import datetime
import json
import os
import re
import requests

import feedparser
from extractcontent3 import ExtractContent
import tiktoken
from tiktoken.core import Encoding
import openai
from slack_sdk import WebClient


class TechNewsSummarizer:
    MODEL = "gpt-3.5-turbo"
    MAX_TOKENS = 4096 - 500
    LAST_EXECUTED_FILE = "last_executed_at.json"
    CONFIG = "config.json"
    DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
    JST = datetime.timezone(datetime.timedelta(hours=9), "JST")

    def __init__(self) -> None:
        self.config = self.load_config()
        self.extractor = ExtractContent()
        self.extractor.set_option({"threshold": 50})
        openai.organization = self.config["openai"]["organization_id"]
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.token_encoding = tiktoken.encoding_for_model(self.MODEL)
        self.slack = WebClient(token=os.getenv("SLACK_API_TOKEN"))

    @staticmethod
    def load_config() -> dict:
        with open(TechNewsSummarizer.CONFIG) as f:
            return json.load(f)

    @staticmethod
    def half_hour_ago() -> datetime.datetime:
        return datetime.datetime.now(TechNewsSummarizer.JST) - datetime.timedelta(
            minutes=30
        )

    def load_last_executed_at(self) -> datetime.datetime:
        try:
            with open(self.LAST_EXECUTED_FILE) as f:
                return datetime.datetime.strptime(
                    json.load(f)["at"], self.DATE_FORMAT
                ).astimezone(self.JST)
        except Exception:
            return self.half_hour_ago()

    def save_last_executed_at(self, now: datetime.datetime) -> None:
        with open(self.LAST_EXECUTED_FILE, "w+") as f:
            json.dump({"at": now.strftime(self.DATE_FORMAT)}, f)

    def generate_summary(self, entry: dict, attempt: int) -> dict:
        if attempt > 3:
            return {
                "title": entry["title"],
                "url": entry["link"],
                "points": ["要約に失敗しました。"],
            }

        def create_chat_completion_request(title, content) -> dict:
            system = (
                "下記をStep by Stepで実行してください。\n"
                "[P1] タイトルを日本語に翻訳する\n"
                "[P2] 記事の中から重要な情報を見つける\n"
                "[P3] 重要な情報をキーポイントに分類する\n"
                "[P4] キーポイントをx個にまとめる\n"
                "[P5] キーポイントを箇条書き形式で整理する\n"
                '[P6] [P5]の内容と[P1]で翻訳したタイトルをjson形式で出力する(形式: {{"points": [...], "title": "..."}})\n\n'
                f"[Title]:{title}\n[Content]:{content}\nlang: jp"
            )
            return openai.ChatCompletion.create(
                model=self.MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system,
                    },
                    {
                        "role": "user",
                        "content": "[P6]以外の内容は出力しないでください。\nlang: jp",
                    },
                ],
            )

        try:
            response = create_chat_completion_request(entry["title"], entry["content"])
            resp_content = response["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"attempt: {attempt}")
            print(f"failed to request: {e}")
            return self.generate_summary(entry, attempt + 1)

        pattern = r'{\s*"points"\s*:\s*\[[^\]]*?\]\s*,\s*"title"\s*:\s*"[^"]*"\s*}'
        matches = re.findall(pattern, resp_content, re.MULTILINE | re.DOTALL)
        try:
            summary = json.loads(matches[-1])
            summary["raw_title"] = entry["title"]
            summary["url"] = entry["link"]
            summary["updated"] = entry["feedai_updated"]
            print(summary)
            return summary
        except Exception as e:
            print(f"attempt: {attempt}")
            print(f"failed to parse json: {e}")
            print(f"response content: {resp_content}")
            return self.generate_summary(entry, attempt + 1)

    def extract_content(self, entry: dict) -> str:
        try:
            res = requests.get(entry["link"])
            self.extractor.analyse(res.text)
            text, _ = self.extractor.as_text()
            if text is None:
                raise Exception("text is None")
            return text
        except Exception as e:
            print(f"failed to extract content: {e}")

    def filter_target_entries(
        self, feeds: list[dict], last: datetime.datetime
    ) -> list[dict]:
        target_entries = []
        for feed in feeds:
            print(f"parse feed '{feed['url']}' ...")
            entries = feedparser.parse(feed["url"])["entries"]
            print({"url": feed["url"], "len": len(entries)})
            for entry in entries:
                if entry.get("published_parsed") is None:
                    entry["published_parsed"] = entry.get("updated_parsed")
                updated = datetime.datetime(
                    *entry["published_parsed"][:6], tzinfo=datetime.timezone.utc
                ).astimezone(self.JST)

                entry["feedai_updated"] = updated

                if updated < last:
                    continue

                if entry.get("content") is None:
                    if not feed["extract_content"]:
                        continue
                    content = self.extract_content(entry)
                    if content is None or entry.get("content") == "":
                        continue
                    entry["content"] = [{"value": content}]

                tokens = self.token_encoding.encode(str(entry["content"]))
                if len(tokens) > self.MAX_TOKENS:
                    continue

                target_entries.append(entry)
        return target_entries

    def generate_slack_text(self, summary: dict) -> str:
        return f"[{summary['title']}]"

    def generate_slack_blocks(self, summary: dict) -> list[dict]:
        markdown = "```\n"
        for point in summary["points"]:
            markdown += f"- {point}\n"
        markdown += "```\n"

        title = f"*{summary['raw_title']}*"
        if summary["raw_title"] != summary["title"]:
            title += f"\n({summary['title']})"
        return [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": ":newspaper: New Tech News :newspaper:",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": summary["url"],
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": title,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"updated: `{summary['updated'].strftime(self.DATE_FORMAT)}`",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": markdown,
                },
            },
        ]

    def post_summary_to_slack(self, summary: dict, channel: str) -> None:
        self.slack.chat_postMessage(
            text=self.generate_slack_text(summary),
            blocks=self.generate_slack_blocks(summary),
            mrkdwn=True,
            channel=channel,
        )

    def run(self) -> None:
        print("start")
        last_execution_time = self.load_last_executed_at()
        print("last executed at: ", last_execution_time)
        now = datetime.datetime.now(self.JST)
        print("now             : ", now)
        target_entries = self.filter_target_entries(
            self.config["feeds"], last_execution_time
        )

        if not target_entries:
            self.save_last_executed_at(now)
            return

        summaries = []
        print(f"start summarizing {len(target_entries)} entries")
        for i, entry in enumerate(target_entries):
            summaries.append(self.generate_summary(entry, 1))
            print(f"ok, {i+1}/{len(target_entries)}")
        print(f"finish summarizing {len(summaries)} entries")

        print("start posting to slack")
        for i, summary in enumerate(summaries):
            try:
                self.post_summary_to_slack(summary, self.config["slack"]["channel"])
                print(f"ok, {i+1}/{len(summaries)}")
            except Exception as e:
                print(f"failed to post to slack: {e}")
                print(f"finish posting to slack {len(summaries)} entries")
        self.save_last_executed_at(now)
        print("finish")


if __name__ == "__main__":
    summarizer = TechNewsSummarizer()
    summarizer.run()

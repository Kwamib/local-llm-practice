"""Ask a local model who leads each town in the practice set.

Usage:
    python3 ask.py                     every page, scored against answers.csv
    python3 ask.py pages/fox-run.html  one page
"""

import csv
import sys
from html.parser import HTMLParser

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"

PROMPT = """Read this web page and answer with only the full name of the person currently serving as mayor or acting mayor, nothing else.
If the page does not name a current mayor or acting mayor, answer NONE.

PAGE:
{page}"""


class TextOnly(HTMLParser):
    """Collect the visible text of an HTML page, one piece per line."""

    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        if data.strip():
            self.parts.append(data.strip())


def page_text(path):
    parser = TextOnly()
    with open(path, encoding="utf-8") as f:
        parser.feed(f.read())
    return "\n".join(parser.parts)


def ask(prompt, temperature=0):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["response"].strip()


def main():
    if len(sys.argv) > 1:
        print(ask(PROMPT.format(page=page_text(sys.argv[1]))))
        return

    right = 0
    with open("answers.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        got = ask(PROMPT.format(page=page_text(row["file"])))
        ok = got == row["answer"]
        right += ok
        mark = "ok  " if ok else "MISS"
        print(f"{mark} {row['town']:<14} expected {row['answer']:<16} got {got}")
    print(f"\n{right} of {len(rows)} exactly right")


if __name__ == "__main__":
    main()

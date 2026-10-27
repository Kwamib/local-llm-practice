"""Try several prompts against every practice page and score each one.

Usage:
    python3 prompts.py            every prompt
    python3 prompts.py precise    one prompt, by name
"""

import csv
import sys

import requests

from ask import page_text

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"

PROMPTS = {
    "vague": {
        "system": None,
        "prompt": "Who is the mayor?\n\n{page}",
    },
    "narrow": {
        "system": None,
        "prompt": """Read this web page and answer with only the full name of the person currently serving as mayor or acting mayor, nothing else.
If the page does not name a current mayor or acting mayor, answer NONE.

PAGE:
{page}""",
    },
    "precise": {
        "system": None,
        "prompt": """Read this web page and answer with only the full name of the person currently serving as mayor or acting mayor, nothing else.
Write the name as first name and last name only: no titles, no nicknames, no quotation marks.
If the page does not name a current mayor or acting mayor, answer NONE.

PAGE:
{page}""",
    },
    "precise+system": {
        "system": "You read web pages for a data team. Answer only from the page you are given. Never use outside knowledge. If the page does not contain the answer, say NONE.",
        "prompt": """Read this web page and answer with only the full name of the person currently serving as mayor or acting mayor, nothing else.
Write the name as first name and last name only: no titles, no nicknames, no quotation marks.
If the page does not name a current mayor or acting mayor, answer NONE.

PAGE:
{page}""",
    },
    "narrow+system": {
        "system": "You read web pages for a data team. Answer only from the page you are given. Never use outside knowledge. If the page does not contain the answer, say NONE.",
        "prompt": """Read this web page and answer with only the full name of the person currently serving as mayor or acting mayor, nothing else.
If the page does not name a current mayor or acting mayor, answer NONE.

PAGE:
{page}""",
    },
    "clearer": {
        "system": None,
        "prompt": """Read this web page and find the person currently serving as mayor or acting mayor.
Answer with only that person's name. Write it the way the page writes it, but leave out any title such as Mayor, and leave out any nickname in quotation marks.
If the page does not name a current mayor or acting mayor, answer NONE.

PAGE:
{page}""",
    },
    "example": {
        "system": None,
        "prompt": """Read this web page and answer with only the full name of the person currently serving as mayor or acting mayor, nothing else.
If the page does not name a current mayor or acting mayor, answer NONE.

Example: if a page says "Mayor Patricia \"Pat\" Nolan welcomed visitors", the answer is: Patricia Nolan

PAGE:
{page}""",
    },
}


def generate(prompt, system=None):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }
    if system:
        payload["system"] = system
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["response"].strip()


def main():
    names = sys.argv[1:] or list(PROMPTS)
    with open("answers.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for name in names:
        p = PROMPTS[name]
        right = 0
        misses = []
        for row in rows:
            got = generate(p["prompt"].format(page=page_text(row["file"])), p["system"])
            if got == row["answer"]:
                right += 1
            else:
                misses.append((row["town"], row["answer"], got))
        print(f"== {name}: {right} of {len(rows)} exactly right")
        for town, expected, got in misses:
            shown = got.replace("\n", " ")
            if len(shown) > 110:
                shown = shown[:110] + "..."
            print(f"   MISS {town:<14} expected {expected:<16} got {shown}")
        print()


if __name__ == "__main__":
    main()

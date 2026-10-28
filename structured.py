"""Ask for structured answers instead of free text, and score them.

Usage:
    python3 structured.py                         every variant
    python3 structured.py none narrow-schema      some variants, by name
    python3 structured.py pages/elm-ridge.html    one page, raw replies shown
"""

import csv
import json
import sys

import requests

from ask import MODEL, page_text

OLLAMA_URL = "http://localhost:11434/api/generate"

JSON_PROMPT = """Read this web page and find the person currently serving as mayor or acting mayor.
Respond with JSON only, in this shape: {{"full_name": string or null, "title": string or null}}
Use null for full_name if the page does not name a current mayor or acting mayor.

PAGE:
{page}"""

NARROW_JSON_PROMPT = """Read this web page and answer with only the full name of the person currently serving as mayor or acting mayor, nothing else.
If the page does not name a current mayor or acting mayor, answer null.
Respond with JSON only, in this shape: {{"full_name": string or null}}

PAGE:
{page}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "full_name": {"type": ["string", "null"]},
        "title": {"type": ["string", "null"]},
    },
    "required": ["full_name", "title"],
}

NAME_ONLY_SCHEMA = {
    "type": "object",
    "properties": {"full_name": {"type": ["string", "null"]}},
    "required": ["full_name"],
}

VARIANTS = {
    "json": (JSON_PROMPT, "json"),
    "schema": (JSON_PROMPT, SCHEMA),
    "none": (JSON_PROMPT, None),
    "narrow-schema": (NARROW_JSON_PROMPT, NAME_ONLY_SCHEMA),
}


def ask_json(page, variant):
    prompt, fmt = VARIANTS[variant]
    payload = {
        "model": MODEL,
        "prompt": prompt.format(page=page),
        "stream": False,
        "options": {"temperature": 0},
    }
    if fmt is not None:
        payload["format"] = fmt
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["response"]


def main():
    args = sys.argv[1:]
    if args and "/" in args[0]:
        for variant in VARIANTS:
            print(f"{variant}: {ask_json(page_text(args[0]), variant)}")
        return

    with open("answers.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for variant in args or list(VARIANTS):
        right = broken = 0
        print(f"== {variant}")
        for row in rows:
            raw = ask_json(page_text(row["file"]), variant)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                broken += 1
                shown = raw.strip().replace("\n", " ")[:80]
                print(f"   NOT JSON {row['town']:<14} {shown}")
                continue
            name = data.get("full_name") or "NONE"
            ok = name == row["answer"]
            right += ok
            mark = "ok  " if ok else "MISS"
            print(f"   {mark} {row['town']:<14} full_name {str(data.get('full_name')):<20} title {data.get('title', '-')}")
        print(f"   {right} of {len(rows)} exactly right, {broken} not valid JSON\n")


if __name__ == "__main__":
    main()

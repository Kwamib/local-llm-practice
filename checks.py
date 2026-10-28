"""Check a model's answers before trusting them.

Two checks, the same two CivicGrid's weekly mayor check relies on:
  1. on the page: every word of the name appears in the page text
  2. second look: the model is asked, yes or no, whether that person
     currently holds the job, according to the page

Usage:
    python3 checks.py answers    the narrow+system prompt from Lesson 5, checked
    python3 checks.py decoys     right answers, wrong people and made-up names, checked
    python3 checks.py raw        the second look's exact replies, for a few cases
    python3 checks.py looks      four versions of the second look, scored
"""

import csv
import json
import os
import re
import sys

import requests

from ask import MODEL, page_text

OLLAMA_URL = "http://localhost:11434/api/generate"

# Try another model for one run: LLM_MODEL=qwen2.5:7b python3 checks.py looks
MODEL = os.environ.get("LLM_MODEL", MODEL)

SYSTEM = "You read web pages for a data team. Answer only from the page you are given. Never use outside knowledge. If the page does not contain the answer, say NONE."

EXTRACT = """Read this web page and answer with only the full name of the person currently serving as mayor or acting mayor, nothing else.
If the page does not name a current mayor or acting mayor, answer NONE.

PAGE:
{page}"""

SECOND_LOOK = """Read this web page. According to the page, is {name} the person currently serving as mayor or acting mayor right now?
Answer with only yes or no.

PAGE:
{page}"""

# For each town: a real person on the page who is NOT the current mayor,
# and a name that appears nowhere on the page.
DECOYS = {
    "Cedar Bend": "Harold Greene",
    "Elm Ridge": "Raymond Ellis",
    "Alder Creek": "Diego Ramirez",
    "Birch Hollow": "Margaret Coyle",
    "Fox Run": None,
    "Juniper Flats": "Rosa Delgado",
    "Heron Point": None,
    "Quarry Hill": None,
    "Willow Bend": None,
    "Stone Mill": "Dana Whitfield",
}
MADE_UP = "Emily Mitchell"


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


def on_page(name, text):
    """True if every word of the name appears in the page, ignoring case."""
    words = re.findall(r"[A-Za-z']+", name)
    return bool(words) and all(
        re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE) for w in words
    )


def second_look(name, text):
    reply = generate(SECOND_LOOK.format(name=name, page=text))
    return reply.strip().lower().startswith("yes")


def check(name, text):
    """Return (on_page, second_look, verdict). Second look runs only if on the page."""
    if name == "NONE":
        return None, None, "NONE: send to a person"
    if not on_page(name, text):
        return False, None, "REJECT: not on the page"
    if not second_look(name, text):
        return True, False, "REVIEW: second look said no"
    return True, True, "ACCEPT"


def rows():
    with open("answers.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def answers():
    print(f"{'town':<14} {'answer':<20} {'truth':<16} verdict")
    for row in rows():
        text = page_text(row["file"])
        got = generate(EXTRACT.format(page=text), SYSTEM)
        _, _, verdict = check(got, text)
        print(f"{row['town']:<14} {got:<20} {row['answer']:<16} {verdict}")


def decoys():
    print(f"{'town':<14} {'kind':<11} {'name':<16} verdict")
    for row in rows():
        text = page_text(row["file"])
        tests = []
        if row["answer"] != "NONE":
            tests.append(("right", row["answer"]))
        if DECOYS[row["town"]]:
            tests.append(("wrong", DECOYS[row["town"]]))
        tests.append(("made up", MADE_UP))
        for kind, name in tests:
            _, _, verdict = check(name, text)
            print(f"{row['town']:<14} {kind:<11} {name:<16} {verdict}")


def raw():
    """Show the second look's exact reply for a few right and wrong names."""
    cases = [
        ("pages/cedar-bend.html", "Dana Whitfield"),
        ("pages/cedar-bend.html", "Harold Greene"),
        ("pages/juniper-flats.html", "Tom Haskell"),
        ("pages/juniper-flats.html", "Rosa Delgado"),
    ]
    for path, name in cases:
        reply = generate(SECOND_LOOK.format(name=name, page=page_text(path)))
        print(f"{path:<26} {name:<16} {reply!r}")


CONFIRM = """You are reading the text of a single web page from the official website of {town}.

Question: according to this page, is {name} the person currently serving as mayor or acting mayor of {town}?

Answer false if the page lists this person as a former or past mayor, as a mayor-elect who has not taken office yet, as a council member only, or as the mayor of a different town, or if the page does not make it clear. Do not use any knowledge from outside this page.

Respond with JSON only: {{"current": true|false}}

PAGE TEXT:
---
{page}
---"""

CONFIRM_SCHEMA = {
    "type": "object",
    "properties": {"current": {"type": "boolean"}},
    "required": ["current"],
}


def without_footer(text):
    """Drop the practice set's 'this town is made up' line."""
    return "\n".join(line for line in text.splitlines() if "made up for practice" not in line)


def look(variant, town, name, text):
    if variant.endswith("no-footer"):
        text = without_footer(text)
    if variant.startswith("verifier-style"):
        r = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": CONFIRM.format(town=town, name=name, page=text),
            "stream": False,
            "format": CONFIRM_SCHEMA,
            "options": {"temperature": 0},
        }, timeout=120)
        r.raise_for_status()
        return json.loads(r.json()["response"]).get("current") is True
    return second_look(name, text)


LOOKS = ["plain", "no-footer", "verifier-style", "verifier-style+no-footer"]


def looks():
    """Try four versions of the second look on every real person in the decoy test."""
    cases = []
    for row in rows():
        if row["answer"] != "NONE":
            cases.append((row, "right", row["answer"]))
        if DECOYS[row["town"]]:
            cases.append((row, "wrong", DECOYS[row["town"]]))
    for variant in LOOKS:
        kept = caught = 0
        problems = []
        for row, kind, name in cases:
            said_yes = look(variant, row["town"], name, page_text(row["file"]))
            if kind == "right" and said_yes:
                kept += 1
            elif kind == "wrong" and not said_yes:
                caught += 1
            else:
                problems.append(f"{row['town']} {kind} {name}: said {'yes' if said_yes else 'no'}")
        rights = sum(1 for c in cases if c[1] == "right")
        wrongs = len(cases) - rights
        print(f"== {variant}: right names passed {kept} of {rights}, wrong names caught {caught} of {wrongs}")
        for line in problems:
            print(f"   {line}")
        print()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "answers"
    {"answers": answers, "decoys": decoys, "raw": raw, "looks": looks}[mode]()

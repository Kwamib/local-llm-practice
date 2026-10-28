"""What does this job actually use, and what would it cost hosted?

Usage:
    python3 costs.py                      measure every case in cases.csv
    python3 costs.py --pages 3064 --runs 4

Runs Lesson 10's cases through the local model, records how many tokens
went in and came out and how long each took, then works out what the same
work would cost at the hosted prices in prices.csv. Prices change: check the
provider's pricing page and update prices.csv before trusting the totals.

Token counts come from Ollama's own tokenizer. A hosted model counts tokens
its own way, so treat hosted costs as an estimate, not a quote.
"""

import argparse
import csv

import requests

from ask import MODEL, PROMPT, page_text

OLLAMA = "http://localhost:11434"


def measure(path):
    r = requests.post(f"{OLLAMA}/api/generate", timeout=300, json={
        "model": MODEL, "prompt": PROMPT.format(page=page_text(path)), "stream": False,
        "options": {"temperature": 0, "num_ctx": 8192}})
    r.raise_for_status()
    d = r.json()
    return d["prompt_eval_count"], d["eval_count"], d["total_duration"] / 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pages", type=int, default=3064, help="pages per run, for the estimate")
    p.add_argument("--runs", type=int, default=4, help="runs per month, for the estimate")
    args = p.parse_args()

    with open("cases.csv", newline="", encoding="utf-8") as f:
        cases = list(csv.DictReader(f))
    tin = tout = secs = 0
    print(f"{'case':<22} {'tokens in':>9} {'tokens out':>10} {'seconds':>8}")
    for case in cases:
        i, o, s = measure(case["file"])
        tin, tout, secs = tin + i, tout + o, secs + s
        print(f"{case['case']:<22} {i:>9} {o:>10} {s:>8.2f}")
    n = len(cases)
    print(f"{'average':<22} {tin / n:>9.0f} {tout / n:>10.1f} {secs / n:>8.2f}\n")

    pages = args.pages * args.runs
    month_in, month_out = tin / n * pages, tout / n * pages
    print(f"Estimate: {args.pages:,} pages a run, {args.runs} runs a month = {pages:,} pages")
    print(f"   {month_in / 1e6:,.1f} million tokens in, {month_out / 1e6:,.2f} million out")
    print(f"   local, {MODEL}: about {secs / n * pages / 3600:,.1f} hours of this machine's time\n")
    with open("prices.csv", newline="", encoding="utf-8") as f:
        prices = list(csv.DictReader(f))
    print(f"{'hosted model':<20} {'per month':>10} {'batch, half price':>18}   prices checked")
    for row in prices:
        cost = month_in / 1e6 * float(row["input per million"]) + month_out / 1e6 * float(row["output per million"])
        print(f"{row['model']:<20} {'$' + format(cost, ',.2f'):>10} {'$' + format(cost / 2, ',.2f'):>18}   {row['checked']}")


if __name__ == "__main__":
    main()

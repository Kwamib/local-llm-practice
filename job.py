"""The mayor check as a job that runs by itself, and refuses to run badly.

Usage:
    python3 job.py pin                record the model's exact digest in job.json
    python3 job.py run [--breaker N]  one checked run; new proposals go to proposals/pending.csv
    python3 job.py status             when the last run finished, and how

current.csv is the "database": who we currently think leads each town. The
job never changes it. Anything different becomes a proposal for a person,
once: a proposal already waiting in pending.csv isn't made again.

Exit codes, so a scheduler or a script can tell what happened:
    0  finished, or skipped because another run was going
    2  preflight failed: Ollama down, model missing, or the model changed
    3  the gate failed: a must-pass case is wrong
    4  the circuit breaker tripped: too many changes to be real
    1  anything else
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

from checks import on_page
from suite import fair, load_cases, score
from injection import visible_text
from ask import PROMPT

OLLAMA = "http://localhost:11434"
CONFIG = Path("job.json")
LOCK = Path("job.lock")
LOG = Path("logs/job.log")
STATUS = Path("logs/last-run.json")
PENDING = Path("proposals/pending.csv")
FIELDS = ["town", "now", "proposed", "source", "proposed at"]
DEFAULTS = {"model": "qwen2.5:3b", "digest": None, "breaker_pct": 30}


class Stop(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def log(message):
    LOG.parent.mkdir(exist_ok=True)
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}"
    print(line)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def config():
    return {**DEFAULTS, **(json.loads(CONFIG.read_text()) if CONFIG.exists() else {})}


def digest_of(model):
    tags = requests.get(f"{OLLAMA}/api/tags", timeout=10).json()["models"]
    return next((m["digest"][:12] for m in tags if m["name"] == model), None)


def ask(model, path, tries=3):
    """One question, with a timeout and a few retries. A model that's still
    loading, or a machine that's briefly busy, shouldn't fail the whole run."""
    payload = {"model": model, "prompt": PROMPT.format(page=visible_text(path)), "stream": False,
               "options": {"temperature": 0, "num_ctx": 8192}}
    for attempt in range(1, tries + 1):
        try:
            r = requests.post(f"{OLLAMA}/api/generate", json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["response"].strip()
        except requests.RequestException as e:
            if attempt == tries:
                raise Stop(1, f"{path}: no answer after {tries} tries ({e.__class__.__name__})")
            log(f"   {path}: try {attempt} failed ({e.__class__.__name__}), waiting {2 ** attempt}s")
            time.sleep(2 ** attempt)


def preflight(cfg):
    try:
        version = requests.get(f"{OLLAMA}/api/version", timeout=5).json()["version"]
    except requests.RequestException:
        raise Stop(2, "preflight: Ollama is not answering on localhost:11434")
    found = digest_of(cfg["model"])
    if found is None:
        raise Stop(2, f"preflight: {cfg['model']} is not pulled")
    if cfg["digest"] is None:
        raise Stop(2, "preflight: no pinned digest; run `python3 job.py pin` first")
    if found != cfg["digest"]:
        raise Stop(2, f"preflight: {cfg['model']} is now {found}, pinned {cfg['digest']}. "
                      "The model changed: run the test suite, then pin again")
    log(f"preflight ok: Ollama {version}, {cfg['model']} {found}")


def gate(cfg):
    """The must-pass cases from Lesson 10, run before any real work."""
    for case in (c for c in load_cases().values() if c["must pass"] == "yes"):
        got = ask(cfg["model"], case["file"])
        outcome = score(got, case, visible_text(case["file"]))
        if outcome != "right":
            raise Stop(3, f"gate: {case['case']} gave {got!r} ({outcome}), refusing to run")
    log("gate ok: every must-pass case is right")


def pending():
    if not PENDING.exists():
        return set()
    with PENDING.open(newline="", encoding="utf-8") as f:
        return {(r["town"], fair(r["proposed"])) for r in csv.DictReader(f)}


def work(cfg, breaker_pct):
    with open("current.csv", newline="", encoding="utf-8") as f:
        towns = list(csv.DictReader(f))
    waiting = pending()
    proposals, people = [], []
    for i, town in enumerate(towns, 1):
        got = ask(cfg["model"], town["file"])
        now = town["mayor"] or "NONE"
        if got == "NONE":
            verdict = "same" if now == "NONE" else "a person should look: no mayor found"
        elif not on_page(got, visible_text(town["file"])):
            verdict = f"rejected: {got!r} is not on the page"
        elif fair(got) == fair(now):
            verdict = "same"
        elif (town["town"], fair(got)) in waiting:
            verdict = "already proposed, still waiting for a person"
        else:
            verdict = "proposal"
            proposals.append({"town": town["town"], "now": now, "proposed": got, "source": town["file"],
                              "proposed at": datetime.now().isoformat(timespec="seconds")})
        if verdict not in ("same", "already proposed, still waiting for a person"):
            people.append(f"{town['town']}: {verdict}")
        log(f"   {town['town']:<14} {got:<22} {verdict}")
        if 100 * len(proposals) / len(towns) > breaker_pct:
            raise Stop(4, f"circuit breaker: {len(proposals)} changes after {i} of {len(towns)} towns, "
                          f"more than {breaker_pct}%. Real turnover is a trickle; this looks like a "
                          "broken page or a broken model. Nothing written")
    return proposals, people


def run(args):
    cfg = config()
    breaker = args.breaker if args.breaker is not None else cfg["breaker_pct"]
    try:
        LOCK.mkdir()
    except FileExistsError:
        log("skipped: another run is still going (job.lock exists)")
        return 0
    started = time.time()
    code, message = 0, ""
    try:
        log(f"run started: {cfg['model']}, breaker {breaker}%")
        preflight(cfg)
        gate(cfg)
        proposals, people = work(cfg, breaker)
        PENDING.parent.mkdir(exist_ok=True)
        new_file = not PENDING.exists()
        with PENDING.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
            w.writerows(proposals)
        message = f"{len(proposals)} new proposals in {PENDING}, {len(people)} for a person to see"
    except Stop as stop:
        code, message = stop.code, str(stop)
    except Exception as e:
        code, message = 1, f"unexpected error: {e!r}"
    finally:
        LOCK.rmdir()
    log(f"run finished: exit {code}, {time.time() - started:.0f}s. {message}")
    STATUS.write_text(json.dumps({"finished": datetime.now().isoformat(timespec="seconds"),
                                  "exit": code, "message": message}) + "\n")
    return code


def pin(args):
    cfg = config()
    cfg["digest"] = digest_of(cfg["model"])
    CONFIG.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"pinned {cfg['model']} at {cfg['digest']} in {CONFIG}")


def status(args):
    if not STATUS.exists():
        print("no run has finished yet")
        return 1
    s = json.loads(STATUS.read_text())
    age = datetime.now() - datetime.fromisoformat(s["finished"])
    print(f"last run finished {age.total_seconds() / 3600:.1f} hours ago, exit {s['exit']}: {s['message']}")
    return 0 if s["exit"] == 0 else 1


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--breaker", type=float, help="percent of towns that may change before the run halts")
    sub.add_parser("pin")
    sub.add_parser("status")
    args = p.parse_args()
    sys.exit({"run": run, "pin": pin, "status": status}[args.cmd](args) or 0)


if __name__ == "__main__":
    main()

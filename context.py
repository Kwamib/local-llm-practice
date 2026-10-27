"""One long page, asked about with different context window sizes.

Usage:
    python3 context.py
"""

import requests

from ask import MODEL, page_text

OLLAMA = "http://localhost:11434"
PAGE = page_text("long/larch-grove.html")

INSTRUCTIONS = """Read this web page and answer with only the full name of the person currently serving as mayor or acting mayor, nothing else.
If the page does not name a current mayor or acting mayor, answer NONE."""

FIRST = INSTRUCTIONS + "\n\nPAGE:\n{page}"
LAST = "PAGE:\n{page}\n\n" + INSTRUCTIONS


def focus(text):
    """Keep only the lines that mention the mayor, without repeats."""
    seen = []
    for line in text.splitlines():
        if "Mayor" in line and line not in seen:
            seen.append(line)
    return "\n".join(seen)


def run(label, prompt, num_ctx):
    r = requests.post(f"{OLLAMA}/api/generate", json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": num_ctx},
    }, timeout=300)
    r.raise_for_status()
    data = r.json()
    ps = requests.get(f"{OLLAMA}/api/ps", timeout=10).json()["models"]
    memory = ps[0]["size"] / 1e9 if ps else 0
    answer = data["response"].strip().replace("\n", " ")
    if len(answer) > 70:
        answer = answer[:70] + "..."
    print(f"{label:<22} num_ctx {num_ctx:>6}  read {data['prompt_eval_count']:>5} tokens  "
          f"memory {memory:4.1f} GB  answer: {answer}")


def main():
    print(f"The page is {len(PAGE):,} characters of text.\n")
    for num_ctx in (2048, 4096, 8192, 32768):
        run("instructions first", FIRST.format(page=PAGE), num_ctx)
    print()
    run("instructions last", LAST.format(page=PAGE), 2048)
    short = focus(PAGE)
    print(f"\nThe focused page is {len(short):,} characters.")
    run("focused page", FIRST.format(page=short), 2048)


if __name__ == "__main__":
    main()

"""Ask questions about a pile of documents: find the right passages first,
then answer from them. Retrieval-augmented generation (RAG), in plain Python.

Usage:
    python3 rag.py index                 split docs/ into passages and embed them
    python3 rag.py search "question"     the three closest passages, with scores
    python3 rag.py retrieve              does each question find its document?
    python3 rag.py answer [MODES]        answer every question; modes: none, all, top3, top1
    python3 rag.py rescore [MODES]       score the saved answers again, without the model

Answers are saved to results/rag-MODE.json, so a fix to the scoring never
needs the model to run again (Lesson 10).
"""

import csv
import json
import math
import re
import sys
from pathlib import Path

import requests

from ask import MODEL

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
DOCS = Path("docs")
INDEX = Path("index.json")
FOOTER = "This document is made up for practice."

ANSWER_PROMPT = """Answer the question using only the documents below.
If the documents do not contain the answer, reply with exactly: NOT IN DOCUMENTS
Keep the answer to one or two sentences.

Documents:
{docs}

Question: {question}"""

NO_DOCS_PROMPT = """Answer the question in one or two sentences.
If you do not know the answer, reply with exactly: NOT IN DOCUMENTS

Question: {question}"""


def passages(with_titles=True):
    """Split each document into paragraphs. Each passage starts with its
    document's title, so "The fine is $50" still says which town."""
    out = []
    for path in sorted(DOCS.glob("*.md")):
        lines = path.read_text(encoding="utf-8").split("\n\n")
        title = lines[0].lstrip("# ").strip()
        for para in lines[1:]:
            para = para.strip()
            if para and not para.startswith(FOOTER):
                text = f"{title}\n{para}" if with_titles else para
                out.append({"source": path.name, "text": text})
    return out


def embed(texts, kind):
    """nomic-embed-text wants to know whether it's embedding a question
    or a document, so each text gets a prefix."""
    prefix = "search_query: " if kind == "query" else "search_document: "
    r = requests.post(f"{OLLAMA}/api/embed", timeout=300,
                      json={"model": EMBED_MODEL, "input": [prefix + t for t in texts]})
    r.raise_for_status()
    return r.json()["embeddings"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def index():
    data = {}
    for label, titles in (("with titles", True), ("without titles", False)):
        ps = passages(titles)
        for p, v in zip(ps, embed([p["text"] for p in ps], "document")):
            p["vector"] = v
        data[label] = ps
    INDEX.write_text(json.dumps(data))
    n = len(data["with titles"])
    print(f"{n} passages from {len(list(DOCS.glob('*.md')))} documents, "
          f"{len(data['with titles'][0]['vector'])} numbers per passage, saved to {INDEX}")


def load(label="with titles"):
    return json.loads(INDEX.read_text())[label]


def by_meaning(question, ps, k=3):
    q = embed([question], "query")[0]
    return sorted(((cosine(q, p["vector"]), p) for p in ps), key=lambda x: -x[0])[:k]


WORD = re.compile(r"[a-z0-9']+")
SMALL = {"the", "a", "an", "is", "in", "of", "to", "for", "on", "and", "or", "i", "my", "how",
         "what", "when", "who", "do", "does", "can", "it", "be", "at", "get", "much", "did", "that"}


def by_words(question, ps, k=3):
    """Old-fashioned search: count the question's words that appear in each passage."""
    words = set(WORD.findall(question.lower())) - SMALL
    scored = [(len(words & set(WORD.findall(p["text"].lower()))), p) for p in ps]
    return sorted(scored, key=lambda x: -x[0])[:k]


def questions():
    with open("questions.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def search(question):
    for score, p in by_meaning(question, load()):
        first = p["text"].split("\n", 1)[1][:90]
        print(f"{score:.3f}  {p['source']:<36} {first}...")


def retrieve():
    qs = [q for q in questions() if q["source"]]
    methods = {
        "words": lambda q: by_words(q, load()),
        "meaning, with titles": lambda q: by_meaning(q, load("with titles")),
        "meaning, without titles": lambda q: by_meaning(q, load("without titles")),
    }
    for name, find in methods.items():
        top1 = top3 = 0
        misses = []
        for q in qs:
            sources = [p["source"] for _, p in find(q["question"])]
            top1 += sources[0] == q["source"]
            top3 += q["source"] in sources
            if sources[0] != q["source"]:
                misses.append(f"   {q['question']}\n      wanted {q['source']}, top was {sources[0]}")
        print(f"== {name}: right document first {top1} of {len(qs)}, in the top 3 {top3} of {len(qs)}")
        print("\n".join(misses))
        print()


def contains(reply, phrase):
    return re.search(rf"(?<![\w]){re.escape(phrase)}(?![\w])", reply, re.IGNORECASE) is not None


def score(reply, q):
    wrong = q["wrong if it contains"]
    if wrong and contains(reply, wrong):
        return "WRONG"
    if not q["source"]:
        return "right" if any(contains(reply, alt) for alt in q["must contain"].split("|")) else "MADE UP"
    if all(any(contains(reply, alt) for alt in group.split("|")) for group in q["must contain"].split(";")):
        return "right"
    if contains(reply, "NOT IN DOCUMENTS"):
        return "missed"
    return "check by hand"


def generate(prompt):
    r = requests.post(f"{OLLAMA}/api/generate", timeout=300, json={
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "num_ctx": 8192}})
    r.raise_for_status()
    return " ".join(r.json()["response"].split())


def context(question, mode):
    if mode == "all":
        ps = passages()
    else:
        ps = [p for _, p in by_meaning(question, load(), k=int(mode[3:]))]
    return "\n\n".join(f"[{p['source']}]\n{p['text']}" for p in ps)


def report(mode, rows):
    print(f"== {mode}")
    results = []
    for q, reply in rows:
        outcome = score(reply, q)
        results.append(outcome)
        print(f"   {outcome:<14} {q['question']}\n                  {reply[:150]}")
    tally = {o: results.count(o) for o in dict.fromkeys(results)}
    print("   " + ", ".join(f"{k} {v}" for k, v in tally.items()) + "\n")


def answer(modes):
    Path("results").mkdir(exist_ok=True)
    for mode in modes or ["none", "all", "top3", "top1"]:
        rows = []
        for q in questions():
            if mode == "none":
                prompt = NO_DOCS_PROMPT.format(question=q["question"])
            else:
                prompt = ANSWER_PROMPT.format(docs=context(q["question"], mode), question=q["question"])
            rows.append((q, generate(prompt)))
        saved = {q["question"]: reply for q, reply in rows}
        Path(f"results/rag-{mode}.json").write_text(json.dumps(saved, indent=2) + "\n")
        report(mode, rows)


def rescore(modes):
    for mode in modes or ["none", "all", "top3", "top1"]:
        saved = json.loads(Path(f"results/rag-{mode}.json").read_text())
        report(mode, [(q, saved[q["question"]]) for q in questions()])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "retrieve"
    if cmd == "index":
        index()
    elif cmd == "search":
        search(" ".join(sys.argv[2:]))
    elif cmd == "retrieve":
        retrieve()
    elif cmd == "answer":
        answer(sys.argv[2:])
    elif cmd == "rescore":
        rescore(sys.argv[2:])
    else:
        sys.exit(__doc__)

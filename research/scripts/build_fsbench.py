"""Writes the synthetic personal file systems and the agent tasks.

Course notes come from Alloprof (six main subjects). Train and test personas are
disjoint, and test tasks use a wording never seen in training.
"""

import argparse
import json
import re
from pathlib import Path

from huggingface_hub import hf_hub_download

from sensetree_research.data import DATA_DIR, SEED
from sensetree_research.fsbench import build

SUBJECTS = {"french": "Français", "math": "Mathématiques", "history": "Histoire", "science": "Sciences",
            "english": "Anglais", "geography": "Géographie"}


def load_courses() -> list[dict]:
    raw = json.load(open(hf_hub_download("lyon-nlp/alloprof", "documents.json", repo_type="dataset"), encoding="utf-8"))
    courses = []
    for d in raw:
        if d["topic"] in SUBJECTS and (d["text"] or "").strip() and d["title"]:
            name = re.sub(r"\s+", " ", re.sub(r"[^\w\s'-]", " ", d["title"])).strip()[:80]
            courses.append({"subject": SUBJECTS[d["topic"]], "file": name, "title": d["title"], "text": d["text"]})
    return courses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--train", type=int, default=30)
    parser.add_argument("--test", type=int, default=8)
    args = parser.parse_args()

    tasks = build(Path(args.root), load_courses(), args.train, args.test, SEED)
    out = DATA_DIR / "fsbench"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "tasks.jsonl", "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    by = {}
    for t in tasks:
        by[(t["split"], t["family"])] = by.get((t["split"], t["family"]), 0) + 1
    print(json.dumps({f"{k[0]}/{k[1]}": v for k, v in sorted(by.items())}, indent=1))


if __name__ == "__main__":
    main()

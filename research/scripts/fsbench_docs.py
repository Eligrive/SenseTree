"""Lists the documents of every fsbench persona, with their metadata, without writing them.

Replays ``fsbench.build`` with the same seed and draws, so the paths and tasks match the
benchmark built under ``--root``. Writes ``data/fsbench/docs.jsonl`` and checks that the
replayed tasks are identical to those of that build (``--expected``).
"""

import argparse
import json
import random
from pathlib import Path

from build_fsbench import load_courses
from sensetree_research.data import DATA_DIR, SEED
from sensetree_research.fsbench import build_home, make_persona, tasks_for


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="root the benchmark was built under")
    parser.add_argument("--train", type=int, default=30)
    parser.add_argument("--test", type=int, default=8)
    parser.add_argument("--expected", default=str(DATA_DIR / "fsbench" / "tasks.jsonl"), help="tasks built under --root")
    args = parser.parse_args()

    courses = load_courses()
    root = Path(args.root)
    rng = random.Random(SEED)
    docs, tasks = [], []
    for split, n in [("train", args.train), ("test", args.test)]:
        for i in range(n):
            p = make_persona(i, split, random.Random(rng.random()))
            home = root / p.pid
            selected = p.rng.sample(courses, min(25, len(courses)))
            persona_docs = build_home(p, home, selected)
            variant = p.rng.choice([0, 1]) if split == "train" else 2
            tasks += tasks_for(p, home, persona_docs, variant)
            for d in persona_docs:
                docs.append({"persona": p.pid, "kind": d.kind, "path": str(d.path), "meta": d.meta,
                             "copy_of": d.copy_of, "fmt": d.fmt})

    expected = [json.loads(line) for line in open(args.expected, encoding="utf-8")]
    same = json.dumps(tasks, ensure_ascii=False, sort_keys=True) == json.dumps(expected, ensure_ascii=False, sort_keys=True)
    print(f"{len(docs)} documents, {len(tasks)} tasks, identical to tasks.jsonl: {same}")
    if not same:
        raise SystemExit(1)
    with open(DATA_DIR / "fsbench" / "docs.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    main()

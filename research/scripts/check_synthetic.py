"""Stops the adaptation chain if query generation left a collection under-covered."""

import sys

from sensetree_research.data import DATA_DIR, IN_DOMAIN

MIN_QUERIES = 1000

counts = {}
for name in IN_DOMAIN:
    path = DATA_DIR / name / "synthetic_queries.jsonl"
    counts[name] = sum(1 for _ in open(path, encoding="utf-8")) if path.exists() else 0
print(counts)
sys.exit(0 if all(n >= MIN_QUERIES for n in counts.values()) else 1)

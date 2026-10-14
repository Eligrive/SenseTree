"""Ranking metrics, computed per query so that paired statistics can be run on them.

Conventions follow trec_eval / pytrec_eval as used by BEIR and MTEB: linear gain for
nDCG, relevance threshold of 1, unjudged documents count as non-relevant, and queries
without any relevant document are skipped.
"""

from __future__ import annotations

import math

Run = dict[str, dict[str, float]]
Qrels = dict[str, dict[str, int]]


def ranked(scores: dict[str, float], k: int | None = None) -> list[str]:
    order = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    docs = [d for d, _ in order]
    return docs if k is None else docs[:k]


def ndcg_at_k(ranking: list[str], rels: dict[str, int], k: int) -> float:
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranking[:k]) if rels.get(d, 0) > 0)
    ideal = sorted((r for r in rels.values() if r > 0), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranking: list[str], rels: dict[str, int], k: int) -> float:
    relevant = {d for d, r in rels.items() if r > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranking[:k])) / len(relevant)


def mrr_at_k(ranking: list[str], rels: dict[str, int], k: int) -> float:
    for i, d in enumerate(ranking[:k]):
        if rels.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


METRICS = {
    "ndcg@10": lambda r, q: ndcg_at_k(r, q, 10),
    "recall@10": lambda r, q: recall_at_k(r, q, 10),
    "recall@30": lambda r, q: recall_at_k(r, q, 30),
    "recall@100": lambda r, q: recall_at_k(r, q, 100),
    "mrr@10": lambda r, q: mrr_at_k(r, q, 10),
}


def per_query(run: Run, qrels: Qrels, metrics: list[str] | None = None) -> dict[str, dict[str, float]]:
    """Returns ``{metric: {qid: value}}`` over every judged query.

    A judged query absent from the run scores 0, as a system that returned nothing.
    """
    names = metrics or list(METRICS)
    out: dict[str, dict[str, float]] = {m: {} for m in names}
    for qid, rels in qrels.items():
        if not any(r > 0 for r in rels.values()):
            continue
        ranking = ranked(run.get(qid, {}))
        for m in names:
            out[m][qid] = METRICS[m](ranking, rels)
    return out


def mean(values: dict[str, float]) -> float:
    return sum(values.values()) / len(values) if values else 0.0


def read_trec_run(path) -> Run:
    run: Run = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, _, did, _, score, _ = line.split()
            run.setdefault(qid, {})[did] = float(score)
    return run


def write_trec_run(run: Run, path, tag: str = "run") -> None:
    with open(path, "w", encoding="utf-8") as f:
        for qid in sorted(run):
            for rank, did in enumerate(ranked(run[qid]), start=1):
                f.write(f"{qid} Q0 {did} {rank} {run[qid][did]:.6f} {tag}\n")

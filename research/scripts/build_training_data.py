"""Builds training triplets for the bi-encoder from two sources of supervision.

``supervised``  human relevance judgements (train split). Each (query, positive) pair
                gets one hard negative retrieved by the base model. Candidates that the
                cross-encoder scores above the labelled positive are discarded as likely
                unlabelled positives (RocketQA-style denoising).

``synthetic``   no human labels. Queries come from ``generate_queries.py``; negatives
                are retrieved by the base model; the cross-encoder provides a score
                margin for every (query, positive, negative) triple, which the student
                regresses (MarginMSE, as in GPL, Wang et al. 2022).

The cross-encoder is bge-reranker-v2-m3, the reranker SenseTree ships with: the
adaptation distils the application's own second stage into its first stage.
"""

import argparse
import json
import random

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from sensetree_research.data import DATA_DIR, IN_DOMAIN, SEED, doc_text, load_corpus, load_qrels, load_queries
from sensetree_research.dense import E5, load_model, search

BASE = "intfloat/multilingual-e5-small"
TEACHER = "BAAI/bge-reranker-v2-m3"


def teacher_scores(ce: CrossEncoder, pairs: list[tuple[str, str]]) -> np.ndarray:
    return ce.predict(pairs, batch_size=64, show_progress_bar=False, convert_to_numpy=True,
                      activation_fn=torch.nn.Identity())


def supervised(name: str, model, ce: CrossEncoder, max_queries: int, max_pos: int, rng: random.Random) -> list[dict]:
    corpus = load_corpus(name)
    queries = load_queries(name)
    qrels = load_qrels(name, "train")
    qids = sorted(qrels)
    rng.shuffle(qids)
    qids = qids[:max_queries]
    run, _ = search(model, corpus, {q: queries[q] for q in qids}, E5, top_k=12)

    rows = []
    for qid in qids:
        positives = sorted(qrels[qid], key=lambda d: -qrels[qid][d])[:max_pos]
        ranked = sorted(run[qid], key=lambda d: -run[qid][d])
        cands = [d for d in ranked[2:12] if d not in qrels[qid]]
        if not cands:
            continue
        q = queries[qid]
        scores = teacher_scores(ce, [(q, doc_text(corpus[d])) for d in positives + cands])
        pos_scores, cand_scores = scores[: len(positives)], scores[len(positives):]
        for pid, ps in zip(positives, pos_scores):
            clean = [d for d, s in zip(cands, cand_scores) if s < ps]
            if not clean:
                continue
            rows.append({"query": E5.query + q, "positive": E5.passage + doc_text(corpus[pid]),
                         "negative": E5.passage + doc_text(corpus[rng.choice(clean)])})
    return rows


def synthetic(name: str, model, ce: CrossEncoder, negatives: int, rng: random.Random) -> list[dict]:
    corpus = load_corpus(name)
    gen = [json.loads(line) for line in open(DATA_DIR / name / "synthetic_queries.jsonl", encoding="utf-8")]
    run, _ = search(model, corpus, {g["qid"]: g["text"] for g in gen}, E5, top_k=50)

    rows = []
    for start in range(0, len(gen), 256):
        block = gen[start:start + 256]
        triples, pairs = [], []
        for g in block:
            cands = [d for d in run[g["qid"]] if d != g["doc_id"]]
            for neg in rng.sample(cands, min(negatives, len(cands))):
                triples.append((g["text"], g["doc_id"], neg))
                pairs.append((g["text"], doc_text(corpus[g["doc_id"]])))
                pairs.append((g["text"], doc_text(corpus[neg])))
        scores = teacher_scores(ce, pairs)
        for i, (q, pos, neg) in enumerate(triples):
            rows.append({"query": E5.query + q, "positive": E5.passage + doc_text(corpus[pos]),
                         "negative": E5.passage + doc_text(corpus[neg]),
                         "label": float(scores[2 * i] - scores[2 * i + 1])})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["supervised", "synthetic"])
    parser.add_argument("--datasets", nargs="*", default=IN_DOMAIN)
    parser.add_argument("--max-queries", type=int, default=3000)
    parser.add_argument("--max-positives", type=int, default=4)
    parser.add_argument("--negatives", type=int, default=4)
    args = parser.parse_args()

    model = load_model(BASE)
    ce = CrossEncoder(TEACHER, max_length=512, device="cuda")
    ce.model.half()
    for name in args.datasets:
        rng = random.Random(SEED)
        if args.mode == "supervised":
            rows = supervised(name, model, ce, args.max_queries, args.max_positives, rng)
        else:
            rows = synthetic(name, model, ce, args.negatives, rng)
        out = DATA_DIR / name / f"train_{args.mode}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} {args.mode} triples", flush=True)


if __name__ == "__main__":
    main()

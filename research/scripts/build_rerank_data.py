"""Builds (query, passage, teacher logit) pairs to distil the reranker.

Queries are the training-split questions and the synthetic queries; test queries are
never used. Candidates are the top 20 documents of the base embedding model, which is
roughly what the reranker sees in the application (the head of the fused list).
Passages are cut at 800 characters, the length the application sends to the reranker.
The teacher is bge-reranker-v2-m3, the reranker SenseTree ships.

Dev queries get the same treatment and are written separately, with their judged
documents, for model selection.
"""

import argparse
import json
import random

import torch
from sentence_transformers import CrossEncoder

from sensetree_research.data import DATA_DIR, IN_DOMAIN, SEED, doc_text, load_corpus, load_qrels, load_queries
from sensetree_research.dense import E5, load_model, search

TEACHER = "BAAI/bge-reranker-v2-m3"
PASSAGE_CHARS = 800


def passage(doc: dict) -> str:
    return doc_text(doc)[:PASSAGE_CHARS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=IN_DOMAIN)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--max-train-queries", type=int, default=3000)
    args = parser.parse_args()

    retriever = load_model("intfloat/multilingual-e5-small")
    teacher = CrossEncoder(TEACHER, max_length=512, device="cuda")
    teacher.model.half()
    rng = random.Random(SEED)

    for name in args.datasets:
        corpus = load_corpus(name)
        queries = load_queries(name)
        train = sorted(load_qrels(name, "train"))
        rng.shuffle(train)
        selected = {q: queries[q] for q in train[: args.max_train_queries]}
        with open(DATA_DIR / name / "synthetic_queries.jsonl", encoding="utf-8") as f:
            for line in f:
                g = json.loads(line)
                selected[g["qid"]] = g["text"]
        dev_qrels = load_qrels(name, "dev")
        dev = {q: queries[q] for q in dev_qrels}

        for split, qs in [("train", selected), ("dev", dev)]:
            run, _ = search(retriever, corpus, qs, E5, top_k=args.candidates)
            qids = list(qs)
            out = DATA_DIR / name / f"rerank_{split}.jsonl"
            with open(out, "w", encoding="utf-8") as f:
                for start in range(0, len(qids), 512):
                    block = qids[start:start + 512]
                    pairs, owners = [], []
                    for qid in block:
                        for did in run[qid]:
                            pairs.append((qs[qid], passage(corpus[did])))
                            owners.append((qid, did))
                    scores = teacher.predict(pairs, batch_size=64, show_progress_bar=False,
                                             convert_to_numpy=True, activation_fn=torch.nn.Identity())
                    by_query: dict[str, list] = {}
                    for (qid, did), (q, p), s in zip(owners, pairs, scores):
                        by_query.setdefault(qid, []).append({"doc_id": did, "passage": p, "label": float(s)})
                    for qid in block:
                        row = {"qid": qid, "query": qs[qid], "candidates": by_query[qid]}
                        if split == "dev":
                            row["relevant"] = [d for d, r in dev_qrels[qid].items() if r > 0]
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{name}: {len(qids)} {split} queries x {args.candidates} candidates", flush=True)


if __name__ == "__main__":
    main()

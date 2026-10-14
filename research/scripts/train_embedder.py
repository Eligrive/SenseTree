"""Fine-tunes multilingual-e5-small on the triplets from ``build_training_data.py``.

Losses
  supervised  InfoNCE over in-batch and mined negatives (cached MNRL, so that a batch of
              128 fits in 8 GB).
  synthetic   LLM-generated queries, filtered by the cross-encoder: a query is kept only
              if the teacher ranks its source document above every sampled negative
              (round-trip consistency, as in Promptagator). The kept triples are trained
              with the same contrastive loss as the supervised ones, so the two settings
              differ only by where the queries come from.
              ``--synthetic-loss marginmse`` regresses the teacher margins instead (GPL).
              With this cosine-trained student it degrades the model; see the report.

Batches are drawn from one dataset at a time, so in-batch negatives come from the same
domain. The checkpoint kept is the one with the best mean nDCG@10 on the dev splits;
the test splits are never read here.
"""

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss, MarginMSELoss
from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers

from sensetree_research.data import DATA_DIR, IN_DOMAIN, SEED, doc_text, load_corpus, load_qrels, load_queries
from sensetree_research.dense import E5

ROOT = Path(__file__).resolve().parent.parent


def load_rows(name: str, kind: str) -> list[dict]:
    path = DATA_DIR / name / f"train_{kind}.jsonl"
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def consistent_triples(rows: list[dict], min_margin: float, rng: random.Random) -> tuple[list[dict], float]:
    """Groups the teacher-scored triples by query and keeps the consistent queries.

    Each query comes with several sampled negatives and the teacher margin
    s(q, source) - s(q, negative). A query survives if every margin is positive; its
    negative is the hardest one whose margin still exceeds ``min_margin``.
    """
    by_query: dict[str, list[dict]] = {}
    for r in rows:
        by_query.setdefault(r["query"], []).append(r)
    kept = []
    for group in by_query.values():
        if min(r["label"] for r in group) <= 0:
            continue
        clean = [r for r in group if r["label"] >= min_margin]
        if not clean:
            continue
        hardest = min(clean, key=lambda r: r["label"])
        kept.append({"query": hardest["query"], "positive": hardest["positive"], "negative": hardest["negative"]})
    rng.shuffle(kept)
    return kept, len(kept) / max(1, len(by_query))


def dev_evaluator(names: list[str], max_corpus: int) -> SequentialEvaluator:
    evaluators = []
    rng = random.Random(SEED)
    for name in names:
        qrels = load_qrels(name, "dev")
        queries = load_queries(name)
        corpus = load_corpus(name)
        relevant = {q: set(d for d, r in rels.items() if r > 0) for q, rels in qrels.items()}
        keep = set().union(*relevant.values())
        others = sorted(set(corpus) - keep)
        rng.shuffle(others)
        doc_ids = sorted(keep) + others[: max(0, max_corpus - len(keep))]
        evaluators.append(InformationRetrievalEvaluator(
            queries={q: E5.query + queries[q] for q in relevant},
            corpus={d: E5.passage + doc_text(corpus[d]) for d in doc_ids},
            relevant_docs=relevant,
            name=name,
            ndcg_at_k=[10], mrr_at_k=[10], accuracy_at_k=[1], precision_recall_at_k=[10], map_at_k=[100],
            batch_size=128, show_progress_bar=False,
        ))
    return SequentialEvaluator(evaluators, main_score_function=lambda scores: sum(scores) / len(scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["supervised", "synthetic", "both"], required=True)
    parser.add_argument("--base", default="intfloat/multilingual-e5-small")
    parser.add_argument("--out", required=True)
    parser.add_argument("--datasets", nargs="*", default=IN_DOMAIN)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--dev-corpus", type=int, default=20000)
    parser.add_argument("--synthetic-loss", choices=["contrastive", "marginmse"], default="contrastive")
    parser.add_argument("--min-margin", type=float, default=1.0)
    args = parser.parse_args()

    model = SentenceTransformer(args.base, device="cuda")
    model.max_seq_length = args.max_seq_length
    if args.mode != "supervised" and args.synthetic_loss == "marginmse":
        # MarginMSE keeps the activations of 3 x batch sequences; recomputing them in
        # the backward pass is what lets a batch of 128 fit in 8 GB.
        model[0].auto_model.gradient_checkpointing_enable()

    train, losses, retention = {}, {}, {}
    mnrl = CachedMultipleNegativesRankingLoss(model, mini_batch_size=32)
    margin = MarginMSELoss(model)
    for name in args.datasets:
        if args.mode in ("supervised", "both"):
            rows = load_rows(name, "supervised")
            train[f"{name}-sup"] = Dataset.from_list([{k: r[k] for k in ("query", "positive", "negative")} for r in rows])
            losses[f"{name}-sup"] = mnrl
        if args.mode in ("synthetic", "both"):
            rows = load_rows(name, "synthetic")
            if args.synthetic_loss == "marginmse":
                train[f"{name}-syn"] = Dataset.from_list(rows)
                losses[f"{name}-syn"] = margin
            else:
                kept, rate = consistent_triples(rows, args.min_margin, random.Random(SEED))
                retention[name] = rate
                train[f"{name}-syn"] = Dataset.from_list(kept)
                losses[f"{name}-syn"] = mnrl
    for key, ds in train.items():
        print(f"{key}: {len(ds)} triples", flush=True)
    for name, rate in retention.items():
        print(f"{name}: {rate:.1%} of synthetic queries pass the consistency filter", flush=True)

    evaluator = dev_evaluator(args.datasets, args.dev_corpus)
    out = ROOT / args.out
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_sequential_score",
        greater_is_better=True,
        logging_steps=20,
        seed=SEED,
        report_to="none",
        dataloader_num_workers=0,
    )
    before = evaluator(model)
    trainer = SentenceTransformerTrainer(
        model=model, args=training_args, train_dataset=train, loss=losses, evaluator=evaluator,
    )
    trainer.train()
    after = evaluator(model)
    model.save_pretrained(str(out / "final"))
    json.dump({"args": vars(args), "retention": retention, "dev_before": before, "dev_after": after,
               "log_history": trainer.state.log_history},
              open(out / "training.json", "w"), indent=1)
    print(f"dev mean nDCG@10: {before['sequential_score']:.4f} -> {after['sequential_score']:.4f}", flush=True)


if __name__ == "__main__":
    main()

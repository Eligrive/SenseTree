"""Distils bge-reranker-v2-m3 into a small cross-encoder.

Losses
  mse      pointwise: regress the teacher logit of every (query, passage) pair.
  listnet  listwise: match the softmax of the teacher logits over each query's
           candidates (cross-entropy between the two distributions).

The student is scored on the dev queries by reranking the same candidates the teacher
saw, against human judgements. ``--epochs 0`` only evaluates (off-the-shelf baselines).
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import ListNetLoss, MSELoss

from sensetree_research.data import DATA_DIR, IN_DOMAIN, SEED
from sensetree_research.metrics import mrr_at_k, ndcg_at_k

ROOT = Path(__file__).resolve().parent.parent


def load(name: str, split: str) -> list[dict]:
    return [json.loads(line) for line in open(DATA_DIR / name / f"rerank_{split}.jsonl", encoding="utf-8")]


def dev_scores(model: CrossEncoder, names: list[str], max_queries: int) -> dict:
    out = {}
    for name in names:
        rows = load(name, "dev")
        random.Random(SEED).shuffle(rows)
        rows = rows[:max_queries]
        pairs = [(r["query"], c["passage"]) for r in rows for c in r["candidates"]]
        t0 = time.perf_counter()
        scores = model.predict(pairs, batch_size=64, show_progress_bar=False, convert_to_numpy=True,
                               activation_fn=torch.nn.Identity())
        elapsed = time.perf_counter() - t0
        ndcg, mrr, agree, i = [], [], [], 0
        for r in rows:
            n = len(r["candidates"])
            s = scores[i:i + n]
            i += n
            ids = [c["doc_id"] for c in r["candidates"]]
            ranking = [ids[j] for j in np.argsort(-s, kind="stable")]
            rels = {d: 1 for d in r["relevant"]}
            ndcg.append(ndcg_at_k(ranking, rels, 10))
            mrr.append(mrr_at_k(ranking, rels, 10))
            teacher = np.array([c["label"] for c in r["candidates"]])
            agree.append(np.corrcoef(teacher, s)[0, 1] if n > 1 and s.std() > 0 else 0.0)
        out[name] = {"ndcg@10": float(np.mean(ndcg)), "mrr@10": float(np.mean(mrr)),
                     "teacher_corr": float(np.nanmean(agree)), "pairs_per_s": len(pairs) / elapsed}
    out["mean_ndcg@10"] = float(np.mean([out[n]["ndcg@10"] for n in names]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--loss", choices=["mse", "listnet"], default="mse")
    parser.add_argument("--datasets", nargs="*", default=IN_DOMAIN)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=64, help="pairs (mse) or queries (listnet)")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--dev-queries", type=int, default=200)
    args = parser.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    # fp32 master weights: some checkpoints are stored in fp16, which mixed-precision
    # training cannot unscale.
    model = CrossEncoder(args.student, num_labels=1, max_length=args.max_length, device="cuda",
                         model_kwargs={"dtype": torch.float32})
    before = dev_scores(model, args.datasets, args.dev_queries)
    print(f"dev before: {before['mean_ndcg@10']:.4f}", flush=True)

    after = before
    if args.epochs > 0:
        rows = [r for name in args.datasets for r in load(name, "train")]
        random.Random(SEED).shuffle(rows)
        if args.loss == "mse":
            data = Dataset.from_list([{"query": r["query"], "passage": c["passage"], "label": c["label"]}
                                      for r in rows for c in r["candidates"]])
            loss = MSELoss(model)
        else:
            data = Dataset.from_list([{"query": r["query"], "docs": [c["passage"] for c in r["candidates"]],
                                       "labels": [c["label"] for c in r["candidates"]]} for r in rows])
            loss = ListNetLoss(model)
        print(f"{len(data)} training examples", flush=True)
        trainer = CrossEncoderTrainer(
            model=model,
            args=CrossEncoderTrainingArguments(
                output_dir=str(out / "checkpoints"),
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.batch_size,
                learning_rate=args.lr,
                warmup_ratio=0.05,
                weight_decay=0.01,
                fp16=True,
                save_strategy="no",
                logging_steps=100,
                seed=SEED,
                report_to="none",
                dataloader_num_workers=0,
            ),
            train_dataset=data,
            loss=loss,
        )
        trainer.train()
        model.save_pretrained(str(out / "final"))
        after = dev_scores(model, args.datasets, args.dev_queries)
        print(f"dev after: {after['mean_ndcg@10']:.4f}", flush=True)

    json.dump({"args": vars(args), "dev_before": before, "dev_after": after},
              open(out / "training.json", "w"), indent=1)


if __name__ == "__main__":
    main()

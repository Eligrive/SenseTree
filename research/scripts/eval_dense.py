"""Evaluates a bi-encoder alone (no BM25, no reranker) at the document level.

This is the MTEB protocol: one vector per document from title + text truncated to the
model's window, exact cosine search. It isolates the embedding model, which is what
fine-tuning changes. The full SenseTree pipeline is measured by the Rust harness.
"""

import argparse
import json
from pathlib import Path

from sensetree_research.data import DATASETS, load_corpus, load_qrels, load_queries
from sensetree_research.dense import load_model, prompts_for, search
from sensetree_research.metrics import mean, per_query, write_trec_run

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=512)
    args = parser.parse_args()

    model = load_model(args.model, args.max_seq_length)
    prompts = prompts_for(args.model)
    out_dir = ROOT / "results" / "dense" / args.split / args.tag
    run_dir = ROOT / "runs" / "dense" / args.split / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        qrels = load_qrels(name, args.split)
        all_queries = load_queries(name)
        queries = {q: all_queries[q] for q in qrels}
        run, timing = search(model, load_corpus(name), queries, prompts, batch_size=args.batch_size)
        scores = per_query(run, qrels)
        summary = {m: mean(v) for m, v in scores.items()}
        write_trec_run(run, run_dir / f"{name}.trec", args.tag)
        json.dump({"model": args.model, "dataset": name, "split": args.split, "n_queries": len(qrels),
                   "metrics": summary, "timing": timing, "per_query": scores},
                  open(out_dir / f"{name}.json", "w"), indent=1)
        print(f"{args.tag:24s} {name:10s} " + " ".join(f"{m}={v:.4f}" for m, v in summary.items()), flush=True)


if __name__ == "__main__":
    main()

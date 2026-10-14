"""Scores the TREC runs written by ``examples/retrieval_eval.rs``.

Reads ``runs/pipeline/<model>/<dataset>/*.trec`` and writes
``results/pipeline/<model>/<dataset>.json`` with mean and per-query metrics for every
system, plus the harness metadata (index size, throughput, CPU latencies).
"""

import argparse
import json
from pathlib import Path

from sensetree_research.data import load_qrels
from sensetree_research.metrics import mean, per_query, read_trec_run

ROOT = Path(__file__).resolve().parent.parent
METRICS = ["ndcg@10", "recall@10", "recall@30", "mrr@10"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*")
    args = parser.parse_args()

    base = ROOT / "runs" / "pipeline"
    models = args.models or sorted(p.name for p in base.iterdir() if p.is_dir())
    for model in models:
        for ds_dir in sorted(p for p in (base / model).iterdir() if (p / "meta.json").exists()):
            meta = json.loads((ds_dir / "meta.json").read_text(encoding="utf-8"))
            qrels = load_qrels(ds_dir.name, meta["split"])
            systems = {}
            for run_path in sorted(ds_dir.glob("*.trec")):
                scores = per_query(read_trec_run(run_path), qrels, METRICS)
                systems[run_path.stem] = {"metrics": {m: mean(v) for m, v in scores.items()}, "per_query": scores}
            out = ROOT / "results" / "pipeline" / model / f"{ds_dir.name}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            json.dump({"meta": meta, "systems": systems}, open(out, "w"), separators=(",", ":"))
            line = " ".join(f"{s}={v['metrics']['ndcg@10']:.3f}" for s, v in systems.items() if "rerank" not in s or s.endswith("-rerank"))
            print(f"{model:16s} {ds_dir.name:9s} {line}")


if __name__ == "__main__":
    main()

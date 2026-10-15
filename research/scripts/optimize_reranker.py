"""Tests whether graph fusion closes the gap between the distilled reranker's FLOPs and
its measured CPU latency.

Variants of the same weights, all scored on the same dev pairs:
  exported   torch.onnx export of the default (SDPA) attention, as shipped so far
  eager      export with eager attention, a graph pattern ONNX Runtime knows how to fuse
  fused      ``eager`` passed through onnxruntime.transformers.optimizer (attention,
             LayerNorm, GELU fusions)
  fused-int8 ``fused`` with per-channel dynamic int8 weights
  exported-int8  the int8 export shipped so far, for reference

Timings alternate between variants (A/B/C/D, repeated) so that background load hits
them equally. Writes the fused models next to the original export and a JSON report.
"""

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.transformers.optimizer import optimize_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sensetree_research.data import DATA_DIR, IN_DOMAIN, SEED

ROOT = Path(__file__).resolve().parent.parent


def export(src: Path, out: Path, attn: str) -> list[str]:
    model = AutoModelForSequenceClassification.from_pretrained(str(src), attn_implementation=attn).eval()
    tok = AutoTokenizer.from_pretrained(str(src))
    sample = tok([("a query", "a passage"), ("another query", "a longer passage")], padding=True, return_tensors="pt")
    names = [n for n in ("input_ids", "attention_mask", "token_type_ids") if n in sample]

    class Scorer(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, *inputs):
            return self.inner(**dict(zip(names, inputs))).logits

    out.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(Scorer(model), tuple(sample[n] for n in names), str(out / "model.onnx"),
                      input_names=names, output_names=["logits"],
                      dynamic_axes={**{n: {0: "batch", 1: "sequence"} for n in names}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    return names


def pairs_for_timing(n_queries: int) -> list[list[tuple[str, str]]]:
    rng = random.Random(SEED)
    groups = []
    for name in IN_DOMAIN:
        rows = [json.loads(line) for line in open(DATA_DIR / name / "rerank_dev.jsonl", encoding="utf-8")]
        for r in rng.sample(rows, n_queries // len(IN_DOMAIN)):
            groups.append([(r["query"], c["passage"]) for c in r["candidates"]])
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/rr-L12-mse/final")
    parser.add_argument("--export", default="models/rr-L12-mse-onnx")
    parser.add_argument("--queries", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    src, base = ROOT / args.model, ROOT / args.export
    tok = AutoTokenizer.from_pretrained(str(base / "fp32"))
    config = json.loads((base / "fp32" / "config.json").read_text())

    eager = base / "eager"
    names = export(src, eager, "eager")
    fused = base / "fused"
    fused.mkdir(exist_ok=True)
    opt = optimize_model(str(eager / "model.onnx"), model_type="bert", num_heads=config["num_attention_heads"],
                         hidden_size=config["hidden_size"], opt_level=1)
    opt.save_model_to_file(str(fused / "model.onnx"))
    fused_int8 = base / "fused-int8"
    fused_int8.mkdir(exist_ok=True)
    # fused operators come from ONNX Runtime's contrib domain, where shape inference gives no type
    quantize_dynamic(str(fused / "model.onnx"), str(fused_int8 / "model.onnx"), weight_type=QuantType.QInt8,
                     per_channel=True, extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT})
    for d in (eager, fused, fused_int8):
        for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json", "sensetree-model.json"):
            shutil.copy(base / "fp32" / f, d / f)

    variants = {"exported": base / "fp32", "eager": eager, "fused": fused, "fused-int8": fused_int8}
    if (base / "int8" / "model.onnx").exists():
        variants["exported-int8"] = base / "int8"
    sessions = {}
    for k, d in variants.items():
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sessions[k] = ort.InferenceSession(str(d / "model.onnx"), so, providers=["CPUExecutionProvider"])

    groups = pairs_for_timing(args.queries)

    def score(sess, pairs):
        enc = tok(pairs, padding=True, truncation=True, max_length=512, return_tensors="np")
        return sess.run(["logits"], {n: enc[n].astype(np.int64) for n in names})[0][:, 0]

    reference = [score(sessions["exported"], g) for g in groups[:10]]
    report = {"fusions": opt.get_fused_operator_statistics(), "variants": {}}
    for k, sess in sessions.items():
        out = [score(sess, g) for g in groups[:10]]
        diffs = np.concatenate([np.abs(a - b) for a, b in zip(out, reference)])
        ranks = np.mean([np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1] for a, b in zip(out, reference)])
        report["variants"][k] = {"max_abs_diff": float(diffs.max()), "rank_correlation": float(ranks), "ms": []}
    for _ in range(args.repeats):
        for g in groups:
            for k, sess in sessions.items():
                t = time.perf_counter()
                score(sess, g)
                report["variants"][k]["ms"].append((time.perf_counter() - t) * 1000)
    for k, v in report["variants"].items():
        v["median_ms_per_20"] = float(np.median(v.pop("ms")))
        v["size_mb"] = (variants[k] / "model.onnx").stat().st_size / 2**20
    json.dump(report, open(base / "fusion.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

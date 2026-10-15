"""Exports a distilled cross-encoder to the folder layout SenseTree loads as a reranker.

Produces ``<out>/fp32`` and ``<out>/int8``: ``model.onnx`` with a ``logits`` output
(what fastembed reads), the tokenizer files and ``sensetree-model.json``. Parity with
PyTorch is checked on dev pairs before anything is kept.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sensetree_research.data import DATA_DIR, IN_DOMAIN, SEED

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json"]


def export(src: Path, out: Path, max_length: int) -> list[str]:
    model = AutoModelForSequenceClassification.from_pretrained(str(src)).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(src))
    sample = tokenizer([("a query", "a passage"), ("another query", "a longer passage here")],
                       padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    names = [n for n in ("input_ids", "attention_mask", "token_type_ids") if n in sample]

    class Scorer(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, *inputs):
            return self.inner(**dict(zip(names, inputs))).logits

    torch.onnx.export(
        Scorer(model), tuple(sample[n] for n in names), str(out / "model.onnx"),
        input_names=names, output_names=["logits"],
        dynamic_axes={**{n: {0: "batch", 1: "sequence"} for n in names}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    tokenizer.save_pretrained(str(out))
    model.config.save_pretrained(str(out))
    json.dump({k: str(v) for k, v in tokenizer.special_tokens_map.items()},
              open(out / "special_tokens_map.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for f in list(out.iterdir()):
        if f.name not in TOKENIZER_FILES + ["model.onnx"]:
            f.unlink() if f.is_file() else shutil.rmtree(f)
    return names


def onnx_scores(path: Path, tokenizer, pairs, names, max_length: int) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = []
    for i in range(0, len(pairs), 32):
        enc = tokenizer(pairs[i:i + 32], padding=True, truncation=True, max_length=max_length, return_tensors="np")
        out.append(session.run(["logits"], {n: enc[n].astype(np.int64) for n in names})[0][:, 0])
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    src, out = ROOT / args.model, ROOT / args.out
    fp32, int8 = out / "fp32", out / "int8"
    for d in (fp32, int8):
        d.mkdir(parents=True, exist_ok=True)
    names = export(src, fp32, args.max_length)
    quantize_dynamic(str(fp32 / "model.onnx"), str(int8 / "model.onnx"), weight_type=QuantType.QInt8, per_channel=True)
    for f in TOKENIZER_FILES:
        shutil.copy(fp32 / f, int8 / f)

    rng = random.Random(SEED)
    pairs = []
    for name in IN_DOMAIN:
        rows = [json.loads(line) for line in open(DATA_DIR / name / "rerank_dev.jsonl", encoding="utf-8")]
        for r in rng.sample(rows, 10):
            pairs += [(r["query"], c["passage"]) for c in r["candidates"][:5]]
    reference = CrossEncoder(str(src), max_length=args.max_length, device="cpu").predict(
        pairs, convert_to_numpy=True, activation_fn=torch.nn.Identity())
    tokenizer = AutoTokenizer.from_pretrained(str(fp32))

    report = {}
    for variant, folder in [("fp32", fp32), ("int8", int8)]:
        s = onnx_scores(folder / "model.onnx", tokenizer, pairs, names, args.max_length)
        report[variant] = {"max_abs_diff": float(np.abs(s - reference).max()),
                           "spearman": float(np.corrcoef(np.argsort(np.argsort(s)), np.argsort(np.argsort(reference)))[0, 1]),
                           "size_mb": (folder / "model.onnx").stat().st_size / 2**20}
        json.dump({"kind": "reranker", "max_length": args.max_length,
                   "description": f"{args.description} ({variant})".strip()},
                  open(folder / "sensetree-model.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    if report["fp32"]["max_abs_diff"] > 1e-2:
        raise SystemExit(f"fp32 export diverges from PyTorch: {report}")
    json.dump(report, open(out / "parity.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

"""Exports a fine-tuned bi-encoder to the folder layout SenseTree loads.

Produces ``<out>/fp32`` and ``<out>/int8`` (dynamic per-channel weight quantisation), each with
``model.onnx``, the tokenizer files and ``sensetree-model.json``. Parity with the
PyTorch model is checked on real corpus passages before anything is written.
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
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from sensetree_research.data import IN_DOMAIN, SEED, doc_text, load_corpus
from sensetree_research.dense import E5

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json"]


def export_encoder(src: Path, out: Path) -> None:
    """Exports the transformer to ONNX with ``last_hidden_state`` as output, the
    layout fastembed expects; pooling stays in the application."""
    model = AutoModel.from_pretrained(str(src)).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(src))
    sample = tokenizer(["passage: a short example", "passage: another one, a little longer"],
                       padding=True, return_tensors="pt")
    class Encoder(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    torch.onnx.export(
        Encoder(model),
        (sample["input_ids"], sample["attention_mask"]),
        str(out / "model.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                      "attention_mask": {0: "batch", 1: "sequence"},
                      "last_hidden_state": {0: "batch", 1: "sequence"}},
        opset_version=17,
        dynamo=False,
    )
    tokenizer.save_pretrained(str(out))
    model.config.save_pretrained(str(out))
    # transformers 5 no longer writes this file; fastembed reads it.
    json.dump({k: str(v) for k, v in tokenizer.special_tokens_map.items()},
              open(out / "special_tokens_map.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def onnx_embed(session: ort.InferenceSession, tokenizer, texts: list[str], max_length: int) -> np.ndarray:
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="np")
    feeds = {i.name: enc[i.name].astype(np.int64) for i in session.get_inputs() if i.name in enc}
    hidden = session.run(["last_hidden_state"], feeds)[0]
    mask = enc["attention_mask"][..., None].astype(np.float32)
    pooled = (hidden * mask).sum(1) / np.clip(mask.sum(1), 1e-9, None)
    return pooled / np.linalg.norm(pooled, axis=1, keepdims=True)


def sample_texts(n: int) -> list[str]:
    rng = random.Random(SEED)
    texts = []
    for name in IN_DOMAIN:
        corpus = load_corpus(name)
        ids = sorted(corpus)
        texts += [E5.passage + doc_text(corpus[d]) for d in rng.sample(ids, n // len(IN_DOMAIN))]
    return texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    src = ROOT / args.model
    out = ROOT / args.out
    fp32, int8 = out / "fp32", out / "int8"
    for d in (fp32, int8):
        d.mkdir(parents=True, exist_ok=True)

    export_encoder(src, fp32)
    for f in list(fp32.iterdir()):
        if f.name not in TOKENIZER_FILES + ["model.onnx"]:
            f.unlink() if f.is_file() else shutil.rmtree(f)
    quantize_dynamic(str(fp32 / "model.onnx"), str(int8 / "model.onnx"), weight_type=QuantType.QInt8, per_channel=True)
    for f in TOKENIZER_FILES:
        shutil.copy(fp32 / f, int8 / f)

    reference = SentenceTransformer(str(src), device="cpu")
    reference.max_seq_length = args.max_length
    tokenizer = AutoTokenizer.from_pretrained(str(fp32))
    texts = sample_texts(250)
    ref = reference.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)

    report = {}
    for variant, folder in [("fp32", fp32), ("int8", int8)]:
        session = ort.InferenceSession(str(folder / "model.onnx"), providers=["CPUExecutionProvider"])
        emb = np.concatenate([onnx_embed(session, tokenizer, texts[i:i + 32], args.max_length)
                              for i in range(0, len(texts), 32)])
        cos = (emb * ref).sum(1)
        report[variant] = {"min_cosine": float(cos.min()), "mean_cosine": float(cos.mean()),
                           "size_mb": (folder / "model.onnx").stat().st_size / 2**20}
        manifest = {"dimensions": int(ref.shape[1]), "query_prefix": E5.query, "passage_prefix": E5.passage,
                    "pooling": "mean", "max_length": args.max_length,
                    "description": f"{args.description} ({variant})".strip()}
        json.dump(manifest, open(folder / "sensetree-model.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    if report["fp32"]["min_cosine"] < 0.999:
        raise SystemExit(f"fp32 export diverges from PyTorch: {report}")
    json.dump(report, open(out / "parity.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

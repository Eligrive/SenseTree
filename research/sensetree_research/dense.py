"""Dense retrieval with sentence-transformers models (GPU side)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from .data import doc_text


@dataclass(frozen=True)
class Prompts:
    query: str
    passage: str


# Instruction prefixes each model family was trained with.
E5 = Prompts("query: ", "passage: ")
NONE = Prompts("", "")


def prompts_for(model_name: str) -> Prompts:
    return E5 if "e5" in model_name.lower() else NONE


def load_model(name: str, max_seq_length: int = 512):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name, device="cuda" if torch.cuda.is_available() else "cpu")
    model.max_seq_length = max_seq_length
    if torch.cuda.is_available():
        model.half()
    return model


def encode(model, texts: list[str], prefix: str, batch_size: int = 64) -> torch.Tensor:
    # Sorting by length keeps padding low; the order is restored afterwards.
    order = np.argsort([-len(t) for t in texts])
    sorted_texts = [prefix + texts[i] for i in order]
    emb = model.encode(sorted_texts, batch_size=batch_size, convert_to_tensor=True,
                       normalize_embeddings=True, show_progress_bar=False)
    out = torch.empty_like(emb)
    out[torch.as_tensor(order, device=emb.device)] = emb
    return out


def search(model, corpus: dict[str, dict], queries: dict[str, str], prompts: Prompts,
           top_k: int = 100, batch_size: int = 64) -> tuple[dict[str, dict[str, float]], dict]:
    doc_ids = list(corpus)
    t0 = time.perf_counter()
    doc_emb = encode(model, [doc_text(corpus[d]) for d in doc_ids], prompts.passage, batch_size)
    t_docs = time.perf_counter() - t0

    qids = list(queries)
    t0 = time.perf_counter()
    q_emb = encode(model, [queries[q] for q in qids], prompts.query, batch_size)
    t_queries = time.perf_counter() - t0

    run: dict[str, dict[str, float]] = {}
    k = min(top_k, len(doc_ids))
    for start in range(0, len(qids), 256):
        scores = q_emb[start:start + 256] @ doc_emb.T
        vals, idx = torch.topk(scores.float(), k, dim=1)
        for row, qid in enumerate(qids[start:start + 256]):
            run[qid] = {doc_ids[j]: float(v) for v, j in zip(vals[row].tolist(), idx[row].tolist())}
    timing = {"docs_per_s": len(doc_ids) / t_docs, "queries_per_s": len(qids) / t_queries}
    return run, timing

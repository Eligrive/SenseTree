"""Disk cost of SenseTree's chunk index, measured without running any model.

Builds LanceDB tables with the app's schema (``vectordb.rs``: id, path, chunk_index,
text, content_hash, mtime, 384-d float32 vector) from real chunks of FiQA, then the
full-text index the app creates on ``text`` (ASCII folding, lower case, English
stemming and stop words). Vectors are random: they have a fixed size, so their values
do not change the bytes on disk. Three table sizes separate the fixed cost from the
cost per chunk. Writes ``results/storage/index_cost.json``.

The app writes with the Rust ``lance`` crate (8.0), this script with the Python
``lancedb`` package; the file formats are the same family, so the sizes are an
estimate of the app's, not a byte-exact copy.
"""

import json
import shutil
import tempfile
import uuid
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa

from sensetree_research.data import DATA_DIR

ROOT = Path(__file__).resolve().parent.parent
DIM = 384
SIZES = [5_000, 20_000, 79_400]


def windows(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(1, len(text) - overlap), step)]


def chunks_of(name: str, limit: int) -> list[tuple[str, int, str]]:
    out = []
    for line in open(DATA_DIR / name / "corpus.jsonl", encoding="utf-8"):
        d = json.loads(line)
        path = f"C:\\Users\\someone\\Documents\\{name}\\{d['_id']}.txt"
        for i, w in enumerate(windows(((d.get("title") or "") + "\n" + (d.get("text") or "")).strip())):
            out.append((path, i, w))
            if len(out) == limit:
                return out
    return out


def size_of(path: Path) -> dict[str, int]:
    parts = {"data": 0, "indices": 0, "other": 0}
    for f in path.rglob("*"):
        if f.is_file():
            key = "data" if "/data/" in f.as_posix() else "indices" if "/_indices/" in f.as_posix() else "other"
            parts[key] += f.stat().st_size
    return parts


def build(rows: list[tuple[str, int, str]], where: Path) -> dict:
    rng = np.random.default_rng(13)
    vectors = rng.standard_normal((len(rows), DIM), dtype=np.float32)
    table = pa.table({
        "id": [str(uuid.uuid4()) for _ in rows],
        "path": [r[0] for r in rows],
        "chunk_index": pa.array([r[1] for r in rows], pa.int32()),
        "text": [r[2] for r in rows],
        "content_hash": ["%064x" % rng.integers(0, 2**62) for _ in rows],
        "mtime": pa.array([1_700_000_000] * len(rows), pa.int64()),
        "vector": pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel()), DIM),
    })
    db = lancedb.connect(str(where))
    t = db.create_table("chunks", table)
    t.create_fts_index("text", use_tantivy=False, language="English", stem=True, remove_stop_words=True,
                       ascii_folding=True, lower_case=True)
    sizes = size_of(where / "chunks.lance")
    text_bytes = sum(len(r[2].encode("utf-8")) for r in rows)
    return {"chunks": len(rows), "text_bytes": text_bytes, **sizes, "total": sum(sizes.values())}


def main() -> None:
    rows = chunks_of("fiqa", max(SIZES))
    results = []
    for n in SIZES:
        where = Path(tempfile.mkdtemp(prefix="sensetree-storage-"))
        try:
            r = build(rows[:n], where)
        finally:
            shutil.rmtree(where, ignore_errors=True)
        results.append(r)
        print(f"{n:>7} chunks: data {r['data'] / n:7.0f} B/chunk, full-text index {r['indices'] / n:6.0f} B/chunk, "
              f"text {r['text_bytes'] / n:5.0f} B/chunk, total {r['total'] / 2**20:7.1f} MiB", flush=True)
    x = np.array([r["chunks"] for r in results], dtype=float)
    fit = {k: dict(zip(("per_chunk", "fixed"), np.polyfit(x, [r[k] for r in results], 1))) for k in ("data", "indices", "total")}
    out = ROOT / "results" / "storage"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"tables": results, "linear_fit_bytes": fit, "dim": DIM, "source": "fiqa chunks, 1000/200 windows"},
              open(out / "index_cost.json", "w"), indent=1)
    print(json.dumps(fit, indent=1))


if __name__ == "__main__":
    main()

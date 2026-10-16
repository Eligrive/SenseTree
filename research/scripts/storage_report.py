"""Index size and indexing time for a disk, from the measured unit costs and a disk scan.

Inputs: ``results/storage/index_cost.json`` (``storage_index.py``) and
``results/storage/disk_<label>.json`` (``storage_disk.py``). Writes
``results/storage/report_<label>.md``.

Rows per file follow the app's defaults: chunks for documents and text, one row for an
image (plus a CLIP vector), a media file (transcription is off by default), a file with
no extractable text, over 50 MB or of an unknown type. A row costs its vector
(4 bytes per dimension) plus its text and full-text index, measured on clean tables.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "storage"
CLIP_DIM = 512
SQLITE_PER_FILE = 24 * 2**20 / 19_412   # real install: sensetree.sqlite / catalogued files
EMBED_CPU_CHUNKS_S = (8.8, 10.9)        # e5-small on the laptop CPU (results/latency-laptop)
NO_COMPACTION = 8306 / 6036             # real index: bytes per row before / after optimize()


def human(b: float) -> str:
    for unit, k in (("TB", 2**40), ("GB", 2**30), ("MB", 2**20), ("KB", 2**10)):
        if b >= k:
            return f"{b / k:.1f} {unit}"
    return f"{b:.0f} B"


def rows_of(disk: dict) -> dict:
    k, est = disk["by_kind"], disk["textual_estimates"]
    doc, txt = est.get("document", {}), est.get("text", {})
    doc_files = doc.get("files_under_cap", 0)
    lo, hi = doc.get("mean_chunks_ci95", [doc.get("mean_chunks", 0)] * 2)
    return {
        "chunks": doc.get("mean_chunks", 0) * doc_files + txt.get("total_chunks", 0),
        "chunks_low": lo * doc_files + txt.get("total_chunks", 0),
        "chunks_high": hi * doc_files + txt.get("total_chunks", 0),
        "single_rows": sum(k.get(x, {}).get("files", 0) for x in ("image", "media", "other"))
        + sum(k.get(x, {}).get("over_cap", 0) for x in ("document", "text"))
        + doc.get("share_without_text", 0) * doc_files,
        "images": k.get("image", {}).get("files", 0),
        "files": sum(v["files"] for v in k.values()),
        "llm_calls": doc_files + txt.get("files_under_cap", 0) + 2 * k.get("image", {}).get("files", 0)
        + sum(k.get(x, {}).get("files", 0) for x in ("media", "other")),
        "bytes": sum(v["bytes"] for v in k.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--scale-to-gb", type=float, default=500)
    args = parser.parse_args()
    cost = json.load(open(RES / "index_cost.json"))
    disk = json.load(open(RES / f"disk_{args.label}.json"))
    full = cost["full_chunks_alloprof"]
    short = cost["tables"][-1]
    text_full = (full["total"] / full["chunks"]) - 4 * cost["dim"]     # text + metadata + full-text index
    text_short = (short["total"] / short["chunks"]) - 4 * cost["dim"]
    r = rows_of(disk)

    def index_bytes(dim: int, chunks: float) -> float:
        return chunks * (4 * dim + text_full) + r["single_rows"] * (4 * dim + text_short) \
            + r["images"] * (4 * CLIP_DIM + 100)

    lines = [f"# Index cost of `{args.label}`", "",
             f"Scanned: {r['files']:,} files, {human(r['bytes'])}. Rows: {r['chunks']:,.0f} chunks "
             f"(95% interval {r['chunks_low']:,.0f} to {r['chunks_high']:,.0f}, from the document sample) and "
             f"{r['single_rows']:,.0f} single-row files.", "",
             "| Embedding model | chunk index | of which vectors | SQLite catalogue | total | without compaction |",
             "|---|---:|---:|---:|---:|---:|"]
    scale = args.scale_to_gb * 2**30 / r["bytes"]
    for name, dim in (("e5-small, 384-d (default)", 384), ("1024-d model (bge-m3 class)", 1024)):
        idx = index_bytes(dim, r["chunks"])
        vec = (r["chunks"] + r["single_rows"]) * 4 * dim
        sql = r["files"] * SQLITE_PER_FILE
        lines.append(f"| {name} | {human(idx)} | {human(vec)} | {human(sql)} | {human(idx + sql)} | "
                     f"{human(idx * NO_COMPACTION + sql)} |")
    lines += ["", f"Scaled to {args.scale_to_gb:.0f} GB with the same composition (x{scale:.1f}):", "",
              "| Embedding model | total index | share of the disk | embedding time, laptop CPU | LLM calls |",
              "|---|---:|---:|---:|---:|"]
    for name, dim in (("384-d", 384), ("1024-d", 1024)):
        total = (index_bytes(dim, r["chunks"]) + r["files"] * SQLITE_PER_FILE) * scale
        rows = (r["chunks"] + r["single_rows"]) * scale
        hours = [rows / s / 3600 for s in EMBED_CPU_CHUNKS_S]
        timing = f"{hours[1]:,.0f} to {hours[0]:,.0f} h" if dim == 384 else "not measured (slower)"
        lines.append(f"| {name} | {human(total)} | {100 * total / (args.scale_to_gb * 2**30):.2f}% | "
                     f"{timing} | {r['llm_calls'] * scale:,.0f} |")
    calls = r["llm_calls"] * scale
    lines += ["", f"LLM descriptions ({calls:,.0f} calls: one per document, text file, media and unknown file, two per "
              f"image) dominate the time if they are on: at an assumed 2 to 4 s per call on a GPU, "
              f"{calls * 2 / 3600:,.0f} to {calls * 4 / 3600:,.0f} h. This rate is an assumption, not a measurement."]
    out = RES / f"report_{args.label}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

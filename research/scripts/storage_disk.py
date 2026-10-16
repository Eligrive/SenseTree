"""What a real disk would cost SenseTree to index, from file metadata and a text sample.

1. Walks one or more folders with the crawler's rules (hidden and technical folders skipped, empty
   files ignored) and routes every file by extension like ``parser.rs``: document, text,
   image, media or other. Only sizes and counts are kept, never names or contents.
2. Draws a random sample of documents and text files under the 50 MB cap and extracts
   their text (pypdf for PDF, the XML of DOCX/PPTX/XLSX, UTF-8 for text), which gives the
   number of chunks per file without indexing anything.
3. Applies the app's defaults: one row per chunk for documents and text (1,000/200
   windows), one row for an image, a media file (transcription is off by default), an
   unreadable or oversized file; a CLIP vector per image; a SQLite catalogue row per file
   and a summary row per indexed file.

Combined with the measured cost of a row (``storage_index.py``), it writes
``results/storage/disk_<label>.json``.
"""

import argparse
import json
import math
import os
import random
import re
import time
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAP_BYTES = 50 * 2**20
SKIP_DIRS = {"node_modules", "target", "AppData", "Windows", "$RECYCLE.BIN", "__pycache__", "__MACOSX", ".venv"}
DOCUMENT = {"pdf", "docx", "pptx", "xlsx"}
IMAGE = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}
MEDIA = set("mp3 wav flac m4a aac ogg oga opus wma aiff aif alac amr ape au caf dsf mka mpga ra voc wv mp4 m4v mkv "
            "avi mov webm wmv mpeg mpg 3gp 3g2 asf divx f4v flv m2ts mts mxf ogv rm rmvb vob".split())
TEXT = set("txt md markdown rst csv tsv log json toml yaml yml xml html htm css scss ini cfg conf env tex bib rs py js "
           "jsx ts tsx c h cpp hpp cc cxx java kt kts go rb php sh bash zsh ps1 bat sql r swift scala lua pl pm vue "
           "svelte dart clj hs ml gradle properties".split())
TAG = re.compile(r"<[^>]+>")


def kind_of(ext: str) -> str:
    return ("document" if ext in DOCUMENT else "text" if ext in TEXT else "image" if ext in IMAGE
            else "media" if ext in MEDIA else "other")


def walk(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS]
        for name in filenames:
            if name.startswith("._") or name.lower() in ("thumbs.db", "desktop.ini", ".ds_store"):
                continue
            path = os.path.join(dirpath, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if size == 0:
                continue
            yield path, name.rsplit(".", 1)[-1].lower() if "." in name else "", size


def text_length(path: str, ext: str) -> int | None:
    try:
        if ext == "pdf":
            from pypdf import PdfReader

            return sum(len(p.extract_text() or "") for p in PdfReader(path).pages)
        if ext in ("docx", "pptx", "xlsx"):
            with zipfile.ZipFile(path) as z:
                parts = [n for n in z.namelist() if n.endswith(".xml") and (
                    n.startswith("word/document") or n.startswith("ppt/slides/slide") or n == "xl/sharedStrings.xml")]
                return sum(len(TAG.sub(" ", z.read(n).decode("utf-8", "ignore")).split()) * 6 for n in parts)
        with open(path, "rb") as f:
            return len(f.read().decode("utf-8", "ignore"))
    except Exception:
        return None


def chunks_for(chars: int) -> int:
    return max(1, math.ceil((chars - 200) / 800)) if chars > 0 else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", nargs="+", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--sample", type=int, default=400, help="documents and text files whose text is extracted")
    parser.add_argument("--time-budget", type=float, default=240, help="seconds spent extracting the sample")
    parser.add_argument("--blocks-at-depth", type=int, default=0,
                        help="scenario: every folder at this depth under a root is indexed as one block")
    args = parser.parse_args()

    t0 = time.time()
    by_kind = defaultdict(lambda: {"files": 0, "bytes": 0, "over_cap": 0})
    by_ext = defaultdict(lambda: [0, 0])
    textual = []
    exact_text = {"files": 0, "chunks": 0, "extract_chars": 0}
    block_chunks = defaultdict(int)
    for path, ext, size in (f for r in args.root for f in walk(Path(r))):
        k = kind_of(ext)
        by_kind[k]["files"] += 1
        by_kind[k]["bytes"] += size
        by_ext[ext][0] += 1
        by_ext[ext][1] += size
        if k in ("document", "text"):
            if size > CAP_BYTES:
                by_kind[k]["over_cap"] += 1
            elif k == "text":
                # UTF-8 text: characters ~ bytes, so the chunk count is exact from the size
                exact_text["files"] += 1
                exact_text["chunks"] += chunks_for(size)
                exact_text["extract_chars"] += min(size, 16_000)
                if args.blocks_at_depth:
                    rel = Path(path).relative_to(next(r for r in args.root if path.startswith(str(Path(r)))))
                    if len(rel.parts) > args.blocks_at_depth:
                        block_chunks[rel.parts[:args.blocks_at_depth]] += chunks_for(size)
            else:
                textual.append((path, ext, k))
    scan_s = time.time() - t0

    rng = random.Random(13)
    sample = rng.sample(textual, min(args.sample, len(textual)))
    measured = defaultdict(list)
    t1 = time.time()
    for path, ext, k in sample:
        if time.time() - t1 > args.time_budget:
            break
        n = text_length(path, ext)
        if n is not None:
            measured[k].append(n)
    extract_s = time.time() - t1

    est = {}
    if exact_text["files"]:
        est["text"] = {"files_under_cap": exact_text["files"], "total_chunks": exact_text["chunks"],
                       "mean_chunks": exact_text["chunks"] / exact_text["files"],
                       "mean_extract_chars": exact_text["extract_chars"] / exact_text["files"],
                       "share_without_text": 0.0, "method": "exact, from file sizes"}
    for k in ("document",):
        lens = measured.get(k, [])
        n_files = by_kind[k]["files"] - by_kind[k]["over_cap"]
        if not lens or not n_files:
            continue
        est[k] = {
            "sampled_files": len(lens),
            "mean_chars": sum(lens) / len(lens),
            "mean_chunks": sum(chunks_for(x) for x in lens) / len(lens),
            "share_without_text": sum(x == 0 for x in lens) / len(lens),
            "mean_extract_chars": sum(min(x, 16_000) for x in lens) / len(lens),
            "files_under_cap": n_files,
        }
        per_file = [chunks_for(x) for x in lens]
        boot = sorted(sum(rng.choices(per_file, k=len(per_file))) / len(per_file) for _ in range(2000))
        est[k]["mean_chunks_ci95"] = [boot[50], boot[1949]]
        est[k]["total_chunks"] = est[k]["mean_chunks"] * n_files
    if args.blocks_at_depth:
        est["blocks"] = {"depth": args.blocks_at_depth, "folders": len(block_chunks),
                         "chunks_replaced": sum(block_chunks.values())}
    top_ext = sorted(by_ext.items(), key=lambda kv: -kv[1][1])[:25]
    out = {"label": args.label, "scan_seconds": scan_s, "extract_seconds": extract_s,
           "by_kind": dict(by_kind), "textual_estimates": est,
           "top_extensions_by_bytes": [{"ext": e, "files": v[0], "bytes": v[1]} for e, v in top_ext]}
    (ROOT / "results" / "storage").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / "results" / "storage" / f"disk_{args.label}.json", "w"), indent=1)
    total = sum(v["bytes"] for v in by_kind.values())
    print(f"scanned {sum(v['files'] for v in by_kind.values()):,} files, {total / 2**30:.1f} GiB in {scan_s:.0f} s; "
          f"text extracted from {sum(len(v) for v in measured.values())} files in {extract_s:.0f} s")
    for k, v in sorted(by_kind.items(), key=lambda kv: -kv[1]["bytes"]):
        print(f"  {k:9s} {v['files']:>9,} files {v['bytes'] / 2**30:8.2f} GiB  over 50 MB: {v['over_cap']}")
    print(json.dumps(est, indent=1))


if __name__ == "__main__":
    main()

"""Copies the harness metadata of the CPU timing runs into results/.

runs/latency/        desktop, Intel i7-9700K (8 cores, no VNNI)
runs/latency-laptop/ laptop, Intel i7-1185G7 (4 cores / 8 threads, AVX-512 VNNI)
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for machine in ("latency", "latency-laptop"):
    for meta in sorted((ROOT / "runs" / machine).glob("*/*/meta.json")):
        out = ROOT / "results" / machine / meta.parent.parent.name / f"{meta.parent.name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(meta.read_text(encoding="utf-8"), encoding="utf-8")
        print(out.relative_to(ROOT))

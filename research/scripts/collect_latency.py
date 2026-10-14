"""Copies the harness metadata of the CPU timing runs into results/latency/."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for meta in sorted((ROOT / "runs" / "latency").glob("*/*/meta.json")):
    out = ROOT / "results" / "latency" / meta.parent.parent.name / f"{meta.parent.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(meta.read_text(encoding="utf-8"), encoding="utf-8")
    print(out.relative_to(ROOT))

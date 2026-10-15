"""Scores agent runs (``runs/agent/*.jsonl``) and writes ``results/agent/``.

For each run: success rate per task family, unrequested deletions, malformed and failed
tool calls, tool calls and seconds per task. Runs are compared pairwise on the tasks
they share (bootstrap interval and randomisation test on per-task success).

A task whose answer is an error from the model server (runner crash, out of memory) says
nothing about the model: it is counted as an infrastructure error and left out.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from sensetree_research.agent_score import score
from sensetree_research.data import DATA_DIR
from sensetree_research.stats import compare

ROOT = Path(__file__).resolve().parent.parent
FAMILIES = ["find", "extract", "aggregate", "organise", "rename", "cleanup", "safety"]


def server_error(record: dict) -> bool:
    return (record.get("answer") or "").startswith("Erreur du modèle")


def load_runs(names: list[str] | None, folder: Path) -> tuple[dict[str, dict[str, dict]], dict[str, int]]:
    runs, errors = {}, {}
    for path in sorted(folder.glob("*.jsonl")):
        if path.stem.startswith("tasks"):
            continue
        if names and path.stem not in names:
            continue
        records, errors[path.stem] = {}, 0
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if server_error(r):
                errors[path.stem] += 1
                continue
            records[r["id"]] = r
        runs[path.stem] = records
    return runs, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*")
    parser.add_argument("--baseline")
    parser.add_argument("--tasks", default=str(DATA_DIR / "fsbench" / "tasks.jsonl"),
                        help="task file whose paths match the runs (Kaggle runs use Linux paths)")
    parser.add_argument("--runs-dir", default=str(ROOT / "runs" / "agent"))
    args = parser.parse_args()

    tasks = {}
    for line in open(args.tasks, encoding="utf-8"):
        t = json.loads(line)
        tasks[t["id"]] = t
    runs, errors = load_runs(args.runs, Path(args.runs_dir))
    out_dir = ROOT / "results" / "agent"
    out_dir.mkdir(parents=True, exist_ok=True)

    scored = {}
    for name, records in runs.items():
        per_task = {tid: score(tasks[tid], r) for tid, r in records.items() if tid in tasks}
        scored[name] = per_task
        fam = defaultdict(list)
        for tid, s in per_task.items():
            fam[s["family"]].append(s["success"])
        summary = {
            "tasks": len(per_task),
            "server_errors_excluded": errors[name],
            "success": float(np.mean([s["success"] for s in per_task.values()])),
            "by_family": {f: float(np.mean(v)) for f, v in fam.items()},
            "unrequested_deletions_per_task": float(np.mean([s["unrequested_deletions"] for s in per_task.values()])),
            "tasks_with_unrequested_deletion": float(np.mean([s["unrequested_deletions"] > 0 for s in per_task.values()])),
            "malformed_calls": int(sum(s["malformed_calls"] for s in per_task.values())),
            "failed_call_rate": float(sum(s["failed_calls"] for s in per_task.values())
                                      / max(1, sum(s["tool_calls"] for s in per_task.values()))),
            "tool_calls_per_task": float(np.mean([s["tool_calls"] for s in per_task.values()])),
            "seconds_per_task": float(np.median([s["seconds"] for s in per_task.values()])),
            "failure_modes": dict(Counter(s["diagnosis"] for s in per_task.values() if s["diagnosis"] != "ok").most_common()),
        }
        json.dump({"summary": summary, "per_task": per_task}, open(out_dir / f"{name}.json", "w"), indent=1)
        fam_line = " ".join(f"{f}={summary['by_family'].get(f, float('nan')):.2f}" for f in FAMILIES)
        print(f"{name:28s} n={summary['tasks']:3d} success={summary['success']:.3f} {fam_line} "
              f"unreq_del={summary['tasks_with_unrequested_deletion']:.2f} calls={summary['tool_calls_per_task']:.1f} "
              f"t={summary['seconds_per_task']:.0f}s")
        print(f"{'':28s} failures: {summary['failure_modes']}")

    if args.baseline and args.baseline in scored:
        base = {t: s["success"] for t, s in scored[args.baseline].items()}
        for name, per_task in scored.items():
            if name == args.baseline:
                continue
            other = {t: s["success"] for t, s in per_task.items()}
            shared = sorted(set(base) & set(other))
            c = compare({t: base[t] for t in shared}, {t: other[t] for t in shared})
            print(f"{name} vs {args.baseline}: Δ success {c.delta:+.3f} [{c.ci_low:+.3f}, {c.ci_high:+.3f}] p={c.p_value:.3f} (n={c.n})")


if __name__ == "__main__":
    main()

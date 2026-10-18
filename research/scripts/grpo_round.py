"""One round of iterative GRPO for the file agent: task sampling, collection of attempts,
rewards and advantages.

  sample   draws the round's training tasks, stratified by family -> runs/grpo/r<n>/round.json
  collect  runs <attempts> attempts per task through the app's agent loop (agent_eval),
           several at a time on one or more Ollama servers. Every attempt of a persona has
           its own copy of the persona's agent state (index, durable memory), so attempts
           never see each other's memories. Resumable: finished tasks are skipped.
  eval     the test people at the app's temperature, for the final comparison
  rewards  scores every attempt with the benchmark's checker, turns it into a reward and
           a group-normalised advantage -> results/training/grpo/r<n>-rewards.json

Reward: 1 for a solved task, 0 otherwise; an organise plan that is not fully right earns
0.5 x purity x coverage; any unrequested deletion (which includes every protected file)
gives -1 whatever the rest. Advantage: (reward - group mean) / group std over the attempts
of the same task; groups whose rewards are all equal carry no signal and are dropped.

Collection runs on the GPU PC (Windows) and in a Kaggle kernel (Linux) at the same time;
each site writes runs/grpo/r<n>/<site>/a<k>/<persona>.jsonl next to its own task file
(paths differ between the sites).
"""

import argparse
import json
import os
import queue
import random
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAMILIES = ["find", "extract", "aggregate", "organise", "rename", "cleanup", "safety"]
# tasks per family and round: the families the base agent already solves weigh less
PER_FAMILY = {"find": 16, "extract": 24, "aggregate": 16, "organise": 16, "rename": 16, "cleanup": 24, "safety": 16}


def load_tasks(path: Path) -> dict[str, dict]:
    return {t["id"]: t for t in (json.loads(l) for l in open(path, encoding="utf-8"))}


def server_error(record: dict) -> bool:
    return (record.get("answer") or "").startswith("Erreur du modèle")


# ---------------------------------------------------------------- sample
def sample(args) -> None:
    tasks = [t for t in load_tasks(Path(args.tasks)).values() if t["split"] == "train"]
    rng = random.Random(1000 + args.round)
    ids = []
    for fam in FAMILIES:
        pool = sorted(t["id"] for t in tasks if t["family"] == fam)
        ids += rng.sample(pool, min(round(PER_FAMILY[fam] * args.fraction), len(pool)))
    out = Path(args.runs) / f"r{args.round}" / "round.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"round": args.round, "ids": sorted(ids), "fraction": args.fraction, "per_family": PER_FAMILY},
              open(out, "w"), indent=1)
    print(f"round {args.round}: {len(ids)} tasks -> {out}")


# ---------------------------------------------------------------- collect
def link_dir(src: Path, dst: Path) -> None:
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)], check=True, stdout=subprocess.DEVNULL)
    else:
        os.symlink(src, dst, target_is_directory=True)


def attempt_state(base: Path, models: Path, dst: Path, memory: bool = False) -> None:
    """A private copy of an indexed persona state: index and settings copied, the
    embedding models (2 GB, read only) shared through a link, durable memory empty
    unless ``memory`` (evaluation: same starting state as the baseline runs)."""
    if (dst / "indexed.json").exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for name in ("lancedb", "ort"):
        if (base / name).exists():
            shutil.copytree(base / name, dst / name)
    shutil.copy(base / "config.json", dst / "config.json")
    if memory and (base / "sensetree.db").exists():
        shutil.copy(base / "sensetree.db", dst / "sensetree.db")
    link_dir(models, dst / "models")
    shutil.copy(base / "indexed.json", dst / "indexed.json")


def finished(out: Path, ids: set[str]) -> bool:
    if not out.exists():
        return False
    done = set()
    for line in open(out, encoding="utf-8"):
        r = json.loads(line)
        if not server_error(r):
            done.add(r["id"])
    return ids <= done


def drop_errors(out: Path) -> None:
    if out.exists():
        lines = [l for l in open(out, encoding="utf-8") if not server_error(json.loads(l))]
        out.write_text("".join(lines), encoding="utf-8")


def collect(args) -> None:
    round_ids = set(json.load(open(Path(args.runs) / f"r{args.round}" / "round.json"))["ids"])
    site_dir = Path(args.runs) / f"r{args.round}" / args.site
    site_dir.mkdir(parents=True, exist_ok=True)
    tasks = {i: t for i, t in load_tasks(Path(args.tasks)).items() if i in round_ids}
    task_file = site_dir / "tasks.jsonl"
    task_file.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks.values()), encoding="utf-8")
    by_persona = defaultdict(set)
    for t in tasks.values():
        by_persona[t["persona"]].add(t["id"])
    personas = args.personas or sorted(by_persona)
    jobs = [(p, k) for p in personas if p in by_persona for k in range(args.attempts)]
    log = open(site_dir / "collect.log", "a", encoding="utf-8")
    lock = threading.Lock()

    def say(msg: str) -> None:
        with lock:
            log.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
            log.flush()
            print(msg, flush=True)

    say(f"{len(jobs)} jobs ({len(personas)} personas x {args.attempts} attempts), model {args.model}, ports {args.ports}")
    stop = site_dir / "stop"
    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)

    def worker(port: int) -> None:
        while not stop.exists():
            try:
                persona, k = q.get_nowait()
            except queue.Empty:
                return
            out = site_dir / f"a{k}" / f"{persona}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            ids = by_persona[persona]
            for _ in range(3):  # a crashed server leaves error answers: retried
                if finished(out, ids):
                    break
                drop_errors(out)
                state = Path(args.work) / f"r{args.round}" / f"a{k}" / persona
                attempt_state(Path(args.base_state) / persona, Path(args.models), state)
                t0 = time.time()
                with open(site_dir / f"a{k}" / f"{persona}.log", "a", encoding="utf-8") as lf:
                    subprocess.run([args.exe, "--root", str(Path(args.fsbench) / persona), "--state", str(state),
                                    "--tasks", str(task_file), "--split", "train", "--model", args.model,
                                    "--effort", args.effort, "--temperature", str(args.temperature),
                                    "--base-url", f"http://127.0.0.1:{port}/v1", "--out", str(out), "--gpu", "false"],
                                   stdout=lf, stderr=subprocess.STDOUT)
                say(f"{persona} a{k} port {port}: {len(ids)} tasks, {time.time() - t0:.0f} s, "
                    f"{'ok' if finished(out, ids) else 'INCOMPLETE'}")

    threads = [threading.Thread(target=worker, args=(port,)) for port in args.ports for _ in range(args.workers)]
    for t in threads:
        t.start()
        time.sleep(2)
    for t in threads:
        t.join()
    complete = sum(finished(site_dir / f"a{k}" / f"{p}.jsonl", by_persona[p]) for p, k in jobs)
    say(f"done: {complete}/{len(jobs)} jobs complete")


def evaluate(args) -> None:
    """The test people through the agent loop at the app's temperature (0.2), thinking on,
    several people at a time; each person starts from a copy of its baseline state."""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    q: queue.Queue = queue.Queue()
    for p in args.personas:
        q.put(p)

    def worker(port: int) -> None:
        while True:
            try:
                persona = q.get_nowait()
            except queue.Empty:
                return
            part = out.with_name(f"{out.stem}.{persona}.jsonl")
            for _ in range(3):
                drop_errors(part)
                state = Path(args.work) / out.stem / persona
                attempt_state(Path(args.base_state) / persona, Path(args.models), state, memory=True)
                with open(out.with_name(f"{out.stem}.{persona}.log"), "a", encoding="utf-8") as lf:
                    subprocess.run([args.exe, "--root", str(Path(args.fsbench) / persona), "--state", str(state),
                                    "--tasks", args.tasks, "--split", "test", "--model", args.model,
                                    "--effort", args.effort, "--base-url", f"http://127.0.0.1:{port}/v1",
                                    "--out", str(part), "--gpu", "false"], stdout=lf, stderr=subprocess.STDOUT)
                if part.exists() and not any(server_error(json.loads(l)) for l in open(part, encoding="utf-8")):
                    break
            print(f"{persona} done", flush=True)

    threads = [threading.Thread(target=worker, args=(port,)) for port in args.ports for _ in range(args.workers)]
    for t in threads:
        t.start()
        time.sleep(2)
    for t in threads:
        t.join()
    with open(out, "w", encoding="utf-8") as merged:
        for p in args.personas:
            part = out.with_name(f"{out.stem}.{p}.jsonl")
            if part.exists():
                merged.write(part.read_text(encoding="utf-8"))
    print(f"-> {out}")


# ---------------------------------------------------------------- rewards
def reward_of(s: dict) -> float:
    if s["unrequested_deletions"] > 0:
        return -1.0
    if s["success"] == 1.0:
        return 1.0
    if s["family"] == "organise" and s.get("proposed_plan"):
        return 0.5 * s.get("purity", 0.0) * s.get("coverage", 0.0)
    return 0.0


def rewards(args) -> None:
    sys.path.insert(0, str(ROOT))
    from sensetree_research.agent_score import score

    rdir = Path(args.runs).resolve() / f"r{args.round}"
    attempts, seen = [], set()
    for site_dir in sorted(p for p in rdir.iterdir() if p.is_dir()):
        tasks = load_tasks(site_dir / "tasks.jsonl")
        for f in sorted(site_dir.glob("a*/*.jsonl")):
            k = int(f.parent.name[1:])
            for n, line in enumerate(open(f, encoding="utf-8")):
                r = json.loads(line)
                if server_error(r) or r["id"] not in tasks or (r["id"], k) in seen:
                    continue
                seen.add((r["id"], k))
                s = score(tasks[r["id"]], r)
                attempts.append({"id": r["id"], "family": s["family"], "attempt": k, "site": site_dir.name,
                                 "file": str(f.relative_to(ROOT)).replace("\\", "/"), "line": n,
                                 "reward": reward_of(s), "success": s["success"],
                                 "unrequested_deletion": s["unrequested_deletions"] > 0,
                                 "tool_calls": s["tool_calls"], "seconds": s["seconds"],
                                 "forced_answer": any(m.get("role") == "system" for m in (r.get("conversation") or [])[1:])})
    groups = defaultdict(list)
    for a in attempts:
        groups[a["id"]].append(a)
    train, zero_var = [], 0
    for tid, g in groups.items():
        rs = [a["reward"] for a in g]
        mean = sum(rs) / len(rs)
        std = (sum((r - mean) ** 2 for r in rs) / len(rs)) ** 0.5
        if len(g) < 2 or std < 1e-6:
            zero_var += 1
            continue
        for a in g:
            a["advantage"] = (a["reward"] - mean) / std
            train.append(a)
    fam = defaultdict(list)
    for a in attempts:
        fam[a["family"]].append(a)
    summary = {
        "round": args.round, "attempts": len(attempts), "tasks": len(groups),
        "attempts_per_task": dict(Counter(len(g) for g in groups.values())),
        "mean_reward": sum(a["reward"] for a in attempts) / max(1, len(attempts)),
        "success": sum(a["success"] for a in attempts) / max(1, len(attempts)),
        "unrequested_deletion_rate": sum(a["unrequested_deletion"] for a in attempts) / max(1, len(attempts)),
        "by_family": {f: {"n": len(v), "success": sum(a["success"] for a in v) / len(v),
                          "mean_reward": sum(a["reward"] for a in v) / len(v)} for f, v in sorted(fam.items())},
        "groups_with_signal": len(groups) - zero_var, "groups_without_signal": zero_var,
        "training_attempts": len(train), "median_seconds": sorted(a["seconds"] for a in attempts)[len(attempts) // 2]
        if attempts else 0.0,
    }
    out = ROOT / "results" / "training" / "grpo" / f"r{args.round}-rewards.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "attempts": attempts, "train": train}, open(out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("sample", "collect", "rewards", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--round", type=int, required=name != "eval")
        p.add_argument("--runs", default=str(ROOT / "runs" / "grpo"))
        p.add_argument("--tasks", default=str(ROOT / "data" / "fsbench" / "tasks.jsonl"))
    sub.choices["sample"].add_argument("--fraction", type=float, default=1.0,
                                       help="share of the per-family counts to draw (time budget)")
    for c in (sub.choices["collect"], sub.choices["eval"]):
        c.add_argument("--exe", required=True)
        c.add_argument("--fsbench", required=True, help="folder holding the persona homes")
        c.add_argument("--base-state", required=True, help="folder holding each persona's indexed agent state")
        c.add_argument("--models", required=True, help="embedding models folder shared by every attempt")
        c.add_argument("--work", required=True, help="where the per-attempt states are created")
        c.add_argument("--model", required=True)
        c.add_argument("--ports", type=int, nargs="+", default=[11510])
        c.add_argument("--workers", type=int, default=2, help="concurrent attempts per Ollama server")
        c.add_argument("--effort", default="medium")
        c.add_argument("--personas", nargs="*", help="personas to run, in this order (default: all of the round)")
    e = sub.choices["eval"]
    e.add_argument("--out", required=True, help="runs/agent/<tag>.jsonl")
    c = sub.choices["collect"]
    c.add_argument("--site", required=True)
    c.add_argument("--attempts", type=int, default=4)
    c.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    {"sample": sample, "collect": collect, "rewards": rewards, "eval": evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()

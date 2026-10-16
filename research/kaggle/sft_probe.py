"""Kaggle job (1 x T4): what fills the GPU when fine-tuning Qwen3.5-4B on agent trajectories.

Five full runs ended with the same out-of-memory error (2.03 GiB requested, 11.09 GiB
allocated, 1.08 GiB reserved but unused) whatever the sequence cap and the data order.
This job trains four steps on the longest sequences, recording every CUDA allocation,
under two allocator settings: the default, and expandable segments, which lets the
reserved-but-unused memory be reused. On failure, the largest live blocks and the code
that allocated them go to ``memory-<setting>.json``.

Inputs  sensetree-src dataset (research/ with runs/agent/train-oracle.jsonl and its scores)
Output  /kaggle/working/probe-<setting>.log, memory-<setting>.json
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

OUT = Path("/kaggle/working")
WORK = Path("/tmp/sensetree")
RESEARCH = WORK / "research"


def sh(cmd: str, **kw) -> int:
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, **kw).returncode


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    src = next(p for p in Path("/kaggle/input").rglob("package.json") if "node_modules" not in str(p)).parent
    if (src / "research").is_dir():
        shutil.copytree(src / "research", RESEARCH)
    else:
        sh(f"mkdir -p {WORK}/_x && tar -xf {src / 'research.tar'} -C {WORK}/_x")
        shutil.move(str(WORK / "_x" / "research") if (WORK / "_x" / "research").is_dir() else str(WORK / "_x"),
                    str(RESEARCH))
    sh("pip install -q peft bitsandbytes accelerate datasets 'transformers==5.19.0' flash-linear-attention")
    sh(f"{sys.executable} -m pip install -q -e {RESEARCH} --no-deps")
    base = (f"cd {RESEARCH} && {sys.executable} scripts/train_agent.py --trajectories train-oracle --per-family 40 "
            f"--max-steps 4 --grad-accum 1 --longest-first")
    for name, extra in [("default", {}), ("expandable", {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})]:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", **extra)
        code = sh(f"{base} --out {OUT}/probe-{name} --memory-report {OUT}/memory-{name}.json "
                  f"> {OUT}/probe-{name}.log 2>&1", env=env)
        print(f"{name}: exit code {code}", flush=True)
        shutil.rmtree(OUT / f"probe-{name}" / "checkpoints", ignore_errors=True)


if __name__ == "__main__":
    main()

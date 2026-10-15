"""Kaggle job (2 x T4): agent SFT on Linux, independent of the run on the Windows machine.

1. Builds the app's agent harness and the file-system benchmark (Linux paths).
2. Records the scripted expert's demonstrations on the train personas (``oracle_agent.py``,
   through the real agent loop) and scores them.
3. GPU 0 fine-tunes Qwen3.5-4B (QLoRA) on the successful ones while GPU 1 evaluates the
   base model on the test personas, both served by ``serve_hf.py``.
4. GPU 0 then evaluates the fine-tuned adapter on the same test personas.

Inputs  sensetree-src dataset (src-tauri/, dist/, package.json, research/)
Output  /kaggle/working/runs/agent/*.jsonl, results-agent/, models/agent-sft/adapter, logs/
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

OUT = Path("/kaggle/working")          # saved with the run: results only
WORK = Path("/tmp/sensetree")           # build, data, models: not saved
APP = WORK / "app"
RESEARCH = WORK / "research"
FS = WORK / "fsbench"
BASE = "Qwen/Qwen3.5-4B"
START = time.time()
BUDGET_S = 11.0 * 3600
PER_FAMILY = int(os.environ.get("SENSETREE_PER_FAMILY", "40"))
EPOCHS = float(os.environ.get("SENSETREE_EPOCHS", "1"))
SKIP_BASE = os.environ.get("SENSETREE_SKIP_BASE") == "1"   # base already evaluated by another run
TAG = os.environ.get("SENSETREE_TAG", "sft")


def sh(cmd: str, **kw) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


def setup() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    sh("apt-get update -qq && apt-get install -y -qq pkg-config libssl-dev libxdo-dev unzip zstd "
       "libwebkit2gtk-4.1-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev > /dev/null")
    protoc = WORK / "protoc"
    if not protoc.exists():
        sh(f"curl -sSL -o /tmp/protoc.zip https://github.com/protocolbuffers/protobuf/releases/download/v29.3/"
           f"protoc-29.3-linux-x86_64.zip && unzip -q /tmp/protoc.zip -d {protoc}")
    os.environ["PROTOC"] = str(protoc / "bin/protoc")
    os.environ["PROTOC_INCLUDE"] = str(protoc / "include")
    ort = WORK / "onnxruntime-linux-x64-1.20.0"
    if not ort.exists():
        sh(f"curl -sSL https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz"
           f" | tar -xz -C {WORK}")
    os.environ["ORT_DYLIB_PATH"] = str(ort / "lib/libonnxruntime.so.1.20.0")
    if not Path.home().joinpath(".cargo/bin/cargo").exists():
        sh("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal")
    if not APP.exists():
        APP.mkdir(parents=True)
        src = next(p for p in Path("/kaggle/input").rglob("package.json") if "node_modules" not in str(p)).parent
        for name, dest in [("src-tauri", APP / "src-tauri"), ("dist", APP / "dist"), ("research", RESEARCH)]:
            if (src / name).is_dir():
                shutil.copytree(src / name, dest)
            else:
                tmp = WORK / "_extract" / name
                tmp.mkdir(parents=True, exist_ok=True)
                sh(f"tar -xf {src / (name + '.tar')} -C {tmp}")
                shutil.move(str(tmp / name) if (tmp / name).is_dir() else str(tmp), str(dest))
        shutil.copy(src / "package.json", APP / "package.json")
    env = dict(os.environ, PATH=f"{Path.home()}/.cargo/bin:" + os.environ["PATH"])
    sh("cargo build --release --example agent_eval", cwd=APP / "src-tauri", env=env)
    sh("pip install -q fpdf2 python-docx peft bitsandbytes accelerate datasets 'transformers==5.19.0' "
       "flash-linear-attention")
    sh(f"{sys.executable} -m pip install -q -e {RESEARCH} --no-deps")
    sh(f"cd {RESEARCH} && {sys.executable} scripts/build_fsbench.py --root {FS}")
    sh(f"cd {RESEARCH}/scripts && {sys.executable} fsbench_docs.py --root {FS}")
    sh(f"{sys.executable} -c \"from huggingface_hub import snapshot_download; snapshot_download('{BASE}')\"")


def agent(tag: str, model: str, split: str, port: int, personas: list[Path]) -> None:
    exe = APP / "src-tauri/target/release/examples/agent_eval"
    out = OUT / "runs/agent" / f"{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    for home in personas:
        if time.time() - START > BUDGET_S:
            print(f"time budget reached, {home.name} skipped", flush=True)
            continue
        subprocess.run([str(exe), "--root", str(home), "--state", str(WORK / "agent-state" / home.name),
                        "--tasks", str(RESEARCH / "data/fsbench/tasks.jsonl"), "--split", split, "--model", model,
                        "--effort", "none", "--base-url", f"http://127.0.0.1:{port}/v1", "--out", str(out),
                        "--gpu", "false"], stdout=open(OUT / f"logs/agent_{tag}.log", "a"), stderr=subprocess.STDOUT)


def personas(split: str) -> list[Path]:
    return sorted(p for p in FS.iterdir() if p.is_dir() and p.name.startswith(split))


def server(script: str, port: int, gpu: int | None, extra: list[str]) -> subprocess.Popen:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu))
    log = OUT / f"logs/{Path(script).stem}_{port}.log"
    p = subprocess.Popen([sys.executable, str(RESEARCH / "scripts" / script), "--port", str(port)] + extra,
                         cwd=RESEARCH, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    for _ in range(360):
        time.sleep(5)
        text = log.read_text(errors="ignore")
        if "serving" in text or "oracle on" in text:
            return p
        if p.poll() is not None:
            break
    raise RuntimeError(f"{script} on port {port} did not start, see {log}")


def demonstrations() -> None:
    """Oracle trajectories on the train personas, two harness processes in parallel."""
    oracle = server("oracle_agent.py", 11700, None, [])
    train = personas("train")
    halves = [threading.Thread(target=agent, args=(f"train-oracle.part{i}", "oracle", "train", 11700, train[i::2]))
              for i in range(2)]
    for t in halves:
        t.start()
    for t in halves:
        t.join()
    oracle.terminate()
    runs = RESEARCH / "runs/agent"
    runs.mkdir(parents=True, exist_ok=True)
    with open(runs / "train-oracle.jsonl", "w", encoding="utf-8") as merged:
        for i in range(2):
            merged.write((OUT / "runs/agent" / f"train-oracle.part{i}.jsonl").read_text(encoding="utf-8"))
    sh(f"cd {RESEARCH} && {sys.executable} scripts/agent_report.py --runs train-oracle")


def main() -> None:
    setup()
    demonstrations()
    sft_tag = f"test-hf-qwen3.5-4b-{TAG}"
    servers, base_eval = [], None
    # Models are loaded one at a time: two concurrent loads of the 4B model (training and
    # the base server) exceeded the VM's memory and killed the session without a log.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    sh(f"cd {RESEARCH} && {sys.executable} scripts/train_agent.py --trajectories train-oracle --model {BASE} "
       f"--out {OUT / 'models/agent-sft'} --per-family {PER_FAMILY} --epochs {EPOCHS} "
       f"> {OUT / 'logs/train.log'} 2>&1", env=env)

    adapter = ["--model", BASE, "--adapter", str(OUT / "models/agent-sft/adapter")]
    if SKIP_BASE:
        # both GPUs serve the adapter, each evaluates half of the test people
        servers += [server("serve_hf.py", 11600 + g, g, adapter) for g in (0, 1)]
        test = personas("test")
        halves = [threading.Thread(target=agent, args=(f"{sft_tag}.part{g}", f"qwen3.5-4b-{TAG}", "test", 11600 + g,
                                                       test[g::2])) for g in (0, 1)]
        for t in halves:
            t.start()
        for t in halves:
            t.join()
        with open(OUT / "runs/agent" / f"{sft_tag}.jsonl", "w", encoding="utf-8") as merged:
            for g in (0, 1):
                part = OUT / "runs/agent" / f"{sft_tag}.part{g}.jsonl"
                if part.exists():
                    merged.write(part.read_text(encoding="utf-8"))
    else:
        servers.append(server("serve_hf.py", 11601, 1, ["--model", BASE]))
        servers.append(server("serve_hf.py", 11600, 0, adapter))
        base_eval = threading.Thread(target=agent, args=("test-hf-qwen3.5-4b", "qwen3.5-4b-base", "test", 11601,
                                                         personas("test")))
        base_eval.start()
        agent(sft_tag, f"qwen3.5-4b-{TAG}", "test", 11600, personas("test"))
        base_eval.join()
    for p in servers:
        p.terminate()
    shutil.rmtree(OUT / "models/agent-sft/checkpoints", ignore_errors=True)
    shutil.copy(RESEARCH / "data/fsbench/tasks.jsonl", OUT / "runs/agent/tasks-kaggle.jsonl")
    runs = f"{sft_tag}" if SKIP_BASE else f"test-hf-qwen3.5-4b {sft_tag} --baseline test-hf-qwen3.5-4b"
    sh(f"cd {RESEARCH} && {sys.executable} scripts/agent_report.py --runs {runs} "
       f"--runs-dir {OUT / 'runs/agent'} --tasks {OUT / 'runs/agent/tasks-kaggle.jsonl'}")
    shutil.copytree(RESEARCH / "results/agent", OUT / "results-agent", dirs_exist_ok=True)


if __name__ == "__main__":
    main()

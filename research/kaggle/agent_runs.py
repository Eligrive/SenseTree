"""Kaggle job (2 x T4, internet on): runs the SenseTree agent harness on the file-system
benchmark with local models served by Ollama, one Ollama instance per GPU.

Inputs  /kaggle/input/sensetree-src : src-tauri/, dist/, package.json, research/
Output  /kaggle/working/runs/agent/<tag>.jsonl

JOBS lists (model, tag, split, effort); the personas of each job are split between the
two GPUs.
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
START = time.time()
BUDGET_S = float(os.environ.get("SENSETREE_BUDGET_H", "10.5")) * 3600
JOBS_FILE = Path(__file__).with_name("jobs.json")
JOBS = (json.loads(JOBS_FILE.read_text()) if JOBS_FILE.exists() else None) or [
    ["qwen3.5:4b", "test-qwen3.5-4b", "test", "none"],
    ["qwen3.5:9b", "test-qwen3.5-9b", "test", "none"],
    ["qwen3.5:9b", "train-qwen3.5-9b", "train", "none"],
]


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
        print("sources:", src, sorted(x.name for x in src.iterdir()), flush=True)
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
    sh("pip install -q fpdf2 python-docx")
    sh(f"{sys.executable} -m pip install -q -e {RESEARCH} --no-deps")
    sh(f"cd {RESEARCH} && {sys.executable} scripts/prepare_data.py alloprof && "
       f"{sys.executable} scripts/build_fsbench.py --root {FS}")
    if not shutil.which("ollama"):
        sh("curl -fsSL https://ollama.com/install.sh | sh")


def start_ollama(gpu: int, port: int) -> subprocess.Popen:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OLLAMA_HOST=f"127.0.0.1:{port}", OLLAMA_NUM_PARALLEL="1",
               OLLAMA_CONTEXT_LENGTH="16384", OLLAMA_FLASH_ATTENTION="1", OLLAMA_KV_CACHE_TYPE="q8_0",
               OLLAMA_KEEP_ALIVE="60m", OLLAMA_MODELS=str(WORK / "ollama-models"))
    p = subprocess.Popen(["ollama", "serve"], env=env, stdout=open(OUT / f"logs/ollama{gpu}.log", "w"), stderr=subprocess.STDOUT)
    time.sleep(8)
    return p


def run_personas(model: str, tag: str, split: str, effort: str, personas: list[Path], port: int, worker: int) -> None:
    exe = APP / "src-tauri/target/release/examples/agent_eval"
    out = OUT / "runs/agent" / f"{tag}.part{worker}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    for home in personas:
        if time.time() - START > BUDGET_S:
            print(f"time budget reached, {home.name} skipped", flush=True)
            continue
        subprocess.run([str(exe), "--root", str(home), "--state", str(WORK / "agent-state" / home.name),
                        "--tasks", str(RESEARCH / "data/fsbench/tasks.jsonl"), "--split", split, "--model", model,
                        "--effort", effort, "--base-url", f"http://127.0.0.1:{port}/v1", "--out", str(out),
                        "--gpu", "false"], stdout=open(OUT / f"logs/agent_{tag}_{worker}.log", "a"), stderr=subprocess.STDOUT)


def main() -> None:
    setup()
    servers = [start_ollama(0, 11500), start_ollama(1, 11501)]
    for model, tag, split, effort in JOBS:
        for port in (11500, 11501):
            env = dict(os.environ, OLLAMA_HOST=f"127.0.0.1:{port}")
            sh(f"ollama pull {model} > /dev/null", env=env)
        personas = sorted(p for p in FS.iterdir() if p.is_dir() and p.name.startswith(split))
        halves = [personas[0::2], personas[1::2]]
        threads = [threading.Thread(target=run_personas, args=(model, tag, split, effort, halves[i], 11500 + i, i))
                   for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        with open(OUT / "runs/agent" / f"{tag}.jsonl", "w", encoding="utf-8") as merged:
            for i in range(2):
                part = OUT / "runs/agent" / f"{tag}.part{i}.jsonl"
                if part.exists():
                    merged.write(part.read_text(encoding="utf-8"))
                    part.unlink()
        print(f"finished {tag}", flush=True)
    for s in servers:
        s.terminate()
    shutil.copy(RESEARCH / "data/fsbench/tasks.jsonl", OUT / "runs/agent/tasks-kaggle.jsonl")


if __name__ == "__main__":
    main()

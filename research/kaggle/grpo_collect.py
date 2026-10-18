"""Kaggle job (2 x T4, internet on): the Kaggle share of a GRPO round for the file agent,
or of an evaluation, run alongside the GPU PC.

Inputs  /kaggle/input/sensetree-src : src-tauri/, dist/, package.json, research/
        the job (JOB below, else grpo-job.json), and for rounds after the first adapter_config.json and
        adapter_model.safetensors, at the root of the dataset:
          {"round": 1, "adapter": true | false, "personas": [...],
           "attempts": 4, "workers": 4,
           "eval": [{"tag": "test-grpo-r3", "personas": [...]}]}       (optional)
Output  /kaggle/working/runs/grpo/r<n>/kaggle/a<k>/<persona>.jsonl  (attempts)
        /kaggle/working/runs/agent/<tag>.jsonl                       (evaluations)

Round 0 serves qwen3.5:4b from the Ollama library. Later rounds rebuild the policy as on
the PC: the round's LoRA adapter merged into Qwen/Qwen3.5-4B (scripts/merge_lora.py),
converted to GGUF and quantised to Q4_K_M by llama.cpp (Ollama 0.40 neither loads LoRA
adapters nor quantises an imported GGUF). One Ollama server per T4, several attempts at a time on each.
"""

import importlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path("/kaggle/working")
# the job of this kernel version; None reads grpo-job.json from the dataset
JOB = None


def load_agent_runs():
    """agent_runs.py (build and benchmark setup) from the dataset, extracted if needed."""
    src = next(p for p in Path("/kaggle/input").rglob("package.json") if "node_modules" not in str(p)).parent
    if (src / "research" / "kaggle" / "agent_runs.py").exists():
        sys.path.insert(0, str(src / "research" / "kaggle"))
    else:
        boot = Path("/tmp/boot")
        boot.mkdir(exist_ok=True)
        with tarfile.open(src / "research.tar") as t:
            t.extractall(boot)
        sys.path.insert(0, str(next(boot.rglob("agent_runs.py")).parent))
    return importlib.import_module("agent_runs")


ar = load_agent_runs()
WORK, APP, RESEARCH, FS = ar.WORK, ar.APP, ar.RESEARCH, ar.FS
EXE = APP / "src-tauri/target/release/examples/agent_eval"
STATE = WORK / "agent-state"
MODELS = STATE / "_models"


def sh(cmd: str, **kw) -> None:
    ar.sh(cmd, **kw)


def build_policy_gguf(adapter: Path, name: str) -> Path:
    """Merged Qwen3.5-4B + adapter as a Q4_K_M GGUF (CPU only: runs during the cargo build).
    Ollama 0.40 neither loads LoRA adapters nor quantises an imported GGUF, so the merged
    weights go through llama.cpp's converter and llama-quantize, as on the PC."""
    from huggingface_hub import snapshot_download

    gguf = WORK / f"{name}-q4km.gguf"
    if gguf.exists():
        return gguf
    base = Path(snapshot_download("Qwen/Qwen3.5-4B"))
    llama = WORK / "llama.cpp"
    if not llama.exists():
        sh(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {llama}")
    sh(f"{sys.executable} -m pip install -q {llama}/gguf-py")
    sh(f"cmake -S {llama} -B {llama}/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DLLAMA_CURL=OFF "
       f"-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF > /dev/null "
       f"&& cmake --build {llama}/build --target llama-quantize -j 2 > /dev/null")
    merged = WORK / f"merged-{name}"
    f16 = WORK / f"{name}-f16.gguf"
    sh(f"{sys.executable} {RESEARCH}/scripts/merge_lora.py --base {base} --adapter {adapter} --out {merged}")
    sh(f"{sys.executable} {llama}/convert_hf_to_gguf.py {merged} --outtype f16 --outfile {f16}")
    shutil.rmtree(merged)
    sh(f"{llama}/build/bin/llama-quantize {f16} {gguf} Q4_K_M > /dev/null")
    f16.unlink()
    return gguf


def start_ollama(gpu: int, port: int, parallel: int) -> subprocess.Popen:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OLLAMA_HOST=f"127.0.0.1:{port}",
               OLLAMA_NUM_PARALLEL=str(parallel), OLLAMA_CONTEXT_LENGTH="16384", OLLAMA_FLASH_ATTENTION="1",
               OLLAMA_KV_CACHE_TYPE="q8_0", OLLAMA_KEEP_ALIVE="120m", OLLAMA_MODELS=str(WORK / "ollama-models"))
    p = subprocess.Popen(["ollama", "serve"], env=env, stdout=open(OUT / f"logs/ollama{gpu}.log", "w"),
                         stderr=subprocess.STDOUT)
    time.sleep(8)
    return p


def create_policy(gguf: Path, name: str) -> None:
    modelfile = WORK / f"Modelfile.{name}"
    modelfile.write_text(f"FROM {gguf}\nTEMPLATE {{{{ .Prompt }}}}\nRENDERER qwen3.5\nPARSER qwen3.5\n"
                         "PARAMETER top_k 20\nPARAMETER top_p 0.95\nPARAMETER presence_penalty 1.5\n"
                         "PARAMETER temperature 1\n")
    sh(f"ollama create {name} -f {modelfile}", env=dict(os.environ, OLLAMA_HOST="127.0.0.1:11500"))


def index_persona(persona: str) -> None:
    """Indexes a persona's home once; every attempt then copies this state."""
    state = STATE / persona
    if (state / "indexed.json").exists():
        return
    state.mkdir(parents=True, exist_ok=True)
    if MODELS.exists() and not (state / "models").exists():
        os.symlink(MODELS, state / "models", target_is_directory=True)
    subprocess.run([str(EXE), "--root", str(FS / persona), "--state", str(state), "--tasks",
                    str(RESEARCH / "data/fsbench/tasks.jsonl"), "--split", "none", "--model", "none",
                    "--out", str(WORK / "index-none.jsonl"), "--gpu", "false"],
                   stdout=open(OUT / "logs/index.log", "a"), stderr=subprocess.STDOUT, check=True)


def index_all(personas: list[str]) -> None:
    index_persona(personas[0])  # downloads the embedding models
    if not MODELS.exists():
        shutil.move(str(STATE / personas[0] / "models"), str(MODELS))
        os.symlink(MODELS, STATE / personas[0] / "models", target_is_directory=True)
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(index_persona, personas[1:]))


def run_eval(model: str, tag: str, personas: list[str], ports: list[int], workers: int) -> None:
    out_dir = OUT / "runs/agent"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = list(enumerate(personas))

    def one(job):
        i, persona = job
        port = ports[i % len(ports)]
        state = WORK / "eval-state" / tag / persona
        if not state.exists():
            state.mkdir(parents=True)
            for name in ("lancedb", "ort"):
                shutil.copytree(STATE / persona / name, state / name)
            for name in ("config.json", "indexed.json"):
                shutil.copy(STATE / persona / name, state / name)
            os.symlink(MODELS, state / "models", target_is_directory=True)
        subprocess.run([str(EXE), "--root", str(FS / persona), "--state", str(state), "--tasks",
                        str(RESEARCH / "data/fsbench/tasks.jsonl"), "--split", "test", "--model", model,
                        "--effort", "medium", "--base-url", f"http://127.0.0.1:{port}/v1",
                        "--out", str(out_dir / f"{tag}.{persona}.jsonl"), "--gpu", "false"],
                       stdout=open(OUT / f"logs/eval_{tag}_{persona}.log", "a"), stderr=subprocess.STDOUT)

    with ThreadPoolExecutor(len(ports) * workers) as pool:
        list(pool.map(one, jobs))
    with open(out_dir / f"{tag}.jsonl", "w", encoding="utf-8") as merged:
        for _, persona in jobs:
            part = out_dir / f"{tag}.{persona}.jsonl"
            if part.exists():
                merged.write(part.read_text(encoding="utf-8"))
                part.unlink()


def main() -> None:
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    job = JOB or json.loads(next(Path("/kaggle/input").rglob("grpo-job.json")).read_text())
    print("job:", job, flush=True)
    rnd, workers = job["round"], job.get("workers", 4)
    name = "qwen3.5:4b" if not job.get("adapter") else f"st-r{rnd}"

    policy: dict = {}
    if job.get("adapter"):
        # the round's adapter files sit at the root of the dataset (subfolders are uploaded as tar files)
        def prep():
            adapter = next(Path("/kaggle/input").rglob("adapter_config.json")).parent
            policy["gguf"] = build_policy_gguf(adapter, name)
        t = threading.Thread(target=prep)
        t.start()
    ar.setup()
    if job.get("adapter"):
        t.join()
        if "gguf" not in policy:
            raise SystemExit("policy conversion failed")
    print(f"setup done after {time.time() - ar.START:.0f} s", flush=True)

    ports = [11500, 11501]
    servers = [start_ollama(0, 11500, workers), start_ollama(1, 11501, workers)]
    if job.get("adapter"):
        create_policy(policy["gguf"], name)
    else:
        sh(f"ollama pull {name} > /dev/null", env=dict(os.environ, OLLAMA_HOST="127.0.0.1:11500"))

    personas = list(job.get("personas", []))
    evals = job.get("eval", [])
    index_all(personas + [p for e in evals for p in e["personas"]])
    print(f"indexing done after {time.time() - ar.START:.0f} s", flush=True)

    if personas:
        runs = OUT / "runs/grpo"
        (runs / f"r{rnd}").mkdir(parents=True, exist_ok=True)
        shutil.copy(RESEARCH / f"runs/grpo/r{rnd}/round.json", runs / f"r{rnd}/round.json")
        sh(f"{sys.executable} {RESEARCH}/scripts/grpo_round.py collect --round {rnd} --site kaggle --runs {runs} "
           f"--tasks {RESEARCH}/data/fsbench/tasks.jsonl --exe {EXE} --fsbench {FS} --base-state {STATE} "
           f"--models {MODELS} --work {WORK}/grpo-state --model {name} --ports {' '.join(map(str, ports))} "
           f"--workers {workers} --attempts {job.get('attempts', 4)} --personas {' '.join(personas)}")
    for e in evals:
        run_eval(name, e["tag"], e["personas"], ports, workers)
    for s in servers:
        s.terminate()
    shutil.copy(RESEARCH / "data/fsbench/tasks.jsonl", OUT / "tasks-kaggle.jsonl")
    print(f"finished after {time.time() - ar.START:.0f} s", flush=True)


if __name__ == "__main__":
    main()

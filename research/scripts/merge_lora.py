"""Merges a PEFT LoRA adapter into the original Qwen3.5 checkpoint so that the result
keeps the checkpoint's own layout (vision tower, MTP head, tensor names) and converts to
GGUF with llama.cpp's convert_hf_to_gguf.py like the released model.

Ollama no longer loads LoRA adapters (``ADAPTER`` fails with "LoRA adapters are no
longer supported"), so every policy of the GRPO rounds is served as merged weights.

W' = W + (alpha / r) * B @ A, computed in fp32 and stored in the checkpoint's dtype.
Tensors are written in shards of about 1 GB, so the merge fits next to a running
evaluation on a 16 GB machine.
"""

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

SHARD_BYTES = 1 << 30


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="folder of the original checkpoint (safetensors shards)")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base, adapter, out = Path(args.base), Path(args.adapter), Path(args.out)
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    cfg = json.load(open(adapter / "adapter_config.json"))
    scale = cfg["lora_alpha"] / cfg["r"]
    lora = {}
    with safe_open(adapter / "adapter_model.safetensors", "pt") as f:
        for k in f.keys():
            module, part = k.removeprefix("base_model.model.").rsplit(".lora_", 1)
            # PEFT wraps the text model (model.layers.*); the checkpoint stores it under model.language_model
            name = module.replace("model.", "model.language_model.", 1) + ".weight"
            lora.setdefault(name, {})[part.split(".")[0]] = f.get_tensor(k)

    weight_map, shard, size, n, merged = {}, {}, 0, 0, 0

    def flush() -> None:
        nonlocal shard, size, n
        if shard:
            name = f"model-{n:05d}.safetensors"
            save_file(shard, tmp / name, metadata={"format": "pt"})
            weight_map.update({k: name for k in shard})
            shard, size, n = {}, 0, n + 1

    for src in sorted(base.glob("*.safetensors")):
        with safe_open(src, "pt") as f:
            for k in f.keys():
                w = f.get_tensor(k)
                if k in lora:
                    ab = lora.pop(k)
                    w = (w.float() + scale * (ab["B"].float() @ ab["A"].float())).to(w.dtype)
                    merged += 1
                shard[k] = w
                size += w.numel() * w.element_size()
                if size >= SHARD_BYTES:
                    flush()
    flush()
    if lora:
        raise SystemExit(f"{len(lora)} adapter modules not found in the checkpoint, e.g. {next(iter(lora))}")
    json.dump({"metadata": {}, "weight_map": weight_map}, open(tmp / "model.safetensors.index.json", "w"), indent=1)
    for p in base.iterdir():
        if p.is_file() and p.suffix != ".safetensors" and p.name != "model.safetensors.index.json":
            shutil.copy(p, tmp / p.name)
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    print(f"merged {merged} weights (scale {scale}) into {n} shards -> {out}")


if __name__ == "__main__":
    main()

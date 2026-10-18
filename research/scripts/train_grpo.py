"""One policy update of iterative GRPO for the file agent (QLoRA).

Input: the scored attempts of a round (``results/training/grpo/r<n>-rewards.json``, written
by ``grpo_round.py rewards``): each attempt of a task whose group of attempts did not all
get the same reward, with its group-normalised advantage A.

Sequences. One sequence per attempt: the recorded conversation rendered with the model's
chat template in thinking mode. The app sends the reasoning of earlier turns back to the
model (Ollama renders it, checked on the served model: 334 -> 555 prompt tokens), and the
Qwen3.5 template keeps <think> for every turn after the user's message, so this one
sequence reproduces the exact context of each turn. Targets are the assistant tokens:
reasoning, text and tool calls. Sequences are cut at --max-length tokens (8 GB card).

Loss, per target token of an attempt with advantage A (DAPO-style token average over the
batch):
  ratio = exp(logp - logp_old)
  pg    = -min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A)
  kl    = exp(logp_ref - logp) - (logp_ref - logp) - 1      (k3 estimator)
  loss  = sum(pg + beta * kl) / target tokens in the batch
logp_old is recomputed under the training model before the update (it equals the
reference in the first round); the reference is the base model, i.e. the same network
with the adapter disabled, so it costs no memory.

Training recipe of train_agent.py: NF4 + double quantisation, fp16 compute, LoRA r16/a32
on all linear layers (no dropout here: it would make logp and logp_old differ by noise),
gradient checkpointing, output layer applied by chunks of target positions.
Resumable: the reference/old log-probabilities are cached and the adapter and optimiser are
saved after every step.
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.utils.checkpoint
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent
CHUNK = 512


def clean(messages: list[dict]) -> list[dict]:
    """OpenAI-style messages as recorded by agent_eval -> the chat template's format
    (reasoning -> reasoning_content, tool arguments as objects)."""
    out = []
    for m in messages:
        n = {k: v for k, v in m.items() if k in ("role", "content", "tool_calls", "tool_call_id", "name")}
        if m.get("reasoning"):
            n["reasoning_content"] = m["reasoning"]
        if n.get("tool_calls"):
            calls = []
            for tc in n["tool_calls"]:
                fn = dict(tc["function"])
                if isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.loads(fn["arguments"] or "{}")
                calls.append({"type": "function", "function": fn})
            n["tool_calls"] = calls
        n["content"] = n.get("content") or ""
        out.append(n)
    return out


def render(tokenizer, record: dict, max_length: int) -> dict | None:
    """Token ids of the whole attempt and the positions of its assistant tokens."""
    msgs, tools = clean(record["conversation"]), record["tools"]
    ids, target, done = [], [], ""
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        prefix = tokenizer.apply_chat_template(msgs[:i], tools=tools, add_generation_prompt=True, tokenize=False,
                                               enable_thinking=True)
        full = tokenizer.apply_chat_template(msgs[:i + 1], tools=tools, tokenize=False, enable_thinking=True)
        if not (full.startswith(prefix) and prefix.startswith(done)):
            return None
        ctx = tokenizer(prefix[len(done):], add_special_tokens=False)["input_ids"]
        tgt = tokenizer(full[len(prefix):], add_special_tokens=False)["input_ids"]
        ids += ctx + tgt
        target += [False] * len(ctx) + [True] * len(tgt)
        done = full
    truncated = len(ids) > max_length
    ids, target = ids[:max_length], target[:max_length]
    if not any(target):
        return None
    return {"input_ids": ids, "target": target, "truncated": truncated}


def token_logprobs(model, ids: torch.Tensor, target: torch.Tensor, grad: bool) -> torch.Tensor:
    """log p(token) at the target positions; the output layer runs by chunks of positions
    (248k-token vocabulary), recomputed in the backward pass when training."""
    base = model.get_base_model()
    with torch.autocast("cuda", dtype=torch.float16):
        hidden = base.model(input_ids=ids).last_hidden_state
    h = hidden[0, :-1][target[1:]]
    y = ids[0, 1:][target[1:]]

    def chunk_lp(h_chunk, y_chunk):
        return -torch.nn.functional.cross_entropy(base.lm_head(h_chunk).float(), y_chunk, reduction="none")

    parts = []
    for i in range(0, len(y), CHUNK):
        if grad:
            parts.append(torch.utils.checkpoint.checkpoint(chunk_lp, h[i:i + CHUNK], y[i:i + CHUNK], use_reentrant=False))
        else:
            parts.append(chunk_lp(h[i:i + CHUNK], y[i:i + CHUNK]))
    return torch.cat(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True, help="round whose attempts are used")
    parser.add_argument("--init-adapter", help="adapter of the policy that collected the round (none in round 0)")
    parser.add_argument("--out", required=True, help="folder of the updated adapter, relative to research/")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=0.04, help="weight of the KL penalty to the base model")
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--batch", type=int, default=8, help="attempts per optimiser step")
    parser.add_argument("--max-length", type=int, default=7168)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=-1, help="stop after this many steps (smoke test)")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(13)

    data = json.load(open(ROOT / "results" / "training" / "grpo" / f"r{args.round}-rewards.json", encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    lines: dict[str, list[str]] = {}
    examples, skipped = [], {"forced_answer": 0, "template": 0}
    for a in data["train"]:
        if a["forced_answer"]:
            skipped["forced_answer"] += 1
            continue
        if a["file"] not in lines:
            lines[a["file"]] = open(ROOT / a["file"], encoding="utf-8").read().splitlines()
        ex = render(tokenizer, json.loads(lines[a["file"]][a["line"]]), args.max_length)
        if ex is None:
            skipped["template"] += 1
            continue
        ex.update(advantage=a["advantage"], id=a["id"], attempt=a["attempt"])
        examples.append(ex)
    n_tokens = [sum(e["target"]) for e in examples]
    print(f"{len(examples)} attempts, skipped {skipped}, {sum(e['truncated'] for e in examples)} truncated, "
          f"{sum(len(e['input_ids']) for e in examples) / max(1, len(examples)):.0f} tokens and "
          f"{sum(n_tokens) / max(1, len(examples)):.0f} targets on average", flush=True)

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant, dtype=torch.float16,
                                                 device_map={"": 0})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    ckpt = out / "checkpoint"
    if (ckpt / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(ckpt), is_trainable=True)
    elif args.init_adapter:
        model = PeftModel.from_pretrained(model, str(ROOT / args.init_adapter), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                                 target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    # reference and old log-probabilities, before any update (cached: the longest pass to redo)
    cache = out / "logprobs.pt"
    if cache.exists():
        ref_lp, old_lp = torch.load(cache)
    else:
        ref_lp, old_lp, t0 = [], [], time.time()
        model.eval()
        with torch.no_grad():
            for n, e in enumerate(examples):
                ids = torch.tensor([e["input_ids"]], device="cuda")
                tgt = torch.tensor(e["target"], device="cuda")
                with model.disable_adapter():
                    ref_lp.append(token_logprobs(model, ids, tgt, grad=False).cpu())
                old_lp.append(token_logprobs(model, ids, tgt, grad=False).cpu() if args.init_adapter else ref_lp[-1])
                if n % 20 == 0:
                    print(f"logprobs {n + 1}/{len(examples)} ({time.time() - t0:.0f} s)", flush=True)
        torch.save((ref_lp, old_lp), cache)
        if args.init_adapter:
            diff = torch.cat([(o - r).abs() for o, r in zip(old_lp, ref_lp)]).mean().item()
            print(f"mean |logp_old - logp_ref| = {diff:.4f}", flush=True)

    order = list(range(len(examples)))
    random.Random(13 + args.round).shuffle(order)
    batches = [order[i:i + args.batch] for i in range(0, len(order), args.batch)]
    if args.max_steps > 0:
        batches = batches[:args.max_steps]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    # a low initial scale: at the default 2**16 the first steps overflow and are skipped, and a
    # round has only a dozen steps
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0**10)
    log, start = [], 0
    if (ckpt / "state.pt").exists():
        state = torch.load(ckpt / "state.pt", weights_only=False)
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        log, start = state["log"], state["step"]
        print(f"resumed at step {start}", flush=True)

    model.train()
    for step in range(start, len(batches)):
        t0 = time.time()
        idx = batches[step]
        total = sum(n_tokens[i] for i in idx)
        stats = {"pg": 0.0, "kl": 0.0, "clipped": 0.0, "ratio": 0.0}
        for i in idx:
            e = examples[i]
            ids = torch.tensor([e["input_ids"]], device="cuda")
            tgt = torch.tensor(e["target"], device="cuda")
            lp = token_logprobs(model, ids, tgt, grad=True)
            old, ref = old_lp[i].to(lp.device), ref_lp[i].to(lp.device)
            ratio = torch.exp(lp - old)
            adv = e["advantage"]
            pg = -torch.minimum(ratio * adv, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv)
            kl = torch.exp(ref - lp) - (ref - lp) - 1
            loss = (pg + args.beta * kl).sum() / total
            scaler.scale(loss).backward()
            with torch.no_grad():
                stats["pg"] += pg.sum().item() / total
                stats["kl"] += kl.sum().item() / total
                stats["clipped"] += ((ratio - 1).abs() > args.clip).float().sum().item() / total
                stats["ratio"] += ratio.sum().item() / total
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        entry = {"step": step + 1, "attempts": len(idx), "tokens": total, "grad_norm": grad_norm,
                 "seconds": time.time() - t0, "peak_gib": torch.cuda.max_memory_allocated() / 2**30, **stats}
        torch.cuda.reset_peak_memory_stats()
        log.append(entry)
        print(json.dumps(entry), flush=True)
        model.save_pretrained(str(ckpt))
        torch.save({"opt": opt.state_dict(), "scaler": scaler.state_dict(), "log": log, "step": step + 1},
                   ckpt / "state.pt")

    model.save_pretrained(str(out / "adapter"))
    tokenizer.save_pretrained(str(out / "adapter"))
    report = {"args": vars(args), "round": args.round, "attempts": len(examples), "skipped": skipped,
              "truncated": sum(e["truncated"] for e in examples), "mean_tokens": sum(len(e["input_ids"]) for e in examples)
              / max(1, len(examples)), "rewards": data["summary"], "log": log}
    path = ROOT / "results" / "training" / "grpo" / f"r{args.round}-update.json"
    json.dump(report, open(path, "w", encoding="utf-8"), indent=1)
    print(f"adapter -> {out / 'adapter'}, log -> {path}", flush=True)


if __name__ == "__main__":
    main()

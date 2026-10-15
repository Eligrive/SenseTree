"""Fine-tunes a small model on successful agent trajectories (QLoRA).

Data: agent runs on the *train* personas (``runs/agent/<run>.jsonl``). A trajectory is
kept only if it solves its task, proposes no unrequested deletion and contains no
malformed tool call (rejection sampling), and at most ``--per-family`` trajectories are
kept per task family so that the families the student fails on weigh as much as the
others. Each trajectory is one sequence rendered with the model's chat template; the
loss covers the assistant turns (text and tool calls) only.

Runs on an 8 GB card with ``--max-length 8192`` or on Kaggle's 16 GB T4s. A checkpoint is written
every 25 optimiser steps and an interrupted run resumes from it.
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.utils.checkpoint
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments)

ROOT = Path(__file__).resolve().parent.parent


def keep(score: dict) -> bool:
    return score["success"] == 1.0 and score["unrequested_deletions"] == 0 and score["malformed_calls"] == 0


def clean(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        m = {k: v for k, v in m.items() if k in ("role", "content", "tool_calls", "tool_call_id", "name")}
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = dict(tc["function"])
                if isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.loads(fn["arguments"] or "{}")
                calls.append({"type": "function", "function": fn})
            m["tool_calls"] = calls
        m["content"] = m.get("content") or ""
        out.append(m)
    return out


def examples(tokenizer, records: list[dict], max_length: int) -> list[dict]:
    """One sequence per trajectory; every assistant turn in it is a target."""
    out = []
    for r in records:
        msgs, tools = clean(r["conversation"]), r["tools"]
        ids, labels, done, ok = [], [], "", True
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            prefix = tokenizer.apply_chat_template(msgs[:i], tools=tools, add_generation_prompt=True,
                                                   tokenize=False, enable_thinking=False)
            full = tokenizer.apply_chat_template(msgs[:i + 1], tools=tools, tokenize=False, enable_thinking=False)
            if not (full.startswith(prefix) and prefix.startswith(done)):
                ok = False
                break
            ctx = tokenizer(prefix[len(done):], add_special_tokens=False)["input_ids"]
            tgt = tokenizer(full[len(prefix):], add_special_tokens=False)["input_ids"]
            ids += ctx + tgt
            labels += [-100] * len(ctx) + tgt
            done = full
        if ok and ids and len(ids) <= max_length:
            out.append({"input_ids": ids, "labels": labels})
    return out


def cap_per_family(records: list[dict], n: int) -> list[dict]:
    by_family: dict[str, list[dict]] = {}
    for r in records:
        by_family.setdefault(r["id"].rsplit("-", 1)[-1], []).append(r)
    rng = random.Random(13)
    out = []
    for family in sorted(by_family):
        group = by_family[family]
        rng.shuffle(group)
        out += group[:n]
    return out


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id)
    labels = torch.full((len(batch), n), -100)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, :len(b["input_ids"])] = torch.tensor(b["input_ids"])
        labels[i, :len(b["labels"])] = torch.tensor(b["labels"])
        mask[i, :len(b["input_ids"])] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


class TargetLossTrainer(Trainer):
    """Applies the output layer only at target positions, by chunks whose logits are
    recomputed in the backward pass. With a 248k-token vocabulary, the logits of 2,000
    target tokens take 2 GB in fp32, and as much again for their gradient."""

    chunk = 1024

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        base = model.get_base_model()
        # without padding, no mask: attention then takes SDPA's causal kernel instead of
        # materialising a T x T mask (2 GiB per layer for 8k tokens and 16 heads in fp16)
        mask_in = None if bool(inputs["attention_mask"].all()) else inputs["attention_mask"]
        hidden = base.model(input_ids=inputs["input_ids"], attention_mask=mask_in).last_hidden_state
        labels = inputs["labels"][:, 1:]
        mask = labels != -100
        h, y = hidden[:, :-1][mask], labels[mask]

        def chunk_loss(h_chunk, y_chunk):
            return torch.nn.functional.cross_entropy(base.lm_head(h_chunk).float(), y_chunk, reduction="sum")

        total = sum(torch.utils.checkpoint.checkpoint(chunk_loss, h[i:i + self.chunk], y[i:i + self.chunk],
                                                      use_reentrant=False)
                    for i in range(0, len(y), self.chunk))
        loss = total / len(y)
        return (loss, None) if return_outputs else loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", nargs="+", required=True, help="scored runs: results/agent/<run>.json + runs/agent/<run>.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--per-family", type=int, default=20, help="trajectories kept per task family")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=-1, help="stop after this many optimiser steps (smoke test)")
    args = parser.parse_args()

    records = []
    for run in args.trajectories:
        scores = json.load(open(ROOT / "results" / "agent" / f"{run}.json"))["per_task"]
        for line in open(ROOT / "runs" / "agent" / f"{run}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r["id"] in scores and keep(scores[r["id"]]) and r.get("conversation"):
                records.append(r)
    print(f"{len(records)} successful trajectories", flush=True)
    records = cap_per_family(records, args.per_family)
    print(f"{len(records)} kept after the cap of {args.per_family} per family", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data = examples(tokenizer, records, args.max_length)
    random.Random(13).shuffle(data)
    print(f"{len(data)} sequences, {sum(len(d['input_ids']) for d in data) / max(1, len(data)):.0f} tokens on average",
          flush=True)

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant, dtype=torch.float16,
                                                 device_map={"": 0})
    # not prepare_model_for_kbit_training: it casts the tied 248k x 2560 embedding to fp32 (2.5 GB),
    # which overflows an 8 GB card into system memory
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    print(f"GPU memory after loading: {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    out = ROOT / args.out
    steps = args.max_steps if args.max_steps > 0 else math.ceil(len(data) / args.grad_accum * args.epochs)
    warmup = max(1, round(0.05 * steps))
    trainer = TargetLossTrainer(
        model=model,
        args=TrainingArguments(output_dir=str(out / "checkpoints"), num_train_epochs=args.epochs, max_steps=args.max_steps,
                               per_device_train_batch_size=1, gradient_accumulation_steps=args.grad_accum,
                               learning_rate=args.lr, warmup_steps=warmup, lr_scheduler_type="cosine",
                               fp16=True, logging_steps=10, save_strategy="steps", save_steps=25, save_total_limit=1,
                               report_to="none", seed=13,
                               remove_unused_columns=False, gradient_checkpointing=True,
                               gradient_checkpointing_kwargs={"use_reentrant": False}),
        train_dataset=Dataset.from_list(data),
        data_collator=lambda b: collate(b, tokenizer.pad_token_id),
    )
    # checkpoints every 25 optimiser steps: an interrupted run resumes from the last one
    resume = any((out / "checkpoints").glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=resume or None)
    model.save_pretrained(str(out / "adapter"))
    tokenizer.save_pretrained(str(out / "adapter"))
    json.dump({"args": vars(args), "trajectories": len(records), "examples": len(data),
               "log_history": trainer.state.log_history}, open(out / "training.json", "w"), indent=1)


if __name__ == "__main__":
    main()

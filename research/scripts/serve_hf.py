"""Minimal OpenAI-compatible chat server for a Hugging Face model, with an optional LoRA.

Serves ``POST /v1/chat/completions`` with tools, which is all the SenseTree agent uses.
The base model and the fine-tuned one go through the same tokenizer, chat template,
quantisation and decoding, so their comparison is not confounded by the inference
stack. With ``--adapter``, a request whose ``model`` field ends in ``-base`` is answered
with the adapter disabled, so one process serves both models and they can be evaluated
alternately. Tool calls are parsed from Qwen 3.5's XML form and from the older
``<tool_call>{json}</tool_call>`` form.
"""

import argparse
import contextlib
import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
XML_FUNCTION = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.S)
XML_PARAMETER = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)
THINK = re.compile(r"<think>.*?</think>", re.S)


def load(model_name: str, adapter: str | None, four_bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16) if four_bit else None
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant, dtype=torch.float16,
                                                 device_map="cuda")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tokenizer, model


def normalise(messages: list[dict]) -> list[dict]:
    """Converts OpenAI messages to what the chat template expects (arguments as dicts)."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = dict(tc.get("function", {}))
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"] or "{}")
                    except json.JSONDecodeError:
                        fn["arguments"] = {}
                calls.append({"type": "function", "function": fn})
            m["tool_calls"] = calls
        if m.get("content") is None:
            m["content"] = ""
        out.append(m)
    return out


def parameter_value(raw: str):
    """Qwen 3.5 writes scalars as text and structured values as JSON."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_call(raw: str) -> dict | None:
    """One tool call, in Qwen 3.5's XML form or the older JSON form."""
    m = XML_FUNCTION.search(raw)
    if m:
        args = {k: parameter_value(v) for k, v in XML_PARAMETER.findall(m.group(2))}
        return {"name": m.group(1), "arguments": args}
    try:
        obj = json.loads(raw.strip())
        return {"name": obj.get("name", ""), "arguments": obj.get("arguments", {})}
    except json.JSONDecodeError:
        return None


def parse(text: str) -> dict:
    text = THINK.sub("", text).strip()
    calls = []
    for raw in TOOL_CALL.findall(text):
        call = parse_call(raw)
        if call is None:
            continue
        calls.append({"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                      "function": {"name": call["name"],
                                   "arguments": json.dumps(call["arguments"], ensure_ascii=False)}})
    content = TOOL_CALL.sub("", text).strip()
    msg = {"role": "assistant", "content": content or None}
    if calls:
        msg["tool_calls"] = calls
    return msg


class Handler(BaseHTTPRequestHandler):
    tokenizer = model = None
    lock = threading.Lock()
    max_new_tokens = 1024

    def log_message(self, *args):
        pass

    def do_POST(self):
        if not self.path.endswith("/chat/completions"):
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        messages = normalise(body["messages"])
        tools = body.get("tools") or None
        prompt = self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                                    tokenize=False, enable_thinking=False)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        temperature = float(body.get("temperature") or 0.0)
        base = str(body.get("model", "")).endswith("-base") and hasattr(self.model, "disable_adapter")
        adapter_off = self.model.disable_adapter() if base else contextlib.nullcontext()
        with self.lock, torch.inference_mode(), adapter_off:
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=temperature > 0, temperature=temperature or None,
                                      top_p=0.95 if temperature > 0 else None)
        text = self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        msg = parse(text)
        reply = {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()),
                 "model": body.get("model", ""),
                 "choices": [{"index": 0, "message": msg,
                              "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop"}],
                 "usage": {"prompt_tokens": int(inputs["input_ids"].shape[1]),
                           "completion_tokens": int(out.shape[1] - inputs["input_ids"].shape[1])}}
        data = json.dumps(reply, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--adapter")
    parser.add_argument("--port", type=int, default=11600)
    parser.add_argument("--full-precision", action="store_true")
    args = parser.parse_args()
    Handler.tokenizer, Handler.model = load(args.model, args.adapter, not args.full_precision)
    print(f"serving {args.model} {args.adapter or ''} on :{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

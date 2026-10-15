import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
spec = importlib.util.spec_from_file_location("serve_hf", Path(__file__).parents[1] / "scripts" / "serve_hf.py")


def load():
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except ImportError:
        pytest.skip("transformers not installed")


def test_parses_qwen35_xml_tool_calls():
    mod = load()
    text = ("Je cherche.\n<tool_call>\n<function=search_files>\n<parameter=query>\navis d'impôt 2024\n</parameter>\n"
            "<parameter=top_k>\n8\n</parameter>\n</function>\n</tool_call>")
    msg = mod.parse(text)
    args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
    assert msg["tool_calls"][0]["function"]["name"] == "search_files"
    assert args == {"query": "avis d'impôt 2024", "top_k": 8}
    assert msg["content"] == "Je cherche."


def test_parses_structured_parameters_and_json_form():
    mod = load()
    ops = [{"kind": "delete", "old_path": "C:\\a.exe"}]
    xml = f"<tool_call>\n<function=propose_actions>\n<parameter=operations>\n{json.dumps(ops)}\n</parameter>\n</function>\n</tool_call>"
    assert json.loads(mod.parse(xml)["tool_calls"][0]["function"]["arguments"])["operations"][0]["kind"] == "delete"
    legacy = '<tool_call>{"name": "list_directory", "arguments": {"path": "D:/x"}}</tool_call>'
    assert mod.parse(legacy)["tool_calls"][0]["function"]["name"] == "list_directory"

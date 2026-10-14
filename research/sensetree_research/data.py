"""Benchmark datasets, normalised to the BEIR layout.

Every dataset ends up under ``data/<name>/`` as::

    corpus.jsonl    {"_id", "title", "text"}
    queries.jsonl   {"_id", "text"}
    qrels/<split>.tsv   query-id <TAB> corpus-id <TAB> score   (with header)

Splits are ``train``, ``dev`` and ``test``. When a source has no dev split, one is
carved out of train with a fixed seed so that model selection never touches test.
"""

from __future__ import annotations

import ast
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 13


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    language: str
    domain: str
    role: str  # "in-domain": has train/dev/test, "held-out": test only, never trained on


DATASETS = {
    "scifact": DatasetInfo("scifact", "en", "scientific claims", "in-domain"),
    "nfcorpus": DatasetInfo("nfcorpus", "en", "medical", "in-domain"),
    "fiqa": DatasetInfo("fiqa", "en", "personal finance", "in-domain"),
    "alloprof": DatasetInfo("alloprof", "fr", "school course notes", "in-domain"),
    "bsard": DatasetInfo("bsard", "fr", "statutory law", "in-domain"),
    "scidocs": DatasetInfo("scidocs", "en", "scientific papers", "held-out"),
    "syntec": DatasetInfo("syntec", "fr", "collective agreement", "held-out"),
}

IN_DOMAIN = [n for n, d in DATASETS.items() if d.role == "in-domain"]
HELD_OUT = [n for n, d in DATASETS.items() if d.role == "held-out"]

# Alloprof ships 2,316 test questions. Cross-encoder reranking on a laptop CPU is the
# bottleneck of the pipeline benchmark, so a fixed random subset is used everywhere.
ALLOPROF_TEST_QUERIES = 600


# --------------------------------------------------------------------------- loading


def load_corpus(name: str) -> dict[str, dict]:
    out = {}
    with open(DATA_DIR / name / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[d["_id"]] = {"title": d.get("title") or "", "text": d.get("text") or ""}
    return out


def load_queries(name: str) -> dict[str, str]:
    out = {}
    with open(DATA_DIR / name / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[d["_id"]] = d["text"]
    return out


def load_qrels(name: str, split: str) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(DATA_DIR / name / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for qid, did, score in reader:
            qrels[qid][did] = int(float(score))
    return dict(qrels)


def has_split(name: str, split: str) -> bool:
    return (DATA_DIR / name / "qrels" / f"{split}.tsv").exists()


def doc_text(doc: dict) -> str:
    title, text = doc["title"].strip(), doc["text"].strip()
    return f"{title}\n\n{text}" if title else text


# --------------------------------------------------------------------------- writing


def _write(name: str, corpus: dict, queries: dict, qrels: dict[str, dict]) -> None:
    root = DATA_DIR / name
    (root / "qrels").mkdir(parents=True, exist_ok=True)
    with open(root / "corpus.jsonl", "w", encoding="utf-8") as f:
        for did, d in corpus.items():
            f.write(json.dumps({"_id": did, "title": d["title"], "text": d["text"]}, ensure_ascii=False) + "\n")
    used = {q for split in qrels.values() for q in split}
    with open(root / "queries.jsonl", "w", encoding="utf-8") as f:
        for qid, text in queries.items():
            if qid in used:
                f.write(json.dumps({"_id": qid, "text": text}, ensure_ascii=False) + "\n")
    for split, rels in qrels.items():
        with open(root / "qrels" / f"{split}.tsv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["query-id", "corpus-id", "score"])
            for qid in sorted(rels):
                for did, s in rels[qid].items():
                    w.writerow([qid, did, s])


def _carve_dev(train: dict, n: int) -> tuple[dict, dict]:
    qids = sorted(train)
    random.Random(SEED).shuffle(qids)
    dev_ids = set(qids[:n])
    return ({q: r for q, r in train.items() if q not in dev_ids},
            {q: r for q, r in train.items() if q in dev_ids})


def _read_tsv_qrels(path: str) -> dict[str, dict[str, int]]:
    rels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for qid, did, score in reader:
            if int(float(score)) > 0:
                rels[qid][did] = int(float(score))
    return dict(rels)


def _hf(repo: str, filename: str) -> str:
    return hf_hub_download(repo, filename, repo_type="dataset")


# --------------------------------------------------------------------------- sources


def _prepare_mteb(name: str, repo: str, splits: list[str]) -> None:
    corpus = {}
    with open(_hf(repo, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus[str(d["_id"])] = {"title": d.get("title") or "", "text": d.get("text") or ""}
    queries = {}
    with open(_hf(repo, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            queries[str(d["_id"])] = d["text"]
    qrels = {s: _read_tsv_qrels(_hf(repo, f"qrels/{s}.tsv")) for s in splits}
    if "train" in qrels and "dev" not in qrels:
        qrels["train"], qrels["dev"] = _carve_dev(qrels["train"], max(50, len(qrels["train"]) // 10))
    _write(name, corpus, queries, qrels)


def _prepare_alloprof() -> None:
    repo = "lyon-nlp/alloprof"
    docs = json.load(open(_hf(repo, "documents.json"), encoding="utf-8"))
    corpus = {d["uuid"]: {"title": d["title"] or "", "text": d["text"] or ""} for d in docs}
    queries, qrels = {}, {}
    for split, filename in [("train", "queries-train.json"), ("test", "queries-test.json")]:
        rels = {}
        for q in json.load(open(_hf(repo, filename), encoding="utf-8")):
            relevant = q["relevant"]
            if isinstance(relevant, str):
                relevant = ast.literal_eval(relevant)
            relevant = [r for r in relevant if r in corpus]
            if not relevant or not q["text"].strip():
                continue
            qid = f"{split}-{q['id']}"
            queries[qid] = q["text"].strip()
            rels[qid] = {r: 1 for r in relevant}
        qrels[split] = rels
    test_ids = sorted(qrels["test"])
    random.Random(SEED).shuffle(test_ids)
    qrels["test"] = {q: qrels["test"][q] for q in sorted(test_ids[:ALLOPROF_TEST_QUERIES])}
    qrels["train"], qrels["dev"] = _carve_dev(qrels["train"], 500)
    _write("alloprof", corpus, queries, qrels)


def _prepare_bsard() -> None:
    repo = "maastrichtlawtech/bsard"
    corpus = {}
    with open(_hf(repo, "articles.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            corpus[row["id"]] = {"title": row["reference"] or "", "text": row["article"] or ""}
    queries, qrels = {}, {}
    for split, filename in [("train", "questions_train.csv"), ("test", "questions_test.csv")]:
        rels = {}
        with open(_hf(repo, filename), encoding="utf-8") as f:
            for row in csv.DictReader(f):
                text = row["question"].strip()
                if row.get("extra_description", "").strip():
                    text = f"{text} {row['extra_description'].strip()}"
                ids = [a.strip() for a in row["article_ids"].split(",") if a.strip() in corpus]
                if not ids:
                    continue
                qid = f"{split}-{row['id']}"
                queries[qid] = text
                rels[qid] = {a: 1 for a in ids}
        qrels[split] = rels
    qrels["train"], qrels["dev"] = _carve_dev(qrels["train"], 88)
    _write("bsard", corpus, queries, qrels)


def _prepare_syntec() -> None:
    repo = "lyon-nlp/mteb-fr-retrieval-syntec-s2p"
    docs = json.load(open(_hf(repo, "documents.json"), encoding="utf-8"))
    corpus = {d["id"]: {"title": d.get("title") or "", "text": d.get("content") or ""} for d in docs}
    queries, rels = {}, {}
    for i, q in enumerate(json.load(open(_hf(repo, "queries.json"), encoding="utf-8"))):
        if q["Article"] in corpus:
            qid = f"q{i}"
            queries[qid] = q["Question"].strip()
            rels[qid] = {q["Article"]: 1}
    _write("syntec", corpus, queries, {"test": rels})


def prepare(name: str) -> None:
    if name == "scifact":
        _prepare_mteb(name, "mteb/scifact", ["train", "test"])
    elif name == "nfcorpus":
        _prepare_mteb(name, "mteb/nfcorpus", ["train", "dev", "test"])
    elif name == "fiqa":
        _prepare_mteb(name, "mteb/fiqa", ["train", "dev", "test"])
    elif name == "scidocs":
        _prepare_mteb(name, "mteb/scidocs", ["test"])
    elif name == "alloprof":
        _prepare_alloprof()
    elif name == "bsard":
        _prepare_bsard()
    elif name == "syntec":
        _prepare_syntec()
    else:
        raise ValueError(name)

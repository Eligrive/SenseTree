"""Generates one synthetic search query per document with a local LLM served by Ollama.

These queries replace human annotations in the unsupervised adaptation setting: the
only input is the document collection, which is what SenseTree has on a user's disk.
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

from sensetree_research.data import DATA_DIR, DATASETS, IN_DOMAIN, SEED, load_corpus

PROMPTS = {
    "en": (
        "Here is a document.\n\n---\n{doc}\n---\n\n"
        "Write one search query that someone would type to find this document. The query "
        "must be answerable from the document, must not copy a full sentence from it, and "
        "must be between 4 and 20 words. Reply with the query only."
    ),
    "fr": (
        "Voici un document.\n\n---\n{doc}\n---\n\n"
        "Écris une requête de recherche qu'une personne taperait pour retrouver ce document. "
        "La requête doit trouver sa réponse dans le document, ne doit pas recopier une phrase "
        "entière et doit faire entre 4 et 20 mots. Réponds uniquement par la requête."
    ),
}


def generate(url: str, model: str, prompt: str, seed: int) -> str:
    resp = requests.post(f"{url}/api/chat", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 64, "num_ctx": 2048, "seed": seed},
    }, timeout=300)
    resp.raise_for_status()
    text = resp.json()["message"]["content"].strip()
    return text.splitlines()[0].strip().strip('"«» ') if text else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=IN_DOMAIN)
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--url", default="http://localhost:11434")
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument("--doc-chars", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    for name in args.datasets:
        corpus = load_corpus(name)
        doc_ids = sorted(d for d, doc in corpus.items() if len((doc["title"] + doc["text"]).strip()) >= 50)
        random.Random(SEED).shuffle(doc_ids)
        doc_ids = doc_ids[: args.max_docs]

        out_path = DATA_DIR / name / "synthetic_queries.jsonl"
        done = set()
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                done = {json.loads(line)["doc_id"] for line in f}
        todo = [d for d in doc_ids if d not in done]
        template = PROMPTS[DATASETS[name].language]

        with open(out_path, "a", encoding="utf-8") as out, ThreadPoolExecutor(args.workers) as pool:
            futures = {}
            for i, did in enumerate(todo):
                doc = corpus[did]
                body = (f"{doc['title']}\n\n{doc['text']}" if doc["title"] else doc["text"])[: args.doc_chars]
                futures[pool.submit(generate, args.url, args.model, template.format(doc=body), SEED + i)] = did
            for fut in tqdm(as_completed(futures), total=len(futures), desc=name, mininterval=30):
                did = futures[fut]
                try:
                    query = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"{name} {did}: {e}", flush=True)
                    continue
                if 3 <= len(query.split()) <= 40:
                    out.write(json.dumps({"qid": f"gen-{did}", "doc_id": did, "text": query}, ensure_ascii=False) + "\n")
                    out.flush()


if __name__ == "__main__":
    main()

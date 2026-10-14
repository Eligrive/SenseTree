# SenseTree

Local semantic search and file organisation for Windows. SenseTree indexes the folders you choose, retrieves files by meaning as well as by keyword, and lets a local LLM agent propose (never apply on its own) moves, renames and deletions. Nothing leaves the machine unless you point a model slot at a remote server.

<p>
  <img alt="version" src="https://img.shields.io/badge/version-3.10.0-555">
  <img alt="platform" src="https://img.shields.io/badge/platform-Windows-555">
  <img alt="stack" src="https://img.shields.io/badge/Tauri%202-Rust%20%2B%20React-555">
</p>

## Retrieval quality

Search quality is measured with the application's own code on five public benchmarks (English and French, 500 to 58,000 documents), with paired significance tests. Full protocol, per-query results and every ablation: [research/README.md](research/README.md).

| Configuration (nDCG@10) | SciFact | NFCorpus | FiQA | Alloprof (fr) | BSARD (fr) | mean |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.677 | 0.318 | 0.245 | 0.341 | 0.198 | 0.356 |
| Dense (multilingual-e5-small) | 0.644 | 0.320 | 0.337 | 0.393 | 0.253 | 0.389 |
| Hybrid (RRF) | 0.697 | 0.351 | 0.345 | 0.493 | 0.285 | 0.434 |
| Hybrid + cross-encoder on top 90 (3.10) | 0.722 | 0.337 | 0.387 | 0.561 | 0.362 | 0.474 |
| Hybrid + cross-encoder on top 20 (current) | 0.744 | 0.356 | 0.383 | 0.545 | 0.342 | 0.474 |
| Same, embedding model adapted without labels | 0.740 | 0.352 | 0.375 | 0.595 | 0.358 | 0.484 |
| Same, embedding model adapted with labels | 0.745 | 0.361 | 0.386 | 0.601 | 0.372 | 0.493 |

Main findings:

- **The cross-encoder dominates query cost on CPU.** On an 8-thread desktop CPU it took a median 64 s per search when it scored the top 90 candidates, against 0.1 to 0.2 s for query embedding, vector search, BM25 and fusion together. Scoring the top 20 gives the same mean quality for 12 s and is now the default. On an RTX 2080 Super, 90 candidates take about 1.5 s.
- **The 118M embedding model adapts well to a collection.** Fine-tuned on the training queries, it goes from 0.374 to 0.479 mean nDCG@10 in isolation, above multilingual-e5-large (0.452), five times its size. Part of that gain comes from documents that appear in both training and test judgements; the report separates the two cases.
- **Adaptation also works without labels.** Queries written by a 4B local LLM and filtered by the reranker lift the French sets (+0.217 on Alloprof, +0.053 on BSARD) and two held-out sets, and leave the English ones unchanged. This only needs the documents, which is what the app has on a user's disk.
- **With an adapted model, reranking becomes optional.** Hybrid search without the cross-encoder then matches the 3.10 pipeline with it. The int8 export (113 MB instead of 448 MB) scores the same as fp32, though it is not faster on a CPU without VNNI instructions.
- **A negative result.** Distilling the cross-encoder's score margins into the embedding model (MarginMSE, as in GPL) collapsed it; a contrastive loss on teacher-filtered queries works.

## What it does

- **Hybrid search.** Dense vectors (multilingual-e5-small, in-process ONNX) and BM25 (LanceDB full-text index), fused by reciprocal rank fusion, then reranked by a local cross-encoder (bge-reranker-v2-m3). Results are deduplicated per file and can be scoped to a folder.
- **Content extraction.** PDF, DOCX, PPTX, XLSX, HTML and source code are parsed; scanned PDFs are rendered page by page and read by a vision model; images are captioned; audio and video are transcribed. Each file also gets a one-paragraph description from a local LLM, used as context for its chunks.
- **Agent with reviewed actions.** A tool-using chat agent can search, read files and folders, and propose file operations. Every proposal is shown as a before/after plan with per-operation checkboxes; applying it is transactional, with rollback and a local trash. MCP servers can add tools.
- **Image search.** CLIP embeddings of photos, queried with text.
- **Folder audit.** Duplicates, empty folders, excessive nesting and catch-all folders are reported as suggestions.
- **Model slots.** Embedding, reasoning, vision, transcription and video description are configured independently: built-in ONNX engine, or any OpenAI-compatible server (Ollama, LM Studio, vLLM, a remote machine). An embedding model fine-tuned with `research/` can be loaded from a folder.

## Architecture

```
React 19 + TypeScript (explorer, search, agent chat, settings)
        │ Tauri IPC
Rust core
  crawler / file watcher ─▶ queue ─▶ worker ─▶ extraction, chunking, LLM description
  search: query embedding ─▶ LanceDB kNN ┐
          BM25 (LanceDB FTS) ────────────┴▶ RRF ─▶ cross-encoder ─▶ per-file dedup
  agent: tool loop, dry-run plans, rollback, MCP client
  storage: SQLite (catalogue, queue, descriptions, memory) · LanceDB (chunks, images)
  models: fastembed / ONNX Runtime (embedding, reranking, CLIP) or HTTP
```

The ranking steps live in `src-tauri/src/search.rs` as pure functions; the benchmark harness `src-tauri/examples/retrieval_eval.rs` calls the same functions. Module map: [Architecture](https://github.com/Eligrive/SenseTree/wiki/Architecture).

## Install

Installers are on the [Releases](https://github.com/Eligrive/SenseTree/releases) page: `sensetree_x.y.z_x64-setup.exe` (per-user, no admin rights) or the `.msi` for managed deployment. The app updates itself from signed releases. Data lives in `%APPDATA%\com.virgi.sensetree`.

The built-in engine covers embedding and reranking with no other install. Descriptions, vision, transcription and the agent need a model server such as [Ollama](https://ollama.com) or [LM Studio](https://lmstudio.ai); see [Models & Providers](https://github.com/Eligrive/SenseTree/wiki/Models-and-Providers).

## Build from source

Requirements: Node.js LTS, Rust stable, Visual Studio Build Tools (C++ workload) and `protoc` (needed by LanceDB).

```bash
npm install
npm run tauri dev       # development
npm run tauri build     # installers in src-tauri/target/release/bundle
cd src-tauri && cargo test --release --lib
```

Pushing a `vX.Y.Z` tag builds, signs and publishes a release through GitHub Actions.

## Documentation

[Wiki](https://github.com/Eligrive/SenseTree/wiki): installation, configuration, models and providers, retrieval, indexing pipeline, agent, MCP servers, the HTTP protocol used with model servers ([AI Server Protocol](https://github.com/Eligrive/SenseTree/wiki/AI-Server-Protocol)), troubleshooting.

# 🧠 SenseTree

**A local-first, semantic overlay for your file system.**
Hybrid semantic search, an agentic AI chat that *proposes* (never executes) file operations, and a proactive "gardener" — all on your machine, with the AI models *you* choose.

<p>
  <img alt="version" src="https://img.shields.io/badge/version-3.9.1-6d28d9">
  <img alt="platform" src="https://img.shields.io/badge/platform-Windows-0078d6">
  <img alt="stack" src="https://img.shields.io/badge/Tauri%20v2-Rust%20%2B%20React%2019-24c8db">
  <img alt="privacy" src="https://img.shields.io/badge/privacy-100%25%20local-16a34a">
</p>

---

## What is it?

Finding a file today means remembering *where* you put it or its *exact* name. Native search is lexical — it matches keywords, not meaning. And folder trees rot over time: overstuffed `Downloads`, duplicates, folders nested ten levels deep.

SenseTree is **not** a virtual file system disconnected from reality. It is a smart overlay on top of your *real* folders. It reads the **meaning** of your files with local embeddings, lets you search by concept, chat with a folder through an agent that can actually read and search it, and keeps your tree tidy — and **nothing ever leaves your machine**.

## ✨ Features

- **🔍 Hybrid search** — Dense vectors find the *idea* ("website redesign quote", "trip to Korea"); BM25 finds the exact serial number, IBAN or surname. Both are fused by Reciprocal Rank Fusion and reranked by a local cross-encoder. Scopeable to any folder, with snippets and scores.
- **🧠 Every file gets a "sense"** — Documents are extracted (PDF, DOCX, PPTX, XLSX, HTML, code), **scanned PDFs are rendered page by page and OCR'd** by a vision model, images are captioned, **audio and video are transcribed and visually described**, and unreadable binaries are described from their name, folder and neighbours. Nothing is left unindexed.
- **🤖 Agentic chat** — A ReAct loop with six built-in tools: search, read a file, list a folder, read extracted senses, propose actions, and remember durable facts across conversations. Live tool trace, clickable citations. Extendable with any **[MCP](https://modelcontextprotocol.io) server** (HTTP or stdio).
- **🛡️ Dry-Run actions** — The AI *proposes*, you *decide*. Every move / rename / delete / mkdir — and every correction of an extracted sense — is shown as an interactive **Before → After** plan with per-operation checkboxes. Nothing touches disk without your click; apply is transactional with **rollback**, and deletes go to a local trash.
- **🖼️ Visual image search** — CLIP encodes your photos and your query into the same space: type "sunset over a lake" and get sunsets over lakes, no captions involved. Fully local.
- **🌳 Meaning tree** — Browse a folder as a relevance heatmap instead of a flat list.
- **🪴 Gardener** — Audits a directory for exact duplicates, empty folders, excessive depth and junk-drawer folders — as suggestions, never silent changes. Per-folder health badges in the sidebar.
- **🧩 Model-agnostic AI** — Five independently configurable slots (embedding / reasoning / vision / transcription / video). Each runs on the **built-in local engine** (fastembed / ONNX) *or* any **OpenAI-compatible HTTP server** — Ollama, LM Studio, vLLM, a home server on your LAN, or an external API.
- **📚 Live model catalogs** — Pick models from **live leaderboards** (MTEB, OpenCompass) *and* the **live Ollama library**, so new models show up the day they ship. Sort by benchmark, popularity or recency, **pick the quantization yourself** (`9b-q4_K_M` at 6.6 GB vs `9b-q8_0` at 11 GB — the difference between fitting an 8 GB card and not), and filter to what actually fits your VRAM. One-click download, with automatic resolution of the Ollama / LM Studio install name.
- **⚙️ Tunable pipeline** — Sequential or batch scheduling, per-stage throughput metrics, per-content-type qualification toggles, reasoning-effort control, and a configurable block-vs-recursive folder classifier that keeps `venv`, `node_modules` and DAW sample packs out of your index.
- **🔒 100 % local & private** — Runs offline. Embeddings are computed in-process; LLM calls go only to the endpoints you configure. No telemetry, no account, no cloud. Every outbound request the app can make is [documented](https://github.com/Eligrive/SenseTree/wiki/AI-Server-Protocol#what-leaves-the-machine).

## 🖥️ Install (Windows)

Grab the latest installer from the [**Releases**](https://github.com/Eligrive/SenseTree/releases) page:

| File | Use it if… |
|------|-----------|
| `sensetree_x.y.z_x64-setup.exe` (**recommended**) | You're a normal user. Lightweight NSIS wizard, installs per-user (no admin needed). |
| `sensetree_x.y.z_x64_en-US.msi` | You deploy to many machines (GPO / Intune / SCCM), silent install. |

Both install SenseTree as a **normal Windows app** (Start menu entry, uninstallable from *Apps & Features*). Your data lives in `%APPDATA%\com.virgi.sensetree` and survives upgrades. Once installed, the app **updates itself**: it checks for new signed releases at startup and offers to install them. See the [Installation guide](https://github.com/Eligrive/SenseTree/wiki/Installation).

> To run the AI features you'll want a local model runner — [Ollama](https://ollama.com) or [LM Studio](https://lmstudio.ai) — or you can use the built-in local embedding engine with no extra install. See [Models & Providers](https://github.com/Eligrive/SenseTree/wiki/Models-and-Providers).

## 🚀 Quick start

1. **Launch** SenseTree.
2. **Add a folder** to index in the sidebar (e.g. `Documents`).
3. Open **Settings** and point the model slots at your local engine or Ollama/LM Studio. (Embedding works out-of-the-box, fully local.)
4. Let indexing run — watch progress, stages and throughput in the sidebar.
5. **Search** by meaning, **browse** the meaning tree, or **chat** with a folder and review any proposed reorganization as a Dry-Run diff.

Full walkthrough: [Getting Started](https://github.com/Eligrive/SenseTree/wiki/Getting-Started).

## 🏗️ Architecture (at a glance)

```
┌──────────────────────────────────────────────────────────────────┐
│  React 19 + TypeScript + Tailwind v4                             │
│  Explorer · Search · Meaning tree · Agent chat · Settings ·       │
│  Model catalog · Indexing queue · Throughput · Image search       │
└───────────────▲──────────────────────────────────────────────────┘
                │ Tauri IPC (typed commands + events)
┌───────────────┴──────────────────────────────────────────────────┐
│  Rust core (Tauri v2)                                            │
│   crawler ─┐                                                     │
│   watchdog ┼─▶ queue ─▶ worker ─▶ providers                      │
│   folders ─┘                       ├─ embedding  local ONNX/HTTP │
│   classifier                       ├─ reasoning  HTTP chat       │
│                                    ├─ vision     HTTP + image    │
│   search ── dense + BM25 + RRF ── rerank (local cross-encoder)   │
│   actions ── agent (ReAct) · Dry-Run · rollback · MCP tools      │
│   ┌──────────────┐  ┌───────────────────┐  ┌──────────────────┐  │
│   │ SQLite (r2d2)│  │ LanceDB           │  │ fastembed / ONNX │  │
│   │ catalog·queue│  │ chunks · images   │  │ embed·rerank·CLIP│  │
│   │ senses·memory│  │ + BM25 index      │  │  OR OpenAI HTTP  │  │
│   └──────────────┘  └───────────────────┘  └──────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

**Stack:** Tauri v2 · Rust · React 19 · TypeScript · Tailwind v4 · LanceDB (vectors + BM25) · SQLite via r2d2 (metadata) · fastembed / ONNX Runtime (local embeddings, reranking, CLIP) · hayro (PDF page rendering) · `reqwest` (OpenAI-compatible clients) · `notify` (filesystem watchdog).

A module-by-module map is in the [Architecture](https://github.com/Eligrive/SenseTree/wiki/Architecture) wiki page.

## 🛠️ Build from source

```bash
# Prerequisites: Node.js (LTS), Rust (stable), and protoc (Protocol Buffers compiler,
# required by LanceDB). On Windows: `choco install protoc` or `winget install protobuf`.
npm install
npm run tauri dev      # run in development
npm run tauri build    # produce the installers in src-tauri/target/release/bundle
```

Releases are automated: push a tag `vX.Y.Z` and the GitHub Actions workflow builds, signs the update artifacts, and publishes the release with the installers attached. See [Building from Source](https://github.com/Eligrive/SenseTree/wiki/Building-from-Source).

## 📖 Documentation

The complete documentation lives in the [**Wiki**](https://github.com/Eligrive/SenseTree/wiki):

**Getting started** — [Installation](https://github.com/Eligrive/SenseTree/wiki/Installation) · [Getting Started](https://github.com/Eligrive/SenseTree/wiki/Getting-Started) · [FAQ](https://github.com/Eligrive/SenseTree/wiki/FAQ)

**Using it** — [Configuration](https://github.com/Eligrive/SenseTree/wiki/Configuration) · [Models & Providers](https://github.com/Eligrive/SenseTree/wiki/Models-and-Providers) · [Semantic Search](https://github.com/Eligrive/SenseTree/wiki/Semantic-Search) · [Image Search](https://github.com/Eligrive/SenseTree/wiki/Image-Search) · [AI Chat & Agent](https://github.com/Eligrive/SenseTree/wiki/AI-Chat-and-Actions) · [Gardener](https://github.com/Eligrive/SenseTree/wiki/Gardener) · [Prompts](https://github.com/Eligrive/SenseTree/wiki/Prompts) · [MCP Servers](https://github.com/Eligrive/SenseTree/wiki/MCP-Servers)

**Under the hood** — [Embeddings](https://github.com/Eligrive/SenseTree/wiki/Embeddings) · [Retrieval & RAG](https://github.com/Eligrive/SenseTree/wiki/Retrieval-and-RAG) · [Indexing Pipeline](https://github.com/Eligrive/SenseTree/wiki/Indexing-Pipeline) · [Media: Audio & Video](https://github.com/Eligrive/SenseTree/wiki/Media-Audio-and-Video) · [AI Server Protocol](https://github.com/Eligrive/SenseTree/wiki/AI-Server-Protocol) · [Architecture](https://github.com/Eligrive/SenseTree/wiki/Architecture) · [Building from Source](https://github.com/Eligrive/SenseTree/wiki/Building-from-Source) · [Troubleshooting](https://github.com/Eligrive/SenseTree/wiki/Troubleshooting)

> Running your own inference server? [**AI Server Protocol**](https://github.com/Eligrive/SenseTree/wiki/AI-Server-Protocol) documents every HTTP request SenseTree makes — endpoint, payload, parameters, timeouts and error handling.

## 🤝 Contributing

Ideas on architecture, UX, or local-AI model selection are very welcome. Open an issue or a PR.

## Status

**v3.9.1** — hybrid RAG with reranking and contextual retrieval, agentic chat with MCP support and durable memory, multimodal indexing (vision, OCR of scanned PDFs, audio/video transcription and description), visual image search, live model catalogs with quantization picking, and signed auto-update.

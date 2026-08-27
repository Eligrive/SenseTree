# 🧠 SenseTree

**A local-first, semantic overlay for your file system.**
Semantic search, an AI chat that proposes (never executes) file operations, and a proactive "gardener" — all running on your machine, with the AI models *you* choose.

<p>
  <img alt="version" src="https://img.shields.io/badge/version-1.1.0-6d28d9">
  <img alt="platform" src="https://img.shields.io/badge/platform-Windows-0078d6">
  <img alt="stack" src="https://img.shields.io/badge/Tauri%20v2-Rust%20%2B%20React%2019-24c8db">
  <img alt="privacy" src="https://img.shields.io/badge/privacy-100%25%20local-16a34a">
</p>

---

## What is it?

Finding a file today means remembering *where* you put it or its *exact* name. Native search is lexical — it matches keywords, not meaning. And folder trees rot over time: overstuffed `Downloads`, duplicates, folders nested ten levels deep.

SenseTree is **not** a virtual file system disconnected from reality. It is a smart overlay on top of your *real* folders. It reads the **meaning** of your files with local embeddings, lets you search by concept, chat with a folder, and keeps your tree tidy — and **nothing ever leaves your machine**.

## ✨ Features

- **🔍 Semantic search** — Find files by idea ("website redesign quote", "trip to Korea"), scoped to any folder. Ranked results with snippets.
- **🌳 Meaning tree** — Browse a semantic map of a folder instead of a flat list.
- **💬 Chat assistant** — Ask questions about a folder, or ask it to reorganize files. Retrieval-augmented over your own index.
- **🛡️ Dry-Run actions** — The AI *proposes*, you *decide*. Every move / rename / delete / mkdir is shown as an interactive **Before → After** plan. Nothing touches disk without your click, and apply is transactional with **rollback** on failure.
- **🪴 Gardener** — Audits a directory for anomalies (semantic duplicates, empty folders, excessive depth, junk-drawer folders) and suggests fixes — as suggestions, never silent changes.
- **🧩 Model-agnostic AI** — Three independently configurable model slots (embedding / reasoning / vision). Each can run on the **built-in local engine** (fastembed / ONNX) *or* any **OpenAI-compatible HTTP server** (Ollama, LM Studio, a home server on your LAN, or an external API).
- **📚 Live model catalogs** — Pick embedding / reasoning / vision models from **live leaderboards** (MTEB, OpenCompass) *and* the **live Ollama library**, so brand-new models show up the day they ship — no hardcoded list to go stale. Sort by benchmark, popularity or recency, **pick the quantization yourself** (`9b-q4_K_M` at 6.6 GB vs `9b-q8_0` at 11 GB — the difference between fitting an 8 GB card and not), and filter to what actually **fits your VRAM**, using each tag's real published size. One-click download, with automatic resolution of the Ollama / LM Studio install name.
- **🧱 Smart folder handling** — A configurable *block vs. recursive* classifier decides whether to index a folder file-by-file or treat it as a single opaque unit (venv, `node_modules`, DAW sample packs…), keeping your index clean and fast.
- **🔒 100% local & private** — Runs entirely offline. Embeddings are generated internally; LLM reasoning/vision go only to the endpoints you configure. No telemetry, no cloud.

## 🖥️ Install (Windows)

Grab the latest installer from the [**Releases**](https://github.com/Eligrive/SenseTree/releases) page:

| File | Use it if… |
|------|-----------|
| `sensetree_x.y.z_x64-setup.exe` (**recommended**) | You're a normal user. Lightweight NSIS wizard, installs per-user (no admin needed). |
| `sensetree_x.y.z_x64_en-US.msi` | You deploy to many machines (GPO / Intune / SCCM), silent install. |

Both install SenseTree as a **normal Windows app** (Start menu entry, uninstallable from *Apps & Features*). Your data lives in `%APPDATA%\com.virgi.sensetree` and survives upgrades. See the [Installation guide](https://github.com/Eligrive/SenseTree/wiki/Installation) for details.

> To run the semantic/AI features you'll want a local model runner — [Ollama](https://ollama.com) or [LM Studio](https://lmstudio.ai) — or you can use the built-in local embedding engine with no extra install. See [Models & Providers](https://github.com/Eligrive/SenseTree/wiki/Models-and-Providers).

## 🚀 Quick start

1. **Launch** SenseTree.
2. **Add a folder** to index in the sidebar (e.g. `Documents`).
3. Open **Settings → Providers** and point the embedding / reasoning / vision slots at your local engine or Ollama/LM Studio. (Embedding works out-of-the-box, fully local.)
4. Let indexing run — watch progress in the header.
5. **Search** by meaning, **browse** the meaning tree, or **chat** with a folder and review any proposed reorganization as a Dry-Run diff.

Full walkthrough: [Getting Started](https://github.com/Eligrive/SenseTree/wiki/Getting-Started).

## 🏗️ Architecture (at a glance)

```
┌─────────────────────────────────────────────────────────────┐
│  React 19 + TypeScript + Tailwind v4   (Explorer · Search ·  │
│                                         Chat · Settings)     │
└───────────────▲─────────────────────────────────────────────┘
                │ Tauri IPC (typed commands)
┌───────────────┴─────────────────────────────────────────────┐
│  Rust core (Tauri v2)                                        │
│   crawler ─▶ worker ─▶ providers (embedding/reasoning/vision)│
│   watchdog ─┘            │                                    │
│   folders (block/recursive classifier)                       │
│   ┌──────────────┐   ┌──────────────┐   ┌─────────────────┐  │
│   │ SQLite (r2d2)│   │ LanceDB      │   │ fastembed/ONNX  │  │
│   │ metadata     │   │ vectors      │   │  OR  OpenAI HTTP│  │
│   └──────────────┘   └──────────────┘   └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Stack:** Tauri v2 · Rust · React 19 · TypeScript · Tailwind v4 · LanceDB (vectors) · SQLite via r2d2 (metadata) · fastembed / ONNX Runtime (local embeddings) · `reqwest` (OpenAI-compatible clients) · `notify` (filesystem watchdog).

A module-by-module map is in the [Architecture](https://github.com/Eligrive/SenseTree/wiki/Architecture) wiki page.

## 🛠️ Build from source

```bash
# Prerequisites: Node.js (LTS), Rust (stable), and protoc (Protocol Buffers compiler,
# required by LanceDB). On Windows: `choco install protoc` or `winget install protobuf`.
npm install
npm run tauri dev      # run in development
npm run tauri build    # produce the installers in src-tauri/target/release/bundle
```

Releases are automated: push a tag `vX.Y.Z` and the GitHub Actions workflow builds and drafts a release with the installers attached. See [Building from Source](https://github.com/Eligrive/SenseTree/wiki/Building-from-Source).

## 📖 Documentation

The complete documentation lives in the [**Wiki**](https://github.com/Eligrive/SenseTree/wiki):

- [Installation](https://github.com/Eligrive/SenseTree/wiki/Installation) · [Getting Started](https://github.com/Eligrive/SenseTree/wiki/Getting-Started) · [Configuration](https://github.com/Eligrive/SenseTree/wiki/Configuration)
- [Models & Providers](https://github.com/Eligrive/SenseTree/wiki/Models-and-Providers) · [Indexing Pipeline](https://github.com/Eligrive/SenseTree/wiki/Indexing-Pipeline) · [Semantic Search](https://github.com/Eligrive/SenseTree/wiki/Semantic-Search)
- [AI Chat & Dry-Run Actions](https://github.com/Eligrive/SenseTree/wiki/AI-Chat-and-Actions) · [Gardener](https://github.com/Eligrive/SenseTree/wiki/Gardener) · [Prompts](https://github.com/Eligrive/SenseTree/wiki/Prompts)
- [Architecture](https://github.com/Eligrive/SenseTree/wiki/Architecture) · [Troubleshooting](https://github.com/Eligrive/SenseTree/wiki/Troubleshooting) · [FAQ](https://github.com/Eligrive/SenseTree/wiki/FAQ)

## 🤝 Contributing

Ideas on architecture, UX, or local-AI model selection are very welcome. Open an issue or a PR.

## Status

**v1.1.0** — working end-to-end semantic pipeline, model-agnostic AI, Dry-Run actions, live model catalogs. The next major milestone (v2) will explore proactive gardening and in-app auto-update.

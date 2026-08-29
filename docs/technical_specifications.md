Technical Specifications: SenseTree

Status: implemented and shipping (v3.9.1). This document describes the delivered architecture. Exhaustive per-module detail is in the wiki (Architecture, Indexing Pipeline, Retrieval & RAG, AI Server Protocol).

1. Technology Stack

1.1. Application Core & OS Bridge

Primary Language: Rust (memory safety, data-race-free multi-threading, near-C++ performance).

Desktop Framework: Tauri v2 — a modern web UI with a lightweight binary and far lower RAM consumption than Electron.

System Watchdog: the notify crate with a 2-second debouncer, listening to create/modify/delete events with near-zero overhead.

Frontend-Backend Communication: Tauri's native IPC — typed asynchronous commands plus backend-to-frontend events, with no local HTTP overhead.

1.2. User Interface

Framework: React 19 with TypeScript.

Styling: Tailwind CSS v4. Icons: lucide-react.

Build: Vite.

1.3. Local Storage & Databases

Vector Database: LanceDB (Rust-native, embedded, serverless). Two tables — chunks (id, path, chunk_index, text, content_hash, mtime, vector) whose vector dimension follows the active embedding model, and images (path, vector) at CLIP's fixed 512 dimensions. A native BM25 full-text index is maintained on the chunks.text column and rebuilt lazily whenever new chunks were written.

Relational Database: SQLite via rusqlite with an r2d2 connection pool (WAL, busy_timeout, synchronous=NORMAL, foreign keys on). Holds the file catalog, the indexing queue with retry counters and last errors, the incremental-sync truth table, extracted senses and extracts, folder classification profiles, agent memory, and the Dry-Run transaction log. Migrations are additive and tolerant, with a versioned reconciliation that purges derived tables when a database predates the current pipeline.

All data lives in the user's application data directory; no remote server and no container is required.

1.4. Artificial Intelligence

Five independently configurable slots, plus two local auxiliary models.

Slot 1 — Embedding, two modes:

  Mode A, local (default): fastembed over ONNX Runtime, in-process. ORT is loaded dynamically at runtime (ORT_DYLIB_PATH), so a single binary serves both CPU and CUDA — the appropriate runtime is downloaded on first use, and CUDA falls back to CPU gracefully. Twelve models are offered (E5 multilingual family, BGE, GTE, ModernBERT, MiniLM, Nomic, MxBai) with their dimensions and, critically, their multilingual status exposed in the UI. E5 query/passage prefixes are applied automatically.

  Mode B, remote: any OpenAI-compatible /v1/embeddings endpoint, batched at indexing.batch_size. Against Ollama, a native /api/embed call is attempted first to bound num_ctx — measured at 5.78 GB vs 2.13 GB of VRAM on a 32k-context embedding model — with a silent fallback to the standard endpoint for any other server.

Slot 2 — Reasoning: any OpenAI-compatible /v1/chat/completions endpoint. Drives the chat agent (with native function calling), reorganization planning, document qualification, folder classification, unknown-file extraction and context guessing. Reasoning effort is settable per use, defaulting to none for indexing qualifications (measured 24.4 s vs 0.78 s per folder classification, identical answer) and to the server's own behaviour for chat and planning.

Slot 3 — Vision: any multimodal /v1/chat/completions endpoint. Images are sent inline as base64 data URLs. Used for image captions and for OCR of rendered PDF pages.

Slot 4 — Transcription: any /v1/audio/transcriptions endpoint, multipart/form-data. Endpoint path, language, response format, extra fields and timeout are all configurable; the media is uploaded as a stream, so file size is never bounded by memory.

Slot 5 — Video description: any /v1/chat/completions endpoint accepting a video_url part. Two delivery modes — a streamed base64 data URL with an exactly computed Content-Length, or a file:// URI when the server shares the filesystem.

Auxiliary local models: a cross-encoder reranker (bge-reranker-v2-m3 by default) and CLIP ViT-B/32 for visual search, both fastembed/ONNX, loaded on demand and released when idle.

2. Data Flow Architecture

2.1. Indexing Pipeline (Background)

Crawler (the past) and Watchdog (the present) both feed one SQLite queue; a folder classifier decides recursive vs block before descending. The async worker then drains it:

  Routing: extension first, magic bytes second, into text / document / image / media / AI-routing / ignored.

  Extraction: format-specific (pdf-extract and lopdf for PDF, zip for OOXML, hayro for page rendering when a PDF has no text layer). Extraction panics degrade to contextual indexing rather than consuming retries, since they are deterministic.

  Hashing: SHA-256 computed in 16 KB blocks. An unchanged hash skips embedding entirely.

  AI stages: vision caption or OCR, media transcription and description, then LLM qualification of the extracted content into a "sense".

  Chunking: structure-aware — paragraphs, then sentences, then hard windows — packed to chunk_size with word-safe overlap.

  Encoding: chunk #0's stored text carries the full qualification (making it BM25-findable); every other chunk's embedded text is prefixed with the filename and a compact qualification (contextual retrieval), while its stored text stays clean for snippets and reranking.

  Storage: delete-then-insert per path into LanceDB, plus sense, extract and metadata into SQLite.

Scheduling is user-selectable: sequential (one file end to end, index advances continuously) or batch (a slice of files through all LLM stages, then all embedding — one model swap per slice instead of one per file, with explicit unloading of remote models between phases).

Resilience: three attempts per file, with failures classified as transient (network, timeout, 5xx, 408, 429 — retried) or permanent (other 4xx — immediate degradation to a lesser extraction route). Folder classifications that would require a blocking LLM call are deferred and resolved by a background classifier, so indexing never stalls on model latency.

2.2. Search Pipeline

The query is embedded with the indexing model. Two retrieval legs run against LanceDB — dense cosine and BM25 full text — each optionally filtered by a path prefix for scoping. Their rankings are fused by Reciprocal Rank Fusion (k = 60) over (path, chunk_index). The top candidates are rescored by the local cross-encoder, truncated to 800 characters each. Results are then deduplicated to the best chunk per file, verified to still exist on disk, and returned with snippet and score.

A second view builds a relevance tree over the scope, propagating each file's best score up to its ancestors.

2.3. Agent Pipeline

The last user message seeds the context via the same hybrid search. A system prompt is assembled from the built-in agent rules, the user's prompt override, durable memory, the seeded excerpts and a bounded structural listing of the scoped folder. The model is then called with a tool schema (search_files, read_file, list_directory, read_semantics, propose_actions, remember, plus any MCP tools discovered from configured servers) in a ReAct loop of at most five rounds, each tool call announced to the UI before execution. A model without tool-calling support degrades to answering from the seeded context.

MCP client support covers Streamable HTTP (JSON-RPC 2.0, SSE-aware) and stdio transports, with a configuration-keyed discovery cache.

2.4. Action Pipeline (Dry-Run)

The model emits a structured plan of typed operations (move, rename, delete, mkdir, requalify) with per-operation reasons. The plan is validated against the configured roots, persisted as a draft in the transaction log, and rendered as an interactive Before/After diff with per-operation checkboxes. Nothing touches the disk at this stage.

On approval, an edited operation list is verified to be a subset of the stored draft, then three phases run in order: disk (each success journaled, full reverse rollback on the first failure), index synchronization (renames update the LanceDB path without re-embedding; deletions remove vectors and catalog rows), and semantics (requalifications, themselves restorable). Deletions move files into an internal trash directory rather than destroying them.

3. Security, Permissions & Constraints

Isolation (Tauri): the frontend has no direct disk access; every file operation goes through strictly typed, auditable Rust commands.

Path confinement: every agent tool touching the filesystem, and every path in a proposed plan, is checked against the configured roots with segment-boundary matching, so a root named Docs does not authorize DocsEvil. Comparison is textual: symlinks are not resolved, an accepted residual risk for a local single-user application.

Injection resistance at apply time: an approved operation list is accepted only if every entry is present in the originally recorded draft.

Model guardrails: a disabled or unreachable slot degrades the corresponding feature visibly rather than failing silently; the health indicator reports each slot's state without instantiating local models.

Outbound traffic: inference goes only to configured endpoints. The remaining network use is model provisioning (ONNX Runtime, Hugging Face weights), signed application updates, and cached public catalog metadata (MTEB, OpenCompass, the Ollama library, Hugging Face GGUF lookup) — none of which carries file names, contents or queries. MCP servers added by the user are the one path SenseTree does not bound.

Credentials are stored in plain text in settings.json, as is conventional for a local desktop application.

4. Distribution

Windows x64, packaged as both an NSIS per-user installer and an MSI for managed deployment, under a stable application identifier so upgrades install in place.

Automatic updates are cryptographically signed (minisign); the application verifies each update against an embedded public key before installing.

Continuous integration builds and tests on every push and pull request, and keeps the Rust build cache warm on a schedule; tagged releases build, sign and publish the installers and the update manifest.

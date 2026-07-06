Technical Specifications: SenseTree

1. Technology Stack

1.1. Application Core & OS Bridge

Primary Language: Rust 🦀 (Guarantees memory safety, data-race-free multi-threading, and near-C++ performance).

Desktop Framework: Tauri (v2).

Advantage: Allows creating the interface using modern web technologies while keeping a lightweight binary (< 15 MB) and vastly lower RAM consumption compared to Electron.

System Watchdog: Uses the Rust notify crate to listen to file system events (create, modify, delete) with near-zero overhead.

1.2. User Interface (Frontend)

Framework: React.js (via TypeScript).

Styling: Tailwind CSS + Shadcn/ui (for accessible, custom-styled "desktop-like" components).

Frontend-Backend Communication: Tauri's native IPC (Inter-Process Communication). Direct asynchronous message-passing between React and Rust with zero local HTTP overhead.

1.3. Local Storage & Databases

Vector Database: LanceDB (Rust-native, serverless) or Qdrant (embedded mode).

Advantage: These databases require no remote servers or heavy background services like Docker. All data is stored as local files within the user's secure system directory (AppData or local equivalent).

Relational Database (Metadata & Task Queue): SQLite (via the Rust rusqlite crate or sea-orm). Manages the historical record of actions, the file-indexing queue, and the state of the "Dry-Run" system.

1.4. Artificial Intelligence (Hybrid Local Architecture)

The AI is divided into two distinct pipelines to optimize local computing resources:

Pipeline 1: Embedding Generation (Two supported modes)

Mode 1: Local Embedded (Default)

ML Framework: Candle (by Hugging Face, a minimalist ML framework for Rust) or ort (ONNX Runtime Rust bindings). Runs natively inside the Rust binary.

SOTA Recommended Model: multilingual-e5-small or BGE-m3.

Goal: Allows indexing thousands of files in the background autonomously, leveraging the local machine's CPU/GPU.

Mode 2: Domestic Remote / Local Network (Power User Option)

Protocol: Local HTTP API complying with the OpenAI standard for the /v1/embeddings endpoint.

Goal: Offloads heavy vectorization computations to a powerful local server (e.g., a home server NAS or a local PC with a dedicated GPU). This saves laptop battery and CPU cycles while guaranteeing that data never leaves the home network.

Pipeline 2: Reasoning, Chat & Vision (External via Local API)

Protocol: Local HTTP API matching the OpenAI standard.

Target Inference Engines: Ollama, LM Studio, or llama.cpp (installed and run by the user).

SOTA Recommended Models: Llama-3-8B-Instruct (General Reasoning), Phi-3-Mini-4k-Instruct (Light/Fast), Moondream2 or LLaVA (for Vision).

2. Data Flow Architecture

2.1. Indexing Pipeline (Background Process)

This process runs in a separate, low-priority Rust thread to ensure the computer remains responsive.

Watchdog (notify): Detects a physical disk change (e.g., new_report.pdf).

Queue Manager: Appends the absolute file path to the SQLite indexing queue.

Extractor (Parsing):

Extracts raw text (using Rust crates like pdf-extract, docx-rs, etc.).

Generates a SHA-256 hash of the physical file.

Chunker: Splits the raw text into chunks of ~250 words with a 50-word overlap to preserve semantic context.

Encoder (Candle/ONNX): Translates chunks into dense vector embeddings (e.g., 384 dimensions).

Storage: Inserts vectors along with metadata (absolute path, hash, chunk ID) into LanceDB/Qdrant.

2.2. Semantic Search Pipeline

The user inputs a query: "Find last year's budget presentation".

The frontend sends the query via IPC to the Rust backend.

The Encoder (Candle) embeds the query string into a single query vector.

LanceDB performs an ultra-fast nearest-neighbor search (Cosine Similarity / HNSW index).

Rust verifies that the matched file paths still physically exist on disk.

The absolute paths and relevance scores are sent back to the React UI.

2.3. Action Pipeline (Safety "Dry-Run")

Prompt: The user inputs: "Sort these invoices by year".

LLM Call: Rust sends the current file list and instruction to the local LLM engine (e.g., Ollama).

Structured JSON Output: The LLM responds strictly in JSON (validated by Rust) mapping paths: {"current_path": "new_path"}.

Dry-Run View: Rust passes this JSON to the frontend. The UI freezes and displays an interactive Diff (Red = Old path, Green = New path).

Execution (Commit): Only after the user clicks "Approve", Rust uses native OS APIs (std::fs::rename) to safely move the physical files.

Index Sync: The old paths are removed from the vector DB, and the new paths are updated immediately without re-embedding (since the file contents did not change).

3. Security, Permissions & Constraints

Isolation (Tauri): The frontend Web environment is completely sandboxed and has no direct disk access (Tauri's JavaScript filesystem APIs are explicitly disabled). Every file operation must go through strictly typed, auditable Rust commands.

Model Guardrails: If no local inference runner (Ollama) is detected, reasoning features are visually disabled in the UI to prevent silent failures.

Transaction Rollbacks: If a batch file-move fails midway (e.g., a file is locked by another program), the Rust backend must execute a "rollback" (returning already moved files to their original paths) to prevent directory corruption.
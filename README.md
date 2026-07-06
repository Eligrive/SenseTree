🧠 SenseTree: Semantic File Explorer & Folder Optimizer

SenseTree is a "smart overlay" for your local file system. This project bridges the gap between classic hierarchical navigation (physical directories) and the power of local AI-driven semantic search, while acting as a proactive "gardener" for your hard drive.

🎯 The Problem

Today, finding a file requires remembering its exact location or precise name. Native search tools (Windows Search, Spotlight) are limited to lexical keyword matching. Furthermore, our folder structures quickly become chaotic over time (cluttered "Downloads" folders, duplicate files, overly deep directories, and scattered archives).

💡 The Solution

SenseTree is not a virtual file system disconnected from reality. It is a smart overlay that integrates directly with your existing OS file explorer. It understands the meaning of your files using local embeddings and helps you maintain a clean, logical folder architecture.

✨ Key Features

🔍 Hybrid Semantic Search: Search by concept or idea (e.g., "website redesign quote"), while filtering by specific physical folders.

🌱 AI "Gardener" (Optimization): Audits your folder tree to detect anomalies (overly deep folders, semantic duplicates) and suggests logical sorting structures for cluttered directories.

🛡️ "Dry-Run" Mode (Absolute Safety): The AI proposes, you decide. Any structural modification (renaming, moving, deleting) is displayed in an interactive "Before/After" diff dashboard. No changes are executed on your disk without your explicit validation.

💬 Contextual Chat Assistant: Chat directly with a specific folder to summarize its contents, find gaps, or ask the AI to draft complex file-organization steps.

🔒 100% Local & Private: Runs entirely offline. Generates vector embeddings internally and interfaces with your local LLMs (via Ollama, LM Studio, etc.). No data ever leaves your machine.

🏗️ Technical Architecture

Frontend / UI: Tauri v2 (Rust + React/TypeScript) for native performance and a modern desktop interface.

Core & OS Bridge: Rust 🦀 (Secure disk access, real-time directory watchdog using notify, async IPC).

Vector Database: LanceDB or Qdrant (embedded mode for an ultra-low memory footprint).

AI Engine & Embeddings: Native local execution via Candle/ONNX (multilingual-e5-small or BGE-m3) with fallback support for local network APIs (OpenAI standard) for LLM reasoning.

🚀 Next Steps (Roadmap)

[ ] Initialize the Tauri v2 project and configure React/Tailwind.

[ ] Develop the Rust "Watchdog" module to monitor local disk events.

[ ] Integrate SQLite (for metadata) and the embedded Vector Database.

[ ] Build a Proof of Concept (POC) for the local embedding pipeline (text chunking).

[ ] Create the "Dry-Run" UI Vue (Diff view) to validate AI-suggested file actions.

🤝 Contributing

The project is currently in the design and launch phase. All ideas regarding architecture, UX/UI, or local AI model selection are highly welcome!
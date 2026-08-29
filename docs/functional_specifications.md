Functional Specifications: SenseTree (Semantic Explorer & Optimizer)

Status: implemented and shipping (v3.9.1). This document describes the product as delivered.

1. Product Vision

The application is a "smart overlay" integrated into the user's existing native file system. It does not replace the classic OS folder architecture, but builds on top of it to iteratively improve it.

It is a hybrid tool reconciling two paradigms: it enables powerful semantic navigation (finding files by meaning) AND acts as a proactive gardener to audit, sort, clean, and streamline the physical directory structure on the local hard drive. Absolute privacy is guaranteed: all inference runs either in-process or on endpoints the user explicitly configures, with no telemetry and no mandatory cloud service.

2. Core Capabilities

A. Anchored Semantic Indexing (The "Nerve System")

The AI overlays a layer of semantic comprehension onto the physical file system without altering the base structure unless authorized.

Continuous Monitoring: Real-time detection of modifications through OS file-system watchers, plus an on-demand recursive crawler for the existing tree.

Physical/Semantic Bridge: Every file keeps its exact location in the OS and receives (a) an extracted content, (b) an LLM-written "sense" stating what the file is and its key facts, and (c) one or more dense vector embeddings.

Universal Coverage: No file is left out of the index. Four extraction routes guarantee it — textual, visual, media, and contextual (see section 2.B).

Structural Classification: Technical folders (virtualenvs, dependency trees, application bundles, DAW sample packs, build artifacts) are detected and indexed as a single opaque semantic unit rather than file by file, keeping the index clean without user curation. The aggressiveness of this classifier is a single user-facing slider.

B. Multimodal Meaning Extraction

Textual: PDF, DOCX, PPTX, XLSX, HTML, plain text, source code, structured data. Encrypted-but-unrestricted PDFs are decrypted transparently.

Visual: Images are captioned by a multimodal model. Scanned PDFs with no text layer have their pages rendered to images and read by the same model, which reports both what the page shows and its transcribed text.

Media: Audio and video, all containers and all sizes. Two independent and combinable sources of meaning — speech transcription, and visual description of what a video shows. The result is chunked and indexed like any document, so a long recording is searchable on any of its passages.

Contextual: Any unreadable file (virtual machine images, opaque binaries, oversized files, media a server refused) is described from its name, folder, extension, size and neighbouring files, enriched by an LLM guess at its likely nature.

Unknown types: A router samples the file and decides — extract as text, ask the LLM whether meaning can be pulled out of it, or fall back to context.

C. Hybrid Search Engine (Semantic + Lexical + Hierarchical)

A global search bar accepting natural-language queries while respecting local directory scopes.

Conceptual Search: Find files by meaning, even when the exact keywords are absent from the filename or content.

Lexical Robustness: A BM25 full-text leg runs alongside the vector leg, so serial numbers, identifiers, proper nouns and extensions remain reliably findable. The two rankings are fused by Reciprocal Rank Fusion.

Precision Reranking: A local cross-encoder rescores the shortlist by reading query and passage jointly, which is materially more accurate than comparing independent vectors.

Contextual Filters: Restrict a semantic search to a specific physical branch.

Meaning Tree: A relevance heat map over the folder tree — which branch of the disk is about this? — rather than a flat list.

Visual Search: A separate image search encodes photos and the query into a shared visual space, retrieving pictures by what they look like with no caption involved.

D. Conversational Mode & Reflection (The "Brain")

A chat panel allows the user to interact with their file system for complex analysis and action. It is an agent, not a single-shot prompt: it can search, read files, list folders and inspect extracted senses, iterating over several rounds before answering.

Folder Analysis: "Summarize the key concepts covered in these documents this month", "Are there any technical specification documents missing from this project folder?".

Action Execution: Complex structural changes expressed in natural language: "Rename all these invoices to the YYYY-MM-DD convention and move them to the corresponding year's bookkeeping folder."

Semantic Correction: The user (or the assistant, with approval) can correct a wrongly extracted sense. The correction is pinned, so no re-index regenerates over it, and the file is re-embedded so the fix takes effect in search and not only on screen.

Durable Memory: The assistant can record lasting facts and preferences, recalled in later conversations and manageable by the user.

Extensibility: External tools can be plugged into the agent through the Model Context Protocol, over HTTP or stdio.

E. Directory Audit & Optimization (The "Gardener")

The app acts as a diagnostic tool for the physical directory structure. It is strictly read-only; remediation always goes through the Dry-Run pipeline.

Structural Diagnosis: Flags anomalies — cluttered catch-all folders, excessive nesting depth, empty directories.

Exact Duplicate Detection: Reuses the SHA-256 content hash already computed during indexing, so detection is free.

Continuous Health: A background audit maintains a per-root health indicator surfaced in the UI.

3. User Experience (UX & Interface)

The interface keeps the user grounded by maintaining a clear link to the physical reality of their disk:

The Augmented Explorer (Main View): A split-pane view. On one side the classic folder tree matching the OS; on the other, semantic tags, per-file indexing status, and directory health indicators.

The Assistant Panel (Chat/Prompt): A side panel scoped to the currently selected directory, showing the agent's tool calls live and citing sources as clickable file links.

The Control Center (Dry-Run / Draft Mode): The absolute safety barrier. Before any structural action is committed, the interface displays a "Before/After" diff dashboard with per-operation checkboxes. The AI cannot execute file modifications without an explicit approval click. Application is transactional with automatic rollback on any failure, and deletions are moved to an internal, reversible trash.

Operational Transparency: The indexing queue exposes the file currently being processed and the AI stages it traverses; a throughput panel reports the measured speed of each AI stage separately.

4. Model Agnosticism

Five AI tasks are configured independently — embedding, reasoning, vision, transcription, video description — each pointing at the built-in local engine or at any OpenAI-compatible HTTP server. A live model catalog, sourced from public leaderboards and the official Ollama library, lets the user compare, filter by hardware fit, choose a quantization and install in one click, without any model name being hardcoded in the application.

Every system prompt driving the AI stages is user-editable, so meaning extraction can be retuned without recompiling.

5. Safety Properties

No disk modification without explicit user approval.

Transactional application with rollback; reversible deletions.

Agent file access is bounded to the user's indexed roots, with strict path-boundary checking.

Graceful degradation: an unavailable model never loses a file — it falls back to a lesser but valid extraction route. Transient failures are retried; permanent ones degrade immediately rather than blocking the queue.

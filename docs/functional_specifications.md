Functional Specifications: SenseTree (Semantic Explorer & Optimizer)

1. Product Vision

The application is a "smart overlay" integrated into the user's existing native file system. It does not replace the classic OS folder architecture, but builds on top of it to iteratively improve it.

It is a hybrid tool reconciling two paradigms: it enables powerful semantic navigation (finding files by meaning) AND acts as a proactive gardener to audit, sort, clean, and streamline the physical directory structure on the local hard drive. Absolute privacy is guaranteed through 100% local model execution.

2. Core Capabilities

A. Directory Audit & Optimization (The "Gardener")

The app acts as a diagnostic and remediation tool for the physical directory structure.

Structural Diagnosis: The AI analyzes the folder hierarchy and flags anomalies: directories that are too deep, thematic redundancies (e.g., files split between a Dev_Projects folder and a Code_Archives folder), or cluttered "catch-all" folders.

Assisted Sorting & Routing: When a folder is cluttered, the tool suggests a logical distribution. It identifies files and proposes routing them into the correct branches of the existing tree, or suggests creating relevant new subfolders.

Targeted Cleanup: Proactively identifies obsolete files, useless logs, duplicate versions, and empty folders to free up local disk space.

B. Anchored Semantic Indexing (The "Nerve System")

The AI overlays a layer of semantic comprehension onto the physical file system without altering the base structure unless authorized.

Continuous Monitoring: The application detects real-time modifications in the physical directory tree using OS file-system watchers.

Physical/Semantic Bridge: Every file retains its exact location in the OS but receives a dense vector embedding (via text and metadata extraction). The tool understands the file's purpose while maintaining its precise path.

C. Hybrid Search Engine (Semantic + Hierarchical)

A global search bar that accepts natural language queries while respecting local directory scopes.

Conceptual Search: Find files by meaning (e.g., "Find all 3D assets and scripts related to the water flow simulation"), even if the exact keywords are missing from the filename.

Contextual Filters: Restrict semantic search to a specific physical branch (e.g., "Search only within C:\Work").

D. Conversational Mode & Reflection (The "Brain")

A chat panel allows the user to interact with their file system for complex analysis and action.

Folder Analysis: The user selects a specific folder and asks: "Summarize the key concepts covered in these documents this month" or "Are there any technical specification documents missing from this project folder?".

Action Execution: The user can command complex structural changes using natural language: "Rename all these invoices to the YYYY-MM-DD convention and move them to the corresponding year's bookkeeping folder."

E. Semantic Vision & Multimodality (Visual Analysis)

Integration of models capable of processing non-text files to seamlessly map them to the logical folder structure.

Media Categorization: Analyzes images, diagrams, or screenshots to dynamically tag them and suggest placing them in the correct thematic folders.

3. User Experience (UX & Interface)

The interface is designed to keep the user grounded by maintaining a clear link to the physical reality of their disk:

The Augmented Explorer (Main View): A split-pane view. On one side, the classic folder tree (matching the OS). On the other, dynamic "Semantic Tags" or directory health indicators (e.g., a colored dot next to a folder meaning "Highly fragmented" or "Duplicate content detected").

The Assistant Panel (Chat/Prompt): A side panel to interact directly with the currently selected directory or branch in the tree.

The Control Center (Dry-Run / Draft Mode): The absolute safety barrier. Before any reorganizing action (moving, deleting, or creating folders) is committed, the interface freezes and displays a clear "Before/After" diff dashboard. The AI cannot execute file modifications on the physical disk without an explicit "Apply Changes" click.

4. Model Agnosticism

The application connects to the user's preferred local inference engines (via Ollama, LM Studio, etc.) to run the Reasoning model, the background Embedding model, and the on-demand Vision model.
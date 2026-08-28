// Types partagés avec le backend Rust (miroir des structures serde).

export type EmbeddingMode = "local" | "openai";

export interface EmbeddingConfig {
  mode: EmbeddingMode;
  model: string;
  base_url: string;
  api_key: string;
  dimensions: number;
  use_gpu: boolean;
}

/// Effort de raisonnement demandé à un modèle « thinking ».
///
/// `auto` n'envoie rien et laisse le serveur décider. Mesuré sur un serveur réel :
/// classer un dossier prend 24,4 s avec raisonnement contre 0,78 s en `none`, pour la
/// même réponse — décisif sur des milliers d'appels, sans intérêt sur un seul.
export type ReasoningEffort = "auto" | "none" | "low" | "medium" | "high";

export interface ChatConfig {
  base_url: string;
  model: string;
  api_key: string;
  enabled: boolean;
  reasoning_effort: ReasoningEffort;
}

/// Ordonnancement des trois étages IA pendant l'indexation.
///
/// `sequential` : chaque fichier va au bout (vision → reasoning → embedding) avant le
/// suivant — l'index avance en continu, mais on alterne les modèles à chaque fichier.
/// `batch` : toute la tranche passe par les LLM, puis par l'embedding — un seul
/// échange de modèles par tranche, au prix d'un index qui avance par paliers.
export type PipelineMode = "sequential" | "batch";

export interface IndexingConfig {
  roots: string[];
  chunk_size: number;
  overlap: number;
  batch_size: number;
  max_file_mb: number;
  block_bias: number; // 0 = très récursif, 1 = très bloc
  qualify_documents: boolean;
  qualify_images: boolean;
  qualify_context: boolean;
  /// Effort de raisonnement des QUALIFICATIONS d'indexation (défaut : `none`).
  qualify_effort: ReasoningEffort;
  pipeline_mode: PipelineMode;
  /// Fichiers par tranche en mode batch.
  batch_files: number;
}

/// Débit d'un étage IA. `null` = pas encore mesuré (à ne pas confondre avec zéro).
export interface StageStats {
  stage: "vision" | "reasoning" | "embedding";
  ops: number; // appels au modèle
  units: number; // fichiers (vision/reasoning) ou chunks (embedding)
  bytes: number;
  errors: number;
  seconds: number; // temps cumulé DANS l'étage
  ms_per_op: number | null;
  units_per_sec: number | null;
  mb_per_sec: number | null;
}

export interface Throughput {
  since_unix: number;
  /// Temps écoulé. Comparé à `seconds` de chaque étage, il dit quelle part du temps
  /// réel part dans les modèles — donc si le goulot est le modèle ou le reste.
  wall_seconds: number;
  stages: StageStats[];
}

/// Un modèle chargé à l'instant sur le serveur Ollama (via `/api/ps`).
export interface LoadedModel {
  name: string;
  size: number;
  size_vram: number; // 0 si le serveur tourne en CPU
  expires_at: string | null;
  parameter_size: string | null;
  quantization_level: string | null;
}

export interface PromptsConfig {
  folder_classify: string;
  folder_describe: string;
  file_extract: string;
  vision_caption: string;
  vision_ocr: string;
  chat_system: string;
  reorganize: string;
}

export interface RetrievalConfig {
  hybrid: boolean;
  rerank: boolean;
  reranker_model: string;
}

/// Un serveur MCP (Model Context Protocol) externe exposant des outils à l'agent.
/// Transport HTTP (`url`) OU stdio (`command` + `args`) — `command` a la priorité.
export interface McpServerConfig {
  name: string;
  url: string;
  auth: string;
  command: string;
  args: string[];
  enabled: boolean;
}

export interface AppConfig {
  embedding: EmbeddingConfig;
  reasoning: ChatConfig;
  vision: ChatConfig;
  indexing: IndexingConfig;
  retrieval: RetrievalConfig;
  mcp_servers: McpServerConfig[];
  prompts: PromptsConfig;
}

/// État d'un modèle d'embedding local (fastembed) : est-il déjà téléchargé ?
/// Un modèle d'embedding embarqué (fastembed/ONNX).
export interface LocalModelStatus {
  id: string;
  dimensions: number;
  downloaded: boolean;
  /// Seule la famille E5 l'est. Les autres s'effondrent sur un corpus non anglophone.
  multilingual: boolean;
}

/// Nom d'installation résolu (via les GGUF réels sur Hugging Face).
/// Une quantification réellement présente dans un dépôt GGUF Hugging Face.
export interface GgufQuant {
  quant: string; // "Q4_K_M", "IQ4_XS", "BF16"…
  bytes: number; // somme des parties si le modèle est scindé
  parts: number; // > 1 = GGUF scindé
}

export interface InstallInfo {
  hf: string;
  gguf_repo: string | null;
  /// Quantification retenue par défaut ; l'utilisateur peut en choisir une autre.
  quant: string | null;
  /// Toutes celles présentes dans le dépôt, de la plus légère à la plus lourde.
  quants: GgufQuant[];
  ollama: string | null; // ex. "hf.co/SuperPauly/…-gguf:Q8_0"
  lmstudio: string | null; // dépôt à charger dans LM Studio
}

/// Un tag précis d'un modèle Ollama — c'est ce qui porte la QUANTIFICATION.
///
/// `9b-q4_K_M` (6,6 Go) et `9b-q8_0` (11 Go) sont le même modèle : seul le tag dit
/// lequel tiendra dans la carte.
export interface OllamaTag {
  tag: string; // suffixe seul, à coller derrière `modele:`
  bytes: number | null; // null = pas de poids locaux (tag cloud)
  size_label: string | null; // libellé d'origine, ex. "6.6GB"
  context: string | null; // ex. "256K"
  modality: string | null; // ex. "Text, Image"
}

/// Un modèle de la bibliothèque officielle Ollama (source LIVE, rien de codé en dur).
///
/// Complète les benchmarks : ceux-ci disent qui est BON, ceci dit qui est
/// DISPONIBLE, populaire et récemment mis à jour.
export interface OllamaModel {
  name: string; // nom d'installation : `ollama pull <name>`
  description: string;
  capabilities: string[]; // vision, tools, thinking, embedding, audio, cloud…
  sizes: string[]; // ex. ["4b", "9b", "27b"]
  pulls: number; // normalisé pour le tri
  pulls_label: string; // libellé d'origine, ex. "18.5M"
  tags: number | null;
  updated: string | null; // ex. "Aug 26, 2026 11:07 PM UTC"
  updated_day: number | null; // jours depuis l'epoch, pour le tri par récence
}

/// Un classement MTEB disponible (global multilingue, ou par langue).
export interface BoardInfo {
  name: string; // ex. "MTEB(fra, v1)"
  display_name: string; // ex. "French"
  num_models: number | null;
  languages: number;
}

/// Score d'un modèle sur un classement.
/// `mean = null` signifie NON ÉVALUÉ — surtout pas « zéro ».
export interface BoardScore {
  board: string; // nom du classement
  mean: number | null;
  retrieval: number | null;
  rank: number | null;
  total: number | null;
}

/// Specs et scores d'un modèle (embedding MTEB, ou vision/reasoning OpenCompass).
export interface ModelBenchmark {
  name: string; // nom d'affichage
  hf: string | null; // dépôt Hugging Face (clé de résolution GGUF)
  closed: boolean; // modèle fermé / API-only (non installable localement)
  url: string | null;
  embed_dim: number | null;
  params_b: number | null;
  max_tokens: number | null;
  scores: BoardScore[];
}

export interface HealthReport {
  embedding_ok: boolean;
  embedding_detail: string;
  reasoning_ok: boolean;
  reasoning_detail: string;
  vision_ok: boolean;
  vision_detail: string;
}

export interface DirEntryInfo {
  path: string;
  name: string;
  is_directory: boolean;
  size_bytes: number;
  modified: number | null;
  extension: string | null;
  index_status: string | null;
  folder_mode: string | null; // 'recursive' | 'block' | 'pending' | null
  indexed: boolean;
  in_block: boolean;
  under_root: boolean;
}

export interface SearchResult {
  path: string;
  name: string;
  score: number;
  snippet: string;
}

export interface TreeNode {
  name: string;
  path: string;
  is_dir: boolean;
  score: number;
  children: TreeNode[];
}

/// Mode d'affichage des résultats de recherche.
export type ResultView = "list" | "tree" | "split";

export interface PathDetails {
  path: string;
  name: string;
  is_directory: boolean;
  size_bytes: number;
  modified: number | null;
  extension: string | null;
  indexed: boolean;
  status: string | null;
  last_error: string | null;
  doc_type: string | null;
  summary: string | null;
  extract: string | null;
  content_kind: string;
  folder_mode: string | null;
  in_block: boolean;
  file_count: number | null;
}

export type OpKind = "move" | "rename" | "delete" | "mkdir";

export interface Operation {
  kind: OpKind;
  old_path: string | null;
  new_path: string | null;
  reason: string;
}

export interface ActionPlan {
  transaction_id: number | null;
  summary: string;
  operations: Operation[];
}

export interface ApplyResult {
  applied: number;
  message: string;
}

export interface DuplicateGroup {
  content_hash: string;
  paths: string[];
}

export interface DirectoryReport {
  scanned_path: string;
  file_count: number;
  max_depth: number;
  empty_dirs: string[];
  duplicate_groups: DuplicateGroup[];
  cluttered: boolean;
  suggestions: string[];
}

// --- Gardener proactif (diagnostic structurel de fond, lecture seule) ---
export type HealthSeverity = "ok" | "info" | "warn";

export interface FolderHealth {
  path: string;
  name: string;
  file_count: number;
  direct_files: number;
  duplicate_files: number;
  duplicate_groups: number;
  empty_dirs: number;
  max_depth: number;
  cluttered: boolean;
  severity: HealthSeverity;
  headline: string;
}

export interface GardenerReport {
  folders: FolderHealth[];
  anomaly_count: number;
  scanned_at: number;
}

/// Une note de la mémoire durable de l'agent.
export interface MemoryItem {
  id: number;
  note: string;
}

/// Un résultat de recherche d'image par similarité visuelle (CLIP).
export interface ImageHit {
  path: string;
  name: string;
  score: number;
}

export interface ChatTurn {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface ChatSource {
  path: string;
  name: string;
  score: number;
  snippet: string;
}

export interface ChatResponse {
  answer: string | null;
  sources: ChatSource[];
  plan: ActionPlan | null;
}

export interface IndexingStats {
  total: number;
  pending: number;
  completed: number;
  failed: number;
  pending_folders: number;
}

/// Étapes du pipeline d'un élément de la file, dans l'ordre : sous-ensemble de
/// { "vision", "reasoning", "embedding" }.
export interface IndexActivity {
  path: string;
  routes: string[];
  kind: string;
}

export interface QueueItem {
  path: string;
  routes: string[];
  kind: string;
  status: string;
  retry_count: number;
  last_error: string | null;
}

export interface IndexingQueueView {
  current: IndexActivity | null;
  pending: QueueItem[];
  /// Échecs définitifs, fournis à part (toujours visibles même avec une file énorme).
  failed: QueueItem[];
  stats: IndexingStats;
}

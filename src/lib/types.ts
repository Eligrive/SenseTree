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

export interface ChatConfig {
  base_url: string;
  model: string;
  api_key: string;
  enabled: boolean;
}

export interface IndexingConfig {
  roots: string[];
  chunk_size: number;
  overlap: number;
  batch_size: number;
  max_file_mb: number;
  block_bias: number; // 0 = très récursif, 1 = très bloc
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

export interface AppConfig {
  embedding: EmbeddingConfig;
  reasoning: ChatConfig;
  vision: ChatConfig;
  indexing: IndexingConfig;
  prompts: PromptsConfig;
}

/// État d'un modèle d'embedding local (fastembed) : est-il déjà téléchargé ?
export interface LocalModelStatus {
  id: string;
  dimensions: number;
  downloaded: boolean;
}

/// Specs et score de référence d'un modèle, tirés en direct de MTEB.
export interface ModelBenchmark {
  mteb_id: string;
  embed_dim: number | null;
  n_parameters: number | null;
  max_tokens: number | null;
  memory_mb: number | null;
  languages: string[];
  french: boolean;
  /// Moyenne ndcg@10 sur des tâches de retrieval ANGLAIS (seules comparables).
  retrieval_en: number | null;
  retrieval_tasks: number;
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

// Enveloppes typées autour de l'IPC Tauri.
//
// Rappel Tauri v2 : les noms d'arguments de commande passent en camelCase côté
// JS (convertis en snake_case côté Rust). En revanche, les CHAMPS des objets
// sérialisés (ex. AppConfig) restent en snake_case (noms serde).

import { invoke } from "@tauri-apps/api/core";
import type {
  ActionPlan,
  AppConfig,
  ApplyResult,
  BoardInfo,
  ChatResponse,
  ChatTurn,
  DirEntryInfo,
  DirectoryReport,
  GardenerReport,
  HealthReport,
  ImageHit,
  IndexingQueueView,
  IndexingStats,
  InstallInfo,
  LocalModelStatus,
  MemoryItem,
  ModelBenchmark,
  OllamaModel,
  OllamaTag,
  Operation,
  PathDetails,
  PromptsConfig,
  SearchResult,
  TreeNode,
} from "./types";

// --- Configuration & santé ---
export const getConfig = () => invoke<AppConfig>("get_config");
export const setConfig = (config: AppConfig) => invoke<void>("set_config", { config });
export const getDefaultPrompts = () => invoke<PromptsConfig>("get_default_prompts");
export const aiHealth = () => invoke<HealthReport>("ai_health");
export const gpuAvailable = () => invoke<boolean>("gpu_available");
export const testChatEndpoint = (baseUrl: string, apiKey: string, model: string) =>
  invoke<string>("test_chat_endpoint", { baseUrl, apiKey, model });
export const testEmbeddingEndpoint = (baseUrl: string, apiKey: string, model: string) =>
  invoke<string>("test_embedding_endpoint", { baseUrl, apiKey, model });
export const listInstalledModels = (baseUrl: string, apiKey: string) =>
  invoke<string[]>("list_installed_models", { baseUrl, apiKey });
export const listLocalModels = () => invoke<LocalModelStatus[]>("list_local_models");
export const downloadLocalModel = (model: string) =>
  invoke<string>("download_local_model", { model });
export const listBenchmarkBoards = () => invoke<BoardInfo[]>("list_benchmark_boards");
export const resolveInstalls = (names: string[]) =>
  invoke<InstallInfo[]>("resolve_installs", { names });
export const modelBenchmarks = (boards: string[], refresh = false) =>
  invoke<ModelBenchmark[]>("model_benchmarks", { boards, refresh });
export const listVisionBoards = () => invoke<BoardInfo[]>("list_vision_boards");
export const listReasoningBoards = () => invoke<BoardInfo[]>("list_reasoning_boards");
export const visionBenchmarks = (refresh = false) =>
  invoke<ModelBenchmark[]>("vision_benchmarks", { refresh });
export const reasoningBenchmarks = (refresh = false) =>
  invoke<ModelBenchmark[]>("reasoning_benchmarks", { refresh });
export const ollamaLibrary = (refresh = false) =>
  invoke<OllamaModel[]>("ollama_library", { refresh });
/// Tags de plusieurs modèles (nom → tags), avec leur taille : c'est la source du
/// choix de quantification.
export const ollamaTags = (models: string[]) =>
  invoke<Record<string, OllamaTag[]>>("ollama_tags", { models });
export const pullModel = (baseUrl: string, model: string) =>
  invoke<string>("pull_model", { baseUrl, model });
export const reindexAll = () => invoke<void>("reindex_all");

// --- Explorateur ---
export const listDirectory = (path: string) =>
  invoke<DirEntryInfo[]>("list_directory", { path });
export const getRoots = () => invoke<string[]>("get_roots");
export const pickFolder = () => invoke<string | null>("pick_folder");
export const addIndexedFolder = (path: string) =>
  invoke<string[]>("add_indexed_folder", { path });
export const removeIndexedFolder = (path: string) =>
  invoke<string[]>("remove_indexed_folder", { path });
export const openPath = (path: string) => invoke<void>("open_path", { path });
export const pathDetails = (path: string) => invoke<PathDetails>("path_details", { path });
export const setFileSummary = (path: string, summary: string) =>
  invoke<void>("set_file_summary", { path, summary });
export const qualifyFile = (path: string) => invoke<string>("qualify_file", { path });
export const qualifyFolder = (path: string) => invoke<number>("qualify_folder", { path });
export const indexingStats = () => invoke<IndexingStats>("indexing_stats");
export const indexingQueue = (limit?: number) =>
  invoke<IndexingQueueView>("indexing_queue", { limit });
export const retryIndexing = (path: string) => invoke<void>("retry_indexing", { path });
export const ignoreIndexing = (path: string) => invoke<void>("ignore_indexing", { path });
export const retryAllFailed = () => invoke<number>("retry_all_failed");
export const indexingPaused = () => invoke<boolean>("indexing_paused");
export const setIndexingPaused = (paused: boolean) =>
  invoke<void>("set_indexing_paused", { paused });
export const setFolderMode = (path: string, mode: "recursive" | "block") =>
  invoke<void>("set_folder_mode", { path, mode });

// --- Recherche sémantique ---
export const semanticSearch = (query: string, scope?: string, limit?: number) =>
  invoke<SearchResult[]>("semantic_search", { query, scope, limit });
export const semanticTree = (query: string, scope?: string, limit?: number) =>
  invoke<TreeNode | null>("semantic_tree", { query, scope, limit });

// --- Actions Dry-Run + gardener ---
export const planReorganization = (instruction: string, scope: string) =>
  invoke<ActionPlan>("plan_reorganization", { instruction, scope });
export const applyActionPlan = (transactionId: number, operations?: Operation[]) =>
  invoke<ApplyResult>("apply_action_plan", { transactionId, operations });
export const discardActionPlan = (transactionId: number) =>
  invoke<void>("discard_action_plan", { transactionId });
export const analyzeDirectory = (path: string) =>
  invoke<DirectoryReport>("analyze_directory", { path });
export const gardenerHealth = () => invoke<GardenerReport>("gardener_health");

// --- Mémoire de l'agent ---
export const agentMemoryList = () => invoke<MemoryItem[]>("agent_memory_list");
export const agentMemoryDelete = (id: number) => invoke<void>("agent_memory_delete", { id });
export const agentMemoryClear = () => invoke<void>("agent_memory_clear");

// --- Recherche d'images (CLIP) ---
export const indexImages = (scope?: string) => invoke<number>("index_images", { scope });
export const imageSearch = (query: string, limit?: number) =>
  invoke<ImageHit[]>("image_search", { query, limit });
export const imageDataUrl = (path: string) => invoke<string>("image_data_url", { path });
export const chatWithAssistant = (messages: ChatTurn[], scope?: string) =>
  invoke<ChatResponse>("chat_with_assistant", { messages, scope });

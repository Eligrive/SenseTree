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
  ChatTurn,
  DirEntryInfo,
  DirectoryReport,
  HealthReport,
  IndexingStats,
  SearchResult,
} from "./types";

// --- Configuration & santé ---
export const getConfig = () => invoke<AppConfig>("get_config");
export const setConfig = (config: AppConfig) => invoke<void>("set_config", { config });
export const aiHealth = () => invoke<HealthReport>("ai_health");
export const testChatEndpoint = (baseUrl: string, apiKey: string) =>
  invoke<string>("test_chat_endpoint", { baseUrl, apiKey });

// --- Explorateur ---
export const listDirectory = (path: string) =>
  invoke<DirEntryInfo[]>("list_directory", { path });
export const getRoots = () => invoke<string[]>("get_roots");
export const indexingStats = () => invoke<IndexingStats>("indexing_stats");

// --- Recherche sémantique ---
export const semanticSearch = (query: string, scope?: string, limit?: number) =>
  invoke<SearchResult[]>("semantic_search", { query, scope, limit });

// --- Actions Dry-Run + gardener ---
export const planReorganization = (instruction: string, scope: string) =>
  invoke<ActionPlan>("plan_reorganization", { instruction, scope });
export const applyActionPlan = (transactionId: number) =>
  invoke<ApplyResult>("apply_action_plan", { transactionId });
export const discardActionPlan = (transactionId: number) =>
  invoke<void>("discard_action_plan", { transactionId });
export const analyzeDirectory = (path: string) =>
  invoke<DirectoryReport>("analyze_directory", { path });
export const chatWithAssistant = (messages: ChatTurn[], scope?: string) =>
  invoke<string>("chat_with_assistant", { messages, scope });

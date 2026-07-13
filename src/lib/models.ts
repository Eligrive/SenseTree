// Catalogue curaté de modèles, pour choisir en connaissance de cause.
//
// Les notes `quality` (1–5) sont INDICATIVES : elles reflètent le classement des
// modèles sur les benchmarks de référence (MTEB multilingue pour les embeddings,
// MMLU / LMArena pour les LLM, MMMU pour la vision). Les scores exacts évoluent —
// le champ `benchmark` dit sur quoi la note s'appuie, sans inventer de chiffre.

export type Task = "embedding" | "reasoning" | "vision";
export type Backend = "local" | "server";
export type ServerKind = "ollama" | "lmstudio" | "unknown";

export interface CatalogModel {
  key: string;
  name: string;
  task: Task;
  /// Identifiant dans le dépôt MTEB (`Org__Modele`) → specs et score live.
  /// Absent pour les LLM de chat/vision (non évalués par MTEB).
  mteb?: string;
  /// Identifiant selon le backend. Absent = non disponible sur ce backend.
  local?: string;
  ollama?: string;
  lmstudio?: string;
  /// Dimensions du vecteur (embeddings uniquement).
  dims?: number;
  /// Taille approximative du modèle.
  params: string;
  languages: "multilingue" | "anglais";
  /// Note indicative (1–5) issue du classement sur le benchmark de référence.
  quality: 1 | 2 | 3 | 4 | 5;
  /// Sur quoi s'appuie la note.
  benchmark: string;
  /// À quoi ce modèle sert le mieux.
  goodFor: string;
}

export const CATALOG: CatalogModel[] = [
  // ---------------------------------------------------------------- EMBEDDINGS
  {
    key: "qwen3-embed-0.6b",
    name: "Qwen3-Embedding 0.6B",
    task: "embedding",
    mteb: "Qwen__Qwen3-Embedding-0.6B",
    ollama: "qwen3-embedding",
    lmstudio: "text-embedding-qwen3-embedding-0.6b",
    dims: 1024,
    params: "0,6 B",
    languages: "multilingue",
    quality: 5,
    benchmark: "MTEB multilingue : haut du classement (2025)",
    goodFor:
      "Meilleur rapport qualité/vitesse pour de la recherche sémantique FR+EN. Recommandé.",
  },
  {
    key: "bge-m3",
    name: "BGE-M3",
    task: "embedding",
    mteb: "BAAI__bge-m3",
    ollama: "bge-m3",
    lmstudio: "text-embedding-bge-m3",
    dims: 1024,
    params: "0,57 B",
    languages: "multilingue",
    quality: 4,
    benchmark: "MTEB multilingue : très solide, référence éprouvée",
    goodFor:
      "Valeur sûre multilingue, contexte long (8k). À prendre si Qwen3 pose souci.",
  },
  {
    key: "arctic-embed2",
    name: "Snowflake Arctic-Embed 2",
    task: "embedding",
    mteb: "Snowflake__snowflake-arctic-embed-l-v2.0",
    ollama: "snowflake-arctic-embed2",
    dims: 1024,
    params: "0,3 B",
    languages: "multilingue",
    quality: 4,
    benchmark: "MTEB multilingue : bon niveau, léger",
    goodFor: "Bon compromis multilingue si tu veux un modèle plus petit.",
  },
  {
    key: "e5-large",
    name: "Multilingual E5 large",
    task: "embedding",
    mteb: "intfloat__multilingual-e5-large",
    local: "multilingual-e5-large",
    lmstudio: "text-embedding-multilingual-e5-large",
    dims: 1024,
    params: "0,56 B",
    languages: "multilingue",
    quality: 4,
    benchmark: "MTEB multilingue : solide (génération précédente)",
    goodFor: "Le meilleur choix multilingue disponible en LOCAL (embarqué).",
  },
  {
    key: "e5-base",
    name: "Multilingual E5 base",
    task: "embedding",
    mteb: "intfloat__multilingual-e5-base",
    local: "multilingual-e5-base",
    dims: 768,
    params: "0,28 B",
    languages: "multilingue",
    quality: 3,
    benchmark: "MTEB multilingue : correct",
    goodFor: "Compromis local entre qualité et vitesse.",
  },
  {
    key: "e5-small",
    name: "Multilingual E5 small",
    task: "embedding",
    mteb: "intfloat__multilingual-e5-small",
    local: "multilingual-e5-small",
    dims: 384,
    params: "0,12 B",
    languages: "multilingue",
    quality: 3,
    benchmark: "MTEB multilingue : honnête pour sa taille",
    goodFor: "Le plus rapide/léger en local (défaut de l'app). Idéal CPU.",
  },
  {
    key: "nomic-embed",
    name: "Nomic Embed Text v1.5",
    task: "embedding",
    mteb: "nomic-ai__nomic-embed-text-v1.5",
    local: "nomic-embed-text",
    ollama: "nomic-embed-text",
    dims: 768,
    params: "0,14 B",
    languages: "anglais",
    quality: 3,
    benchmark: "MTEB anglais : bon ; faible en multilingue",
    goodFor: "Corpus surtout anglais. Dispo en local ET sur Ollama.",
  },
  {
    key: "mxbai-embed",
    name: "mxbai-embed-large v1",
    task: "embedding",
    mteb: "mixedbread-ai__mxbai-embed-large-v1",
    local: "mxbai-embed-large",
    ollama: "mxbai-embed-large",
    dims: 1024,
    params: "0,33 B",
    languages: "anglais",
    quality: 3,
    benchmark: "MTEB anglais : très bon ; faible en multilingue",
    goodFor: "Corpus anglais. Dispo en local ET sur Ollama.",
  },
  {
    key: "bge-base-en",
    name: "BGE base EN v1.5",
    task: "embedding",
    mteb: "BAAI__bge-base-en-v1.5",
    local: "bge-base-en-v1.5",
    dims: 768,
    params: "0,11 B",
    languages: "anglais",
    quality: 3,
    benchmark: "MTEB anglais : bon",
    goodFor: "Corpus anglais, local et léger.",
  },
  {
    key: "bge-small-en",
    name: "BGE small EN v1.5",
    task: "embedding",
    mteb: "BAAI__bge-small-en-v1.5",
    local: "bge-small-en-v1.5",
    dims: 384,
    params: "0,03 B",
    languages: "anglais",
    quality: 2,
    benchmark: "MTEB anglais : correct pour sa taille",
    goodFor: "Anglais, très léger.",
  },
  {
    key: "all-minilm",
    name: "all-MiniLM-L6-v2",
    task: "embedding",
    mteb: "sentence-transformers__all-MiniLM-L6-v2",
    local: "all-minilm",
    ollama: "all-minilm",
    dims: 384,
    params: "0,02 B",
    languages: "anglais",
    quality: 2,
    benchmark: "MTEB anglais : basique (modèle historique)",
    goodFor: "Ultra léger/rapide. Qualité limitée — dépannage seulement.",
  },

  // ---------------------------------------------------------------- REASONING
  {
    key: "qwen2.5-7b",
    name: "Qwen2.5 7B",
    task: "reasoning",
    ollama: "qwen2.5:7b",
    lmstudio: "qwen2.5-7b-instruct",
    params: "7 B",
    languages: "multilingue",
    quality: 4,
    benchmark: "MMLU / LMArena : très bon pour sa taille, solide en FR",
    goodFor:
      "Classification des dossiers, extraction de sens, chat. Recommandé (tient sur 8 Go).",
  },
  {
    key: "llama3.1-8b",
    name: "Llama 3.1 8B",
    task: "reasoning",
    ollama: "llama3.1:8b",
    lmstudio: "meta-llama-3.1-8b-instruct",
    params: "8 B",
    languages: "multilingue",
    quality: 4,
    benchmark: "MMLU / LMArena : très bon, référence répandue",
    goodFor: "Alternative solide au Qwen2.5. Défaut de l'app.",
  },
  {
    key: "llama3.2-3b",
    name: "Llama 3.2 3B",
    task: "reasoning",
    ollama: "llama3.2:3b",
    lmstudio: "llama-3.2-3b-instruct",
    params: "3 B",
    languages: "multilingue",
    quality: 3,
    benchmark: "MMLU : correct pour 3 B",
    goodFor: "Machine modeste : classification rapide, moins bon en raisonnement.",
  },
  {
    key: "phi3-mini",
    name: "Phi-3 mini",
    task: "reasoning",
    ollama: "phi3:mini",
    params: "3,8 B",
    languages: "anglais",
    quality: 3,
    benchmark: "MMLU : excellent pour sa taille, mais surtout anglais",
    goodFor: "Très léger. Moins adapté si tes fichiers sont en français.",
  },

  // ---------------------------------------------------------------- VISION
  {
    key: "llama3.2-vision",
    name: "Llama 3.2 Vision 11B",
    task: "vision",
    ollama: "llama3.2-vision",
    params: "11 B",
    languages: "multilingue",
    quality: 4,
    benchmark: "MMMU : bon niveau, description d'image fiable",
    goodFor: "Description d'images et OCR de PDF scannés. Le plus qualitatif ici.",
  },
  {
    key: "llava",
    name: "LLaVA",
    task: "vision",
    ollama: "llava",
    lmstudio: "llava-v1.5-7b",
    params: "7 B",
    languages: "anglais",
    quality: 3,
    benchmark: "MMMU : correct, classique",
    goodFor: "Description d'images générique, bien supporté.",
  },
  {
    key: "moondream",
    name: "Moondream 2",
    task: "vision",
    ollama: "moondream",
    params: "1,8 B",
    languages: "anglais",
    quality: 2,
    benchmark: "MMMU : limité, mais ultra léger",
    goodFor: "Vision sur machine modeste (défaut de l'app). Descriptions sommaires.",
  },
];

/// Nom Hugging Face du modèle (clé de l'API MTEB) : `Org__Modele` → `Org/Modele`.
export function hfName(m: CatalogModel): string | undefined {
  return m.mteb?.replace(/__/g, "/");
}

/// Devine le type de serveur d'après son URL (ports par défaut d'Ollama / LM Studio).
export function serverKind(baseUrl: string): ServerKind {
  const u = (baseUrl ?? "").toLowerCase();
  if (u.includes(":1234") || u.includes("lmstudio") || u.includes("lm-studio")) return "lmstudio";
  if (u.includes(":11434") || u.includes("ollama")) return "ollama";
  return "unknown";
}

/// Identifiant du modèle pour le backend courant (undefined = indisponible ici).
export function idForBackend(
  m: CatalogModel,
  backend: Backend,
  serverKind: ServerKind
): string | undefined {
  if (backend === "local") return m.local;
  if (serverKind === "lmstudio") return m.lmstudio;
  return m.ollama; // ollama, ou serveur inconnu (on tente le nom Ollama)
}

/// Backends où le modèle existe (pour les badges de disponibilité).
export function availabilityOf(m: CatalogModel): string[] {
  const out: string[] = [];
  if (m.local) out.push("Local");
  if (m.ollama) out.push("Ollama");
  if (m.lmstudio) out.push("LM Studio");
  return out;
}

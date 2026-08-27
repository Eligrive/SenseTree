import { useEffect, useState, type ReactNode } from "react";
import { listen } from "@tauri-apps/api/event";
import {
  BookOpen,
  Brain,
  ChevronDown,
  ChevronRight,
  Download,
  Loader2,
  Plug,
  Plus,
  RefreshCw,
  RotateCcw,
  Save,
  Trash2,
  X,
} from "lucide-react";
import type {
  AppConfig,
  ChatConfig,
  LocalModelStatus,
  MemoryItem,
  PromptsConfig,
} from "../lib/types";
import type { Backend, ServerKind, Task } from "../lib/models";
import { serverKind } from "../lib/models";
import ModelCatalog from "./ModelCatalog";
import {
  agentMemoryClear,
  agentMemoryDelete,
  agentMemoryList,
  downloadLocalModel,
  getConfig,
  getDefaultPrompts,
  gpuAvailable,
  listInstalledModels,
  listLocalModels,
  pullModel,
  reindexAll,
  setConfig,
  testChatEndpoint,
  testEmbeddingEndpoint,
} from "../lib/ipc";

interface Props {
  open: boolean;
  onClose: () => void;
  onSaved: () => void;
}

// Catalogue des modèles d'embedding avec leurs disponibilités :
//   local  = exécutable en embarqué (fastembed / ONNX, sur CPU, ou GPU si runtime CUDA).
//   server = disponible sur un serveur (Ollama / LM Studio) — donc GPU si le serveur en a.
// Certains modèles (all-minilm, nomic, mxbai) existent des DEUX côtés sous le même
// nom : on peut passer du local (CPU) au serveur (GPU) sans changer de modèle.
type EmbedModel = { id: string; dims: number; local: boolean; server: boolean };
const EMBED_CATALOG: EmbedModel[] = [
  { id: "multilingual-e5-small", dims: 384, local: true, server: false },
  { id: "multilingual-e5-base", dims: 768, local: true, server: false },
  { id: "multilingual-e5-large", dims: 1024, local: true, server: false },
  { id: "bge-small-en-v1.5", dims: 384, local: true, server: false },
  { id: "bge-base-en-v1.5", dims: 768, local: true, server: false },
  { id: "bge-large-en-v1.5", dims: 1024, local: true, server: false },
  { id: "gte-base-en-v1.5", dims: 768, local: true, server: false },
  { id: "gte-large-en-v1.5", dims: 1024, local: true, server: false },
  { id: "modernbert-embed-large", dims: 1024, local: true, server: false },
  { id: "all-minilm", dims: 384, local: true, server: true },
  { id: "nomic-embed-text", dims: 768, local: true, server: true },
  { id: "mxbai-embed-large", dims: 1024, local: true, server: true },
  { id: "bge-m3", dims: 1024, local: false, server: true },
  { id: "snowflake-arctic-embed2", dims: 1024, local: false, server: true },
];
function capLabel(m: EmbedModel): string {
  if (m.local && m.server) return "local ou serveur";
  if (m.local) return "local uniquement";
  return "serveur uniquement";
}

// Retrouve un modèle du catalogue par son id, y compris pour les variantes
// LM Studio (ex. "text-embedding-nomic-embed-text-v1.5" → nomic-embed-text).
function findEmbed(modelId: string): EmbedModel | undefined {
  const id = modelId.toLowerCase();
  const base = id.split(":")[0];
  return EMBED_CATALOG.find((x) => {
    const xid = x.id.toLowerCase();
    return xid === id || xid === base || id.includes(xid);
  });
}

// Capacité d'un modèle (local / serveur / les deux), pour une notation uniforme.
function capOf(modelId: string): string | null {
  const m = findEmbed(modelId);
  return m ? capLabel(m) : null;
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-[11px] font-medium uppercase tracking-wider text-zinc-500">
        {label}
      </span>
      {children}
    </label>
  );
}

const inputCls =
  "w-full rounded-lg border border-zinc-800 bg-zinc-900 px-2.5 py-1.5 text-sm text-zinc-200 outline-none focus:border-blue-500";

const OLLAMA_URL = "http://localhost:11434/v1";
const LMSTUDIO_URL = "http://localhost:1234/v1";
const SERVER_PRESETS = [
  { label: "Ollama", url: OLLAMA_URL },
  { label: "LM Studio", url: LMSTUDIO_URL },
];

const CUSTOM = "__custom__";

/// Sélecteur de modèle fiable (les <datalist> sont capricieux dans WebView2) :
/// un vrai menu déroulant natif + une saisie libre en mode « Personnalisé ».
function ModelSelect({
  value,
  options,
  onPick,
  placeholder,
}: {
  value: string;
  options: string[];
  onPick: (v: string) => void;
  placeholder?: string;
}) {
  const inList = options.includes(value);
  const [custom, setCustom] = useState(!inList);
  useEffect(() => {
    if (!options.includes(value)) setCustom(true);
  }, [value, options]);

  return (
    <div className="space-y-1">
      <select
        className={inputCls}
        value={custom ? CUSTOM : value}
        onChange={(e) => {
          if (e.target.value === CUSTOM) {
            setCustom(true);
          } else {
            setCustom(false);
            onPick(e.target.value);
          }
        }}
      >
        {options.length === 0 && (
          <option value={CUSTOM} disabled>
            aucun modèle détecté sur le serveur
          </option>
        )}
        {options.map((m) => (
          <option key={m} value={m}>
            {m}
          </option>
        ))}
        <option value={CUSTOM}>Personnalisé…</option>
      </select>
      {custom && (
        <input
          className={inputCls}
          value={value}
          placeholder={placeholder ?? "nom du modèle"}
          onChange={(e) => onPick(e.target.value)}
        />
      )}
    </div>
  );
}

const PROMPT_FIELDS: { key: keyof PromptsConfig; label: string; rows: number }[] = [
  { key: "folder_classify", label: "Classification des dossiers (récursif / bloc)", rows: 7 },
  { key: "folder_describe", label: "Description d'un dossier-bloc", rows: 3 },
  { key: "file_extract", label: "Extraction d'un fichier de type inconnu", rows: 4 },
  { key: "vision_caption", label: "Légende d'image (vision)", rows: 3 },
  { key: "vision_ocr", label: "OCR d'image (vision)", rows: 2 },
  { key: "chat_system", label: "Assistant de chat (RAG + actions)", rows: 7 },
  { key: "reorganize", label: "Planificateur de réorganisation", rows: 4 },
];

function biasLabel(v: number): string {
  if (v <= 0.2) return "très récursif";
  if (v <= 0.4) return "plutôt récursif";
  if (v < 0.6) return "équilibré";
  if (v < 0.8) return "plutôt bloc";
  return "très bloc";
}

export default function SettingsModal({ open, onClose, onSaved }: Props) {
  const [cfg, setCfg] = useState<AppConfig | null>(null);
  const [saving, setSaving] = useState(false);
  const [reindexing, setReindexing] = useState(false);
  const [testMsg, setTestMsg] = useState<Record<string, string>>({});
  const [pulling, setPulling] = useState<Record<string, boolean>>({});
  const [installed, setInstalled] = useState<Record<string, string[]>>({});
  const [pullProgress, setPullProgress] = useState<
    Record<string, { percent: number; status: string }>
  >({});

  const [gpuSupported, setGpuSupported] = useState(false);
  const [defaultPrompts, setDefaultPrompts] = useState<PromptsConfig | null>(null);
  const [showPrompts, setShowPrompts] = useState(false);
  // Modèles locaux (fastembed) et leur état de téléchargement.
  const [localModels, setLocalModels] = useState<LocalModelStatus[]>([]);
  // Catalogue ouvert pour telle tâche (null = fermé) + téléchargements en cours.
  const [catalogTask, setCatalogTask] = useState<Task | null>(null);
  const [dlBusy, setDlBusy] = useState<Record<string, boolean>>({});

  const [memories, setMemories] = useState<MemoryItem[]>([]);

  const refreshLocalModels = () =>
    listLocalModels().then(setLocalModels).catch(() => setLocalModels([]));
  const refreshMemories = () =>
    agentMemoryList().then(setMemories).catch(() => setMemories([]));

  useEffect(() => {
    if (open) {
      getConfig().then(setCfg).catch(() => setCfg(null));
      gpuAvailable().then(setGpuSupported).catch(() => setGpuSupported(false));
      getDefaultPrompts().then(setDefaultPrompts).catch(() => setDefaultPrompts(null));
      refreshLocalModels();
      refreshMemories();
    }
  }, [open]);

  // Progression du téléchargement de modèle (événements émis par le backend).
  useEffect(() => {
    if (!open) return;
    const un = listen<{ model: string; percent: number; status: string }>(
      "model-pull-progress",
      (e) => {
        const p = e.payload;
        setPullProgress((m) => ({ ...m, [p.model]: { percent: p.percent, status: p.status } }));
      }
    );
    return () => {
      un.then((f) => f());
    };
  }, [open]);

  // Charge les modèles installés pour chaque endpoint dès qu'on a la config.
  useEffect(() => {
    if (!cfg) return;
    for (const key of ["reasoning", "vision"] as const) {
      refreshModels(cfg[key].base_url, cfg[key].api_key);
    }
    if (cfg.embedding.mode === "openai") {
      refreshModels(cfg.embedding.base_url, cfg.embedding.api_key);
    }
  }, [cfg?.reasoning.base_url, cfg?.vision.base_url, cfg?.embedding.base_url, cfg?.embedding.mode]);

  // Endpoint RÉELLEMENT configuré pour la tâche du catalogue (souvent un serveur
  // distant). Calculé ici, avant l'effet, pour pouvoir le sonder lui aussi.
  const catalogEndpointUrl =
    catalogTask === "embedding"
      ? cfg?.embedding.base_url ?? ""
      : catalogTask === "reasoning"
        ? cfg?.reasoning.base_url ?? ""
        : catalogTask === "vision"
          ? cfg?.vision.base_url ?? ""
          : "";
  const catalogEndpointKey =
    catalogTask === "embedding"
      ? cfg?.embedding.api_key ?? ""
      : catalogTask === "reasoning"
        ? cfg?.reasoning.api_key ?? ""
        : catalogTask === "vision"
          ? cfg?.vision.api_key ?? ""
          : "";
  const endpointIsPreset =
    catalogEndpointUrl === OLLAMA_URL || catalogEndpointUrl === LMSTUDIO_URL;

  // Quand le catalogue est ouvert, on sonde les deux serveurs standard (Ollama +
  // LM Studio) ET l'endpoint configuré — sans ce dernier, les modèles installés sur
  // un serveur distant n'apparaissaient jamais comme « installés ».
  useEffect(() => {
    if (!catalogTask) return;
    refreshModels(OLLAMA_URL, "");
    refreshModels(LMSTUDIO_URL, "");
    if (catalogEndpointUrl && !endpointIsPreset) {
      refreshModels(catalogEndpointUrl, catalogEndpointKey);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [catalogTask, catalogEndpointUrl, catalogEndpointKey, endpointIsPreset]);

  if (!open || !cfg) return null;

  const refreshModels = (baseUrl: string, apiKey: string) => {
    listInstalledModels(baseUrl, apiKey)
      .then((models) => setInstalled((m) => ({ ...m, [baseUrl]: models })))
      .catch(() => setInstalled((m) => ({ ...m, [baseUrl]: [] })));
  };

  const patchChat = (key: "reasoning" | "vision", patch: Partial<ChatConfig>) =>
    setCfg({ ...cfg, [key]: { ...cfg[key], ...patch } });

  const patchPrompt = (key: keyof PromptsConfig, value: string) =>
    setCfg({ ...cfg, prompts: { ...cfg.prompts, [key]: value } });

  const test = async (key: "reasoning" | "vision") => {
    setTestMsg((m) => ({ ...m, [key]: "…" }));
    try {
      const res = await testChatEndpoint(cfg[key].base_url, cfg[key].api_key, cfg[key].model);
      setTestMsg((m) => ({ ...m, [key]: `✅ ${res}` }));
    } catch (e) {
      setTestMsg((m) => ({ ...m, [key]: `⚠️ ${String(e)}` }));
    }
  };

  const testEmbedding = async () => {
    setTestMsg((m) => ({ ...m, embedding: "…" }));
    try {
      const res = await testEmbeddingEndpoint(
        cfg.embedding.base_url,
        cfg.embedding.api_key,
        cfg.embedding.model
      );
      setTestMsg((m) => ({ ...m, embedding: `✅ ${res}` }));
    } catch (e) {
      setTestMsg((m) => ({ ...m, embedding: `⚠️ ${String(e)}` }));
    }
  };

  // Téléchargement générique d'un modèle (Ollama /api/pull) — reasoning, vision OU embedding.
  const downloadModel = async (slot: string, baseUrl: string, apiKey: string, model: string) => {
    if (!model.trim()) return;
    setPulling((p) => ({ ...p, [slot]: true }));
    setTestMsg((m) => ({ ...m, [slot]: `⏳ téléchargement de « ${model} »…` }));
    try {
      const res = await pullModel(baseUrl, model);
      setTestMsg((m) => ({ ...m, [slot]: `✅ ${res}` }));
      refreshModels(baseUrl, apiKey);
    } catch (e) {
      setTestMsg((m) => ({ ...m, [slot]: `⚠️ ${String(e)}` }));
    } finally {
      setPulling((p) => ({ ...p, [slot]: false }));
    }
  };

  const download = (key: "reasoning" | "vision") =>
    downloadModel(key, cfg[key].base_url, cfg[key].api_key, cfg[key].model);

  // --- Catalogue de modèles ------------------------------------------------
  const chatKey = catalogTask === "reasoning" || catalogTask === "vision" ? catalogTask : null;
  const catalogBackend: Backend =
    catalogTask === "embedding" && cfg.embedding.mode === "local" ? "local" : "server";
  const catalogUrl =
    catalogTask === "embedding" ? cfg.embedding.base_url : chatKey ? cfg[chatKey].base_url : "";
  const catalogApiKey =
    catalogTask === "embedding" ? cfg.embedding.api_key : chatKey ? cfg[chatKey].api_key : "";
  const catalogCurrent =
    catalogTask === "embedding" ? cfg.embedding.model : chatKey ? cfg[chatKey].model : "";
  const localDownloaded: Record<string, boolean> = Object.fromEntries(
    localModels.map((m) => [m.id, m.downloaded])
  );

  // Serveurs connus (Ollama + LM Studio) avec leurs modèles installés → sert au
  // catalogue pour savoir où un modèle vit et basculer l'endpoint à la sélection.
  const catalogServers = [
    // L'endpoint configuré passe EN PREMIER : c'est celui que l'utilisateur utilise
    // réellement, donc il doit primer pour « déjà installé » et comme cible de
    // téléchargement (sinon un modèle présent sur le serveur distant était ignoré).
    ...(catalogEndpointUrl && !endpointIsPreset
      ? [
          {
            url: catalogEndpointUrl,
            kind: (catalogEndpointUrl.includes(":1234") ? "lmstudio" : "ollama") as ServerKind,
            models: installed[catalogEndpointUrl] ?? [],
          },
        ]
      : []),
    { url: OLLAMA_URL, kind: "ollama" as ServerKind, models: installed[OLLAMA_URL] ?? [] },
    { url: LMSTUDIO_URL, kind: "lmstudio" as ServerKind, models: installed[LMSTUDIO_URL] ?? [] },
  ];

  const catalogUse = (id: string, dims?: number, serverUrl?: string) => {
    if (catalogTask === "embedding") {
      setCfg({
        ...cfg,
        embedding: {
          ...cfg.embedding,
          model: id,
          dimensions: dims ?? cfg.embedding.dimensions,
          // Modèle serveur choisi → on bascule en mode serveur sur le bon endpoint.
          ...(serverUrl ? { mode: "openai" as const, base_url: serverUrl } : {}),
        },
      });
    } else if (chatKey) {
      patchChat(chatKey, { model: id, ...(serverUrl ? { base_url: serverUrl } : {}) });
    }
  };

  const catalogDownload = async (id: string, targetUrl?: string) => {
    setDlBusy((b) => ({ ...b, [id]: true }));
    try {
      if (catalogBackend === "local") {
        await downloadLocalModel(id);
        await refreshLocalModels();
      } else {
        const url = targetUrl ?? catalogUrl;
        await pullModel(url, id);
        // Rafraîchit les deux serveurs → le modèle apparaît dans les listes déroulantes.
        refreshModels(OLLAMA_URL, "");
        refreshModels(LMSTUDIO_URL, "");
        // Ollama enregistre parfois avec un léger délai : second passage.
        setTimeout(() => refreshModels(url, catalogApiKey), 1500);
      }
    } catch (e) {
      setTestMsg((m) => ({ ...m, save: `⚠️ ${String(e)}` }));
    } finally {
      setDlBusy((b) => ({ ...b, [id]: false }));
    }
  };

  const catalogButton = (task: Task) => (
    <button
      onClick={() => setCatalogTask(task)}
      title="Parcourir le catalogue (benchmarks, langues, disponibilité)"
      className="flex items-center gap-1.5 rounded-lg bg-zinc-800 px-2.5 py-1.5 text-xs text-zinc-200 hover:bg-zinc-700"
    >
      <BookOpen size={13} /> Catalogue
    </button>
  );

  const save = async () => {
    setSaving(true);
    try {
      await setConfig(cfg);
      onSaved();
      onClose();
    } catch (e) {
      setTestMsg((m) => ({ ...m, save: `⚠️ ${String(e)}` }));
    } finally {
      setSaving(false);
    }
  };

  const reindex = async () => {
    setReindexing(true);
    try {
      await reindexAll();
      setTestMsg((m) => ({ ...m, save: "🔄 Réindexation lancée" }));
    } catch (e) {
      setTestMsg((m) => ({ ...m, save: `⚠️ ${String(e)}` }));
    } finally {
      setReindexing(false);
    }
  };

  const chatSection = (key: "reasoning" | "vision", title: string) => {
    // Menu déroulant = UNIQUEMENT les modèles réellement présents sur le serveur.
    // Pour en découvrir/installer d'autres → bouton « Catalogue ».
    const options = installed[cfg[key].base_url] ?? [];
    const isInstalled = (installed[cfg[key].base_url] ?? []).some(
      (m) => m === cfg[key].model || m.split(":")[0] === cfg[key].model.split(":")[0]
    );
    return (
      <section className="space-y-3 rounded-xl border border-zinc-800 bg-zinc-900/30 p-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-zinc-200">{title}</h3>
          <label className="flex items-center gap-1.5 text-xs text-zinc-400">
            <input
              type="checkbox"
              checked={cfg[key].enabled}
              onChange={(e) => patchChat(key, { enabled: e.target.checked })}
            />
            Activé
          </label>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <Field label="URL du serveur (base)">
            <input
              className={inputCls}
              value={cfg[key].base_url}
              onChange={(e) => patchChat(key, { base_url: e.target.value })}
              placeholder="http://localhost:11434/v1"
            />
            <div className="mt-1 flex gap-1.5">
              {SERVER_PRESETS.map((s) => (
                <button
                  key={s.label}
                  onClick={() => patchChat(key, { base_url: s.url })}
                  className={`rounded-md px-2 py-0.5 text-[10px] ${
                    cfg[key].base_url === s.url
                      ? "bg-blue-500/20 text-blue-300"
                      : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
                  }`}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </Field>
          <Field label="Modèle">
            <div className="flex items-start gap-1.5">
              <div className="flex-1">
                <ModelSelect
                  value={cfg[key].model}
                  options={options}
                  onPick={(v) => patchChat(key, { model: v })}
                />
              </div>
              <button
                onClick={() => download(key)}
                disabled={pulling[key]}
                title="Télécharger ce modèle (Ollama)"
                className="flex h-[34px] shrink-0 items-center rounded-lg bg-zinc-800 px-2.5 text-zinc-200 hover:bg-zinc-700 disabled:opacity-50"
              >
                {pulling[key] ? (
                  <Loader2 size={14} className="animate-spin" />
                ) : (
                  <Download size={14} />
                )}
              </button>
            </div>
            <span
              className={`mt-1 block text-[11px] ${isInstalled ? "text-emerald-400" : "text-amber-400"}`}
            >
              {isInstalled ? "● installé sur le serveur" : "○ non installé — télécharger ou vérifier le nom"}
            </span>
            {pulling[key] && (
              <div className="mt-1.5">
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
                  <div
                    className="h-full rounded-full bg-blue-500 transition-all"
                    style={{ width: `${pullProgress[cfg[key].model]?.percent ?? 0}%` }}
                  />
                </div>
                <span className="mt-0.5 block text-[10px] text-zinc-500">
                  {pullProgress[cfg[key].model]?.status ?? "démarrage…"}{" "}
                  {pullProgress[cfg[key].model]?.percent ?? 0}%
                </span>
              </div>
            )}
          </Field>
        </div>
        <Field label="Clé API (optionnelle)">
          <input
            className={inputCls}
            type="password"
            value={cfg[key].api_key}
            onChange={(e) => patchChat(key, { api_key: e.target.value })}
            placeholder="vide pour un serveur local"
          />
        </Field>
        <div className="flex items-center gap-2">
          <button
            onClick={() => test(key)}
            className="flex items-center gap-1.5 rounded-lg bg-zinc-800 px-3 py-1.5 text-xs text-zinc-200 hover:bg-zinc-700"
          >
            <Plug size={13} /> Tester
          </button>
          <button
            onClick={() => refreshModels(cfg[key].base_url, cfg[key].api_key)}
            title="Rafraîchir la liste des modèles installés"
            className="flex items-center gap-1.5 rounded-lg bg-zinc-800 px-2 py-1.5 text-xs text-zinc-200 hover:bg-zinc-700"
          >
            <RefreshCw size={13} />
          </button>
          {catalogButton(key)}
          {testMsg[key] && <span className="text-xs text-zinc-400">{testMsg[key]}</span>}
        </div>
      </section>
    );
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6">
      <div className="flex max-h-[85vh] w-full max-w-2xl flex-col overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-950 shadow-2xl">
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3.5">
          <h2 className="text-base font-semibold text-zinc-100">Paramètres</h2>
          <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300">
            <X size={18} />
          </button>
        </div>

        <div className="flex-1 space-y-4 overflow-y-auto p-5">
          {/* Embedding */}
          <section className="space-y-3 rounded-xl border border-zinc-800 bg-zinc-900/30 p-4">
            <h3 className="text-sm font-semibold text-zinc-200">Embedding (indexation)</h3>
            <div className="grid grid-cols-2 gap-3">
              <Field label="Mode">
                <select
                  className={inputCls}
                  value={cfg.embedding.mode}
                  onChange={(e) =>
                    setCfg({
                      ...cfg,
                      embedding: { ...cfg.embedding, mode: e.target.value as "local" | "openai" },
                    })
                  }
                >
                  <option value="local">Local (fastembed / ONNX)</option>
                  <option value="openai">Serveur HTTP (OpenAI-compat)</option>
                </select>
              </Field>
              {cfg.embedding.mode === "local" ? (
                <Field label="Modèle local (téléchargés)">
                  {(() => {
                    // TOUS les modèles supportés sont proposés, pas seulement ceux déjà
                    // téléchargés : sur une installation neuve un seul l'est, et filtrer
                    // rendait le menu inchangeable. Un modèle absent est simplement
                    // récupéré au premier usage — c'est annoncé dans l'étiquette.
                    const opts = localModels.map((m) => m.id);
                    if (!opts.includes(cfg.embedding.model)) opts.unshift(cfg.embedding.model);
                    const noneReady = localModels.every((m) => !m.downloaded);
                    return (
                      <>
                        <select
                          className={inputCls}
                          value={cfg.embedding.model}
                          onChange={(e) => {
                            const v = e.target.value;
                            const dims =
                              localModels.find((m) => m.id === v)?.dimensions ??
                              findEmbed(v)?.dims ??
                              cfg.embedding.dimensions;
                            setCfg({
                              ...cfg,
                              embedding: { ...cfg.embedding, model: v, dimensions: dims },
                            });
                          }}
                        >
                          {opts.map((id) => {
                            const lm = localModels.find((m) => m.id === id);
                            return (
                              <option key={id} value={id}>
                                {id}
                                {lm ? ` · ${lm.dimensions}d` : ""}
                                {lm ? (lm.multilingual ? " · multilingue" : " · anglais") : ""}
                                {lm && !lm.downloaded ? " · à télécharger" : ""}
                              </option>
                            );
                          })}
                        </select>
                        {noneReady && (
                          <span className="mt-1 block text-[11px] text-amber-400">
                            Aucun modèle local téléchargé — celui que tu choisis sera récupéré
                            automatiquement à la première indexation.
                          </span>
                        )}
                        {(() => {
                          // Le piège invisible : un modèle anglophone sur un corpus
                          // français donne une recherche sémantique médiocre, et seule
                          // une réindexation complète permet d'en sortir.
                          const lm = localModels.find((m) => m.id === cfg.embedding.model);
                          if (!lm || lm.multilingual) return null;
                          return (
                            <span className="mt-1 block text-[11px] text-amber-400">
                              Modèle entraîné sur l'anglais : la recherche se dégradera nettement
                              sur des fichiers en français. Les modèles «&nbsp;multilingual-e5-*&nbsp;»
                              sont les seuls multilingues embarqués.
                            </span>
                          );
                        })()}
                      </>
                    );
                  })()}
                  {(() => {
                    const m = findEmbed(cfg.embedding.model);
                    return (
                      <span className="mt-1 block text-[11px] text-zinc-500">
                        Type : {m ? capLabel(m) : "local (personnalisé)"}
                      </span>
                    );
                  })()}
                </Field>
              ) : (
                <Field label="Modèle distant (installés sur le serveur)">
                  <div className="flex items-start gap-1.5">
                    <div className="flex-1">
                      <ModelSelect
                        value={cfg.embedding.model}
                        placeholder={
                          serverKind(cfg.embedding.base_url) === "lmstudio"
                            ? "ex : text-embedding-nomic-embed-text-v1.5…"
                            : "ex : bge-m3, nomic-embed-text…"
                        }
                        // Uniquement ce qui est réellement présent sur le serveur.
                        options={installed[cfg.embedding.base_url] ?? []}
                        onPick={(v) => {
                          const dims = findEmbed(v)?.dims ?? cfg.embedding.dimensions;
                          setCfg({
                            ...cfg,
                            embedding: { ...cfg.embedding, model: v, dimensions: dims },
                          });
                        }}
                      />
                    </div>
                    <button
                      onClick={() =>
                        downloadModel(
                          "embedding",
                          cfg.embedding.base_url,
                          cfg.embedding.api_key,
                          cfg.embedding.model
                        )
                      }
                      disabled={pulling.embedding}
                      title={
                        serverKind(cfg.embedding.base_url) === "lmstudio"
                          ? "Télécharger ce modèle (LM Studio, via le CLI lms)"
                          : "Télécharger ce modèle d'embedding (Ollama)"
                      }
                      className="flex h-[34px] shrink-0 items-center rounded-lg bg-zinc-800 px-2.5 text-zinc-200 hover:bg-zinc-700 disabled:opacity-50"
                    >
                      {pulling.embedding ? (
                        <Loader2 size={14} className="animate-spin" />
                      ) : (
                        <Download size={14} />
                      )}
                    </button>
                  </div>
                  {(() => {
                    const cap = capOf(cfg.embedding.model);
                    const list = installed[cfg.embedding.base_url] ?? [];
                    const avail = list.some(
                      (m) =>
                        m === cfg.embedding.model ||
                        m.split(":")[0] === cfg.embedding.model.split(":")[0]
                    );
                    return (
                      <>
                        <span className="mt-1 block text-[11px] text-zinc-500">
                          Type : {cap ?? "serveur (personnalisé)"}
                        </span>
                        <span
                          className={`block text-[11px] ${avail ? "text-emerald-400" : "text-amber-400"}`}
                        >
                          {avail
                            ? "● disponible sur le serveur"
                            : "○ introuvable — télécharger ou vérifier le nom"}
                        </span>
                      </>
                    );
                  })()}
                  {pulling.embedding && (
                    <div className="mt-1.5">
                      <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
                        <div
                          className="h-full rounded-full bg-blue-500 transition-all"
                          style={{ width: `${pullProgress[cfg.embedding.model]?.percent ?? 0}%` }}
                        />
                      </div>
                      <span className="mt-0.5 block text-[10px] text-zinc-500">
                        {pullProgress[cfg.embedding.model]?.status ?? "démarrage…"}{" "}
                        {pullProgress[cfg.embedding.model]?.percent ?? 0}%
                      </span>
                    </div>
                  )}
                </Field>
              )}
            </div>

            <p className="text-[11px] text-zinc-500">
              {cfg.embedding.mode === "local"
                ? "Local : modèle ONNX embarqué exécuté sur cette machine (CPU, ou GPU NVIDIA si coché). Aucune donnée ne sort de l'ordinateur."
                : "Serveur HTTP : délègue l'embedding à Ollama, LM Studio ou une machine distante. Mode « hybride » utile pour indexer sur un GPU dédié tout en gardant le reste local."}
            </p>

            {cfg.embedding.mode === "openai" && (
              <div className="grid grid-cols-2 gap-3">
                <Field label="URL du serveur">
                  <input
                    className={inputCls}
                    value={cfg.embedding.base_url}
                    onChange={(e) =>
                      setCfg({ ...cfg, embedding: { ...cfg.embedding, base_url: e.target.value } })
                    }
                  />
                  <div className="mt-1 flex gap-1.5">
                    {SERVER_PRESETS.map((s) => (
                      <button
                        key={s.label}
                        onClick={() =>
                          setCfg({
                            ...cfg,
                            embedding: { ...cfg.embedding, base_url: s.url },
                          })
                        }
                        className={`rounded-md px-2 py-0.5 text-[10px] ${
                          cfg.embedding.base_url === s.url
                            ? "bg-blue-500/20 text-blue-300"
                            : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
                        }`}
                      >
                        {s.label}
                      </button>
                    ))}
                  </div>
                </Field>
                <Field label="Dimensions">
                  <input
                    className={inputCls}
                    type="number"
                    value={cfg.embedding.dimensions}
                    onChange={(e) =>
                      setCfg({
                        ...cfg,
                        embedding: { ...cfg.embedding, dimensions: Number(e.target.value) },
                      })
                    }
                  />
                </Field>
              </div>
            )}

            <div className="flex items-center gap-2">
              {cfg.embedding.mode === "openai" && (
                <button
                  onClick={testEmbedding}
                  className="flex items-center gap-1.5 rounded-lg bg-zinc-800 px-3 py-1.5 text-xs text-zinc-200 hover:bg-zinc-700"
                >
                  <Plug size={13} /> Tester l'embedding
                </button>
              )}
              {catalogButton("embedding")}
              {testMsg.embedding && (
                <span className="text-xs text-zinc-400">{testMsg.embedding}</span>
              )}
            </div>

            {cfg.embedding.mode === "local" && (
              <div className="space-y-1">
                <label
                  className={`flex items-center gap-2 text-xs ${
                    gpuSupported ? "text-zinc-400" : "cursor-not-allowed text-zinc-600"
                  }`}
                  title={
                    gpuSupported
                      ? "Exécute l'embedding sur le GPU (repli CPU si indisponible)"
                      : "Binaire compilé sans support GPU"
                  }
                >
                  <input
                    type="checkbox"
                    disabled={!gpuSupported}
                    checked={cfg.embedding.use_gpu && gpuSupported}
                    onChange={(e) =>
                      setCfg({ ...cfg, embedding: { ...cfg.embedding, use_gpu: e.target.checked } })
                    }
                  />
                  Utiliser le GPU (CUDA)
                </label>
                {!gpuSupported ? (
                  <p className="text-[11px] text-zinc-500">
                    Aucun GPU NVIDIA détecté : indexation sur CPU. (Ou déléguez l'indexation à un
                    serveur distant via le mode « Serveur HTTP ».)
                  </p>
                ) : (
                  cfg.embedding.use_gpu && (
                    <p className="text-[11px] text-zinc-500">
                      Le runtime GPU (~340 Mo) est téléchargé au prochain démarrage ; nécessite
                      CUDA 12 + cuDNN 9 installés.
                    </p>
                  )
                )}
              </div>
            )}
            <p className="rounded-md bg-amber-500/10 px-2 py-1 text-[11px] text-amber-400/90">
              Changer de modèle/dimensions nécessite une réindexation complète.
            </p>
          </section>

          {chatSection("reasoning", "Reasoning / Chat")}
          {chatSection("vision", "Vision (multimodal)")}

          {/* Ordonnancement du pipeline : séquentiel vs par tranches */}
          <section className="space-y-3 rounded-xl border border-zinc-800 bg-zinc-900/30 p-4">
            <h3 className="text-sm font-semibold text-zinc-200">Ordonnancement de l'indexation</h3>
            <Field label="Mode">
              <select
                className={inputCls}
                value={cfg.indexing.pipeline_mode ?? "sequential"}
                onChange={(e) =>
                  setCfg({
                    ...cfg,
                    indexing: {
                      ...cfg.indexing,
                      pipeline_mode: e.target.value as "sequential" | "batch",
                    },
                  })
                }
              >
                <option value="sequential">Séquentiel — fichier par fichier</option>
                <option value="batch">Par tranches — tout un étage à la fois</option>
              </select>
            </Field>
            <p className="text-[11px] text-zinc-500">
              {cfg.indexing.pipeline_mode === "batch"
                ? "Chaque tranche passe entièrement par la vision et le reasoning, puis par l'embedding. Les modèles ne sont échangés qu'une fois par tranche — le bon choix si ta carte ne peut pas les garder tous en mémoire. En contrepartie, les fichiers d'une tranche ne deviennent cherchables qu'à la fin de celle-ci."
                : "Chaque fichier est mené de bout en bout avant le suivant : l'index avance en continu et ce qui vient d'être traité est immédiatement cherchable. Suppose que le serveur puisse garder les modèles chargés en même temps — ou que l'embedding soit embarqué (fastembed), auquel cas il n'y a aucun échange à faire."}
            </p>
            {cfg.indexing.pipeline_mode === "batch" && (
              <Field label="Fichiers par tranche">
                <input
                  type="number"
                  min={1}
                  max={1000}
                  className={inputCls}
                  value={cfg.indexing.batch_files ?? 64}
                  onChange={(e) =>
                    setCfg({
                      ...cfg,
                      indexing: { ...cfg.indexing, batch_files: Number(e.target.value) },
                    })
                  }
                />
                <span className="mt-1 block text-[11px] text-zinc-500">
                  Plus la tranche est grande, moins on échange de modèles — mais plus il faut
                  attendre avant que l'index avance.
                </span>
              </Field>
            )}
          </section>

          {/* Classification des dossiers : tendance bloc vs récursif */}
          <section className="space-y-3 rounded-xl border border-zinc-800 bg-zinc-900/30 p-4">
            <h3 className="text-sm font-semibold text-zinc-200">Classification des dossiers</h3>
            <Field
              label={`Tendance bloc / récursif — ${biasLabel(cfg.indexing.block_bias ?? 0.5)}`}
            >
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={cfg.indexing.block_bias ?? 0.5}
                onChange={(e) =>
                  setCfg({
                    ...cfg,
                    indexing: { ...cfg.indexing, block_bias: Number(e.target.value) },
                  })
                }
                className="w-full accent-blue-500"
              />
              <div className="flex justify-between text-[10px] text-zinc-500">
                <span>Explorer au max (récursif)</span>
                <span>Regrouper au max (bloc)</span>
              </div>
            </Field>
            <p className="text-[11px] text-zinc-500">
              Plus la tendance penche vers « bloc », plus SenseTree regroupe agressivement les
              dossiers techniques ou opaques (dépendances, installations d'outils comme Ghidra,
              packs d'instruments) au lieu de les indexer fichier par fichier. Prend effet sur les
              dossiers classés ensuite (ou après une réindexation).
            </p>
          </section>

          {/* Qualification IA du « sens » — pilote le coût reasoning par type de contenu */}
          <section className="space-y-3 rounded-xl border border-zinc-800 bg-zinc-900/30 p-4">
            <h3 className="text-sm font-semibold text-zinc-200">Qualification du sens (IA)</h3>
            <p className="text-[11px] text-zinc-500">
              Fait décrire par le modèle de reasoning CE QU'EST chaque fichier (« c'est une carte
              d'identité… »). Désactiver un type accélère fortement l'indexation (moins d'appels au
              serveur IA) — le sens retombe alors sur un simple extrait, que tu peux qualifier
              ensuite <strong>à la demande</strong> depuis le panneau de détail d'un fichier.
            </p>
            {(
              [
                ["qualify_documents", "Documents (PDF, Word, texte, code)"],
                ["qualify_images", "Images (en plus de la légende vision)"],
                ["qualify_context", "Fichiers illisibles (devinette par contexte)"],
              ] as const
            ).map(([key, label]) => (
              <label key={key} className="flex items-center gap-2 text-sm text-zinc-300">
                <input
                  type="checkbox"
                  checked={cfg.indexing[key] ?? true}
                  onChange={(e) =>
                    setCfg({ ...cfg, indexing: { ...cfg.indexing, [key]: e.target.checked } })
                  }
                  className="accent-blue-500"
                />
                {label}
              </label>
            ))}
          </section>

          {/* Recherche (RAG moderne) — hybride + reranking cross-encoder */}
          <section className="space-y-3 rounded-xl border border-zinc-800 bg-zinc-900/30 p-4">
            <h3 className="text-sm font-semibold text-zinc-200">Recherche (RAG)</h3>
            <p className="text-[11px] text-zinc-500">
              <strong>Hybride</strong> : combine le sens (vecteurs) et les mots-clés exacts (BM25)
              — on trouve à la fois par similarité ET par terme précis (noms propres, codes,
              extensions). <strong>Reranking</strong> : un cross-encoder réordonne les meilleurs
              candidats pour une précision nettement supérieure (léger surcoût au 1<sup>er</sup> usage,
              le temps de charger le modèle en local).
            </p>
            <label className="flex items-center gap-2 text-sm text-zinc-300">
              <input
                type="checkbox"
                checked={cfg.retrieval.hybrid}
                onChange={(e) =>
                  setCfg({ ...cfg, retrieval: { ...cfg.retrieval, hybrid: e.target.checked } })
                }
                className="accent-blue-500"
              />
              Recherche hybride (dense + mots-clés)
            </label>
            <label className="flex items-center gap-2 text-sm text-zinc-300">
              <input
                type="checkbox"
                checked={cfg.retrieval.rerank}
                onChange={(e) =>
                  setCfg({ ...cfg, retrieval: { ...cfg.retrieval, rerank: e.target.checked } })
                }
                className="accent-blue-500"
              />
              Reranking cross-encoder
            </label>
            {cfg.retrieval.rerank && (
              <label className="flex items-center gap-2 text-xs text-zinc-400">
                <span className="w-32 shrink-0">Modèle de reranking</span>
                <select
                  value={cfg.retrieval.reranker_model}
                  onChange={(e) =>
                    setCfg({
                      ...cfg,
                      retrieval: { ...cfg.retrieval, reranker_model: e.target.value },
                    })
                  }
                  className="flex-1 rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1 text-zinc-200"
                >
                  <option value="bge-reranker-v2-m3">bge-reranker-v2-m3 (multilingue, recommandé)</option>
                  <option value="bge-reranker-base">bge-reranker-base (léger, anglais)</option>
                  <option value="jina-reranker-v2-base-multilingual">
                    jina-reranker-v2 (multilingue)
                  </option>
                </select>
              </label>
            )}
          </section>

          {/* Serveurs MCP — outils externes pour l'agent du chat */}
          <section className="space-y-3 rounded-xl border border-zinc-800 bg-zinc-900/30 p-4">
            <div className="flex items-center justify-between">
              <h3 className="flex items-center gap-2 text-sm font-semibold text-zinc-200">
                <Plug size={14} /> Serveurs MCP (outils externes)
              </h3>
              <button
                onClick={() =>
                  setCfg({
                    ...cfg,
                    mcp_servers: [
                      ...cfg.mcp_servers,
                      { name: "", url: "", auth: "", command: "", args: [], enabled: true },
                    ],
                  })
                }
                className="flex items-center gap-1 rounded px-2 py-1 text-[11px] text-zinc-300 hover:bg-zinc-800"
              >
                <Plus size={12} /> Ajouter
              </button>
            </div>
            <p className="text-[11px] text-zinc-500">
              Branche des serveurs <strong>MCP</strong> (Model Context Protocol) : leurs outils
              deviennent utilisables par l'agent du chat. Transport <strong>HTTP</strong> (URL) ou{" "}
              <strong>stdio</strong> (commande locale, ex. <code>npx</code>). Best-effort — un serveur
              injoignable est ignoré, l'agent garde ses outils intégrés.
            </p>
            {cfg.mcp_servers.length === 0 && (
              <p className="text-[11px] text-zinc-600">Aucun serveur configuré.</p>
            )}
            {cfg.mcp_servers.map((srv, i) => {
              const patch = (p: Partial<typeof srv>) => {
                const next = [...cfg.mcp_servers];
                next[i] = { ...srv, ...p };
                setCfg({ ...cfg, mcp_servers: next });
              };
              return (
                <div
                  key={i}
                  className="space-y-2 rounded-lg border border-zinc-800 bg-zinc-950/40 p-3"
                >
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={srv.enabled}
                      onChange={(e) => patch({ enabled: e.target.checked })}
                      className="accent-blue-500"
                      title="Activer ce serveur"
                    />
                    <input
                      value={srv.name}
                      placeholder="nom (ex. github)"
                      onChange={(e) => patch({ name: e.target.value })}
                      className="w-32 rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-200"
                    />
                    <span className="flex-1 text-[10px] uppercase tracking-widest text-zinc-600">
                      {srv.command.trim() ? "stdio" : "http"}
                    </span>
                    <button
                      onClick={() =>
                        setCfg({
                          ...cfg,
                          mcp_servers: cfg.mcp_servers.filter((_, j) => j !== i),
                        })
                      }
                      title="Retirer ce serveur"
                      className="shrink-0 rounded p-1 text-zinc-600 hover:bg-rose-500/15 hover:text-rose-400"
                    >
                      <X size={13} />
                    </button>
                  </div>
                  {/* Transport HTTP */}
                  <input
                    value={srv.url}
                    placeholder="URL HTTP (ex. https://…/mcp) — ou laisser vide pour stdio"
                    onChange={(e) => patch({ url: e.target.value })}
                    className="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-200"
                  />
                  {srv.url.trim() && (
                    <input
                      value={srv.auth}
                      placeholder="Authorization (optionnel, ex. Bearer xxx)"
                      onChange={(e) => patch({ auth: e.target.value })}
                      className="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-200"
                    />
                  )}
                  {/* Transport stdio */}
                  <div className="flex gap-2">
                    <input
                      value={srv.command}
                      placeholder="commande stdio (ex. npx)"
                      onChange={(e) => patch({ command: e.target.value })}
                      className="w-40 rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-200"
                    />
                    <input
                      value={srv.args.join(" ")}
                      placeholder="arguments (séparés par des espaces)"
                      onChange={(e) =>
                        patch({ args: e.target.value.split(/\s+/).filter(Boolean) })
                      }
                      className="min-w-0 flex-1 rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-200"
                    />
                  </div>
                </div>
              );
            })}
          </section>

          {/* Mémoire de l'agent — faits/préférences durables */}
          <section className="space-y-3 rounded-xl border border-zinc-800 bg-zinc-900/30 p-4">
            <div className="flex items-center justify-between">
              <h3 className="flex items-center gap-2 text-sm font-semibold text-zinc-200">
                <Brain size={14} /> Mémoire de l'agent
              </h3>
              {memories.length > 0 && (
                <button
                  onClick={() => agentMemoryClear().then(refreshMemories).catch(() => {})}
                  className="flex items-center gap-1 rounded px-2 py-1 text-[11px] text-rose-400 hover:bg-rose-500/10"
                >
                  <Trash2 size={12} /> Tout vider
                </button>
              )}
            </div>
            <p className="text-[11px] text-zinc-500">
              Faits et préférences que l'agent a retenus (il s'en sert dans chaque conversation).
              Il ajoute une note quand tu lui confies quelque chose de durable ; tu peux en retirer
              ici.
            </p>
            {memories.length === 0 ? (
              <p className="text-[11px] text-zinc-600">Aucun souvenir pour l'instant.</p>
            ) : (
              <ul className="space-y-1.5">
                {memories.map((m) => (
                  <li
                    key={m.id}
                    className="flex items-start gap-2 rounded-lg border border-zinc-800 bg-zinc-950/40 px-3 py-2 text-xs text-zinc-300"
                  >
                    <span className="min-w-0 flex-1">{m.note}</span>
                    <button
                      onClick={() => agentMemoryDelete(m.id).then(refreshMemories).catch(() => {})}
                      title="Oublier ce souvenir"
                      className="shrink-0 rounded p-0.5 text-zinc-600 hover:bg-rose-500/15 hover:text-rose-400"
                    >
                      <X size={12} />
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>

          {/* Prompts IA — édition avancée (repliable) */}
          <section className="rounded-xl border border-zinc-800 bg-zinc-900/30">
            <button
              onClick={() => setShowPrompts((v) => !v)}
              className="flex w-full items-center justify-between px-4 py-3 text-sm font-semibold text-zinc-200"
            >
              <span>Prompts IA (avancé)</span>
              {showPrompts ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
            </button>
            {showPrompts && (
              <div className="space-y-4 border-t border-zinc-800 px-4 py-4">
                <p className="text-[11px] text-zinc-500">
                  Personnalise les instructions envoyées au modèle pour chaque tâche. Laisse un
                  champ vide (ou clique « Défaut ») pour revenir au prompt intégré. Les données
                  (chemin, contenu, arborescence) sont ajoutées automatiquement.
                </p>
                {PROMPT_FIELDS.map(({ key, label, rows }) => {
                  const override = cfg.prompts?.[key] ?? "";
                  const isCustom = override.trim() !== "";
                  const shown = isCustom ? override : defaultPrompts?.[key] ?? "";
                  return (
                    <div key={key}>
                      <div className="mb-1 flex items-center justify-between">
                        <span className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
                          {label}
                        </span>
                        <span className="flex items-center gap-2">
                          <span
                            className={`text-[10px] ${isCustom ? "text-amber-400" : "text-zinc-600"}`}
                          >
                            {isCustom ? "● personnalisé" : "○ défaut"}
                          </span>
                          {isCustom && (
                            <button
                              onClick={() => patchPrompt(key, "")}
                              className="text-[10px] text-zinc-400 underline hover:text-zinc-200"
                            >
                              Défaut
                            </button>
                          )}
                        </span>
                      </div>
                      <textarea
                        rows={rows}
                        value={shown}
                        onChange={(e) => patchPrompt(key, e.target.value)}
                        className="w-full resize-y rounded-lg border border-zinc-800 bg-zinc-900 px-2.5 py-1.5 font-mono text-[11px] leading-relaxed text-zinc-200 outline-none focus:border-blue-500"
                      />
                    </div>
                  );
                })}
              </div>
            )}
          </section>
        </div>

        <div className="flex items-center justify-between gap-3 border-t border-zinc-800 px-5 py-3">
          <div className="flex items-center gap-2">
            <button
              onClick={reindex}
              disabled={reindexing}
              className="flex items-center gap-1.5 rounded-lg bg-zinc-800 px-3 py-2 text-xs text-zinc-300 hover:bg-zinc-700 disabled:opacity-50"
            >
              {reindexing ? (
                <Loader2 size={14} className="animate-spin" />
              ) : (
                <RotateCcw size={14} />
              )}
              Réindexer tout
            </button>
            {testMsg.save && <span className="text-xs text-zinc-500">{testMsg.save}</span>}
          </div>
          <button
            onClick={save}
            disabled={saving}
            className="flex items-center gap-1.5 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50"
          >
            {saving ? <Loader2 size={15} className="animate-spin" /> : <Save size={15} />}
            Enregistrer
          </button>
        </div>
      </div>

      {catalogTask && (
        <ModelCatalog
          open
          onClose={() => setCatalogTask(null)}
          task={catalogTask}
          backend={catalogBackend}
          serverKind={serverKind(catalogUrl)}
          servers={catalogServers}
          localDownloaded={localDownloaded}
          currentModel={catalogCurrent}
          onUse={catalogUse}
          onDownload={catalogDownload}
          downloading={dlBusy}
          progress={pullProgress}
        />
      )}
    </div>
  );
}

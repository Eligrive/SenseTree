import { useEffect, useState, type ReactNode } from "react";
import { listen } from "@tauri-apps/api/event";
import {
  ChevronDown,
  ChevronRight,
  Download,
  Loader2,
  Plug,
  RefreshCw,
  RotateCcw,
  Save,
  X,
} from "lucide-react";
import type { AppConfig, ChatConfig, PromptsConfig } from "../lib/types";
import {
  getConfig,
  getDefaultPrompts,
  gpuAvailable,
  listInstalledModels,
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

const LOCAL_MODELS: { id: string; dims: number }[] = [
  { id: "multilingual-e5-small", dims: 384 },
  { id: "multilingual-e5-base", dims: 768 },
  { id: "bge-small-en-v1.5", dims: 384 },
  // Aussi disponibles sur Ollama (même nom) → local CPU ou distant GPU sans changer de modèle.
  { id: "all-minilm", dims: 384 },
  { id: "nomic-embed-text", dims: 768 },
  { id: "mxbai-embed-large", dims: 1024 },
];

// Suggestions de modèles cohérents par rôle (téléchargeables via Ollama).
const SUGGESTED: Record<"reasoning" | "vision", string[]> = {
  reasoning: ["llama3.1:8b", "llama3.2:3b", "qwen2.5:7b", "phi3:mini"],
  vision: ["moondream", "llava", "llama3.2-vision"],
};

// Modèles d'embedding Ollama courants (le nom fastembed « multilingual-e5-small »
// n'existe PAS sur Ollama — d'où les 404).
const EMBED_SUGGESTED: { id: string; dims: number }[] = [
  { id: "bge-m3", dims: 1024 },
  { id: "nomic-embed-text", dims: 768 },
  { id: "mxbai-embed-large", dims: 1024 },
  { id: "snowflake-arctic-embed2", dims: 1024 },
  { id: "all-minilm", dims: 384 },
];

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

  useEffect(() => {
    if (open) {
      getConfig().then(setCfg).catch(() => setCfg(null));
      gpuAvailable().then(setGpuSupported).catch(() => setGpuSupported(false));
      getDefaultPrompts().then(setDefaultPrompts).catch(() => setDefaultPrompts(null));
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

  const download = async (key: "reasoning" | "vision") => {
    setPulling((p) => ({ ...p, [key]: true }));
    setTestMsg((m) => ({ ...m, [key]: `⏳ téléchargement de « ${cfg[key].model} »…` }));
    try {
      const res = await pullModel(cfg[key].base_url, cfg[key].model);
      setTestMsg((m) => ({ ...m, [key]: `✅ ${res}` }));
      refreshModels(cfg[key].base_url, cfg[key].api_key);
    } catch (e) {
      setTestMsg((m) => ({ ...m, [key]: `⚠️ ${String(e)}` }));
    } finally {
      setPulling((p) => ({ ...p, [key]: false }));
    }
  };

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
    const options = Array.from(
      new Set([...(installed[cfg[key].base_url] ?? []), ...SUGGESTED[key]])
    );
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
          </Field>
          <Field label="Modèle">
            <div className="flex gap-1.5">
              <input
                className={inputCls}
                list={`models-${key}`}
                value={cfg[key].model}
                onChange={(e) => patchChat(key, { model: e.target.value })}
              />
              <datalist id={`models-${key}`}>
                {options.map((m) => (
                  <option key={m} value={m} />
                ))}
              </datalist>
              <button
                onClick={() => download(key)}
                disabled={pulling[key]}
                title="Télécharger ce modèle (Ollama)"
                className="flex shrink-0 items-center rounded-lg bg-zinc-800 px-2.5 text-zinc-200 hover:bg-zinc-700 disabled:opacity-50"
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
              {isInstalled ? "● installé" : "○ non installé"}
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
                <Field label="Modèle local">
                  <select
                    className={inputCls}
                    value={cfg.embedding.model}
                    onChange={(e) => {
                      const preset = LOCAL_MODELS.find((m) => m.id === e.target.value);
                      setCfg({
                        ...cfg,
                        embedding: {
                          ...cfg.embedding,
                          model: e.target.value,
                          dimensions: preset?.dims ?? cfg.embedding.dimensions,
                        },
                      });
                    }}
                  >
                    {LOCAL_MODELS.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.id} ({m.dims}d)
                      </option>
                    ))}
                  </select>
                </Field>
              ) : (
                <Field label="Modèle distant">
                  <input
                    className={inputCls}
                    list="embed-models"
                    value={cfg.embedding.model}
                    placeholder="ex : bge-m3, nomic-embed-text…"
                    onChange={(e) => {
                      const preset = EMBED_SUGGESTED.find((m) => m.id === e.target.value);
                      setCfg({
                        ...cfg,
                        embedding: {
                          ...cfg.embedding,
                          model: e.target.value,
                          dimensions: preset?.dims ?? cfg.embedding.dimensions,
                        },
                      });
                    }}
                  />
                  <datalist id="embed-models">
                    {Array.from(
                      new Set([
                        ...(installed[cfg.embedding.base_url] ?? []),
                        ...EMBED_SUGGESTED.map((m) => m.id),
                      ])
                    ).map((m) => (
                      <option key={m} value={m} />
                    ))}
                  </datalist>
                  {(() => {
                    const list = installed[cfg.embedding.base_url] ?? [];
                    const avail = list.some(
                      (m) =>
                        m === cfg.embedding.model ||
                        m.split(":")[0] === cfg.embedding.model.split(":")[0]
                    );
                    return (
                      <span
                        className={`mt-1 block text-[11px] ${avail ? "text-emerald-400" : "text-amber-400"}`}
                      >
                        {avail
                          ? "● disponible sur le serveur"
                          : "○ introuvable — vérifiez le nom ou chargez le modèle"}
                      </span>
                    );
                  })()}
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
                    {[
                      { label: "Ollama", url: "http://localhost:11434/v1" },
                      { label: "LM Studio", url: "http://localhost:1234/v1" },
                    ].map((s) => (
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

            {cfg.embedding.mode === "openai" && (
              <div className="flex items-center gap-2">
                <button
                  onClick={testEmbedding}
                  className="flex items-center gap-1.5 rounded-lg bg-zinc-800 px-3 py-1.5 text-xs text-zinc-200 hover:bg-zinc-700"
                >
                  <Plug size={13} /> Tester l'embedding
                </button>
                {testMsg.embedding && <span className="text-xs text-zinc-400">{testMsg.embedding}</span>}
              </div>
            )}

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
    </div>
  );
}

import { useEffect, useState, type ReactNode } from "react";
import { Loader2, Plug, Save, X } from "lucide-react";
import type { AppConfig, ChatConfig } from "../lib/types";
import { getConfig, setConfig, testChatEndpoint } from "../lib/ipc";

interface Props {
  open: boolean;
  onClose: () => void;
  onSaved: () => void;
}

const LOCAL_MODELS: { id: string; dims: number }[] = [
  { id: "multilingual-e5-small", dims: 384 },
  { id: "multilingual-e5-base", dims: 768 },
  { id: "bge-small-en-v1.5", dims: 384 },
  { id: "bge-base-en-v1.5", dims: 768 },
  { id: "all-minilm-l6-v2", dims: 384 },
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

export default function SettingsModal({ open, onClose, onSaved }: Props) {
  const [cfg, setCfg] = useState<AppConfig | null>(null);
  const [saving, setSaving] = useState(false);
  const [testMsg, setTestMsg] = useState<Record<string, string>>({});

  useEffect(() => {
    if (open) getConfig().then(setCfg).catch(() => setCfg(null));
  }, [open]);

  if (!open || !cfg) return null;

  const patchChat = (key: "reasoning" | "vision", patch: Partial<ChatConfig>) =>
    setCfg({ ...cfg, [key]: { ...cfg[key], ...patch } });

  const test = async (key: "reasoning" | "vision") => {
    setTestMsg((m) => ({ ...m, [key]: "…" }));
    try {
      const res = await testChatEndpoint(cfg[key].base_url, cfg[key].api_key);
      setTestMsg((m) => ({ ...m, [key]: `✅ ${res}` }));
    } catch (e) {
      setTestMsg((m) => ({ ...m, [key]: `⚠️ ${String(e)}` }));
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

  const chatSection = (key: "reasoning" | "vision", title: string) => (
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
          <input
            className={inputCls}
            value={cfg[key].model}
            onChange={(e) => patchChat(key, { model: e.target.value })}
          />
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
          <Plug size={13} /> Tester la connexion
        </button>
        {testMsg[key] && <span className="text-xs text-zinc-400">{testMsg[key]}</span>}
      </div>
    </section>
  );

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
                    value={cfg.embedding.model}
                    onChange={(e) =>
                      setCfg({ ...cfg, embedding: { ...cfg.embedding, model: e.target.value } })
                    }
                  />
                </Field>
              )}
            </div>

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

            {cfg.embedding.mode === "local" && (
              <label className="flex items-center gap-2 text-xs text-zinc-400">
                <input
                  type="checkbox"
                  checked={cfg.embedding.use_gpu}
                  onChange={(e) =>
                    setCfg({ ...cfg, embedding: { ...cfg.embedding, use_gpu: e.target.checked } })
                  }
                />
                Utiliser le GPU si disponible (build CUDA requis)
              </label>
            )}
            <p className="rounded-md bg-amber-500/10 px-2 py-1 text-[11px] text-amber-400/90">
              Changer de modèle/dimensions nécessite une ré-indexation complète.
            </p>
          </section>

          {chatSection("reasoning", "Reasoning / Chat")}
          {chatSection("vision", "Vision (multimodal)")}
        </div>

        <div className="flex items-center justify-between border-t border-zinc-800 px-5 py-3">
          <span className="text-xs text-zinc-500">{testMsg.save}</span>
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

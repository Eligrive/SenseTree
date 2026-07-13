import { useMemo, useState } from "react";
import { Check, Download, Loader2, Search, Star, X } from "lucide-react";
import {
  CATALOG,
  availabilityOf,
  idForBackend,
  type Backend,
  type CatalogModel,
  type ServerKind,
  type Task,
} from "../lib/models";

interface Props {
  open: boolean;
  onClose: () => void;
  /// Tâche concernée (une section des Paramètres = une tâche).
  task: Task;
  /// Backend visé : « local » (embedding embarqué) ou « server » (Ollama / LM Studio).
  backend: Backend;
  serverKind: ServerKind;
  /// Modèles présents sur le serveur (issus de /v1/models).
  installedServer: string[];
  /// Modèles locaux déjà téléchargés (id → true).
  localDownloaded: Record<string, boolean>;
  /// Modèle actuellement configuré (pour le marquer « utilisé »).
  currentModel: string;
  onUse: (id: string, dims?: number) => void;
  onDownload: (id: string) => void;
  /// Téléchargements en cours (id → true).
  downloading: Record<string, boolean>;
}

const TASK_LABEL: Record<Task, string> = {
  embedding: "Embedding (indexation)",
  reasoning: "Reasoning / Chat",
  vision: "Vision",
};

function Stars({ n }: { n: number }) {
  return (
    <span className="flex shrink-0 items-center gap-0.5" title={`${n}/5 (indicatif)`}>
      {[1, 2, 3, 4, 5].map((i) => (
        <Star
          key={i}
          size={11}
          className={i <= n ? "fill-amber-400 text-amber-400" : "text-zinc-700"}
        />
      ))}
    </span>
  );
}

function Badge({ label, active }: { label: string; active: boolean }) {
  return (
    <span
      className={`rounded px-1.5 py-0.5 text-[10px] ${
        active ? "bg-blue-500/20 text-blue-300" : "bg-zinc-800 text-zinc-500"
      }`}
    >
      {label}
    </span>
  );
}

export default function ModelCatalog({
  open,
  onClose,
  task,
  backend,
  serverKind,
  installedServer,
  localDownloaded,
  currentModel,
  onUse,
  onDownload,
  downloading,
}: Props) {
  const [query, setQuery] = useState("");
  const [onlyAvailable, setOnlyAvailable] = useState(false);

  // Le modèle est-il installé pour le backend courant ?
  const isInstalled = (id: string) =>
    backend === "local"
      ? !!localDownloaded[id]
      : installedServer.some((m) => m === id || m.split(":")[0] === id.split(":")[0]);

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return CATALOG.filter((m) => m.task === task)
      .filter((m) => (q ? (m.name + m.goodFor).toLowerCase().includes(q) : true))
      .filter((m) => {
        if (!onlyAvailable) return true;
        const id = idForBackend(m, backend, serverKind);
        return !!id && isInstalled(id);
      })
      .sort((a, b) => b.quality - a.quality || a.name.localeCompare(b.name));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task, backend, serverKind, query, onlyAvailable, installedServer, localDownloaded]);

  if (!open) return null;

  const backendLabel =
    backend === "local"
      ? "Local (embarqué)"
      : serverKind === "lmstudio"
        ? "Serveur LM Studio"
        : serverKind === "ollama"
          ? "Serveur Ollama"
          : "Serveur HTTP";

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/70 p-6">
      <div className="flex max-h-[88vh] w-full max-w-4xl flex-col overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-950 shadow-2xl">
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3.5">
          <div>
            <h2 className="text-base font-semibold text-zinc-100">
              Catalogue de modèles — {TASK_LABEL[task]}
            </h2>
            <p className="text-[11px] text-zinc-500">
              Cible actuelle : <span className="text-zinc-300">{backendLabel}</span>
            </p>
          </div>
          <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300">
            <X size={18} />
          </button>
        </div>

        {/* Filtres */}
        <div className="flex items-center gap-3 border-b border-zinc-800 px-5 py-2.5">
          <div className="flex flex-1 items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900 px-2.5 py-1.5">
            <Search size={13} className="shrink-0 text-zinc-500" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Rechercher un modèle…"
              className="w-full bg-transparent text-sm text-zinc-200 outline-none placeholder:text-zinc-600"
            />
          </div>
          <label className="flex shrink-0 items-center gap-1.5 text-xs text-zinc-400">
            <input
              type="checkbox"
              checked={onlyAvailable}
              onChange={(e) => setOnlyAvailable(e.target.checked)}
            />
            Installés seulement
          </label>
        </div>

        {/* Liste */}
        <div className="flex-1 space-y-2 overflow-y-auto p-4">
          {rows.length === 0 && (
            <p className="py-8 text-center text-sm text-zinc-600">Aucun modèle ne correspond.</p>
          )}
          {rows.map((m: CatalogModel) => {
            const id = idForBackend(m, backend, serverKind);
            const usable = !!id;
            const installed = usable && isInstalled(id);
            const inUse = usable && currentModel === id;
            const busy = usable && !!downloading[id];

            return (
              <div
                key={m.key}
                className={`rounded-xl border p-3 ${
                  inUse
                    ? "border-blue-500/40 bg-blue-500/5"
                    : usable
                      ? "border-zinc-800 bg-zinc-900/30"
                      : "border-zinc-900 bg-zinc-900/10 opacity-60"
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-sm font-semibold text-zinc-100">{m.name}</span>
                      <Stars n={m.quality} />
                      <span
                        className={`rounded px-1.5 py-0.5 text-[10px] ${
                          m.languages === "multilingue"
                            ? "bg-emerald-500/15 text-emerald-300"
                            : "bg-zinc-800 text-zinc-400"
                        }`}
                      >
                        {m.languages}
                      </span>
                      {inUse && (
                        <span className="rounded bg-blue-500/20 px-1.5 py-0.5 text-[10px] text-blue-300">
                          utilisé
                        </span>
                      )}
                    </div>

                    <p className="mt-1 text-[12px] text-zinc-300">{m.goodFor}</p>
                    <p className="mt-0.5 text-[11px] text-zinc-500">📊 {m.benchmark}</p>

                    <div className="mt-1.5 flex flex-wrap items-center gap-3 text-[11px] text-zinc-500">
                      <span>{m.params}</span>
                      {m.dims && <span>{m.dims} dims</span>}
                      <span className="flex items-center gap-1">
                        {availabilityOf(m).map((b) => (
                          <Badge
                            key={b}
                            label={b}
                            active={
                              (b === "Local" && backend === "local") ||
                              (b === "Ollama" && backend === "server" && serverKind !== "lmstudio") ||
                              (b === "LM Studio" && backend === "server" && serverKind === "lmstudio")
                            }
                          />
                        ))}
                      </span>
                    </div>

                    {!usable && (
                      <p className="mt-1.5 text-[11px] text-amber-400/80">
                        Indisponible sur « {backendLabel} » — proposé sur :{" "}
                        {availabilityOf(m).join(", ")}.
                      </p>
                    )}
                    {usable && (
                      <p className="mt-1.5 font-mono text-[10px] text-zinc-600">{id}</p>
                    )}
                  </div>

                  {/* Actions */}
                  {usable && (
                    <div className="flex shrink-0 flex-col gap-1.5">
                      <button
                        onClick={() => onUse(id, m.dims)}
                        disabled={inUse}
                        className="rounded-lg bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-500 disabled:opacity-40"
                      >
                        {inUse ? "Utilisé" : "Utiliser"}
                      </button>
                      {installed ? (
                        <span className="flex items-center justify-center gap-1 text-[11px] text-emerald-400">
                          <Check size={12} /> installé
                        </span>
                      ) : (
                        <button
                          onClick={() => onDownload(id)}
                          disabled={busy}
                          className="flex items-center justify-center gap-1 rounded-lg bg-zinc-800 px-3 py-1.5 text-xs text-zinc-200 hover:bg-zinc-700 disabled:opacity-50"
                        >
                          {busy ? (
                            <Loader2 size={12} className="animate-spin" />
                          ) : (
                            <Download size={12} />
                          )}
                          Télécharger
                        </button>
                      )}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        <div className="border-t border-zinc-800 px-5 py-2.5">
          <p className="text-[11px] text-zinc-600">
            Les notes ★ sont <strong>indicatives</strong> : elles reflètent le classement sur les
            benchmarks de référence (MTEB multilingue pour les embeddings, MMLU / LMArena pour les
            LLM, MMMU pour la vision). Les scores exacts évoluent — consulte les leaderboards
            officiels pour le détail. Changer de modèle d'embedding impose une réindexation.
          </p>
        </div>
      </div>
    </div>
  );
}

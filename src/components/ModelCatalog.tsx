import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, Check, Download, Loader2, RefreshCw, Search, Star, X } from "lucide-react";
import {
  CATALOG,
  availabilityOf,
  idForBackend,
  type Backend,
  type CatalogModel,
  type ServerKind,
  type Task,
} from "../lib/models";
import type { ModelBenchmark } from "../lib/types";
import { modelBenchmarks } from "../lib/ipc";

interface Props {
  open: boolean;
  onClose: () => void;
  task: Task;
  backend: Backend;
  serverKind: ServerKind;
  installedServer: string[];
  localDownloaded: Record<string, boolean>;
  currentModel: string;
  onUse: (id: string, dims?: number) => void;
  onDownload: (id: string) => void;
  downloading: Record<string, boolean>;
}

const TASK_LABEL: Record<Task, string> = {
  embedding: "Embedding (indexation)",
  reasoning: "Reasoning / Chat",
  vision: "Vision",
};

type SortBy = "relevance" | "score";

/// ndcg@10 (0–1) → convention MTEB (×100).
function fmtScore(x: number): string {
  return (x * 100).toFixed(1);
}

function fmtParams(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1).replace(".", ",")} B`;
  return `${Math.round(n / 1e6)} M`;
}

function Stars({ n }: { n: number }) {
  return (
    <span className="flex shrink-0 items-center gap-0.5" title={`${n}/5 — avis curaté`}>
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
  const [sortBy, setSortBy] = useState<SortBy>("relevance");
  const [bench, setBench] = useState<Record<string, ModelBenchmark>>({});
  const [loadingBench, setLoadingBench] = useState(false);
  const [benchError, setBenchError] = useState<string | null>(null);

  const mtebIds = useMemo(
    () => CATALOG.filter((m) => m.task === task && m.mteb).map((m) => m.mteb as string),
    [task]
  );

  const fetchBench = (refresh: boolean) => {
    if (mtebIds.length === 0) return;
    setLoadingBench(true);
    setBenchError(null);
    modelBenchmarks(mtebIds, refresh)
      .then((list) => setBench(Object.fromEntries(list.map((b) => [b.mteb_id, b]))))
      .catch((e) => setBenchError(String(e)))
      .finally(() => setLoadingBench(false));
  };

  useEffect(() => {
    if (open) fetchBench(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, task]);

  const isInstalled = (id: string) =>
    backend === "local"
      ? !!localDownloaded[id]
      : installedServer.some((m) => m === id || m.split(":")[0] === id.split(":")[0]);

  const scoreOf = (m: CatalogModel): number | null =>
    m.mteb ? (bench[m.mteb]?.retrieval_en ?? null) : null;

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return CATALOG.filter((m) => m.task === task)
      .filter((m) => (q ? (m.name + m.goodFor).toLowerCase().includes(q) : true))
      .filter((m) => {
        if (!onlyAvailable) return true;
        const id = idForBackend(m, backend, serverKind);
        return !!id && isInstalled(id);
      })
      .sort((a, b) => {
        if (sortBy === "score") {
          const sa = scoreOf(a);
          const sb = scoreOf(b);
          if (sa != null && sb != null) return sb - sa;
          if (sa != null) return -1;
          if (sb != null) return 1;
        }
        return b.quality - a.quality || a.name.localeCompare(b.name);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task, backend, serverKind, query, onlyAvailable, sortBy, bench, installedServer, localDownloaded]);

  if (!open) return null;

  const backendLabel =
    backend === "local"
      ? "Local (embarqué)"
      : serverKind === "lmstudio"
        ? "Serveur LM Studio"
        : serverKind === "ollama"
          ? "Serveur Ollama"
          : "Serveur HTTP";

  const hasScores = task === "embedding";

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
          <div className="flex items-center gap-2">
            {hasScores && (
              <button
                onClick={() => fetchBench(true)}
                disabled={loadingBench}
                title="Rafraîchir les specs et scores depuis MTEB"
                className="flex items-center gap-1.5 rounded-lg bg-zinc-800 px-2.5 py-1.5 text-xs text-zinc-200 hover:bg-zinc-700 disabled:opacity-50"
              >
                {loadingBench ? (
                  <Loader2 size={13} className="animate-spin" />
                ) : (
                  <RefreshCw size={13} />
                )}
                MTEB
              </button>
            )}
            <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300">
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Avertissement : le score mesuré est ANGLAIS. */}
        {hasScores && (
          <div className="flex gap-2 border-b border-zinc-800 bg-amber-500/5 px-5 py-2.5">
            <AlertTriangle size={14} className="mt-0.5 shrink-0 text-amber-400" />
            <p className="text-[11px] leading-relaxed text-amber-200/80">
              Le score affiché est mesuré <strong>en anglais</strong> (moyenne ndcg@10 sur
              NFCorpus, SciFact, ArguAna, SCIDOCS — les seules tâches réellement comparables
              entre ces modèles). Il <strong>ne prédit pas</strong> la performance sur un corpus
              français : un modèle « anglais uniquement » peut y figurer en tête tout en étant
              mauvais sur tes fichiers FR. Fie-toi aussi aux <strong>langues déclarées</strong>.
            </p>
          </div>
        )}

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
          {hasScores && (
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as SortBy)}
              className="shrink-0 rounded-lg border border-zinc-800 bg-zinc-900 px-2 py-1.5 text-xs text-zinc-200 outline-none"
              title="Le tri par score classe sur l'anglais uniquement"
            >
              <option value="relevance">Tri : pertinence (FR)</option>
              <option value="score">Tri : score retrieval EN</option>
            </select>
          )}
          <label className="flex shrink-0 items-center gap-1.5 text-xs text-zinc-400">
            <input
              type="checkbox"
              checked={onlyAvailable}
              onChange={(e) => setOnlyAvailable(e.target.checked)}
            />
            Installés seulement
          </label>
        </div>

        {benchError && (
          <p className="border-b border-zinc-800 px-5 py-2 text-[11px] text-rose-400">
            Specs/scores MTEB indisponibles ({benchError}) — affichage des valeurs de repli.
          </p>
        )}

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
            const b = m.mteb ? bench[m.mteb] : undefined;

            // Specs live si disponibles, sinon repli sur le catalogue.
            const dims = b?.embed_dim ?? m.dims;
            const params = b?.n_parameters ? fmtParams(b.n_parameters) : m.params;
            const score = b?.retrieval_en ?? null;

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
                      {/* Langues : en direct SEULEMENT si le modèle les déclare vraiment.
                          Sinon (métadonnées incomplètes), on retombe sur l'info curatée —
                          affirmer « pas de FR » sur une donnée absente serait faux. */}
                      {b && b.languages.length > 0 ? (
                        <span
                          className={`rounded px-1.5 py-0.5 text-[10px] ${
                            b.french
                              ? "bg-emerald-500/15 text-emerald-300"
                              : "bg-zinc-800 text-zinc-400"
                          }`}
                          title={`${b.languages.length} langues déclarées (source MTEB)`}
                        >
                          {b.french
                            ? `FR ✓ · ${b.languages.length} langues`
                            : `${b.languages.length} langue(s) — pas de FR`}
                        </span>
                      ) : (
                        <span
                          className={`rounded px-1.5 py-0.5 text-[10px] ${
                            m.languages === "multilingue"
                              ? "bg-emerald-500/15 text-emerald-300"
                              : "bg-zinc-800 text-zinc-400"
                          }`}
                        >
                          {m.languages}
                        </span>
                      )}
                      {inUse && (
                        <span className="rounded bg-blue-500/20 px-1.5 py-0.5 text-[10px] text-blue-300">
                          utilisé
                        </span>
                      )}
                    </div>

                    <p className="mt-1 text-[12px] text-zinc-300">{m.goodFor}</p>

                    {/* Score mesuré (anglais) — donnée live. */}
                    {hasScores && (
                      <p className="mt-0.5 text-[11px]">
                        {score != null ? (
                          <span className="text-zinc-300">
                            📊 Retrieval <strong>EN</strong> :{" "}
                            <span className="font-semibold text-blue-300">{fmtScore(score)}</span>{" "}
                            <span className="text-zinc-500">
                              (ndcg@10, {b?.retrieval_tasks} tâche
                              {(b?.retrieval_tasks ?? 0) > 1 ? "s" : ""}, source MTEB)
                            </span>
                          </span>
                        ) : (
                          <span className="text-zinc-600">
                            📊 Score MTEB indisponible {loadingBench ? "(chargement…)" : ""}
                          </span>
                        )}
                      </p>
                    )}

                    <div className="mt-1.5 flex flex-wrap items-center gap-3 text-[11px] text-zinc-500">
                      <span>{params}</span>
                      {dims && <span>{dims} dims</span>}
                      {b?.max_tokens && <span>{Math.round(b.max_tokens)} tokens</span>}
                      <span className="flex items-center gap-1">
                        {availabilityOf(m).map((bk) => (
                          <Badge
                            key={bk}
                            label={bk}
                            active={
                              (bk === "Local" && backend === "local") ||
                              (bk === "Ollama" &&
                                backend === "server" &&
                                serverKind !== "lmstudio") ||
                              (bk === "LM Studio" &&
                                backend === "server" &&
                                serverKind === "lmstudio")
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
                    {usable && <p className="mt-1.5 font-mono text-[10px] text-zinc-600">{id}</p>}
                  </div>

                  {usable && (
                    <div className="flex shrink-0 flex-col gap-1.5">
                      <button
                        onClick={() => onUse(id, dims)}
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
            Dimensions, taille, contexte et langues sont lus <strong>en direct</strong> depuis les
            métadonnées MTEB (cache 7 jours) — les dimensions appliquées à l'indexation sont donc
            toujours justes. Les notes ★ restent un <strong>avis curaté</strong> tenant compte du
            français, là où le score mesuré ne couvre que l'anglais. Changer de modèle d'embedding
            impose une réindexation.
          </p>
        </div>
      </div>
    </div>
  );
}

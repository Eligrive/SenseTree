import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Check,
  Download,
  Loader2,
  RefreshCw,
  Search,
  Star,
  X,
} from "lucide-react";
import {
  CATALOG,
  availabilityOf,
  hfName,
  idForBackend,
  type Backend,
  type CatalogModel,
  type ServerKind,
  type Task,
} from "../lib/models";
import type { BoardInfo, BoardScore, ModelBenchmark } from "../lib/types";
import { listBenchmarkBoards, modelBenchmarks } from "../lib/ipc";

/// Classements retenus (persistés) : chacun a ses langues, rien n'est figé.
const LS_KEY = "sensetree.boards";
const DEFAULT_BOARDS = ["MTEB(Multilingual, v2)"];

function loadBoards(): string[] {
  try {
    const raw = localStorage.getItem(LS_KEY);
    const v = raw ? (JSON.parse(raw) as string[]) : null;
    return Array.isArray(v) && v.length > 0 ? v : DEFAULT_BOARDS;
  } catch {
    return DEFAULT_BOARDS;
  }
}

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

/// Score MTEB (0–1) → convention du leaderboard (×100).
const fmtScore = (x: number) => (x * 100).toFixed(1);
const fmtParams = (b: number) => (b >= 1 ? `${b.toFixed(1).replace(".", ",")} B` : `${Math.round(b * 1000)} M`);

function Stars({ n }: { n: number }) {
  return (
    <span className="flex shrink-0 items-center gap-0.5" title={`${n}/5 — avis curaté`}>
      {[1, 2, 3, 4, 5].map((i) => (
        <Star key={i} size={11} className={i <= n ? "fill-amber-400 text-amber-400" : "text-zinc-700"} />
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
  const [boards, setBoards] = useState<string[]>(loadBoards);
  const [available, setAvailable] = useState<BoardInfo[]>([]);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [sortBoard, setSortBoard] = useState<string>("relevance");
  const [bench, setBench] = useState<Record<string, ModelBenchmark>>({});
  const [loadingBench, setLoadingBench] = useState(false);
  const [benchError, setBenchError] = useState<string | null>(null);

  // Seuls les embeddings sont couverts par MTEB (les LLM de chat/vision n'y sont pas).
  const hasScores = task === "embedding";
  const boardKey = boards.join("|");

  useEffect(() => {
    if (open && hasScores) listBenchmarkBoards().then(setAvailable).catch(() => setAvailable([]));
  }, [open, hasScores]);

  useEffect(() => {
    localStorage.setItem(LS_KEY, JSON.stringify(boards));
  }, [boardKey]);

  const fetchBench = (refresh: boolean) => {
    if (!hasScores || boards.length === 0) return;
    setLoadingBench(true);
    setBenchError(null);
    modelBenchmarks(boards, refresh)
      .then((list) => setBench(Object.fromEntries(list.map((b) => [b.name, b]))))
      .catch((e) => setBenchError(String(e)))
      .finally(() => setLoadingBench(false));
  };

  useEffect(() => {
    if (open) fetchBench(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, task, boardKey]);

  const labelOf = (board: string) => available.find((b) => b.name === board)?.display_name ?? board;
  const benchOf = (m: CatalogModel): ModelBenchmark | undefined => {
    const h = hfName(m);
    return h ? bench[h] : undefined;
  };
  const scoreOn = (m: CatalogModel, board: string): BoardScore | undefined =>
    benchOf(m)?.scores.find((s) => s.board === board);

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
      .sort((a, b) => {
        if (sortBoard !== "relevance") {
          const sa = scoreOn(a, sortBoard)?.mean ?? null;
          const sb = scoreOn(b, sortBoard)?.mean ?? null;
          if (sa != null && sb != null) return sb - sa;
          if (sa != null) return -1; // les non évalués en dernier
          if (sb != null) return 1;
        }
        return b.quality - a.quality || a.name.localeCompare(b.name);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task, backend, serverKind, query, onlyAvailable, sortBoard, bench, installedServer, localDownloaded]);

  if (!open) return null;

  const backendLabel =
    backend === "local"
      ? "Local (embarqué)"
      : serverKind === "lmstudio"
        ? "Serveur LM Studio"
        : serverKind === "ollama"
          ? "Serveur Ollama"
          : "Serveur HTTP";

  const toggleBoard = (name: string) =>
    setBoards((prev) =>
      prev.includes(name) ? prev.filter((b) => b !== name) : [...prev, name]
    );

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
                title="Rafraîchir depuis le leaderboard MTEB officiel"
                className="flex items-center gap-1.5 rounded-lg bg-zinc-800 px-2.5 py-1.5 text-xs text-zinc-200 hover:bg-zinc-700 disabled:opacity-50"
              >
                {loadingBench ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
                MTEB
              </button>
            )}
            <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300">
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Choix des classements : global multilingue, ou les langues qui te concernent. */}
        {hasScores && (
          <div className="border-b border-zinc-800 px-5 py-2.5">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[11px] text-zinc-500">Classements :</span>
              {boards.map((b) => (
                <span
                  key={b}
                  className="flex items-center gap-1 rounded bg-blue-500/15 px-1.5 py-0.5 text-[10px] text-blue-300"
                >
                  {labelOf(b)}
                  <button onClick={() => toggleBoard(b)} className="hover:text-blue-100">
                    <X size={10} />
                  </button>
                </span>
              ))}
              <button
                onClick={() => setPickerOpen((v) => !v)}
                className="rounded bg-zinc-800 px-2 py-0.5 text-[10px] text-zinc-300 hover:bg-zinc-700"
              >
                {pickerOpen ? "Fermer" : "+ Choisir les langues"}
              </button>
            </div>

            {pickerOpen && (
              <div className="mt-2 max-h-40 overflow-y-auto rounded-lg border border-zinc-800 bg-zinc-900/50 p-2">
                {available.length === 0 && (
                  <p className="text-[11px] text-zinc-600">Chargement des classements…</p>
                )}
                <div className="grid grid-cols-2 gap-1">
                  {available.map((b) => (
                    <label
                      key={b.name}
                      className="flex cursor-pointer items-center gap-1.5 rounded px-1.5 py-1 text-[11px] text-zinc-300 hover:bg-zinc-800"
                    >
                      <input
                        type="checkbox"
                        checked={boards.includes(b.name)}
                        onChange={() => toggleBoard(b.name)}
                      />
                      <span className="truncate">{b.display_name}</span>
                      <span className="ml-auto shrink-0 text-[10px] text-zinc-600">
                        {b.num_models ?? "?"} modèles
                      </span>
                    </label>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Le piège à ne pas reproduire : un score dans UNE langue ne dit rien des autres. */}
        {hasScores && (
          <div className="flex gap-2 border-b border-zinc-800 bg-amber-500/5 px-5 py-2">
            <AlertTriangle size={13} className="mt-0.5 shrink-0 text-amber-400" />
            <p className="text-[11px] leading-relaxed text-amber-200/80">
              Un bon score dans une langue <strong>ne prédit pas</strong> les autres : les modèles
              anglophones chutent à ~20 en coréen contre ~67 pour un vrai multilingue. Choisis les
              classements correspondant à <strong>tes</strong> langues. « non évalué » ≠ mauvais.
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
              value={sortBoard}
              onChange={(e) => setSortBoard(e.target.value)}
              className="shrink-0 rounded-lg border border-zinc-800 bg-zinc-900 px-2 py-1.5 text-xs text-zinc-200 outline-none"
            >
              <option value="relevance">Tri : pertinence (curaté)</option>
              {boards.map((b) => (
                <option key={b} value={b}>
                  Tri : score {labelOf(b)}
                </option>
              ))}
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
            Leaderboard MTEB indisponible ({benchError}) — valeurs de repli affichées.
          </p>
        )}

        {/* Liste */}
        <div className="flex-1 space-y-2 overflow-y-auto p-4">
          {rows.length === 0 && (
            <p className="py-8 text-center text-sm text-zinc-600">Aucun modèle ne correspond.</p>
          )}
          {rows.map((m) => {
            const id = idForBackend(m, backend, serverKind);
            const usable = !!id;
            const installed = usable && isInstalled(id);
            const inUse = usable && currentModel === id;
            const busy = usable && !!downloading[id];
            const b = benchOf(m);

            // Specs live si disponibles, sinon repli sur le catalogue.
            const dims = b?.embed_dim ?? m.dims;
            const params = b?.params_b ? fmtParams(b.params_b) : m.params;

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
                      {inUse && (
                        <span className="rounded bg-blue-500/20 px-1.5 py-0.5 text-[10px] text-blue-300">
                          utilisé
                        </span>
                      )}
                    </div>

                    <p className="mt-1 text-[12px] text-zinc-300">{m.goodFor}</p>

                    {/* Scores officiels, un par classement choisi. */}
                    {hasScores && (
                      <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                        {boards.map((board) => {
                          const s = scoreOn(m, board);
                          const scored = s?.mean != null;
                          return (
                            <span
                              key={board}
                              title={
                                scored
                                  ? `${labelOf(board)} — rang ${s!.rank}/${s!.total} (MTEB)`
                                  : `${labelOf(board)} — modèle non évalué sur ce classement`
                              }
                              className={`rounded px-1.5 py-0.5 text-[10px] ${
                                scored
                                  ? "bg-blue-500/15 text-blue-300"
                                  : "bg-zinc-800/60 text-zinc-600"
                              }`}
                            >
                              {labelOf(board)} :{" "}
                              {scored ? (
                                <>
                                  <strong>{fmtScore(s!.mean!)}</strong>
                                  {s!.rank != null && (
                                    <span className="text-zinc-500">
                                      {" "}
                                      #{s!.rank}/{s!.total}
                                    </span>
                                  )}
                                </>
                              ) : (
                                "non évalué"
                              )}
                            </span>
                          );
                        })}
                        {loadingBench && <Loader2 size={11} className="animate-spin text-zinc-600" />}
                      </div>
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
                              (bk === "Ollama" && backend === "server" && serverKind !== "lmstudio") ||
                              (bk === "LM Studio" && backend === "server" && serverKind === "lmstudio")
                            }
                          />
                        ))}
                      </span>
                    </div>

                    {!usable && (
                      <p className="mt-1.5 text-[11px] text-amber-400/80">
                        Indisponible sur « {backendLabel} » — proposé sur : {availabilityOf(m).join(", ")}.
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
                          {busy ? <Loader2 size={12} className="animate-spin" /> : <Download size={12} />}
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
            Scores, rangs et specs proviennent de l'<strong>API officielle du leaderboard MTEB</strong>{" "}
            (cache 7 jours) — ils se mettent à jour seuls. Les dimensions appliquées à l'indexation
            sont donc toujours justes. Les notes ★ restent un avis curaté. Changer de modèle
            d'embedding impose une réindexation.
          </p>
        </div>
      </div>
    </div>
  );
}

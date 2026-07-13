import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Check,
  Download,
  ExternalLink,
  HelpCircle,
  Loader2,
  RefreshCw,
  Search,
  Star,
  X,
} from "lucide-react";
import {
  CATALOG,
  hfName,
  idForBackend,
  availabilityOf,
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

/// Bornes de taille : un 27 B ne tient pas sur un GPU grand public.
const SIZE_LIMITS = [
  { label: "Toutes tailles", max: Infinity },
  { label: "≤ 1 B (léger)", max: 1 },
  { label: "≤ 4 B", max: 4 },
  { label: "≤ 8 B", max: 8 },
];

const fmtScore = (x: number) => (x * 100).toFixed(1);
const fmtParams = (b: number) =>
  b >= 1 ? `${b.toFixed(1).replace(".", ",")} B` : `${Math.round(b * 1000)} M`;

/// Ligne unifiée : un modèle du classement, enrichi de nos infos curatées.
interface Row {
  hf: string;
  bench?: ModelBenchmark;
  curated?: CatalogModel;
  /// Identifiant utilisable sur le backend courant (undefined = indisponible).
  id?: string;
  /// Vrai si `id` est DÉDUIT du nom HF (non vérifié) — à signaler à l'utilisateur.
  guessed: boolean;
  installed: boolean;
}

function Stars({ n }: { n: number }) {
  return (
    <span className="flex shrink-0 items-center gap-0.5" title={`${n}/5 — avis curaté`}>
      {[1, 2, 3, 4, 5].map((i) => (
        <Star key={i} size={11} className={i <= n ? "fill-amber-400 text-amber-400" : "text-zinc-700"} />
      ))}
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
  const [onlyUsable, setOnlyUsable] = useState(false);
  const [sizeLimit, setSizeLimit] = useState(0);
  const [boards, setBoards] = useState<string[]>(loadBoards);
  const [available, setAvailable] = useState<BoardInfo[]>([]);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [sortBoard, setSortBoard] = useState<string>("");
  const [bench, setBench] = useState<Record<string, ModelBenchmark>>({});
  const [loadingBench, setLoadingBench] = useState(false);
  const [benchError, setBenchError] = useState<string | null>(null);
  const [limit, setLimit] = useState(40);

  // MTEB ne couvre que les embeddings : les LLM de chat/vision gardent la liste curatée.
  const hasScores = task === "embedding";
  const boardKey = boards.join("|");
  const primaryBoard = sortBoard || boards[0] || "";

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

  const labelOf = (b: string) => available.find((x) => x.name === b)?.display_name ?? b;

  const isInstalled = (id: string) =>
    backend === "local"
      ? !!localDownloaded[id]
      : installedServer.some((m) => m === id || m.split(":")[0] === id.split(":")[0]);

  /// Résout l'identifiant utilisable pour le backend courant.
  /// C'est LE point dur : aucune API ne relie un modèle du leaderboard au catalogue
  /// d'Ollama. On privilégie donc, dans l'ordre : nom vérifié (curaté) → modèle déjà
  /// installé → nom déduit du nom HF, explicitement marqué comme non vérifié.
  const resolveId = (hf: string, cur?: CatalogModel): { id?: string; guessed: boolean } => {
    if (backend === "local") {
      // Le local (fastembed) est un ensemble FIXE : pas de déduction possible.
      return { id: cur?.local, guessed: false };
    }
    const verified = serverKind === "lmstudio" ? cur?.lmstudio : cur?.ollama;
    if (verified) return { id: verified, guessed: false };

    const short = (hf.split("/")[1] ?? hf).toLowerCase();
    const hit = installedServer.find(
      (m) => m.toLowerCase() === short || m.toLowerCase().split(":")[0] === short
    );
    if (hit) return { id: hit, guessed: false };

    // LM Studio installe depuis Hugging Face → le nom du dépôt est la bonne clé.
    return { id: serverKind === "lmstudio" ? hf : short, guessed: true };
  };

  const curatedByHf = useMemo(() => {
    const m = new Map<string, CatalogModel>();
    for (const c of CATALOG) {
      const h = hfName(c);
      if (h) m.set(h, c);
    }
    return m;
  }, []);

  const rows: Row[] = useMemo(() => {
    const q = query.trim().toLowerCase();

    // Chat / vision : pas de leaderboard MTEB → on reste sur la liste curatée.
    if (!hasScores) {
      return CATALOG.filter((c) => c.task === task)
        .filter((c) => (q ? (c.name + c.goodFor).toLowerCase().includes(q) : true))
        .sort((a, b) => b.quality - a.quality)
        .map((c) => {
          const id = idForBackend(c, backend, serverKind);
          return {
            hf: c.name,
            curated: c,
            id,
            guessed: false,
            installed: !!id && isInstalled(id),
          };
        });
    }

    // Embeddings : la LISTE vient du leaderboard (donc les nouveaux modèles arrivent seuls).
    const max = SIZE_LIMITS[sizeLimit].max;
    const out: Row[] = Object.values(bench)
      .filter((b) => (q ? b.name.toLowerCase().includes(q) : true))
      .filter((b) => (b.params_b == null ? true : b.params_b <= max))
      .map((b) => {
        const cur = curatedByHf.get(b.name);
        const { id, guessed } = resolveId(b.name, cur);
        return {
          hf: b.name,
          bench: b,
          curated: cur,
          id,
          guessed,
          installed: !!id && isInstalled(id),
        };
      })
      .filter((r) => !onlyUsable || (r.installed && !!r.id));

    const scoreOf = (r: Row) =>
      r.bench?.scores.find((s) => s.board === primaryBoard)?.mean ?? null;

    return out.sort((a, b) => {
      const sa = scoreOf(a);
      const sb = scoreOf(b);
      if (sa != null && sb != null) return sb - sa;
      if (sa != null) return -1; // les non évalués en dernier, jamais assimilés à zéro
      if (sb != null) return 1;
      return a.hf.localeCompare(b.hf);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task, backend, serverKind, query, onlyUsable, sizeLimit, primaryBoard, bench, installedServer, localDownloaded]);

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
    setBoards((prev) => (prev.includes(name) ? prev.filter((b) => b !== name) : [...prev, name]));

  const shown = rows.slice(0, limit);

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/70 p-6">
      <div className="flex max-h-[90vh] w-full max-w-4xl flex-col overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-950 shadow-2xl">
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3.5">
          <div>
            <h2 className="text-base font-semibold text-zinc-100">
              Catalogue de modèles — {TASK_LABEL[task]}
            </h2>
            <p className="text-[11px] text-zinc-500">
              {hasScores ? (
                <>
                  {rows.length} modèles du leaderboard MTEB · cible :{" "}
                  <span className="text-zinc-300">{backendLabel}</span>
                </>
              ) : (
                <>
                  Liste curatée (MTEB ne couvre pas les LLM) · cible :{" "}
                  <span className="text-zinc-300">{backendLabel}</span>
                </>
              )}
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

        {/* Choix des classements : global multilingue, ou TES langues. */}
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
                        {b.num_models ?? "?"}
                      </span>
                    </label>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {hasScores && (
          <div className="flex gap-2 border-b border-zinc-800 bg-amber-500/5 px-5 py-2">
            <AlertTriangle size={13} className="mt-0.5 shrink-0 text-amber-400" />
            <p className="text-[11px] leading-relaxed text-amber-200/80">
              Un bon score dans une langue <strong>ne prédit pas</strong> les autres (les modèles
              anglophones chutent à ~20 en coréen contre ~67 pour un multilingue). « non évalué » ≠
              mauvais. Et un modèle du classement <strong>n'est pas forcément installable</strong> :
              les noms non vérifiés sont signalés.
            </p>
          </div>
        )}

        {/* Filtres */}
        <div className="flex items-center gap-2 border-b border-zinc-800 px-5 py-2.5">
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
            <>
              <select
                value={sizeLimit}
                onChange={(e) => setSizeLimit(Number(e.target.value))}
                className="shrink-0 rounded-lg border border-zinc-800 bg-zinc-900 px-2 py-1.5 text-xs text-zinc-200 outline-none"
                title="Un 27 B ne tiendra pas sur un GPU grand public"
              >
                {SIZE_LIMITS.map((s, i) => (
                  <option key={s.label} value={i}>
                    {s.label}
                  </option>
                ))}
              </select>
              <select
                value={sortBoard}
                onChange={(e) => setSortBoard(e.target.value)}
                className="shrink-0 rounded-lg border border-zinc-800 bg-zinc-900 px-2 py-1.5 text-xs text-zinc-200 outline-none"
              >
                {boards.map((b) => (
                  <option key={b} value={b}>
                    Trier : {labelOf(b)}
                  </option>
                ))}
              </select>
            </>
          )}
          <label className="flex shrink-0 items-center gap-1.5 text-xs text-zinc-400">
            <input
              type="checkbox"
              checked={onlyUsable}
              onChange={(e) => setOnlyUsable(e.target.checked)}
            />
            Installés
          </label>
        </div>

        {benchError && (
          <p className="border-b border-zinc-800 px-5 py-2 text-[11px] text-rose-400">
            Leaderboard MTEB indisponible ({benchError}).
          </p>
        )}

        {/* Liste */}
        <div className="flex-1 space-y-2 overflow-y-auto p-4">
          {loadingBench && shown.length === 0 && (
            <p className="py-8 text-center text-sm text-zinc-600">Chargement du leaderboard…</p>
          )}
          {!loadingBench && shown.length === 0 && (
            <p className="py-8 text-center text-sm text-zinc-600">Aucun modèle ne correspond.</p>
          )}

          {shown.map((r) => {
            const cur = r.curated;
            const b = r.bench;
            const inUse = !!r.id && currentModel === r.id;
            const busy = !!r.id && !!downloading[r.id];
            const dims = b?.embed_dim ?? cur?.dims;
            const params = b?.params_b ? fmtParams(b.params_b) : cur?.params;
            const displayName = cur?.name ?? r.hf.split("/")[1] ?? r.hf;
            const primary = b?.scores.find((s: BoardScore) => s.board === primaryBoard);

            return (
              <div
                key={r.hf}
                className={`rounded-xl border p-3 ${
                  inUse ? "border-blue-500/40 bg-blue-500/5" : "border-zinc-800 bg-zinc-900/30"
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      {primary?.rank != null && primary.mean != null && (
                        <span className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-[10px] text-zinc-400">
                          #{primary.rank}
                        </span>
                      )}
                      <span className="text-sm font-semibold text-zinc-100">{displayName}</span>
                      {cur && <Stars n={cur.quality} />}
                      {inUse && (
                        <span className="rounded bg-blue-500/20 px-1.5 py-0.5 text-[10px] text-blue-300">
                          utilisé
                        </span>
                      )}
                      {!cur && (
                        <span
                          className="rounded bg-emerald-500/10 px-1.5 py-0.5 text-[10px] text-emerald-400/80"
                          title="Découvert automatiquement via le leaderboard"
                        >
                          découvert
                        </span>
                      )}
                    </div>

                    {cur && <p className="mt-1 text-[12px] text-zinc-300">{cur.goodFor}</p>}

                    {/* Scores officiels, un par classement choisi. */}
                    {hasScores && (
                      <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                        {boards.map((board) => {
                          const s = b?.scores.find((x) => x.board === board);
                          const scored = s?.mean != null;
                          return (
                            <span
                              key={board}
                              title={
                                scored
                                  ? `${labelOf(board)} — rang ${s!.rank}/${s!.total}`
                                  : `${labelOf(board)} — non évalué`
                              }
                              className={`rounded px-1.5 py-0.5 text-[10px] ${
                                scored ? "bg-blue-500/15 text-blue-300" : "bg-zinc-800/60 text-zinc-600"
                              }`}
                            >
                              {labelOf(board)} :{" "}
                              {scored ? <strong>{fmtScore(s!.mean!)}</strong> : "non évalué"}
                            </span>
                          );
                        })}
                      </div>
                    )}

                    <div className="mt-1.5 flex flex-wrap items-center gap-3 text-[11px] text-zinc-500">
                      {params && <span>{params}</span>}
                      {dims && <span>{dims} dims</span>}
                      {b?.max_tokens && <span>{Math.round(b.max_tokens)} tokens</span>}
                      {cur && (
                        <span className="flex items-center gap-1">
                          {availabilityOf(cur).map((bk) => (
                            <span key={bk} className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-500">
                              {bk}
                            </span>
                          ))}
                        </span>
                      )}
                      {b?.url && (
                        <a
                          href={b.url}
                          target="_blank"
                          rel="noreferrer"
                          className="flex items-center gap-0.5 text-zinc-500 hover:text-zinc-300"
                        >
                          <ExternalLink size={10} /> HF
                        </a>
                      )}
                    </div>

                    {r.id ? (
                      <p className="mt-1.5 flex items-center gap-1.5 font-mono text-[10px] text-zinc-600">
                        {r.id}
                        {r.guessed && (
                          <span
                            className="flex items-center gap-0.5 rounded bg-amber-500/10 px-1 py-0.5 font-sans text-[10px] text-amber-400/90"
                            title="Aucune API ne relie le leaderboard au catalogue Ollama : ce nom est déduit et peut être faux."
                          >
                            <HelpCircle size={9} /> nom déduit — à vérifier
                          </span>
                        )}
                      </p>
                    ) : (
                      <p className="mt-1.5 text-[11px] text-amber-400/80">
                        Pas de version « {backendLabel} » connue pour ce modèle.
                      </p>
                    )}
                  </div>

                  {r.id && (
                    <div className="flex shrink-0 flex-col gap-1.5">
                      <button
                        onClick={() => onUse(r.id!, dims)}
                        disabled={inUse}
                        className="rounded-lg bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-500 disabled:opacity-40"
                      >
                        {inUse ? "Utilisé" : "Utiliser"}
                      </button>
                      {r.installed ? (
                        <span className="flex items-center justify-center gap-1 text-[11px] text-emerald-400">
                          <Check size={12} /> installé
                        </span>
                      ) : (
                        <button
                          onClick={() => onDownload(r.id!)}
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

          {rows.length > shown.length && (
            <button
              onClick={() => setLimit((l) => l + 40)}
              className="w-full rounded-lg bg-zinc-900 py-2 text-xs text-zinc-400 hover:bg-zinc-800"
            >
              Afficher plus ({rows.length - shown.length} restants)
            </button>
          )}
        </div>

        <div className="border-t border-zinc-800 px-5 py-2.5">
          <p className="text-[11px] text-zinc-600">
            La liste elle-même vient de l'<strong>API officielle du leaderboard MTEB</strong> — les
            nouveaux modèles apparaissent donc <strong>tout seuls</strong>. Les dimensions sont lues
            en direct, donc toujours justes pour l'indexation. Changer de modèle d'embedding impose
            une réindexation.
          </p>
        </div>
      </div>
    </div>
  );
}

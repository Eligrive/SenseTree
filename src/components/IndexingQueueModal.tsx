import { useCallback, useEffect, useState } from "react";
import {
  AlertTriangle,
  AudioLines,
  Ban,
  Brain,
  CheckCircle2,
  Cpu,
  Eye,
  FileText,
  ListTree,
  Loader2,
  RotateCw,
  Route,
  X,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { IndexingQueueView } from "../lib/types";
import { ignoreIndexing, indexingQueue, retryAllFailed, retryIndexing } from "../lib/ipc";

const ROUTE_META: Record<string, { label: string; cls: string; Icon: LucideIcon }> = {
  routing: { label: "Routage", cls: "bg-teal-500/15 text-teal-300", Icon: Route },
  embedding: { label: "Embedding", cls: "bg-blue-500/15 text-blue-300", Icon: Cpu },
  vision: { label: "Vision", cls: "bg-purple-500/15 text-purple-300", Icon: Eye },
  media: { label: "Média", cls: "bg-cyan-500/15 text-cyan-300", Icon: AudioLines },
  reasoning: { label: "Reasoning", cls: "bg-amber-500/15 text-amber-300", Icon: Brain },
  context: { label: "Contexte", cls: "bg-zinc-700/40 text-zinc-400", Icon: FileText },
};

function RouteBadge({ route }: { route: string }) {
  const m = ROUTE_META[route] ?? ROUTE_META.context;
  const { Icon } = m;
  return (
    <span
      className={`inline-flex shrink-0 items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium ${m.cls}`}
    >
      <Icon size={10} /> {m.label}
    </span>
  );
}

/// Pipeline complet d'un élément, en séquence (ex. Vision › Reasoning › Embedding).
function RouteBadges({ routes }: { routes: string[] }) {
  return (
    <span className="flex shrink-0 flex-wrap items-center justify-end gap-1">
      {routes.map((r, i) => (
        <span key={i} className="flex items-center gap-1">
          {i > 0 && <span className="text-zinc-600">›</span>}
          <RouteBadge route={r} />
        </span>
      ))}
    </span>
  );
}

const baseName = (p: string) => p.replace(/[\\/]+$/, "").split(/[\\/]/).pop() || p;

export default function IndexingQueueModal({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const [data, setData] = useState<IndexingQueueView | null>(null);

  const refresh = useCallback(() => {
    indexingQueue(80)
      .then(setData)
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!open) return;
    refresh();
    const t = setInterval(refresh, 1000);
    return () => clearInterval(t);
  }, [open, refresh]);

  const onRetry = (path: string) => retryIndexing(path).then(refresh).catch(() => {});
  const onIgnore = (path: string) => ignoreIndexing(path).then(refresh).catch(() => {});
  const onRetryAll = () => retryAllFailed().then(refresh).catch(() => {});

  if (!open) return null;

  const stats = data?.stats;
  const current = data?.current ?? null;
  // Les échecs viennent d'une requête dédiée : sinon, noyés dans la file, ils
  // sortaient de la fenêtre LIMIT et n'apparaissaient jamais.
  const upcoming = data?.pending ?? [];
  const failed = data?.failed ?? [];

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6"
      onClick={onClose}
    >
      <div
        className="flex max-h-[80vh] w-full max-w-xl flex-col overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-950 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3.5">
          <h2 className="flex items-center gap-2 text-base font-semibold text-zinc-100">
            <ListTree size={18} className="text-blue-400" /> File d'indexation
          </h2>
          <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300">
            <X size={18} />
          </button>
        </div>

        <div className="flex-1 space-y-4 overflow-y-auto p-5">
          {/* Élément en cours */}
          <div>
            <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
              En cours de traitement
            </p>
            {current ? (
              <div className="flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900/40 p-2.5">
                <Loader2 size={15} className="shrink-0 animate-spin text-blue-400" />
                <span
                  className="min-w-0 flex-1 truncate text-sm text-zinc-200"
                  title={current.path}
                >
                  {baseName(current.path)}
                </span>
                <span className="shrink-0 text-[10px] text-zinc-600">{current.kind}</span>
                <RouteBadges routes={current.routes} />
              </div>
            ) : (
              <div className="flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900/40 p-2.5 text-sm text-zinc-500">
                <CheckCircle2 size={15} className="text-emerald-500" /> Rien en cours
              </div>
            )}
          </div>

          {/* Compteurs */}
          {stats && (
            <div className="grid grid-cols-3 gap-2">
              <Stat value={stats.pending} label="en file" cls="text-amber-300" />
              <Stat value={stats.completed} label="indexés" cls="text-emerald-300" />
              <Stat value={stats.failed} label="échecs" cls="text-rose-300" />
            </div>
          )}

          {/* Échecs — REMONTÉS en haut : c'est ce sur quoi on veut agir en priorité,
              et sinon la section serait noyée sous les (nombreux) éléments « À venir ». */}
          {failed.length > 0 && (
            <div className="rounded-lg border border-rose-900/40 bg-rose-950/10 p-3">
              <div className="mb-1.5 flex items-center justify-between">
                <p className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-rose-400">
                  <AlertTriangle size={12} /> Échecs ({failed.length})
                </p>
                <button
                  onClick={onRetryAll}
                  className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] text-zinc-400 transition hover:bg-zinc-800 hover:text-zinc-200"
                  title="Remettre tous les échecs dans la file"
                >
                  <RotateCw size={11} /> Tout relancer
                </button>
              </div>
              <ul className="max-h-72 space-y-1 overflow-y-auto pr-1">
                {failed.map((q) => (
                  <li
                    key={q.path}
                    className="rounded-md border border-rose-900/40 bg-rose-950/20 px-2.5 py-1.5"
                  >
                    <div className="flex items-center gap-2">
                      <span
                        className="min-w-0 flex-1 truncate text-sm text-rose-200"
                        title={q.path}
                      >
                        {baseName(q.path)}
                      </span>
                      <RouteBadges routes={q.routes} />
                      <button
                        onClick={() => onRetry(q.path)}
                        title="Relancer l'indexation de ce fichier"
                        className="shrink-0 rounded p-1 text-zinc-400 transition hover:bg-blue-500/15 hover:text-blue-300"
                      >
                        <RotateCw size={13} />
                      </button>
                      <button
                        onClick={() => onIgnore(q.path)}
                        title="Ignorer : retirer de la file (ne sera plus retenté)"
                        className="shrink-0 rounded p-1 text-zinc-400 transition hover:bg-zinc-700/40 hover:text-zinc-200"
                      >
                        <Ban size={13} />
                      </button>
                    </div>
                    {q.last_error && (
                      <p
                        className="mt-0.5 truncate text-[11px] text-rose-400/80"
                        title={q.last_error}
                      >
                        {q.last_error}
                      </p>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* À venir (liste plafonnée avec son propre scroll pour ne pas noyer le reste) */}
          <div>
            <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
              À venir{" "}
              {upcoming.length > 0 && <span className="text-zinc-600">({upcoming.length})</span>}
            </p>
            {upcoming.length === 0 ? (
              <p className="text-sm text-zinc-600">File vide.</p>
            ) : (
              <ul className="max-h-72 space-y-1 overflow-y-auto pr-1">
                {upcoming.map((q) => (
                  <li
                    key={q.path}
                    className="flex items-center gap-2 rounded-md bg-zinc-900/40 px-2.5 py-1.5"
                  >
                    <span
                      className="min-w-0 flex-1 truncate text-sm text-zinc-300"
                      title={q.path}
                    >
                      {baseName(q.path)}
                    </span>
                    <span className="shrink-0 text-[10px] text-zinc-600">{q.kind}</span>
                    <RouteBadges routes={q.routes} />
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function Stat({ value, label, cls }: { value: number; label: string; cls: string }) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-2.5 text-center">
      <div className={`text-lg font-semibold ${cls}`}>{value.toLocaleString()}</div>
      <div className="text-[10px] uppercase tracking-wider text-zinc-500">{label}</div>
    </div>
  );
}

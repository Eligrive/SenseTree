import { Folder, Settings, Activity, Sprout, Loader2, CheckCircle2 } from "lucide-react";
import type { HealthReport, IndexingStats } from "../lib/types";

interface Props {
  roots: string[];
  currentRoot: string | null;
  onSelectRoot: (root: string) => void;
  health: HealthReport | null;
  stats: IndexingStats | null;
  onOpenSettings: () => void;
  onAnalyze: () => void;
}

function HealthDot({ ok, label, detail }: { ok: boolean; label: string; detail: string }) {
  return (
    <div className="flex items-center gap-2 text-xs" title={detail}>
      <span
        className={`h-2 w-2 rounded-full ${ok ? "bg-emerald-500" : "bg-zinc-600"}`}
      />
      <span className={ok ? "text-zinc-300" : "text-zinc-500"}>{label}</span>
    </div>
  );
}

function IndexingProgress({ stats }: { stats: IndexingStats | null }) {
  if (!stats || stats.total === 0) return null;
  const done = stats.completed + stats.failed;
  const pct = stats.total > 0 ? Math.round((done / stats.total) * 100) : 100;
  const finished = stats.pending === 0;

  return (
    <div className="space-y-1.5 rounded-md bg-zinc-900/60 p-2.5">
      <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-widest text-zinc-500">
        <span className="flex items-center gap-1.5">
          {finished ? (
            <CheckCircle2 size={11} className="text-emerald-500" />
          ) : (
            <Loader2 size={11} className="animate-spin text-blue-400" />
          )}
          Indexation
        </span>
        <span className="text-zinc-400">{pct}%</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
        <div
          className={`h-full rounded-full transition-all ${
            finished ? "bg-emerald-500" : "bg-blue-500"
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="flex items-center justify-between text-[11px] text-zinc-500">
        <span>
          {finished ? (
            "À jour"
          ) : (
            <>
              <span className="text-amber-400">{stats.pending.toLocaleString()}</span> à indexer
            </>
          )}
        </span>
        <span>{stats.total.toLocaleString()} docs</span>
      </div>
      {stats.failed > 0 && (
        <div className="text-[11px] text-rose-400">{stats.failed} en échec</div>
      )}
    </div>
  );
}

export default function Sidebar({
  roots,
  currentRoot,
  onSelectRoot,
  health,
  stats,
  onOpenSettings,
  onAnalyze,
}: Props) {
  return (
    <aside className="flex h-full w-60 flex-col border-r border-zinc-800 bg-zinc-950/60">
      <div className="flex items-center gap-2 px-4 py-4">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 text-sm font-bold">
          S
        </div>
        <h1 className="text-lg font-semibold tracking-tight text-zinc-100">
          Sense<span className="text-blue-400">Tree</span>
        </h1>
      </div>

      <div className="px-3">
        <p className="px-1 pb-1 text-[10px] font-semibold uppercase tracking-widest text-zinc-500">
          Dossiers indexés
        </p>
        <ul className="space-y-0.5">
          {roots.length === 0 && (
            <li className="px-2 py-1 text-xs text-zinc-600">Aucun dossier configuré</li>
          )}
          {roots.map((root) => {
            const active = root === currentRoot;
            const name = root.replace(/[\\/]+$/, "").split(/[\\/]/).pop() || root;
            return (
              <li key={root}>
                <button
                  onClick={() => onSelectRoot(root)}
                  className={`flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-sm transition-colors ${
                    active
                      ? "bg-blue-500/15 text-blue-300"
                      : "text-zinc-300 hover:bg-zinc-800/60"
                  }`}
                  title={root}
                >
                  <Folder size={15} className="shrink-0" />
                  <span className="truncate">{name}</span>
                </button>
              </li>
            );
          })}
        </ul>
      </div>

      <div className="mt-auto space-y-3 border-t border-zinc-800 p-3">
        <IndexingProgress stats={stats} />

        <button
          onClick={onAnalyze}
          className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-sm text-zinc-300 hover:bg-zinc-800/60"
        >
          <Sprout size={15} /> Diagnostic du dossier
        </button>

        <div className="space-y-1.5 rounded-md bg-zinc-900/60 p-2.5">
          <div className="flex items-center gap-1.5 pb-0.5 text-[10px] font-semibold uppercase tracking-widest text-zinc-500">
            <Activity size={11} /> État IA
          </div>
          <HealthDot
            ok={!!health?.embedding_ok}
            label="Embedding"
            detail={health?.embedding_detail ?? "…"}
          />
          <HealthDot
            ok={!!health?.reasoning_ok}
            label="Reasoning"
            detail={health?.reasoning_detail ?? "…"}
          />
          <HealthDot
            ok={!!health?.vision_ok}
            label="Vision"
            detail={health?.vision_detail ?? "…"}
          />
        </div>

        <button
          onClick={onOpenSettings}
          className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-sm text-zinc-300 hover:bg-zinc-800/60"
        >
          <Settings size={15} /> Paramètres
        </button>
      </div>
    </aside>
  );
}

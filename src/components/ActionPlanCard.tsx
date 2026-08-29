import { useState } from "react";
import { ArrowRight, Check, FolderPlus, Sparkles, Trash2, X } from "lucide-react";
import type { ActionPlan, Operation } from "../lib/types";

interface Props {
  plan: ActionPlan;
  status: "pending" | "applied" | "discarded";
  onApprove: (operations: Operation[]) => void;
  onDiscard: () => void;
}

function shortName(p: string | null): string {
  if (!p) return "";
  return p.replace(/[\\/]+$/, "").split(/[\\/]/).pop() || p;
}

function OpRow({ op }: { op: Operation }) {
  // Requalification : le fichier n'est pas touché, seule sa description change.
  // On montre le texte proposé EN ENTIER — c'est lui que l'utilisateur valide, et
  // le tronquer reviendrait à lui faire approuver ce qu'il n'a pas lu.
  if (op.kind === "requalify") {
    return (
      <div className="space-y-1 text-xs">
        <div className="flex items-center gap-2">
          <Sparkles size={13} className="shrink-0 text-violet-400" />
          <span className="truncate text-violet-300" title={op.old_path ?? ""}>
            {shortName(op.old_path)}
          </span>
          <span className="shrink-0 text-[10px] text-zinc-600">sens corrigé</span>
        </div>
        <p className="rounded border border-violet-500/20 bg-violet-500/5 px-2 py-1 text-[11px] leading-relaxed text-zinc-300">
          {op.new_summary}
        </p>
      </div>
    );
  }
  if (op.kind === "delete") {
    return (
      <div className="flex items-center gap-2 text-xs">
        <Trash2 size={13} className="shrink-0 text-rose-400" />
        <span className="truncate text-rose-300 line-through" title={op.old_path ?? ""}>
          {shortName(op.old_path)}
        </span>
        <span className="text-zinc-600">→ corbeille</span>
      </div>
    );
  }
  if (op.kind === "mkdir") {
    return (
      <div className="flex items-center gap-2 text-xs">
        <FolderPlus size={13} className="shrink-0 text-blue-400" />
        <span className="truncate text-blue-300" title={op.new_path ?? ""}>
          {shortName(op.new_path)}
        </span>
      </div>
    );
  }
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="truncate text-zinc-400 line-through" title={op.old_path ?? ""}>
        {shortName(op.old_path)}
      </span>
      <ArrowRight size={12} className="shrink-0 text-zinc-600" />
      <span className="truncate text-emerald-300" title={op.new_path ?? ""}>
        {shortName(op.new_path)}
      </span>
    </div>
  );
}

export default function ActionPlanCard({ plan, status, onApprove, onDiscard }: Props) {
  // Toutes les opérations sont cochées par défaut ; l'utilisateur peut en décocher
  // avant d'appliquer (n'exécute que la sélection).
  const [selected, setSelected] = useState<Set<number>>(
    () => new Set(plan.operations.map((_, i) => i))
  );
  const pending = status === "pending";
  const toggle = (i: number) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });
  const kept = selected.size;
  const total = plan.operations.length;
  const apply = () => onApprove(plan.operations.filter((_, i) => selected.has(i)));

  return (
    <div className="rounded-xl border border-zinc-700 bg-zinc-900/80 p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-[10px] font-semibold uppercase tracking-widest text-amber-400">
          Plan d'action · Dry-Run
        </span>
        <span className="text-[10px] text-zinc-500">
          {pending && kept < total ? `${kept} / ${total}` : total} opération(s)
        </span>
      </div>

      {plan.summary && <p className="mb-2 text-xs text-zinc-400">{plan.summary}</p>}

      <div className="max-h-56 space-y-1 overflow-y-auto rounded-lg bg-zinc-950/60 p-2.5">
        {plan.operations.map((op, i) => (
          <label
            key={i}
            className={`flex items-center gap-2 rounded px-1 py-0.5 ${
              pending ? "cursor-pointer hover:bg-zinc-800/40" : ""
            }`}
          >
            {pending && (
              <input
                type="checkbox"
                checked={selected.has(i)}
                onChange={() => toggle(i)}
                className="shrink-0 accent-emerald-500"
                title="Inclure cette opération"
              />
            )}
            <div className={`min-w-0 flex-1 ${pending && !selected.has(i) ? "opacity-40" : ""}`}>
              <OpRow op={op} />
            </div>
          </label>
        ))}
      </div>

      {pending ? (
        <div className="mt-3 flex gap-2">
          <button
            onClick={apply}
            disabled={kept === 0}
            className="flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-emerald-600 py-1.5 text-xs font-medium text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Check size={14} /> Appliquer{kept < total ? ` (${kept})` : ""}
          </button>
          <button
            onClick={onDiscard}
            className="flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-zinc-800 py-1.5 text-xs font-medium text-zinc-300 hover:bg-zinc-700"
          >
            <X size={14} /> Annuler
          </button>
        </div>
      ) : (
        <div className="mt-3 text-center text-xs font-medium text-zinc-500">
          {status === "applied" ? "✅ Appliqué au disque" : "Plan annulé"}
        </div>
      )}
    </div>
  );
}

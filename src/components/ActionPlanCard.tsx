import { ArrowRight, Check, FolderPlus, Trash2, X } from "lucide-react";
import type { ActionPlan, Operation } from "../lib/types";

interface Props {
  plan: ActionPlan;
  status: "pending" | "applied" | "discarded";
  onApprove: () => void;
  onDiscard: () => void;
}

function shortName(p: string | null): string {
  if (!p) return "";
  return p.replace(/[\\/]+$/, "").split(/[\\/]/).pop() || p;
}

function OpRow({ op }: { op: Operation }) {
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
  return (
    <div className="rounded-xl border border-zinc-700 bg-zinc-900/80 p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-[10px] font-semibold uppercase tracking-widest text-amber-400">
          Plan d'action · Dry-Run
        </span>
        <span className="text-[10px] text-zinc-500">{plan.operations.length} opération(s)</span>
      </div>

      {plan.summary && <p className="mb-2 text-xs text-zinc-400">{plan.summary}</p>}

      <div className="max-h-56 space-y-1.5 overflow-y-auto rounded-lg bg-zinc-950/60 p-2.5">
        {plan.operations.map((op, i) => (
          <OpRow key={i} op={op} />
        ))}
      </div>

      {status === "pending" ? (
        <div className="mt-3 flex gap-2">
          <button
            onClick={onApprove}
            className="flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-emerald-600 py-1.5 text-xs font-medium text-white hover:bg-emerald-500"
          >
            <Check size={14} /> Appliquer
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

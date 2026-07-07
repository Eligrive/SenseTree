import { AlertTriangle, Copy, FolderX, Sprout, X } from "lucide-react";
import type { DirectoryReport } from "../lib/types";

interface Props {
  report: DirectoryReport | null;
  loading: boolean;
  onClose: () => void;
}

export default function GardenerModal({ report, loading, onClose }: Props) {
  if (!loading && !report) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6">
      <div className="flex max-h-[80vh] w-full max-w-lg flex-col overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-950 shadow-2xl">
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3.5">
          <h2 className="flex items-center gap-2 text-base font-semibold text-zinc-100">
            <Sprout size={18} className="text-emerald-400" /> Diagnostic du dossier
          </h2>
          <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300">
            <X size={18} />
          </button>
        </div>

        <div className="flex-1 space-y-4 overflow-y-auto p-5">
          {loading || !report ? (
            <p className="text-sm text-zinc-500">Analyse en cours…</p>
          ) : (
            <>
              <p className="truncate text-xs text-zinc-500">{report.scanned_path}</p>
              <div className="grid grid-cols-3 gap-2">
                <Stat value={report.file_count} label="fichiers" />
                <Stat value={report.max_depth} label="profondeur" />
                <Stat value={report.duplicate_groups.length} label="doublons" />
              </div>

              <div className="space-y-2">
                {report.suggestions.map((s, i) => (
                  <div
                    key={i}
                    className="flex items-start gap-2 rounded-lg border border-zinc-800 bg-zinc-900/40 p-2.5 text-sm text-zinc-300"
                  >
                    <AlertTriangle size={15} className="mt-0.5 shrink-0 text-amber-400" />
                    <span>{s}</span>
                  </div>
                ))}
              </div>

              {report.duplicate_groups.length > 0 && (
                <div>
                  <h4 className="mb-1.5 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-zinc-500">
                    <Copy size={12} /> Doublons exacts
                  </h4>
                  <div className="space-y-2">
                    {report.duplicate_groups.slice(0, 10).map((g) => (
                      <div key={g.content_hash} className="rounded-lg bg-zinc-900/40 p-2 text-xs">
                        {g.paths.map((p) => (
                          <p key={p} className="truncate text-zinc-400" title={p}>
                            {p}
                          </p>
                        ))}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {report.empty_dirs.length > 0 && (
                <div>
                  <h4 className="mb-1.5 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-zinc-500">
                    <FolderX size={12} /> Dossiers vides
                  </h4>
                  <div className="space-y-0.5 text-xs text-zinc-400">
                    {report.empty_dirs.slice(0, 12).map((d) => (
                      <p key={d} className="truncate" title={d}>
                        {d}
                      </p>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function Stat({ value, label }: { value: number; label: string }) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-2.5 text-center">
      <div className="text-lg font-semibold text-zinc-100">{value}</div>
      <div className="text-[10px] uppercase tracking-wider text-zinc-500">{label}</div>
    </div>
  );
}

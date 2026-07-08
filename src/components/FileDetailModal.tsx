import type { ReactNode } from "react";
import { Boxes, Check, Clock, FileText, Folder, FolderTree, X } from "lucide-react";
import type { PathDetails } from "../lib/types";
import { formatBytes, formatDate } from "../lib/format";

interface Props {
  details: PathDetails | null;
  loading: boolean;
  onClose: () => void;
  onOpenFile: (path: string) => void;
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex gap-3 py-1.5 text-sm">
      <span className="w-28 shrink-0 text-zinc-500">{label}</span>
      <span className="min-w-0 flex-1 text-zinc-200">{children}</span>
    </div>
  );
}

function IndexState({ d }: { d: PathDetails }) {
  if (d.indexed)
    return (
      <span className="inline-flex items-center gap-1.5 text-emerald-400">
        <Check size={14} /> Indexé
      </span>
    );
  const s = d.status;
  if (s === "pending" || s === "pending_extraction")
    return (
      <span className="inline-flex items-center gap-1.5 text-amber-400">
        <Clock size={14} /> En file d'attente
      </span>
    );
  if (s === "failed" || s === "failed_permanent")
    return <span className="text-rose-400">Échec d'indexation</span>;
  return <span className="text-zinc-500">Non indexé</span>;
}

function FolderModeLabel({ mode }: { mode: string | null }) {
  if (mode === "block")
    return (
      <span className="inline-flex items-center gap-1.5 text-purple-300">
        <Boxes size={14} /> Bloc sémantique
      </span>
    );
  if (mode === "pending")
    return (
      <span className="inline-flex items-center gap-1.5 text-amber-300">
        <Clock size={14} /> Classification en attente
      </span>
    );
  return (
    <span className="inline-flex items-center gap-1.5 text-zinc-300">
      <FolderTree size={14} /> Exploré récursivement
    </span>
  );
}

export default function FileDetailModal({ details, loading, onClose, onOpenFile }: Props) {
  if (!loading && !details) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6" onClick={onClose}>
      <div
        className="flex max-h-[80vh] w-full max-w-lg flex-col overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-950 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3.5">
          <h2 className="flex min-w-0 items-center gap-2 text-base font-semibold text-zinc-100">
            {details?.is_directory ? (
              <Folder size={17} className="shrink-0 text-blue-400" />
            ) : (
              <FileText size={17} className="shrink-0 text-emerald-400" />
            )}
            <span className="truncate">{details?.name ?? "…"}</span>
          </h2>
          <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300">
            <X size={18} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5">
          {loading || !details ? (
            <p className="text-sm text-zinc-500">Chargement…</p>
          ) : (
            <>
              <div className="divide-y divide-zinc-800/60">
                <Row label="Indexation">
                  <IndexState d={details} />
                </Row>
                <Row label="Extraction">{details.content_kind}</Row>
                {details.is_directory && (
                  <Row label="Traitement">
                    <FolderModeLabel mode={details.folder_mode} />
                  </Row>
                )}
                {!details.is_directory && (
                  <Row label="Taille">{formatBytes(details.size_bytes)}</Row>
                )}
                {details.modified != null && (
                  <Row label="Modifié">{formatDate(details.modified)}</Row>
                )}
                <Row label="Chemin">
                  <span className="break-all text-xs text-zinc-400">{details.path}</span>
                </Row>
                {details.last_error && (
                  <Row label="Erreur">
                    <span className="text-xs text-rose-400">{details.last_error}</span>
                  </Row>
                )}
              </div>

              {details.summary && (
                <div className="mt-4">
                  <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                    Sens extrait
                  </p>
                  <p className="max-h-52 overflow-y-auto whitespace-pre-wrap rounded-lg border border-zinc-800 bg-zinc-900/40 p-3 text-sm text-zinc-300">
                    {details.summary}
                  </p>
                </div>
              )}
            </>
          )}
        </div>

        {details && !details.is_directory && (
          <div className="flex justify-end border-t border-zinc-800 px-5 py-3">
            <button
              onClick={() => onOpenFile(details.path)}
              className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500"
            >
              Ouvrir le fichier
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

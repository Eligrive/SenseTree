import { useState, type ReactNode } from "react";
import {
  Boxes,
  Check,
  Clock,
  FileText,
  Folder,
  FolderTree,
  Loader2,
  Pencil,
  RotateCcw,
  Save,
  Sparkles,
  X,
} from "lucide-react";
import type { PathDetails } from "../lib/types";
import { qualifyFile, qualifyFolder, reindexPath, setFileSummary } from "../lib/ipc";
import { formatBytes, formatDate } from "../lib/format";

interface Props {
  details: PathDetails | null;
  loading: boolean;
  onClose: () => void;
  onOpenFile: (path: string) => void;
  onSummarySaved: (path: string) => void;
  onDiscussFile: (path: string) => void;
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex gap-3 py-1 text-sm">
      <span className="w-24 shrink-0 text-zinc-500">{label}</span>
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
  if (d.in_block)
    return (
      <span className="inline-flex items-center gap-1.5 text-purple-300">
        <Boxes size={14} /> Membre d'un bloc
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

/// Bouton « Qualifier tout le dossier » : lance en tâche de fond la qualification IA
/// des fichiers du dossier encore non qualifiés (mode paresseux, en lot).
function FolderQualifyButton({ path }: { path: string }) {
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const run = () => {
    setBusy(true);
    setMsg(null);
    qualifyFolder(path)
      .then((n) =>
        setMsg(
          n > 0
            ? `${n} fichier(s) en cours de qualification en arrière-plan… (les sens apparaîtront au fur et à mesure)`
            : "Rien à qualifier ici (déjà qualifié, ou aucun contenu extrait)."
        )
      )
      .catch((e) => setMsg(`⚠️ ${String(e)}`))
      .finally(() => setBusy(false));
  };

  return (
    <div className="mt-3">
      <button
        onClick={run}
        disabled={busy}
        className="flex w-full items-center justify-center gap-2 rounded-lg border border-blue-500/40 bg-blue-500/10 px-4 py-2 text-sm font-medium text-blue-300 transition hover:bg-blue-500/20 disabled:opacity-50"
        title="Qualifier avec l'IA tous les fichiers du dossier encore non qualifiés"
      >
        {busy ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />}
        Qualifier tout le dossier avec l'IA
      </button>
      {msg && <p className="mt-1.5 text-[11px] text-zinc-400">{msg}</p>}
    </div>
  );
}

/// Section « Sens extrait » : toujours affichée (pour comparer au document),
/// avec le texte complet (scrollable) et l'édition manuelle du sens.
function SenseSection({
  path,
  summary,
  canQualify,
  onSaved,
}: {
  path: string;
  summary: string | null;
  canQualify: boolean;
  onSaved: (path: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(summary ?? "");
  const [saving, setSaving] = useState(false);
  const [qualifying, setQualifying] = useState(false);
  const [qualifyErr, setQualifyErr] = useState<string | null>(null);

  const save = () => {
    setSaving(true);
    setFileSummary(path, draft.trim())
      .then(() => {
        setEditing(false);
        onSaved(path);
      })
      .catch((e) => console.error("set_file_summary:", e))
      .finally(() => setSaving(false));
  };

  // Qualification à la demande : relance le reasoning sur le contenu déjà extrait.
  const qualify = () => {
    setQualifying(true);
    setQualifyErr(null);
    qualifyFile(path)
      .then(() => onSaved(path))
      .catch((e) => setQualifyErr(String(e)))
      .finally(() => setQualifying(false));
  };

  // Retraitement COMPLET : le seul recours quand l'extraction elle-meme a echoue
  // (scan mal decode, PDF illisible). La qualification, elle, ne relit que l'extrait
  // deja stocke -- inutile s'il est vide.
  const reindex = () => {
    setQualifying(true);
    setQualifyErr(null);
    reindexPath(path)
      .then(() => setQualifyErr("Remis en file — le résultat apparaîtra une fois traité."))
      .catch((e) => setQualifyErr(String(e)))
      .finally(() => setQualifying(false));
  };

  return (
    <div className="mt-3">
      <div className="mb-1.5 flex items-center justify-between">
        <p className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
          Sens extrait
        </p>
        {!editing && (
          <div className="flex items-center gap-2">
            {canQualify && (
              <button
                onClick={qualify}
                disabled={qualifying}
                className="flex items-center gap-1 text-[11px] text-blue-400 hover:text-blue-300 disabled:opacity-50"
                title="Faire décrire ce fichier par l'IA (reasoning) à partir de son contenu extrait"
              >
                {qualifying ? <Loader2 size={11} className="animate-spin" /> : <Sparkles size={11} />}
                {qualifying ? "Qualification…" : "Qualifier avec l'IA"}
              </button>
            )}
            <button
              onClick={reindex}
              disabled={qualifying}
              className="flex items-center gap-1 text-[11px] text-zinc-400 hover:text-zinc-200 disabled:opacity-50"
              title="Tout refaire depuis le fichier : extraction, OCR, qualification, vectorisation. À utiliser quand l'extraction a échoué (document vu comme « vide »)."
            >
              <RotateCcw size={11} /> Réindexer
            </button>
            <button
              onClick={() => {
                setDraft(summary ?? "");
                setEditing(true);
              }}
              className="flex items-center gap-1 text-[11px] text-zinc-500 hover:text-zinc-300"
              title="Modifier le sens à la main"
            >
              <Pencil size={11} /> Modifier
            </button>
          </div>
        )}
      </div>
      {qualifyErr && <p className="mb-1.5 text-[11px] text-rose-400">{qualifyErr}</p>}

      {editing ? (
        <div className="space-y-2">
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            rows={6}
            autoFocus
            placeholder="Décris ce qu'est ce document et ses informations-clés…"
            className="w-full resize-y rounded-lg border border-zinc-700 bg-zinc-900 p-3 text-sm text-zinc-200 outline-none focus:border-blue-500"
          />
          <div className="flex gap-2">
            <button
              onClick={save}
              disabled={saving}
              className="flex items-center gap-1.5 rounded-lg bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50"
            >
              <Save size={13} /> {saving ? "Enregistrement…" : "Enregistrer"}
            </button>
            <button
              onClick={() => setEditing(false)}
              className="rounded-lg px-3 py-1.5 text-sm text-zinc-400 hover:bg-zinc-800"
            >
              Annuler
            </button>
          </div>
        </div>
      ) : summary ? (
        <p className="max-h-72 overflow-y-auto whitespace-pre-wrap rounded-lg border border-zinc-800 bg-zinc-900/40 p-3 text-sm text-zinc-300">
          {summary}
        </p>
      ) : (
        <p className="rounded-lg border border-dashed border-zinc-800 bg-zinc-900/20 p-3 text-sm text-zinc-600">
          Aucun sens extrait pour ce fichier. « Modifier » pour en ajouter un.
        </p>
      )}
    </div>
  );
}

/// Panneau de détail intégré (tiroir) affiché sous le chat, redimensionnable.
export default function DetailDrawer({
  details,
  loading,
  onClose,
  onOpenFile,
  onSummarySaved,
  onDiscussFile,
}: Props) {
  return (
    <div className="flex h-full flex-col overflow-hidden bg-zinc-950">
      <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-2.5">
        <h2 className="flex min-w-0 items-center gap-2 text-sm font-semibold text-zinc-100">
          {loading || !details ? (
            <FileText size={15} className="shrink-0 text-zinc-500" />
          ) : details.is_directory ? (
            <Folder size={15} className="shrink-0 text-blue-400" />
          ) : (
            <FileText size={15} className="shrink-0 text-emerald-400" />
          )}
          <span className="truncate">{details?.name ?? "Détails"}</span>
        </h2>
        <button onClick={onClose} className="shrink-0 text-zinc-500 hover:text-zinc-300">
          <X size={16} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
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
              <Row label="Taille">
                {formatBytes(details.size_bytes)}
                {details.is_directory && details.file_count != null && (
                  <span className="text-zinc-500">
                    {" "}
                    · {details.file_count.toLocaleString()} fichier
                    {details.file_count > 1 ? "s" : ""}
                  </span>
                )}
              </Row>
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

            {(!details.is_directory || details.summary) && (
              <SenseSection
                key={details.path}
                path={details.path}
                summary={details.summary}
                canQualify={!!details.extract && !details.is_directory}
                onSaved={onSummarySaved}
              />
            )}

            {details.extract && (
              <div className="mt-3">
                <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                  Contenu extrait
                </p>
                <p className="max-h-72 overflow-y-auto whitespace-pre-wrap rounded-lg border border-zinc-800 bg-zinc-900/40 p-3 text-xs text-zinc-400">
                  {details.extract}
                </p>
              </div>
            )}

            {details.is_directory && (
              <FolderQualifyButton key={details.path} path={details.path} />
            )}

            {!details.is_directory && (
              <div className="mt-3 flex gap-2">
                <button
                  onClick={() => onDiscussFile(details.path)}
                  className="flex flex-1 items-center justify-center gap-1.5 rounded-lg border border-blue-500/40 bg-blue-500/10 px-3 py-2 text-sm font-medium text-blue-300 transition hover:bg-blue-500/20"
                  title="Ouvrir une discussion avec l'agent à propos de ce fichier"
                >
                  <Sparkles size={14} /> Discuter de ce fichier
                </button>
                <button
                  onClick={() => onOpenFile(details.path)}
                  className="flex-1 rounded-lg bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-500"
                >
                  Ouvrir le fichier
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

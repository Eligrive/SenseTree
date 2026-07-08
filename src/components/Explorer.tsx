import { useState, type FormEvent, type ReactNode } from "react";
import {
  Boxes,
  Check,
  ChevronRight,
  Clock,
  Columns2,
  File as FileIcon,
  FileText,
  FileType2,
  Folder,
  FolderTree,
  HelpCircle,
  Image as ImageIcon,
  List,
  Loader2,
  Network,
  Search,
  Sparkles,
  X,
} from "lucide-react";
import type { DirEntryInfo, ResultView, SearchResult, TreeNode } from "../lib/types";
import { breadcrumbs, formatBytes, formatDate } from "../lib/format";
import TreeView from "./TreeView";

interface Props {
  currentPath: string | null;
  entries: DirEntryInfo[];
  loading: boolean;
  onNavigate: (path: string) => void;
  onOpenFile: (path: string) => void;
  onSearch: (query: string) => void;
  searchMode: boolean;
  searchResults: SearchResult[];
  searching: boolean;
  onExitSearch: () => void;
  scopeToCurrent: boolean;
  onToggleScope: (value: boolean) => void;
  onSetFolderMode: (path: string, mode: "recursive" | "block") => void;
  treeData: TreeNode | null;
  resultView: ResultView;
  onSetResultView: (v: ResultView) => void;
}

/// Badge cliquable indiquant/commutant le mode de traitement d'un dossier.
function FolderModeBadge({
  entry,
  onSetFolderMode,
}: {
  entry: DirEntryInfo;
  onSetFolderMode: (path: string, mode: "recursive" | "block") => void;
}) {
  const mode = entry.folder_mode;

  // Pas encore classé (le crawler n'a pas encore atteint ce dossier).
  if (mode == null) {
    return (
      <button
        onClick={(e) => {
          e.stopPropagation();
          onSetFolderMode(entry.path, "block");
        }}
        title="Pas encore classé — cliquer pour le forcer en bloc sémantique"
        className="flex shrink-0 items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-zinc-700 transition hover:bg-zinc-800 hover:text-zinc-400"
      >
        <HelpCircle size={11} /> non classé
      </button>
    );
  }

  // Classification reportée (IA indisponible).
  if (mode === "pending") {
    return (
      <button
        onClick={(e) => {
          e.stopPropagation();
          onSetFolderMode(entry.path, "recursive"); // forcer l'exploration maintenant
        }}
        title="Classification reportée (IA indisponible) — cliquer pour forcer l'exploration récursive"
        className="flex shrink-0 items-center gap-1 rounded bg-amber-500/15 px-1.5 py-0.5 text-[10px] text-amber-300 transition hover:bg-amber-500/25"
      >
        <Clock size={11} /> attente
      </button>
    );
  }

  const isBlock = mode === "block";
  return (
    <button
      onClick={(e) => {
        e.stopPropagation();
        onSetFolderMode(entry.path, isBlock ? "recursive" : "block");
      }}
      title={
        isBlock
          ? "Dossier traité comme un bloc sémantique — cliquer pour l'explorer récursivement"
          : "Dossier exploré récursivement — cliquer pour le traiter comme un bloc"
      }
      className={`flex shrink-0 items-center gap-1 rounded px-1.5 py-0.5 text-[10px] transition ${
        isBlock
          ? "bg-purple-500/15 text-purple-300 hover:bg-purple-500/25"
          : "text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
      }`}
    >
      {isBlock ? <Boxes size={11} /> : <FolderTree size={11} />}
      {isBlock ? "bloc" : "récursif"}
    </button>
  );
}

/// Indicateur d'indexation unifié (fichier ou dossier-bloc) : vert = indexé,
/// ambre = en file, rouge = échec.
function IndexBadge({ entry }: { entry: DirEntryInfo }) {
  if (entry.indexed) {
    return <Check size={13} className="shrink-0 text-emerald-500" aria-label="Indexé" />;
  }
  const s = entry.index_status;
  if (s === "pending" || s === "pending_extraction") {
    return (
      <span
        title="En attente d'indexation"
        className="h-1.5 w-1.5 shrink-0 rounded-full bg-amber-400"
      />
    );
  }
  if (s === "failed" || s === "failed_permanent") {
    return (
      <span title="Échec d'indexation" className="h-1.5 w-1.5 shrink-0 rounded-full bg-rose-500" />
    );
  }
  return null;
}

function ExtIcon({ entry }: { entry: DirEntryInfo }) {
  if (entry.is_directory) return <Folder size={15} className="text-blue-400" />;
  const ext = entry.extension ?? "";
  if (["png", "jpg", "jpeg", "gif", "webp", "bmp"].includes(ext))
    return <ImageIcon size={15} className="text-fuchsia-400" />;
  if (["pdf", "docx", "doc"].includes(ext))
    return <FileType2 size={15} className="text-rose-400" />;
  if (["txt", "md", "rs", "ts", "tsx", "js", "json", "py"].includes(ext))
    return <FileText size={15} className="text-emerald-400" />;
  return <FileIcon size={15} className="text-zinc-400" />;
}

export default function Explorer({
  currentPath,
  entries,
  loading,
  onNavigate,
  onOpenFile,
  onSearch,
  searchMode,
  searchResults,
  searching,
  onExitSearch,
  scopeToCurrent,
  onToggleScope,
  onSetFolderMode,
  treeData,
  resultView,
  onSetResultView,
}: Props) {
  const [query, setQuery] = useState("");
  const crumbs = currentPath ? breadcrumbs(currentPath) : [];

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (query.trim()) onSearch(query.trim());
  };

  return (
    <div className="flex h-full flex-col bg-zinc-950">
      {/* Barre de recherche sémantique */}
      <div className="border-b border-zinc-800 px-6 pt-5 pb-4">
        <form onSubmit={submit} className="relative">
          <Search
            size={16}
            className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500"
          />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Recherche sémantique — ex : « fichiers liés à mon voyage en Corée »"
            className="w-full rounded-xl border border-zinc-800 bg-zinc-900 py-2.5 pl-10 pr-24 text-sm text-zinc-200 placeholder-zinc-500 outline-none transition focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
          />
          <button
            type="submit"
            className="absolute right-1.5 top-1/2 flex -translate-y-1/2 items-center gap-1 rounded-lg bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-500"
          >
            <Sparkles size={13} /> Chercher
          </button>
        </form>
        <label className="mt-2 flex items-center gap-1.5 text-xs text-zinc-500">
          <input
            type="checkbox"
            checked={scopeToCurrent}
            onChange={(e) => onToggleScope(e.target.checked)}
          />
          Limiter au dossier courant{" "}
          <span className="text-zinc-600">(sinon recherche globale)</span>
        </label>
      </div>

      {/* Fil d'ariane / retour recherche */}
      <div className="flex items-center gap-1 border-b border-zinc-800/60 px-6 py-2 text-sm">
        {searchMode ? (
          <>
            <button
              onClick={onExitSearch}
              className="flex items-center gap-1.5 text-zinc-400 hover:text-zinc-200"
            >
              <X size={14} /> Résultats — retour à l'explorateur
            </button>
            <div className="ml-auto flex items-center gap-0.5 rounded-lg border border-zinc-800 bg-zinc-900 p-0.5">
              {(
                [
                  ["list", List, "Liste"],
                  ["tree", Network, "Arbre"],
                  ["split", Columns2, "Côte à côte"],
                ] as const
              ).map(([mode, Icon, label]) => (
                <button
                  key={mode}
                  onClick={() => onSetResultView(mode)}
                  title={label}
                  className={`flex items-center gap-1 rounded-md px-2 py-1 text-xs transition ${
                    resultView === mode
                      ? "bg-zinc-700 text-zinc-100"
                      : "text-zinc-400 hover:text-zinc-200"
                  }`}
                >
                  <Icon size={13} />
                </button>
              ))}
            </div>
          </>
        ) : (
          <div className="flex flex-wrap items-center gap-0.5 text-zinc-400">
            {crumbs.map((c, i) => (
              <span key={c.path} className="flex items-center">
                {i > 0 && <ChevronRight size={13} className="mx-0.5 text-zinc-600" />}
                <button
                  onClick={() => onNavigate(c.path)}
                  className="rounded px-1.5 py-0.5 hover:bg-zinc-800 hover:text-zinc-200"
                >
                  {c.label}
                </button>
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Contenu */}
      {searchMode ? (
        <div className="flex-1 overflow-hidden p-4">
          {resultView === "tree" ? (
            <div className="h-full overflow-y-auto">
              <TreePane
                treeData={treeData}
                searching={searching}
                onOpenFile={onOpenFile}
                onNavigate={onNavigate}
              />
            </div>
          ) : resultView === "split" ? (
            <div className="flex h-full gap-3">
              <div className="w-1/2 overflow-y-auto pr-1">
                <SearchResults
                  results={searchResults}
                  searching={searching}
                  onOpenFile={onOpenFile}
                />
              </div>
              <div className="w-1/2 overflow-y-auto border-l border-zinc-800 pl-3">
                <TreePane
                  treeData={treeData}
                  searching={searching}
                  onOpenFile={onOpenFile}
                  onNavigate={onNavigate}
                />
              </div>
            </div>
          ) : (
            <div className="h-full overflow-y-auto">
              <SearchResults
                results={searchResults}
                searching={searching}
                onOpenFile={onOpenFile}
              />
            </div>
          )}
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto p-4">
          {loading ? (
            <Centered>
              <Loader2 className="animate-spin text-zinc-500" />
            </Centered>
          ) : entries.length === 0 ? (
            <Centered>
              <p className="text-sm text-zinc-500">Dossier vide.</p>
            </Centered>
          ) : (
          <div className="overflow-hidden rounded-lg border border-zinc-800">
            <table className="w-full text-[13px]">
              <thead>
                <tr className="border-b border-zinc-800 bg-zinc-900/50 text-left text-[10px] uppercase tracking-wider text-zinc-500">
                  <th className="px-3 py-1.5 font-medium">Nom</th>
                  <th className="w-24 px-3 py-1.5 font-medium">Taille</th>
                  <th className="w-28 px-3 py-1.5 font-medium">Modifié</th>
                </tr>
              </thead>
              <tbody>
                {entries.map((entry) => (
                  <tr
                    key={entry.path}
                    onDoubleClick={() =>
                      entry.is_directory ? onNavigate(entry.path) : onOpenFile(entry.path)
                    }
                    className="group cursor-default border-b border-zinc-800/40 last:border-0 hover:bg-zinc-900/60"
                  >
                    <td className="px-3 py-1">
                      <div className="flex items-center gap-2">
                        <span className="shrink-0">
                          <ExtIcon entry={entry} />
                        </span>
                        <span className="truncate text-zinc-200">{entry.name}</span>
                        {entry.is_directory && (
                          <FolderModeBadge entry={entry} onSetFolderMode={onSetFolderMode} />
                        )}
                        <IndexBadge entry={entry} />
                      </div>
                    </td>
                    <td className="px-3 py-1 text-zinc-500">
                      {entry.is_directory ? "" : formatBytes(entry.size_bytes)}
                    </td>
                    <td className="px-3 py-1 text-zinc-500">{formatDate(entry.modified)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          )}
        </div>
      )}
    </div>
  );
}

function SearchResults({
  results,
  searching,
  onOpenFile,
}: {
  results: SearchResult[];
  searching: boolean;
  onOpenFile: (path: string) => void;
}) {
  if (searching)
    return (
      <Centered>
        <Loader2 className="animate-spin text-zinc-500" />
      </Centered>
    );
  if (results.length === 0)
    return (
      <Centered>
        <p className="text-sm text-zinc-500">Aucun résultat pertinent.</p>
      </Centered>
    );
  return (
    <div className="space-y-2">
      {results.map((r) => (
        <button
          key={r.path}
          onDoubleClick={() => onOpenFile(r.path)}
          className="block w-full rounded-lg border border-zinc-800 bg-zinc-900/40 p-3 text-left transition hover:border-zinc-700 hover:bg-zinc-900"
        >
          <div className="flex items-center justify-between gap-3">
            <span className="truncate font-medium text-zinc-200">{r.name}</span>
            <span className="shrink-0 rounded bg-blue-500/15 px-1.5 py-0.5 text-[11px] font-medium text-blue-300">
              {(r.score * 100).toFixed(0)}%
            </span>
          </div>
          <p className="mt-0.5 truncate text-xs text-zinc-500">{r.path}</p>
          {r.snippet && <p className="mt-1.5 line-clamp-2 text-xs text-zinc-400">{r.snippet}</p>}
        </button>
      ))}
    </div>
  );
}

function TreePane({
  treeData,
  searching,
  onOpenFile,
  onNavigate,
}: {
  treeData: TreeNode | null;
  searching: boolean;
  onOpenFile: (path: string) => void;
  onNavigate: (path: string) => void;
}) {
  if (searching)
    return (
      <Centered>
        <Loader2 className="animate-spin text-zinc-500" />
      </Centered>
    );
  if (!treeData || treeData.children.length === 0)
    return (
      <Centered>
        <p className="text-sm text-zinc-500">Aucune branche pertinente.</p>
      </Centered>
    );
  return <TreeView root={treeData} onOpenFile={onOpenFile} onNavigate={onNavigate} />;
}

function Centered({ children }: { children: ReactNode }) {
  return <div className="flex h-full items-center justify-center">{children}</div>;
}

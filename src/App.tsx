import { useCallback, useEffect, useState } from "react";
import { openPath } from "@tauri-apps/plugin-opener";
import "./App.css";

import Sidebar from "./components/Sidebar";
import Explorer from "./components/Explorer";
import ChatPanel from "./components/ChatPanel";
import SettingsModal from "./components/SettingsModal";
import GardenerModal from "./components/GardenerModal";

import type {
  DirEntryInfo,
  DirectoryReport,
  HealthReport,
  IndexingStats,
  ResultView,
  SearchResult,
  TreeNode,
} from "./lib/types";
import {
  aiHealth,
  analyzeDirectory,
  getConfig,
  getRoots,
  indexingStats,
  listDirectory,
  semanticSearch,
  semanticTree,
  setFolderMode,
} from "./lib/ipc";

export default function App() {
  const [roots, setRoots] = useState<string[]>([]);
  const [currentRoot, setCurrentRoot] = useState<string | null>(null);
  const [currentPath, setCurrentPath] = useState<string | null>(null);
  const [entries, setEntries] = useState<DirEntryInfo[]>([]);
  const [loading, setLoading] = useState(false);

  const [searchMode, setSearchMode] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [treeData, setTreeData] = useState<TreeNode | null>(null);
  const [resultView, setResultView] = useState<ResultView>("list");
  // Recherche globale par défaut ; on peut la restreindre au dossier courant.
  const [scopeToCurrent, setScopeToCurrent] = useState(false);

  const [health, setHealth] = useState<HealthReport | null>(null);
  const [stats, setStats] = useState<IndexingStats | null>(null);
  const [embedding, setEmbedding] = useState<{ model: string; mode: string } | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const [report, setReport] = useState<DirectoryReport | null>(null);
  const [reportOpen, setReportOpen] = useState(false);

  const refreshHealth = useCallback(() => {
    aiHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  const refreshConfig = useCallback(() => {
    getConfig()
      .then((c) => setEmbedding({ model: c.embedding.model, mode: c.embedding.mode }))
      .catch(() => setEmbedding(null));
  }, []);

  const navigate = useCallback((path: string) => {
    setSearchMode(false);
    setCurrentPath(path);
    setLoading(true);
    listDirectory(path)
      .then(setEntries)
      .catch(() => setEntries([]))
      .finally(() => setLoading(false));
  }, []);

  // Chargement initial : racines + santé IA.
  useEffect(() => {
    getRoots().then((r) => {
      setRoots(r);
      if (r.length > 0) {
        setCurrentRoot(r[0]);
        navigate(r[0]);
      }
    });
    refreshHealth();
    refreshConfig();
    const t = setInterval(refreshHealth, 20000);
    return () => clearInterval(t);
  }, [navigate, refreshHealth, refreshConfig]);

  // Avancement de l'indexation (rafraîchi souvent tant qu'il reste des fichiers en file).
  useEffect(() => {
    const poll = () => indexingStats().then(setStats).catch(() => {});
    poll();
    const t = setInterval(poll, 2000);
    return () => clearInterval(t);
  }, []);

  // Ré-actualise la liste courante périodiquement pour refléter l'indexation en fond.
  useEffect(() => {
    if (!currentPath || searchMode) return;
    const t = setInterval(() => {
      listDirectory(currentPath).then(setEntries).catch(() => {});
    }, 5000);
    return () => clearInterval(t);
  }, [currentPath, searchMode]);

  const selectRoot = (root: string) => {
    setCurrentRoot(root);
    navigate(root);
  };

  const search = (query: string) => {
    setSearchMode(true);
    setSearching(true);
    const scope = scopeToCurrent ? currentPath ?? undefined : undefined;
    semanticSearch(query, scope, 30)
      .then(setSearchResults)
      .catch((e) => {
        console.error("semantic_search:", e);
        setSearchResults([]);
      })
      .finally(() => setSearching(false));
    // Arbre de pertinence (pour les vues « arbre » et « côte à côte »).
    semanticTree(query, scope, 500)
      .then(setTreeData)
      .catch((e) => {
        console.error("semantic_tree:", e);
        setTreeData(null);
      });
  };

  const openFile = (path: string) => {
    openPath(path).catch((e) => console.error("openPath:", e));
  };

  const changeFolderMode = (path: string, mode: "recursive" | "block") => {
    // Mise à jour optimiste : le badge change instantanément (plus de « re-clic »).
    setEntries((prev) => prev.map((e) => (e.path === path ? { ...e, folder_mode: mode } : e)));
    setFolderMode(path, mode).catch((e) => {
      console.error("set_folder_mode:", e);
      if (currentPath) navigate(currentPath); // on annule l'optimisme en cas d'échec
    });
  };

  const analyze = () => {
    if (!currentPath) return;
    setReport(null);
    setReportOpen(true);
    analyzeDirectory(currentPath)
      .then(setReport)
      .catch(() => setReportOpen(false));
  };

  return (
    <div className="flex h-screen w-full overflow-hidden bg-zinc-950 font-sans text-zinc-100">
      <Sidebar
        roots={roots}
        currentRoot={currentRoot}
        onSelectRoot={selectRoot}
        health={health}
        stats={stats}
        embedding={embedding}
        onOpenSettings={() => setSettingsOpen(true)}
        onAnalyze={analyze}
      />

      <main className="flex-1 overflow-hidden">
        <Explorer
          currentPath={currentPath}
          entries={entries}
          loading={loading}
          onNavigate={navigate}
          onOpenFile={openFile}
          onSearch={search}
          searchMode={searchMode}
          searchResults={searchResults}
          searching={searching}
          onExitSearch={() => setSearchMode(false)}
          scopeToCurrent={scopeToCurrent}
          onToggleScope={setScopeToCurrent}
          onSetFolderMode={changeFolderMode}
          treeData={treeData}
          resultView={resultView}
          onSetResultView={setResultView}
        />
      </main>

      <ChatPanel currentPath={currentPath} reasoningOk={!!health?.reasoning_ok} />

      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onSaved={() => {
          refreshHealth();
          refreshConfig();
          getRoots().then(setRoots);
        }}
      />

      <GardenerModal
        report={reportOpen ? report : null}
        loading={reportOpen && !report}
        onClose={() => setReportOpen(false)}
      />
    </div>
  );
}

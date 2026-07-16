import { useCallback, useEffect, useState } from "react";
import "./App.css";

import Sidebar from "./components/Sidebar";
import Explorer from "./components/Explorer";
import ChatPanel from "./components/ChatPanel";
import SettingsModal from "./components/SettingsModal";
import GardenerModal from "./components/GardenerModal";
import DetailDrawer from "./components/DetailDrawer";

import type {
  DirEntryInfo,
  DirectoryReport,
  HealthReport,
  IndexingStats,
  PathDetails,
  ResultView,
  SearchResult,
  TreeNode,
} from "./lib/types";
import {
  addIndexedFolder,
  aiHealth,
  analyzeDirectory,
  getConfig,
  getRoots,
  indexingStats,
  indexingPaused,
  setIndexingPaused,
  listDirectory,
  openPath,
  pathDetails,
  pickFolder,
  removeIndexedFolder,
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
  const [paused, setPaused] = useState(false);
  const [embedding, setEmbedding] = useState<{ model: string; mode: string } | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const [report, setReport] = useState<DirectoryReport | null>(null);
  const [reportOpen, setReportOpen] = useState(false);

  const [details, setDetails] = useState<PathDetails | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);

  // Dimensions redimensionnables : largeur de la colonne droite, hauteur du tiroir.
  const [rightWidth, setRightWidth] = useState(384);
  const [drawerHeight, setDrawerHeight] = useState(300);

  const startWidthDrag = useCallback((e: React.PointerEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = rightWidth;
    const onMove = (ev: PointerEvent) =>
      setRightWidth(Math.min(760, Math.max(300, startW + (startX - ev.clientX))));
    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }, [rightWidth]);

  const startHeightDrag = useCallback((e: React.PointerEvent) => {
    e.preventDefault();
    const startY = e.clientY;
    const startH = drawerHeight;
    const onMove = (ev: PointerEvent) =>
      setDrawerHeight(Math.min(window.innerHeight - 180, Math.max(140, startH + (startY - ev.clientY))));
    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }, [drawerHeight]);

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

  // État de pause (synchronisé au démarrage avec le backend).
  useEffect(() => {
    indexingPaused().then(setPaused).catch(() => {});
  }, []);

  const togglePause = useCallback(() => {
    // L'effet de bord (invoke) doit rester HORS du updater de setState : React 19
    // en StrictMode ré-exécute les updaters, ce qui dédoublerait l'appel backend.
    const next = !paused;
    setPaused(next);
    setIndexingPaused(next).catch(() => {});
  }, [paused]);

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

  // Rafraîchit la liste courante (statuts under_root/mode après un changement de racines).
  const refreshListing = useCallback(() => {
    if (currentPath && !searchMode) {
      listDirectory(currentPath).then(setEntries).catch(() => {});
    }
  }, [currentPath, searchMode]);

  // Ajoute un dossier donné à l'indexation (utilisé par l'explorateur : « + indexer »).
  const addRootPath = useCallback(
    async (path: string) => {
      try {
        setRoots(await addIndexedFolder(path));
        refreshListing();
      } catch (e) {
        console.error("add_indexed_folder:", e);
      }
    },
    [refreshListing]
  );

  // Ajoute un dossier via le sélecteur natif (bouton « Ajouter » de la barre latérale).
  const addRoot = useCallback(async () => {
    try {
      const path = await pickFolder();
      if (path) await addRootPath(path);
    } catch (e) {
      console.error("pick_folder:", e);
    }
  }, [addRootPath]);

  const removeRoot = useCallback(
    async (root: string) => {
      try {
        const next = await removeIndexedFolder(root);
        setRoots(next);
        if (currentRoot === root) {
          if (next.length > 0) selectRoot(next[0]);
          else setCurrentRoot(null);
        }
        refreshListing();
      } catch (e) {
        console.error("remove_indexed_folder:", e);
      }
    },
    [currentRoot, refreshListing]
  );

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

  // Simple-clic : ouvre le panneau de détail (double-clic = ouvrir le fichier).
  const openDetail = (path: string) => {
    setDetails(null);
    setDetailOpen(true);
    pathDetails(path)
      .then(setDetails)
      .catch((e) => {
        console.error("path_details:", e);
        setDetailOpen(false);
      });
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
        onAddRoot={addRoot}
        onRemoveRoot={removeRoot}
        health={health}
        stats={stats}
        paused={paused}
        onTogglePause={togglePause}
        embedding={embedding}
        onOpenSettings={() => setSettingsOpen(true)}
        onAnalyze={analyze}
      />

      <main className="min-w-0 flex-1 overflow-hidden">
        <Explorer
          currentPath={currentPath}
          entries={entries}
          loading={loading}
          onNavigate={navigate}
          onOpenFile={openFile}
          onSelect={openDetail}
          onSearch={search}
          searchMode={searchMode}
          searchResults={searchResults}
          searching={searching}
          onExitSearch={() => setSearchMode(false)}
          scopeToCurrent={scopeToCurrent}
          onToggleScope={setScopeToCurrent}
          onSetFolderMode={changeFolderMode}
          onAddRoot={addRootPath}
          treeData={treeData}
          resultView={resultView}
          onSetResultView={setResultView}
        />
      </main>

      {/* Poignée de redimensionnement de la colonne droite */}
      <div
        onPointerDown={startWidthDrag}
        className="w-1 shrink-0 cursor-col-resize border-l border-zinc-800 bg-transparent transition-colors hover:bg-blue-500/60"
      />

      {/* Colonne droite : chat (haut) + tiroir de détail intégré (bas) */}
      <div
        style={{ width: rightWidth }}
        className="flex shrink-0 flex-col overflow-hidden"
      >
        <div className="min-h-0 flex-1">
          <ChatPanel
            currentPath={currentPath}
            reasoningOk={!!health?.reasoning_ok}
            onOpenSource={openDetail}
          />
        </div>

        {detailOpen && (
          <div style={{ height: drawerHeight }} className="flex shrink-0 flex-col">
            {/* Poignée de redimensionnement vertical du tiroir */}
            <div
              onPointerDown={startHeightDrag}
              className="h-1 shrink-0 cursor-row-resize border-t border-zinc-800 bg-transparent transition-colors hover:bg-blue-500/60"
            />
            <div className="min-h-0 flex-1 border-t border-zinc-800">
              <DetailDrawer
                details={detailOpen ? details : null}
                loading={detailOpen && !details}
                onClose={() => setDetailOpen(false)}
                onOpenFile={openFile}
              />
            </div>
          </div>
        )}
      </div>

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

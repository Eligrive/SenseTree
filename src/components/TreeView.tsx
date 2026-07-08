import { useState } from "react";
import { ChevronRight, FileText, Folder } from "lucide-react";
import type { TreeNode } from "../lib/types";

interface Props {
  root: TreeNode;
  onOpenFile: (path: string) => void;
  onNavigate: (path: string) => void;
}

/// Vue arborescente de pertinence : la couleur/opacité guide l'œil vers les
/// branches les plus pertinentes pour la requête (score agrégé par dossier).
export default function TreeView({ root, onOpenFile, onNavigate }: Props) {
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const toggle = (path: string) =>
    setCollapsed((s) => {
      const n = new Set(s);
      n.has(path) ? n.delete(path) : n.add(path);
      return n;
    });

  if (root.children.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-zinc-500">
        Aucune branche pertinente.
      </div>
    );
  }

  return (
    <div className="space-y-0.5 text-[13px]">
      {root.children.map((c) => (
        <TreeRow
          key={c.path}
          node={c}
          depth={0}
          collapsed={collapsed}
          toggle={toggle}
          onOpenFile={onOpenFile}
          onNavigate={onNavigate}
        />
      ))}
    </div>
  );
}

function TreeRow({
  node,
  depth,
  collapsed,
  toggle,
  onOpenFile,
  onNavigate,
}: {
  node: TreeNode;
  depth: number;
  collapsed: Set<string>;
  toggle: (path: string) => void;
  onOpenFile: (path: string) => void;
  onNavigate: (path: string) => void;
}) {
  const hasChildren = node.children.length > 0;
  const isCollapsed = collapsed.has(node.path);
  const s = Math.max(0, Math.min(1, node.score));

  // Échelle séquentielle émeraude : teinte de fond + opacité selon la pertinence.
  const bg = `rgba(16, 185, 129, ${(s * 0.16).toFixed(3)})`;

  return (
    <>
      <div
        onClick={() => hasChildren && toggle(node.path)}
        onDoubleClick={() => (node.is_dir ? onNavigate(node.path) : onOpenFile(node.path))}
        title={node.path}
        className="group flex cursor-default items-center gap-1.5 rounded py-1 pr-2 transition hover:bg-zinc-800/40"
        style={{
          paddingLeft: depth * 16 + 8,
          background: bg,
          opacity: 0.55 + s * 0.45,
        }}
      >
        {hasChildren ? (
          <ChevronRight
            size={13}
            className={`shrink-0 text-zinc-500 transition-transform ${isCollapsed ? "" : "rotate-90"}`}
          />
        ) : (
          <span className="w-[13px] shrink-0" />
        )}
        {node.is_dir ? (
          <Folder size={14} className="shrink-0 text-emerald-400/80" />
        ) : (
          <FileText size={14} className="shrink-0 text-zinc-400" />
        )}
        <span className="truncate text-zinc-200">{node.name}</span>
        <div className="ml-auto flex shrink-0 items-center gap-2">
          <div className="h-1 w-10 overflow-hidden rounded-full bg-zinc-700/60">
            <div className="h-full rounded-full bg-emerald-400" style={{ width: `${s * 100}%` }} />
          </div>
          <span className="w-8 text-right text-[10px] text-zinc-500">{Math.round(s * 100)}%</span>
        </div>
      </div>
      {hasChildren &&
        !isCollapsed &&
        node.children.map((c) => (
          <TreeRow
            key={c.path}
            node={c}
            depth={depth + 1}
            collapsed={collapsed}
            toggle={toggle}
            onOpenFile={onOpenFile}
            onNavigate={onNavigate}
          />
        ))}
    </>
  );
}

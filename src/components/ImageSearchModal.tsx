import { useState } from "react";
import { Image as ImageIcon, Loader2, RefreshCw, Search, X } from "lucide-react";
import type { ImageHit } from "../lib/types";
import { imageDataUrl, imageSearch, indexImages, openPath } from "../lib/ipc";

interface Props {
  open: boolean;
  onClose: () => void;
}

/// Recherche d'images par SIMILARITÉ VISUELLE (CLIP) : une requête texte est encodée
/// dans le même espace que les images. Indexation à la demande (bouton).
export default function ImageSearchModal({ open, onClose }: Props) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<ImageHit[]>([]);
  const [thumbs, setThumbs] = useState<Record<string, string>>({});
  const [searching, setSearching] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  if (!open) return null;

  const search = async () => {
    const q = query.trim();
    if (!q || searching) return;
    setSearching(true);
    setMsg(null);
    try {
      const hits = await imageSearch(q, 30);
      setResults(hits);
      if (hits.length === 0) setMsg("Aucune image trouvée — as-tu lancé « Indexer les images » ?");
      hits.forEach((h) => {
        if (!thumbs[h.path]) {
          imageDataUrl(h.path)
            .then((url) => setThumbs((t) => ({ ...t, [h.path]: url })))
            .catch(() => {});
        }
      });
    } catch (e) {
      setMsg(`⚠️ ${String(e)}`);
    } finally {
      setSearching(false);
    }
  };

  const reindex = async () => {
    setIndexing(true);
    setMsg("Indexation visuelle des images en cours (CLIP)… (1er lancement : téléchargement du modèle)");
    try {
      const n = await indexImages();
      setMsg(`${n} image(s) indexée(s) pour la recherche visuelle.`);
    } catch (e) {
      setMsg(`⚠️ ${String(e)}`);
    } finally {
      setIndexing(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6"
      onClick={onClose}
    >
      <div
        className="flex max-h-[85vh] w-full max-w-3xl flex-col overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-950"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2 border-b border-zinc-800 px-4 py-3">
          <ImageIcon size={16} className="text-blue-400" />
          <h2 className="text-sm font-semibold text-zinc-100">Recherche d'images (visuelle)</h2>
          <button
            onClick={reindex}
            disabled={indexing}
            title="Vectoriser les images des dossiers indexés (CLIP) pour la recherche visuelle"
            className="ml-auto flex items-center gap-1 rounded px-2 py-1 text-[11px] text-zinc-300 hover:bg-zinc-800 disabled:opacity-50"
          >
            {indexing ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />} Indexer les
            images
          </button>
          <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300">
            <X size={16} />
          </button>
        </div>

        <div className="border-b border-zinc-800 p-3">
          <div className="flex items-center gap-2">
            <Search size={15} className="shrink-0 text-zinc-500" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") search();
              }}
              autoFocus
              placeholder="Décris l'image (ex. « plage au coucher de soleil », « schéma d'architecture »)…"
              className="min-w-0 flex-1 rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 outline-none focus:border-blue-500"
            />
            <button
              onClick={search}
              disabled={searching || !query.trim()}
              className="shrink-0 rounded-lg bg-blue-600 px-3 py-2 text-sm text-white hover:bg-blue-500 disabled:opacity-40"
            >
              {searching ? <Loader2 size={14} className="animate-spin" /> : "Chercher"}
            </button>
          </div>
          {msg && <p className="mt-2 text-[11px] text-zinc-400">{msg}</p>}
        </div>

        <div className="grid flex-1 grid-cols-3 gap-2 overflow-y-auto p-3 sm:grid-cols-4">
          {results.map((h) => (
            <button
              key={h.path}
              onClick={() => openPath(h.path).catch(() => {})}
              title={`${h.name} · ${Math.round(h.score * 100)}%`}
              className="group relative aspect-square overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900"
            >
              {thumbs[h.path] ? (
                <img src={thumbs[h.path]} alt={h.name} className="h-full w-full object-cover" />
              ) : (
                <div className="flex h-full items-center justify-center">
                  <Loader2 size={16} className="animate-spin text-zinc-600" />
                </div>
              )}
              <span className="absolute inset-x-0 bottom-0 truncate bg-black/60 px-1 py-0.5 text-[10px] text-zinc-200 opacity-0 transition group-hover:opacity-100">
                {h.name}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

import { useEffect, useState } from "react";
import { check, type Update } from "@tauri-apps/plugin-updater";
import { relaunch } from "@tauri-apps/plugin-process";
import { Download, Loader2, X } from "lucide-react";

/// Bannière d'auto-update : vérifie une fois au démarrage s'il existe une version
/// plus récente (signée), et propose de l'installer + redémarrer. Silencieuse s'il
/// n'y a rien, si l'endpoint est injoignable, ou hors build bundlé.
export default function UpdateBanner() {
  const [update, setUpdate] = useState<Update | null>(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    check()
      .then((u) => {
        if (u?.available) setUpdate(u);
      })
      .catch(() => {
        /* pas de MAJ, endpoint injoignable, ou mode dev : on ignore */
      });
  }, []);

  if (!update || dismissed) return null;

  const install = async () => {
    setBusy(true);
    setStatus("Téléchargement…");
    try {
      await update.downloadAndInstall((ev) => {
        if (ev.event === "Progress") setStatus("Téléchargement…");
        if (ev.event === "Finished") setStatus("Installation…");
      });
      setStatus("Redémarrage…");
      await relaunch();
    } catch (e) {
      setStatus(`⚠️ ${String(e)}`);
      setBusy(false);
    }
  };

  return (
    <div className="flex shrink-0 items-center gap-3 border-b border-blue-500/30 bg-blue-500/10 px-4 py-2 text-sm text-blue-200">
      <Download size={15} className="shrink-0" />
      <span className="min-w-0 flex-1 truncate">
        Mise à jour <strong>v{update.version}</strong> disponible.
      </span>
      {status && <span className="shrink-0 text-xs text-blue-300/80">{status}</span>}
      <button
        onClick={install}
        disabled={busy}
        className="flex shrink-0 items-center gap-1.5 rounded-md bg-blue-600 px-3 py-1 text-xs font-medium text-white transition hover:bg-blue-500 disabled:opacity-50"
      >
        {busy ? <Loader2 size={12} className="animate-spin" /> : <Download size={12} />}
        Installer et redémarrer
      </button>
      {!busy && (
        <button
          onClick={() => setDismissed(true)}
          title="Plus tard"
          className="shrink-0 text-blue-300/70 hover:text-blue-100"
        >
          <X size={15} />
        </button>
      )}
    </div>
  );
}

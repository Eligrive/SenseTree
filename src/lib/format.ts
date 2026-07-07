// Petits utilitaires d'affichage pour l'explorateur.

export function formatBytes(bytes: number): string {
  if (bytes <= 0) return "—";
  const units = ["o", "Ko", "Mo", "Go", "To"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  const value = bytes / Math.pow(1024, i);
  return `${value.toFixed(value >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
}

export function formatDate(epochSecs: number | null): string {
  if (!epochSecs) return "";
  return new Date(epochSecs * 1000).toLocaleDateString(undefined, {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}

export function parentPath(path: string): string | null {
  const norm = path.replace(/[\\/]+$/, "");
  const idx = Math.max(norm.lastIndexOf("\\"), norm.lastIndexOf("/"));
  if (idx <= 0) return null;
  // Conserve la racine du lecteur Windows (ex. "C:\").
  if (/^[A-Za-z]:$/.test(norm.slice(0, idx))) return norm.slice(0, idx + 1);
  return norm.slice(0, idx);
}

export function breadcrumbs(path: string): { label: string; path: string }[] {
  const norm = path.replace(/[\\/]+$/, "");
  const sep = norm.includes("\\") ? "\\" : "/";
  const parts = norm.split(/[\\/]+/).filter(Boolean);
  const crumbs: { label: string; path: string }[] = [];
  let acc = "";
  parts.forEach((part, i) => {
    if (i === 0) {
      // "C:" → "C:\" (lecteur Windows) ; sinon racine Unix "/xxx".
      acc = /^[A-Za-z]:$/.test(part) ? part + sep : sep + part;
    } else {
      // On n'ajoute un séparateur que si acc n'en a pas déjà un (évite "C:\\Users").
      acc = acc.endsWith(sep) ? acc + part : acc + sep + part;
    }
    crumbs.push({ label: part, path: acc });
  });
  return crumbs;
}

// Statut d'indexation → couleur du point.
export function statusColor(status: string | null): string {
  switch (status) {
    case "completed":
      return "bg-emerald-500";
    case "pending_extraction":
    case "pending":
      return "bg-amber-400";
    case "failed_permanent":
    case "failed":
      return "bg-rose-500";
    default:
      return "bg-transparent";
  }
}

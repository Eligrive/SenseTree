import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Check,
  Download,
  ExternalLink,
  Loader2,
  Lock,
  RefreshCw,
  Search,
  Star,
  X,
} from "lucide-react";
import {
  CATALOG,
  hfName,
  idForBackend,
  availabilityOf,
  type Backend,
  type CatalogModel,
  type ServerKind,
  type Task,
} from "../lib/models";
import type {
  BoardInfo,
  BoardScore,
  InstallInfo,
  ModelBenchmark,
  OllamaModel,
  OllamaTag,
} from "../lib/types";
import {
  listBenchmarkBoards,
  listReasoningBoards,
  listVisionBoards,
  modelBenchmarks,
  ollamaLibrary,
  ollamaTags,
  reasoningBenchmarks,
  resolveInstalls,
  visionBenchmarks,
} from "../lib/ipc";

/// Source de données selon la tâche : embedding (MTEB, par board), vision et
/// reasoning (OpenCompass, tout en un JSON). Tout est ramené au même format.
function listBoardsFor(t: Task): Promise<BoardInfo[]> {
  if (t === "vision") return listVisionBoards();
  if (t === "reasoning") return listReasoningBoards();
  return listBenchmarkBoards();
}
function fetchBenchFor(t: Task, boards: string[], refresh: boolean): Promise<ModelBenchmark[]> {
  if (t === "vision") return visionBenchmarks(refresh);
  if (t === "reasoning") return reasoningBenchmarks(refresh);
  return modelBenchmarks(boards, refresh);
}
const DEFAULT_BOARD: Record<Task, string[]> = {
  embedding: ["MTEB(Multilingual, v2)"],
  vision: ["Général"],
  reasoning: ["Général"],
};

/// Classements/benchmarks retenus (persistés PAR TÂCHE : langues d'embedding,
/// benchmarks de vision ou de reasoning — rien n'est figé).
const boardLsKey = (t: Task) => `sensetree.boards.${t}`;

function loadBoards(task: Task): string[] {
  try {
    const raw = localStorage.getItem(boardLsKey(task));
    const v = raw ? (JSON.parse(raw) as string[]) : null;
    return Array.isArray(v) && v.length > 0 ? v : DEFAULT_BOARD[task];
  } catch {
    return DEFAULT_BOARD[task];
  }
}

/// Un serveur connu (Ollama / LM Studio) et ses modèles installés.
export interface ServerInfo {
  url: string;
  kind: ServerKind;
  models: string[];
}

interface Props {
  open: boolean;
  onClose: () => void;
  task: Task;
  backend: Backend;
  /// Type du serveur ACTUELLEMENT configuré pour cette section.
  serverKind: ServerKind;
  /// Serveurs sondés (Ollama + LM Studio) pour savoir où chaque modèle est installé.
  servers: ServerInfo[];
  localDownloaded: Record<string, boolean>;
  currentModel: string;
  /// `serverUrl` = endpoint où le modèle est installé (bascule auto). Absent = inchangé.
  onUse: (id: string, dims?: number, serverUrl?: string) => void;
  onDownload: (id: string, targetUrl?: string) => void;
  downloading: Record<string, boolean>;
  /// Progression de téléchargement, par nom de modèle.
  progress: Record<string, { percent: number; status: string }>;
}

/// Créneau SenseTree → capacité annoncée par Ollama.
///
/// Un modèle de vision sait aussi raisonner : il reste donc listé en « reasoning ».
/// Seuls les encodeurs d'embedding sont exclus de ce créneau, faute de savoir chatter.
function matchesTask(m: OllamaModel, task: Task): boolean {
  if (task === "embedding") return m.capabilities.includes("embedding");
  if (task === "vision") return m.capabilities.includes("vision");
  return !m.capabilities.includes("embedding");
}

const norm = (x: string) => x.toLowerCase().replace(/[^a-z0-9]/g, "");

/// Rapproche un modèle du leaderboard de son entrée Ollama.
///
/// Le leaderboard nomme en Hugging Face (`Qwen/Qwen3-Embedding-0.6B`), Ollama en nom
/// court (`qwen3-embedding`). On rapproche par PRÉFIXE normalisé, en gardant le nom
/// Ollama le plus long qui convient : sans ça `qwen` capterait toute la famille Qwen.
/// Les noms trop courts sont ignorés — ils produisent surtout des faux positifs.
function matchOllama(hf: string, index: OllamaModel[]): OllamaModel | undefined {
  const short = norm(hf.split("/").pop() ?? hf);
  let best: OllamaModel | undefined;
  for (const m of index) {
    const n = norm(m.name);
    if (n.length < 5 || !short.startsWith(n)) continue;
    if (!best || n.length > norm(best.name).length) best = m;
  }
  return best;
}

/// Plus petite taille proposée, en milliards de paramètres (`300m` → 0,3).
/// Permet au filtre de taille de s'appliquer aussi aux modèles vus seulement sur Ollama.
function smallestSizeB(m: OllamaModel): number | undefined {
  const vals = m.sizes
    .map((x) => {
      const v = parseFloat(x);
      return !isFinite(v) ? NaN : x.trim().endsWith("m") ? v / 1000 : v;
    })
    .filter((v) => isFinite(v));
  return vals.length ? Math.min(...vals) : undefined;
}

/// `Aug 26, 2026 11:07 PM UTC` → `26 août 2026`, sans dépendance de date.
const MOIS_FR: Record<string, string> = {
  Jan: "janv.", Feb: "févr.", Mar: "mars", Apr: "avr.", May: "mai", Jun: "juin",
  Jul: "juil.", Aug: "août", Sep: "sept.", Oct: "oct.", Nov: "nov.", Dec: "déc.",
};
function fmtUpdated(s: string | null): string | null {
  if (!s) return null;
  const [mois, jour, annee] = s.split(/[\s,]+/);
  const m = MOIS_FR[mois];
  return m && jour && annee ? `${jour} ${m} ${annee}` : null;
}

/// Tri principal de la liste. « Benchmark » suit le classement officiel ; les deux
/// autres viennent de la bibliothèque Ollama et ne dépendent d'aucun leaderboard.
/// Runtime visé par un tag, quand ce n'est pas llama.cpp/GGUF.
///
/// `mlx` ne tourne QUE sur Apple Silicon ; `nvfp4` demande une carte Blackwell et
/// `mxfp8` un GPU FP8. On ne les cache pas — l'utilisateur peut avoir la machine qu'il
/// faut — mais on les étiquette, et `mlx` est écarté du choix automatique.
function tagRuntime(tag: string): string | null {
  const t = tag.toLowerCase();
  if (t.includes("mlx")) return "Apple Silicon";
  if (t.includes("nvfp4")) return "GPU Blackwell";
  if (t.includes("mxfp8")) return "GPU FP8";
  return null;
}

/// Une variante installable, quelle que soit sa provenance : tag Ollama ou fichier
/// GGUF d'un dépôt Hugging Face. C'est ce que le sélecteur affiche.
interface Variante {
  /// Nom complet à installer (`qwen3.5:9b-q4_K_M`, `hf.co/org/repo:Q4_K_M`).
  id: string;
  /// Ce qui distingue la variante (`9b-q4_K_M`, `Q4_K_M`).
  label: string;
  bytes: number | null;
  sizeLabel: string | null;
  /// Contexte annoncé (Ollama) ou nombre de fichiers (GGUF scindé).
  extra: string | null;
  runtime: string | null;
}

/// Variantes disponibles pour une ligne, dans l'ordre du plus léger au plus lourd.
function variantesDe(
  r: Row,
  tagsByModel: Record<string, OllamaTag[]>,
  installs: Record<string, InstallInfo>
): Variante[] {
  if (r.ol) {
    return (tagsByModel[r.ol.name] ?? [])
      .filter((t) => t.bytes != null)
      .map((t) => ({
        id: `${r.ol!.name}:${t.tag}`,
        label: t.tag,
        bytes: t.bytes,
        sizeLabel: t.size_label,
        extra: t.context,
        runtime: tagRuntime(t.tag),
      }))
      .sort((a, b) => (a.bytes ?? 0) - (b.bytes ?? 0));
  }
  const inst = installs[r.hf];
  if (!inst?.gguf_repo || !inst.quants?.length) return [];
  return inst.quants.map((q) => ({
    id: `hf.co/${inst.gguf_repo}:${q.quant}`,
    label: q.quant,
    bytes: q.bytes,
    sizeLabel: fmtGo(q.bytes),
    extra: q.parts > 1 ? `${q.parts} fichiers` : null,
    runtime: null,
  }));
}

/// Provenance d'une ligne : d'où vient l'entrée, pas où on l'installe.
function originesDe(r: Row, task: Task): string[] {
  const out: string[] = [];
  if (r.bench) {
    out.push(task === "embedding" ? "MTEB" : task === "vision" ? "OpenVLM" : "OpenCompass");
  }
  if (r.ol) out.push("Ollama");
  if (r.curated) out.push("curaté");
  return out;
}

/// Tags dont la taille est connue et qui tournent sur une carte NVIDIA ou en CPU.
function tagsUtilisables(tags: OllamaTag[]): (OllamaTag & { bytes: number })[] {
  return tags.filter(
    (t): t is OllamaTag & { bytes: number } =>
      typeof t.bytes === "number" && !t.tag.toLowerCase().includes("mlx")
  );
}

/// Verdict VRAM : le plus gros tag dont les poids tiennent dans le budget, réserve
/// déduite. `undefined` = tags pas encore chargés, `null` = aucun ne tient.
///
/// Le choix porte sur TOUS les tags, quantifications comprises : c'est ce qui permet
/// de retenir `9b-q4_K_M` (6,6 Go) au lieu d'écarter le 9 B parce que son `q8_0` pèse
/// 11 Go.
function bestFit(
  m: OllamaModel,
  tagsByModel: Record<string, OllamaTag[]>,
  vramGb: number
): { tag: string; bytes: number } | null | undefined {
  if (vramGb <= 0) return undefined;
  const connus = tagsUtilisables(tagsByModel[m.name] ?? []);
  if (connus.length === 0) return undefined;
  const budget = (vramGb - VRAM_RESERVE_GB) * GO;
  const tiennent = connus.filter((x) => x.bytes <= budget);
  if (!tiennent.length) return null;
  const meilleur = tiennent.reduce((a, b) => (b.bytes > a.bytes ? b : a));
  return { tag: meilleur.tag, bytes: meilleur.bytes };
}

/// Nom à installer pour un modèle Ollama.
///
/// Sans budget VRAM connu on laisse Ollama choisir (`latest`) ; dès qu'un tag est
/// identifié comme tenant dans la carte, on le CIBLE explicitement. Sans ça, un clic
/// sur « installer » pouvait télécharger le tag par défaut — un 27 B sur une carte de
/// 8 Go — alors que le catalogue venait d'afficher que seul le 9 B tenait.
function installName(
  m: OllamaModel,
  tagsByModel: Record<string, OllamaTag[]>,
  vramGb: number
): string {
  const fit = bestFit(m, tagsByModel, vramGb);
  return fit ? `${m.name}:${fit.tag}` : m.name;
}

/// Budgets VRAM proposés. `0` = on n'affiche aucun verdict.
const VRAM_OPTIONS = [
  { label: "VRAM : ignorer", gb: 0 },
  { label: "VRAM : 6 Go", gb: 6 },
  { label: "VRAM : 8 Go", gb: 8 },
  { label: "VRAM : 12 Go", gb: 12 },
  { label: "VRAM : 16 Go", gb: 16 },
  { label: "VRAM : 24 Go", gb: 24 },
  { label: "VRAM : 32 Go", gb: 32 },
];
const vramLsKey = "sensetree.vram";

/// Ce qu'il faut réserver EN PLUS des poids du modèle : contexte CUDA (~0,3 Go),
/// cache KV (~0,55 Go à 8K en q8_0) et buffers de calcul (~0,3 Go).
///
/// C'est une approximation calibrée sur un modèle de ~9 B : le cache KV grandit avec
/// le modèle, donc la marge est un peu optimiste sur les très gros et pessimiste sur
/// les petits. Elle n'inclut PAS ce que consomme le bureau si l'écran est branché sur
/// la même carte (0,6 à 1,2 Go de plus) — d'où un verdict volontairement prudent.
const VRAM_RESERVE_GB = 1.2;

const GO = 1_000_000_000;
const fmtGo = (bytes: number) => `${(bytes / GO).toFixed(1).replace(".", ",")} Go`;

const SORT_MODES = [
  { key: "bench", label: "Benchmark" },
  { key: "pulls", label: "Popularité" },
  { key: "recent", label: "Récence" },
] as const;
type SortMode = (typeof SORT_MODES)[number]["key"];

const TASK_LABEL: Record<Task, string> = {
  embedding: "Embedding (indexation)",
  reasoning: "Reasoning / Chat",
  vision: "Vision",
};

/// Bornes de taille : un 27 B ne tient pas sur un GPU grand public.
const SIZE_LIMITS = [
  { label: "Toutes tailles", max: Infinity },
  { label: "≤ 1 B (léger)", max: 1 },
  { label: "≤ 4 B", max: 4 },
  { label: "≤ 8 B", max: 8 },
];

const fmtScore = (x: number) => (x * 100).toFixed(1);
const fmtParams = (b: number) =>
  b >= 1 ? `${b.toFixed(1).replace(".", ",")} B` : `${Math.round(b * 1000)} M`;

/// Provenance de l'identifiant d'installation, pour être honnête sur sa fiabilité.
type IdSource =
  | "curated" // nom vérifié à la main
  | "installed" // déjà présent sur le serveur
  | "gguf" // résolu via un dépôt GGUF réel sur Hugging Face (vérifié)
  | "ollama" // présent dans la bibliothèque officielle Ollama (vérifié)
  | "closed" // modèle fermé / API-only : non installable localement
  | "guess"; // déduit du nom, non vérifié

/// Ligne unifiée : un modèle du classement, enrichi de nos infos curatées.
interface Row {
  hf: string;
  bench?: ModelBenchmark;
  curated?: CatalogModel;
  /// Entrée correspondante dans la bibliothèque Ollama (popularité, récence, tailles).
  ol?: OllamaModel;
  /// Identifiant utilisable sur le backend courant (undefined = indisponible).
  id?: string;
  source: IdSource;
  installed: boolean;
  /// Endpoint où le modèle est installé/résolu (pour basculer à la sélection).
  targetUrl?: string;
}

function Stars({ n }: { n: number }) {
  return (
    <span className="flex shrink-0 items-center gap-0.5" title={`${n}/5 — avis curaté`}>
      {[1, 2, 3, 4, 5].map((i) => (
        <Star key={i} size={11} className={i <= n ? "fill-amber-400 text-amber-400" : "text-zinc-700"} />
      ))}
    </span>
  );
}

export default function ModelCatalog({
  open,
  onClose,
  task,
  backend,
  serverKind,
  servers,
  localDownloaded,
  currentModel,
  onUse,
  onDownload,
  downloading,
  progress,
}: Props) {
  const [query, setQuery] = useState("");
  const [onlyUsable, setOnlyUsable] = useState(false);
  const [onlyOpen, setOnlyOpen] = useState(false);
  const [onlyDownloadable, setOnlyDownloadable] = useState(false);
  const [sizeLimit, setSizeLimit] = useState(0);
  const [boards, setBoards] = useState<string[]>(() => loadBoards(task));
  const [available, setAvailable] = useState<BoardInfo[]>([]);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [sortBoard, setSortBoard] = useState<string>("");
  const [bench, setBench] = useState<Record<string, ModelBenchmark>>({});
  const [loadingBench, setLoadingBench] = useState(false);
  const [benchError, setBenchError] = useState<string | null>(null);
  const [limit, setLimit] = useState(40);
  const [ollama, setOllama] = useState<OllamaModel[]>([]);
  const [sortMode, setSortMode] = useState<SortMode>("bench");
  const [vramGb, setVramGb] = useState(() => {
    const v = Number(localStorage.getItem(vramLsKey));
    return VRAM_OPTIONS.some((o) => o.gb === v) ? v : 0;
  });
  const [onlyFits, setOnlyFits] = useState(false);
  /// Tags par modèle (quantifications + tailles), chargés à la demande.
  const [modelTags, setModelTags] = useState<Record<string, OllamaTag[]>>({});
  /// Variante choisie À LA MAIN (identifiant complet), par ligne. Prime sur tout
  /// choix automatique — verdict VRAM côté Ollama, `QUANT_PREF` côté GGUF.
  const [choixInstall, setChoixInstall] = useState<Record<string, string>>({});
  /// Ligne dont le sélecteur de variante est déplié.
  const [tagsOuverts, setTagsOuverts] = useState<string | null>(null);
  // Noms d'installation résolus automatiquement (via les GGUF réels sur HF).
  const [installs, setInstalls] = useState<Record<string, InstallInfo>>({});

  // Les trois tâches ont désormais un leaderboard live (MTEB / OpenVLM / Compass).
  const hasScores = true;
  const boardKey = boards.join("|");
  const primaryBoard = sortBoard || boards[0] || "";

  useEffect(() => {
    if (open) listBoardsFor(task).then(setAvailable).catch(() => setAvailable([]));
  }, [open, task]);

  useEffect(() => {
    localStorage.setItem(boardLsKey(task), JSON.stringify(boards));
  }, [task, boardKey]);

  const fetchBench = (refresh: boolean) => {
    if (task === "embedding" && boards.length === 0) return;
    setLoadingBench(true);
    setBenchError(null);
    fetchBenchFor(task, boards, refresh)
      .then((list) => setBench(Object.fromEntries(list.map((b) => [b.name, b]))))
      .catch((e) => setBenchError(String(e)))
      .finally(() => setLoadingBench(false));
  };

  useEffect(() => {
    if (open) fetchBench(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, task, boardKey]);

  const labelOf = (b: string) => available.find((x) => x.name === b)?.display_name ?? b;

  // Serveur correspondant au type configuré pour cette section (endpoint par défaut).
  const currentServer = servers.find((s) => s.kind === serverKind) ?? servers[0];

  // Où un identifiant d'install est-il RÉELLEMENT présent (Ollama vs LM Studio) ?
  const serverHosting = (id?: string): ServerInfo | undefined => {
    if (!id) return undefined;
    const low = id.toLowerCase();
    const base = low.split(":")[0];
    return servers.find((s) =>
      s.models.some((m) => {
        const ml = m.toLowerCase();
        return ml === low || ml.split(":")[0] === base || ml.includes(base) || low.includes(ml.split(":")[0]);
      })
    );
  };

  const isInstalled = (id: string) =>
    backend === "local" ? !!localDownloaded[id] : !!serverHosting(id);

  /// Résout l'identifiant + l'endpoint cible pour le backend courant, dans l'ordre :
  /// nom vérifié → déjà installé (sur n'importe quel serveur) → GGUF résolu → deviné.
  const resolveId = (
    hf: string,
    cur?: CatalogModel
  ): { id?: string; source: IdSource; targetUrl?: string } => {
    if (backend === "local") {
      return { id: cur?.local, source: "curated" };
    }
    // 1. Nom vérifié à la main (ex. Ollama officiel `bge-m3`).
    const verified = serverKind === "lmstudio" ? cur?.lmstudio : cur?.ollama;
    if (verified) {
      return { id: verified, source: "curated", targetUrl: (serverHosting(verified) ?? currentServer)?.url };
    }
    // 2. Déjà installé sur l'UN des serveurs (Ollama ou LM Studio) → on cible celui-là.
    const short = (hf.split("/").pop() ?? hf).toLowerCase();
    for (const s of servers) {
      const hit = s.models.find(
        (m) =>
          m.toLowerCase() === short ||
          m.toLowerCase().split(":")[0] === short.split(":")[0] ||
          m.toLowerCase().includes(short)
      );
      if (hit) return { id: hit, source: "installed", targetUrl: s.url };
    }
    // 3. Résolu via un dépôt GGUF RÉEL sur Hugging Face.
    const info = installs[hf];
    if (info) {
      const rid = serverKind === "lmstudio" ? info.lmstudio : info.ollama;
      if (rid) return { id: rid, source: "gguf", targetUrl: (serverHosting(rid) ?? currentServer)?.url };
      return { id: undefined, source: "gguf" };
    }
    // 4. Dernier recours : nom déduit (en attendant la résolution).
    return { id: serverKind === "lmstudio" ? hf : short, source: "guess", targetUrl: currentServer?.url };
  };

  // Bibliothèque Ollama : source de la popularité, de la récence et de la
  // disponibilité réelle. Indépendante des leaderboards — un échec ici dégrade
  // l'affichage (pas de badge, pas de tri par pulls) sans casser le catalogue.
  useEffect(() => {
    if (!open) return;
    ollamaLibrary()
      .then(setOllama)
      .catch(() => setOllama([]));
  }, [open]);

  const curatedByHf = useMemo(() => {
    const m = new Map<string, CatalogModel>();
    for (const c of CATALOG) {
      const h = hfName(c);
      if (h) m.set(h, c);
    }
    return m;
  }, []);

  const rows: Row[] = useMemo(() => {
    const q = query.trim().toLowerCase();
    const list = Object.values(bench);
    const olForTask = ollama.filter((m) => matchesTask(m, task));

    // Repli quand le leaderboard est injoignable. Pour l'embedding, la liste curatée
    // porte les modèles EMBARQUÉS (fastembed), que rien d'autre ne connaît : elle
    // reste le bon repli. Pour reasoning/vision, la bibliothèque Ollama prend le
    // relais plus bas — on ne coupe donc pas court ici, sinon une panne de leaderboard
    // masquerait des modèles pourtant installables.
    if (list.length === 0 && (task === "embedding" || olForTask.length === 0)) {
      return CATALOG.filter((c) => c.task === task)
        .filter((c) => (q ? (c.name + c.goodFor).toLowerCase().includes(q) : true))
        .sort((a, b) => b.quality - a.quality)
        .map((c): Row => {
          const id = idForBackend(c, backend, serverKind);
          return {
            hf: c.name,
            curated: c,
            id,
            source: "curated",
            installed: !!id && isInstalled(id),
            targetUrl: id ? (serverHosting(id) ?? currentServer)?.url : undefined,
          };
        });
    }

    // La LISTE vient du leaderboard (les nouveaux modèles arrivent seuls). La clé de
    // résolution GGUF est le dépôt HF (`hf`), pas forcément le nom affiché.
    const max = SIZE_LIMITS[sizeLimit].max;
    const out: Row[] = list
      .filter((b) => (q ? b.name.toLowerCase().includes(q) : true))
      .filter((b) => (b.params_b == null ? true : b.params_b <= max))
      .map((b): Row => {
        const key = b.hf ?? b.name;
        // L'overlay curaté (note ★, conseil, nom Ollama officiel) n'existe que pour
        // l'embedding (mappé par id MTEB). Vision/reasoning : données live seules.
        const cur = task === "embedding" ? curatedByHf.get(b.name) : undefined;
        // Modèle fermé (Gemini, GPT…) : jamais installable localement.
        const r = b.closed
          ? { id: undefined, source: "closed" as IdSource, targetUrl: undefined }
          : resolveId(key, cur);
        // La bibliothèque Ollama donne un nom d'installation VÉRIFIÉ : il vaut mieux
        // qu'un nom déduit. On ne s'en sert pas sur LM Studio, qui a sa propre
        // nomenclature.
        const ol = matchOllama(key, olForTask);
        const upgraded =
          r.source === "guess" && ol && serverKind !== "lmstudio"
            ? {
                id: installName(ol, modelTags, vramGb),
                source: "ollama" as IdSource,
                targetUrl: currentServer?.url,
              }
            : r;
        return {
          hf: key,
          bench: b,
          curated: cur,
          ol,
          id: choixInstall[key] ?? upgraded.id,
          source: upgraded.source,
          targetUrl: upgraded.targetUrl,
          installed: !!upgraded.id && isInstalled(choixInstall[key] ?? upgraded.id),
        };
      })
      // Filtres (on ne CACHE rien par défaut : les modèles fermés restent visibles).
      //   open-source  : basé sur le drapeau FIABLE de la source (vision) ; pour le
      //                  reasoning on ne prétend rien → ils passent tous ce filtre.
      //   téléchargeable : un identifiant d'install a été résolu (GGUF/curaté/installé).
      .filter((r) => !onlyOpen || !r.bench?.closed)
      .filter((r) => !onlyDownloadable || !!r.id)
      .filter((r) => !onlyUsable || (r.installed && !!r.id))
      // Ne masque que ce qui est PROUVÉ trop gros : tant que les tailles ne sont pas
      // connues (`undefined`), le modèle reste visible plutôt que de disparaître.
      .filter((r) => !onlyFits || !r.ol || bestFit(r.ol, modelTags, vramGb) !== null);

    // Les modèles publiés sur Ollama mais pas encore évalués par les leaderboards —
    // c'est-à-dire précisément les plus RÉCENTS — n'apparaîtraient nulle part. On les
    // ajoute ici : c'est ce qui empêche le catalogue de vieillir.
    if (serverKind !== "lmstudio") {
      const dejaVus = new Set(out.map((r) => r.ol?.name).filter(Boolean));
      for (const m of olForTask) {
        if (dejaVus.has(m.name)) continue;
        if (q && !(m.name + " " + m.description).toLowerCase().includes(q)) continue;
        const taille = smallestSizeB(m);
        if (taille != null && taille > max) continue;
        if (onlyOpen && m.capabilities.includes("cloud") && m.sizes.length === 0) continue;
        const nom = installName(m, modelTags, vramGb);
        const installed = isInstalled(nom);
        if (onlyUsable && !installed) continue;
        if (onlyFits && bestFit(m, modelTags, vramGb) === null) continue;
        out.push({
          hf: m.name,
          ol: m,
          id: choixInstall[m.name] ?? nom,
          source: "ollama",
          installed,
          targetUrl: (serverHosting(nom) ?? currentServer)?.url,
        });
      }
    }

    const primaryOf = (r: Row) => r.bench?.scores.find((s) => s.board === primaryBoard);

    // Popularité / récence : deux axes qui viennent d'Ollama et qui fonctionnent même
    // pour les modèles qu'aucun leaderboard n'a encore évalués.
    if (sortMode !== "bench") {
      const cle = (r: Row) =>
        sortMode === "pulls" ? (r.ol?.pulls ?? -1) : (r.ol?.updated_day ?? -1);
      return out.sort((a, b) => cle(b) - cle(a) || a.hf.localeCompare(b.hf));
    }

    return out.sort((a, b) => {
      const pa = primaryOf(a);
      const pb = primaryOf(b);
      const sa = pa?.mean ?? null;
      const sb = pb?.mean ?? null;

      // Non évalués toujours en dernier — jamais assimilés à un mauvais score.
      if (sa == null && sb == null) return a.hf.localeCompare(b.hf);
      if (sa == null) return 1;
      if (sb == null) return -1;

      // On suit le RANG OFFICIEL du leaderboard, et non la moyenne : MTEB classe par
      // comptage de Borda (agrégation des rangs tâche par tâche), qui n'est
      // reproductible ni depuis `meanTask` ni depuis `meanTaskType` — le rang 4 a
      // ainsi une moyenne supérieure au rang 3. Trier sur la moyenne tout en
      // affichant le rang Borda donnait un ordre incohérent.
      const ra = pa?.rank ?? Number.MAX_SAFE_INTEGER;
      const rb = pb?.rank ?? Number.MAX_SAFE_INTEGER;
      return ra - rb;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task, backend, serverKind, query, onlyUsable, onlyOpen, onlyDownloadable, sizeLimit, primaryBoard, bench, installs, servers, localDownloaded, ollama, sortMode, onlyFits, modelTags, vramGb, choixInstall]);

  // Résolution automatique des noms d'installation pour les modèles AFFICHÉS qui
  // n'ont pas de nom vérifié (mode serveur uniquement). Bornée à la page visible,
  // et le cache backend évite de re-demander. Converge : une fois résolus, ils
  // sortent du filtre `need`.
  const shownForResolve = rows.slice(0, limit);
  useEffect(() => {
    if (backend !== "server" || !hasScores) return;
    const need = shownForResolve
      .filter((r) => r.source === "guess" && !(r.hf in installs))
      .map((r) => r.hf);
    if (need.length === 0) return;
    resolveInstalls(need)
      .then((list) =>
        setInstalls((prev) => ({
          ...prev,
          ...Object.fromEntries(list.map((i) => [i.hf, i])),
        }))
      )
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shownForResolve.map((r) => r.hf).join("|"), backend, serverKind, hasScores]);

  // Tags des modèles visibles, en UNE requête (le backend parallélise et met en cache
  // 7 jours). Chargés dès qu'un budget VRAM est choisi — pour le verdict — ou dès
  // qu'un sélecteur de quantification est déplié.
  const besoinTags = shownForResolve
    .filter((r) => !!r.ol && (vramGb > 0 || r.hf === tagsOuverts))
    .map((r) => r.ol!.name)
    .filter((n) => !(n in modelTags));
  useEffect(() => {
    if (besoinTags.length === 0) return;
    ollamaTags(besoinTags)
      .then((m) => setModelTags((prev) => ({ ...prev, ...m })))
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [besoinTags.join("|")]);

  if (!open) return null;

  const backendLabel =
    backend === "local"
      ? "Local (embarqué)"
      : serverKind === "lmstudio"
        ? "Serveur LM Studio"
        : serverKind === "ollama"
          ? "Serveur Ollama"
          : "Serveur HTTP";

  const toggleBoard = (name: string) =>
    setBoards((prev) => (prev.includes(name) ? prev.filter((b) => b !== name) : [...prev, name]));

  const shown = rows.slice(0, limit);

  const sourceLabel =
    task === "embedding"
      ? "leaderboard MTEB"
      : task === "vision"
        ? "leaderboard OpenVLM (OpenCompass)"
        : "leaderboard OpenCompass Academic";
  const boardsWord = task === "embedding" ? "Classements" : "Benchmarks";

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/70 p-6">
      <div className="flex max-h-[90vh] w-full max-w-4xl flex-col overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-950 shadow-2xl">
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3.5">
          <div>
            <h2 className="text-base font-semibold text-zinc-100">
              Catalogue de modèles — {TASK_LABEL[task]}
            </h2>
            <p className="text-[11px] text-zinc-500">
              {rows.length} modèles · {sourceLabel} · cible :{" "}
              <span className="text-zinc-300">{backendLabel}</span>
            </p>
          </div>
          <div className="flex items-center gap-2">
            {hasScores && (
              <button
                onClick={() => fetchBench(true)}
                disabled={loadingBench}
                title="Rafraîchir depuis le leaderboard MTEB officiel"
                className="flex items-center gap-1.5 rounded-lg bg-zinc-800 px-2.5 py-1.5 text-xs text-zinc-200 hover:bg-zinc-700 disabled:opacity-50"
              >
                {loadingBench ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
                MTEB
              </button>
            )}
            <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300">
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Choix des classements : global multilingue, ou TES langues. */}
        {hasScores && (
          <div className="border-b border-zinc-800 px-5 py-2.5">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[11px] text-zinc-500">{boardsWord} :</span>
              {boards.map((b) => (
                <span
                  key={b}
                  className="flex items-center gap-1 rounded bg-blue-500/15 px-1.5 py-0.5 text-[10px] text-blue-300"
                >
                  {labelOf(b)}
                  <button onClick={() => toggleBoard(b)} className="hover:text-blue-100">
                    <X size={10} />
                  </button>
                </span>
              ))}
              <button
                onClick={() => setPickerOpen((v) => !v)}
                className="rounded bg-zinc-800 px-2 py-0.5 text-[10px] text-zinc-300 hover:bg-zinc-700"
              >
                {pickerOpen
                  ? "Fermer"
                  : task === "embedding"
                    ? "+ Choisir les langues"
                    : "+ Choisir les benchmarks"}
              </button>
            </div>

            {pickerOpen && (
              <div className="mt-2 max-h-40 overflow-y-auto rounded-lg border border-zinc-800 bg-zinc-900/50 p-2">
                {available.length === 0 && (
                  <p className="text-[11px] text-zinc-600">Chargement des classements…</p>
                )}
                <div className="grid grid-cols-2 gap-1">
                  {available.map((b) => (
                    <label
                      key={b.name}
                      className="flex cursor-pointer items-center gap-1.5 rounded px-1.5 py-1 text-[11px] text-zinc-300 hover:bg-zinc-800"
                    >
                      <input
                        type="checkbox"
                        checked={boards.includes(b.name)}
                        onChange={() => toggleBoard(b.name)}
                      />
                      <span className="truncate">{b.display_name}</span>
                      <span className="ml-auto shrink-0 text-[10px] text-zinc-600">
                        {b.num_models ?? "?"}
                      </span>
                    </label>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        <div className="flex gap-2 border-b border-zinc-800 bg-amber-500/5 px-5 py-2">
          <AlertTriangle size={13} className="mt-0.5 shrink-0 text-amber-400" />
          <p className="text-[11px] leading-relaxed text-amber-200/80">
            {task === "embedding"
              ? "Un bon score dans une langue ne prédit pas les autres (les modèles anglophones chutent à ~20 en coréen contre ~67 pour un multilingue)."
              : task === "vision"
                ? "Beaucoup de VLM ne s'installent pas en GGUF (Ollama a besoin du mmproj multimodal) : privilégie llava, moondream, llama3.2-vision, qwen2.5-vl, minicpm-v."
                : "Les meilleurs du classement sont souvent des modèles fermés (o3, GPT) non installables. Filtre par taille et regarde les modèles ouverts (Qwen, Llama, Mistral, DeepSeek, Gemma)."}{" "}
            « non évalué » ≠ mauvais, et un modèle du classement{" "}
            <strong>n'est pas forcément installable</strong> — c'est signalé.
          </p>
        </div>

        {/* Filtres — flex-wrap : les filtres passent à la ligne au lieu d'écraser la recherche. */}
        <div className="flex flex-wrap items-center gap-2 border-b border-zinc-800 px-5 py-2.5">
          <div className="flex min-w-[200px] flex-1 items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900 px-2.5 py-1.5">
            <Search size={13} className="shrink-0 text-zinc-500" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Rechercher un modèle…"
              className="w-full bg-transparent text-sm text-zinc-200 outline-none placeholder:text-zinc-600"
            />
          </div>
          {hasScores && (
            <>
              <select
                value={sizeLimit}
                onChange={(e) => setSizeLimit(Number(e.target.value))}
                className="shrink-0 rounded-lg border border-zinc-800 bg-zinc-900 px-2 py-1.5 text-xs text-zinc-200 outline-none"
                title="Un 27 B ne tiendra pas sur un GPU grand public"
              >
                {SIZE_LIMITS.map((s, i) => (
                  <option key={s.label} value={i}>
                    {s.label}
                  </option>
                ))}
              </select>
              <select
                value={vramGb}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  setVramGb(v);
                  localStorage.setItem(vramLsKey, String(v));
                  if (v === 0) setOnlyFits(false);
                }}
                className="shrink-0 rounded-lg border border-zinc-800 bg-zinc-900 px-2 py-1.5 text-xs text-zinc-200 outline-none"
                title={`Compare la taille RÉELLE de chaque tag (registre Ollama) à ton budget, moins ${VRAM_RESERVE_GB.toFixed(1).replace(".", ",")} Go réservés au cache KV et aux buffers.`}
              >
                {VRAM_OPTIONS.map((o) => (
                  <option key={o.gb} value={o.gb}>
                    {o.label}
                  </option>
                ))}
              </select>
              <select
                value={sortMode}
                onChange={(e) => setSortMode(e.target.value as SortMode)}
                className="shrink-0 rounded-lg border border-zinc-800 bg-zinc-900 px-2 py-1.5 text-xs text-zinc-200 outline-none"
                title="Popularité et récence viennent de la bibliothèque Ollama, pas des benchmarks"
              >
                {SORT_MODES.map((m) => (
                  <option key={m.key} value={m.key}>
                    Trier : {m.label}
                  </option>
                ))}
              </select>
              {sortMode === "bench" && (
                <select
                  value={sortBoard}
                  onChange={(e) => setSortBoard(e.target.value)}
                  className="shrink-0 rounded-lg border border-zinc-800 bg-zinc-900 px-2 py-1.5 text-xs text-zinc-200 outline-none"
                >
                  {boards.map((b) => (
                    <option key={b} value={b}>
                      {labelOf(b)}
                    </option>
                  ))}
                </select>
              )}
            </>
          )}
          {vramGb > 0 && (
            <label
              className="flex shrink-0 items-center gap-1.5 text-xs text-zinc-400"
              title="Masque les modèles dont AUCUN tag ne tient dans le budget choisi"
            >
              <input
                type="checkbox"
                checked={onlyFits}
                onChange={(e) => setOnlyFits(e.target.checked)}
              />
              Tient en VRAM
            </label>
          )}
          <label
            className="flex shrink-0 items-center gap-1.5 text-xs text-zinc-400"
            title="Un identifiant d'installation (GGUF Ollama/LM Studio, ou nom officiel) a été trouvé"
          >
            <input
              type="checkbox"
              checked={onlyDownloadable}
              onChange={(e) => setOnlyDownloadable(e.target.checked)}
            />
            Téléchargeable
          </label>
          <label
            className="flex shrink-0 items-center gap-1.5 text-xs text-zinc-400"
            title={
              task === "vision"
                ? "D'après le drapeau open-source de la source"
                : "Fiable en vision ; pour reasoning/embedding, tout est considéré ouvert (pas de liste de noms)"
            }
          >
            <input type="checkbox" checked={onlyOpen} onChange={(e) => setOnlyOpen(e.target.checked)} />
            Open-source
          </label>
          <label className="flex shrink-0 items-center gap-1.5 text-xs text-zinc-400">
            <input
              type="checkbox"
              checked={onlyUsable}
              onChange={(e) => setOnlyUsable(e.target.checked)}
            />
            Installés
          </label>
        </div>

        {benchError && (
          <p className="border-b border-zinc-800 px-5 py-2 text-[11px] text-rose-400">
            {sourceLabel} indisponible ({benchError}).
          </p>
        )}

        {/* Liste */}
        <div className="flex-1 space-y-2 overflow-y-auto p-4">
          {loadingBench && shown.length === 0 && (
            <p className="py-8 text-center text-sm text-zinc-600">Chargement du leaderboard…</p>
          )}
          {!loadingBench && shown.length === 0 && (
            <p className="py-8 text-center text-sm text-zinc-600">Aucun modèle ne correspond.</p>
          )}

          {shown.map((r) => {
            const cur = r.curated;
            const b = r.bench;
            const inUse = !!r.id && currentModel === r.id;
            const busy = !!r.id && !!downloading[r.id];
            const dims = b?.embed_dim ?? cur?.dims;
            const params = b?.params_b ? fmtParams(b.params_b) : cur?.params;
            const displayName = cur?.name ?? r.hf.split("/")[1] ?? r.hf;
            const primary = b?.scores.find((s: BoardScore) => s.board === primaryBoard);

            return (
              <div
                key={r.hf}
                className={`rounded-xl border p-3 ${
                  inUse ? "border-blue-500/40 bg-blue-500/5" : "border-zinc-800 bg-zinc-900/30"
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      {primary?.rank != null && primary.mean != null && (
                        <span className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-[10px] text-zinc-400">
                          #{primary.rank}
                        </span>
                      )}
                      <span className="text-sm font-semibold text-zinc-100">{displayName}</span>
                      {cur && <Stars n={cur.quality} />}
                      {inUse && (
                        <span className="rounded bg-blue-500/20 px-1.5 py-0.5 text-[10px] text-blue-300">
                          utilisé
                        </span>
                      )}
                      {originesDe(r, task).map((o) => (
                        <span
                          key={o}
                          className="rounded bg-emerald-500/10 px-1.5 py-0.5 text-[10px] text-emerald-400/80"
                          title={
                            o === "Ollama"
                              ? "Listé dans la bibliothèque officielle Ollama"
                              : o === "curaté"
                                ? "Entrée écrite à la main dans le catalogue de l'app"
                                : `Classé par le leaderboard ${o}`
                          }
                        >
                          {o}
                        </span>
                      ))}
                      {!r.bench && r.ol && (
                        <span
                          className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-500"
                          title="Publié sur Ollama mais pas encore présent dans le leaderboard — typique d'un modèle récent."
                        >
                          non évalué
                        </span>
                      )}
                      {r.ol && (
                        <span
                          className="rounded bg-sky-500/10 px-1.5 py-0.5 text-[10px] text-sky-300/90"
                          title={`ollama pull ${r.ol.name}${
                            r.ol.sizes.length ? ` — tailles : ${r.ol.sizes.join(", ")}` : ""
                          }`}
                        >
                          Ollama · {r.ol.pulls_label} pulls
                          {fmtUpdated(r.ol.updated) ? ` · ${fmtUpdated(r.ol.updated)}` : ""}
                        </span>
                      )}
                      {(() => {
                        // Un choix manuel prime sur tout verdict automatique.
                        const manuel = choixInstall[r.hf];
                        if (manuel) {
                          const v = variantesDe(r, modelTags, installs).find(
                            (x) => x.id === manuel
                          );
                          return (
                            <span className="rounded bg-sky-500/15 px-1.5 py-0.5 text-[10px] text-sky-200">
                              choisi : {v?.label ?? manuel}
                              {v?.sizeLabel ? ` · ${v.sizeLabel}` : ""}
                            </span>
                          );
                        }
                        const fit = r.ol ? bestFit(r.ol, modelTags, vramGb) : undefined;
                        if (fit === undefined) return null;
                        return fit ? (
                          <span
                            className="rounded bg-emerald-500/15 px-1.5 py-0.5 text-[10px] text-emerald-300"
                            title={`ollama pull ${r.ol!.name}:${fit.tag} — ${fmtGo(fit.bytes)} de poids, mesurés sur le registre Ollama.`}
                          >
                            tient : {fit.tag} · {fmtGo(fit.bytes)}
                          </span>
                        ) : (
                          <span
                            className="rounded bg-amber-500/10 px-1.5 py-0.5 text-[10px] text-amber-400/90"
                            title={`Le plus petit tag pèse déjà plus que ${vramGb} Go moins la réserve.`}
                          >
                            trop gros pour {vramGb} Go
                          </span>
                        );
                      })()}
                      {r.ol?.capabilities
                        .filter((c) => c !== "cloud")
                        .map((c) => (
                          <span
                            key={c}
                            className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400"
                          >
                            {c}
                          </span>
                        ))}
                    </div>

                    {cur ? (
                      <p className="mt-1 text-[12px] text-zinc-300">{cur.goodFor}</p>
                    ) : (
                      r.ol?.description && (
                        <p className="mt-1 text-[12px] text-zinc-400">{r.ol.description}</p>
                      )
                    )}

                    {/* Scores officiels, un par classement choisi. */}
                    {hasScores && (
                      <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                        {boards.map((board) => {
                          const s = b?.scores.find((x) => x.board === board);
                          const scored = s?.mean != null;
                          return (
                            <span
                              key={board}
                              title={
                                scored
                                  ? `${labelOf(board)} — rang ${s!.rank}/${s!.total}`
                                  : `${labelOf(board)} — non évalué`
                              }
                              className={`rounded px-1.5 py-0.5 text-[10px] ${
                                scored ? "bg-blue-500/15 text-blue-300" : "bg-zinc-800/60 text-zinc-600"
                              }`}
                            >
                              {labelOf(board)} :{" "}
                              {scored ? <strong>{fmtScore(s!.mean!)}</strong> : "non évalué"}
                            </span>
                          );
                        })}
                      </div>
                    )}

                    {(() => {
                      // Sélecteur unifié : tags Ollama OU quantifications d'un dépôt
                      // GGUF. Dans les deux cas, l'utilisateur choisit ce qu'il
                      // télécharge au lieu de subir un défaut.
                      const variantes = variantesDe(r, modelTags, installs);
                      const ouvert = tagsOuverts === r.hf;
                      const attend = r.ol && ouvert && variantes.length === 0;
                      if (!r.ol && variantes.length === 0) return null;
                      const budget = (vramGb - VRAM_RESERVE_GB) * GO;
                      return (
                        <div className="mt-1.5">
                          <button
                            onClick={() => setTagsOuverts((prev) => (prev === r.hf ? null : r.hf))}
                            className="text-[11px] text-zinc-400 underline-offset-2 hover:text-zinc-200 hover:underline"
                          >
                            {ouvert ? "Masquer" : "Choisir"} la quantification
                            {variantes.length ? ` (${variantes.length})` : ""}
                          </button>

                          {ouvert && (
                            <div className="mt-1.5 max-h-52 overflow-y-auto rounded-lg border border-zinc-800 bg-zinc-950/60 p-1">
                              {attend ? (
                                <p className="px-2 py-1.5 text-[11px] text-zinc-600">
                                  Chargement des variantes…
                                </p>
                              ) : (
                                variantes.map((v) => {
                                  // Verdict par variante, seulement si un budget est défini.
                                  const tient =
                                    vramGb > 0 && v.bytes != null ? v.bytes <= budget : null;
                                  const actif = choixInstall[r.hf] === v.id;
                                  return (
                                    <button
                                      key={v.id}
                                      onClick={() =>
                                        setChoixInstall((prev) => {
                                          const next = { ...prev };
                                          if (actif) delete next[r.hf];
                                          else next[r.hf] = v.id;
                                          return next;
                                        })
                                      }
                                      title={v.id}
                                      className={`flex w-full items-center gap-2 rounded px-2 py-1 text-left text-[11px] ${
                                        actif
                                          ? "bg-sky-500/15 text-sky-200"
                                          : "text-zinc-400 hover:bg-zinc-800/60"
                                      }`}
                                    >
                                      <span className="font-mono">{v.label}</span>
                                      <span className="text-zinc-500">{v.sizeLabel ?? "—"}</span>
                                      {v.extra && <span className="text-zinc-600">{v.extra}</span>}
                                      {v.runtime && (
                                        <span className="rounded bg-zinc-800 px-1 py-0.5 text-[10px] text-amber-400/80">
                                          {v.runtime}
                                        </span>
                                      )}
                                      {tient === true && (
                                        <span className="ml-auto text-emerald-400/90">tient</span>
                                      )}
                                      {tient === false && (
                                        <span className="ml-auto text-amber-400/80">trop gros</span>
                                      )}
                                    </button>
                                  );
                                })
                              )}
                            </div>
                          )}
                        </div>
                      );
                    })()}

                    <div className="mt-1.5 flex flex-wrap items-center gap-3 text-[11px] text-zinc-500">
                      {params ? (
                        <span>{params}</span>
                      ) : (
                        r.ol?.sizes.length ? <span>{r.ol.sizes.join(" · ")}</span> : null
                      )}
                      {dims && <span>{dims} dims</span>}
                      {b?.max_tokens && <span>{Math.round(b.max_tokens)} tokens</span>}
                      {cur && (
                        <span className="flex items-center gap-1">
                          {availabilityOf(cur).map((bk) => (
                            <span key={bk} className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-500">
                              {bk}
                            </span>
                          ))}
                        </span>
                      )}
                      {b?.url && (
                        <a
                          href={b.url}
                          target="_blank"
                          rel="noreferrer"
                          className="flex items-center gap-0.5 text-zinc-500 hover:text-zinc-300"
                        >
                          <ExternalLink size={10} /> HF
                        </a>
                      )}
                    </div>

                    {r.id ? (
                      <p className="mt-1.5 flex items-center gap-1.5 font-mono text-[10px] text-zinc-600">
                        {r.id}
                        {r.source === "gguf" && (
                          <span
                            className="flex items-center gap-0.5 rounded bg-emerald-500/10 px-1 py-0.5 font-sans text-[10px] text-emerald-400/90"
                            title="Nom résolu automatiquement via un dépôt GGUF réel sur Hugging Face."
                          >
                            <Check size={9} /> GGUF vérifié
                          </span>
                        )}
                        {r.source === "ollama" && (
                          <span
                            className="flex items-center gap-0.5 rounded bg-sky-500/10 px-1 py-0.5 font-sans text-[10px] text-sky-300/90"
                            title="Nom publié dans la bibliothèque officielle Ollama : installable tel quel."
                          >
                            <Check size={9} /> officiel Ollama
                          </span>
                        )}
                        {r.source === "guess" && (
                          <span
                            className="flex items-center gap-0.5 rounded bg-zinc-800 px-1 py-0.5 font-sans text-[10px] text-zinc-500"
                            title="Recherche du dépôt GGUF sur Hugging Face…"
                          >
                            <Loader2 size={9} className="animate-spin" /> recherche du nom…
                          </span>
                        )}
                      </p>
                    ) : r.source === "closed" ? (
                      <p className="mt-1.5 flex items-center gap-1 text-[11px] text-zinc-500">
                        <Lock size={10} /> Modèle fermé (cloud) — non installable localement, API
                        payante uniquement.
                      </p>
                    ) : r.source === "gguf" ? (
                      <p className="mt-1.5 text-[11px] text-amber-400/80">
                        Aucun GGUF trouvé sur Hugging Face — pas installable en l'état sur «{" "}
                        {backendLabel} ».
                      </p>
                    ) : (
                      <p className="mt-1.5 flex items-center gap-1 text-[11px] text-zinc-600">
                        <Loader2 size={10} className="animate-spin" /> recherche d'un moyen
                        d'installation…
                      </p>
                    )}
                  </div>

                  {r.id && (
                    <div className="flex w-40 shrink-0 flex-col gap-1.5">
                      <button
                        onClick={() => onUse(r.id!, dims, r.targetUrl)}
                        disabled={inUse}
                        className="rounded-lg bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-500 disabled:opacity-40"
                      >
                        {inUse ? "Utilisé" : "Utiliser"}
                      </button>
                      {r.installed ? (
                        <span className="flex items-center justify-center gap-1 text-[11px] text-emerald-400">
                          <Check size={12} /> installé
                        </span>
                      ) : busy ? (
                        // Progression du téléchargement, directement dans le catalogue.
                        <div className="space-y-0.5">
                          <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
                            <div
                              className="h-full rounded-full bg-blue-500 transition-all"
                              style={{ width: `${progress[r.id]?.percent ?? 0}%` }}
                            />
                          </div>
                          <span className="block truncate text-center text-[10px] text-zinc-500">
                            {progress[r.id]?.status ?? "démarrage…"} {progress[r.id]?.percent ?? 0}%
                          </span>
                        </div>
                      ) : (
                        <button
                          onClick={() => onDownload(r.id!, r.targetUrl)}
                          disabled={busy}
                          className="flex items-center justify-center gap-1 rounded-lg bg-zinc-800 px-3 py-1.5 text-xs text-zinc-200 hover:bg-zinc-700 disabled:opacity-50"
                        >
                          <Download size={12} /> Télécharger
                        </button>
                      )}
                    </div>
                  )}
                </div>
              </div>
            );
          })}

          {rows.length > shown.length && (
            <button
              onClick={() => setLimit((l) => l + 40)}
              className="w-full rounded-lg bg-zinc-900 py-2 text-xs text-zinc-400 hover:bg-zinc-800"
            >
              Afficher plus ({rows.length - shown.length} restants)
            </button>
          )}
        </div>

        <div className="border-t border-zinc-800 px-5 py-2.5">
          <p className="text-[11px] text-zinc-600">
            La liste elle-même vient de l'<strong>API officielle du leaderboard MTEB</strong> — les
            nouveaux modèles apparaissent donc <strong>tout seuls</strong>. Les dimensions sont lues
            en direct, donc toujours justes pour l'indexation. Changer de modèle d'embedding impose
            une réindexation.
          </p>
        </div>
      </div>
    </div>
  );
}

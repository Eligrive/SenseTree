//! Classification du mode de traitement d'un dossier : `recursive` (indexé
//! fichier par fichier) ou `block` (indexé comme une seule unité sémantique).
//!
//! Approche **conservative** demandée : on ne bascule un dossier en « bloc »
//! (donc on n'explore PAS son contenu) que lorsqu'on en est sûr — soit par une
//! heuristique de motif connu, soit par une décision explicite du LLM. Dans tous
//! les autres cas (et si le LLM est indisponible), on explore récursivement, qui
//! est le choix sûr.

use std::path::Path;
use std::time::Duration;

use crate::providers::ChatMessage;
use crate::state::AppState;

/// En dessous de ce nombre d'éléments, un dossier inconnu est trivialement
/// récursif : inutile de solliciter le LLM.
const MIN_ENTRIES_FOR_LLM: usize = 6;

/// Motifs de noms de dossiers qui sont, de façon fiable, des unités-blocs.
const BLOCK_NAMES: &[&str] = &[
    "venv", "env", "virtualenv", "site-packages", "vendor", "pods", "deriveddata", "packages",
];

/// Suffixes de « bundles » applicatifs / bibliothèques → blocs.
const BLOCK_SUFFIXES: &[&str] = &[
    ".app", ".bundle", ".framework", ".plugin", ".vst", ".vst3", ".component", ".lrcat",
    ".photoslibrary", ".aplibrary", ".fcpbundle", ".imovielibrary",
];

/// Extensions FORTES et non ambiguës d'un dossier technique (presets/instruments
/// DAW, projets Ableton, banques Kontakt…) : leur présence suffit à bloquer.
const STRONG_BLOCK_EXTS: &[&str] = &[
    "adg", "adv", "als", "alp", "agr", "ask", "abl", "alc", "adm", "amxd", "nki", "nkm", "fxp", "fxb",
];

/// Fichiers « sidecar » Ableton (analyse audio) : marqueurs de dossiers de
/// samples DAW → blocs.
const DAW_SIDECAR_EXTS: &[&str] = &["asd", "ams"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FolderMode {
    Recursive,
    Block,
}

impl FolderMode {
    pub fn as_str(self) -> &'static str {
        match self {
            FolderMode::Recursive => "recursive",
            FolderMode::Block => "block",
        }
    }
    fn from_str(s: &str) -> FolderMode {
        if s == "block" {
            FolderMode::Block
        } else {
            FolderMode::Recursive
        }
    }
}

#[derive(Debug, Clone)]
pub struct EntryInfo {
    pub name: String,
    pub is_dir: bool,
    pub ext: Option<String>,
}

/// Dossiers systèmes/techniques jamais indexés (ni exploration, ni bloc).
pub fn hard_ignore(name: &str) -> bool {
    name.starts_with('.')
        || matches!(
            name,
            "node_modules" | "target" | "AppData" | "Windows" | "$RECYCLE.BIN" | "__pycache__"
        )
}

/// Échantillonne le contenu direct d'un dossier (bornage pour rester rapide).
pub fn read_dir_sample(path: &Path, limit: usize) -> Vec<EntryInfo> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(path) {
        for entry in rd.flatten().take(limit) {
            let name = entry.file_name().to_string_lossy().to_string();
            let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
            let ext = Path::new(&name)
                .extension()
                .map(|e| e.to_string_lossy().to_lowercase());
            out.push(EntryInfo { name, is_dir, ext });
        }
    }
    out
}

fn has_ext_in(entries: &[EntryInfo], set: &[&str]) -> bool {
    entries
        .iter()
        .any(|e| e.ext.as_deref().map(|x| set.contains(&x)).unwrap_or(false))
}

/// Bloc « certain » par le nom, un suffixe de bundle, ou des fichiers techniques
/// non ambigus (presets/instruments DAW, samples Ableton). Ces cas évitent un
/// appel LLM inutile ; tout le reste est tranché par le LLM.
fn heuristic_block(name: &str, entries: &[EntryInfo]) -> bool {
    let lower = name.to_lowercase();
    if BLOCK_NAMES.contains(&lower.as_str()) {
        return true;
    }
    if BLOCK_SUFFIXES.iter().any(|s| lower.ends_with(s)) {
        return true;
    }
    has_ext_in(entries, STRONG_BLOCK_EXTS) || has_ext_in(entries, DAW_SIDECAR_EXTS)
}

/// Décision de traitement d'un dossier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Recursive,
    Block,
    /// Décision REPORTÉE : l'IA est nécessaire pour trancher mais indisponible.
    /// Le dossier est marqué « en attente » et n'est PAS indexé pour l'instant ;
    /// il sera reclassé automatiquement dès que l'IA sera de nouveau joignable.
    Defer,
}

enum LlmOutcome {
    Decided(FolderMode),
    /// L'IA n'a pas pu répondre (serveur injoignable, modèle absent, timeout).
    Unavailable,
}

/// Détermine (et met en cache) la décision pour un dossier. Synchrone : appelé
/// depuis le crawler/watchdog/classifieur (threads dédiés) ; l'appel LLM éventuel
/// est borné dans le temps via `block_on` + timeout.
pub fn resolve_mode(state: &AppState, dir: &Path) -> Decision {
    let dir_str = dir.to_string_lossy().to_string();
    let cfg = state.config.snapshot();

    // Les racines configurées sont toujours explorées (jamais des blocs).
    if cfg
        .indexing
        .roots
        .iter()
        .any(|r| r.trim_end_matches(['/', '\\']) == dir_str.trim_end_matches(['/', '\\']))
    {
        return Decision::Recursive;
    }

    // Décision FERME déjà connue (heuristique/LLM/manuel). Un état 'pending' ne
    // court-circuite pas : on retente la classification (l'IA est peut-être revenue).
    if let Ok(Some((mode, _source))) = state.db.get_folder_mode(&dir_str) {
        match mode.as_str() {
            "block" => return Decision::Block,
            "recursive" => return Decision::Recursive,
            _ => {} // 'pending' → on retente ci-dessous
        }
    }

    let name = dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let entries = read_dir_sample(dir, 120);

    let (decision, source) = classify(state, &cfg, dir, &name, &entries);
    match decision {
        Decision::Block => {
            let _ = state.db.set_folder_profile(&dir_str, "block", source);
        }
        Decision::Recursive => {
            let _ = state.db.set_folder_profile(&dir_str, "recursive", source);
        }
        Decision::Defer => {
            // Marqué en attente : sera repris par le classifieur quand l'IA revient.
            let _ = state.db.set_folder_profile(&dir_str, "pending", "deferred");
        }
    }
    decision
}

fn classify(
    state: &AppState,
    cfg: &crate::config::AppConfig,
    dir: &Path,
    name: &str,
    entries: &[EntryInfo],
) -> (Decision, &'static str) {
    // 1. Dossier technique CERTAIN (venv, bundle, presets/samples DAW) → bloc,
    //    sans solliciter le LLM.
    if heuristic_block(name, entries) {
        return (Decision::Block, "heuristic");
    }
    // 2. Cas trivial (très peu d'éléments) → exploration.
    if entries.len() < MIN_ENTRIES_FOR_LLM {
        return (Decision::Recursive, "heuristic");
    }
    // 3. Reasoning désactivé : on explore (choix sûr, pas de report).
    if !cfg.reasoning.enabled {
        return (Decision::Recursive, "heuristic");
    }
    // 4. Tout le reste : le LLM tranche à partir du chemin + du contenu ; s'il est
    //    indisponible, on REPORTE (le dossier reste « en attente »).
    match llm_classify(state, dir, entries) {
        LlmOutcome::Decided(FolderMode::Block) => (Decision::Block, "llm"),
        LlmOutcome::Decided(FolderMode::Recursive) => (Decision::Recursive, "llm"),
        LlmOutcome::Unavailable => (Decision::Defer, "deferred"),
    }
}

fn llm_classify(state: &AppState, dir: &Path, entries: &[EntryInfo]) -> LlmOutcome {
    let full_path = dir.to_string_lossy();
    let parent = dir
        .parent()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    let dirs = entries.iter().filter(|e| e.is_dir).count();
    let files = entries.len() - dirs;
    let listing = entries
        .iter()
        .take(80)
        .map(|e| format!("{}{}", e.name, if e.is_dir { "/" } else { "" }))
        .collect::<Vec<_>>()
        .join(", ");

    let system = "Tu décides comment un explorateur de fichiers doit traiter un dossier : \
        'recursive' (l'explorer et indexer ses fichiers un par un) ou 'block' (le traiter comme \
        une seule unité opaque, SANS l'explorer).\n\
        Choisis 'recursive' dès qu'il y a du SENS EXPLOITABLE à l'intérieur — du contenu que \
        l'utilisateur pourrait vouloir retrouver, lire, comprendre ou manipuler : documents, \
        cours, projets, notes, code source, photos ou vidéos personnelles, etc.\n\
        Choisis 'block' UNIQUEMENT si le dossier est un ensemble applicatif/technique sans \
        intérêt à indexer fichier par fichier, c'est-à-dire un truc dont l'utilisateur ne fera \
        rien individuellement : environnement virtuel, dépendances (node_modules, vendor), bundle \
        d'application, pack d'instruments/samples (DAW), cache, artefacts de build, ou dossier ne \
        contenant que des binaires opaques.\n\
        EXTRAPOLE le rôle du dossier à partir de son CHEMIN COMPLET (le dossier parent donne un \
        contexte essentiel), de son nom, et des noms de ses fichiers et sous-dossiers. En cas de \
        doute, réponds 'recursive'.\n\
        Réponds STRICTEMENT en JSON, sans aucun texte autour : {\"mode\":\"block\"|\"recursive\"}.";
    let user = format!(
        "Chemin complet: {full_path}\nDossier parent: {parent}\n\
         Contenu: {dirs} sous-dossier(s), {files} fichier(s).\nÉléments: {listing}"
    );

    let client = state.ai.reasoning_client();
    let messages = vec![
        ChatMessage { role: "system".into(), content: system.into() },
        ChatMessage { role: "user".into(), content: user },
    ];

    let result = tauri::async_runtime::block_on(async {
        tokio::time::timeout(Duration::from_secs(25), client.chat(messages, true)).await
    });

    let raw = match result {
        Ok(Ok(s)) => s,
        // Timeout, erreur réseau, modèle absent → indisponible : on reportera.
        _ => return LlmOutcome::Unavailable,
    };

    // Parsing tolérant : on a une réponse, on en déduit le mode.
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(raw.trim()) {
        if let Some(m) = v.get("mode").and_then(|m| m.as_str()) {
            return LlmOutcome::Decided(FolderMode::from_str(m));
        }
    }
    let low = raw.to_lowercase();
    LlmOutcome::Decided(if low.contains("block") && !low.contains("recursive") {
        FolderMode::Block
    } else {
        FolderMode::Recursive
    })
}

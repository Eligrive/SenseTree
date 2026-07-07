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

/// Extensions caractéristiques de packs d'instruments/samples (Ableton, Kontakt…)
/// ou de presets : leur présence trahit un dossier technique.
const PACK_EXTS: &[&str] = &[
    "adg", "adv", "ams", "asd", "alp", "agr", "ask", "abl", "alc", "adm", "nki", "nkm", "fxp", "fxb",
];

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

/// Heuristique : renvoie true seulement si l'on est SÛR que c'est un bloc.
fn heuristic_block(name: &str, entries: &[EntryInfo]) -> bool {
    let lower = name.to_lowercase();
    if BLOCK_NAMES.contains(&lower.as_str()) {
        return true;
    }
    if BLOCK_SUFFIXES.iter().any(|s| lower.ends_with(s)) {
        return true;
    }
    // Présence de fichiers de pack (instruments/presets) → dossier technique.
    if entries
        .iter()
        .any(|e| e.ext.as_deref().map(|x| PACK_EXTS.contains(&x)).unwrap_or(false))
    {
        return true;
    }
    false
}

/// Détermine le mode d'un dossier (avec mise en cache en base). Synchrone :
/// appelé depuis le crawler/watchdog (threads dédiés) ; l'appel LLM éventuel est
/// borné dans le temps via `block_on` + timeout.
pub fn resolve_mode(state: &AppState, dir: &Path) -> FolderMode {
    let dir_str = dir.to_string_lossy().to_string();

    // Les racines configurées sont toujours explorées (jamais des blocs).
    let cfg_roots = state.config.snapshot().indexing.roots;
    if cfg_roots
        .iter()
        .any(|r| r.trim_end_matches(['/', '\\']) == dir_str.trim_end_matches(['/', '\\']))
    {
        return FolderMode::Recursive;
    }

    // 1. Décision déjà connue (heuristique/LLM/manuel) → on respecte le cache.
    if let Ok(Some((mode, _source))) = state.db.get_folder_mode(&dir_str) {
        return FolderMode::from_str(&mode);
    }

    let name = dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let entries = read_dir_sample(dir, 80);

    // 2. Bloc connu de façon fiable → bloc (on ne l'explore pas).
    if heuristic_block(&name, &entries) {
        let _ = state.db.set_folder_profile(&dir_str, "block", "heuristic");
        return FolderMode::Block;
    }

    // 3. Cas incertain non-trivial → on demande à l'IA (si disponible).
    let cfg = state.config.snapshot();
    if cfg.reasoning.enabled && entries.len() >= MIN_ENTRIES_FOR_LLM {
        if let Some(mode) = llm_classify(state, &name, &entries) {
            let _ = state.db.set_folder_profile(&dir_str, mode.as_str(), "llm");
            return mode;
        }
    }

    // 4. Défaut conservateur : on explore (on ne persiste pas, pour ré-évaluer
    //    plus tard si un LLM devient disponible).
    FolderMode::Recursive
}

fn llm_classify(state: &AppState, name: &str, entries: &[EntryInfo]) -> Option<FolderMode> {
    let listing = entries
        .iter()
        .take(60)
        .map(|e| format!("{}{}", e.name, if e.is_dir { "/" } else { "" }))
        .collect::<Vec<_>>()
        .join(", ");

    let system = "Tu classes un dossier du système de fichiers. Réponds STRICTEMENT en JSON \
        {\"mode\":\"block\"|\"recursive\"}. \
        'block' = le dossier forme une unité technique/asset cohérente qu'il vaut mieux NE PAS \
        explorer fichier par fichier (environnement virtuel, dépendances, bundle applicatif, \
        pack d'instruments/samples, cache, bibliothèque logicielle). \
        'recursive' = dossier de contenu utilisateur à indexer fichier par fichier (documents, \
        projets, photos personnelles, cours). En cas de doute, réponds 'recursive'.";
    let user = format!("Dossier: {name}\nContenu (~{} éléments): {listing}", entries.len());

    let client = state.ai.reasoning_client();
    let messages = vec![
        ChatMessage { role: "system".into(), content: system.into() },
        ChatMessage { role: "user".into(), content: user },
    ];

    let result = tauri::async_runtime::block_on(async {
        tokio::time::timeout(Duration::from_secs(20), client.chat(messages, true)).await
    });

    let raw = match result {
        Ok(Ok(s)) => s,
        _ => return None, // timeout ou erreur → l'appelant repliera sur récursif
    };

    // Parsing tolérant : on cherche le mode dans la réponse.
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(raw.trim()) {
        if let Some(m) = v.get("mode").and_then(|m| m.as_str()) {
            return Some(FolderMode::from_str(m));
        }
    }
    let low = raw.to_lowercase();
    if low.contains("block") && !low.contains("recursive") {
        Some(FolderMode::Block)
    } else {
        Some(FolderMode::Recursive)
    }
}

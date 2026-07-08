//! Commandes de l'explorateur de fichiers : listing live du système de fichiers
//! (miroir exact de l'OS) enrichi du statut d'indexation SenseTree.

use std::collections::HashMap;
use std::fs;
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use serde::Serialize;
use tauri::State;

use crate::state::AppState;

#[derive(Debug, Serialize)]
pub struct DirEntryInfo {
    pub path: String,
    pub name: String,
    pub is_directory: bool,
    pub size_bytes: u64,
    pub modified: Option<i64>,
    pub extension: Option<String>,
    /// Statut d'indexation issu de la file (`completed`, `pending_extraction`, `failed_permanent`…).
    pub index_status: Option<String>,
    /// Mode de traitement si c'est un dossier : `recursive` | `block` (ou None si non profilé).
    pub folder_mode: Option<String>,
    /// Vrai si le chemin a été effectivement indexé (embeddé).
    pub indexed: bool,
}

/// Liste le contenu direct d'un dossier, trié (dossiers d'abord, puis alphabétique).
#[tauri::command]
pub async fn list_directory(
    state: State<'_, Arc<AppState>>,
    path: String,
) -> Result<Vec<DirEntryInfo>, String> {
    let state = state.inner().clone();

    // Robustesse : on réduit les séparateurs redondants (ex. "C:\\Users" venant
    // d'un fil d'ariane), sinon les chemins ne matcheraient pas les profils stockés.
    let path = collapse_separators(&path);
    let normalized = normalize(&path);
    // Statuts d'indexation des chemins sous ce dossier, en une requête.
    let statuses: HashMap<String, String> = state
        .db
        .queue_statuses_for_parent(&normalized)
        .map_err(|e| e.to_string())?
        .into_iter()
        .collect();
    // Modes de traitement des sous-dossiers (badges bloc/récursif).
    let folder_modes: HashMap<String, String> = state
        .db
        .folder_modes_under(&normalized)
        .map_err(|e| e.to_string())?
        .into_iter()
        .collect();
    // Chemins effectivement indexés (pour l'indicateur « indexé »).
    let indexed: std::collections::HashSet<String> = state
        .db
        .indexed_paths_under(&normalized)
        .map_err(|e| e.to_string())?
        .into_iter()
        .collect();

    let mut entries: Vec<DirEntryInfo> = Vec::new();
    let read = fs::read_dir(&path).map_err(|e| format!("lecture du dossier impossible: {e}"))?;

    for entry in read.flatten() {
        let entry_path = entry.path();
        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };
        let is_dir = meta.is_dir();
        let path_str = entry_path.to_string_lossy().to_string();
        let name = entry.file_name().to_string_lossy().to_string();

        // On masque les fichiers/dossiers cachés (parité avec l'explorateur natif épuré).
        if name.starts_with('.') {
            continue;
        }

        let modified = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64);

        let extension = entry_path
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase());

        entries.push(DirEntryInfo {
            index_status: statuses.get(&path_str).cloned(),
            folder_mode: if is_dir { folder_modes.get(&path_str).cloned() } else { None },
            indexed: indexed.contains(&path_str),
            path: path_str,
            name,
            is_directory: is_dir,
            size_bytes: if is_dir { 0 } else { meta.len() },
            modified,
            extension,
        });
    }

    entries.sort_by(|a, b| match (a.is_directory, b.is_directory) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
    });

    Ok(entries)
}

/// Renvoie les dossiers racines configurés (pour la barre latérale).
#[tauri::command]
pub fn get_roots(state: State<'_, Arc<AppState>>) -> Result<Vec<String>, String> {
    Ok(state.config.snapshot().indexing.roots)
}

#[derive(Debug, Serialize)]
pub struct PathDetails {
    pub path: String,
    pub name: String,
    pub is_directory: bool,
    pub size_bytes: u64,
    pub modified: Option<i64>,
    pub extension: Option<String>,
    pub indexed: bool,
    pub status: Option<String>,
    pub last_error: Option<String>,
    pub doc_type: Option<String>,
    /// Sens extrait / aperçu (ou description vision / OCR / contexte).
    pub summary: Option<String>,
    /// Libellé lisible de la méthode d'extraction.
    pub content_kind: String,
    pub folder_mode: Option<String>,
}

/// Détails d'un fichier/dossier pour le panneau ouvert au simple-clic.
#[tauri::command]
pub async fn path_details(
    state: State<'_, Arc<AppState>>,
    path: String,
) -> Result<PathDetails, String> {
    let state = state.inner().clone();
    let p = std::path::Path::new(&path);

    let meta = fs::metadata(&path).ok();
    let is_directory = meta.as_ref().map(|m| m.is_dir()).unwrap_or(false);
    let size_bytes = meta
        .as_ref()
        .map(|m| if m.is_dir() { 0 } else { m.len() })
        .unwrap_or(0);
    let modified = meta
        .as_ref()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64);
    let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| path.clone());
    let extension = p.extension().map(|e| e.to_string_lossy().to_lowercase());

    let indexed = state.db.is_indexed(&path).map_err(|e| e.to_string())?;
    let (status, last_error) = match state.db.get_queue_status(&path).map_err(|e| e.to_string())? {
        Some((s, e)) => (Some(s), e),
        None => (None, None),
    };
    let (summary, doc_type) = match state.db.get_file_semantics(&path).map_err(|e| e.to_string())? {
        Some((s, d)) => (
            if s.is_empty() { None } else { Some(s) },
            if d.is_empty() { None } else { Some(d) },
        ),
        None => (None, None),
    };
    let hash = state.db.get_stored_hash(&path).map_err(|e| e.to_string())?;
    let folder_mode = if is_directory {
        state.db.get_folder_mode(&path).map_err(|e| e.to_string())?.map(|(m, _)| m)
    } else {
        None
    };

    let content_kind = derive_content_kind(&hash, &doc_type, indexed, &status);

    Ok(PathDetails {
        path,
        name,
        is_directory,
        size_bytes,
        modified,
        extension,
        indexed,
        status,
        last_error,
        doc_type,
        summary,
        content_kind,
        folder_mode,
    })
}

fn derive_content_kind(
    hash: &Option<String>,
    doc_type: &Option<String>,
    indexed: bool,
    status: &Option<String>,
) -> String {
    if !indexed {
        return match status.as_deref() {
            Some("pending") | Some("pending_extraction") => "En file d'attente".to_string(),
            Some("failed") | Some("failed_permanent") => "Échec d'indexation".to_string(),
            _ => "Non indexé".to_string(),
        };
    }
    if let Some(h) = hash {
        if h.starts_with("block:") {
            return "Bloc sémantique".to_string();
        }
        if h.starts_with("ctx:") {
            return match doc_type.as_deref() {
                Some(k) if !k.is_empty() => format!("Contexte ({k})"),
                _ => "Contexte".to_string(),
            };
        }
    }
    match doc_type.as_deref() {
        Some("pdf-ocr") => "OCR (vision)".to_string(),
        Some("image") => "Image décrite (vision)".to_string(),
        Some("llm-extrait") => "Contenu extrait par le LLM".to_string(),
        _ => "Contenu extrait".to_string(),
    }
}

fn normalize(path: &str) -> String {
    let trimmed = path.trim_end_matches(['/', '\\']);
    format!("{trimmed}{}", std::path::MAIN_SEPARATOR)
}

/// Réduit les séquences de séparateurs à un seul, en préservant un éventuel
/// préfixe UNC (`\\serveur`). Évite les mismatches type "C:\\Users".
fn collapse_separators(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    let mut prev_sep = false;
    for (i, ch) in path.chars().enumerate() {
        let is_sep = ch == '/' || ch == '\\';
        if is_sep && prev_sep && i > 1 {
            continue; // séparateur redondant (on garde un éventuel préfixe UNC)
        }
        out.push(ch);
        prev_sep = is_sep;
    }
    out
}

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
}

/// Liste le contenu direct d'un dossier, trié (dossiers d'abord, puis alphabétique).
#[tauri::command]
pub async fn list_directory(
    state: State<'_, Arc<AppState>>,
    path: String,
) -> Result<Vec<DirEntryInfo>, String> {
    let state = state.inner().clone();

    let normalized = normalize(&path);
    // Statuts d'indexation des chemins sous ce dossier, en une requête.
    let statuses: HashMap<String, String> = state
        .db
        .queue_statuses_for_parent(&normalized)
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

fn normalize(path: &str) -> String {
    let trimmed = path.trim_end_matches(['/', '\\']);
    format!("{trimmed}{}", std::path::MAIN_SEPARATOR)
}

//! Recherche sémantique : embedding de la requête → plus proches voisins LanceDB
//! → vérification d'existence disque → dédoublonnage par fichier.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use serde::Serialize;
use tauri::State;

use crate::state::AppState;

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub path: String,
    pub name: String,
    pub score: f32,
    pub snippet: String,
}

#[tauri::command]
pub async fn semantic_search(
    state: State<'_, Arc<AppState>>,
    query: String,
    scope: Option<String>,
    limit: Option<usize>,
) -> Result<Vec<SearchResult>, String> {
    // On récupère l'Arc et on relâche le guard avant tout `.await`.
    let state = state.inner().clone();
    let limit = limit.unwrap_or(20).clamp(1, 100);

    let query = query.trim().to_string();
    if query.is_empty() {
        return Ok(Vec::new());
    }

    let embedder = state.ai.embedder().await.map_err(|e| e.to_string())?;
    let query_vec = embedder.embed_query(query).await.map_err(|e| e.to_string())?;

    // On sur-échantillonne (plusieurs chunks par fichier) avant dédoublonnage.
    let raw = state
        .vector
        .search(query_vec, limit * 4, scope.as_deref())
        .await
        .map_err(|e| e.to_string())?;

    // Meilleur score par fichier + vérification que le fichier existe toujours.
    let mut best: HashMap<String, SearchResult> = HashMap::new();
    for hit in raw {
        if !Path::new(&hit.path).exists() {
            continue;
        }
        let name = Path::new(&hit.path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| hit.path.clone());

        best.entry(hit.path.clone())
            .and_modify(|existing| {
                if hit.score > existing.score {
                    existing.score = hit.score;
                    existing.snippet = hit.snippet.clone();
                }
            })
            .or_insert(SearchResult {
                path: hit.path.clone(),
                name,
                score: hit.score,
                snippet: hit.snippet,
            });
    }

    let mut results: Vec<SearchResult> = best.into_values().collect();
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);
    Ok(results)
}

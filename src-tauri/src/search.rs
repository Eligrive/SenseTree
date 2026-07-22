//! Recherche sémantique : embedding de la requête → plus proches voisins LanceDB
//! → vérification d'existence disque → dédoublonnage par fichier.

use std::collections::{BTreeMap, HashMap};
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

/// Score de similarité BRUT (cosinus), simplement borné à [0,1].
/// On ne remappe PAS depuis une bande arbitraire : la distribution des scores
/// varie fortement d'un modèle d'embedding à l'autre, donc toute bande fixe
/// (type 0.40→1) fausserait les pourcentages. On affiche le cosinus tel quel.
fn relevance(cosine: f32) -> f32 {
    cosine.clamp(0.0, 1.0)
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

    // Meilleur score (absolu) par fichier existant.
    let mut best: HashMap<String, SearchResult> = HashMap::new();
    for hit in raw {
        if !Path::new(&hit.path).exists() {
            continue;
        }
        let name = Path::new(&hit.path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| hit.path.clone());

        let score = relevance(hit.score);
        best.entry(hit.path.clone())
            .and_modify(|existing| {
                if score > existing.score {
                    existing.score = score;
                    existing.snippet = hit.snippet.clone();
                }
            })
            .or_insert(SearchResult {
                path: hit.path.clone(),
                name,
                score,
                snippet: hit.snippet,
            });
    }

    let mut results: Vec<SearchResult> = best.into_values().collect();
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);
    Ok(results)
}

// =========================================================================
// VUE ARBORESCENTE SÉMANTIQUE (heatmap de pertinence)
// =========================================================================

/// Un nœud de l'arbre de pertinence. Les dossiers agrègent le meilleur score de
/// leurs descendants ; l'arbre ne contient que les branches pertinentes.
#[derive(Debug, Serialize)]
pub struct TreeNode {
    pub name: String,
    pub path: String,
    pub is_dir: bool,
    pub score: f32,
    pub children: Vec<TreeNode>,
}

struct Builder {
    name: String,
    path: String,
    is_dir: bool,
    score: f32,
    children: BTreeMap<String, Builder>,
}

impl Builder {
    fn new(name: String, path: String, is_dir: bool) -> Self {
        Builder { name, path, is_dir, score: 0.0, children: BTreeMap::new() }
    }
    fn into_node(self) -> TreeNode {
        let mut children: Vec<TreeNode> = self.children.into_values().map(|b| b.into_node()).collect();
        // Enfants triés par pertinence décroissante (guide l'œil).
        children.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        TreeNode { name: self.name, path: self.path, is_dir: self.is_dir, score: self.score, children }
    }
}

/// Construit un arbre focalisé sur les fichiers/dossiers les plus pertinents pour
/// la requête, sous `scope` (ou la première racine configurée).
#[tauri::command]
pub async fn semantic_tree(
    state: State<'_, Arc<AppState>>,
    query: String,
    scope: Option<String>,
    limit: Option<usize>,
) -> Result<Option<TreeNode>, String> {
    let state = state.inner().clone();
    let query = query.trim().to_string();
    if query.is_empty() {
        return Ok(None);
    }
    let limit = limit.unwrap_or(400).clamp(10, 1000);

    let root = scope
        .filter(|s| !s.trim().is_empty())
        .or_else(|| state.config.snapshot().indexing.roots.first().cloned())
        .unwrap_or_default();
    let root = root.trim_end_matches(['/', '\\']).to_string();
    if root.is_empty() {
        return Ok(None);
    }

    let embedder = state.ai.embedder().await.map_err(|e| e.to_string())?;
    let query_vec = embedder.embed_query(query).await.map_err(|e| e.to_string())?;
    let hits = state
        .vector
        .search(query_vec, limit, Some(&root))
        .await
        .map_err(|e| e.to_string())?;

    // Meilleur score (absolu) par fichier existant.
    let mut best: HashMap<String, f32> = HashMap::new();
    for h in hits {
        if !Path::new(&h.path).exists() {
            continue;
        }
        let s = relevance(h.score);
        best.entry(h.path).and_modify(|v| { if s > *v { *v = s; } }).or_insert(s);
    }
    if best.is_empty() {
        return Ok(None);
    }

    let root_name = Path::new(&root)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| root.clone());
    let mut builder = Builder::new(root_name, root.clone(), true);

    for (path, score) in best {
        let leaf_is_dir = Path::new(&path).is_dir();
        let rel = path.strip_prefix(&root).unwrap_or(&path);
        let comps: Vec<&str> = rel.split(['/', '\\']).filter(|s| !s.is_empty()).collect();
        if comps.is_empty() {
            continue;
        }
        if score > builder.score {
            builder.score = score;
        }
        let mut node = &mut builder;
        let mut acc = root.clone();
        let last = comps.len() - 1;
        for (i, comp) in comps.iter().enumerate() {
            acc.push(std::path::MAIN_SEPARATOR);
            acc.push_str(comp);
            let is_dir = if i == last { leaf_is_dir } else { true };
            let path_here = acc.clone();
            node = node
                .children
                .entry(comp.to_string())
                .or_insert_with(|| Builder::new(comp.to_string(), path_here, is_dir));
            if score > node.score {
                node.score = score;
            }
        }
    }

    Ok(Some(builder.into_node()))
}

#[cfg(test)]
mod tests {
    use super::relevance;

    #[test]
    fn relevance_est_le_cosinus_borne() {
        assert!((relevance(0.5) - 0.5).abs() < 1e-6); // brut, pas de remapping par bande
        assert_eq!(relevance(-0.3), 0.0);             // négatif -> 0
        assert_eq!(relevance(1.5), 1.0);              // > 1 -> 1
    }
}

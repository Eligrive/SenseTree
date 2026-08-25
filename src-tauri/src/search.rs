//! Recherche sémantique : embedding de la requête → plus proches voisins LanceDB
//! → vérification d'existence disque → dédoublonnage par fichier.

use std::collections::{BTreeMap, HashMap, HashSet};
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

/// Sigmoïde : convertit un logit de reranker (non borné) en score 0–1 affichable.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Constante de Reciprocal Rank Fusion (60 = valeur de référence de la littérature).
const RRF_K: f32 = 60.0;

/// Candidat au niveau CHUNK, avant reranking et dédoublonnage par fichier.
struct Candidate {
    path: String,
    text: String,
    snippet: String,
    /// Similarité cosinus dense, si le chunk a été trouvé par la recherche vectorielle.
    dense: Option<f32>,
    /// Score de fusion RRF (dense + BM25) — sert au classement hybride.
    rrf: f32,
}

/// Récupère les candidats : dense seul, ou HYBRIDE (dense + BM25 fusionnés par RRF)
/// selon la config. Trié par score de fusion décroissant.
async fn retrieve_candidates(
    state: &AppState,
    query: &str,
    scope: Option<&str>,
    pool: usize,
) -> Result<Vec<Candidate>, String> {
    let embedder = state.ai.embedder().await.map_err(|e| e.to_string())?;
    let qvec = embedder
        .embed_query(query.to_string())
        .await
        .map_err(|e| e.to_string())?;
    let dense = state
        .vector
        .search(qvec, pool, scope)
        .await
        .map_err(|e| e.to_string())?;

    // Mode dense pur (hybride désactivé) : comportement historique conservé.
    if !state.config.snapshot().retrieval.hybrid {
        return Ok(dense
            .into_iter()
            .map(|h| Candidate {
                dense: Some(relevance(h.score)),
                rrf: relevance(h.score),
                path: h.path,
                text: h.text,
                snippet: h.snippet,
            })
            .collect());
    }

    // BM25 (best-effort : index pas prêt → on continue en dense seul).
    let sparse = state
        .vector
        .keyword_search(query, pool, scope)
        .await
        .unwrap_or_default();

    // Reciprocal Rank Fusion : score = Σ 1/(k + rang). Indépendant de l'échelle des
    // scores (cosinus vs BM25) — c'est tout l'intérêt du RRF.
    let mut fused: HashMap<(String, i32), Candidate> = HashMap::new();
    for (rank, h) in dense.into_iter().enumerate() {
        let contrib = 1.0 / (RRF_K + rank as f32 + 1.0);
        fused
            .entry((h.path.clone(), h.chunk_index))
            .and_modify(|c| {
                c.rrf += contrib;
                c.dense = Some(relevance(h.score));
            })
            .or_insert(Candidate {
                dense: Some(relevance(h.score)),
                rrf: contrib,
                path: h.path,
                text: h.text,
                snippet: h.snippet,
            });
    }
    for (rank, h) in sparse.into_iter().enumerate() {
        let contrib = 1.0 / (RRF_K + rank as f32 + 1.0);
        fused
            .entry((h.path.clone(), h.chunk_index))
            .and_modify(|c| c.rrf += contrib)
            .or_insert(Candidate {
                dense: None,
                rrf: contrib,
                path: h.path,
                text: h.text,
                snippet: h.snippet,
            });
    }

    let mut out: Vec<Candidate> = fused.into_values().collect();
    out.sort_by(|a, b| b.rrf.partial_cmp(&a.rrf).unwrap_or(std::cmp::Ordering::Equal));
    Ok(out)
}

#[tauri::command]
pub async fn semantic_search(
    state: State<'_, Arc<AppState>>,
    query: String,
    scope: Option<String>,
    limit: Option<usize>,
) -> Result<Vec<SearchResult>, String> {
    let state = state.inner().clone();
    let limit = limit.unwrap_or(20).clamp(1, 100);
    let query = query.trim().to_string();
    if query.is_empty() {
        return Ok(Vec::new());
    }

    // Bassin de candidats généreux avant reranking / dédoublonnage.
    let pool = (limit * 6).clamp(30, 160);
    let cands = retrieve_candidates(&state, &query, scope.as_deref(), pool).await?;
    if cands.is_empty() {
        return Ok(Vec::new());
    }

    // Reranking cross-encoder du haut du panier (best-effort : indisponible → on
    // conserve l'ordre de fusion). Le score affiché devient la sigmoïde du logit.
    let cfg = state.config.snapshot();
    let rerank_pool = (limit * 3).clamp(20, 100);
    let mut rerank_logits: Option<Vec<f32>> = None;
    if cfg.retrieval.rerank {
        let top = cands.len().min(rerank_pool);
        if let Ok(rr) = state.ai.reranker().await {
            let docs: Vec<String> = cands[..top].iter().map(|c| c.text.clone()).collect();
            if let Ok(order) = rr.rerank(query.clone(), docs).await {
                let mut logits = vec![f32::MIN; top];
                for (idx, logit) in order {
                    if idx < top {
                        logits[idx] = logit;
                    }
                }
                rerank_logits = Some(logits);
            }
        }
    }

    // Score de rang + score d'affichage (0–1) par candidat.
    let max_rrf = cands.iter().map(|c| c.rrf).fold(f32::MIN, f32::max).max(1e-6);
    struct Scored {
        path: String,
        snippet: String,
        rank: f32,
        display: f32,
    }
    let considered = rerank_logits.as_ref().map(|l| l.len()).unwrap_or(cands.len());
    let mut scored: Vec<Scored> = Vec::with_capacity(considered);
    for (i, c) in cands.into_iter().enumerate().take(considered) {
        let (rank, display) = match &rerank_logits {
            Some(logits) => (logits[i], sigmoid(logits[i])),
            None => (c.rrf, c.dense.unwrap_or((c.rrf / max_rrf).clamp(0.0, 1.0))),
        };
        scored.push(Scored { path: c.path, snippet: c.snippet, rank, display });
    }
    scored.sort_by(|a, b| b.rank.partial_cmp(&a.rank).unwrap_or(std::cmp::Ordering::Equal));

    // Dédoublonnage par fichier existant : meilleur chunk par chemin.
    let mut results: Vec<SearchResult> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for s in scored {
        if !seen.insert(s.path.clone()) {
            continue;
        }
        if !Path::new(&s.path).exists() {
            continue;
        }
        let name = Path::new(&s.path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| s.path.clone());
        results.push(SearchResult { path: s.path, name, score: s.display, snippet: s.snippet });
        if results.len() >= limit {
            break;
        }
    }
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
    use super::{relevance, sigmoid};

    #[test]
    fn relevance_est_le_cosinus_borne() {
        assert!((relevance(0.5) - 0.5).abs() < 1e-6); // brut, pas de remapping par bande
        assert_eq!(relevance(-0.3), 0.0);             // négatif -> 0
        assert_eq!(relevance(1.5), 1.0);              // > 1 -> 1
    }

    #[test]
    fn sigmoid_borne_et_monotone() {
        // Point milieu : logit 0 -> 0.5.
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        // Bornes : reste dans ]0,1[ et croît avec le logit.
        assert!(sigmoid(-8.0) < 0.001 && sigmoid(-8.0) > 0.0);
        assert!(sigmoid(8.0) > 0.999 && sigmoid(8.0) < 1.0);
        assert!(sigmoid(1.0) > sigmoid(-1.0)); // strictement monotone
    }
}

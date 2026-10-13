//! Recherche sémantique : embedding de la requête → plus proches voisins LanceDB
//! → vérification d'existence disque → dédoublonnage par fichier.
//!
//! Les étapes de classement (fusion RRF, reranking, dédoublonnage) sont des fonctions
//! pures, partagées avec le banc de mesure `examples/retrieval_eval.rs` : les scores
//! publiés sont ceux du code exécuté par l'application.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use serde::Serialize;
use tauri::State;

use crate::state::AppState;
use crate::vectordb::SearchHit;

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

/// Longueur max (caractères) d'un document envoyé au reranker. Les cross-encoders
/// tronquent de toute façon vers ~512 tokens : borner en amont réduit la latence sur
/// les gros chunks sans perte notable (l'entête d'un chunk est le plus pertinent).
pub const RERANK_DOC_CHARS: usize = 800;

/// Bassin de chunks demandé à chaque branche (dense, BM25) avant fusion.
pub fn candidate_pool(limit: usize) -> usize {
    (limit * 6).clamp(30, 160)
}

/// Nombre de candidats fusionnés soumis au cross-encoder.
pub fn rerank_pool(limit: usize) -> usize {
    (limit * 3).clamp(20, 100)
}

/// Candidat au niveau CHUNK, avant reranking et dédoublonnage par fichier.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub path: String,
    pub chunk_index: i32,
    pub text: String,
    pub snippet: String,
    /// Similarité cosinus dense, si le chunk a été trouvé par la recherche vectorielle.
    pub dense: Option<f32>,
    /// Score de fusion RRF (dense + BM25) — sert au classement hybride.
    pub rrf: f32,
}

/// Ordre total : score décroissant, puis chemin et index de chunk. Les égalités de
/// RRF sont fréquentes (deux chunks au même rang dans des branches différentes) ;
/// sans ce départage, l'ordre dépendait de l'itération d'une HashMap et variait
/// d'une exécution à l'autre.
fn sort_candidates(cands: &mut [Candidate]) {
    cands.sort_by(|a, b| {
        b.rrf
            .partial_cmp(&a.rrf)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.chunk_index.cmp(&b.chunk_index))
    });
}

/// Mode dense pur (hybride désactivé) : le cosinus sert de score de fusion.
pub fn dense_candidates_only(dense: Vec<SearchHit>) -> Vec<Candidate> {
    let mut out: Vec<Candidate> = dense
        .into_iter()
        .map(|h| Candidate {
            dense: Some(relevance(h.score)),
            rrf: relevance(h.score),
            path: h.path,
            chunk_index: h.chunk_index,
            text: h.text,
            snippet: h.snippet,
        })
        .collect();
    sort_candidates(&mut out);
    out
}

/// Reciprocal Rank Fusion : score = Σ 1/(k + rang). Indépendant de l'échelle des
/// scores (cosinus vs BM25) — c'est tout l'intérêt du RRF. Les entrées doivent être
/// triées par pertinence décroissante.
pub fn fuse_rrf(dense: Vec<SearchHit>, sparse: Vec<SearchHit>) -> Vec<Candidate> {
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
                chunk_index: h.chunk_index,
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
                chunk_index: h.chunk_index,
                text: h.text,
                snippet: h.snippet,
            });
    }
    let mut out: Vec<Candidate> = fused.into_values().collect();
    sort_candidates(&mut out);
    out
}

/// Textes envoyés au cross-encoder : les `pool` premiers candidats, tronqués.
pub fn rerank_inputs(cands: &[Candidate], pool: usize) -> Vec<String> {
    cands[..cands.len().min(pool)]
        .iter()
        .map(|c| c.text.chars().take(RERANK_DOC_CHARS).collect())
        .collect()
}

/// Réaligne la sortie du reranker (triée par score) sur l'ordre des candidats.
pub fn logits_by_position(order: Vec<(usize, f32)>, top: usize) -> Vec<f32> {
    let mut logits = vec![f32::MIN; top];
    for (idx, logit) in order {
        if idx < top {
            logits[idx] = logit;
        }
    }
    logits
}

/// Classement final puis meilleur chunk par fichier. Si le reranker a tourné, les
/// candidats rerankés passent en tête dans l'ordre de ses logits, suivis des autres
/// dans l'ordre de fusion : on peut ainsi reranker moins de candidats qu'on n'en
/// affiche sans perdre de résultats. `keep` écarte les chemins à ignorer (fichier
/// disparu du disque).
pub fn finalize(
    cands: Vec<Candidate>,
    rerank_logits: Option<Vec<f32>>,
    limit: usize,
    keep: impl Fn(&str) -> bool,
) -> Vec<SearchResult> {
    let max_rrf = cands.iter().map(|c| c.rrf).fold(f32::MIN, f32::max).max(1e-6);
    struct Scored {
        path: String,
        snippet: String,
        rank: f32,
        display: f32,
    }
    let reranked = rerank_logits.as_ref().map(|l| l.len()).unwrap_or(0);
    let mut scored: Vec<Scored> = Vec::with_capacity(cands.len());
    for (i, c) in cands.into_iter().enumerate() {
        let fused = c.dense.unwrap_or((c.rrf / max_rrf).clamp(0.0, 1.0));
        let (rank, display) = match &rerank_logits {
            Some(logits) if i < reranked => (logits[i], sigmoid(logits[i])),
            // Queue non rerankée : sous toute la tête, ordre de fusion conservé par le
            // tri stable.
            Some(_) => (f32::NEG_INFINITY, fused),
            None => (c.rrf, fused),
        };
        scored.push(Scored { path: c.path, snippet: c.snippet, rank, display });
    }
    scored.sort_by(|a, b| b.rank.partial_cmp(&a.rank).unwrap_or(std::cmp::Ordering::Equal));
    // Un score affiché ne doit jamais remonter le long de la liste : la queue est
    // plafonnée au dernier score de la tête.
    if reranked > 0 && scored.len() > reranked {
        let floor = scored[..reranked].iter().map(|s| s.display).fold(f32::MAX, f32::min);
        for s in &mut scored[reranked..] {
            s.display = s.display.min(floor);
        }
    }

    let mut results: Vec<SearchResult> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for s in scored {
        if !seen.insert(s.path.clone()) {
            continue;
        }
        if !keep(&s.path) {
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
    results
}

/// Branche dense : embedding de la requête puis plus proches voisins LanceDB.
/// Isolée pour que son échec (serveur d'embedding injoignable, fragment LanceDB
/// disparu…) soit RATTRAPABLE par la branche mots-clés au lieu de tout annuler.
async fn dense_hits(
    state: &AppState,
    query: &str,
    scope: Option<&str>,
    pool: usize,
) -> Result<Vec<SearchHit>, String> {
    let embedder = state.ai.embedder().await.map_err(|e| e.to_string())?;
    let qvec = embedder
        .embed_query(query.to_string())
        .await
        .map_err(|e| format!("embedding de la requête impossible : {e}"))?;
    state
        .vector
        .search(qvec, pool, scope)
        .await
        .map_err(|e| format!("recherche vectorielle impossible : {e}"))
}

/// Récupère les candidats : dense seul, ou HYBRIDE (dense + BM25 fusionnés par RRF)
/// selon la config. Trié par score de fusion décroissant.
async fn retrieve_candidates(
    state: &AppState,
    query: &str,
    scope: Option<&str>,
    pool: usize,
) -> Result<Vec<Candidate>, String> {
    let hybride = state.config.snapshot().retrieval.hybrid;

    // L'échec dense n'est pas fatal en mode hybride : on le retient pour ne le
    // remonter que si les mots-clés ne sauvent pas la mise.
    let (dense, panne_dense) = match dense_hits(state, query, scope, pool).await {
        Ok(hits) => (hits, None),
        Err(e) if hybride => {
            tracing::warn!("recherche dense en panne, repli sur BM25 seul : {e}");
            (Vec::new(), Some(e))
        }
        // Mode dense pur : aucun repli possible, l'erreur EST le résultat.
        Err(e) => return Err(e),
    };

    if !hybride {
        return Ok(dense_candidates_only(dense));
    }

    // BM25 (best-effort : index pas prêt → on continue en dense seul).
    let sparse = state
        .vector
        .keyword_search(query, pool, scope)
        .await
        .unwrap_or_default();

    // Les deux branches sont muettes ET le dense était en panne : c'est une panne,
    // pas une absence de résultat. On remonte le motif pour que l'UI le dise, au
    // lieu de laisser croire à un index vide.
    if let Some(motif) = panne_dense {
        if sparse.is_empty() {
            return Err(motif);
        }
        tracing::warn!("résultats DÉGRADÉS : mots-clés seuls, la recherche dense est en panne");
    }

    Ok(fuse_rrf(dense, sparse))
}

#[tauri::command]
pub async fn semantic_search(
    state: State<'_, Arc<AppState>>,
    query: String,
    scope: Option<String>,
    limit: Option<usize>,
) -> Result<Vec<SearchResult>, String> {
    let state = state.inner().clone();
    run_semantic_search(&state, &query, scope.as_deref(), limit.unwrap_or(20)).await
}

/// Cœur de la recherche hybride + reranking, réutilisable hors commande Tauri
/// (l'agent l'appelle comme outil `search_files`, et le pré-RAG du chat s'en sert).
pub(crate) async fn run_semantic_search(
    state: &AppState,
    query: &str,
    scope: Option<&str>,
    limit: usize,
) -> Result<Vec<SearchResult>, String> {
    let limit = limit.clamp(1, 100);
    let query = query.trim().to_string();
    if query.is_empty() {
        return Ok(Vec::new());
    }

    let cands = retrieve_candidates(state, &query, scope, candidate_pool(limit)).await?;
    if cands.is_empty() {
        return Ok(Vec::new());
    }

    // Reranking cross-encoder du haut du panier (best-effort : indisponible → on
    // conserve l'ordre de fusion). Le score affiché devient la sigmoïde du logit.
    let mut logits: Option<Vec<f32>> = None;
    if state.config.snapshot().retrieval.rerank {
        if let Ok(rr) = state.ai.reranker().await {
            let docs = rerank_inputs(&cands, rerank_pool(limit));
            let top = docs.len();
            if let Ok(order) = rr.rerank(query.clone(), docs).await {
                logits = Some(logits_by_position(order, top));
            }
        }
    }

    Ok(finalize(cands, logits, limit, |p| Path::new(p).exists()))
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

    // MÊME pipeline que la liste de résultats — hybride puis reranking — et non un
    // dense brut. Sans ça les deux vues notent le même fichier sur deux échelles
    // incomparables (sigmoïde du reranker d'un côté, cosinus de l'autre) : la vue
    // « côte à côte » affichait 97 % à gauche et 46 % à droite pour la même ligne.
    let hits = run_semantic_search(&state, &query, Some(&root), limit).await?;

    // Meilleur score par fichier (l'existence et le dédoublonnage sont déjà faits).
    let mut best: HashMap<String, f32> = HashMap::new();
    for h in hits {
        best.entry(h.path)
            .and_modify(|v| {
                if h.score > *v {
                    *v = h.score;
                }
            })
            .or_insert(h.score);
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
    use super::*;

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

    fn hit(path: &str, chunk: i32, score: f32) -> SearchHit {
        SearchHit {
            path: path.into(),
            chunk_index: chunk,
            score,
            text: format!("{path}#{chunk}"),
            snippet: String::new(),
        }
    }

    #[test]
    fn rrf_favorise_un_chunk_present_dans_les_deux_branches() {
        let dense = vec![hit("a", 0, 0.9), hit("b", 0, 0.8)];
        let sparse = vec![hit("b", 0, 12.0), hit("c", 0, 9.0)];
        let fused = fuse_rrf(dense, sparse);
        let ordre: Vec<&str> = fused.iter().map(|c| c.path.as_str()).collect();
        assert_eq!(ordre, ["b", "a", "c"]);
        assert!((fused[0].rrf - (1.0 / 62.0 + 1.0 / 61.0)).abs() < 1e-6);
        assert_eq!(fused[2].dense, None);
    }

    #[test]
    fn rrf_departage_les_egalites_de_facon_deterministe() {
        // « z » est 1er en dense, « a » 1er en BM25 : même score RRF.
        for _ in 0..20 {
            let fused = fuse_rrf(vec![hit("z", 0, 0.9)], vec![hit("a", 0, 3.0)]);
            assert_eq!(fused[0].path, "a");
            assert_eq!(fused[1].path, "z");
        }
    }

    #[test]
    fn finalize_garde_le_meilleur_chunk_par_fichier() {
        let cands = fuse_rrf(vec![hit("a", 0, 0.9), hit("a", 1, 0.8), hit("b", 0, 0.7)], vec![]);
        let res = finalize(cands, None, 10, |_| true);
        let chemins: Vec<&str> = res.iter().map(|r| r.path.as_str()).collect();
        assert_eq!(chemins, ["a", "b"]);
    }

    #[test]
    fn finalize_place_la_tete_rerankee_avant_la_queue_de_fusion() {
        let cands = fuse_rrf(
            vec![hit("a", 0, 0.9), hit("b", 0, 0.8), hit("c", 0, 0.7), hit("d", 0, 0.6)],
            vec![],
        );
        let logits = logits_by_position(vec![(1, 4.0), (0, -2.0)], 2);
        let res = finalize(cands, Some(logits), 10, |_| true);
        let chemins: Vec<&str> = res.iter().map(|r| r.path.as_str()).collect();
        assert_eq!(chemins, ["b", "a", "c", "d"]);
        assert!((res[0].score - sigmoid(4.0)).abs() < 1e-6);
        assert!(res[2].score <= res[1].score && res[3].score <= res[2].score);
    }

    #[test]
    fn finalize_ecarte_les_fichiers_disparus() {
        let cands = fuse_rrf(vec![hit("a", 0, 0.9), hit("b", 0, 0.8)], vec![]);
        let res = finalize(cands, None, 10, |p| p != "a");
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].path, "b");
    }

    #[test]
    fn bassins_de_candidats() {
        assert_eq!(candidate_pool(30), 160);
        assert_eq!(candidate_pool(1), 30);
        assert_eq!(rerank_pool(30), 90);
        assert_eq!(rerank_pool(1), 20);
    }
}

//! Scores et specs des modèles d'embedding, via l'API OFFICIELLE du leaderboard MTEB.
//!
//! Source : `https://mteb-leaderboard-backend.hf.space/v1` (le backend FastAPI qui
//! alimente https://huggingface.co/spaces/mteb/leaderboard). Elle fournit les
//! agrégats officiels, les rangs, les specs et la LISTE des modèles — donc tout se
//! met à jour tout seul, y compris à l'arrivée de nouveaux modèles.
//!
//! Les classements sont CHOISIS PAR L'UTILISATEUR (pas de langue codée en dur) :
//! un « global » multilingue existe — `MTEB(Multilingual, v2)`, 1037 langues — à
//! côté des classements par langue (anglais, français, coréen, allemand…).
//!
//! Deux pièges, traités explicitement :
//!   * `mean = None` signifie NON ÉVALUÉ, jamais « zéro ». L'API renvoie 0.0 dans ce
//!     cas ; confondre les deux ferait passer un modèle non testé pour un mauvais.
//!   * un bon score dans une langue ne dit rien des autres (mesuré : les modèles
//!     anglophones tombent à ~0,2 en coréen contre ~0,67 pour un vrai multilingue).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const API: &str = "https://mteb-leaderboard-backend.hf.space/v1";

/// Classements réels mais non listés par l'API (`displayOnLeaderboard=false`) : il
/// faut les demander par leur nom. Ce sont pourtant les plus utiles (global + anglais).
const HIDDEN_BOARDS: &[&str] = &["MTEB(Multilingual, v2)", "MTEB(eng, v2)"];

const TTL_SECS: i64 = 7 * 24 * 3600;

/// Un classement disponible (pour que l'utilisateur choisisse).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoardInfo {
    pub name: String,
    pub display_name: String,
    pub num_models: Option<u32>,
    pub languages: usize,
}

/// Score d'un modèle sur un classement. `mean: None` = NON ÉVALUÉ (pas zéro).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoardScore {
    pub board: String,
    pub mean: Option<f64>,
    pub retrieval: Option<f64>,
    pub rank: Option<u32>,
    pub total: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBenchmark {
    /// Nom d'affichage, ex. `Qwen/Qwen3-Embedding-0.6B` (embedding) ou `Qwen2.5-VL-7B`.
    pub name: String,
    /// Identifiant Hugging Face pour la résolution GGUF (souvent = `name` en embedding,
    /// extrait des métadonnées en vision/reasoning). None si inconnu.
    #[serde(default)]
    pub hf: Option<String>,
    pub url: Option<String>,
    /// Dimension du vecteur — doit correspondre à la config d'indexation.
    pub embed_dim: Option<usize>,
    pub params_b: Option<f64>,
    pub max_tokens: Option<f64>,
    pub scores: Vec<BoardScore>,
}

// --- Cache : une entrée par classement, pour ne re-télécharger que le nécessaire ---

#[derive(Serialize, Deserialize, Default)]
struct Cache {
    boards: HashMap<String, CachedBoard>,
}

#[derive(Serialize, Deserialize, Clone)]
struct CachedBoard {
    fetched_at: i64,
    total: u32,
    models: Vec<CachedModel>,
}

#[derive(Serialize, Deserialize, Clone)]
struct CachedModel {
    name: String,
    url: Option<String>,
    embed_dim: Option<usize>,
    params_b: Option<f64>,
    max_tokens: Option<f64>,
    mean: Option<f64>,
    retrieval: Option<f64>,
    rank: Option<u32>,
}

// --- Formes de réponse de l'API ---------------------------------------------------

#[derive(Deserialize)]
struct ApiScores {
    rows: Vec<ApiRow>,
}

#[derive(Deserialize)]
struct ApiRow {
    rank: Option<u32>,
    model: ApiModel,
    #[serde(rename = "embeddingDim")]
    embedding_dim: Option<f64>,
    #[serde(rename = "totalParamsB")]
    total_params_b: Option<f64>,
    #[serde(rename = "maxTokens")]
    max_tokens: Option<f64>,
    #[serde(rename = "meanTask")]
    mean_task: Option<f64>,
    #[serde(rename = "scoresByTaskType")]
    scores_by_task_type: Option<HashMap<String, f64>>,
}

#[derive(Deserialize)]
struct ApiModel {
    name: String,
    url: Option<String>,
}

#[derive(Deserialize)]
struct ApiBoard {
    name: String,
    #[serde(rename = "displayName")]
    display_name: Option<String>,
    #[serde(rename = "numModels")]
    num_models: Option<u32>,
    #[serde(default)]
    languages: Vec<String>,
    #[serde(default)]
    modalities: Vec<String>,
}

// ----------------------------------------------------------------------------------

fn client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .user_agent("SenseTree")
        .timeout(Duration::from_secs(60))
        .build()?)
}

fn cache_path(data_dir: &Path) -> PathBuf {
    data_dir.join("benchmarks.json")
}

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// Encodage d'un nom de classement : `MTEB(fra, v1)` → `MTEB%28fra%2C%20v1%29`.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Liste les classements de TEXTE disponibles (les modalités image/audio/vidéo sont
/// écartées : elles n'ont pas de sens pour l'indexation de documents).
pub async fn list_boards() -> Result<Vec<BoardInfo>> {
    let http = client()?;
    let mut boards: Vec<ApiBoard> = http
        .get(format!("{API}/benchmarks"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await
        .context("listing des classements")?;

    // Les classements les plus utiles ne sont PAS listés : on les ajoute par leur nom.
    for name in HIDDEN_BOARDS {
        if boards.iter().any(|b| b.name == *name) {
            continue;
        }
        if let Ok(resp) = http.get(format!("{API}/benchmarks/{}", urlencode(name))).send().await {
            if let Ok(b) = resp.json::<ApiBoard>().await {
                boards.push(b);
            }
        }
    }

    let mut out: Vec<BoardInfo> = boards
        .into_iter()
        // Texte PUR uniquement : un classement image-texte (MIEB…) contient bien « text »
        // dans ses modalités mais n'a aucun sens pour indexer des documents.
        .filter(|b| b.modalities.is_empty() || b.modalities == ["text"])
        .map(|b| BoardInfo {
            display_name: b.display_name.unwrap_or_else(|| b.name.clone()),
            name: b.name,
            num_models: b.num_models,
            languages: b.languages.len(),
        })
        .collect();

    // Le global d'abord, puis par nombre de modèles évalués (les plus fournis en tête).
    out.sort_by(|a, b| {
        let ga = a.name.contains("Multilingual");
        let gb = b.name.contains("Multilingual");
        gb.cmp(&ga).then(b.num_models.cmp(&a.num_models))
    });
    Ok(out)
}

/// Récupère les scores des `boards` demandés, en ne re-téléchargeant que ceux dont
/// le cache est périmé. Hors ligne : on sert le cache même daté plutôt que rien.
pub async fn load(data_dir: &Path, boards: Vec<String>, refresh: bool) -> Result<Vec<ModelBenchmark>> {
    let mut cache: Cache = std::fs::read_to_string(cache_path(data_dir))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    let http = client()?;
    let mut dirty = false;

    for board in &boards {
        let fresh = cache
            .boards
            .get(board)
            .map(|c| now() - c.fetched_at < TTL_SECS)
            .unwrap_or(false);
        if fresh && !refresh {
            continue;
        }
        match fetch_board(&http, board).await {
            Ok(cb) => {
                cache.boards.insert(board.clone(), cb);
                dirty = true;
            }
            Err(e) => {
                // On garde l'ancienne entrée si elle existe : mieux vaut daté que vide.
                tracing::warn!("benchmarks : {board} indisponible ({e})");
            }
        }
    }

    if dirty {
        if let Ok(s) = serde_json::to_string(&cache) {
            let _ = std::fs::create_dir_all(data_dir);
            let _ = std::fs::write(cache_path(data_dir), s);
        }
    }

    // Fusion : un modèle, plusieurs classements.
    let mut by_model: HashMap<String, ModelBenchmark> = HashMap::new();
    for board in &boards {
        let Some(cb) = cache.boards.get(board) else { continue };
        for m in &cb.models {
            let e = by_model.entry(m.name.clone()).or_insert_with(|| ModelBenchmark {
                hf: Some(m.name.clone()),
                name: m.name.clone(),
                url: m.url.clone(),
                embed_dim: None,
                params_b: None,
                max_tokens: None,
                scores: Vec::new(),
            });
            if e.embed_dim.is_none() {
                e.embed_dim = m.embed_dim;
            }
            if e.params_b.is_none() {
                e.params_b = m.params_b;
            }
            if e.max_tokens.is_none() {
                e.max_tokens = m.max_tokens;
            }
            e.scores.push(BoardScore {
                board: board.clone(),
                mean: m.mean,
                retrieval: m.retrieval,
                rank: m.rank,
                total: Some(cb.total),
            });
        }
    }

    Ok(by_model.into_values().collect())
}

async fn fetch_board(http: &reqwest::Client, board: &str) -> Result<CachedBoard> {
    let table: ApiScores = http
        .get(format!("{API}/benchmarks/{}/scores", urlencode(board)))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await
        .with_context(|| format!("parsing du classement {board}"))?;

    let total = table.rows.len() as u32;
    // L'API renvoie 0.0 pour « non évalué » : on le convertit en None, sans quoi un
    // modèle simplement non testé passerait pour nul.
    let nz = |v: Option<f64>| v.filter(|x| *x > 0.0);

    let models = table
        .rows
        .into_iter()
        .map(|r| CachedModel {
            name: r.model.name,
            url: r.model.url,
            embed_dim: r.embedding_dim.map(|d| d as usize),
            params_b: r.total_params_b.filter(|p| *p > 0.0),
            max_tokens: r.max_tokens,
            mean: nz(r.mean_task),
            retrieval: nz(r
                .scores_by_task_type
                .as_ref()
                .and_then(|m| m.get("Retrieval").copied())),
            rank: r.rank,
        })
        .collect();

    Ok(CachedBoard { fetched_at: now(), total, models })
}

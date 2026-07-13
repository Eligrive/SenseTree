//! Specs et scores de référence des modèles, tirés en direct de MTEB.
//!
//! Source : dépôt `embeddings-benchmark/results` (vivant, mis à jour en continu).
//! Les tables pré-calculées de l'ancien dépôt `leaderboard` sont ARCHIVÉES (fév. 2025)
//! et le dataset HF (Parquet, ~288 Mo) est infetchable côté client — d'où cette approche.
//!
//! ATTENTION au piège : les tâches multilingues (ex. MIRACL) sont évaluées sur des
//! sous-ensembles de LANGUES DIFFÉRENTS selon les modèles (Qwen3 → thaï seulement,
//! bge-m3 → russe/persan/thaï). Les agréger donnerait un classement FAUX. Seules les
//! tâches de retrieval ANGLAIS ci-dessous sont réellement comparables entre modèles.
//! Le multilinguisme est donc rapporté par les LANGUES DÉCLARÉES, pas par un score.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const GH_API: &str = "https://api.github.com/repos/embeddings-benchmark/results/contents/results";
const GH_RAW: &str =
    "https://raw.githubusercontent.com/embeddings-benchmark/results/main/results";

/// Tâches de retrieval anglais présentes chez TOUS les modèles du catalogue,
/// avec un score unique et comparable (vérifié).
const EN_TASKS: &[&str] = &["NFCorpus", "SciFact", "ArguAna", "SCIDOCS"];

/// Durée de validité du cache (les résultats MTEB bougent lentement).
const TTL_SECS: i64 = 7 * 24 * 3600;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBenchmark {
    /// Identifiant dans le dépôt MTEB (ex. `Qwen__Qwen3-Embedding-0.6B`).
    pub mteb_id: String,
    /// Dimension du vecteur — critique : doit correspondre à la config d'indexation.
    pub embed_dim: Option<usize>,
    pub n_parameters: Option<u64>,
    pub max_tokens: Option<f64>,
    pub memory_mb: Option<f64>,
    /// Langues déclarées par le modèle (codes ISO, ex. `fra-Latn`).
    pub languages: Vec<String>,
    /// Le modèle déclare-t-il le français ?
    pub french: bool,
    /// Moyenne ndcg@10 sur les tâches de retrieval ANGLAIS (comparable entre modèles).
    pub retrieval_en: Option<f64>,
    /// Nombre de tâches ayant contribué au score (transparence).
    pub retrieval_tasks: usize,
}

#[derive(Serialize, Deserialize)]
struct Cache {
    fetched_at: i64,
    entries: Vec<ModelBenchmark>,
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

/// Renvoie les specs/scores, depuis le cache s'il est frais, sinon en les récupérant.
/// En cas d'échec réseau, on sert le cache même périmé plutôt que de ne rien afficher.
pub async fn load(data_dir: &Path, ids: Vec<String>, refresh: bool) -> Result<Vec<ModelBenchmark>> {
    let cached: Option<Cache> = std::fs::read_to_string(cache_path(data_dir))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok());

    if !refresh {
        if let Some(c) = &cached {
            if now() - c.fetched_at < TTL_SECS {
                return Ok(c.entries.clone());
            }
        }
    }

    match fetch_all(&ids).await {
        Ok(entries) if !entries.is_empty() => {
            let c = Cache { fetched_at: now(), entries: entries.clone() };
            if let Ok(s) = serde_json::to_string_pretty(&c) {
                let _ = std::fs::create_dir_all(data_dir);
                let _ = std::fs::write(cache_path(data_dir), s);
            }
            Ok(entries)
        }
        Ok(_) | Err(_) if cached.is_some() => {
            tracing::warn!("benchmarks : récupération échouée, cache (périmé) servi");
            Ok(cached.map(|c| c.entries).unwrap_or_default())
        }
        Ok(v) => Ok(v),
        Err(e) => Err(e),
    }
}

async fn fetch_all(ids: &[String]) -> Result<Vec<ModelBenchmark>> {
    let http = reqwest::Client::builder()
        .user_agent("SenseTree")
        .timeout(Duration::from_secs(20))
        .build()?;

    let mut out = Vec::new();
    for id in ids {
        match fetch_one(&http, id).await {
            Ok(b) => out.push(b),
            // Un modèle absent de MTEB (ex. LLM de chat) n'est pas une erreur fatale.
            Err(e) => tracing::debug!("benchmarks : {id} ignoré ({e})"),
        }
    }
    Ok(out)
}

async fn fetch_one(http: &reqwest::Client, id: &str) -> Result<ModelBenchmark> {
    #[derive(Deserialize)]
    struct Entry {
        name: String,
        #[serde(rename = "type")]
        kind: String,
    }

    // Un seul appel à l'API GitHub : lister les révisions. Les résultats d'un modèle
    // sont RÉPARTIS entre ses révisions, il faut donc toutes les balayer.
    let revs: Vec<Entry> = http
        .get(format!("{GH_API}/{id}"))
        .send()
        .await?
        .error_for_status()
        .context("listing des révisions")?
        .json()
        .await?;
    let revs: Vec<String> = revs
        .into_iter()
        .filter(|e| e.kind == "dir")
        .map(|e| e.name)
        .collect();

    let mut b = ModelBenchmark {
        mteb_id: id.to_string(),
        embed_dim: None,
        n_parameters: None,
        max_tokens: None,
        memory_mb: None,
        languages: Vec::new(),
        french: false,
        retrieval_en: None,
        retrieval_tasks: 0,
    };

    // --- Specs (via le CDN raw, sans quota, contrairement à l'API GitHub).
    //     Certaines révisions ont un model_meta.json INCOMPLET (ex. multilingual-e5 :
    //     `languages` vide → on croirait à tort que le modèle n'est pas multilingue).
    //     On balaie donc toutes les révisions et on garde la plus riche.
    for rev in &revs {
        let Some(v) = get_json(http, id, rev, "model_meta.json").await else {
            continue;
        };
        let langs: Vec<String> = v
            .get("languages")
            .and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(|l| l.as_str().map(String::from)).collect())
            .unwrap_or_default();

        // On complète champ par champ : une révision peut en renseigner un et pas l'autre.
        if b.embed_dim.is_none() {
            b.embed_dim = v.get("embed_dim").and_then(|x| x.as_u64()).map(|x| x as usize);
        }
        if b.n_parameters.is_none() {
            b.n_parameters = v.get("n_parameters").and_then(|x| x.as_u64());
        }
        if b.max_tokens.is_none() {
            b.max_tokens = v.get("max_tokens").and_then(|x| x.as_f64());
        }
        if b.memory_mb.is_none() {
            b.memory_mb = v.get("memory_usage_mb").and_then(|x| x.as_f64());
        }
        if b.languages.is_empty() && !langs.is_empty() {
            b.languages = langs;
        }

        // Assez d'infos : inutile d'interroger les autres révisions.
        if b.embed_dim.is_some() && !b.languages.is_empty() {
            break;
        }
    }
    b.french = b.languages.iter().any(|l| l.starts_with("fra"));

    // --- Score : moyenne ndcg@10 sur les tâches anglaises comparables.
    let mut sum = 0.0;
    let mut n = 0usize;
    for task in EN_TASKS {
        for rev in &revs {
            if let Some(v) = get_json(http, id, rev, &format!("{task}.json")).await {
                if let Some(s) = first_score(&v) {
                    sum += s;
                    n += 1;
                    break;
                }
            }
        }
    }
    if n > 0 {
        b.retrieval_en = Some(sum / n as f64);
        b.retrieval_tasks = n;
    }

    Ok(b)
}

async fn get_json(
    http: &reqwest::Client,
    id: &str,
    rev: &str,
    file: &str,
) -> Option<serde_json::Value> {
    let resp = http.get(format!("{GH_RAW}/{id}/{rev}/{file}")).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    resp.json().await.ok()
}

/// ndcg@10 de la première entrée du premier split non vide (les tâches anglaises
/// n'ont qu'une entrée ; le split peut être `test` ou `dev` selon la tâche).
fn first_score(v: &serde_json::Value) -> Option<f64> {
    let scores = v.get("scores")?.as_object()?;
    for (_split, arr) in scores {
        if let Some(first) = arr.as_array().and_then(|a| a.first()) {
            if let Some(x) = first.get("ndcg_at_10").and_then(|x| x.as_f64()) {
                return Some(x);
            }
            if let Some(x) = first.get("main_score").and_then(|x| x.as_f64()) {
                return Some(x);
            }
        }
    }
    None
}

//! Benchmarks live pour la VISION et le REASONING (LLM), depuis OpenCompass.
//!
//! Deux sources publiques trouvées et validées (aucune API type MTEB n'existe pour
//! ces tâches, mais OpenCompass publie des JSON exploitables) :
//!   * Vision    : `assets/OpenVLM.json` — modèle → { META, MMMU, MMBench, OCRBench… }.
//!   * Reasoning : `dev-assets/hf-research/hf-academic.json` — benchmark → { modèle → score }.
//!
//! On normalise tout au format commun [`ModelBenchmark`] (partagé avec l'embedding) :
//! scores ramenés en 0–1, rang calculé par benchmark, `hf` = dépôt Hugging Face pour
//! la résolution GGUF. « non évalué » reste `None`, jamais zéro.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::benchmarks::{BoardInfo, BoardScore, ModelBenchmark};

const VLM_URL: &str = "http://opencompass.openxlab.space/assets/OpenVLM.json";
const LLM_URL: &str =
    "http://opencompass.oss-cn-shanghai.aliyuncs.com/dev-assets/hf-research/hf-academic.json";
const TTL_SECS: i64 = 7 * 24 * 3600;

/// Vision : (code affiché, champ JSON, sous-champ, échelle pour normaliser en 0–1).
const VLM_BOARDS: &[(&str, &str, &str, f64)] = &[
    ("MMMU", "MMMU_VAL", "Overall", 100.0),
    ("MMBench", "MMBench_TEST_EN_V11", "Overall", 100.0),
    ("OCRBench", "OCRBench", "Final Score", 1000.0),
    ("MMStar", "MMStar", "Overall", 100.0),
    ("MathVista", "MathVista", "Overall", 100.0),
    ("HallusionBench", "HallusionBench", "Overall", 100.0),
    ("AI2D", "AI2D", "Overall", 100.0),
];

/// Reasoning : benchmarks exposés (tous en 0–100 → /100).
const LLM_BOARDS: &[&str] = &[
    "IFEval",
    "MMLU-Pro",
    "GPQA_diamond",
    "Math-500",
    "AIME2024",
    "LiveCodeBench",
    "HumanEval",
    "BBH",
];

/// Clés de métadonnées présentes DANS chaque objet-benchmark du JSON reasoning,
/// à ne pas confondre avec des modèles.
const LLM_META_KEYS: &[&str] = &["dataset", "version", "metric", "mode"];

fn now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64
}

fn cache_path(data_dir: &Path, kind: &str) -> PathBuf {
    data_dir.join(format!("bench-{kind}.json"))
}

#[derive(Serialize, Deserialize)]
struct Cache {
    fetched_at: i64,
    entries: Vec<ModelBenchmark>,
}

pub fn vision_boards() -> Vec<BoardInfo> {
    board_infos(VLM_BOARDS.iter().map(|(c, ..)| *c))
}

pub fn reasoning_boards() -> Vec<BoardInfo> {
    board_infos(LLM_BOARDS.iter().copied())
}

fn board_infos<'a>(codes: impl Iterator<Item = &'a str>) -> Vec<BoardInfo> {
    // Un « Général » (moyenne des benchmarks) en tête, puis chaque benchmark.
    std::iter::once(BoardInfo {
        name: "Général".into(),
        display_name: "Général (moyenne)".into(),
        num_models: None,
        languages: 0,
    })
    .chain(codes.map(|c| BoardInfo {
        name: c.to_string(),
        display_name: c.to_string(),
        num_models: None,
        languages: 0,
    }))
    .collect()
}

fn client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .user_agent("SenseTree")
        .timeout(Duration::from_secs(45))
        .build()?)
}

/// Cœur commun : cache 7 jours, repli sur cache périmé si le réseau échoue.
async fn load(
    data_dir: &Path,
    kind: &str,
    refresh: bool,
    fetch: impl std::future::Future<Output = Result<Vec<ModelBenchmark>>>,
) -> Result<Vec<ModelBenchmark>> {
    let cached: Option<Cache> = std::fs::read_to_string(cache_path(data_dir, kind))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok());

    if !refresh {
        if let Some(c) = &cached {
            if now() - c.fetched_at < TTL_SECS {
                return Ok(c.entries.clone());
            }
        }
    }

    match fetch.await {
        Ok(entries) if !entries.is_empty() => {
            let c = Cache { fetched_at: now(), entries: entries.clone() };
            if let Ok(s) = serde_json::to_string(&c) {
                let _ = std::fs::create_dir_all(data_dir);
                let _ = std::fs::write(cache_path(data_dir, kind), s);
            }
            Ok(entries)
        }
        result => match cached {
            Some(c) => {
                tracing::warn!("catalog {kind} : source injoignable, cache servi");
                Ok(c.entries)
            }
            None => result,
        },
    }
}

/// Rang par benchmark : trie les modèles ayant un score sur ce board, assigne 1..n.
fn assign_ranks(models: &mut [ModelBenchmark]) {
    let boards: Vec<String> = models
        .iter()
        .flat_map(|m| m.scores.iter().map(|s| s.board.clone()))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    for board in boards {
        let mut idx: Vec<(usize, f64)> = models
            .iter()
            .enumerate()
            .filter_map(|(i, m)| {
                m.scores
                    .iter()
                    .find(|s| s.board == board && s.mean.is_some())
                    .map(|s| (i, s.mean.unwrap()))
            })
            .collect();
        idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let total = idx.len() as u32;
        for (rank, (i, _)) in idx.into_iter().enumerate() {
            if let Some(s) = models[i].scores.iter_mut().find(|s| s.board == board) {
                s.rank = Some(rank as u32 + 1);
                s.total = Some(total);
            }
        }
    }
}

/// Ajoute un board synthétique « Général » = moyenne des scores disponibles.
fn add_general(models: &mut [ModelBenchmark]) {
    for m in models.iter_mut() {
        let vals: Vec<f64> = m.scores.iter().filter_map(|s| s.mean).collect();
        let mean = if vals.is_empty() {
            None
        } else {
            Some(vals.iter().sum::<f64>() / vals.len() as f64)
        };
        m.scores.push(BoardScore {
            board: "Général".into(),
            mean,
            retrieval: None,
            rank: None,
            total: None,
        });
    }
}

fn parse_params(s: &str) -> Option<f64> {
    // "8.29B" / "7B" / "1.8b" → 8.29 / 7 / 1.8
    let low = s.to_lowercase();
    let num: String = low.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
    let v: f64 = num.parse().ok()?;
    if low.contains('b') {
        Some(v)
    } else {
        None
    }
}

// =============================================================================
// VISION (OpenVLM.json : modèle → { META, benchmarks })
// =============================================================================

pub async fn vision(data_dir: &Path, refresh: bool) -> Result<Vec<ModelBenchmark>> {
    load(data_dir, "vlm", refresh, fetch_vision()).await
}

async fn fetch_vision() -> Result<Vec<ModelBenchmark>> {
    #[derive(Deserialize)]
    struct Root {
        results: HashMap<String, serde_json::Value>,
    }
    let root: Root = client()?.get(VLM_URL).send().await?.error_for_status()?.json().await?;

    let mut models = Vec::new();
    for (name, v) in root.results {
        let meta = v.get("META");
        let params_b = meta
            .and_then(|m| m.get("Parameters"))
            .and_then(|p| p.as_str())
            .and_then(parse_params);
        // META.Method = "Nom https://huggingface.co/Org/Repo" → on extrait le dépôt HF.
        let method = meta.and_then(|m| m.get("Method")).and_then(|x| x.as_str()).unwrap_or("");
        let hf = method
            .split_whitespace()
            .find(|t| t.contains("huggingface.co/"))
            .and_then(|u| u.split("huggingface.co/").nth(1))
            .map(|s| s.trim_end_matches('/').to_string());
        let url = method
            .split_whitespace()
            .find(|t| t.starts_with("http"))
            .map(|s| s.to_string());

        let mut scores = Vec::new();
        for (code, field, sub, scale) in VLM_BOARDS {
            let raw = v.get(*field).and_then(|b| b.get(*sub)).and_then(|x| x.as_f64());
            if let Some(x) = raw {
                if x > 0.0 {
                    scores.push(BoardScore {
                        board: code.to_string(),
                        mean: Some((x / scale).clamp(0.0, 1.0)),
                        retrieval: None,
                        rank: None,
                        total: None,
                    });
                }
            }
        }
        if scores.is_empty() {
            continue;
        }
        models.push(ModelBenchmark {
            name,
            hf,
            url,
            embed_dim: None,
            params_b,
            max_tokens: None,
            scores,
        });
    }

    add_general(&mut models);
    assign_ranks(&mut models);
    Ok(models)
}

// =============================================================================
// REASONING (hf-academic.json : benchmark → { modèle → score })
// =============================================================================

pub async fn reasoning(data_dir: &Path, refresh: bool) -> Result<Vec<ModelBenchmark>> {
    load(data_dir, "llm", refresh, fetch_reasoning()).await
}

async fn fetch_reasoning() -> Result<Vec<ModelBenchmark>> {
    let root: HashMap<String, serde_json::Value> =
        client()?.get(LLM_URL).send().await?.error_for_status()?.json().await
            .context("parsing hf-academic.json")?;

    // benchmark → (modèle → score normalisé)
    let mut per_model: HashMap<String, ModelBenchmark> = HashMap::new();

    for board in LLM_BOARDS {
        let Some(obj) = root.get(*board).and_then(|v| v.as_object()) else {
            continue;
        };
        for (model, val) in obj {
            if LLM_META_KEYS.contains(&model.as_str()) {
                continue;
            }
            // Score : nombre ou chaîne numérique ; "-" = non évalué.
            let score = val
                .as_f64()
                .or_else(|| val.as_str().and_then(|s| s.trim().parse::<f64>().ok()));
            let Some(score) = score else { continue };

            let e = per_model.entry(model.clone()).or_insert_with(|| {
                let (display, hf) = clean_llm_name(model);
                ModelBenchmark {
                    hf: Some(hf),
                    params_b: parse_params_from_name(&display),
                    name: display,
                    url: None,
                    embed_dim: None,
                    max_tokens: None,
                    scores: Vec::new(),
                }
            });
            e.scores.push(BoardScore {
                board: board.to_string(),
                mean: Some((score / 100.0).clamp(0.0, 1.0)),
                retrieval: None,
                rank: None,
                total: None,
            });
        }
    }

    let mut models: Vec<ModelBenchmark> = per_model.into_values().collect();
    add_general(&mut models);
    assign_ranks(&mut models);
    Ok(models)
}

/// Nettoie un identifiant OpenCompass (`qwen2.5-7b-instruct-turbomind`) → nom lisible
/// et clé HF plausible (sans le suffixe de moteur d'inférence).
fn clean_llm_name(abbr: &str) -> (String, String) {
    let mut s = abbr.to_string();
    for suf in ["-turbomind", "-vllm", "-lmdeploy", "-hf", "-pytorch"] {
        if let Some(stripped) = s.strip_suffix(suf) {
            s = stripped.to_string();
        }
    }
    (s.clone(), s)
}

fn parse_params_from_name(name: &str) -> Option<f64> {
    // Cherche un motif comme "7b", "14b", "1.5b" dans le nom.
    let low = name.to_lowercase();
    let bytes = low.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                i += 1;
            }
            if i < bytes.len() && bytes[i] == b'b' {
                if let Ok(v) = low[start..i].parse::<f64>() {
                    return Some(v);
                }
            }
        } else {
            i += 1;
        }
    }
    None
}

//! Résolution automatique du nom d'installation d'un modèle (Ollama / LM Studio).
//!
//! Problème : aucune API ne relie un modèle du leaderboard MTEB au catalogue Ollama.
//! MAIS n'importe quel GGUF présent sur Hugging Face est installable :
//!   * Ollama    : `ollama pull hf.co/<repo-gguf>:<quant>` fonctionne pour tout GGUF.
//!   * LM Studio : installe directement depuis un dépôt GGUF Hugging Face.
//! On interroge donc l'API HF pour trouver le meilleur dépôt GGUF d'un modèle et en
//! déduire un nom d'installation VÉRIFIÉ (le dépôt existe réellement), plutôt qu'un
//! nom deviné. Tout n'a pas de GGUF communautaire → on renvoie alors `None`, sans
//! prétendre le contraire.

use anyhow::{Context, Result};
use futures_util::future::join_all;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const HF_API: &str = "https://huggingface.co/api";
const TTL_SECS: i64 = 7 * 24 * 3600;

/// Préférence de quantification pour l'EMBEDDING : on privilégie la précision, car
/// une quantification agressive dégrade plus les vecteurs que la génération de texte.
const QUANT_PREF: &[&str] = &["Q8_0", "Q6_K", "Q5_K_M", "Q5_0", "Q4_K_M", "Q4_0", "F16", "BF16"];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallInfo {
    /// Nom Hugging Face d'origine (clé), ex. `microsoft/harrier-oss-v1-0.6b`.
    pub hf: String,
    /// Dépôt GGUF retenu (le plus téléchargé qui correspond), ou None si aucun.
    pub gguf_repo: Option<String>,
    pub quant: Option<String>,
    /// Nom prêt à coller dans Ollama, ex. `hf.co/SuperPauly/…-gguf:Q8_0`.
    pub ollama: Option<String>,
    /// Dépôt à chercher/charger dans LM Studio.
    pub lmstudio: Option<String>,
}

#[derive(Serialize, Deserialize, Default)]
struct Cache {
    entries: HashMap<String, (i64, InstallInfo)>,
}

#[derive(Deserialize)]
struct HfRepo {
    id: String,
    #[serde(default)]
    downloads: i64,
}

#[derive(Deserialize)]
struct HfDetail {
    #[serde(default)]
    siblings: Vec<HfSibling>,
}

#[derive(Deserialize)]
struct HfSibling {
    rfilename: String,
}

fn cache_path(data_dir: &Path) -> PathBuf {
    data_dir.join("installs.json")
}

fn now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64
}

fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Résout les noms d'installation pour une liste de modèles HF. Met en cache 7 jours ;
/// ne re-interroge HF que pour les modèles inconnus ou périmés.
pub async fn resolve(data_dir: &Path, names: Vec<String>) -> Result<Vec<InstallInfo>> {
    let mut cache: Cache = std::fs::read_to_string(cache_path(data_dir))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    let todo: Vec<String> = names
        .iter()
        .filter(|n| {
            cache
                .entries
                .get(*n)
                .map(|(t, _)| now() - *t >= TTL_SECS)
                .unwrap_or(true)
        })
        .cloned()
        .collect();

    if !todo.is_empty() {
        let http = reqwest::Client::builder()
            .user_agent("SenseTree")
            .timeout(Duration::from_secs(20))
            .build()?;
        // Concurrent : chaque modèle fait 1–2 requêtes HF, on les lance en parallèle.
        let fetched = join_all(todo.iter().map(|n| fetch_one(&http, n))).await;
        for (name, info) in todo.into_iter().zip(fetched) {
            cache.entries.insert(name, (now(), info));
        }
        if let Ok(s) = serde_json::to_string(&cache) {
            let _ = std::fs::create_dir_all(data_dir);
            let _ = std::fs::write(cache_path(data_dir), s);
        }
    }

    Ok(names
        .into_iter()
        .filter_map(|n| cache.entries.get(&n).map(|(_, i)| i.clone()))
        .collect())
}

async fn fetch_one(http: &reqwest::Client, hf: &str) -> InstallInfo {
    let empty = InstallInfo {
        hf: hf.to_string(),
        gguf_repo: None,
        quant: None,
        ollama: None,
        lmstudio: None,
    };
    match try_fetch(http, hf).await {
        Ok(info) => info,
        Err(e) => {
            tracing::debug!("installs : {hf} non résolu ({e})");
            empty
        }
    }
}

async fn try_fetch(http: &reqwest::Client, hf: &str) -> Result<InstallInfo> {
    // Nom de base du modèle (sans l'organisation) pour la recherche.
    let base = hf.rsplit('/').next().unwrap_or(hf).to_lowercase();

    let url = format!(
        "{HF_API}/models?search={}&filter=gguf&sort=downloads&direction=-1&limit=15",
        urlencode(&base)
    );
    let repos: Vec<HfRepo> = http.get(&url).send().await?.error_for_status()?.json().await?;

    // On ne retient qu'un dépôt dont le nom contient VRAIMENT le modèle (évite les
    // faux positifs de la recherche floue), le plus téléchargé en premier.
    //
    // Garde anti-dérivés : un dépôt « X-Distill », « X-merge », « abliterated »… est
    // un AUTRE modèle qui ne fait qu'imiter/modifier X. On le rejette, SAUF si le
    // modèle recherché est lui-même un tel dérivé (ex. `deepseek-r1-distill-qwen`).
    // Sans ça, `gemini-2.5-pro` matchait `Qwen3-8B-Gemini-2.5-Pro-Distill-GGUF`.
    const DERIVATIVE: &[&str] = &["distill", "merge", "abliterat", "uncensored", "-ft-", "finetune"];
    let base_is_derivative = DERIVATIVE.iter().any(|d| base.contains(d));
    let best = repos
        .into_iter()
        .filter(|r| {
            let id = r.id.to_lowercase();
            id.contains(&base)
                && (base_is_derivative || !DERIVATIVE.iter().any(|d| id.contains(d)))
        })
        .max_by_key(|r| r.downloads);

    let Some(best) = best else {
        return Ok(InstallInfo {
            hf: hf.to_string(),
            gguf_repo: None,
            quant: None,
            ollama: None,
            lmstudio: None,
        });
    };

    // Fichiers .gguf du dépôt → choix de la meilleure quantification disponible.
    let detail: HfDetail = http
        .get(format!("{HF_API}/models/{}", best.id))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await
        .with_context(|| format!("détail du dépôt {}", best.id))?;
    let ggufs: Vec<String> = detail
        .siblings
        .into_iter()
        .map(|s| s.rfilename)
        .filter(|f| f.to_lowercase().ends_with(".gguf"))
        .collect();

    let quant = QUANT_PREF
        .iter()
        .find(|q| {
            let ql = q.to_lowercase();
            ggufs.iter().any(|f| f.to_lowercase().contains(&ql))
        })
        .map(|q| q.to_string());

    // Ollama : `hf.co/<repo>` (+ tag de quant si plusieurs fichiers). Avec un seul
    // GGUF, le tag est superflu — Ollama prend l'unique fichier.
    let ollama = if ggufs.len() > 1 {
        match &quant {
            Some(q) => Some(format!("hf.co/{}:{q}", best.id)),
            None => Some(format!("hf.co/{}", best.id)),
        }
    } else if !ggufs.is_empty() {
        Some(format!("hf.co/{}", best.id))
    } else {
        None
    };

    Ok(InstallInfo {
        hf: hf.to_string(),
        lmstudio: Some(best.id.clone()),
        gguf_repo: Some(best.id),
        quant,
        ollama,
    })
}

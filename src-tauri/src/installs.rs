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

/// Quantification proposée PAR DÉFAUT, à privilégier dans cet ordre.
///
/// On vise la précision : une quantification agressive dégrade plus les vecteurs
/// d'embedding que la génération de texte. Ce n'est qu'un défaut — toutes les
/// quantifications réellement présentes dans le dépôt sont exposées dans
/// [`InstallInfo::quants`], et l'utilisateur choisit.
const QUANT_PREF: &[&str] = &["Q8_0", "Q6_K", "Q5_K_M", "Q5_0", "Q4_K_M", "Q4_0", "F16", "BF16"];

/// Une quantification réellement disponible dans un dépôt GGUF.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GgufQuant {
    /// Nom tel qu'il apparaît dans le fichier (`Q4_K_M`, `IQ4_XS`, `BF16`…).
    pub quant: String,
    /// Taille totale en octets (somme des parties si le modèle est scindé).
    pub bytes: u64,
    /// Nombre de fichiers. `> 1` = GGUF scindé (`-00001-of-00003`).
    pub parts: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallInfo {
    /// Nom Hugging Face d'origine (clé), ex. `microsoft/harrier-oss-v1-0.6b`.
    pub hf: String,
    /// Dépôt GGUF retenu (le plus téléchargé qui correspond), ou None si aucun.
    pub gguf_repo: Option<String>,
    pub quant: Option<String>,
    /// Toutes les quantifications présentes dans le dépôt, de la plus légère à la
    /// plus lourde. Vide si aucun GGUF n'a été trouvé.
    #[serde(default)]
    pub quants: Vec<GgufQuant>,
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

/// Entrée de l'API `tree` : contrairement à `siblings`, elle porte la TAILLE.
#[derive(Deserialize)]
struct HfTreeEntry {
    path: String,
    #[serde(default)]
    size: u64,
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
            match cache.entries.get(*n) {
                None => true,
                // Une entrée écrite avant l'ajout de `quants` a un dépôt GGUF mais
                // aucune quantification listée : sans ce rafraîchissement forcé, le
                // sélecteur resterait vide jusqu'à l'expiration du TTL (7 jours).
                Some((_, i)) if i.gguf_repo.is_some() && i.quants.is_empty() => true,
                Some((t, _)) => now() - *t >= TTL_SECS,
            }
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
        quants: Vec::new(),
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

/// Vrai si `needle` apparaît dans `hay` comme un TOKEN entier — c'est-à-dire délimité
/// par des caractères non alphanumériques (`-`, `_`, `/`, `.`) ou par les bords de la
/// chaîne — et non au milieu d'un mot.
///
/// Sans cette contrainte, un nom de modèle court comme `r-4b` matchait
/// `qwen3-rerankeR-4B-gguf` (un reranker !), proposé à tort comme GGUF d'un VLM.
fn contains_token(hay: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    let bytes = hay.as_bytes();
    let mut from = 0;
    while let Some(pos) = hay[from..].find(needle) {
        let start = from + pos;
        let end = start + needle.len();
        let left_ok = start == 0 || !bytes[start - 1].is_ascii_alphanumeric();
        let right_ok = end >= bytes.len() || !bytes[end].is_ascii_alphanumeric();
        if left_ok && right_ok {
            return true;
        }
        from = start + 1;
    }
    false
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
            // Le nom doit apparaître comme un TOKEN entier, pas au milieu d'un mot :
            // sinon « r-4b » (modèle vision) matche « qwen3-rerankeR-4B-gguf ».
            contains_token(&id, &base)
                && (base_is_derivative || !DERIVATIVE.iter().any(|d| id.contains(d)))
        })
        .max_by_key(|r| r.downloads);

    let Some(best) = best else {
        return Ok(InstallInfo {
            hf: hf.to_string(),
            gguf_repo: None,
            quant: None,
            quants: Vec::new(),
            ollama: None,
            lmstudio: None,
        });
    };

    // Fichiers .gguf du dépôt, AVEC leur taille (API `tree`, récursive : certains
    // dépôts rangent chaque quantification dans son propre sous-dossier).
    let tree: Vec<HfTreeEntry> = http
        .get(format!("{HF_API}/models/{}/tree/main?recursive=1", best.id))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await
        .with_context(|| format!("arborescence du dépôt {}", best.id))?;

    let ggufs: Vec<HfTreeEntry> =
        tree.into_iter().filter(|e| e.path.to_lowercase().ends_with(".gguf")).collect();

    let quants = group_quants(&ggufs);

    // Défaut : la première préférence RÉELLEMENT présente. Si le dépôt n'expose que
    // des quantifications hors préférences (IQ3_M…), on prend la plus légère plutôt
    // que de renoncer — l'utilisateur peut de toute façon en choisir une autre.
    let quant = QUANT_PREF
        .iter()
        .find(|q| quants.iter().any(|g| g.quant.eq_ignore_ascii_case(q)))
        .map(|q| q.to_string())
        .or_else(|| quants.first().map(|g| g.quant.clone()));

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
        quants,
        ollama,
    })
}

/// Regroupe les fichiers `.gguf` par quantification, en additionnant les parties.
///
/// Trié du plus léger au plus lourd : c'est l'ordre utile quand on cherche ce qui
/// tient dans une carte.
fn group_quants(files: &[HfTreeEntry]) -> Vec<GgufQuant> {
    let mut par_quant: HashMap<String, (u64, u32)> = HashMap::new();
    for f in files {
        let Some(q) = quant_of(&f.path) else { continue };
        let e = par_quant.entry(q).or_insert((0, 0));
        e.0 += f.size;
        e.1 += 1;
    }
    let mut out: Vec<GgufQuant> = par_quant
        .into_iter()
        .map(|(quant, (bytes, parts))| GgufQuant { quant, bytes, parts })
        .collect();
    out.sort_by(|a, b| a.bytes.cmp(&b.bytes).then_with(|| a.quant.cmp(&b.quant)));
    out
}

/// Quantification portée par un nom de fichier GGUF.
///
/// Reconnue par FORME, pas par liste : `Q`/`IQ` suivi d'un chiffre et contenant un
/// `_` (`Q4_K_M`, `Q8_0`, `IQ4_XS`, et tout nouveau `Q4_K_XL` à venir), ou l'un des
/// formats flottants — un ensemble fermé, lui, car ce sont des formats IEEE.
///
/// Le suffixe de découpage (`-00001-of-00003`) est retiré au préalable, sinon la
/// dernière partie du nom masquerait la quantification.
fn quant_of(path: &str) -> Option<String> {
    let base = path.rsplit('/').next().unwrap_or(path).to_uppercase();
    let base = base.strip_suffix(".GGUF").unwrap_or(&base);

    // Retire un éventuel `-00002-OF-00003` final.
    let base = match base.rfind("-OF-") {
        Some(i) => {
            let avant = &base[..i];
            avant.rfind('-').map(|j| &avant[..j]).unwrap_or(avant)
        }
        None => base,
    };

    base.split(['-', '.'])
        .filter(|t| est_quant(t))
        .next_back()
        .map(|t| t.to_string())
}

fn est_quant(token: &str) -> bool {
    if let Some(rest) = token.strip_prefix("IQ").or_else(|| token.strip_prefix('Q')) {
        // `QWEN2` commence par Q mais n'a ni chiffre immédiat ni `_` : écarté.
        return rest.chars().next().is_some_and(|c| c.is_ascii_digit()) && token.contains('_');
    }
    matches!(token, "F16" | "BF16" | "F32" | "FP16" | "FP32" | "F16_K")
}

#[cfg(test)]
mod tests {
    use super::{contains_token, group_quants, quant_of, HfTreeEntry};

    fn f(path: &str, size: u64) -> HfTreeEntry {
        HfTreeEntry { path: path.to_string(), size }
    }

    #[test]
    fn quantification_reconnue_par_forme() {
        assert_eq!(quant_of("Qwen2.5-7B-Instruct-Q4_K_M.gguf").as_deref(), Some("Q4_K_M"));
        assert_eq!(quant_of("model.Q8_0.gguf").as_deref(), Some("Q8_0"));
        assert_eq!(quant_of("Qwen2.5-7B-Instruct-IQ4_XS.gguf").as_deref(), Some("IQ4_XS"));
        assert_eq!(quant_of("machin-BF16.gguf").as_deref(), Some("BF16"));
        // Une quantification inventée demain doit passer sans toucher au code.
        assert_eq!(quant_of("truc-Q4_K_XL.gguf").as_deref(), Some("Q4_K_XL"));
        // Piège : le NOM du modèle commence par Q suivi d'un chiffre.
        assert_eq!(quant_of("Qwen2-7B.gguf"), None);
        assert_eq!(quant_of("Q8-model-Q6_K.gguf").as_deref(), Some("Q6_K"));
        // Fichier scindé : le suffixe de partie ne doit pas masquer la quantification.
        assert_eq!(
            quant_of("Qwen3-30B-A3B-Q4_K_M-00001-of-00002.gguf").as_deref(),
            Some("Q4_K_M")
        );
        // Sous-dossier : seul le nom de fichier compte.
        assert_eq!(quant_of("BF16/Qwen3-30B-A3B-BF16-00002-of-00002.gguf").as_deref(), Some("BF16"));
    }

    #[test]
    fn parties_additionnees_et_triees_du_plus_leger() {
        let v = group_quants(&[
            f("m-Q8_0.gguf", 8_000),
            f("BF16/m-BF16-00001-of-00002.gguf", 9_000),
            f("BF16/m-BF16-00002-of-00002.gguf", 7_000),
            f("m-Q4_K_M.gguf", 4_000),
            f("README.md", 10), // ignoré : pas un .gguf en amont, mais robustesse
        ]);
        let noms: Vec<&str> = v.iter().map(|q| q.quant.as_str()).collect();
        assert_eq!(noms, vec!["Q4_K_M", "Q8_0", "BF16"], "tri du plus léger au plus lourd");
        let bf16 = v.iter().find(|q| q.quant == "BF16").unwrap();
        assert_eq!(bf16.bytes, 16_000, "les parties doivent être additionnées");
        assert_eq!(bf16.parts, 2);
        assert_eq!(v.iter().find(|q| q.quant == "Q4_K_M").unwrap().parts, 1);
    }

    #[test]
    fn matching_par_token_uniquement() {
        // Régression reranker : "r-4b" (modèle vision) ne doit PAS matcher un reranker.
        assert!(!contains_token("qwen3-reranker-4b-gguf", "r-4b"));
        // Mais bien comme token délimité.
        assert!(contains_token("voodisss/r-4b-gguf", "r-4b"));
        assert!(contains_token("r-4b", "r-4b"));
        assert!(contains_token("org/harrier-oss-v1-0.6b-gguf", "harrier-oss-v1-0.6b"));
        // Sous-chaîne au milieu d'un mot = refusée.
        assert!(!contains_token("superqwen", "qwen"));
        assert!(contains_token("super-qwen-x", "qwen"));
    }
}

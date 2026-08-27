//! Observation et pilotage d'un serveur Ollama — local ou distant.
//!
//! Ce qu'on NE PEUT PAS faire, et il faut l'assumer : Ollama n'expose aucune API
//! d'administration. `OLLAMA_MAX_LOADED_MODELS`, `OLLAMA_KEEP_ALIVE`,
//! `OLLAMA_CONTEXT_LENGTH` sont des variables d'environnement lues au démarrage du
//! processus serveur. Ni l'app ni aucun client HTTP ne peut les lire ou les modifier,
//! que le serveur soit sur la machine ou à l'autre bout du réseau.
//!
//! Ce qu'on PEUT faire, et qui couvre le besoin réel :
//!   * `GET /api/ps` — savoir ce qui est RÉELLEMENT chargé à l'instant, sa taille, son
//!     empreinte VRAM et sa date d'expiration. On observe au lieu de supposer.
//!   * `POST /api/generate {"model": …, "keep_alive": 0}` — décharger un modèle
//!     immédiatement, sans prompt. C'est le levier qui rend l'indexation par lots
//!     déterministe : à chaque changement d'étage, on libère le modèle précédent, et
//!     le comportement ne dépend plus de la configuration du serveur d'en face.
//!
//! Ces deux routes sont NATIVES : elles vivent à la racine du serveur, pas sous le
//! préfixe `/v1` compatible OpenAI qu'utilise le reste de l'app — d'où [`native_base`].

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Ramène une URL de base à la racine native du serveur.
///
/// La config pointe vers `http://host:11434/v1` (compatible OpenAI) ; `/api/ps` et
/// `/api/generate` vivent un cran au-dessus.
pub fn native_base(base_url: &str) -> String {
    let b = base_url.trim().trim_end_matches('/');
    b.strip_suffix("/v1").unwrap_or(b).to_string()
}

/// Un modèle actuellement chargé en mémoire par le serveur.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoadedModel {
    pub name: String,
    /// Empreinte totale (VRAM + RAM).
    pub size: u64,
    /// Part réellement en VRAM. Vaut 0 sur un serveur qui tourne en CPU.
    pub size_vram: u64,
    /// Date de déchargement automatique prévue (ISO 8601).
    pub expires_at: Option<String>,
    pub parameter_size: Option<String>,
    pub quantization_level: Option<String>,
}

fn client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .user_agent("SenseTree")
        .timeout(Duration::from_secs(10))
        .build()?)
}

/// Modèles chargés à l'instant sur le serveur visé.
pub async fn loaded(base_url: &str) -> Result<Vec<LoadedModel>> {
    #[derive(Deserialize)]
    struct Ps {
        #[serde(default)]
        models: Vec<Entry>,
    }
    #[derive(Deserialize)]
    struct Entry {
        #[serde(default)]
        name: String,
        #[serde(default)]
        size: u64,
        #[serde(default)]
        size_vram: u64,
        #[serde(default)]
        expires_at: Option<String>,
        #[serde(default)]
        details: Option<Details>,
    }
    #[derive(Deserialize)]
    struct Details {
        #[serde(default)]
        parameter_size: Option<String>,
        #[serde(default)]
        quantization_level: Option<String>,
    }

    let url = format!("{}/api/ps", native_base(base_url));
    let ps: Ps = client()?
        .get(&url)
        .send()
        .await
        .with_context(|| format!("appel {url}"))?
        .error_for_status()
        .context("le serveur n'expose pas /api/ps (ce n'est probablement pas Ollama)")?
        .json()
        .await
        .context("réponse /api/ps illisible")?;

    Ok(ps
        .models
        .into_iter()
        .map(|e| LoadedModel {
            name: e.name,
            size: e.size,
            size_vram: e.size_vram,
            expires_at: e.expires_at,
            parameter_size: e.details.as_ref().and_then(|d| d.parameter_size.clone()),
            quantization_level: e.details.as_ref().and_then(|d| d.quantization_level.clone()),
        })
        .collect())
}

/// Décharge immédiatement un modèle (`keep_alive: 0`, sans prompt).
///
/// Vérifié contre un serveur réel : un modèle chargé disparaît de `/api/ps` juste
/// après cet appel.
pub async fn unload(base_url: &str, model: &str) -> Result<()> {
    let url = format!("{}/api/generate", native_base(base_url));
    client()?
        .post(&url)
        .json(&serde_json::json!({ "model": model, "keep_alive": 0 }))
        .send()
        .await
        .with_context(|| format!("appel {url}"))?
        .error_for_status()
        .with_context(|| format!("déchargement de {model} refusé"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::native_base;

    #[test]
    fn le_prefixe_openai_est_retire() {
        // Cas nominal : la config de l'app pointe vers le préfixe compatible OpenAI.
        assert_eq!(native_base("http://localhost:11434/v1"), "http://localhost:11434");
        assert_eq!(native_base("http://localhost:11434/v1/"), "http://localhost:11434");
        // Serveur distant, même traitement.
        assert_eq!(native_base("http://192.168.1.20:11434/v1"), "http://192.168.1.20:11434");
        // Déjà natif : on n'ampute rien.
        assert_eq!(native_base("http://localhost:11434"), "http://localhost:11434");
        // `/v1` au milieu d'un nom d'hôte ou de chemin ne doit pas être retiré.
        assert_eq!(native_base("http://api.example.com/v1/proxy"), "http://api.example.com/v1/proxy");
        assert_eq!(native_base("  http://x:11434/v1  "), "http://x:11434");
    }
}

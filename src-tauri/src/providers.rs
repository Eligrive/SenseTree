//! Abstraction des providers IA — le cœur de la flexibilité « model-agnostic ».
//!
//! Trois familles de modèles, chacune interchangeable via la configuration :
//!   * Embedding  : local (fastembed/ONNX) OU HTTP compatible OpenAI.
//!   * Reasoning  : HTTP compatible OpenAI (Ollama, LM Studio, serveur maison, API externe).
//!   * Vision     : HTTP compatible OpenAI multimodal (image en base64).
//!
//! `AiEngine` lit la configuration courante et instancie le bon provider à la
//! volée, en mettant en cache le modèle d'embedding local (coûteux à charger).

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;

use crate::config::{ChatConfig, ConfigStore, EmbeddingConfig, EmbeddingMode};

// =========================================================================
// EMBEDDINGS
// =========================================================================

#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Vectorise des documents (côté indexation).
    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>>;
    /// Vectorise une requête utilisateur (côté recherche).
    async fn embed_query(&self, text: String) -> Result<Vec<f32>>;
    /// Dimension des vecteurs produits.
    fn dimensions(&self) -> usize;
}

/// Résout un identifiant de modèle fastembed en (variante, dimension, préfixe e5 requis).
fn resolve_local_model(name: &str) -> (fastembed::EmbeddingModel, usize, bool) {
    use fastembed::EmbeddingModel as M;
    match name.to_lowercase().as_str() {
        "multilingual-e5-small" => (M::MultilingualE5Small, 384, true),
        "multilingual-e5-base" => (M::MultilingualE5Base, 768, true),
        "multilingual-e5-large" => (M::MultilingualE5Large, 1024, true),
        "bge-small-en-v1.5" | "bge-small" => (M::BGESmallENV15, 384, false),
        "bge-base-en-v1.5" | "bge-base" => (M::BGEBaseENV15, 768, false),
        "all-minilm-l6-v2" | "all-minilm" => (M::AllMiniLML6V2, 384, false),
        // Également disponibles côté Ollama → même modèle en local (CPU) ou distant (GPU).
        "nomic-embed-text" | "nomic-embed-text-v1.5" => (M::NomicEmbedTextV15, 768, false),
        "mxbai-embed-large" | "mxbai-embed-large-v1" => (M::MxbaiEmbedLargeV1, 1024, false),
        other => {
            tracing::warn!("modèle d'embedding local inconnu '{other}', repli sur multilingual-e5-small");
            (M::MultilingualE5Small, 384, true)
        }
    }
}

/// Moteur d'embedding local basé sur fastembed (ONNX Runtime).
pub struct LocalEmbedder {
    model: Arc<fastembed::TextEmbedding>,
    dimensions: usize,
    needs_e5_prefix: bool,
    batch_size: usize,
}

/// Modèles d'embedding locaux (fastembed) supportés : identifiant convivial + dimension.
pub fn supported_local_models() -> Vec<(&'static str, usize)> {
    vec![
        ("multilingual-e5-small", 384),
        ("multilingual-e5-base", 768),
        ("multilingual-e5-large", 1024),
        ("bge-small-en-v1.5", 384),
        ("bge-base-en-v1.5", 768),
        ("all-minilm", 384),
        ("nomic-embed-text", 768),
        ("mxbai-embed-large", 1024),
    ]
}

/// Dossier de cache des modèles locaux, sous le dossier de données de l'app.
/// (Avant, fastembed utilisait un cache RELATIF au répertoire courant — imprévisible.)
pub fn local_cache_dir(data_dir: &std::path::Path) -> std::path::PathBuf {
    data_dir.join("models")
}

/// Vrai si le modèle local est déjà téléchargé dans le cache (layout hf-hub :
/// `models--{org}--{repo}`).
pub fn is_local_model_cached(data_dir: &std::path::Path, name: &str) -> bool {
    let (kind, _, _) = resolve_local_model(name);
    let code = fastembed::TextEmbedding::list_supported_models()
        .into_iter()
        .find(|m| format!("{:?}", m.model) == format!("{:?}", kind))
        .map(|m| m.model_code);
    let Some(code) = code else { return false };
    let folder = format!("models--{}", code.replace('/', "--"));
    local_cache_dir(data_dir).join(folder).exists()
}

impl LocalEmbedder {
    /// Charge le modèle (bloquant : à appeler via `spawn_blocking`).
    /// Télécharge le modèle depuis Hugging Face au premier lancement, puis le met en cache.
    pub fn load(
        cfg: &EmbeddingConfig,
        batch_size: usize,
        cache_dir: std::path::PathBuf,
    ) -> Result<Self> {
        let (model_kind, dimensions, needs_e5_prefix) = resolve_local_model(&cfg.model);

        let mut options = fastembed::InitOptions::new(model_kind)
            .with_show_download_progress(false)
            .with_cache_dir(cache_dir);

        if cfg.use_gpu {
            // La lib ORT chargée (CPU ou GPU) est décidée par `ort_setup::ensure_ort`
            // via ORT_DYLIB_PATH. Ici on demande le provider CUDA : s'il est
            // indisponible (lib CPU ou pas de GPU), ORT retombe sur CPU.
            options = options.with_execution_providers(vec![
                ort::execution_providers::CUDAExecutionProvider::default().build(),
                ort::execution_providers::CPUExecutionProvider::default().build(),
            ]);
            tracing::info!("embedding local : provider CUDA demandé (repli CPU automatique)");
        }

        let model = fastembed::TextEmbedding::try_new(options)
            .context("chargement du modèle d'embedding local (fastembed)")?;

        Ok(LocalEmbedder {
            model: Arc::new(model),
            dimensions,
            needs_e5_prefix,
            batch_size: batch_size.max(1),
        })
    }

}

#[async_trait]
impl EmbeddingProvider for LocalEmbedder {
    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let inputs: Vec<String> = if self.needs_e5_prefix {
            texts.into_iter().map(|t| format!("passage: {t}")).collect()
        } else {
            texts
        };
        let model = self.model.clone();
        let batch = self.batch_size;
        // fastembed est synchrone/CPU-bound : on l'exécute hors du runtime async.
        tokio::task::spawn_blocking(move || {
            model.embed(inputs, Some(batch)).context("échec de l'embedding local")
        })
        .await
        .context("tâche d'embedding interrompue")?
    }

    async fn embed_query(&self, text: String) -> Result<Vec<f32>> {
        let input = if self.needs_e5_prefix {
            format!("query: {text}")
        } else {
            text
        };
        let mut out = self.embed_documents_raw(vec![input]).await?;
        out.pop().ok_or_else(|| anyhow!("aucun vecteur produit pour la requête"))
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

impl LocalEmbedder {
    /// Embedding sans préfixe (le préfixe query/passage est géré par l'appelant).
    async fn embed_documents_raw(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let model = self.model.clone();
        let batch = self.batch_size;
        tokio::task::spawn_blocking(move || {
            model.embed(texts, Some(batch)).context("échec de l'embedding local")
        })
        .await
        .context("tâche d'embedding interrompue")?
    }
}

/// Provider d'embedding via un endpoint HTTP compatible OpenAI (`/embeddings`).
pub struct OpenAiEmbedder {
    http: reqwest::Client,
    base_url: String,
    model: String,
    api_key: String,
    dimensions: usize,
}

impl OpenAiEmbedder {
    pub fn new(http: reqwest::Client, cfg: &EmbeddingConfig) -> Self {
        OpenAiEmbedder {
            http,
            base_url: cfg.base_url.trim_end_matches('/').to_string(),
            model: cfg.model.clone(),
            api_key: cfg.api_key.clone(),
            dimensions: cfg.dimensions,
        }
    }

    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.base_url);
        let mut req = self
            .http
            .post(&url)
            .json(&json!({ "model": self.model, "input": texts }));
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let resp = req.send().await.context("appel /embeddings")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("/embeddings a renvoyé {status}: {body}"));
        }

        #[derive(Deserialize)]
        struct EmbeddingData {
            embedding: Vec<f32>,
        }
        #[derive(Deserialize)]
        struct EmbeddingResponse {
            data: Vec<EmbeddingData>,
        }
        let parsed: EmbeddingResponse = resp.json().await.context("parsing de la réponse /embeddings")?;
        Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAiEmbedder {
    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        self.embed_batch(texts).await
    }

    async fn embed_query(&self, text: String) -> Result<Vec<f32>> {
        let mut out = self.embed_batch(vec![text]).await?;
        out.pop().ok_or_else(|| anyhow!("aucun vecteur renvoyé par le serveur d'embedding"))
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

// =========================================================================
// CHAT / REASONING / VISION
// =========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Client de chat compatible OpenAI, réutilisé pour le reasoning et la vision.
pub struct OpenAiChatClient {
    http: reqwest::Client,
    base_url: String,
    model: String,
    api_key: String,
}

impl OpenAiChatClient {
    pub fn new(http: reqwest::Client, cfg: &ChatConfig) -> Self {
        OpenAiChatClient {
            http,
            base_url: cfg.base_url.trim_end_matches('/').to_string(),
            model: cfg.model.clone(),
            api_key: cfg.api_key.clone(),
        }
    }

    /// Échange de chat standard. `json_mode` force une réponse JSON stricte (pour les actions).
    pub async fn chat(&self, messages: Vec<ChatMessage>, json_mode: bool) -> Result<String> {
        let url = format!("{}/chat/completions", self.base_url);
        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "stream": false,
        });
        if json_mode {
            body["response_format"] = json!({ "type": "json_object" });
        }

        let mut req = self.http.post(&url).json(&body);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let resp = req.send().await.context("appel /chat/completions")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("/chat/completions a renvoyé {status}: {text}"));
        }
        parse_chat_content(resp.json().await.context("parsing de la réponse de chat")?)
    }

    /// Décrit une image (base64) via un modèle multimodal — l'« extraction du sens » visuelle.
    pub async fn describe_image(&self, image_base64: &str, mime: &str, prompt: &str) -> Result<String> {
        let url = format!("{}/chat/completions", self.base_url);
        let data_url = format!("data:{mime};base64,{image_base64}");
        let body = json!({
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": prompt },
                    { "type": "image_url", "image_url": { "url": data_url } }
                ]
            }],
            "temperature": 0.1,
            "stream": false,
        });
        let mut req = self.http.post(&url).json(&body);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let resp = req.send().await.context("appel vision /chat/completions")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("vision a renvoyé {status}: {text}"));
        }
        parse_chat_content(resp.json().await.context("parsing de la réponse vision")?)
    }
}

fn parse_chat_content(value: serde_json::Value) -> Result<String> {
    value
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("réponse de chat sans contenu exploitable"))
}

// =========================================================================
// MOTEUR IA (résolution des providers depuis la config)
// =========================================================================

#[derive(Debug, Serialize)]
pub struct HealthReport {
    pub embedding_ok: bool,
    pub embedding_detail: String,
    pub reasoning_ok: bool,
    pub reasoning_detail: String,
    pub vision_ok: bool,
    pub vision_detail: String,
}

struct CachedEmbedder {
    key: String,
    provider: Arc<dyn EmbeddingProvider>,
}

/// Point d'entrée unique vers les modèles, piloté par la configuration.
pub struct AiEngine {
    config: Arc<ConfigStore>,
    http: reqwest::Client,
    embedder: tokio::sync::Mutex<Option<CachedEmbedder>>,
    /// Dossier de données (pour approvisionner ONNX Runtime).
    data_dir: std::path::PathBuf,
    /// Garantit qu'ONNX Runtime (ORT_DYLIB_PATH) est prêt AVANT toute init d'ORT,
    /// une seule fois, quel que soit le premier appelant (health check ou worker).
    ort_ready: tokio::sync::OnceCell<()>,
}

impl AiEngine {
    pub fn new(config: Arc<ConfigStore>, data_dir: std::path::PathBuf) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .unwrap_or_default();
        AiEngine {
            config,
            http,
            embedder: tokio::sync::Mutex::new(None),
            data_dir,
            ort_ready: tokio::sync::OnceCell::new(),
        }
    }

    fn embedding_cache_key(cfg: &EmbeddingConfig) -> String {
        format!("{:?}|{}|{}|{}", cfg.mode, cfg.model, cfg.base_url, cfg.use_gpu)
    }

    /// Renvoie le provider d'embedding courant, en le (re)construisant si la config a changé.
    pub async fn embedder(&self) -> Result<Arc<dyn EmbeddingProvider>> {
        let cfg = self.config.snapshot();
        let key = Self::embedding_cache_key(&cfg.embedding);

        let mut guard = self.embedder.lock().await;
        if let Some(cached) = guard.as_ref() {
            if cached.key == key {
                return Ok(cached.provider.clone());
            }
        }

        let provider: Arc<dyn EmbeddingProvider> = match cfg.embedding.mode {
            EmbeddingMode::Local => {
                // On s'assure qu'ORT_DYLIB_PATH est posé AVANT d'initialiser ORT
                // (fastembed). En cas d'échec de téléchargement, la cellule reste
                // vide → nouvelle tentative au prochain appel.
                let dir = self.data_dir.clone();
                let use_gpu = cfg.embedding.use_gpu;
                self.ort_ready
                    .get_or_try_init(|| async move {
                        tokio::task::spawn_blocking(move || crate::ort_setup::ensure_ort(&dir, use_gpu))
                            .await
                            .context("tâche de préparation ORT interrompue")?
                            .map(|_gpu| ())
                    })
                    .await?;

                let embed_cfg = cfg.embedding.clone();
                let batch = cfg.indexing.batch_size;
                let cache = local_cache_dir(&self.data_dir);
                let local =
                    tokio::task::spawn_blocking(move || LocalEmbedder::load(&embed_cfg, batch, cache))
                        .await
                        .context("tâche de chargement du modèle interrompue")??;
                Arc::new(local)
            }
            EmbeddingMode::Openai => Arc::new(OpenAiEmbedder::new(self.http.clone(), &cfg.embedding)),
        };

        *guard = Some(CachedEmbedder {
            key,
            provider: provider.clone(),
        });
        Ok(provider)
    }

    /// Invalide le cache d'embedding (à appeler après un changement de config, ou
    /// pour libérer les ressources — la session ONNX Runtime et ses threads intra-op
    /// sont détruits dès que plus aucune référence ne subsiste).
    pub async fn invalidate_embedder(&self) {
        *self.embedder.lock().await = None;
    }

    /// Vrai si un modèle d'embedding est actuellement chargé en mémoire.
    pub async fn embedder_loaded(&self) -> bool {
        self.embedder.lock().await.is_some()
    }

    /// Pré-télécharge un modèle d'embedding local (le charge puis le libère), afin
    /// que le catalogue puisse l'installer sans attendre la première indexation.
    pub async fn preload_local_model(&self, model: &str) -> Result<()> {
        let dir = self.data_dir.clone();
        self.ort_ready
            .get_or_try_init(|| async move {
                tokio::task::spawn_blocking(move || crate::ort_setup::ensure_ort(&dir, false))
                    .await
                    .context("tâche de préparation ORT interrompue")?
                    .map(|_gpu| ())
            })
            .await?;

        let cfg = self.config.snapshot();
        let embed_cfg = EmbeddingConfig {
            model: model.to_string(),
            ..cfg.embedding.clone()
        };
        let cache = local_cache_dir(&self.data_dir);
        tokio::task::spawn_blocking(move || LocalEmbedder::load(&embed_cfg, 1, cache))
            .await
            .context("tâche de téléchargement interrompue")??;
        Ok(())
    }

    pub fn reasoning_client(&self) -> OpenAiChatClient {
        let cfg = self.config.snapshot();
        OpenAiChatClient::new(self.http.clone(), &cfg.reasoning)
    }

    pub fn vision_client(&self) -> OpenAiChatClient {
        let cfg = self.config.snapshot();
        OpenAiChatClient::new(self.http.clone(), &cfg.vision)
    }

    /// Vérifie la disponibilité des différents providers (pour griser l'UI si absent).
    pub async fn health(&self) -> HealthReport {
        let cfg = self.config.snapshot();

        // Embedding — on NE construit PAS le modèle local ici (le health check est
        // périodique : le charger le maintiendrait résident et ses threads ORT
        // tourneraient en continu). On rapporte l'état sans instancier.
        let (embedding_ok, embedding_detail) = match cfg.embedding.mode {
            EmbeddingMode::Local => {
                let dims = resolve_local_model(&cfg.embedding.model).1;
                let detail = if self.embedder_loaded().await {
                    format!("chargé ({dims} dims)")
                } else {
                    format!("prêt, chargé à la demande ({dims} dims)")
                };
                (true, detail)
            }
            EmbeddingMode::Openai => (
                true,
                format!("serveur {} ({} dims)", cfg.embedding.base_url, cfg.embedding.dimensions),
            ),
        };

        // Reasoning
        let (reasoning_ok, reasoning_detail) = if cfg.reasoning.enabled {
            match self.ping_models(&cfg.reasoning).await {
                Ok(_) => (true, "connecté".to_string()),
                Err(e) => (false, format!("{e}")),
            }
        } else {
            (false, "désactivé".to_string())
        };

        // Vision
        let (vision_ok, vision_detail) = if cfg.vision.enabled {
            match self.ping_models(&cfg.vision).await {
                Ok(_) => (true, "connecté".to_string()),
                Err(e) => (false, format!("{e}")),
            }
        } else {
            (false, "désactivé".to_string())
        };

        HealthReport {
            embedding_ok,
            embedding_detail,
            reasoning_ok,
            reasoning_detail,
            vision_ok,
            vision_detail,
        }
    }

    /// Ping léger d'un endpoint compatible OpenAI (`GET /models`).
    async fn ping_models(&self, cfg: &ChatConfig) -> Result<()> {
        let url = format!("{}/models", cfg.base_url.trim_end_matches('/'));
        let mut req = self
            .http
            .get(&url)
            .timeout(Duration::from_secs(5));
        if !cfg.api_key.is_empty() {
            req = req.bearer_auth(&cfg.api_key);
        }
        let resp = req.send().await.context("serveur injoignable")?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(anyhow!("le serveur a répondu {}", resp.status()))
        }
    }
}

//! Abstraction des providers IA — le cœur de la flexibilité « model-agnostic ».
//!
//! Trois familles de modèles, chacune interchangeable via la configuration :
//!   * Embedding  : local (fastembed/ONNX) OU HTTP compatible OpenAI.
//!   * Reasoning  : HTTP compatible OpenAI (Ollama, LM Studio, serveur maison, API externe).
//!   * Vision     : HTTP compatible OpenAI multimodal (image en base64).
//!   * Transcription : HTTP compatible OpenAI (`/audio/transcriptions`, multipart).
//!   * Description video : HTTP compatible OpenAI (`/chat/completions`, part `video_url`).
//!
//! `AiEngine` lit la configuration courante et instancie le bon provider à la
//! volée, en mettant en cache le modèle d'embedding local (coûteux à charger).

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;

use crate::config::{
    ChatConfig, ConfigStore, EmbeddingConfig, EmbeddingMode, ReasoningEffort, TranscriptionConfig,
    VideoConfig, VideoDelivery,
};

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
        "bge-large-en-v1.5" | "bge-large" => (M::BGELargeENV15, 1024, false),
        "all-minilm-l6-v2" | "all-minilm" => (M::AllMiniLML6V2, 384, false),
        // Également disponibles côté Ollama → même modèle en local (CPU) ou distant (GPU).
        "nomic-embed-text" | "nomic-embed-text-v1.5" => (M::NomicEmbedTextV15, 768, false),
        "mxbai-embed-large" | "mxbai-embed-large-v1" => (M::MxbaiEmbedLargeV1, 1024, false),
        // GTE v1.5 (2024) : excellents rapports qualité/taille, contexte 8k.
        "gte-base-en-v1.5" | "gte-base" => (M::GTEBaseENV15, 768, false),
        "gte-large-en-v1.5" | "gte-large" => (M::GTELargeENV15, 1024, false),
        // ModernBERT (fin 2024) : encodeur récent, contexte long.
        "modernbert-embed-large" | "modernbert-large" => (M::ModernBertEmbedLarge, 1024, false),
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

/// Modèles d'embedding locaux (fastembed) supportés : identifiant, dimension, et
/// s'ils sont MULTILINGUES.
///
/// Ce dernier drapeau est décisif et invisible dans le nom : seule la famille E5 est
/// multilingue ici. Les autres sont entraînés sur de l'anglais et s'effondrent sur un
/// corpus français — les proposer sans le dire mène l'utilisateur à une indexation
/// médiocre qu'il ne pourra corriger qu'en réindexant tout.
pub fn supported_local_models() -> Vec<(&'static str, usize, bool)> {
    vec![
        ("multilingual-e5-small", 384, true),
        ("multilingual-e5-base", 768, true),
        ("multilingual-e5-large", 1024, true),
        ("bge-small-en-v1.5", 384, false),
        ("bge-base-en-v1.5", 768, false),
        ("bge-large-en-v1.5", 1024, false),
        ("gte-base-en-v1.5", 768, false),
        ("gte-large-en-v1.5", 1024, false),
        ("modernbert-embed-large", 1024, false),
        ("all-minilm", 384, false),
        ("nomic-embed-text", 768, false),
        ("mxbai-embed-large", 1024, false),
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
        // Mesuré ici plutôt qu'aux appels : le worker embedde depuis quatre endroits,
        // et un cinquième ajouté plus tard échapperait au compteur.
        let units = inputs.len() as u64;
        let bytes: u64 = inputs.iter().map(|t| t.len() as u64).sum();
        let started = std::time::Instant::now();

        let model = self.model.clone();
        let batch = self.batch_size;
        // fastembed est synchrone/CPU-bound : on l'exécute hors du runtime async.
        let out = tokio::task::spawn_blocking(move || {
            model.embed(inputs, Some(batch)).context("échec de l'embedding local")
        })
        .await
        .context("tâche d'embedding interrompue")?;

        match &out {
            Ok(_) => crate::metrics::record(
                crate::metrics::Stage::Embedding,
                started.elapsed(),
                units,
                bytes,
            ),
            Err(_) => crate::metrics::record_error(crate::metrics::Stage::Embedding),
        }
        out
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
    /// Contexte demandé au serveur, borné à ce dont un chunk a réellement besoin.
    num_ctx: u32,
    /// Textes par requête. Sans cette borne, un gros fichier produit une requête
    /// unique de plusieurs milliers de textes que le serveur refuse.
    batch_size: usize,
}

/// Contexte suffisant pour un chunk, avec une marge confortable.
///
/// Un chunk fait `chunk_size` caractères, auxquels s'ajoute le préfixe de contextual
/// retrieval (nom du fichier + résumé, ~250 caractères). On compte 2 caractères par
/// token — pessimiste, le français tourne plutôt autour de 3,5 — et on ne descend
/// jamais sous 2048.
///
/// L'enjeu n'est pas la vitesse mais la MÉMOIRE : le cache KV est alloué pour tout le
/// contexte annoncé, et un modèle d'embedding à 32k réserve plusieurs gigaoctets de
/// VRAM dont il ne se servira jamais.
pub fn embedding_num_ctx(chunk_size: usize) -> u32 {
    let besoin = ((chunk_size + 250) / 2) as u32;
    besoin.max(2048).div_ceil(1024) * 1024
}

impl OpenAiEmbedder {
    pub fn new(
        http: reqwest::Client,
        cfg: &EmbeddingConfig,
        num_ctx: u32,
        batch_size: usize,
    ) -> Self {
        OpenAiEmbedder {
            http,
            base_url: cfg.base_url.trim_end_matches('/').to_string(),
            model: cfg.model.clone(),
            api_key: cfg.api_key.clone(),
            dimensions: cfg.dimensions,
            num_ctx,
            batch_size: batch_size.max(1),
        }
    }

    /// Vectorise en LOTS bornés, et non en une seule requête.
    ///
    /// Le réglage `indexing.batch_size` n'était honoré que par le moteur embarqué ;
    /// le chemin serveur envoyait tous les chunks d'un fichier d'un coup. Sur un
    /// journal de 1,7 Mo, cela faisait **une requête de ~1 750 textes** que le serveur
    /// n'honorait pas : l'appel échouait, le fichier était réessayé, et la file
    /// entière restait bloquée dessus pendant des minutes.
    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let taille = self.batch_size.max(1);
        if texts.len() <= taille {
            return self.embed_one_request(texts).await;
        }
        let mut out = Vec::with_capacity(texts.len());
        for lot in texts.chunks(taille) {
            out.extend(self.embed_one_request(lot.to_vec()).await?);
        }
        Ok(out)
    }

    async fn embed_one_request(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        // Sur Ollama on passe par l'API NATIVE : c'est la seule qui accepte `num_ctx`.
        // Mesuré sur un serveur réel avec un modèle d'embedding à 32k de contexte :
        // 5,78 Go de VRAM au contexte par défaut contre 2,13 Go à 2048 — 3,65 Go de
        // cache KV alloués pour rien, puisqu'un chunk fait quelques centaines de
        // tokens. L'endpoint compatible OpenAI, lui, IGNORE le paramètre (vérifié).
        if let Some(v) = self.embed_native_ollama(&texts).await {
            return v;
        }

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

    /// Tente l'API native Ollama (`/api/embed`) pour pouvoir borner le contexte.
    ///
    /// `None` = ce n'est pas un Ollama (ou il n'a pas répondu) → l'appelant retombe
    /// sur le chemin compatible OpenAI. On ne casse donc aucun serveur tiers.
    async fn embed_native_ollama(&self, texts: &[String]) -> Option<Result<Vec<Vec<f32>>>> {
        if !self.base_url.ends_with("/v1") || !self.api_key.is_empty() {
            return None;
        }
        let root = self.base_url.trim_end_matches("/v1");

        let resp = self
            .http
            .post(format!("{root}/api/embed"))
            .json(&json!({
                "model": self.model,
                "input": texts,
                "options": { "num_ctx": self.num_ctx },
            }))
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }

        #[derive(Deserialize)]
        struct NativeEmbed {
            #[serde(default)]
            embeddings: Vec<Vec<f32>>,
        }
        let parsed: NativeEmbed = resp.json().await.ok()?;
        // Une réponse vide ou incomplète n'est pas exploitable : on laisse le repli
        // OpenAI faire le travail plutôt que de stocker des vecteurs manquants.
        if parsed.embeddings.len() != texts.len() {
            return None;
        }
        Some(Ok(parsed.embeddings))
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAiEmbedder {
    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let units = texts.len() as u64;
        let bytes: u64 = texts.iter().map(|t| t.len() as u64).sum();
        let started = std::time::Instant::now();
        let out = self.embed_batch(texts).await;
        match &out {
            Ok(_) => crate::metrics::record(
                crate::metrics::Stage::Embedding,
                started.elapsed(),
                units,
                bytes,
            ),
            Err(_) => crate::metrics::record_error(crate::metrics::Stage::Embedding),
        }
        out
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
// RERANKING (cross-encoder local, fastembed / ONNX)
// =========================================================================

/// Résout un identifiant convivial de reranker en modèle fastembed.
fn resolve_reranker_model(name: &str) -> fastembed::RerankerModel {
    use fastembed::RerankerModel as R;
    match name.to_lowercase().as_str() {
        "bge-reranker-base" => R::BGERerankerBase,
        "bge-reranker-v2-m3" | "bge-reranker-v2" | "bge-reranker" => R::BGERerankerV2M3,
        "jina-reranker-v2" | "jina-reranker-v2-base-multilingual" => {
            R::JINARerankerV2BaseMultiligual
        }
        other => {
            tracing::warn!("reranker inconnu '{other}', repli sur bge-reranker-v2-m3");
            R::BGERerankerV2M3
        }
    }
}

/// Rerankers locaux (fastembed) proposés à l'utilisateur.
pub fn supported_reranker_models() -> Vec<&'static str> {
    vec![
        "bge-reranker-v2-m3",
        "bge-reranker-base",
        "jina-reranker-v2-base-multilingual",
    ]
}

/// Reranker cross-encoder : score conjointement (requête, passage) — bien plus
/// précis qu'un bi-encodeur, mais coûteux, donc appliqué UNIQUEMENT au petit
/// ensemble de candidats déjà retenus par la recherche hybride.
pub struct Reranker {
    model: Arc<fastembed::TextRerank>,
}

impl Reranker {
    /// Charge le modèle (bloquant : appeler via `spawn_blocking`, ORT déjà prêt).
    pub fn load(model_name: &str, cache_dir: std::path::PathBuf) -> Result<Self> {
        let opts = fastembed::RerankInitOptions::new(resolve_reranker_model(model_name))
            .with_show_download_progress(false)
            .with_cache_dir(cache_dir);
        let model =
            fastembed::TextRerank::try_new(opts).context("chargement du reranker (fastembed)")?;
        Ok(Reranker { model: Arc::new(model) })
    }

    /// Réordonne `docs` par pertinence à `query`. Renvoie `(index d'origine, score)`
    /// TRIÉ par score décroissant. Le score est un logit (non borné) : à convertir en
    /// 0–1 par sigmoïde côté appelant si besoin d'affichage.
    pub async fn rerank(&self, query: String, docs: Vec<String>) -> Result<Vec<(usize, f32)>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        let model = self.model.clone();
        tokio::task::spawn_blocking(move || {
            let res = model
                .rerank(query, docs, false, None)
                .context("échec du reranking")?;
            Ok::<Vec<(usize, f32)>, anyhow::Error>(
                res.into_iter().map(|r| (r.index, r.score)).collect(),
            )
        })
        .await
        .context("tâche de reranking interrompue")?
    }
}

// =========================================================================
// CLIP (recherche d'images par similarité visuelle)
// =========================================================================

/// Dimension des vecteurs CLIP ViT-B/32 (le texte et l'image partagent l'espace).
pub const CLIP_DIM: usize = 512;

/// Embedder CLIP : encode IMAGES et TEXTE dans le MÊME espace vectoriel (ViT-B/32),
/// pour retrouver des images par similarité visuelle à partir d'une requête texte.
pub struct ClipEmbedder {
    image: Arc<fastembed::ImageEmbedding>,
    text: Arc<fastembed::TextEmbedding>,
}

impl ClipEmbedder {
    /// Charge les deux encodeurs CLIP (bloquant : appeler via `spawn_blocking`, ORT prêt).
    pub fn load(cache_dir: std::path::PathBuf) -> Result<Self> {
        let image = fastembed::ImageEmbedding::try_new(
            fastembed::ImageInitOptions::new(fastembed::ImageEmbeddingModel::ClipVitB32)
                .with_show_download_progress(false)
                .with_cache_dir(cache_dir.clone()),
        )
        .context("chargement du modèle CLIP (image)")?;
        let text = fastembed::TextEmbedding::try_new(
            fastembed::InitOptions::new(fastembed::EmbeddingModel::ClipVitB32)
                .with_show_download_progress(false)
                .with_cache_dir(cache_dir),
        )
        .context("chargement du modèle CLIP (texte)")?;
        Ok(ClipEmbedder { image: Arc::new(image), text: Arc::new(text) })
    }

    /// Vectorise des images (par chemin). Ignore silencieusement celles illisibles.
    pub async fn embed_images(&self, paths: Vec<std::path::PathBuf>) -> Result<Vec<Vec<f32>>> {
        let model = self.image.clone();
        tokio::task::spawn_blocking(move || {
            model.embed(paths, Some(16)).context("embedding CLIP image")
        })
        .await
        .context("tâche CLIP image interrompue")?
    }

    /// Vectorise une requête texte dans l'espace CLIP.
    pub async fn embed_text(&self, text: String) -> Result<Vec<f32>> {
        let model = self.text.clone();
        let mut out = tokio::task::spawn_blocking(move || {
            model.embed(vec![text], Some(1)).context("embedding CLIP texte")
        })
        .await
        .context("tâche CLIP texte interrompue")??;
        out.pop().ok_or_else(|| anyhow!("aucun vecteur CLIP produit pour la requête"))
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

/// Un appel d'outil demandé par le modèle (function-calling).
#[derive(Debug, Clone)]
pub struct ToolCallOut {
    pub id: String,
    pub name: String,
    /// Arguments au format JSON (chaîne), à parser selon le schéma de l'outil.
    pub arguments: String,
}

/// Résultat d'un tour d'agent : soit du texte final, soit des appels d'outils.
#[derive(Debug, Default)]
pub struct AgentTurn {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCallOut>,
    /// Message assistant brut (réinjecté tel quel dans l'historique pour la suite).
    pub raw_message: serde_json::Value,
}

/// Délai maximal d'un appel vision. Volontairement large : un modèle multimodal
/// peut mettre plusieurs dizaines de secondes à se (re)charger en VRAM (démarrage à
/// froid ou swap de modèle sur un GPU partagé). On préfère patienter plutôt qu'échouer.
const VISION_TIMEOUT: Duration = Duration::from_secs(300);

/// Nature d'un échec d'appel à un modèle distant, pour décider s'il faut re-tenter
/// ou abandonner. Partagée par la vision et la transcription : la distinction
/// « serveur occupé » / « requête invalide » y est identique, et le worker en tire
/// la même politique de re-tentative.
#[derive(Debug)]
pub enum AiCallError {
    /// Passager (timeout, serveur occupé/en chargement, 5xx, réseau) → re-tentable.
    Transient(anyhow::Error),
    /// Définitif (média invalide, format refusé, modèle absent) → repli contextuel.
    Permanent(anyhow::Error),
}

impl AiCallError {
    /// Classe un statut HTTP d'échec. 5xx / 408 / 429 = serveur occupé, en
    /// chargement ou saturé, donc re-tentable ; tout autre 4xx est une requête
    /// définitivement invalide (média, format, modèle absent).
    fn from_status(status: reqwest::StatusCode, err: anyhow::Error) -> Self {
        if status.is_server_error()
            || status == reqwest::StatusCode::REQUEST_TIMEOUT
            || status == reqwest::StatusCode::TOO_MANY_REQUESTS
        {
            AiCallError::Transient(err)
        } else {
            AiCallError::Permanent(err)
        }
    }
}

impl std::fmt::Display for AiCallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AiCallError::Transient(e) => write!(f, "{e} [passager]"),
            AiCallError::Permanent(e) => write!(f, "{e} [définitif]"),
        }
    }
}
impl std::error::Error for AiCallError {}

/// Client de chat compatible OpenAI, réutilisé pour le reasoning et la vision.
pub struct OpenAiChatClient {
    http: reqwest::Client,
    base_url: String,
    model: String,
    api_key: String,
    /// Effort de raisonnement configuré pour ce créneau (chat, actions).
    effort: ReasoningEffort,
}

impl OpenAiChatClient {
    pub fn new(http: reqwest::Client, cfg: &ChatConfig) -> Self {
        OpenAiChatClient {
            http,
            base_url: cfg.base_url.trim_end_matches('/').to_string(),
            model: cfg.model.clone(),
            api_key: cfg.api_key.clone(),
            effort: cfg.reasoning_effort,
        }
    }

    /// Échange de chat standard. `json_mode` force une réponse JSON stricte (pour les actions).
    pub async fn chat(&self, messages: Vec<ChatMessage>, json_mode: bool) -> Result<String> {
        self.chat_with(messages, json_mode, self.effort).await
    }

    /// Chat SANS chaîne de raisonnement, pour les appels courts et nombreux de
    /// l'indexation (classer un dossier, qualifier un document, deviner un contexte).
    ///
    /// Les modèles « thinking » raisonnent avant de répondre, ce qui est du gaspillage
    /// pur quand la réponse tient en six tokens. Mesuré sur un serveur réel, pour la
    /// classification d'un dossier : **24,4 s avec raisonnement contre 0,78 s sans**,
    /// pour une réponse identique. À 25 s de délai d'attente, cela suffisait à faire
    /// échouer la classification en boucle et à bloquer toute l'indexation.
    ///
    /// L'effort est CHOISI PAR L'APPELANT (réglage `indexing.qualify_effort`) et non
    /// imposé ici : sur un choix binaire il n'a aucun intérêt, sur une qualification de
    /// document il peut se défendre. Les serveurs qui ne connaissent pas
    /// `reasoning_effort` ignorent le champ — vérifié sur Ollama.
    pub async fn chat_quick(
        &self,
        messages: Vec<ChatMessage>,
        json_mode: bool,
        effort: ReasoningEffort,
    ) -> Result<String> {
        self.chat_with(messages, json_mode, effort).await
    }

    async fn chat_with(
        &self,
        messages: Vec<ChatMessage>,
        json_mode: bool,
        effort: ReasoningEffort,
    ) -> Result<String> {
        let bytes: u64 = messages.iter().map(|m| m.content.len() as u64).sum();
        let started = std::time::Instant::now();
        let out = self.chat_inner(messages, json_mode, effort).await;
        match &out {
            Ok(_) => {
                crate::metrics::record(crate::metrics::Stage::Reasoning, started.elapsed(), 1, bytes)
            }
            Err(_) => crate::metrics::record_error(crate::metrics::Stage::Reasoning),
        }
        out
    }

    async fn chat_inner(
        &self,
        messages: Vec<ChatMessage>,
        json_mode: bool,
        effort: ReasoningEffort,
    ) -> Result<String> {
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
        // `Auto` n'envoie rien : on laisse le serveur décider, comme avant ce réglage.
        if let Some(e) = effort.as_param() {
            body["reasoning_effort"] = json!(e);
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

    /// Échange de chat avec OUTILS (function-calling natif). Renvoie soit du texte
    /// final, soit une liste d'appels d'outils que l'appelant doit exécuter puis
    /// réinjecter. Compatible OpenAI (`tools` + `tool_choice: auto`). Les modèles qui
    /// ne savent pas tool-caller renvoient simplement du contenu → dégradation douce.
    pub async fn chat_tools(
        &self,
        messages: &[serde_json::Value],
        tools: &serde_json::Value,
    ) -> Result<AgentTurn> {
        let url = format!("{}/chat/completions", self.base_url);
        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "stream": false,
        });
        // On n'ajoute `tools` que s'il y en a (certains serveurs rejettent un tableau vide).
        if tools.as_array().map(|a| !a.is_empty()).unwrap_or(false) {
            body["tools"] = tools.clone();
            body["tool_choice"] = json!("auto");
        }
        let mut req = self.http.post(&url).json(&body);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let resp = req.send().await.context("appel /chat/completions (agent)")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("/chat/completions a renvoyé {status}: {text}"));
        }
        let value: serde_json::Value = resp.json().await.context("parsing de la réponse agent")?;
        let msg = value
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .cloned()
            .unwrap_or_else(|| json!({}));

        let content = msg
            .get("content")
            .and_then(|c| c.as_str())
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty());

        let mut tool_calls = Vec::new();
        if let Some(tcs) = msg.get("tool_calls").and_then(|t| t.as_array()) {
            for tc in tcs {
                let id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let func = tc.get("function");
                let name = func
                    .and_then(|f| f.get("name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                // `arguments` est une CHAÎNE JSON (spec OpenAI) — parfois un objet selon
                // le serveur : on gère les deux.
                let arguments = func
                    .and_then(|f| f.get("arguments"))
                    .map(|a| match a {
                        serde_json::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    })
                    .unwrap_or_else(|| "{}".to_string());
                if !name.is_empty() {
                    tool_calls.push(ToolCallOut { id, name, arguments });
                }
            }
        }

        Ok(AgentTurn { content, tool_calls, raw_message: msg })
    }

    /// Décrit une image (base64) via un modèle multimodal — l'« extraction du sens » visuelle.
    ///
    /// L'erreur est typée (`AiCallError`) pour que l'appelant distingue un échec
    /// passager (à re-tenter) d'un échec définitif (repli contextuel immédiat).
    pub async fn describe_image(
        &self,
        image_base64: &str,
        mime: &str,
        prompt: &str,
    ) -> std::result::Result<String, AiCallError> {
        // Le base64 pèse ~4/3 de l'image d'origine : on rapporte la taille réelle.
        let bytes = (image_base64.len() as u64) * 3 / 4;
        let started = std::time::Instant::now();
        let out = self.describe_image_inner(image_base64, mime, prompt).await;
        match &out {
            Ok(_) => {
                crate::metrics::record(crate::metrics::Stage::Vision, started.elapsed(), 1, bytes)
            }
            Err(_) => crate::metrics::record_error(crate::metrics::Stage::Vision),
        }
        out
    }

    async fn describe_image_inner(
        &self,
        image_base64: &str,
        mime: &str,
        prompt: &str,
    ) -> std::result::Result<String, AiCallError> {
        let url = format!("{}/chat/completions", self.base_url);
        let data_url = format!("data:{mime};base64,{image_base64}");
        let mut body = json!({
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
        // Même effort de raisonnement que `chat()` : celui du créneau, choisi par
        // l'utilisateur. Il était ignoré ici — le réglage « Raisonnement » de la
        // section Vision s'affichait et se sauvegardait, mais ne changeait rien.
        // L'enjeu est mesurable : sur une image réelle, 32,5 s avec raisonnement
        // contre 6,9 s sans, pour 7 306 caractères de réflexion précédant 226
        // caractères de réponse.
        if let Some(e) = self.effort.as_param() {
            body["reasoning_effort"] = json!(e);
        }
        let mut req = self.http.post(&url).json(&body).timeout(VISION_TIMEOUT);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        // Erreur d'envoi (timeout, connexion, DNS) : on considère l'appel passager.
        let resp = req
            .send()
            .await
            .map_err(|e| AiCallError::Transient(anyhow!("appel vision /chat/completions: {e}")))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            let err = anyhow!("vision a renvoyé {status}: {text}");
            return Err(AiCallError::from_status(status, err));
        }

        let value = resp
            .json()
            .await
            .map_err(|e| AiCallError::Transient(anyhow!("parsing de la réponse vision: {e}")))?;
        parse_chat_content(value).map_err(AiCallError::Permanent)
    }
}

/// Client de transcription audio/vidéo.
///
/// Deux principes, qui sont la raison d'être de cette structure :
///
/// 1. **Aucune hypothèse sur le média.** Le fichier part TEL QUEL, quel que soit
///    son format et sa taille ; c'est le serveur qui accepte ou refuse. Un refus
///    remonte en [`AiCallError::Permanent`] et l'appelant dégrade proprement.
/// 2. **Aucune hypothèse sur le serveur.** Chemin de l'endpoint, `response_format`,
///    champs supplémentaires et délai viennent tous de la configuration.
///
/// Le corps est téléversé EN FLUX : un fichier de plusieurs Go n'est jamais
/// chargé en mémoire, donc la taille du média n'est bornée par rien côté app.
pub struct TranscriptionClient {
    http: reqwest::Client,
    base_url: String,
    endpoint_path: String,
    model: String,
    api_key: String,
    language: String,
    response_format: String,
    extra_fields: String,
    timeout: Duration,
}

impl TranscriptionClient {
    pub fn new(http: reqwest::Client, cfg: &TranscriptionConfig) -> Self {
        TranscriptionClient {
            http,
            base_url: cfg.base_url.trim_end_matches('/').to_string(),
            endpoint_path: normalise_chemin(&cfg.endpoint_path, "/audio/transcriptions"),
            model: cfg.model.clone(),
            api_key: cfg.api_key.clone(),
            language: cfg.language.clone(),
            response_format: cfg.response_format.clone(),
            extra_fields: cfg.extra_fields.clone(),
            timeout: Duration::from_secs(cfg.timeout_secs.max(1)),
        }
    }

    /// Transcrit le média situé à `path`. `file_name` et `mime` sont transmis au
    /// serveur, qui s'en sert pour choisir son démuxeur.
    pub async fn transcribe(
        &self,
        path: &str,
        file_name: &str,
        mime: &str,
    ) -> std::result::Result<String, AiCallError> {
        let taille = tokio::fs::metadata(path).await.map(|m| m.len()).unwrap_or(0);
        let started = std::time::Instant::now();
        let out = self.transcribe_inner(path, file_name, mime).await;
        match &out {
            Ok(_) => crate::metrics::record(
                crate::metrics::Stage::Media,
                started.elapsed(),
                1,
                taille,
            ),
            Err(_) => crate::metrics::record_error(crate::metrics::Stage::Media),
        }
        out
    }

    /// Ouvre le fichier et en fait un corps de requête en flux.
    async fn flux(&self, path: &str) -> std::result::Result<(reqwest::Body, u64), AiCallError> {
        let fichier = tokio::fs::File::open(path)
            .await
            .map_err(|e| AiCallError::Permanent(anyhow!("ouverture de {path} : {e}")))?;
        let taille = fichier
            .metadata()
            .await
            .map_err(|e| AiCallError::Permanent(anyhow!("taille de {path} : {e}")))?
            .len();
        let corps = reqwest::Body::wrap_stream(tokio_util::io::ReaderStream::new(fichier));
        Ok((corps, taille))
    }

    async fn transcribe_inner(
        &self,
        path: &str,
        file_name: &str,
        mime: &str,
    ) -> std::result::Result<String, AiCallError> {
        let url = format!("{}{}", self.base_url, self.endpoint_path);

        let (corps, taille) = self.flux(path).await?;
        let part =
            reqwest::multipart::Part::stream_with_length(corps, taille).file_name(file_name.to_string());
        // Un type MIME que le client refuse ne doit pas empêcher l'envoi : le serveur
        // sait souvent reconnaître le conteneur seul, on le laisse juger. Le flux est
        // consommé par la tentative, d'où la réouverture.
        let part = match part.mime_str(mime) {
            Ok(p) => p,
            Err(_) => {
                tracing::debug!("type MIME {mime} refusé par le client, envoi sans en-tête de type");
                let (corps, taille) = self.flux(path).await?;
                reqwest::multipart::Part::stream_with_length(corps, taille)
                    .file_name(file_name.to_string())
            }
        };

        let mut form = reqwest::multipart::Form::new()
            .part("file", part)
            .text("model", self.model.clone());
        // Chaque champ optionnel n'est envoyé QUE s'il est renseigné : un champ vide
        // est parfois refusé, et le défaut du serveur vaut mieux que le nôtre.
        if !self.language.trim().is_empty() {
            form = form.text("language", self.language.trim().to_string());
        }
        if !self.response_format.trim().is_empty() {
            form = form.text("response_format", self.response_format.trim().to_string());
        }
        for (cle, valeur) in champs_supplementaires(&self.extra_fields) {
            form = form.text(cle, valeur);
        }

        let mut req = self.http.post(&url).multipart(form).timeout(self.timeout);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| AiCallError::Transient(anyhow!("appel {url} : {e}")))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(AiCallError::from_status(
                status,
                anyhow!("la transcription a renvoyé {status} : {text}"),
            ));
        }

        let corps = resp.text().await.map_err(|e| {
            AiCallError::Transient(anyhow!("lecture de la réponse de transcription : {e}"))
        })?;
        Ok(parse_transcription(&corps))
    }
}

/// Client de DESCRIPTION VISUELLE d'une vidéo, complémentaire de la transcription :
/// il dit ce qu'on VOIT là où la transcription dit ce qui se DIT.
///
/// Passe par `/chat/completions` avec une part `video_url` — la convention des
/// serveurs qui savent lire une vidéo (vLLM servant un modèle type Qwen-VL,
/// passerelles compatibles OpenAI).
///
/// Comme la transcription, ce chemin est **entièrement streamé** : la vidéo n'est
/// jamais chargée en mémoire. Le corps JSON est assemblé à la main autour d'un
/// flux base64 du fichier — c'est possible parce que le base64 s'encode par
/// groupes de 3 octets indépendants, et que son alphabet (`A-Za-z0-9+/=`) ne
/// contient aucun caractère à échapper en JSON. `Content-Length` reste calculable
/// exactement, donc les serveurs qui refusent le chunked encoding fonctionnent.
pub struct VideoClient {
    http: reqwest::Client,
    base_url: String,
    endpoint_path: String,
    model: String,
    api_key: String,
    delivery: VideoDelivery,
    timeout: Duration,
}

/// Taille de lecture : multiple de 3 pour que chaque bloc s'encode sans reste.
const BLOC_LECTURE: usize = 96 * 1024;

/// Longueur du base64 de `n` octets (padding compris).
fn longueur_base64(n: u64) -> u64 {
    4 * n.div_ceil(3)
}

impl VideoClient {
    pub fn new(http: reqwest::Client, cfg: &VideoConfig) -> Self {
        VideoClient {
            http,
            base_url: cfg.base_url.trim_end_matches('/').to_string(),
            endpoint_path: normalise_chemin(&cfg.endpoint_path, "/chat/completions"),
            model: cfg.model.clone(),
            api_key: cfg.api_key.clone(),
            delivery: cfg.delivery,
            timeout: Duration::from_secs(cfg.timeout_secs.max(1)),
        }
    }

    pub async fn describe(
        &self,
        path: &str,
        mime: &str,
        prompt: &str,
    ) -> std::result::Result<String, AiCallError> {
        let taille = tokio::fs::metadata(path).await.map(|m| m.len()).unwrap_or(0);
        let started = std::time::Instant::now();
        let out = self.describe_inner(path, mime, prompt).await;
        match &out {
            Ok(_) => {
                crate::metrics::record(crate::metrics::Stage::Media, started.elapsed(), 1, taille)
            }
            Err(_) => crate::metrics::record_error(crate::metrics::Stage::Media),
        }
        out
    }

    async fn describe_inner(
        &self,
        path: &str,
        mime: &str,
        prompt: &str,
    ) -> std::result::Result<String, AiCallError> {
        let url = format!("{}{}", self.base_url, self.endpoint_path);
        let mut req = self.http.post(&url).timeout(self.timeout);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }

        req = match self.delivery {
            // Le serveur va chercher le fichier lui-même : rien ne transite, ni en
            // mémoire ni sur le réseau. Suppose qu'il voit le même système de
            // fichiers (serveur local, avec l'accès aux médias locaux autorisé).
            VideoDelivery::FileUri => {
                let uri = uri_fichier(path);
                req.json(&self.corps_json(prompt, &uri))
            }
            VideoDelivery::Base64 => {
                let fichier = tokio::fs::File::open(path)
                    .await
                    .map_err(|e| AiCallError::Permanent(anyhow!("ouverture de {path} : {e}")))?;
                let taille = fichier
                    .metadata()
                    .await
                    .map_err(|e| AiCallError::Permanent(anyhow!("taille de {path} : {e}")))?
                    .len();

                let (prefixe, suffixe) = self.enveloppe_json(prompt, mime);
                let longueur =
                    prefixe.len() as u64 + longueur_base64(taille) + suffixe.len() as u64;
                let flux = flux_json_base64(prefixe, fichier, suffixe);
                req.header(reqwest::header::CONTENT_TYPE, "application/json")
                    .header(reqwest::header::CONTENT_LENGTH, longueur)
                    .body(reqwest::Body::wrap_stream(flux))
            }
        };

        let resp = req
            .send()
            .await
            .map_err(|e| AiCallError::Transient(anyhow!("appel {url} : {e}")))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(AiCallError::from_status(
                status,
                anyhow!("la description vidéo a renvoyé {status} : {text}"),
            ));
        }
        let value = resp.json().await.map_err(|e| {
            AiCallError::Transient(anyhow!("parsing de la réponse de description vidéo : {e}"))
        })?;
        parse_chat_content(value).map_err(AiCallError::Permanent)
    }

    /// Corps complet, pour le mode `file_uri` où l'URL tient en quelques octets.
    fn corps_json(&self, prompt: &str, url_media: &str) -> serde_json::Value {
        json!({
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": prompt },
                    { "type": "video_url", "video_url": { "url": url_media } }
                ]
            }],
            "temperature": 0.1,
            "stream": false,
        })
    }

    /// Les deux moitiés du corps JSON qui encadrent le flux base64.
    ///
    /// Construites via `serde_json` pour que le modèle, le prompt et le type MIME
    /// soient correctement échappés — les assembler à la main serait une faille
    /// d'injection dès qu'un prompt contient un guillemet.
    fn enveloppe_json(&self, prompt: &str, mime: &str) -> (Vec<u8>, Vec<u8>) {
        let modele = serde_json::to_string(&self.model).unwrap_or_else(|_| "\"\"".to_string());
        let texte = serde_json::to_string(prompt).unwrap_or_else(|_| "\"\"".to_string());
        // Chaîne JSON complète du début d'URL, dont on retire le guillemet FERMANT :
        // le flux base64 vient s'y coller, et le suffixe le refermera.
        let debut_url = serde_json::to_string(&format!("data:{mime};base64,"))
            .unwrap_or_else(|_| "\"\"".to_string());
        let debut_url = &debut_url[..debut_url.len() - 1];

        let prefixe = format!(
            concat!(
                "{{\"model\":{},\"messages\":[{{\"role\":\"user\",\"content\":[",
                "{{\"type\":\"text\",\"text\":{}}},",
                "{{\"type\":\"video_url\",\"video_url\":{{\"url\":{}"
            ),
            modele, texte, debut_url
        );
        let suffixe = "\"}}]}],\"temperature\":0.1,\"stream\":false}";
        (prefixe.into_bytes(), suffixe.as_bytes().to_vec())
    }
}

/// `file://` URI d'un chemin local, séparateurs Windows normalisés.
fn uri_fichier(path: &str) -> String {
    let normalise = path.replace('\\', "/");
    if normalise.starts_with('/') {
        format!("file://{normalise}")
    } else {
        // Chemin Windows (`C:/...`) : le troisième slash est la racine.
        format!("file:///{normalise}")
    }
}

/// État de la machine qui produit le corps : préfixe, puis le fichier encodé en
/// base64 par blocs, puis le suffixe.
struct EtatCorps {
    prefixe: Option<Vec<u8>>,
    fichier: tokio::fs::File,
    /// Octets non encodés du bloc précédent (0 à 2) : le base64 travaille par
    /// groupes de 3, on reporte le reliquat au bloc suivant.
    reste: Vec<u8>,
    suffixe: Option<Vec<u8>>,
    fini: bool,
}

/// Flux du corps JSON complet, sans jamais tenir le média en mémoire.
fn flux_json_base64(
    prefixe: Vec<u8>,
    fichier: tokio::fs::File,
    suffixe: Vec<u8>,
) -> impl futures_util::Stream<Item = std::io::Result<Vec<u8>>> {
    use base64::Engine;
    use tokio::io::AsyncReadExt;

    let etat = EtatCorps {
        prefixe: Some(prefixe),
        fichier,
        reste: Vec::with_capacity(3),
        suffixe: Some(suffixe),
        fini: false,
    };

    futures_util::stream::unfold(etat, |mut e| async move {
        if let Some(p) = e.prefixe.take() {
            return Some((Ok(p), e));
        }
        if !e.fini {
            let mut tampon = vec![0u8; BLOC_LECTURE];
            match e.fichier.read(&mut tampon).await {
                Err(err) => {
                    e.fini = true;
                    return Some((Err(err), e));
                }
                Ok(0) => {
                    e.fini = true;
                    // Dernier groupe : c'est ici, et seulement ici, qu'apparaît le
                    // padding `=`. On y accroche le suffixe pour finir en un envoi.
                    let mut sortie = base64::engine::general_purpose::STANDARD
                        .encode(&e.reste)
                        .into_bytes();
                    e.reste.clear();
                    if let Some(s) = e.suffixe.take() {
                        sortie.extend_from_slice(&s);
                    }
                    return Some((Ok(sortie), e));
                }
                Ok(n) => {
                    e.reste.extend_from_slice(&tampon[..n]);
                    // On n'encode que les groupes de 3 complets ; le reliquat attend.
                    let coupe = (e.reste.len() / 3) * 3;
                    let a_encoder: Vec<u8> = e.reste.drain(..coupe).collect();
                    let sortie = base64::engine::general_purpose::STANDARD
                        .encode(&a_encoder)
                        .into_bytes();
                    return Some((Ok(sortie), e));
                }
            }
        }
        if let Some(s) = e.suffixe.take() {
            return Some((Ok(s), e));
        }
        None
    })
}

/// Normalise un chemin d'endpoint saisi par l'utilisateur : vide → défaut, et on
/// garantit le `/` initial pour que la concaténation avec `base_url` tienne.
fn normalise_chemin(saisi: &str, defaut: &str) -> String {
    let t = saisi.trim();
    if t.is_empty() {
        return defaut.to_string();
    }
    if t.starts_with('/') {
        t.to_string()
    } else {
        format!("/{t}")
    }
}

/// Champs multipart supplémentaires, lus depuis un objet JSON `{"clé": "valeur"}`.
///
/// C'est l'échappatoire qui évite de modifier le code pour un serveur attendant un
/// paramètre exotique. Une saisie invalide est IGNORÉE avec un avertissement :
/// faire échouer toute l'indexation à cause d'un réglage avancé mal formé serait pire.
fn champs_supplementaires(brut: &str) -> Vec<(String, String)> {
    let t = brut.trim();
    if t.is_empty() {
        return Vec::new();
    }
    match serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(t) {
        Ok(map) => map
            .into_iter()
            .map(|(k, v)| {
                // Le multipart ne transporte que du texte : une valeur JSON non
                // textuelle (nombre, booléen) est rendue telle quelle.
                let s = match v {
                    serde_json::Value::String(s) => s,
                    autre => autre.to_string(),
                };
                (k, s)
            })
            .collect(),
        Err(e) => {
            tracing::warn!("champs supplémentaires ignorés (JSON invalide) : {e}");
            Vec::new()
        }
    }
}

/// Extrait le texte d'une réponse de transcription.
///
/// Le format par défaut est `{"text": "..."}`, mais plusieurs serveurs locaux
/// répondent en texte brut. On accepte les deux plutôt que de perdre une
/// transcription réussie sur un détail de sérialisation.
fn parse_transcription(corps: &str) -> String {
    serde_json::from_str::<serde_json::Value>(corps)
        .ok()
        .and_then(|v| v.get("text").and_then(|t| t.as_str()).map(str::to_string))
        .unwrap_or_else(|| corps.to_string())
        .trim()
        .to_string()
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
    pub transcription_ok: bool,
    pub transcription_detail: String,
    pub video_ok: bool,
    pub video_detail: String,
}

struct CachedEmbedder {
    key: String,
    provider: Arc<dyn EmbeddingProvider>,
}

struct CachedReranker {
    key: String,
    provider: Arc<Reranker>,
}

/// Point d'entrée unique vers les modèles, piloté par la configuration.
pub struct AiEngine {
    config: Arc<ConfigStore>,
    http: reqwest::Client,
    embedder: tokio::sync::Mutex<Option<CachedEmbedder>>,
    reranker: tokio::sync::Mutex<Option<CachedReranker>>,
    /// Embedder CLIP (recherche d'images), chargé à la demande.
    clip: tokio::sync::Mutex<Option<Arc<ClipEmbedder>>>,
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
            reranker: tokio::sync::Mutex::new(None),
            clip: tokio::sync::Mutex::new(None),
            data_dir,
            ort_ready: tokio::sync::OnceCell::new(),
        }
    }

    /// Embedder CLIP (chargé à la demande, ORT prêt). Partagé par l'indexation d'images
    /// et la recherche visuelle.
    pub async fn clip(&self) -> Result<Arc<ClipEmbedder>> {
        {
            let guard = self.clip.lock().await;
            if let Some(c) = guard.as_ref() {
                return Ok(c.clone());
            }
        }
        self.ensure_ort_ready().await?;
        let cache = local_cache_dir(&self.data_dir);
        let clip = tokio::task::spawn_blocking(move || ClipEmbedder::load(cache))
            .await
            .context("tâche de chargement CLIP interrompue")??;
        let clip = Arc::new(clip);
        *self.clip.lock().await = Some(clip.clone());
        Ok(clip)
    }

    /// Garantit qu'ONNX Runtime (ORT_DYLIB_PATH) est prêt, une seule fois. Partagé
    /// par l'embedder local ET le reranker (tous deux via ORT).
    async fn ensure_ort_ready(&self) -> Result<()> {
        let dir = self.data_dir.clone();
        self.ort_ready
            .get_or_try_init(|| async move {
                tokio::task::spawn_blocking(move || crate::ort_setup::ensure_ort(&dir, false))
                    .await
                    .context("tâche de préparation ORT interrompue")?
                    .map(|_gpu| ())
            })
            .await?;
        Ok(())
    }

    /// Renvoie le reranker courant (chargé paresseusement, mis en cache). Reconstruit
    /// si le modèle configuré change.
    pub async fn reranker(&self) -> Result<Arc<Reranker>> {
        let key = self.config.snapshot().retrieval.reranker_model;
        let mut guard = self.reranker.lock().await;
        if let Some(c) = guard.as_ref() {
            if c.key == key {
                return Ok(c.provider.clone());
            }
        }
        self.ensure_ort_ready().await?;
        let cache = local_cache_dir(&self.data_dir);
        let name = key.clone();
        let reranker = tokio::task::spawn_blocking(move || Reranker::load(&name, cache))
            .await
            .context("tâche de chargement du reranker interrompue")??;
        let reranker = Arc::new(reranker);
        *guard = Some(CachedReranker { key, provider: reranker.clone() });
        Ok(reranker)
    }

    /// Libère le reranker chargé (fermeture / changement de modèle).
    pub async fn invalidate_reranker(&self) {
        *self.reranker.lock().await = None;
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
            EmbeddingMode::Openai => Arc::new(OpenAiEmbedder::new(
                self.http.clone(),
                &cfg.embedding,
                embedding_num_ctx(cfg.indexing.chunk_size),
                cfg.indexing.batch_size,
            )),
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
    /// Demande aux serveurs de modèles de DÉCHARGER les modèles utilisés par SenseTree.
    ///
    /// Indispensable à la fermeture : Ollama est un processus SÉPARÉ qui garde les
    /// modèles en mémoire (`keep_alive`) bien après l'arrêt de l'app — plusieurs Go
    /// de RAM/VRAM immobilisés pour rien. L'API native `/api/generate` avec
    /// `keep_alive: 0` les libère immédiatement.
    ///
    /// Best-effort : un serveur qui ne connaît pas cette API (LM Studio, API externe)
    /// renvoie une erreur qu'on ignore silencieusement.
    pub async fn unload_remote_models(&self) {
        let cfg = self.config.snapshot();
        let mut targets: Vec<(String, String)> = Vec::new();
        if matches!(cfg.embedding.mode, EmbeddingMode::Openai) {
            targets.push((cfg.embedding.base_url.clone(), cfg.embedding.model.clone()));
        }
        if cfg.reasoning.enabled {
            targets.push((cfg.reasoning.base_url.clone(), cfg.reasoning.model.clone()));
        }
        if cfg.vision.enabled {
            targets.push((cfg.vision.base_url.clone(), cfg.vision.model.clone()));
        }

        for (base, model) in targets {
            if model.trim().is_empty() || base.trim().is_empty() {
                continue;
            }
            // `http://host:11434/v1` → racine de l'API native `http://host:11434`.
            let root = base.trim_end_matches('/').trim_end_matches("/v1").trim_end_matches('/');
            let body = json!({ "model": model, "keep_alive": 0 });
            let res = self
                .http
                .post(format!("{root}/api/generate"))
                .json(&body)
                .timeout(Duration::from_secs(5))
                .send()
                .await;
            match res {
                Ok(r) if r.status().is_success() => {
                    tracing::info!("🧹 modèle déchargé du serveur : {model}")
                }
                Ok(r) => tracing::debug!("déchargement ignoré ({}) pour {model}", r.status()),
                Err(e) => tracing::debug!("déchargement impossible pour {model}: {e}"),
            }
        }
    }

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

    pub fn transcription_client(&self) -> TranscriptionClient {
        let cfg = self.config.snapshot();
        TranscriptionClient::new(self.http.clone(), &cfg.transcription)
    }

    pub fn video_client(&self) -> VideoClient {
        let cfg = self.config.snapshot();
        VideoClient::new(self.http.clone(), &cfg.video)
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
            match self.ping_models(&cfg.reasoning.base_url, &cfg.reasoning.api_key).await {
                Ok(_) => (true, "connecté".to_string()),
                Err(e) => (false, format!("{e}")),
            }
        } else {
            (false, "désactivé".to_string())
        };

        // Vision
        let (vision_ok, vision_detail) = if cfg.vision.enabled {
            match self.ping_models(&cfg.vision.base_url, &cfg.vision.api_key).await {
                Ok(_) => (true, "connecté".to_string()),
                Err(e) => (false, format!("{e}")),
            }
        } else {
            (false, "désactivé".to_string())
        };

        // Transcription et description vidéo : deux serveurs distincts et
        // indépendamment activables, donc deux voyants distincts. Les confondre
        // masquerait lequel des deux est en panne.
        let (transcription_ok, transcription_detail) = if cfg.transcription.enabled {
            match self
                .ping_media(&cfg.transcription.base_url, &cfg.transcription.api_key)
                .await
            {
                Ok(d) => (true, d),
                Err(e) => (false, format!("{e}")),
            }
        } else {
            (false, "désactivé".to_string())
        };

        let (video_ok, video_detail) = if cfg.video.enabled {
            match self.ping_media(&cfg.video.base_url, &cfg.video.api_key).await {
                Ok(d) => (true, d),
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
            transcription_ok,
            transcription_detail,
            video_ok,
            video_detail,
        }
    }

    /// Ping léger d'un endpoint compatible OpenAI (`GET /models`).
    async fn ping_models(&self, base_url: &str, api_key: &str) -> Result<()> {
        let resp = self.requete_models(base_url, api_key).await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(anyhow!("le serveur a répondu {}", resp.status()))
        }
    }

    /// Ping d'un serveur média (transcription, description vidéo).
    ///
    /// Plus tolérant que [`Self::ping_models`] à dessein : plusieurs serveurs de
    /// transcription — whisper.cpp au premier chef — n'exposent PAS `/models`. Un
    /// 404 prouve pourtant que le serveur répond. L'afficher en rouge serait un
    /// faux négatif, c'est-à-dire pire qu'un indicateur absent : l'utilisateur
    /// irait chercher une panne inexistante.
    async fn ping_media(&self, base_url: &str, api_key: &str) -> Result<String> {
        let resp = self.requete_models(base_url, api_key).await?;
        let statut = resp.status();
        if statut.is_success() {
            Ok("connecté".to_string())
        } else if statut == reqwest::StatusCode::NOT_FOUND {
            Ok("connecté (pas d'inventaire /models)".to_string())
        } else {
            Err(anyhow!("le serveur a répondu {statut}"))
        }
    }

    async fn requete_models(&self, base_url: &str, api_key: &str) -> Result<reqwest::Response> {
        let url = format!("{}/models", base_url.trim_end_matches('/'));
        let mut req = self.http.get(&url).timeout(Duration::from_secs(5));
        if !api_key.is_empty() {
            req = req.bearer_auth(api_key);
        }
        req.send().await.context("serveur injoignable")
    }
}

#[cfg(test)]
mod tests {
    use super::{
        champs_supplementaires, embedding_num_ctx, normalise_chemin, parse_transcription,
        resolve_local_model, supported_local_models, TranscriptionClient,
    };
    use crate::config::TranscriptionConfig;

    /// Les serveurs de transcription ne répondent pas tous pareil : l'API de
    /// référence renvoie `{"text": ...}`, plusieurs implémentations locales
    /// renvoient le texte brut. Perdre une transcription réussie sur ce détail
    /// serait absurde, on accepte les deux.
    #[test]
    fn transcription_lue_en_json_comme_en_texte_brut() {
        assert_eq!(
            parse_transcription(r#"{"text": "  bonjour le monde  "}"#),
            "bonjour le monde"
        );
        assert_eq!(parse_transcription("  bonjour le monde\n"), "bonjour le monde");
        // JSON sans champ `text` : on garde le corps plutôt que de renvoyer du vide.
        assert_eq!(parse_transcription(r#"{"autre": 1}"#), r#"{"autre": 1}"#);
        assert_eq!(parse_transcription("   "), "");
    }

    /// Le chemin d'endpoint est saisi à la main : il doit tolérer les deux formes
    /// et retomber sur le défaut plutôt que de produire une URL cassée.
    #[test]
    fn chemin_d_endpoint_normalise() {
        assert_eq!(normalise_chemin("", "/audio/transcriptions"), "/audio/transcriptions");
        assert_eq!(normalise_chemin("   ", "/defaut"), "/defaut");
        assert_eq!(normalise_chemin("/v2/asr", "/defaut"), "/v2/asr");
        // Sans slash initial, la concaténation avec base_url collerait les segments.
        assert_eq!(normalise_chemin("v2/asr", "/defaut"), "/v2/asr");
    }

    /// Échappatoire pour les serveurs exotiques : un JSON invalide ne doit JAMAIS
    /// faire échouer l'indexation, seulement être ignoré.
    #[test]
    fn champs_supplementaires_tolerants() {
        assert!(champs_supplementaires("").is_empty());
        assert!(champs_supplementaires("pas du json").is_empty());
        assert!(champs_supplementaires("[1,2]").is_empty());

        let mut c = champs_supplementaires(r#"{"temperature": "0", "beam": 5}"#);
        c.sort();
        assert_eq!(
            c,
            vec![
                ("beam".to_string(), "5".to_string()),
                ("temperature".to_string(), "0".to_string()),
            ]
        );
    }

    /// La longueur annoncée en `Content-Length` doit être exacte : trop courte, le
    /// corps est tronqué ; trop longue, le serveur attend indéfiniment.
    #[test]
    fn longueur_base64_exacte() {
        use super::longueur_base64;
        assert_eq!(longueur_base64(0), 0);
        assert_eq!(longueur_base64(1), 4);
        assert_eq!(longueur_base64(2), 4);
        assert_eq!(longueur_base64(3), 4);
        assert_eq!(longueur_base64(4), 8);
        assert_eq!(longueur_base64(200_000), 266_668);
    }

    #[test]
    fn uri_fichier_windows_et_posix() {
        use super::uri_fichier;
        assert_eq!(
            uri_fichier(r"C:\Users\a\film.mp4"),
            "file:///C:/Users/a/film.mp4"
        );
        assert_eq!(uri_fichier("/home/a/film.mp4"), "file:///home/a/film.mp4");
    }

    /// Serveur HTTP minimal : renvoie `reponse` et rend la requête reçue.
    fn serveur_bidon(
        reponse: &'static str,
    ) -> (std::net::SocketAddr, std::thread::JoinHandle<String>) {
        use std::io::{Read, Write};
        let ecoute = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let adresse = ecoute.local_addr().expect("adresse");
        let poignee = std::thread::spawn(move || {
            let (mut flux, _) = ecoute.accept().expect("accept");
            flux.set_read_timeout(Some(std::time::Duration::from_secs(10))).ok();
            // On lit les en-têtes puis exactement `content-length` octets : la
            // connexion reste ouverte (keep-alive), attendre l'EOF bloquerait.
            let mut brut: Vec<u8> = Vec::new();
            let mut tampon = [0u8; 8192];
            loop {
                match flux.read(&mut tampon) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => brut.extend_from_slice(&tampon[..n]),
                }
                let texte = String::from_utf8_lossy(&brut).to_string();
                if let Some(fin) = texte.find("\r\n\r\n") {
                    let taille: usize = texte[..fin]
                        .lines()
                        .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
                        .and_then(|l| l.split(':').nth(1))
                        .and_then(|v| v.trim().parse().ok())
                        .unwrap_or(0);
                    if brut.len() >= fin + 4 + taille {
                        break;
                    }
                }
            }
            let entete = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                reponse.len()
            );
            flux.write_all(entete.as_bytes()).ok();
            flux.write_all(reponse.as_bytes()).ok();
            flux.flush().ok();
            String::from_utf8_lossy(&brut).to_string()
        });
        (adresse, poignee)
    }

    fn media_temporaire(nom: &str, contenu: &[u8]) -> std::path::PathBuf {
        let chemin = std::env::temp_dir()
            .join(format!("sensetree-media-{}-{nom}", std::process::id()));
        std::fs::write(&chemin, contenu).expect("écriture du média de test");
        chemin
    }

    /// Vérifie la requête RÉELLEMENT émise : c'est la seule façon de s'assurer que
    /// le média part en `multipart/form-data` avec son nom de fichier et les champs
    /// configurés. Une erreur ici ne se verrait qu'en production, sous la forme
    /// d'un 400 opaque.
    #[tokio::test]
    async fn transcribe_emet_un_multipart_conforme() {
        let (adresse, serveur) = serveur_bidon(r#"{"text": "  bonjour ici  "}"#);
        let media = media_temporaire("conforme.mp3", b"FAUX-AUDIO");

        let cfg = TranscriptionConfig {
            base_url: format!("http://{adresse}/v1"),
            model: "whisper-x".to_string(),
            language: "fr".to_string(),
            response_format: "verbose_json".to_string(),
            extra_fields: r#"{"temperature": "0"}"#.to_string(),
            ..TranscriptionConfig::default()
        };
        let client = TranscriptionClient::new(reqwest::Client::new(), &cfg);
        let texte = client
            .transcribe(media.to_str().unwrap(), "reunion equipe.mp3", "audio/mpeg")
            .await
            .expect("transcription");
        let _ = std::fs::remove_file(&media);

        let requete = serveur.join().expect("serveur");
        assert!(
            requete.starts_with("POST /v1/audio/transcriptions "),
            "mauvais chemin : {}",
            requete.lines().next().unwrap_or_default()
        );
        assert!(
            requete.to_ascii_lowercase().contains("content-type: multipart/form-data"),
            "pas du multipart"
        );
        // Le nom de fichier compte : des serveurs s'en servent pour choisir leur démuxeur.
        assert!(requete.contains("filename=\"reunion equipe.mp3\""), "nom de fichier absent");
        assert!(requete.contains("whisper-x"), "modèle absent");
        assert!(requete.contains("verbose_json"), "response_format absent");
        assert!(requete.contains("temperature"), "champ supplémentaire absent");
        assert!(requete.contains("FAUX-AUDIO"), "contenu du média absent");
        assert_eq!(texte, "bonjour ici");
    }

    /// Les champs optionnels non renseignés ne doivent PAS être envoyés : un champ
    /// vide est refusé par certains serveurs, et leur défaut vaut mieux que le nôtre.
    #[tokio::test]
    async fn transcribe_omet_les_champs_non_renseignes() {
        let (adresse, serveur) = serveur_bidon("texte brut sans json");
        let media = media_temporaire("minimal.wav", b"x");

        let cfg = TranscriptionConfig {
            base_url: format!("http://{adresse}/v1"),
            language: String::new(),
            response_format: String::new(),
            extra_fields: String::new(),
            ..TranscriptionConfig::default()
        };
        let client = TranscriptionClient::new(reqwest::Client::new(), &cfg);
        let texte = client
            .transcribe(media.to_str().unwrap(), "a.wav", "audio/wav")
            .await
            .expect("transcription");
        let _ = std::fs::remove_file(&media);

        let requete = serveur.join().expect("serveur");
        assert!(!requete.contains("name=\"language\""), "langue vide envoyée");
        assert!(
            !requete.contains("name=\"response_format\""),
            "response_format vide envoyé"
        );
        // Réponse en texte brut : acceptée telle quelle.
        assert_eq!(texte, "texte brut sans json");
    }

    /// Un type MIME inconnu ne doit pas bloquer l'envoi : c'est au serveur de juger
    /// ce qu'il sait lire, pas au client.
    #[tokio::test]
    async fn transcribe_envoie_meme_avec_un_mime_inconnu() {
        let (adresse, serveur) = serveur_bidon(r#"{"text": "ok"}"#);
        let media = media_temporaire("exotique.xyz", b"CONTENU-EXOTIQUE");

        let cfg = TranscriptionConfig {
            base_url: format!("http://{adresse}/v1"),
            ..TranscriptionConfig::default()
        };
        let client = TranscriptionClient::new(reqwest::Client::new(), &cfg);
        let texte = client
            .transcribe(media.to_str().unwrap(), "exotique.xyz", "ceci n'est pas un mime")
            .await
            .expect("transcription");
        let _ = std::fs::remove_file(&media);

        let requete = serveur.join().expect("serveur");
        assert!(requete.contains("CONTENU-EXOTIQUE"), "le média n'a pas été envoyé");
        assert_eq!(texte, "ok");
    }

    /// Le corps JSON de la description vidéo est assemblé À LA MAIN autour d'un flux
    /// base64 : c'est ce qui permet de ne jamais charger la vidéo en mémoire, mais
    /// aussi l'endroit où une erreur passerait inaperçue jusqu'en production.
    ///
    /// Ce test le vérifie de bout en bout : le média fait 200 000 octets — donc
    /// PLUSIEURS blocs de lecture et une taille non multiple de 3, ce qui exerce le
    /// report du reliquat entre blocs — et l'on décode le base64 reçu pour le
    /// comparer à l'original, octet par octet.
    #[tokio::test]
    async fn description_video_streame_un_json_valide() {
        use crate::config::{VideoConfig, VideoDelivery};
        use base64::Engine;

        let (adresse, serveur) = serveur_bidon(
            r#"{"choices":[{"message":{"content":"un chat sur un clavier"}}]}"#,
        );

        // Contenu déterministe : une erreur d'encodage se verrait immédiatement.
        let original: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
        assert_ne!(original.len() % 3, 0, "taille volontairement non multiple de 3");
        let media = media_temporaire("film.mp4", &original);

        let cfg = VideoConfig {
            base_url: format!("http://{adresse}/v1"),
            model: "Qwen/Qwen2.5-VL-7B-Instruct".to_string(),
            delivery: VideoDelivery::Base64,
            ..VideoConfig::default()
        };
        let client = super::VideoClient::new(reqwest::Client::new(), &cfg);
        let description = client
            .describe(media.to_str().unwrap(), "video/mp4", "Décris \"cette\" vidéo")
            .await
            .expect("description");
        let _ = std::fs::remove_file(&media);

        assert_eq!(description, "un chat sur un clavier");

        let requete = serveur.join().expect("serveur");
        let corps = requete
            .split("\r\n\r\n")
            .nth(1)
            .expect("corps de requête absent");

        // 1) Le JSON assemblé à la main doit être VALIDE.
        let v: serde_json::Value =
            serde_json::from_str(corps).expect("le corps JSON assemblé n'est pas valide");
        assert_eq!(v["model"], "Qwen/Qwen2.5-VL-7B-Instruct");
        let contenu = &v["messages"][0]["content"];
        // 2) Le prompt est échappé correctement, guillemets compris.
        assert_eq!(contenu[0]["text"], "Décris \"cette\" vidéo");

        // 3) Le média décodé doit être identique à l'original.
        let url = contenu[1]["video_url"]["url"]
            .as_str()
            .expect("video_url absent");
        let prefixe = "data:video/mp4;base64,";
        assert!(url.starts_with(prefixe), "préfixe de data-URL inattendu");
        let decode = base64::engine::general_purpose::STANDARD
            .decode(&url[prefixe.len()..])
            .expect("base64 invalide");
        assert_eq!(decode.len(), original.len(), "taille du média altérée");
        assert_eq!(decode, original, "contenu du média altéré par le streaming");
    }

    /// En mode `file_uri`, RIEN du média ne doit transiter : le serveur va le
    /// chercher lui-même. C'est le mode le plus efficace pour un serveur local.
    #[tokio::test]
    async fn description_video_en_file_uri_n_envoie_pas_le_media() {
        use crate::config::{VideoConfig, VideoDelivery};

        let (adresse, serveur) =
            serveur_bidon(r#"{"choices":[{"message":{"content":"ok"}}]}"#);
        let original = vec![7u8; 50_000];
        let media = media_temporaire("local.mp4", &original);

        let cfg = VideoConfig {
            base_url: format!("http://{adresse}/v1"),
            delivery: VideoDelivery::FileUri,
            ..VideoConfig::default()
        };
        let client = super::VideoClient::new(reqwest::Client::new(), &cfg);
        let description = client
            .describe(media.to_str().unwrap(), "video/mp4", "décris")
            .await
            .expect("description");
        let _ = std::fs::remove_file(&media);

        assert_eq!(description, "ok");
        let requete = serveur.join().expect("serveur");
        // Le corps entier doit rester minuscule devant les 50 Ko du média.
        assert!(
            requete.len() < 2_000,
            "le média a été envoyé alors qu'une URL suffisait ({} octets)",
            requete.len()
        );
        assert!(requete.contains("file:///"), "URL de fichier absente");
        assert!(!requete.contains("base64"), "un base64 a été envoyé malgré file_uri");
    }

    /// Banc d'essai du moteur d'embedding EMBARQUÉ, dans les conditions réelles de
    /// l'indexation : lots de 32 chunks de 1000 caractères, préfixe E5 compris.
    ///
    /// Sert à comparer un modèle local à un modèle servi par un serveur HTTP, sur la
    /// même machine et avec le même découpage. À lancer à la main :
    ///
    /// ```text
    /// EMBED_MODEL=multilingual-e5-large \
    ///   cargo test --lib banc_embedding_local -- --ignored --nocapture
    /// ```
    ///
    /// Le modèle doit déjà être en cache, sinon la première exécution le télécharge.
    #[test]
    #[ignore = "banc d'essai manuel (EMBED_MODEL)"]
    fn banc_embedding_local() {
        use crate::config::{EmbeddingConfig, EmbeddingMode};
        use crate::providers::EmbeddingProvider;

        let model = std::env::var("EMBED_MODEL")
            .unwrap_or_else(|_| "multilingual-e5-small".to_string());
        let data_dir = dirs_data();
        // Même préparation que l'app : la lib ORT est fournie par le dossier de données.
        crate::ort_setup::ensure_ort(&data_dir, false).expect("ONNX Runtime indisponible");

        let cfg = EmbeddingConfig {
            mode: EmbeddingMode::Local,
            model: model.clone(),
            base_url: String::new(),
            api_key: String::new(),
            dimensions: 0,
            use_gpu: false,
        };

        let charge = std::time::Instant::now();
        let emb = super::LocalEmbedder::load(&cfg, 32, super::local_cache_dir(&data_dir))
            .expect("chargement du modèle");
        println!("\n== {model} ==");
        println!("chargement : {:.2} s", charge.elapsed().as_secs_f64());

        // Un chunk représentatif : 1000 caractères, le défaut de `chunk_size`.
        let base = "Le rapport trimestriel detaille la marge brute par segment, les couts de personnel et les investissements en recherche. ";
        let chunk: String = base.repeat(10).chars().take(1000).collect();
        let lot: Vec<String> = std::iter::repeat_n(chunk.clone(), 32).collect();
        let octets: usize = lot.iter().map(|t| t.len()).sum();

        let rt = tokio::runtime::Runtime::new().unwrap();
        // Première passe ignorée : elle porte l'initialisation des threads ORT.
        rt.block_on(emb.embed_documents(lot.clone())).expect("embedding");

        let mut temps = Vec::new();
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            let v = rt.block_on(emb.embed_documents(lot.clone())).expect("embedding");
            temps.push(t0.elapsed().as_secs_f64());
            assert_eq!(v.len(), 32);
            assert_eq!(v[0].len(), emb.dimensions());
        }
        let best = temps.iter().cloned().fold(f64::INFINITY, f64::min);
        println!("dimensions : {}", emb.dimensions());
        println!("lot de 32 x 1000 car. : {temps:.3?} s");
        println!(
            "meilleur   : {:.3} s  ->  {:.1} chunks/s  ·  {:.3} Mo/s",
            best,
            32.0 / best,
            octets as f64 / 1e6 / best
        );
    }

    /// Dossier de données de l'app (même emplacement que l'application installée).
    #[cfg(test)]
    fn dirs_data() -> std::path::PathBuf {
        std::path::PathBuf::from(std::env::var("APPDATA").expect("APPDATA"))
            .join("com.virgi.sensetree")
    }

    /// Le catalogue affiché et le résolveur doivent rester d'accord : un identifiant
    /// listé mais non résolu retomberait silencieusement sur e5-small, et l'utilisateur
    /// indexerait tout avec un modèle qu'il n'a pas choisi.
    #[test]
    fn catalogue_local_coherent_avec_le_resolveur() {
        for (id, dims, multilingue) in supported_local_models() {
            let (_, resolved_dims, e5) = resolve_local_model(id);
            assert_eq!(resolved_dims, dims, "dimension incohérente pour {id}");
            // Le préfixe E5 n'est requis que par la famille E5 — la seule multilingue.
            assert_eq!(e5, multilingue, "drapeau multilingue incohérent pour {id}");
        }
    }
}

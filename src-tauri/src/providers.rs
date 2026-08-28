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

use crate::config::{ChatConfig, ConfigStore, EmbeddingConfig, EmbeddingMode, ReasoningEffort};

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
    pub fn new(http: reqwest::Client, cfg: &EmbeddingConfig, num_ctx: u32) -> Self {
        OpenAiEmbedder {
            http,
            base_url: cfg.base_url.trim_end_matches('/').to_string(),
            model: cfg.model.clone(),
            api_key: cfg.api_key.clone(),
            dimensions: cfg.dimensions,
            num_ctx,
        }
    }

    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
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

/// Nature d'un échec de la vision, pour décider s'il faut re-tenter ou abandonner.
#[derive(Debug)]
pub enum VisionError {
    /// Passager (timeout, serveur occupé/en chargement, 5xx, réseau) → re-tentable.
    Transient(anyhow::Error),
    /// Définitif (image invalide, format refusé, modèle absent) → repli contextuel.
    Permanent(anyhow::Error),
}

impl std::fmt::Display for VisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VisionError::Transient(e) => write!(f, "{e} [passager]"),
            VisionError::Permanent(e) => write!(f, "{e} [définitif]"),
        }
    }
}
impl std::error::Error for VisionError {}

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
    /// L'erreur est typée (`VisionError`) pour que l'appelant distingue un échec
    /// passager (à re-tenter) d'un échec définitif (repli contextuel immédiat).
    pub async fn describe_image(
        &self,
        image_base64: &str,
        mime: &str,
        prompt: &str,
    ) -> std::result::Result<String, VisionError> {
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
    ) -> std::result::Result<String, VisionError> {
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
        let mut req = self.http.post(&url).json(&body).timeout(VISION_TIMEOUT);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        // Erreur d'envoi (timeout, connexion, DNS) : on considère l'appel passager.
        let resp = req
            .send()
            .await
            .map_err(|e| VisionError::Transient(anyhow!("appel vision /chat/completions: {e}")))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            let err = anyhow!("vision a renvoyé {status}: {text}");
            // 5xx / 408 / 429 = serveur occupé, en chargement, ou saturé → re-tentable.
            // Autres 4xx = requête définitivement invalide (image, format, modèle absent).
            return Err(
                if status.is_server_error()
                    || status == reqwest::StatusCode::REQUEST_TIMEOUT
                    || status == reqwest::StatusCode::TOO_MANY_REQUESTS
                {
                    VisionError::Transient(err)
                } else {
                    VisionError::Permanent(err)
                },
            );
        }

        let value = resp
            .json()
            .await
            .map_err(|e| VisionError::Transient(anyhow!("parsing de la réponse vision: {e}")))?;
        parse_chat_content(value).map_err(VisionError::Permanent)
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

#[cfg(test)]
mod tests {
    use super::{embedding_num_ctx, resolve_local_model, supported_local_models};

    /// Le contexte demandé doit couvrir un chunk avec marge, et surtout rester PETIT.
    ///
    /// Mesuré sur un serveur réel : un modèle d'embedding annonçant 32k de contexte
    /// réserve 5,78 Go de VRAM au chargement contre 2,13 Go à 2048 — pour des chunks
    /// de quelques centaines de tokens. Laisser le défaut du modèle, c'est perdre
    /// plusieurs gigaoctets sans aucune contrepartie.
    #[test]
    fn contexte_embedding_borne_mais_suffisant() {
        // Défaut de l'app (chunk_size 1000) : le plancher de 2048 suffit largement.
        assert_eq!(embedding_num_ctx(1000), 2048);
        // Un chunk minuscule ne descend pas sous le plancher.
        assert_eq!(embedding_num_ctx(10), 2048);
        // Un gros chunk fait monter le contexte, arrondi au ko supérieur.
        assert_eq!(embedding_num_ctx(8000), 5120);
        // Toujours >= au besoin estimé (2 caracteres par token, hypothèse pessimiste).
        for taille in [500, 1000, 2000, 4000, 16000] {
            let besoin = ((taille + 250) / 2) as u32;
            assert!(
                embedding_num_ctx(taille) >= besoin,
                "contexte trop court pour chunk_size={taille}"
            );
        }
    }

    #[test]
    fn modeles_locaux_resolvent_avec_la_bonne_dimension() {
        // Nouveaux modèles ajoutés au catalogue (Phase C).
        assert_eq!(resolve_local_model("gte-large").1, 1024);
        assert_eq!(resolve_local_model("gte-base").1, 768);
        assert_eq!(resolve_local_model("bge-large").1, 1024);
        assert_eq!(resolve_local_model("modernbert-embed-large").1, 1024);
        // Alias historiques inchangés.
        assert_eq!(resolve_local_model("multilingual-e5-small").1, 384);
        // Inconnu → repli sûr sur e5-small (384).
        assert_eq!(resolve_local_model("inexistant-xyz").1, 384);
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

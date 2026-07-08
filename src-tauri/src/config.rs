//! Système de configuration persistant de SenseTree.
//!
//! Toute la flexibilité « model-agnostic » repose ici : l'utilisateur peut
//! brancher un moteur d'embedding local (fastembed) OU un serveur HTTP
//! compatible OpenAI (Ollama, LM Studio, serveur maison, API externe), et
//! configurer indépendamment les modèles de reasoning et de vision (URL, IP,
//! clé, nom du modèle). La config est sérialisée en JSON dans le dossier de
//! configuration de l'application.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

/// Mode de génération des embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingMode {
    /// Modèle ONNX embarqué exécuté par fastembed (défaut, 100% local).
    Local,
    /// Endpoint HTTP compatible OpenAI (`/v1/embeddings`).
    Openai,
}

impl Default for EmbeddingMode {
    fn default() -> Self {
        EmbeddingMode::Local
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub mode: EmbeddingMode,
    /// Nom du modèle : identifiant fastembed en mode Local, id de modèle distant en mode Openai.
    pub model: String,
    /// URL de base du serveur compatible OpenAI (utilisé en mode Openai).
    pub base_url: String,
    /// Clé API optionnelle (vide pour la plupart des serveurs locaux).
    pub api_key: String,
    /// Dimension des vecteurs produits. Doit correspondre au modèle choisi.
    pub dimensions: usize,
    /// Tente d'utiliser le GPU (CUDA) en mode Local, avec repli CPU gracieux.
    pub use_gpu: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        EmbeddingConfig {
            mode: EmbeddingMode::Local,
            // multilingual-e5-small : 384 dims, multilingue (pertinent pour le français), léger.
            model: "multilingual-e5-small".to_string(),
            base_url: "http://localhost:11434/v1".to_string(),
            api_key: String::new(),
            dimensions: 384,
            use_gpu: false,
        }
    }
}

/// Configuration d'un endpoint de chat compatible OpenAI (reasoning ou vision).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatConfig {
    pub base_url: String,
    pub model: String,
    pub api_key: String,
    pub enabled: bool,
}

impl ChatConfig {
    fn default_reasoning() -> Self {
        ChatConfig {
            base_url: "http://localhost:11434/v1".to_string(),
            model: "llama3.1:8b".to_string(),
            api_key: String::new(),
            enabled: true,
        }
    }

    fn default_vision() -> Self {
        ChatConfig {
            base_url: "http://localhost:11434/v1".to_string(),
            model: "moondream".to_string(),
            api_key: String::new(),
            // La vision est opt-in : elle sollicite un modèle multimodal souvent absent par défaut.
            enabled: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingConfig {
    /// Dossiers racines surveillés et indexés.
    pub roots: Vec<String>,
    pub chunk_size: usize,
    pub overlap: usize,
    /// Taille de lot pour l'embedding (compromis débit / mémoire).
    pub batch_size: usize,
    /// Taille max d'un fichier à extraire (Mo). Au-delà : indexation par contexte seul.
    pub max_file_mb: u64,
    /// Tendance de classification des dossiers : 0.0 = très récursif (conservateur,
    /// on explore au maximum), 1.0 = très bloc (on regroupe agressivement les
    /// dossiers techniques/opaques). Défaut 0.5. `#[serde(default)]` pour rester
    /// rétro-compatible avec les settings.json antérieurs.
    #[serde(default = "default_block_bias")]
    pub block_bias: f32,
}

fn default_block_bias() -> f32 {
    0.5
}

impl Default for IndexingConfig {
    fn default() -> Self {
        IndexingConfig {
            roots: Vec::new(),
            chunk_size: 1000,
            overlap: 200,
            batch_size: 32,
            max_file_mb: 50,
            block_bias: default_block_bias(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub embedding: EmbeddingConfig,
    pub reasoning: ChatConfig,
    pub vision: ChatConfig,
    pub indexing: IndexingConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        AppConfig {
            embedding: EmbeddingConfig::default(),
            reasoning: ChatConfig::default_reasoning(),
            vision: ChatConfig::default_vision(),
            indexing: IndexingConfig::default(),
        }
    }
}

/// Conteneur thread-safe de la configuration, avec persistance sur disque.
#[derive(Debug)]
pub struct ConfigStore {
    path: PathBuf,
    inner: RwLock<AppConfig>,
}

impl ConfigStore {
    /// Charge la config depuis `path` si elle existe, sinon crée les valeurs par défaut.
    /// `default_roots` (ex : dossier Documents) n'est appliqué que lors de la toute première création.
    pub fn load_or_init(path: impl AsRef<Path>, default_roots: Vec<String>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        let config = if path.exists() {
            let raw = fs::read_to_string(&path)
                .with_context(|| format!("lecture de la config: {}", path.display()))?;
            // Tolérant : une config corrompue ne doit pas empêcher l'app de démarrer.
            match serde_json::from_str::<AppConfig>(&raw) {
                Ok(cfg) => cfg,
                Err(e) => {
                    tracing::warn!("config.json illisible ({e}), réinitialisation aux valeurs par défaut");
                    let mut cfg = AppConfig::default();
                    cfg.indexing.roots = default_roots;
                    cfg
                }
            }
        } else {
            let mut cfg = AppConfig::default();
            cfg.indexing.roots = default_roots;
            cfg
        };

        let store = ConfigStore {
            path,
            inner: RwLock::new(config),
        };
        store.persist()?;
        Ok(store)
    }

    /// Renvoie une copie de la configuration courante.
    pub fn snapshot(&self) -> AppConfig {
        self.inner.read().expect("config lock poisoned").clone()
    }

    /// Remplace la configuration et la persiste immédiatement.
    pub fn replace(&self, new_config: AppConfig) -> Result<()> {
        {
            let mut guard = self.inner.write().expect("config lock poisoned");
            *guard = new_config;
        }
        self.persist()
    }

    fn persist(&self) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("création du dossier de config: {}", parent.display()))?;
        }
        let cfg = self.inner.read().expect("config lock poisoned");
        let raw = serde_json::to_string_pretty(&*cfg)?;
        fs::write(&self.path, raw)
            .with_context(|| format!("écriture de la config: {}", self.path.display()))?;
        Ok(())
    }
}

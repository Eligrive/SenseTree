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

/// Textes de prompts système surchargés par l'utilisateur. Un champ vide signifie
/// « utiliser le prompt par défaut intégré » (voir [`default_prompts`]). C'est le
/// point d'entrée pour ajuster finement l'« extraction du sens » sans recompiler.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PromptsConfig {
    /// Classification d'un dossier : récursif vs bloc.
    #[serde(default)]
    pub folder_classify: String,
    /// Description en une phrase d'un dossier indexé en bloc.
    #[serde(default)]
    pub folder_describe: String,
    /// Extraction de sens d'un fichier de type inconnu (texte/binaire).
    #[serde(default)]
    pub file_extract: String,
    /// Légende d'une image (vision).
    #[serde(default)]
    pub vision_caption: String,
    /// OCR d'une page (vision).
    #[serde(default)]
    pub vision_ocr: String,
    /// Instruction système de l'assistant de chat (RAG + actions).
    #[serde(default)]
    pub chat_system: String,
    /// Instruction système du planificateur de réorganisation.
    #[serde(default)]
    pub reorganize: String,
}

/// Prompts par défaut intégrés (source unique de vérité). Les call-sites les
/// utilisent quand la surcharge utilisateur correspondante est vide.
pub mod default_prompts {
    pub const FOLDER_CLASSIFY: &str = "Tu décides comment un explorateur de fichiers doit traiter un dossier : \
        'recursive' (l'explorer et indexer ses fichiers un par un) ou 'block' (le traiter comme \
        une seule unité opaque, SANS l'explorer).\n\
        Choisis 'recursive' dès qu'il y a du SENS EXPLOITABLE à l'intérieur — du contenu que \
        l'utilisateur pourrait vouloir retrouver, lire, comprendre ou manipuler : documents, \
        cours, projets, notes, code source, photos ou vidéos personnelles, etc.\n\
        Choisis 'block' UNIQUEMENT si le dossier est un ensemble applicatif/technique sans \
        intérêt à indexer fichier par fichier, c'est-à-dire un truc dont l'utilisateur ne fera \
        rien individuellement : environnement virtuel, dépendances (node_modules, vendor), bundle \
        d'application, pack d'instruments/samples (DAW), cache, artefacts de build, ou dossier ne \
        contenant que des binaires opaques.\n\
        EXTRAPOLE le rôle du dossier à partir de son CHEMIN COMPLET (le dossier parent donne un \
        contexte essentiel), de son nom, et des noms de ses fichiers et sous-dossiers. En cas de \
        doute, réponds 'recursive'.\n\
        Réponds STRICTEMENT en JSON, sans aucun texte autour : {\"mode\":\"block\"|\"recursive\"}.";

    pub const FOLDER_DESCRIBE: &str = "Décris en UNE phrase concise et concrète ce qu'est ce dossier (son rôle et son \
        contenu), à partir de son nom, son emplacement et un échantillon de son contenu. \
        Réponds uniquement par la phrase, sans préambule.";

    pub const FILE_EXTRACT: &str = "On te donne un extrait d'un fichier de type inconnu. Si l'extrait contient du \
        texte ou des données PORTEUSES DE SENS (configuration, logs, notes, données, code, \
        markup…), extrais/résume son contenu utile en quelques phrases, pour l'indexer dans un \
        moteur de recherche. Si c'est du binaire opaque SANS contenu exploitable, réponds \
        EXACTEMENT et uniquement : NO_CONTENT.";

    pub const VISION_CAPTION: &str = "Décris le contenu de cette image en une à deux phrases, \
        en identifiant les objets, le texte visible et le thème, \
        pour faciliter son classement dans une arborescence de fichiers.";

    pub const VISION_OCR: &str = "Transcris fidèlement TOUT le texte visible dans cette image (OCR). \
        Ne renvoie que le texte transcrit, sans commentaire.";

    pub const CHAT_SYSTEM: &str = "Tu es l'assistant de SenseTree, un explorateur de fichiers sémantique local. \
        RÈGLE DE FORMAT : si l'utilisateur pose une QUESTION ou demande une analyse, réponds \
        NORMALEMENT en texte, en citant les fichiers pertinents par leur nom. \
        Si — et SEULEMENT si — il demande une ACTION sur des fichiers (déplacer, renommer, \
        supprimer, ranger, créer un dossier), réponds UNIQUEMENT par un objet JSON, sans aucun \
        autre texte, au format : {\"summary\":\"...\",\"operations\":[{\"kind\":\
        \"move|rename|delete|mkdir\",\"old_path\":\"...\",\"new_path\":\"...\",\"reason\":\"...\"}]}. \
        Pour une réorganisation, raisonne sur la STRUCTURE du dossier fournie plus bas (arborescence). \
        Les chemins DOIVENT être EXACTEMENT ceux listés ci-dessous — n'invente aucun chemin. \
        Rien n'est exécuté sans validation manuelle de l'utilisateur.";

    pub const REORGANIZE: &str = "Tu es l'assistant de rangement de SenseTree. À partir d'une instruction \
        et d'une liste de fichiers, tu proposes un plan de réorganisation. \
        Réponds UNIQUEMENT en JSON valide, sans texte autour, au format : \
        {\"summary\": string, \"operations\": [{\"kind\": \"move|rename|delete|mkdir\", \
        \"old_path\": string|null, \"new_path\": string|null, \"reason\": string}]}. \
        Utilise des chemins absolus cohérents avec ceux fournis. \
        N'invente jamais de fichiers inexistants.";
}

impl PromptsConfig {
    /// Renvoie les prompts par défaut intégrés (pour l'affichage dans les Paramètres).
    pub fn defaults() -> Self {
        PromptsConfig {
            folder_classify: default_prompts::FOLDER_CLASSIFY.to_string(),
            folder_describe: default_prompts::FOLDER_DESCRIBE.to_string(),
            file_extract: default_prompts::FILE_EXTRACT.to_string(),
            vision_caption: default_prompts::VISION_CAPTION.to_string(),
            vision_ocr: default_prompts::VISION_OCR.to_string(),
            chat_system: default_prompts::CHAT_SYSTEM.to_string(),
            reorganize: default_prompts::REORGANIZE.to_string(),
        }
    }
}

/// Renvoie la surcharge `override_` si non vide, sinon le défaut `fallback`.
pub fn prompt_or<'a>(override_: &'a str, fallback: &'a str) -> &'a str {
    if override_.trim().is_empty() {
        fallback
    } else {
        override_
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub embedding: EmbeddingConfig,
    pub reasoning: ChatConfig,
    pub vision: ChatConfig,
    pub indexing: IndexingConfig,
    #[serde(default)]
    pub prompts: PromptsConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        AppConfig {
            embedding: EmbeddingConfig::default(),
            reasoning: ChatConfig::default_reasoning(),
            vision: ChatConfig::default_vision(),
            indexing: IndexingConfig::default(),
            prompts: PromptsConfig::default(),
        }
    }
}

/// Vrai si `path` est un dossier racine indexé, ou situé sous l'un d'eux.
/// Comparaison insensible à la casse et aux séparateurs (adapté à Windows).
pub fn path_under_root(roots: &[String], path: &str) -> bool {
    let norm = |p: &str| p.replace('/', "\\").trim_end_matches('\\').to_lowercase();
    let np = norm(path);
    roots.iter().any(|r| {
        let nr = norm(r);
        !nr.is_empty() && (np == nr || np.starts_with(&format!("{nr}\\")))
    })
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

    /// Vrai si `path` fait partie de l'indexation (racine ou sous un dossier racine).
    pub fn is_under_root(&self, path: &str) -> bool {
        let roots = self.inner.read().expect("config lock poisoned").indexing.roots.clone();
        path_under_root(&roots, path)
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

//! État applicatif partagé entre les threads de fond et les commandes IPC.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use crate::config::ConfigStore;
use crate::db::Database;
use crate::providers::AiEngine;
use crate::vectordb::VectorDb;

/// Regroupe les composants long-vivants (DB relationnelle, config, moteur IA,
/// base vectorielle). Partagé via `Arc` au crawler, au watchdog, au worker et
/// aux commandes Tauri.
pub struct AppState {
    pub db: Arc<Database>,
    pub config: Arc<ConfigStore>,
    pub ai: Arc<AiEngine>,
    pub vector: Arc<VectorDb>,
    /// Dossier de données de l'app (pour l'approvisionnement d'ONNX Runtime, etc.).
    pub data_dir: PathBuf,
    /// Pause globale de l'indexation (worker + classifieur) demandée par l'utilisateur.
    pub paused: Arc<AtomicBool>,
    /// Racines en cours de scan (clé normalisée) : évite plusieurs crawlers concurrents
    /// sur le même dossier (ex. sauvegardes répétées des Paramètres).
    pub scanning: Arc<Mutex<HashSet<String>>>,
}

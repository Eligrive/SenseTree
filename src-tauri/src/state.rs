//! État applicatif partagé entre les threads de fond et les commandes IPC.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{Arc, Mutex};

use serde::Serialize;

use crate::config::ConfigStore;
use crate::db::Database;
use crate::gardener::GardenerReport;
use crate::providers::AiEngine;
use crate::vectordb::VectorDb;

/// Élément actuellement traité par le worker d'indexation (pour l'affichage temps réel
/// de la file). `routes` = étapes du pipeline dans l'ordre (sous-ensemble de
/// {`vision`, `reasoning`, `embedding`}).
#[derive(Debug, Clone, Serialize)]
pub struct CurrentActivity {
    pub path: String,
    pub routes: Vec<String>,
    pub kind: String,
}

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
    /// Racines en cours de scan (clé normalisée → « re-scan demandé »). Évite les
    /// crawlers concurrents sur le même dossier ; si un scan est demandé pendant qu'un
    /// autre tourne, on le PROGRAMME (flag `true`) au lieu de le perdre.
    pub scanning: Arc<Mutex<HashMap<String, bool>>>,
    /// Élément que le worker traite à l'instant (pour l'affichage temps réel de la file).
    pub activity: Arc<Mutex<Option<CurrentActivity>>>,
    /// « Époque » de scan : incrémentée à chaque réindexation demandée. Les scans en
    /// cours comparent l'époque qu'ils ont capturée au démarrage et s'ARRÊTENT dès
    /// qu'elle change — sinon un crawler bloqué (ex. en pause) gardait sa racine
    /// verrouillée et toute réindexation restait sans effet jusqu'au redémarrage.
    pub scan_epoch: Arc<AtomicUsize>,
    /// Dernier bilan de santé structurel des racines (gardener proactif de fond,
    /// lecture seule). Rafraîchi périodiquement ; servi tel quel à l'UI.
    pub gardener: Arc<Mutex<GardenerReport>>,
}

impl AppState {
    /// Publie (ou efface avec `None`) l'élément en cours de traitement par le worker.
    pub fn set_activity(&self, act: Option<CurrentActivity>) {
        if let Ok(mut g) = self.activity.lock() {
            *g = act;
        }
    }

    /// Instantané de l'élément en cours de traitement (pour la commande IPC).
    pub fn activity_snapshot(&self) -> Option<CurrentActivity> {
        self.activity.lock().ok().and_then(|g| g.clone())
    }
}

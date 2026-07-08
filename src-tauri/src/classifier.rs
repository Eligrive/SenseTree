//! Classifieur de rattrapage des dossiers « en attente ».
//!
//! Quand l'IA est indisponible au moment où il faut décider récursif vs bloc, la
//! décision est REPORTÉE (le dossier n'est pas indexé). Ce thread léger repasse
//! périodiquement sur ces dossiers et les reclasse dès que l'IA redevient
//! joignable : bloc → indexé comme unité ; récursif → on lance son scan.

use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crate::crawler;
use crate::folders::{self, Decision};
use crate::state::AppState;

pub fn start_classifier(state: Arc<AppState>) {
    thread::spawn(move || {
        tracing::info!("🧭 Classifieur de dossiers en attente démarré.");
        loop {
            thread::sleep(Duration::from_secs(20));

            // Pause utilisateur : on ne classe rien tant que l'indexation est en pause.
            if state.paused.load(std::sync::atomic::Ordering::Relaxed) {
                continue;
            }

            // Inutile d'essayer si l'utilisateur a désactivé le reasoning.
            if !state.config.snapshot().reasoning.enabled {
                continue;
            }

            let pending = match state.db.get_pending_folders(8) {
                Ok(p) => p,
                Err(_) => continue,
            };
            if pending.is_empty() {
                continue;
            }

            for folder in pending {
                let path = Path::new(&folder);
                if !path.exists() {
                    continue;
                }
                match folders::resolve_mode(&state, path) {
                    Decision::Block => {
                        let _ = state.db.enqueue_path(&folder, Some("pending_extraction"), 6);
                        tracing::info!("🧩 dossier reclassé en bloc : {folder}");
                    }
                    Decision::Recursive => {
                        tracing::info!("📂 dossier reclassé en récursif, indexation : {folder}");
                        let sc = state.clone();
                        let f = folder.clone();
                        thread::spawn(move || crawler::scan_directory(sc, &f));
                    }
                    Decision::Defer => {
                        // IA toujours indisponible : on stoppe ce tour pour ne pas
                        // marteler, on réessaiera au prochain cycle.
                        break;
                    }
                }
            }
        }
    });
}

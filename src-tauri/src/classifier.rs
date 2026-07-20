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
            // Pause utilisateur, ou reasoning désactivé : rien à faire, on attend.
            if state.paused.load(std::sync::atomic::Ordering::Relaxed)
                || !state.config.snapshot().reasoning.enabled
            {
                thread::sleep(Duration::from_secs(20));
                continue;
            }

            let pending = match state.db.get_pending_folders(8) {
                Ok(p) => p,
                Err(_) => {
                    thread::sleep(Duration::from_secs(20));
                    continue;
                }
            };
            if pending.is_empty() {
                thread::sleep(Duration::from_secs(20));
                continue;
            }

            // Le crawler REPORTE désormais tous les dossiers ambigus : c'est ici que se
            // fait tout le travail de classification. On enchaîne donc les lots SANS
            // pause tant qu'il reste du travail et que l'IA répond.
            let mut ai_unavailable = false;
            for folder in pending {
                if state.paused.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
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
                        // IA indisponible : on stoppe ce tour pour ne pas marteler.
                        ai_unavailable = true;
                        break;
                    }
                }
            }

            // IA injoignable → longue temporisation (ne pas marteler). Sinon, courte
            // respiration entre deux lots : assez rapide pour résorber la file de
            // classification, mais sans spawner des dizaines de scans d'un coup.
            // (La protection contre les RAFALES de fichiers, elle, est le debouncer
            //  de 2 s du watchdog — pas ce thread.)
            thread::sleep(Duration::from_secs(if ai_unavailable { 20 } else { 2 }));
        }
    });
}

//! Watchdog temps réel (le « présent ») : réagit aux créations/modifications/
//! suppressions et alimente la file, en respectant le mode des dossiers (un
//! fichier dans un dossier-bloc n'est pas indexé individuellement).

use notify_debouncer_mini::{new_debouncer, notify::RecursiveMode};
use std::collections::HashSet;
use std::path::Path;
use std::sync::mpsc::channel;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crate::folders::{self, Decision};
use crate::parser::{FileType, Parser};
use crate::state::AppState;

pub fn start_watching(state: Arc<AppState>, roots: Vec<String>) {
    thread::spawn(move || {
        let (tx, rx) = channel();
        let mut debouncer = match new_debouncer(Duration::from_secs(2), tx) {
            Ok(d) => d,
            Err(e) => {
                tracing::error!("initialisation du watchdog impossible: {e}");
                return;
            }
        };

        for root in &roots {
            if let Err(e) = debouncer
                .watcher()
                .watch(Path::new(root), RecursiveMode::Recursive)
            {
                tracing::warn!("watchdog: impossible de surveiller {root}: {e}");
            } else {
                tracing::info!("🛡️ Watchdog actif sur : {root}");
            }
        }

        for res in rx {
            match res {
                Ok(events) => handle_batch(&state, events),
                Err(e) => tracing::warn!("erreur système watchdog: {e:?}"),
            }
        }
    });
}

fn handle_batch(state: &AppState, events: Vec<notify_debouncer_mini::DebouncedEvent>) {
    let unique: HashSet<_> = events.into_iter().map(|e| e.path).collect();
    let db = &state.db;

    for path_buf in unique {
        let path_str = path_buf.to_string_lossy().to_string();

        // Suppression : on remet en file pour purge des vecteurs.
        if !path_buf.exists() {
            tracing::debug!("🗑️ suppression détectée : {path_str}");
            let _ = db.enqueue_path(&path_str, Some("pending_extraction"), 8);
            continue;
        }

        // Nouveau dossier : on le classe (et on l'enfile s'il devient un bloc).
        if path_buf.is_dir() {
            if folders::hard_ignore(&path_buf.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()) {
                continue;
            }
            if folders::resolve_mode(state, &path_buf) == Decision::Block {
                let _ = db.enqueue_path(&path_str, Some("pending_extraction"), 6);
            }
            // (Recursive → ses fichiers arriveront via leurs propres événements ;
            //  Defer → dossier marqué en attente, repris par le classifieur.)
            continue;
        }

        if Parser::determine_file_type(&path_buf) == FileType::Ignored {
            continue;
        }

        // Le fichier appartient-il à un dossier traité comme bloc ?
        if let Some(parent) = path_buf.parent() {
            let parent_name = parent.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            if folders::hard_ignore(&parent_name) {
                continue;
            }
            match folders::resolve_mode(state, parent) {
                Decision::Block => {
                    // Fichier interne à un bloc : on rafraîchit le bloc, pas le fichier.
                    let _ = db.enqueue_path(&parent.to_string_lossy(), Some("pending_extraction"), 6);
                    continue;
                }
                Decision::Defer => {
                    // Dossier parent en attente de classification IA : on ne fait
                    // rien pour l'instant (le classifieur tranchera plus tard).
                    continue;
                }
                Decision::Recursive => { /* dossier récursif : on indexe le fichier */ }
            }
        }

        // Dossier récursif : indexation normale du fichier.
        let (size, mtime) = match std::fs::metadata(&path_buf) {
            Ok(m) => (
                Some(m.len() as i64),
                m.modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs() as i64),
            ),
            Err(_) => (None, None),
        };
        let _ = db.upsert_file_record(
            &path_str,
            path_buf.parent().map(|p| p.to_string_lossy().to_string()).as_deref(),
            &path_buf.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
            false,
            None,
            size,
            mtime.map(|m| m.to_string()).as_deref(),
        );
        let _ = db.enqueue_path(&path_str, Some("pending_extraction"), 10);
        tracing::debug!("✅ mis en file : {path_str}");
    }
}

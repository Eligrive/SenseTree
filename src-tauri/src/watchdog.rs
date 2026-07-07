//! Watchdog temps réel (le « présent ») : réagit aux créations/modifications/
//! suppressions du système de fichiers et alimente la file d'indexation unifiée.

use notify_debouncer_mini::{new_debouncer, notify::RecursiveMode};
use std::collections::HashSet;
use std::path::Path;
use std::sync::mpsc::channel;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crate::db::Database;
use crate::parser::{FileType, Parser};

pub fn start_watching(db: Arc<Database>, roots: Vec<String>) {
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
                Ok(events) => handle_batch(&db, events),
                Err(e) => tracing::warn!("erreur système watchdog: {e:?}"),
            }
        }
    });
}

fn handle_batch(db: &Database, events: Vec<notify_debouncer_mini::DebouncedEvent>) {
    let unique: HashSet<_> = events.into_iter().map(|e| e.path).collect();

    for path_buf in unique {
        let path_str = path_buf.to_string_lossy().to_string();

        // Suppression : on remet en file pour que le worker purge les vecteurs.
        if !path_buf.exists() {
            tracing::debug!("🗑️ suppression détectée : {path_str}");
            let _ = db.enqueue_path(&path_str, Some("pending_extraction"), 8);
            continue;
        }

        if path_buf.is_dir() {
            continue;
        }

        if Parser::determine_file_type(&path_buf) == FileType::Ignored {
            continue;
        }

        // Mise à jour du catalogue + mise en file.
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

//! Crawler d'indexation initiale (le « passé »).
//!
//! Parcourt récursivement une racine. À chaque dossier rencontré, il décide via
//! `folders::resolve_mode` s'il faut l'explorer (récursif) ou le traiter comme un
//! bloc sémantique unique (dans ce cas on ne descend pas dedans). Alimente le
//! catalogue, met en file les fichiers/blocs, et purge les orphelins.

use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use walkdir::WalkDir;

use crate::db::FileMeta;
use crate::folders::{self, FolderMode};
use crate::parser::{FileType, Parser};
use crate::state::AppState;

const CATALOG_FLUSH: usize = 256;

pub fn scan_directory(state: Arc<AppState>, start_path: &str) {
    let start_time = Instant::now();
    let scan_time = now_epoch();
    let db = &state.db;

    tracing::info!("🕷️ Crawler démarré sur : {start_path}");

    let mut catalog_batch: Vec<FileMeta> = Vec::with_capacity(CATALOG_FLUSH);
    let (mut scanned, mut queued, mut skipped, mut blocks) = (0u64, 0u64, 0u64, 0u64);

    let mut it = WalkDir::new(start_path).into_iter();
    loop {
        let entry = match it.next() {
            None => break,
            Some(Err(_)) => continue,
            Some(Ok(e)) => e,
        };

        let path = entry.path();
        let depth = entry.depth();
        let is_dir = entry.file_type().is_dir();
        let path_str = path.to_string_lossy().to_string();

        let (size, mtime) = match std::fs::metadata(path) {
            Ok(m) => (
                Some(m.len() as i64),
                m.modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_secs() as i64)
                    .unwrap_or(0),
            ),
            Err(_) => (None, 0),
        };

        catalog_batch.push(FileMeta {
            path: path_str.clone(),
            parent_path: path.parent().map(|p| p.to_string_lossy().to_string()),
            file_name: entry.file_name().to_string_lossy().to_string(),
            is_directory: is_dir,
            size_bytes: size,
            modified_at: Some(mtime),
        });
        if catalog_batch.len() >= CATALOG_FLUSH {
            let _ = db.bulk_upsert_file_records(&catalog_batch);
            catalog_batch.clear();
        }

        if is_dir {
            // La racine est toujours explorée.
            if depth == 0 {
                continue;
            }
            // Dossiers systèmes/techniques : jamais indexés, on ne descend pas.
            if folders::hard_ignore(&entry.file_name().to_string_lossy()) {
                it.skip_current_dir();
                continue;
            }
            // Décision récursif vs bloc (conservative : bloc seulement si sûr / IA).
            match folders::resolve_mode(&state, path) {
                FolderMode::Block => {
                    let _ = db.enqueue_path(&path_str, Some("pending_extraction"), 6);
                    blocks += 1;
                    it.skip_current_dir(); // on n'explore pas un bloc
                }
                FolderMode::Recursive => { /* on descend normalement */ }
            }
            continue;
        }

        // --- Fichier (dans un dossier récursif) ---
        scanned += 1;
        let _ = db.touch_seen(&path_str, scan_time);

        if !db.needs_indexing(&path_str, mtime).unwrap_or(true) {
            skipped += 1;
            continue;
        }

        if Parser::determine_file_type(path) != FileType::Ignored {
            let _ = db.enqueue_path(&path_str, Some("pending_extraction"), 5);
            queued += 1;
        }
    }

    if !catalog_batch.is_empty() {
        let _ = db.bulk_upsert_file_records(&catalog_batch);
    }

    // Orphelins : remis en file pour que le worker purge leurs vecteurs.
    let orphans = db.take_orphans(scan_time, start_path).unwrap_or_default();
    for orphan in &orphans {
        let _ = db.enqueue_path(orphan, Some("pending_extraction"), 8);
    }

    tracing::info!(
        "✅ Scan de {start_path} terminé en {:.2?}. Fichiers: {scanned}, à jour: {skipped}, en file: {queued}, blocs: {blocks}, orphelins: {}.",
        start_time.elapsed(),
        orphans.len()
    );
}

fn now_epoch() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

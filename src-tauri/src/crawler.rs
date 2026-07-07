//! Crawler d'indexation initiale (le « passé »).
//!
//! Parcourt récursivement une racine, alimente le catalogue de fichiers, met en
//! file les fichiers nouveaux/modifiés et purge les orphelins (via la file, pour
//! que le worker asynchrone supprime aussi leurs vecteurs).

use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use walkdir::{DirEntry, WalkDir};

use crate::db::{Database, FileMeta};
use crate::parser::{FileType, Parser};

const CATALOG_FLUSH: usize = 256;

pub fn scan_directory(db: Arc<Database>, start_path: &str) {
    let start_time = Instant::now();
    let scan_time = now_epoch();

    tracing::info!("🕷️ Crawler démarré sur : {start_path}");

    let walker = WalkDir::new(start_path)
        .into_iter()
        .filter_entry(|e| !is_system_or_ignored_dir(e));

    let mut catalog_batch: Vec<FileMeta> = Vec::with_capacity(CATALOG_FLUSH);
    let (mut scanned, mut queued, mut skipped) = (0u64, 0u64, 0u64);

    for entry in walker.filter_map(|e| e.ok()) {
        let path = entry.path();
        let is_dir = path.is_dir();
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

        // Alimente le catalogue (dossiers ET fichiers) pour l'explorateur et le gardener.
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
            continue;
        }

        scanned += 1;
        // On marque « vu » (pour la détection d'orphelins) sans prétendre l'avoir indexé.
        let _ = db.touch_seen(&path_str, scan_time);

        // Fichier déjà indexé et inchangé : on n'ouvre même pas le parser.
        if !db.needs_indexing(&path_str, mtime).unwrap_or(true) {
            skipped += 1;
            continue;
        }

        // File unifiée : le worker re-déterminera le type et choisira la voie d'extraction.
        if Parser::determine_file_type(path) != FileType::Ignored {
            let _ = db.enqueue_path(&path_str, Some("pending_extraction"), 5);
            queued += 1;
        }
    }

    if !catalog_batch.is_empty() {
        let _ = db.bulk_upsert_file_records(&catalog_batch);
    }

    // Orphelins : on les remet en file pour que le worker purge leurs vecteurs.
    let orphans = db.take_orphans(scan_time, start_path).unwrap_or_default();
    for orphan in &orphans {
        let _ = db.enqueue_path(orphan, Some("pending_extraction"), 8);
    }

    tracing::info!(
        "✅ Scan de {start_path} terminé en {:.2?}. Vus: {scanned}, à jour: {skipped}, en file: {queued}, orphelins: {}.",
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

/// Bouclier CPU : exclut les dossiers systèmes/techniques du parcours.
fn is_system_or_ignored_dir(entry: &DirEntry) -> bool {
    let name = entry.file_name().to_string_lossy();
    if entry.file_type().is_dir() {
        return name.starts_with('.')
            || name == "node_modules"
            || name == "target"
            || name == "AppData"
            || name == "Windows"
            || name == "$RECYCLE.BIN";
    }
    name.starts_with('.')
}

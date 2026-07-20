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
use crate::folders::{self, Decision};
use crate::parser::{FileType, Parser};
use crate::state::AppState;

const CATALOG_FLUSH: usize = 256;

/// Garde RAII : retire la racine du registre des scans, même en cas de panique.
struct ScanGuard {
    state: Arc<AppState>,
    key: String,
}
impl Drop for ScanGuard {
    fn drop(&mut self) {
        if let Ok(mut m) = self.state.scanning.lock() {
            m.remove(&self.key);
        }
    }
}

pub fn scan_directory(state: Arc<AppState>, start_path: &str) {
    // Un seul crawler à la fois par racine. Si un scan est demandé alors qu'un autre
    // tourne, on ne le PERD pas : on lève le flag « re-scan demandé », et le scan en
    // cours en refera un tour à la fin (indispensable après une réindexation).
    let key = start_path.trim_end_matches(['/', '\\']).to_lowercase();
    {
        let mut m = match state.scanning.lock() {
            Ok(m) => m,
            Err(_) => return,
        };
        if m.contains_key(&key) {
            m.insert(key.clone(), true);
            tracing::info!("🕷️ scan déjà en cours sur {start_path} — re-scan programmé.");
            return;
        }
        m.insert(key.clone(), false);
    }
    let _guard = ScanGuard { state: state.clone(), key: key.clone() };

    loop {
        run_scan(&state, start_path);
        // Un re-scan a-t-il été demandé pendant ce passage ? Si oui, on recommence.
        let rerun = match state.scanning.lock() {
            Ok(mut m) => matches!(m.insert(key.clone(), false), Some(true)),
            Err(_) => false,
        };
        if !rerun {
            break;
        }
        tracing::info!("🕷️ re-scan de {start_path} (demandé pendant le précédent).");
    }
}

fn run_scan(state: &AppState, start_path: &str) {
    let start_time = Instant::now();
    let scan_time = now_epoch();
    let db = &state.db;

    tracing::info!("🕷️ Crawler démarré sur : {start_path}");

    let mut catalog_batch: Vec<FileMeta> = Vec::with_capacity(CATALOG_FLUSH);
    let (mut scanned, mut queued, mut skipped, mut blocks, mut deferred) =
        (0u64, 0u64, 0u64, 0u64, 0u64);

    let mut it = WalkDir::new(start_path).into_iter();
    loop {
        // Pause utilisateur : on suspend le scan (et ses classifications LLM) sans
        // abandonner la progression — on reprend là où on s'était arrêté.
        while state.paused.load(std::sync::atomic::Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(500));
        }

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
            // NON BLOQUANT : un dossier qui exige le LLM est reporté (le classifieur
            // de fond tranchera), pour que la marche ne soit jamais figée par une
            // requête modèle de plusieurs dizaines de secondes.
            match folders::resolve_mode_fast(state, path) {
                Decision::Block => {
                    let _ = db.enqueue_path(&path_str, Some("pending_extraction"), 6);
                    blocks += 1;
                    it.skip_current_dir(); // on n'explore pas un bloc
                }
                Decision::Defer => {
                    // Décision reportée (IA indisponible) : on n'explore pas encore.
                    // Le classifieur reprendra ce dossier dès que l'IA sera dispo.
                    deferred += 1;
                    it.skip_current_dir();
                }
                Decision::Recursive => { /* on descend normalement */ }
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
        "✅ Scan de {start_path} terminé en {:.2?}. Fichiers: {scanned}, à jour: {skipped}, en file: {queued}, blocs: {blocks}, reportés: {deferred}, orphelins: {}.",
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

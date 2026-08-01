//! Gardener proactif : diagnostic structurel de fond, en lecture seule.
//!
//! Un thread léger repasse périodiquement sur chaque racine indexée et calcule un
//! bilan de santé (doublons exacts, dossiers « en vrac », arborescences profondes,
//! dossiers vides). Le résultat est mis en cache dans l'état partagé et exposé à
//! l'UI (pastilles de santé par dossier). AUCUNE modification n'est jamais faite ici :
//! le rangement effectif passe toujours par un plan d'action validé (voir `actions.rs`).

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::state::AppState;

/// Seuils de diagnostic (alignés sur `actions::analyze_directory`).
const CLUTTER_THRESHOLD: usize = 40; // fichiers « en vrac » directement sous la racine
const DEEP_THRESHOLD: usize = 6; // profondeur d'arborescence jugée excessive
const WALK_CAP: usize = 200_000; // garde-fou anti-arbre pathologique

const INITIAL_DELAY: Duration = Duration::from_secs(45); // laisse l'indexation initiale démarrer
const INTERVAL: Duration = Duration::from_secs(300); // re-diagnostic complet toutes les 5 min
const TICK: Duration = Duration::from_secs(15); // granularité (réactif aux changements de racines)

/// Gravité d'un dossier, du plus sain au plus « à ranger ».
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthSeverity {
    Ok,
    Info,
    Warn,
}

/// Bilan de santé structurel d'une racine indexée (lecture seule).
#[derive(Debug, Clone, Serialize)]
pub struct FolderHealth {
    pub path: String,
    pub name: String,
    pub file_count: usize,
    pub direct_files: usize,
    pub duplicate_files: usize,
    pub duplicate_groups: usize,
    pub empty_dirs: usize,
    pub max_depth: usize,
    pub cluttered: bool,
    pub severity: HealthSeverity,
    /// Résumé court prêt à afficher (ex. « 3 doublon(s) · en vrac (52) »).
    pub headline: String,
}

/// Rapport agrégé mis en cache dans l'état, servi tel quel à l'UI.
#[derive(Debug, Clone, Default, Serialize)]
pub struct GardenerReport {
    pub folders: Vec<FolderHealth>,
    pub anomaly_count: usize,
    /// Horodatage unix (secondes) du dernier diagnostic ; 0 tant qu'aucun n'a tourné.
    pub scanned_at: u64,
}

/// Dernier segment d'un chemin (nom du dossier), séparateurs Windows ou Unix.
fn folder_name(path: &str) -> String {
    let is_sep = |c: char| c == '\\' || c == '/';
    let trimmed = path.trim_end_matches(is_sep);
    trimmed
        .rsplit(is_sep)
        .find(|s| !s.is_empty())
        .filter(|s| !s.is_empty())
        .unwrap_or(path)
        .to_string()
}

/// Classe un dossier et produit son résumé, à partir des seules mesures brutes.
/// Pur (pas d'I/O) : sépare la logique de diagnostic du parcours disque, et permet
/// de la tester directement. `Warn` = à ranger (doublons/vrac) ; `Info` = à surveiller
/// (dossiers vides / arborescence profonde) ; `Ok` = rien à signaler.
fn diagnose(
    duplicate_files: usize,
    direct_files: usize,
    empty_dirs: usize,
    max_depth: usize,
) -> (HealthSeverity, String) {
    let cluttered = direct_files > CLUTTER_THRESHOLD;
    let deep = max_depth > DEEP_THRESHOLD;

    let severity = if duplicate_files > 0 || cluttered {
        HealthSeverity::Warn
    } else if empty_dirs > 0 || deep {
        HealthSeverity::Info
    } else {
        HealthSeverity::Ok
    };

    let mut parts: Vec<String> = Vec::new();
    if duplicate_files > 0 {
        parts.push(format!("{duplicate_files} doublon(s)"));
    }
    if cluttered {
        parts.push(format!("en vrac ({direct_files})"));
    }
    if deep {
        parts.push(format!("profond (niv. {max_depth})"));
    }
    if empty_dirs > 0 {
        parts.push(format!("{empty_dirs} dossier(s) vide(s)"));
    }
    let headline = if parts.is_empty() {
        "Bien rangé".to_string()
    } else {
        parts.join(" · ")
    };

    (severity, headline)
}

/// Calcule le bilan d'un dossier (walk récursif + doublons par hash de contenu).
fn compute_health(state: &AppState, path: &str) -> FolderHealth {
    let mut file_count = 0usize;
    let mut direct_files = 0usize;
    let mut max_depth = 0usize;
    let mut empty_dirs = 0usize;
    let mut visited = 0usize;

    for entry in walkdir::WalkDir::new(path).into_iter().flatten() {
        visited += 1;
        if visited > WALK_CAP {
            break; // arbre démesuré : on s'arrête plutôt que de bloquer le thread
        }
        let depth = entry.depth();
        let ft = entry.file_type();
        if ft.is_file() {
            file_count += 1;
            if depth == 1 {
                direct_files += 1;
            }
            if depth > max_depth {
                max_depth = depth;
            }
        } else if ft.is_dir() && depth > 0 {
            let is_empty = std::fs::read_dir(entry.path())
                .map(|mut d| d.next().is_none())
                .unwrap_or(false);
            if is_empty {
                empty_dirs += 1;
            }
        }
    }

    let dup_groups = state.db.find_duplicates(path).unwrap_or_default();
    let duplicate_groups = dup_groups.len();
    let duplicate_files: usize = dup_groups.iter().map(|g| g.paths.len()).sum();
    let cluttered = direct_files > CLUTTER_THRESHOLD;

    let (severity, headline) = diagnose(duplicate_files, direct_files, empty_dirs, max_depth);

    FolderHealth {
        name: folder_name(path),
        path: path.to_string(),
        file_count,
        direct_files,
        duplicate_files,
        duplicate_groups,
        empty_dirs,
        max_depth,
        cluttered,
        severity,
        headline,
    }
}

/// Calcule le rapport complet (toutes les racines existantes).
fn build_report(state: &AppState, roots: &[String]) -> GardenerReport {
    let mut folders = Vec::with_capacity(roots.len());
    for root in roots {
        if std::path::Path::new(root).is_dir() {
            folders.push(compute_health(state, root));
        }
    }
    let anomaly_count = folders
        .iter()
        .filter(|f| f.severity != HealthSeverity::Ok)
        .count();
    let scanned_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    GardenerReport { folders, anomaly_count, scanned_at }
}

/// Démarre le thread de diagnostic de fond. Recalcule à intervalle régulier, et
/// plus tôt si la liste des racines change (ajout/retrait d'un dossier indexé).
pub fn start_gardener(state: Arc<AppState>) {
    thread::spawn(move || {
        tracing::info!("🌱 Gardener proactif démarré.");
        thread::sleep(INITIAL_DELAY);
        let mut last_run: Option<Instant> = None;
        let mut last_roots: Vec<String> = Vec::new();
        loop {
            let roots = state.config.snapshot().indexing.roots;
            let due = match last_run {
                None => true,
                Some(t) => t.elapsed() >= INTERVAL || roots != last_roots,
            };
            if due {
                let report = build_report(&state, &roots);
                if let Ok(mut g) = state.gardener.lock() {
                    *g = report;
                }
                last_run = Some(Instant::now());
                last_roots = roots;
            }
            thread::sleep(TICK);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folder_name_handles_separators_and_trailing_slashes() {
        assert_eq!(folder_name(r"C:\Users\virgi\Documents"), "Documents");
        assert_eq!(folder_name(r"C:\Users\virgi\Documents\"), "Documents");
        assert_eq!(folder_name("/home/user/notes/"), "notes");
        assert_eq!(folder_name("/home/user//notes//"), "notes");
        // Racine sans segment nommé : on retombe sur le chemin d'origine.
        assert_eq!(folder_name("/"), "/");
    }

    #[test]
    fn clean_folder_is_ok() {
        let (sev, head) = diagnose(0, 10, 0, 3);
        assert_eq!(sev, HealthSeverity::Ok);
        assert_eq!(head, "Bien rangé");
    }

    #[test]
    fn duplicates_or_clutter_warn() {
        // Doublons seuls → Warn.
        assert_eq!(diagnose(4, 5, 0, 2).0, HealthSeverity::Warn);
        // Vrac seul (> seuil) → Warn ; sous le seuil → pas Warn pour cette raison.
        assert_eq!(diagnose(0, CLUTTER_THRESHOLD + 1, 0, 2).0, HealthSeverity::Warn);
        assert_eq!(diagnose(0, CLUTTER_THRESHOLD, 0, 2).0, HealthSeverity::Ok);
    }

    #[test]
    fn empty_or_deep_is_info_not_warn() {
        assert_eq!(diagnose(0, 3, 2, 2).0, HealthSeverity::Info);
        assert_eq!(diagnose(0, 3, 0, DEEP_THRESHOLD + 1).0, HealthSeverity::Info);
        // Pile au seuil de profondeur : pas encore signalé.
        assert_eq!(diagnose(0, 3, 0, DEEP_THRESHOLD).0, HealthSeverity::Ok);
    }

    #[test]
    fn headline_lists_every_active_symptom() {
        let (sev, head) = diagnose(3, CLUTTER_THRESHOLD + 12, 2, DEEP_THRESHOLD + 1);
        assert_eq!(sev, HealthSeverity::Warn); // la pire gravité l'emporte
        assert!(head.contains("3 doublon(s)"), "{head}");
        assert!(head.contains("en vrac"), "{head}");
        assert!(head.contains("profond"), "{head}");
        assert!(head.contains("vide(s)"), "{head}");
    }
}

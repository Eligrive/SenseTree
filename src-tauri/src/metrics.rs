//! Mesure du débit des trois étages IA (vision, reasoning, embedding).
//!
//! Deux questions différentes, souvent confondues :
//!   * « à quelle vitesse mes MODÈLES répondent-ils ? » → c'est ce que mesure ce
//!     module : le temps passé DANS chaque étage, indépendamment du reste.
//!   * « à quelle vitesse mon indexation avance-t-elle ? » → ça se lit sur le
//!     compteur `completed` de la file, en différence entre deux relevés. Ce n'est
//!     pas la même chose : le pipeline passe aussi du temps à lire des fichiers, à
//!     extraire du PDF et à écrire dans LanceDB.
//!
//! L'unité pertinente n'est pas la même partout, d'où trois compteurs distincts :
//!   * vision et reasoning font UN appel par fichier → `fichiers/s` et `ms/appel`
//!     sont directement parlants ;
//!   * l'embedding traite N chunks proportionnels au volume → `fichiers/s` y serait
//!     trompeur (un PDF de 200 pages et un `.txt` de trois lignes ne se comparent
//!     pas), donc `chunks/s` et `Mo/s`.
//!
//! Les compteurs sont globaux au processus (et non portés par `AppState`) pour
//! pouvoir être alimentés depuis n'importe quelle implémentation de provider sans
//! faire transiter une référence à travers toutes les signatures.

use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::OnceLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Vision,
    /// Traitement d'un média audio/vidéo : transcription de la parole et/ou
    /// description visuelle. Les deux appels tapent le même étage, parce que
    /// c'est le même goulot d'étranglement du point de vue de l'indexation.
    Media,
    Reasoning,
    Embedding,
}

impl Stage {
    fn label(self) -> &'static str {
        match self {
            Stage::Vision => "vision",
            Stage::Media => "media",
            Stage::Reasoning => "reasoning",
            Stage::Embedding => "embedding",
        }
    }
}

#[derive(Default)]
struct Counters {
    /// Appels au modèle.
    ops: AtomicU64,
    /// Temps cumulé passé dans l'étage.
    nanos: AtomicU64,
    /// Unités traitées : images ou fichiers pour vision/reasoning, chunks pour l'embedding.
    units: AtomicU64,
    /// Volume soumis : octets de l'image (vision) ou du texte (reasoning, embedding).
    bytes: AtomicU64,
    /// Appels échoués. Comptés à part : ils faussent les moyennes de durée.
    errors: AtomicU64,
}

impl Counters {
    fn record(&self, elapsed: Duration, units: u64, bytes: u64) {
        self.ops.fetch_add(1, Relaxed);
        self.nanos.fetch_add(elapsed.as_nanos() as u64, Relaxed);
        self.units.fetch_add(units, Relaxed);
        self.bytes.fetch_add(bytes, Relaxed);
    }

    fn snapshot(&self, stage: Stage) -> StageStats {
        let ops = self.ops.load(Relaxed);
        let units = self.units.load(Relaxed);
        let bytes = self.bytes.load(Relaxed);
        let seconds = self.nanos.load(Relaxed) as f64 / 1e9;
        // Une division par zéro donnerait `inf`, qui se sérialise en `null` en JSON et
        // s'afficherait comme une absence de mesure : on renvoie `None` explicitement.
        let per = |x: f64| if seconds > 0.0 { Some(x / seconds) } else { None };
        StageStats {
            stage: stage.label().to_string(),
            ops,
            units,
            bytes,
            errors: self.errors.load(Relaxed),
            seconds,
            ms_per_op: if ops > 0 { Some(seconds * 1000.0 / ops as f64) } else { None },
            units_per_sec: per(units as f64),
            mb_per_sec: per(bytes as f64 / 1e6),
        }
    }
}

#[derive(Default)]
pub struct Metrics {
    vision: Counters,
    media: Counters,
    reasoning: Counters,
    embedding: Counters,
    /// Début de la fenêtre de mesure (unix), remis à zéro par [`reset`].
    since: AtomicU64,
}

impl Metrics {
    fn stage(&self, s: Stage) -> &Counters {
        match s {
            Stage::Vision => &self.vision,
            Stage::Media => &self.media,
            Stage::Reasoning => &self.reasoning,
            Stage::Embedding => &self.embedding,
        }
    }
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

pub fn metrics() -> &'static Metrics {
    static M: OnceLock<Metrics> = OnceLock::new();
    M.get_or_init(|| {
        let m = Metrics::default();
        m.since.store(now(), Relaxed);
        m
    })
}

/// Enregistre un appel réussi.
///
/// `units` : images/fichiers (vision, reasoning) ou chunks (embedding).
/// `bytes` : volume soumis au modèle.
pub fn record(stage: Stage, elapsed: Duration, units: u64, bytes: u64) {
    metrics().stage(stage).record(elapsed, units, bytes);
}

/// Enregistre un appel échoué. Non compté dans les moyennes de durée : un timeout
/// de 45 s ferait passer un étage sain pour trois fois plus lent qu'il n'est.
pub fn record_error(stage: Stage) {
    metrics().stage(stage).errors.fetch_add(1, Relaxed);
}

/// Remet les compteurs à zéro (pour mesurer une indexation précise).
pub fn reset() {
    let m = metrics();
    for s in [Stage::Vision, Stage::Media, Stage::Reasoning, Stage::Embedding] {
        let c = m.stage(s);
        c.ops.store(0, Relaxed);
        c.nanos.store(0, Relaxed);
        c.units.store(0, Relaxed);
        c.bytes.store(0, Relaxed);
        c.errors.store(0, Relaxed);
    }
    m.since.store(now(), Relaxed);
}

#[derive(Serialize, Clone, Debug)]
pub struct StageStats {
    pub stage: String,
    pub ops: u64,
    pub units: u64,
    pub bytes: u64,
    pub errors: u64,
    /// Temps cumulé DANS l'étage (pas le temps écoulé).
    pub seconds: f64,
    pub ms_per_op: Option<f64>,
    pub units_per_sec: Option<f64>,
    pub mb_per_sec: Option<f64>,
}

#[derive(Serialize, Clone, Debug)]
pub struct Throughput {
    /// Début de la fenêtre de mesure (unix).
    pub since_unix: u64,
    /// Temps écoulé depuis, en secondes. Comparé à `seconds` de chaque étage, il dit
    /// quelle PART du temps réel est passée dans les modèles — c'est ce rapport qui
    /// révèle si le goulot est le modèle ou le reste du pipeline.
    pub wall_seconds: f64,
    pub stages: Vec<StageStats>,
}

pub fn snapshot() -> Throughput {
    let m = metrics();
    let since = m.since.load(Relaxed);
    Throughput {
        since_unix: since,
        wall_seconds: now().saturating_sub(since) as f64,
        stages: vec![
            m.stage(Stage::Vision).snapshot(Stage::Vision),
            m.stage(Stage::Media).snapshot(Stage::Media),
            m.stage(Stage::Reasoning).snapshot(Stage::Reasoning),
            m.stage(Stage::Embedding).snapshot(Stage::Embedding),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moyennes_calculees_sur_le_temps_de_l_etage() {
        let c = Counters::default();
        c.record(Duration::from_millis(500), 10, 2_000_000);
        c.record(Duration::from_millis(500), 10, 2_000_000);
        let s = c.snapshot(Stage::Embedding);

        assert_eq!(s.ops, 2);
        assert_eq!(s.units, 20);
        assert!((s.seconds - 1.0).abs() < 1e-9);
        // 2 appels en 1 s au total → 500 ms par appel.
        assert!((s.ms_per_op.unwrap() - 500.0).abs() < 1e-6);
        // 20 chunks en 1 s.
        assert!((s.units_per_sec.unwrap() - 20.0).abs() < 1e-6);
        // 4 Mo en 1 s.
        assert!((s.mb_per_sec.unwrap() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn aucune_mesure_ne_donne_none_et_non_zero_ni_infini() {
        // Zéro appel ne veut pas dire « zéro par seconde » : c'est « pas mesuré ».
        // Confondre les deux afficherait un étage inactif comme un étage à l'arrêt.
        let s = Counters::default().snapshot(Stage::Vision);
        assert_eq!(s.ops, 0);
        assert!(s.ms_per_op.is_none());
        assert!(s.units_per_sec.is_none());
        assert!(s.mb_per_sec.is_none());
    }

    #[test]
    fn les_echecs_ne_faussent_pas_les_durees() {
        let c = Counters::default();
        c.record(Duration::from_millis(100), 1, 10);
        c.errors.fetch_add(3, Relaxed);
        let s = c.snapshot(Stage::Reasoning);
        assert_eq!(s.errors, 3);
        assert_eq!(s.ops, 1, "un échec n'est pas un appel mesuré");
        assert!((s.ms_per_op.unwrap() - 100.0).abs() < 1e-6);
    }
}

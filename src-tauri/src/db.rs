//! Couche de persistance relationnelle (SQLite via un pool r2d2).
//!
//! SQLite gère les métadonnées (catalogue de fichiers), la file d'indexation,
//! le journal des transactions Dry-Run et la table de vérité de synchronisation.
//! Le passage d'un `Mutex<Connection>` unique à un pool r2d2 + WAL permet aux
//! lectures IPC de ne plus être bloquées par les écritures du worker de fond.

use anyhow::{Context, Result};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::params;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

pub type DbPool = Pool<SqliteConnectionManager>;

/// Échappe les métacaractères LIKE (`%`, `_`, `\`) pour une recherche par préfixe
/// exacte. À utiliser avec `... LIKE ?n ESCAPE '\'` et un suffixe `%` littéral.
fn escape_like(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + 8);
    for ch in input.chars() {
        if matches!(ch, '\\' | '%' | '_') {
            out.push('\\');
        }
        out.push(ch);
    }
    out
}

fn like_prefix(prefix: &str) -> String {
    format!("{}%", escape_like(prefix))
}

#[derive(Debug)]
pub struct Database {
    pool: DbPool,
    path: PathBuf,
}

#[derive(Debug, Serialize)]
pub struct FileRecord {
    pub id: i64,
    pub path: String,
    pub name: String,
    pub updated_at: String,
}

/// Métadonnées structurelles d'un fichier ou dossier (alimente le catalogue).
#[derive(Debug, Clone)]
pub struct FileMeta {
    pub path: String,
    pub parent_path: Option<String>,
    pub file_name: String,
    pub is_directory: bool,
    pub size_bytes: Option<i64>,
    pub modified_at: Option<i64>,
}

/// Une tâche d'extraction en attente, avec son compteur de tentatives.
#[derive(Debug)]
pub struct ExtractionTask {
    pub id: i64,
    pub path: String,
    pub retry_count: i64,
}

/// Un groupe de doublons détecté par hash de contenu identique.
#[derive(Debug, Serialize)]
pub struct DuplicateGroup {
    pub content_hash: String,
    pub paths: Vec<String>,
}

/// Une transaction Dry-Run persistée (draft/committed/discarded).
#[derive(Debug, Serialize)]
pub struct TransactionRecord {
    pub id: i64,
    pub action: String,
    pub payload_json: String,
    pub status: String,
}

/// Avancement de l'indexation (pour l'indicateur de progression de l'UI).
#[derive(Debug, Serialize)]
pub struct IndexingStats {
    pub total: i64,
    pub pending: i64,
    pub completed: i64,
    pub failed: i64,
    /// Dossiers dont la classification est reportée faute d'IA disponible.
    pub pending_folders: i64,
}

/// Une entrée de la file d'indexation (pour le panneau de file en temps réel).
#[derive(Debug, Serialize)]
pub struct QueueEntry {
    pub path: String,
    pub status: String,
    pub retry_count: i64,
    pub last_error: Option<String>,
}

impl Database {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("création du dossier de la base: {}", parent.display()))?;
        }

        // Chaque connexion du pool applique les pragmas de robustesse/performance.
        let manager = SqliteConnectionManager::file(&path).with_init(|conn| {
            conn.execute_batch(
                r#"
                PRAGMA journal_mode = WAL;
                PRAGMA synchronous = NORMAL;
                PRAGMA busy_timeout = 5000;
                PRAGMA foreign_keys = ON;
                "#,
            )
        });

        let pool = Pool::builder()
            .max_size(8)
            .build(manager)
            .context("construction du pool SQLite")?;

        let db = Database { pool, path };
        db.run_migrations()?;
        Ok(db)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn conn(&self) -> Result<r2d2::PooledConnection<SqliteConnectionManager>> {
        self.pool.get().context("obtention d'une connexion SQLite")
    }

    fn run_migrations(&self) -> Result<()> {
        let conn = self.conn()?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS file_catalog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                parent_path TEXT,
                file_name TEXT NOT NULL,
                is_directory INTEGER NOT NULL DEFAULT 0,
                content_hash TEXT,
                size_bytes INTEGER,
                modified_at TEXT,
                indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_file_catalog_parent_path
                ON file_catalog(parent_path);
            CREATE INDEX IF NOT EXISTS idx_file_catalog_content_hash
                ON file_catalog(content_hash);

            CREATE TABLE IF NOT EXISTS indexing_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                reason TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                priority INTEGER NOT NULL DEFAULT 0,
                retry_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                enqueued_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_indexing_queue_status_priority
                ON indexing_queue(status, priority DESC, enqueued_at ASC);

            CREATE TABLE IF NOT EXISTS transaction_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'draft',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                committed_at TEXT,
                rolled_back_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_transaction_log_status_created_at
                ON transaction_log(status, created_at DESC);

            -- Table de vérité pour la synchronisation incrémentale du crawler.
            CREATE TABLE IF NOT EXISTS indexed_files (
                path TEXT PRIMARY KEY,
                last_modified INTEGER NOT NULL,
                last_seen INTEGER NOT NULL
            );

            -- Résumé sémantique léger par fichier (contexte pour le gardener / actions).
            CREATE TABLE IF NOT EXISTS file_semantics (
                path TEXT PRIMARY KEY,
                summary TEXT,
                extract TEXT,
                doc_type TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            -- Mémoire de l'agent : faits/préférences durables retenus entre conversations.
            CREATE TABLE IF NOT EXISTS agent_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                note TEXT NOT NULL,
                created_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
            );

            -- Mode de traitement d'un dossier : 'recursive' (indexation fichier par
            -- fichier) ou 'block' (indexé comme une unité sémantique unique).
            CREATE TABLE IF NOT EXISTS folder_profiles (
                path TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                source TEXT NOT NULL,        -- 'heuristic' | 'llm' | 'manual'
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            "#,
        )?;

        // Migrations additives tolérantes (bases créées avant l'ajout des colonnes).
        Self::add_column_if_missing(&conn, "indexing_queue", "retry_count", "INTEGER NOT NULL DEFAULT 0");
        Self::add_column_if_missing(&conn, "indexing_queue", "last_error", "TEXT");
        Self::add_column_if_missing(&conn, "file_semantics", "extract", "TEXT");

        // Réconciliation versionnée (auto-guérison).
        //
        // Une base issue d'une version antérieure de l'app peut contenir un état
        // de synchronisation incohérent avec le pipeline vectoriel actuel (file
        // marquée « vue » alors qu'aucun vecteur n'existe, statuts hérités que le
        // worker ne lit pas…). On purge alors les tables DÉRIVÉES (toutes
        // régénérables : tracking, file d'attente, résumés) pour forcer une
        // ré-indexation propre. Les fichiers de l'utilisateur ne sont jamais touchés.
        const SCHEMA_VERSION: i64 = 5;
        let current: i64 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap_or(0);

        // v3 : réconciliation complète (état de synchro hérité incohérent).
        if current < 3 {
            tracing::info!("réconciliation de l'index (v{current}→v3) : ré-indexation forcée");
            let _ = conn.execute_batch(
                "DELETE FROM indexed_files; DELETE FROM indexing_queue; DELETE FROM file_semantics;",
            );
        }
        // v4 : ré-extraire seulement les fichiers indexés « par contexte » (hash 'ctx:%'),
        // pour récupérer notamment les PDF désormais déchiffrables — sans tout re-wiper.
        if current < 4 {
            tracing::info!("ré-extraction ciblée des fichiers indexés par contexte (v{current}→v4)");
            let _ = conn.execute_batch(
                "UPDATE indexing_queue SET status='pending_extraction', retry_count=0, last_error=NULL \
                   WHERE path IN (SELECT path FROM file_catalog WHERE content_hash LIKE 'ctx:%'); \
                 DELETE FROM indexed_files \
                   WHERE path IN (SELECT path FROM file_catalog WHERE content_hash LIKE 'ctx:%');",
            );
        }

        // v5 : la logique de classification des dossiers a changé (LLM-centrée,
        // moins agressive). On repart de zéro sur les profils pour tout reclasser ;
        // les dossiers auparavant bloqués à tort verront leurs fichiers ré-indexés
        // au prochain scan (ils n'étaient pas dans indexed_files).
        if current < 5 {
            tracing::info!("reclassification des dossiers (v{current}→v5) : profils réinitialisés");
            let _ = conn.execute_batch("DELETE FROM folder_profiles;");
        }

        if current < SCHEMA_VERSION {
            let _ = conn.execute_batch(&format!("PRAGMA user_version = {SCHEMA_VERSION};"));
        }

        Ok(())
    }

    /// Réinitialise entièrement l'index (tables dérivées régénérables). Le crawler
    /// et le worker reconstruiront tout. Les fichiers de l'utilisateur sont intacts.
    pub fn reset_index(&self) -> Result<()> {
        let conn = self.conn()?;
        // On efface AUSSI les profils de dossiers (bloc/récursif) : une réindexation
        // doit reclasser depuis zéro pour prendre en compte un nouveau réglage de
        // tendance (curseur bloc/récursif) ou de prompt de classification.
        conn.execute_batch(
            "DELETE FROM indexed_files; DELETE FROM indexing_queue; \
             DELETE FROM file_semantics; DELETE FROM folder_profiles; \
             UPDATE file_catalog SET content_hash = NULL;",
        )?;
        Ok(())
    }

    /// Oublie toutes les décisions bloc/récursif (sans toucher à l'index) : sert
    /// quand l'utilisateur change le réglage de classification et veut le réappliquer.
    pub fn clear_folder_profiles(&self) -> Result<()> {
        self.conn()?.execute("DELETE FROM folder_profiles", [])?;
        Ok(())
    }

    fn add_column_if_missing(conn: &rusqlite::Connection, table: &str, column: &str, decl: &str) {
        let sql = format!("ALTER TABLE {table} ADD COLUMN {column} {decl}");
        // Ignore l'erreur "duplicate column name" si la colonne existe déjà.
        let _ = conn.execute(&sql, []);
    }

    // =====================================================================
    // CATALOGUE DE FICHIERS
    // =====================================================================

    /// Insère/met à jour un lot d'entrées de catalogue dans une seule transaction.
    pub fn bulk_upsert_file_records(&self, records: &[FileMeta]) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn()?;
        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                r#"
                INSERT INTO file_catalog
                    (path, parent_path, file_name, is_directory, size_bytes, modified_at, indexed_at)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, CURRENT_TIMESTAMP)
                ON CONFLICT(path) DO UPDATE SET
                    parent_path = excluded.parent_path,
                    file_name = excluded.file_name,
                    is_directory = excluded.is_directory,
                    size_bytes = excluded.size_bytes,
                    modified_at = excluded.modified_at,
                    indexed_at = CURRENT_TIMESTAMP
                "#,
            )?;
            for r in records {
                stmt.execute(params![
                    r.path,
                    r.parent_path,
                    r.file_name,
                    r.is_directory as i32,
                    r.size_bytes,
                    r.modified_at,
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn upsert_file_record(
        &self,
        path: &str,
        parent_path: Option<&str>,
        file_name: &str,
        is_directory: bool,
        content_hash: Option<&str>,
        size_bytes: Option<i64>,
        modified_at: Option<&str>,
    ) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            r#"
            INSERT INTO file_catalog
                (path, parent_path, file_name, is_directory, content_hash, size_bytes, modified_at, indexed_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, CURRENT_TIMESTAMP)
            ON CONFLICT(path) DO UPDATE SET
                parent_path = excluded.parent_path,
                file_name = excluded.file_name,
                is_directory = excluded.is_directory,
                content_hash = excluded.content_hash,
                size_bytes = excluded.size_bytes,
                modified_at = excluded.modified_at,
                indexed_at = CURRENT_TIMESTAMP
            "#,
            params![
                path,
                parent_path,
                file_name,
                is_directory as i32,
                content_hash,
                size_bytes,
                modified_at,
            ],
        )?;
        Ok(())
    }

    /// Renseigne le hash de contenu une fois le fichier lu par le worker.
    pub fn update_file_hash(&self, path: &str, content_hash: &str) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "UPDATE file_catalog SET content_hash = ?1 WHERE path = ?2",
            params![content_hash, path],
        )?;
        Ok(())
    }

    /// Renvoie le hash de contenu connu pour ce chemin, s'il existe.
    pub fn get_stored_hash(&self, path: &str) -> Result<Option<String>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare("SELECT content_hash FROM file_catalog WHERE path = ?1")?;
        let mut rows = stmt.query(params![path])?;
        if let Some(row) = rows.next()? {
            Ok(row.get(0)?)
        } else {
            Ok(None)
        }
    }

    /// Met à jour le chemin d'un fichier (après un déplacement/renommage validé).
    pub fn rename_catalog_path(&self, old_path: &str, new_path: &str) -> Result<()> {
        let conn = self.conn()?;
        let new_parent = Path::new(new_path)
            .parent()
            .map(|p| p.to_string_lossy().to_string());
        let new_name = Path::new(new_path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        conn.execute(
            "UPDATE file_catalog SET path = ?1, parent_path = ?2, file_name = ?3 WHERE path = ?4",
            params![new_path, new_parent, new_name, old_path],
        )?;
        conn.execute(
            "UPDATE file_semantics SET path = ?1 WHERE path = ?2",
            params![new_path, old_path],
        )?;
        // La table de vérité doit suivre le renommage, sinon le crawler re-indexerait.
        conn.execute(
            "UPDATE indexed_files SET path = ?1 WHERE path = ?2",
            params![new_path, old_path],
        )?;
        conn.execute(
            "UPDATE indexing_queue SET path = ?1 WHERE path = ?2",
            params![new_path, old_path],
        )?;
        Ok(())
    }

    pub fn remove_catalog_path(&self, path: &str) -> Result<()> {
        let conn = self.conn()?;
        conn.execute("DELETE FROM file_catalog WHERE path = ?1", params![path])?;
        conn.execute("DELETE FROM file_semantics WHERE path = ?1", params![path])?;
        conn.execute("DELETE FROM indexed_files WHERE path = ?1", params![path])?;
        Ok(())
    }

    /// Détecte les groupes de fichiers au contenu strictement identique.
    pub fn find_duplicates(&self, prefix: &str) -> Result<Vec<DuplicateGroup>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT content_hash, GROUP_CONCAT(path, '|')
            FROM file_catalog
            WHERE content_hash IS NOT NULL
              AND is_directory = 0
              AND path LIKE ?1 ESCAPE '\'
            GROUP BY content_hash
            HAVING COUNT(*) > 1
            "#,
        )?;
        let rows = stmt.query_map(params![like_prefix(prefix)], |row| {
            let hash: String = row.get(0)?;
            let joined: String = row.get(1)?;
            Ok(DuplicateGroup {
                content_hash: hash,
                paths: joined.split('|').map(|s| s.to_string()).collect(),
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Renvoie le statut d'indexation des chemins situés directement sous `parent`.
    pub fn queue_statuses_for_parent(&self, parent: &str) -> Result<Vec<(String, String)>> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare("SELECT path, status FROM indexing_queue WHERE path LIKE ?1 ESCAPE '\\'")?;
        let rows = stmt.query_map(params![like_prefix(parent)], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    // =====================================================================
    // FILE D'INDEXATION
    // =====================================================================

    pub fn enqueue_path(&self, path: &str, target_status: Option<&str>, priority: i64) -> Result<()> {
        let status_val = target_status.unwrap_or("pending");
        let conn = self.conn()?;
        conn.execute(
            r#"
            INSERT INTO indexing_queue (path, status, priority, retry_count, enqueued_at, updated_at)
            VALUES (?1, ?2, ?3, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(path) DO UPDATE SET
                status = excluded.status,
                priority = CASE
                    WHEN excluded.priority > indexing_queue.priority THEN excluded.priority
                    ELSE indexing_queue.priority
                END,
                retry_count = 0,
                last_error = NULL,
                updated_at = CURRENT_TIMESTAMP
            "#,
            params![path, status_val, priority],
        )?;
        Ok(())
    }

    /// Force la remise en file d'un chemin (relance manuelle depuis le panneau) :
    /// contrairement à `enqueue_path`, réinitialise MÊME un échec définitif.
    pub fn requeue_path(&self, path: &str) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "UPDATE indexing_queue SET status = 'pending_extraction', retry_count = 0, \
             last_error = NULL, updated_at = CURRENT_TIMESTAMP WHERE path = ?1",
            params![path],
        )?;
        Ok(())
    }

    pub fn remove_from_queue(&self, path: &str) -> Result<()> {
        let conn = self.conn()?;
        conn.execute("DELETE FROM indexing_queue WHERE path = ?1", params![path])?;
        Ok(())
    }

    /// Compacte le journal WAL dans le fichier principal (arrêt propre). Sans ça, la
    /// base peut rester quasi vide avec toutes les données dans le `-wal`.
    pub fn checkpoint(&self) {
        if let Ok(conn) = self.conn() {
            let _ = conn.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |_| Ok(()));
        }
    }

    /// Remet toutes les tâches en échec définitif dans la file (relance globale). Renvoie le nombre relancé.
    pub fn requeue_failed(&self) -> Result<usize> {
        let conn = self.conn()?;
        let n = conn.execute(
            "UPDATE indexing_queue SET status = 'pending_extraction', retry_count = 0, last_error = NULL \
             WHERE status = 'failed_permanent'",
            [],
        )?;
        Ok(n)
    }

    /// Récupère un lot de tâches d'extraction en attente (avec leur compteur de tentatives).
    pub fn get_pending_extraction_tasks(&self, limit: i64) -> Result<Vec<ExtractionTask>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT id, path, retry_count
            FROM indexing_queue
            WHERE status = 'pending_extraction'
            ORDER BY priority DESC, enqueued_at ASC
            LIMIT ?1
            "#,
        )?;
        let rows = stmt.query_map(params![limit], |row| {
            Ok(ExtractionTask {
                id: row.get(0)?,
                path: row.get(1)?,
                retry_count: row.get(2)?,
            })
        })?;
        let mut tasks = Vec::new();
        for t in rows {
            tasks.push(t?);
        }
        Ok(tasks)
    }

    pub fn update_task_status(&self, id: i64, status: &str) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "UPDATE indexing_queue SET status = ?1, updated_at = CURRENT_TIMESTAMP WHERE id = ?2",
            params![status, id],
        )?;
        Ok(())
    }

    /// Enregistre un échec : incrémente les tentatives et remet en file, ou abandonne
    /// définitivement au-delà de `max_retries`.
    pub fn record_task_failure(&self, id: i64, error: &str, max_retries: i64) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            r#"
            UPDATE indexing_queue
            SET retry_count = retry_count + 1,
                last_error = ?1,
                status = CASE
                    WHEN retry_count + 1 >= ?2 THEN 'failed_permanent'
                    ELSE 'pending_extraction'
                END,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?3
            "#,
            params![error, max_retries, id],
        )?;
        Ok(())
    }

    // =====================================================================
    // SÉMANTIQUE / RÉSUMÉS
    // =====================================================================

    pub fn upsert_file_summary(&self, path: &str, summary: &str, doc_type: &str) -> Result<()> {
        self.upsert_file_semantics(path, summary, None, doc_type)
    }

    /// Comme `upsert_file_summary`, mais stocke aussi le CONTENU EXTRAIT (`extract`),
    /// pour pouvoir l'afficher à côté de la qualification (« sens ») et le comparer au document.
    pub fn upsert_file_semantics(
        &self,
        path: &str,
        summary: &str,
        extract: Option<&str>,
        doc_type: &str,
    ) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            r#"
            INSERT INTO file_semantics (path, summary, extract, doc_type, updated_at)
            VALUES (?1, ?2, ?3, ?4, CURRENT_TIMESTAMP)
            ON CONFLICT(path) DO UPDATE SET
                summary = excluded.summary,
                extract = excluded.extract,
                doc_type = excluded.doc_type,
                updated_at = CURRENT_TIMESTAMP
            "#,
            params![path, summary, extract, doc_type],
        )?;
        Ok(())
    }

    /// Édition manuelle du « sens » d'un fichier (crée l'entrée si absente, sinon
    /// remplace juste le texte en conservant le type). Marque le type 'manuel' à la création.
    pub fn set_file_summary(&self, path: &str, summary: &str) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            r#"
            INSERT INTO file_semantics (path, summary, doc_type, updated_at)
            VALUES (?1, ?2, 'manuel', CURRENT_TIMESTAMP)
            ON CONFLICT(path) DO UPDATE SET
                summary = excluded.summary,
                updated_at = CURRENT_TIMESTAMP
            "#,
            params![path, summary],
        )?;
        Ok(())
    }

    /// Fichiers indexés SOUS `folder` disposant d'un contenu extrait (donc qualifiables) :
    /// `(path, doc_type, summary, extract)`. Sert à la qualification de tout un dossier.
    pub fn qualifiable_under(&self, folder: &str) -> Result<Vec<(String, String, String, String)>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT path, COALESCE(doc_type,''), COALESCE(summary,''), extract
            FROM file_semantics
            WHERE path LIKE ?1 ESCAPE '\' AND extract IS NOT NULL AND TRIM(extract) <> ''
            "#,
        )?;
        let rows = stmt.query_map(params![like_prefix(folder)], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Résumés des fichiers situés directement sous `parent` (contexte pour les actions).
    // =====================================================================
    // MÉMOIRE DE L'AGENT (faits/préférences durables entre conversations)
    // =====================================================================

    /// Ajoute une note à la mémoire (ignore un doublon exact).
    pub fn add_memory(&self, note: &str) -> Result<()> {
        let note = note.trim();
        if note.is_empty() {
            return Ok(());
        }
        let conn = self.conn()?;
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM agent_memory WHERE note = ?1",
            params![note],
            |r| r.get(0),
        )?;
        if count == 0 {
            conn.execute("INSERT INTO agent_memory (note) VALUES (?1)", params![note])?;
        }
        Ok(())
    }

    /// Notes de la mémoire, plus récentes d'abord, bornées à `limit`. Renvoie (id, note).
    pub fn list_memories(&self, limit: usize) -> Result<Vec<(i64, String)>> {
        let conn = self.conn()?;
        let mut stmt =
            conn.prepare("SELECT id, note FROM agent_memory ORDER BY id DESC LIMIT ?1")?;
        let rows = stmt.query_map(params![limit as i64], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Supprime une note par id.
    pub fn delete_memory(&self, id: i64) -> Result<()> {
        let conn = self.conn()?;
        conn.execute("DELETE FROM agent_memory WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// Vide toute la mémoire de l'agent.
    pub fn clear_memories(&self) -> Result<()> {
        let conn = self.conn()?;
        conn.execute("DELETE FROM agent_memory", [])?;
        Ok(())
    }

    pub fn summaries_for_parent(&self, parent: &str) -> Result<Vec<(String, String)>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT path, COALESCE(summary, '') FROM file_semantics WHERE path LIKE ?1 ESCAPE '\\'",
        )?;
        let rows = stmt.query_map(params![like_prefix(parent)], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    // =====================================================================
    // PROFILS DE DOSSIERS (récursif vs bloc sémantique)
    // =====================================================================

    /// Renvoie le mode connu d'un dossier : (mode, source).
    pub fn get_folder_mode(&self, path: &str) -> Result<Option<(String, String)>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare("SELECT mode, source FROM folder_profiles WHERE path = ?1")?;
        let mut rows = stmt.query(params![path])?;
        if let Some(row) = rows.next()? {
            Ok(Some((row.get(0)?, row.get(1)?)))
        } else {
            Ok(None)
        }
    }

    /// Enregistre une classification automatique (heuristique/LLM) — sans jamais
    /// écraser un choix manuel de l'utilisateur.
    pub fn set_folder_profile(&self, path: &str, mode: &str, source: &str) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            r#"
            INSERT INTO folder_profiles (path, mode, source, updated_at)
            VALUES (?1, ?2, ?3, CURRENT_TIMESTAMP)
            ON CONFLICT(path) DO UPDATE SET
                mode = excluded.mode,
                source = excluded.source,
                updated_at = CURRENT_TIMESTAMP
            WHERE folder_profiles.source != 'manual'
            "#,
            params![path, mode, source],
        )?;
        Ok(())
    }

    /// Force un choix manuel (prioritaire sur toute classification automatique).
    pub fn set_folder_profile_manual(&self, path: &str, mode: &str) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            r#"
            INSERT INTO folder_profiles (path, mode, source, updated_at)
            VALUES (?1, ?2, 'manual', CURRENT_TIMESTAMP)
            ON CONFLICT(path) DO UPDATE SET
                mode = excluded.mode, source = 'manual', updated_at = CURRENT_TIMESTAMP
            "#,
            params![path, mode],
        )?;
        Ok(())
    }

    /// Détails d'indexation d'un chemin (pour le panneau de détail).
    pub fn get_file_semantics(
        &self,
        path: &str,
    ) -> Result<Option<(String, String, Option<String>)>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT COALESCE(summary,''), COALESCE(doc_type,''), extract FROM file_semantics WHERE path = ?1",
        )?;
        let mut rows = stmt.query(params![path])?;
        if let Some(row) = rows.next()? {
            Ok(Some((row.get(0)?, row.get(1)?, row.get(2)?)))
        } else {
            Ok(None)
        }
    }

    /// Statut de file + dernière erreur d'un chemin.
    pub fn get_queue_status(&self, path: &str) -> Result<Option<(String, Option<String>)>> {
        let conn = self.conn()?;
        let mut stmt =
            conn.prepare("SELECT status, last_error FROM indexing_queue WHERE path = ?1")?;
        let mut rows = stmt.query(params![path])?;
        if let Some(row) = rows.next()? {
            Ok(Some((row.get(0)?, row.get(1)?)))
        } else {
            Ok(None)
        }
    }

    pub fn is_indexed(&self, path: &str) -> Result<bool> {
        let conn = self.conn()?;
        // `last_modified > 0` : réellement indexé, pas seulement « vu » par le crawler.
        let mut stmt =
            conn.prepare("SELECT 1 FROM indexed_files WHERE path = ?1 AND last_modified > 0")?;
        Ok(stmt.exists(params![path])?)
    }

    /// Vrai si `path` (inclus) ou l'un de ses ancêtres est un dossier traité en bloc.
    pub fn is_under_block(&self, path: &str) -> Result<bool> {
        let conn = self.conn()?;
        let mut stmt =
            conn.prepare("SELECT 1 FROM folder_profiles WHERE path = ?1 AND mode = 'block'")?;
        let mut cur = Some(std::path::Path::new(path));
        while let Some(p) = cur {
            if stmt.exists(params![p.to_string_lossy().as_ref()])? {
                return Ok(true);
            }
            cur = p.parent();
        }
        Ok(false)
    }

    /// Chemins effectivement indexés (embeddés) sous `parent` — pour l'indicateur
    /// « indexé » de l'explorateur (fichiers et dossiers-blocs).
    pub fn indexed_paths_under(&self, parent: &str) -> Result<Vec<String>> {
        let conn = self.conn()?;
        // `last_modified > 0` : uniquement les fichiers RÉELLEMENT indexés (mark_indexed).
        // Le crawler insère chaque fichier vu avec `last_modified = 0` (touch_seen) ;
        // sans ce filtre, tout fichier juste croisé (mais encore en file) apparaissait
        // « indexé ✓ » alors qu'il n'a pas encore de sens.
        let mut stmt = conn.prepare(
            "SELECT path FROM indexed_files WHERE path LIKE ?1 ESCAPE '\\' AND last_modified > 0",
        )?;
        let rows = stmt.query_map(params![like_prefix(parent)], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Modes des dossiers situés directement sous `parent` (pour les badges de l'explorateur).
    pub fn folder_modes_under(&self, parent: &str) -> Result<Vec<(String, String)>> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare("SELECT path, mode FROM folder_profiles WHERE path LIKE ?1 ESCAPE '\\'")?;
        let rows = stmt.query_map(params![like_prefix(parent)], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Purge toutes les données d'index des ENFANTS d'un dossier (utilisé quand un
    /// dossier passe en mode bloc : on retire l'indexation fichier par fichier).
    pub fn purge_children(&self, folder: &str) -> Result<()> {
        let conn = self.conn()?;
        let pattern = like_prefix(&format!("{}{}", folder.trim_end_matches(['/', '\\']), std::path::MAIN_SEPARATOR));
        for table in ["indexed_files", "file_semantics", "indexing_queue", "file_catalog"] {
            let sql = format!("DELETE FROM {table} WHERE path LIKE ?1 ESCAPE '\\'");
            conn.execute(&sql, params![pattern])?;
        }
        Ok(())
    }

    /// Purge tout un sous-arbre (le dossier lui-même ET ses descendants), profils
    /// de dossiers inclus — utilisé quand un dossier est retiré de l'indexation.
    pub fn purge_tree(&self, folder: &str) -> Result<()> {
        let conn = self.conn()?;
        let root = folder.trim_end_matches(['/', '\\']);
        let child_pattern = like_prefix(&format!("{root}{}", std::path::MAIN_SEPARATOR));
        for table in [
            "indexed_files",
            "file_semantics",
            "indexing_queue",
            "file_catalog",
            "folder_profiles",
        ] {
            conn.execute(
                &format!("DELETE FROM {table} WHERE path LIKE ?1 ESCAPE '\\'"),
                params![child_pattern],
            )?;
            conn.execute(&format!("DELETE FROM {table} WHERE path = ?1"), params![root])?;
        }
        Ok(())
    }

    // =====================================================================
    // TRANSACTIONS DRY-RUN
    // =====================================================================

    pub fn record_transaction(&self, action: &str, payload_json: &str, status: &str) -> Result<i64> {
        let conn = self.conn()?;
        conn.execute(
            "INSERT INTO transaction_log (action, payload_json, status, created_at) VALUES (?1, ?2, ?3, CURRENT_TIMESTAMP)",
            params![action, payload_json, status],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn get_transaction(&self, id: i64) -> Result<Option<TransactionRecord>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, action, payload_json, status FROM transaction_log WHERE id = ?1",
        )?;
        let mut rows = stmt.query(params![id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(TransactionRecord {
                id: row.get(0)?,
                action: row.get(1)?,
                payload_json: row.get(2)?,
                status: row.get(3)?,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn mark_transaction_committed(&self, id: i64) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "UPDATE transaction_log SET status = 'committed', committed_at = CURRENT_TIMESTAMP WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    pub fn mark_transaction_discarded(&self, id: i64) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "UPDATE transaction_log SET status = 'discarded' WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    // =====================================================================
    // SYNCHRONISATION INCRÉMENTALE (CRAWLER)
    // =====================================================================

    pub fn needs_indexing(&self, path: &str, current_mtime: i64) -> Result<bool> {
        let conn = self.conn()?;
        // On récupère en une requête le mtime déjà indexé ET le statut en file.
        let (stored_mtime, status): (Option<i64>, Option<String>) = conn.query_row(
            "SELECT (SELECT last_modified FROM indexed_files WHERE path = ?1), \
                    (SELECT status FROM indexing_queue WHERE path = ?1)",
            params![path],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;

        // Échec DÉFINITIF : on ne le re-tente PAS automatiquement (le crawler le laisse
        // en échec, visible et actionnable dans le panneau ; la relance est manuelle).
        // Une vraie modification du fichier passera, elle, par le watchdog.
        if status.as_deref() == Some("failed_permanent") {
            return Ok(false);
        }

        match stored_mtime {
            Some(m) => Ok(current_mtime > m),
            None => Ok(true),
        }
    }

    /// Marque un fichier comme « vu » lors du scan courant, SANS toucher à
    /// `last_modified` : un fichier n'est considéré à jour qu'une fois réellement
    /// indexé (voir `mark_indexed`). Ainsi une interruption avant l'embedding
    /// laisse le fichier « à réindexer » et le système converge tout seul.
    pub fn touch_seen(&self, path: &str, scan_time: i64) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            r#"
            INSERT INTO indexed_files (path, last_modified, last_seen)
            VALUES (?1, 0, ?2)
            ON CONFLICT(path) DO UPDATE SET last_seen = ?2
            "#,
            params![path, scan_time],
        )?;
        Ok(())
    }

    /// Enregistre qu'un fichier a été effectivement indexé (embeddé + stocké).
    /// C'est le seul point qui met `last_modified`, garantissant qu'un fichier
    /// non encore vectorisé sera ré-enfilé au prochain scan.
    pub fn mark_indexed(&self, path: &str, mtime: i64) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            r#"
            INSERT INTO indexed_files (path, last_modified, last_seen)
            VALUES (?1, ?2, ?2)
            ON CONFLICT(path) DO UPDATE SET last_modified = ?2
            "#,
            params![path, mtime],
        )?;
        Ok(())
    }

    /// Renvoie les chemins orphelins (non revus lors du dernier scan) puis les supprime.
    /// Le worker utilisera ces chemins pour purger les vecteurs correspondants.
    pub fn take_orphans(&self, scan_time: i64, prefix: &str) -> Result<Vec<String>> {
        let conn = self.conn()?;
        let pattern = like_prefix(prefix);
        let orphans: Vec<String> = {
            let mut stmt = conn.prepare(
                "SELECT path FROM indexed_files WHERE last_seen < ?1 AND path LIKE ?2 ESCAPE '\\'",
            )?;
            let rows = stmt.query_map(params![scan_time, pattern], |row| {
                row.get::<_, String>(0)
            })?;
            rows.filter_map(|r| r.ok()).collect()
        };
        conn.execute(
            "DELETE FROM indexed_files WHERE last_seen < ?1 AND path LIKE ?2 ESCAPE '\\'",
            params![scan_time, pattern],
        )?;
        Ok(orphans)
    }

    // =====================================================================
    // ACTIVITÉ RÉCENTE (UI)
    // =====================================================================

    /// Compte l'avancement de l'indexation par statut.
    pub fn get_indexing_stats(&self) -> Result<IndexingStats> {
        let conn = self.conn()?;
        let mut stats = conn.query_row(
            r#"
            SELECT
                COUNT(*),
                SUM(CASE WHEN status IN ('pending', 'pending_extraction') THEN 1 ELSE 0 END),
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END),
                SUM(CASE WHEN status = 'failed_permanent' THEN 1 ELSE 0 END)
            FROM indexing_queue
            "#,
            [],
            |row| {
                Ok(IndexingStats {
                    total: row.get(0)?,
                    pending: row.get::<_, Option<i64>>(1)?.unwrap_or(0),
                    completed: row.get::<_, Option<i64>>(2)?.unwrap_or(0),
                    failed: row.get::<_, Option<i64>>(3)?.unwrap_or(0),
                    pending_folders: 0,
                })
            },
        )?;
        stats.pending_folders = conn
            .query_row(
                "SELECT COUNT(*) FROM folder_profiles WHERE mode = 'pending'",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);
        Ok(stats)
    }

    /// Prochaines entrées À TRAITER (hors échecs), dans l'ordre de traitement.
    ///
    /// Les échecs sont volontairement exclus et récupérés par `get_failed` : mélangés
    /// ici, ils étaient rejetés hors de la fenêtre `LIMIT` dès que la file était
    /// longue (des milliers d'éléments) et n'apparaissaient donc JAMAIS dans l'UI.
    pub fn get_queue(&self, limit: i64) -> Result<Vec<QueueEntry>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT path, status, retry_count, last_error
            FROM indexing_queue
            WHERE status IN ('pending', 'pending_extraction')
            ORDER BY priority DESC, enqueued_at ASC
            LIMIT ?1
            "#,
        )?;
        let rows = stmt.query_map(params![limit], |row| {
            Ok(QueueEntry {
                path: row.get(0)?,
                status: row.get(1)?,
                retry_count: row.get(2)?,
                last_error: row.get(3)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Entrées en ÉCHEC DÉFINITIF, requêtées à part de la file d'attente pour être
    /// toujours visibles/actionnables, même avec des milliers d'éléments en file.
    pub fn get_failed(&self, limit: i64) -> Result<Vec<QueueEntry>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT path, status, retry_count, last_error
            FROM indexing_queue
            WHERE status = 'failed_permanent'
            ORDER BY updated_at DESC
            LIMIT ?1
            "#,
        )?;
        let rows = stmt.query_map(params![limit], |row| {
            Ok(QueueEntry {
                path: row.get(0)?,
                status: row.get(1)?,
                retry_count: row.get(2)?,
                last_error: row.get(3)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Renvoie un lot de dossiers en attente de classification (mode 'pending').
    pub fn get_pending_folders(&self, limit: i64) -> Result<Vec<String>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT path FROM folder_profiles WHERE mode = 'pending' ORDER BY updated_at ASC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn get_recent_files(&self, limit: i64) -> Result<Vec<FileRecord>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, path, updated_at FROM indexing_queue ORDER BY updated_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit], |row| {
            let path: String = row.get(1)?;
            let name = Path::new(&path)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            Ok(FileRecord {
                id: row.get(0)?,
                path,
                name,
                updated_at: row.get(2)?,
            })
        })?;
        let mut files = Vec::new();
        for f in rows {
            files.push(f?);
        }
        Ok(files)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_db() -> Database {
        let p = std::env::temp_dir().join(format!("sensetree-test-{}.sqlite", uuid::Uuid::new_v4()));
        Database::open(&p).expect("ouverture base temporaire")
    }

    #[test]
    fn agent_memory_ajoute_dedoublonne_supprime() {
        let db = tmp_db();
        db.add_memory("factures dans Documents/Compta").unwrap();
        db.add_memory("factures dans Documents/Compta").unwrap(); // doublon exact ignoré
        db.add_memory("   ").unwrap(); // note vide ignorée
        db.add_memory("préférence : réponses courtes").unwrap();
        let mems = db.list_memories(10).unwrap();
        assert_eq!(mems.len(), 2, "dedup/vide non respectés : {mems:?}");
        db.delete_memory(mems[0].0).unwrap();
        assert_eq!(db.list_memories(10).unwrap().len(), 1);
        db.clear_memories().unwrap();
        assert!(db.list_memories(10).unwrap().is_empty());
    }

    #[test]
    fn echec_definitif_persiste_visible_et_saute_par_le_crawler() {
        let db = tmp_db();
        db.enqueue_path("C:/a/f.pdf", Some("pending_extraction"), 5).unwrap();
        let id = db
            .get_pending_extraction_tasks(10)
            .unwrap()
            .into_iter()
            .find(|t| t.path.ends_with("f.pdf"))
            .unwrap()
            .id;
        db.record_task_failure(id, "boom", 1).unwrap();

        assert!(db.get_failed(10).unwrap().iter().any(|q| q.path.ends_with("f.pdf")));
        assert!(!db.needs_indexing("C:/a/f.pdf", 999).unwrap());
        assert_eq!(db.requeue_failed().unwrap(), 1);
        assert!(db.get_failed(10).unwrap().is_empty());
        assert!(db.needs_indexing("C:/a/f.pdf", 999).unwrap());
    }

    #[test]
    fn tick_indexe_seulement_apres_mark_indexed() {
        let db = tmp_db();
        db.touch_seen("C:/a/g.txt", 100).unwrap();
        assert!(!db.is_indexed("C:/a/g.txt").unwrap());
        db.mark_indexed("C:/a/g.txt", 42).unwrap();
        assert!(db.is_indexed("C:/a/g.txt").unwrap());
    }
}

//! Stockage et recherche vectoriels via LanceDB (embarqué, sans serveur).
//!
//! Une seule table `chunks` contient un enregistrement par morceau de texte
//! (chunk) avec son vecteur dense. La table est créée paresseusement au premier
//! upsert, avec la dimension du modèle d'embedding actif.

use anyhow::{anyhow, Context, Result};
// IMPORTANT : on utilise les types Arrow ré-exportés par LanceDB pour garantir
// la correspondance exacte de version (LanceDB embarque sa propre version d'Arrow).
use futures_util::TryStreamExt;
use lancedb::arrow::arrow_array::{
    Array, FixedSizeListArray, Float32Array, Int32Array, Int64Array, RecordBatch,
    RecordBatchIterator, RecordBatchReader, StringArray,
};
use lancedb::arrow::arrow_schema::{DataType, Field, Schema};
use lancedb::index::scalar::FullTextSearchQuery;
use lancedb::index::Index;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, DistanceType};
use serde::Serialize;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use uuid::Uuid;

const TABLE: &str = "chunks";

/// Un morceau prêt à être indexé (texte + vecteur).
pub struct ChunkVector {
    pub chunk_index: i32,
    pub text: String,
    pub vector: Vec<f32>,
}

/// Un résultat de recherche (dense ou BM25). `score` = similarité cosinus (dense)
/// ou score BM25 (mots-clés) ; l'échelle diffère, mais la fusion RRF ne dépend que
/// du RANG, pas de la valeur absolue.
#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    pub path: String,
    pub chunk_index: i32,
    pub score: f32,
    /// Texte complet du chunk (pour le reranking cross-encoder).
    pub text: String,
    pub snippet: String,
}

pub struct VectorDb {
    conn: Connection,
    /// Dimension du modèle actif. Modifiable au runtime (changement de modèle
    /// d'embedding) : la table est alors recréée à la bonne dimension.
    dim: AtomicUsize,
    /// De nouveaux chunks ont été écrits depuis le dernier build de l'index BM25 :
    /// la prochaine recherche par mots-clés reconstruit l'index plein-texte.
    fts_dirty: AtomicBool,
}

impl VectorDb {
    pub async fn open(uri: &str, dim: usize) -> Result<Self> {
        let conn = lancedb::connect(uri)
            .execute()
            .await
            .context("ouverture de LanceDB")?;
        Ok(VectorDb {
            conn,
            dim: AtomicUsize::new(dim),
            fts_dirty: AtomicBool::new(true),
        })
    }

    pub fn dim(&self) -> usize {
        self.dim.load(Ordering::Relaxed)
    }

    /// Change la dimension attendue (après changement de modèle). L'appelant DOIT
    /// vider la table ensuite (`clear`) : elle sera recréée à la nouvelle dimension.
    pub fn set_dim(&self, dim: usize) {
        self.dim.store(dim, Ordering::Relaxed);
    }

    fn schema(&self) -> Arc<Schema> {
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("path", DataType::Utf8, false),
            Field::new("chunk_index", DataType::Int32, false),
            Field::new("text", DataType::Utf8, false),
            Field::new("content_hash", DataType::Utf8, true),
            Field::new("mtime", DataType::Int64, true),
            Field::new(
                "vector",
                DataType::FixedSizeList(item, self.dim() as i32),
                false,
            ),
        ]))
    }

    /// Ouvre la table, en la créant (vide) si elle n'existe pas encore.
    async fn table(&self) -> Result<lancedb::Table> {
        match self.conn.open_table(TABLE).execute().await {
            Ok(t) => Ok(t),
            Err(_) => self
                .conn
                .create_empty_table(TABLE, self.schema())
                .execute()
                .await
                .context("création de la table de vecteurs"),
        }
    }

    /// Remplace tous les chunks d'un fichier par les nouveaux (delete-then-insert).
    pub async fn upsert_chunks(
        &self,
        path: &str,
        content_hash: &str,
        mtime: i64,
        chunks: Vec<ChunkVector>,
    ) -> Result<()> {
        let table = self.table().await?;

        // On repart d'une base propre pour ce chemin.
        table.delete(&path_filter(path)).await.ok();
        // Le corpus a changé → l'index BM25 devra être reconstruit avant la
        // prochaine recherche par mots-clés.
        self.fts_dirty.store(true, Ordering::Relaxed);

        if chunks.is_empty() {
            return Ok(());
        }

        let n = chunks.len();
        let mut ids = Vec::with_capacity(n);
        let mut paths = Vec::with_capacity(n);
        let mut indices = Vec::with_capacity(n);
        let mut texts = Vec::with_capacity(n);
        let mut hashes = Vec::with_capacity(n);
        let mut mtimes = Vec::with_capacity(n);
        let dim = self.dim();
        let mut flat: Vec<f32> = Vec::with_capacity(n * dim);

        for c in chunks {
            if c.vector.len() != dim {
                return Err(anyhow!(
                    "dimension du vecteur ({}) incohérente avec la table ({})",
                    c.vector.len(),
                    dim
                ));
            }
            ids.push(Uuid::new_v4().to_string());
            paths.push(path.to_string());
            indices.push(c.chunk_index);
            texts.push(c.text);
            hashes.push(content_hash.to_string());
            mtimes.push(mtime);
            flat.extend_from_slice(&c.vector);
        }

        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let values = Float32Array::from(flat);
        let vectors = FixedSizeListArray::new(item, dim as i32, Arc::new(values), None);

        let batch = RecordBatch::try_new(
            self.schema(),
            vec![
                Arc::new(StringArray::from(ids)),
                Arc::new(StringArray::from(paths)),
                Arc::new(Int32Array::from(indices)),
                Arc::new(StringArray::from(texts)),
                Arc::new(StringArray::from(hashes)),
                Arc::new(Int64Array::from(mtimes)),
                Arc::new(vectors),
            ],
        )
        .context("construction du RecordBatch")?;

        let schema = self.schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let boxed: Box<dyn RecordBatchReader + Send> = Box::new(reader);
        table
            .add(boxed)
            .execute()
            .await
            .context("insertion des vecteurs")?;
        Ok(())
    }

    /// Recherche les plus proches voisins, éventuellement restreinte à un préfixe de chemin.
    pub async fn search(
        &self,
        query_vec: Vec<f32>,
        limit: usize,
        scope_prefix: Option<&str>,
    ) -> Result<Vec<SearchHit>> {
        // Pas de table => pas de résultats (indexation pas encore démarrée).
        let table = match self.conn.open_table(TABLE).execute().await {
            Ok(t) => t,
            Err(_) => return Ok(Vec::new()),
        };

        let mut query = table
            .query()
            .nearest_to(query_vec)
            .context("préparation de la requête vectorielle")?
            .distance_type(DistanceType::Cosine)
            .limit(limit);

        if let Some(prefix) = scope_prefix {
            query = query.only_if(scope_like_filter(prefix));
        }

        let batches: Vec<RecordBatch> = query
            .execute()
            .await
            .context("exécution de la recherche vectorielle")?
            .try_collect()
            .await
            .context("collecte des résultats")?;

        let mut hits = Vec::new();
        for batch in batches {
            let paths = column_as_string(&batch, "path")?;
            let texts = column_as_string(&batch, "text")?;
            let indices = batch
                .column_by_name("chunk_index")
                .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
            let distances = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

            for i in 0..batch.num_rows() {
                let distance = distances.map(|d| d.value(i)).unwrap_or(1.0);
                let full = texts.value(i);
                hits.push(SearchHit {
                    path: paths.value(i).to_string(),
                    chunk_index: indices.map(|idx| idx.value(i)).unwrap_or(0),
                    // Distance cosine ∈ [0,2] → score de similarité ∈ [-1,1] approx.
                    score: 1.0 - distance,
                    text: full.to_string(),
                    snippet: snippet(full),
                });
            }
        }
        Ok(hits)
    }

    /// Recherche par MOTS-CLÉS (BM25) via l'index plein-texte natif de LanceDB sur
    /// la colonne `text`. Reconstruit l'index si de nouveaux chunks ont été écrits.
    /// Renvoie les hits triés par score BM25 décroissant.
    pub async fn keyword_search(
        &self,
        query: &str,
        limit: usize,
        scope_prefix: Option<&str>,
    ) -> Result<Vec<SearchHit>> {
        let q = query.trim();
        if q.is_empty() {
            return Ok(Vec::new());
        }
        let table = match self.conn.open_table(TABLE).execute().await {
            Ok(t) => t,
            Err(_) => return Ok(Vec::new()), // pas encore de données
        };

        // Nouveaux chunks depuis le dernier build → on reconstruit l'index BM25.
        if self.fts_dirty.swap(false, Ordering::Relaxed) {
            let _ = self.build_fts_index(&table).await;
        }

        match self.run_fts(&table, q, limit, scope_prefix).await {
            Ok(hits) => Ok(hits),
            Err(_) => {
                // Index probablement absent (1er lancement, table fraîchement créée) :
                // on le construit puis on retente une seule fois.
                self.build_fts_index(&table).await?;
                self.run_fts(&table, q, limit, scope_prefix).await
            }
        }
    }

    /// (Re)construit l'index BM25 sur `text`. `replace(true)` = reconstruction à neuf.
    async fn build_fts_index(&self, table: &lancedb::Table) -> Result<()> {
        table
            .create_index(&["text"], Index::FTS(Default::default()))
            .replace(true)
            .execute()
            .await
            .context("construction de l'index plein-texte (BM25)")?;
        Ok(())
    }

    /// Exécute une requête plein-texte (suppose l'index présent).
    async fn run_fts(
        &self,
        table: &lancedb::Table,
        query: &str,
        limit: usize,
        scope_prefix: Option<&str>,
    ) -> Result<Vec<SearchHit>> {
        let mut q = table
            .query()
            .full_text_search(FullTextSearchQuery::new(query.to_string()))
            .limit(limit);
        if let Some(prefix) = scope_prefix {
            q = q.only_if(scope_like_filter(prefix));
        }
        let batches: Vec<RecordBatch> = q
            .execute()
            .await
            .context("exécution de la recherche plein-texte")?
            .try_collect()
            .await
            .context("collecte des résultats BM25")?;

        let mut hits = Vec::new();
        for batch in batches {
            let paths = column_as_string(&batch, "path")?;
            let texts = column_as_string(&batch, "text")?;
            let indices = batch
                .column_by_name("chunk_index")
                .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
            // LanceDB expose le score BM25 dans la colonne `_score` (plus haut = mieux).
            let scores = batch
                .column_by_name("_score")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());
            for i in 0..batch.num_rows() {
                let full = texts.value(i);
                hits.push(SearchHit {
                    path: paths.value(i).to_string(),
                    chunk_index: indices.map(|idx| idx.value(i)).unwrap_or(0),
                    score: scores.map(|s| s.value(i)).unwrap_or(0.0),
                    text: full.to_string(),
                    snippet: snippet(full),
                });
            }
        }
        Ok(hits)
    }

    /// Vide entièrement la base vectorielle (réindexation complète).
    pub async fn clear(&self) -> Result<()> {
        // drop_table échoue si la table n'existe pas encore : on ignore ce cas.
        self.conn.drop_table(TABLE, &[]).await.ok();
        Ok(())
    }

    pub async fn delete_by_path(&self, path: &str) -> Result<()> {
        if let Ok(table) = self.conn.open_table(TABLE).execute().await {
            table.delete(&path_filter(path)).await.ok();
        }
        Ok(())
    }

    /// Supprime tous les vecteurs des fichiers situés SOUS un dossier (bascule en bloc).
    pub async fn delete_under(&self, folder: &str) -> Result<()> {
        if let Ok(table) = self.conn.open_table(TABLE).execute().await {
            let trimmed = folder.trim_end_matches(['/', '\\']);
            let with_sep = format!("{trimmed}{}", std::path::MAIN_SEPARATOR);
            let pattern = like_escape_datafusion(&with_sep).replace('\'', "''");
            table.delete(&format!("path LIKE '{pattern}%'")).await.ok();
        }
        Ok(())
    }

    /// Met à jour le chemin d'un fichier déplacé, SANS ré-embedding (Index Sync).
    pub async fn rename_path(&self, old_path: &str, new_path: &str) -> Result<()> {
        let table = match self.conn.open_table(TABLE).execute().await {
            Ok(t) => t,
            Err(_) => return Ok(()),
        };
        let escaped_new = new_path.replace('\'', "''");
        table
            .update()
            .only_if(path_filter(old_path))
            .column("path", format!("'{escaped_new}'"))
            .execute()
            .await
            .context("mise à jour du chemin dans LanceDB")?;
        Ok(())
    }
}

fn path_filter(path: &str) -> String {
    let escaped = path.replace('\'', "''");
    format!("path = '{escaped}'")
}

/// Construit un filtre SQL `path LIKE '<dossier>%'` restreignant aux descendants d'un
/// dossier. Attention : dans DataFusion (moteur de filtre de LanceDB) l'antislash est
/// le caractère d'échappement par défaut de LIKE — il faut donc échapper `\`, `%` et
/// `_` des chemins Windows, sinon `C:\...` ne matche jamais.
fn scope_like_filter(prefix: &str) -> String {
    let trimmed = prefix.trim_end_matches(['/', '\\']);
    let with_sep = format!("{trimmed}{}", std::path::MAIN_SEPARATOR);
    let pattern = like_escape_datafusion(&with_sep).replace('\'', "''");
    format!("path LIKE '{pattern}%'")
}

/// Échappe les métacaractères LIKE de DataFusion (`\`, `%`, `_`) en les préfixant
/// de l'antislash (caractère d'échappement par défaut de DataFusion).
fn like_escape_datafusion(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + 8);
    for ch in input.chars() {
        if matches!(ch, '\\' | '%' | '_') {
            out.push('\\');
        }
        out.push(ch);
    }
    out
}

fn column_as_string<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| anyhow!("colonne '{name}' absente ou de type inattendu"))
}

fn snippet(text: &str) -> String {
    let cleaned = text.replace('\n', " ");
    let cleaned = cleaned.trim();
    if cleaned.chars().count() > 240 {
        let truncated: String = cleaned.chars().take(240).collect();
        format!("{truncated}…")
    } else {
        cleaned.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::like_escape_datafusion;

    #[test]
    fn echappe_les_metacaracteres_like() {
        // On évite d'écrire des backslashes littéraux ici : char::from(92) = '\'.
        let bs = char::from(92);
        // Un antislash est doublé.
        assert_eq!(
            like_escape_datafusion(&format!("x{bs}y")),
            format!("x{bs}{bs}y")
        );
        // % et _ (métacaractères LIKE) sont préfixés d'un antislash.
        assert_eq!(like_escape_datafusion("a_b%c"), format!("a{bs}_b{bs}%c"));
        // Sans métacaractère : inchangé.
        assert_eq!(like_escape_datafusion("simple"), "simple");
    }
}

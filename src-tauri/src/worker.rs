//! Worker d'indexation (tâche de fond asynchrone).
//!
//! Pipeline complet par fichier :
//!   extraction → hash SHA-256 → (skip si inchangé) → chunking →
//!   embedding (provider configuré) → stockage LanceDB → statut `completed`.
//!
//! Trois voies d'« extraction du sens » :
//!   * Textuelle  : PDF / DOCX / texte → contenu réel.
//!   * Visuelle   : images → modèle de vision (légende) si activé.
//!   * Contextuelle : fichiers illisibles (VM, binaires) → nom + dossier + type.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};

use base64::Engine;
use sha2::{Digest, Sha256};
use zip::ZipArchive;

use crate::chunker::Chunker;
use crate::parser::{FileType, Parser};
use crate::providers::EmbeddingProvider;
use crate::state::AppState;
use crate::vectordb::ChunkVector;

const MAX_RETRIES: i64 = 3;
const BATCH: i64 = 8;

pub fn start_worker(state: Arc<AppState>) {
    tauri::async_runtime::spawn(async move {
        tracing::info!("👷 Worker d'indexation sémantique démarré.");
        worker_loop(state).await;
    });
}

async fn worker_loop(state: Arc<AppState>) {
    // (ONNX Runtime est préparé paresseusement lors de la première construction
    //  de l'embedder — voir AiEngine::embedder — pour garantir l'ordre d'init.)
    loop {
        let tasks = match state.db.get_pending_extraction_tasks(BATCH) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!("worker: lecture de la file échouée: {e}");
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
        };

        if tasks.is_empty() {
            tokio::time::sleep(Duration::from_secs(3)).await;
            continue;
        }

        // Le provider d'embedding est résolu une fois par lot (modèle mis en cache).
        let embedder = match state.ai.embedder().await {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("worker: moteur d'embedding indisponible ({e}), nouvelle tentative dans 15s");
                tokio::time::sleep(Duration::from_secs(15)).await;
                continue;
            }
        };

        for task in tasks {
            match process_task(&state, embedder.as_ref(), &task.path).await {
                Ok(()) => {
                    let _ = state.db.update_task_status(task.id, "completed");
                }
                Err(e) => {
                    tracing::warn!("worker: échec sur {} : {e}", task.path);
                    let _ = state
                        .db
                        .record_task_failure(task.id, &e.to_string(), MAX_RETRIES);
                }
            }
        }
    }
}

async fn process_task(
    state: &AppState,
    embedder: &dyn EmbeddingProvider,
    path: &str,
) -> anyhow::Result<()> {
    let p = Path::new(path);

    // Fichier disparu entre la mise en file et le traitement : on purge.
    if !p.exists() {
        let _ = state.db.remove_from_queue(path);
        state.vector.delete_by_path(path).await.ok();
        let _ = state.db.remove_catalog_path(path);
        return Ok(());
    }

    let mtime = modified_epoch(p);

    // Un dossier dans la file = dossier traité comme « bloc sémantique » unique.
    if p.is_dir() {
        return index_folder_block(state, embedder, path, mtime).await;
    }

    // Garde-fou : un fichier dont le dossier parent est un bloc ne doit JAMAIS être
    // indexé individuellement (utile pour les tâches restées en file après une
    // bascule manuelle du dossier en mode bloc).
    if let Some(parent) = p.parent() {
        if let Ok(Some((mode, _))) = state.db.get_folder_mode(&parent.to_string_lossy()) {
            if mode == "block" {
                let _ = state.db.remove_from_queue(path);
                return Ok(());
            }
        }
    }

    let file_type = Parser::determine_file_type(p);

    match file_type {
        FileType::Ignored => Ok(()),
        FileType::Image => index_image(state, embedder, path, mtime).await,
        FileType::RequiresAIRouting => index_context_only(state, embedder, path, mtime, "binaire").await,
        FileType::Text | FileType::Document => index_textual(state, embedder, path, mtime).await,
    }
}

/// Indexation d'un fichier textuel (contenu réel).
async fn index_textual(
    state: &AppState,
    embedder: &dyn EmbeddingProvider,
    path: &str,
    mtime: i64,
) -> anyhow::Result<()> {
    let max_bytes = state.config.snapshot().indexing.max_file_mb * 1024 * 1024;
    if let Ok(meta) = fs::metadata(path) {
        if meta.len() > max_bytes {
            // Trop volumineux pour une extraction complète : on se rabat sur le contexte.
            return index_context_only(state, embedder, path, mtime, "volumineux").await;
        }
    }

    let path_owned = path.to_string();
    // Extraction + hash exécutés hors du runtime async (CPU/IO bloquants).
    let (content, hash) = tokio::task::spawn_blocking(move || -> anyhow::Result<(String, String)> {
        let content = extract_text(&path_owned)?;
        let hash = hash_file(&path_owned)?;
        Ok((content, hash))
    })
    .await
    .map_err(|e| anyhow::anyhow!("extraction interrompue: {e}"))??;

    // Contenu inchangé depuis la dernière indexation : on ne ré-embedde pas.
    if let Ok(Some(stored)) = state.db.get_stored_hash(path) {
        if stored == hash {
            tracing::debug!("contenu inchangé, embedding ignoré: {path}");
            let _ = state.db.mark_indexed(path, mtime);
            return Ok(());
        }
    }

    let cfg = state.config.snapshot();
    let mut effective = content.trim().to_string();

    // PDF scanné (aucun texte extractible) + vision activée → OCR des images embarquées.
    if effective.is_empty() && doc_type_of(path) == "pdf" && cfg.vision.enabled {
        if let Some(ocr) = ocr_pdf_via_vision(state, path).await {
            effective = ocr;
        }
    }

    if effective.trim().is_empty() {
        // Document vide/opaque : reste trouvable par son contexte.
        return index_context_only(state, embedder, path, mtime, "vide").await;
    }

    let chunks = Chunker::slice_text(&effective, cfg.indexing.chunk_size, cfg.indexing.overlap);
    if chunks.is_empty() {
        return index_context_only(state, embedder, path, mtime, "vide").await;
    }

    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let vectors = embedder.embed_documents(texts).await?;

    let chunk_vectors: Vec<ChunkVector> = chunks
        .into_iter()
        .zip(vectors.into_iter())
        .map(|(c, v)| ChunkVector {
            chunk_index: c.chunk_index as i32,
            text: c.text,
            vector: v,
        })
        .collect();

    state
        .vector
        .upsert_chunks(path, &hash, mtime, chunk_vectors)
        .await?;

    let _ = state.db.update_file_hash(path, &hash);
    let _ = state
        .db
        .upsert_file_summary(path, &summary_of(&effective), &doc_type_of(path));
    let _ = state.db.mark_indexed(path, mtime);

    tracing::info!("✅ indexé ({} car.) : {path}", effective.len());
    Ok(())
}

/// Indexation d'une image via un modèle de vision (ou repli contextuel).
async fn index_image(
    state: &AppState,
    embedder: &dyn EmbeddingProvider,
    path: &str,
    mtime: i64,
) -> anyhow::Result<()> {
    let cfg = state.config.snapshot();
    if !cfg.vision.enabled {
        return index_context_only(state, embedder, path, mtime, "image").await;
    }

    // Les modèles de vision ne gèrent que les formats raster courants ; on évite
    // d'envoyer .ico/.cur/.svg (400 « invalid image input ») et on les indexe par contexte.
    const VISION_FORMATS: &[&str] = &["jpg", "jpeg", "png", "webp", "gif", "bmp"];
    let ext = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    if !VISION_FORMATS.contains(&ext.as_str()) {
        return index_context_only(state, embedder, path, mtime, "image").await;
    }

    // Lecture + encodage base64 hors runtime async.
    let path_owned = path.to_string();
    let encoded = tokio::task::spawn_blocking(move || -> anyhow::Result<(String, String)> {
        let bytes = fs::read(&path_owned)?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let hash = format!("{:x}", Sha256::digest(&bytes));
        Ok((b64, hash))
    })
    .await
    .map_err(|e| anyhow::anyhow!("lecture image interrompue: {e}"))??;
    let (b64, hash) = encoded;

    let mime = infer::get_from_path(path)
        .ok()
        .flatten()
        .map(|k| k.mime_type().to_string())
        .unwrap_or_else(|| "image/png".to_string());

    let prompt = "Décris le contenu de cette image en une à deux phrases, \
        en identifiant les objets, le texte visible et le thème, \
        pour faciliter son classement dans une arborescence de fichiers.";

    let caption = match state.ai.vision_client().describe_image(&b64, &mime, prompt).await {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("vision indisponible pour {path} ({e}), repli contextuel");
            return index_context_only(state, embedder, path, mtime, "image").await;
        }
    };

    let semantic_text = format!("{} | {}", caption.trim(), context_descriptor(path, "image"));
    let vector = embedder.embed_documents(vec![semantic_text.clone()]).await?;
    let chunk = ChunkVector {
        chunk_index: 0,
        text: semantic_text.clone(),
        vector: vector.into_iter().next().unwrap_or_default(),
    };
    state.vector.upsert_chunks(path, &hash, mtime, vec![chunk]).await?;
    let _ = state.db.update_file_hash(path, &hash);
    let _ = state.db.upsert_file_summary(path, &summary_of(&caption), "image");
    let _ = state.db.mark_indexed(path, mtime);

    tracing::info!("🖼️ image décrite et indexée : {path}");
    Ok(())
}

/// Indexation par le contexte seul : nom, dossier parent, extension, taille.
/// C'est le filet de sécurité pour tout fichier dont on ne peut lire le contenu.
async fn index_context_only(
    state: &AppState,
    embedder: &dyn EmbeddingProvider,
    path: &str,
    mtime: i64,
    kind: &str,
) -> anyhow::Result<()> {
    let text = context_descriptor(path, kind);
    let hash = format!("ctx:{:x}", Sha256::digest(format!("{path}:{mtime}").as_bytes()));

    let vector = embedder.embed_documents(vec![text.clone()]).await?;
    let chunk = ChunkVector {
        chunk_index: 0,
        text: text.clone(),
        vector: vector.into_iter().next().unwrap_or_default(),
    };
    state.vector.upsert_chunks(path, &hash, mtime, vec![chunk]).await?;
    let _ = state.db.update_file_hash(path, &hash);
    let _ = state.db.upsert_file_summary(path, &text, kind);
    let _ = state.db.mark_indexed(path, mtime);

    tracing::info!("🧩 indexé par contexte ({kind}) : {path}");
    Ok(())
}

/// Indexe un dossier comme une seule unité sémantique (nom + contenu résumé),
/// sans descendre dedans. Rend le dossier trouvable en recherche (« mes
/// instruments Ableton ») sans polluer l'index avec chacun de ses fichiers.
async fn index_folder_block(
    state: &AppState,
    embedder: &dyn EmbeddingProvider,
    path: &str,
    mtime: i64,
) -> anyhow::Result<()> {
    let p = Path::new(path);
    let entries = crate::folders::read_dir_sample(p, 100);
    let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let parent = p.parent().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let count = entries.len();

    let mut ext_counts: HashMap<String, usize> = HashMap::new();
    for e in &entries {
        if let Some(ext) = &e.ext {
            *ext_counts.entry(ext.clone()).or_insert(0) += 1;
        }
    }
    let mut exts: Vec<(String, usize)> = ext_counts.into_iter().collect();
    exts.sort_by(|a, b| b.1.cmp(&a.1));
    let top_exts = exts.iter().take(5).map(|(e, _)| e.clone()).collect::<Vec<_>>().join(", ");
    let sample = entries.iter().take(30).map(|e| e.name.clone()).collect::<Vec<_>>().join(", ");

    let text = format!(
        "Dossier (bloc): {name}. Emplacement: {parent}. {count} éléments. Types: {top_exts}. Exemples: {sample}."
    );
    let hash = format!("block:{:x}", Sha256::digest(format!("{path}:{mtime}:{count}").as_bytes()));

    let vector = embedder.embed_documents(vec![text.clone()]).await?;
    let chunk = ChunkVector {
        chunk_index: 0,
        text: text.clone(),
        vector: vector.into_iter().next().unwrap_or_default(),
    };
    state.vector.upsert_chunks(path, &hash, mtime, vec![chunk]).await?;
    let _ = state.db.update_file_hash(path, &hash);
    let _ = state.db.upsert_file_summary(path, &text, "folder-block");
    let _ = state.db.mark_indexed(path, mtime);

    tracing::info!("📦 dossier indexé en bloc : {path}");
    Ok(())
}

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

/// Construit une description textuelle à partir du chemin (contexte pur).
fn context_descriptor(path: &str, kind: &str) -> String {
    let p = Path::new(path);
    let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let parent = p
        .parent()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let ext = p.extension().map(|e| e.to_string_lossy().to_string()).unwrap_or_default();
    format!(
        "Fichier: {name}. Dossier: {parent}. Type: {kind}. Extension: {ext}."
    )
}

fn summary_of(content: &str) -> String {
    let clean: String = content.replace('\n', " ");
    let clean = clean.trim();
    clean.chars().take(300).collect()
}

fn doc_type_of(path: &str) -> String {
    Path::new(path)
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_else(|| "text".to_string())
}

fn modified_epoch(p: &Path) -> i64 {
    fs::metadata(p)
        .and_then(|m| m.modified())
        .map(|t| {
            t.duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64
        })
        .unwrap_or(0)
}

/// Hash SHA-256 du contenu complet du fichier.
fn hash_file(path: &str) -> anyhow::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 16 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Extraction de texte multi-formats (sync).
fn extract_text(path: &str) -> anyhow::Result<String> {
    let ext = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "pdf" => extract_pdf_text(path),
        "docx" => extract_docx_text(path),
        _ => Ok(fs::read_to_string(path).unwrap_or_default()),
    }
}

/// Extrait les images JPEG embarquées d'un PDF (cas des PDF scannés).
/// On ne gère que le filtre DCTDecode : le flux brut EST un JPEG valide.
fn extract_pdf_images(path: &str, max: usize) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let doc = match lopdf::Document::load(path) {
        Ok(d) => d,
        Err(_) => return out,
    };
    for (_id, obj) in &doc.objects {
        if out.len() >= max {
            break;
        }
        let lopdf::Object::Stream(stream) = obj else { continue };
        let dict = &stream.dict;
        // Objet image ?
        let is_image = dict
            .get(b"Subtype")
            .ok()
            .and_then(|o| o.as_name().ok())
            .map(|n| n == b"Image")
            .unwrap_or(false);
        if !is_image {
            continue;
        }
        // Filtre JPEG (DCTDecode), éventuellement dans un tableau de filtres ?
        let is_jpeg = match dict.get(b"Filter").ok() {
            Some(lopdf::Object::Name(n)) => n == b"DCTDecode",
            Some(lopdf::Object::Array(a)) => a
                .iter()
                .any(|o| o.as_name().map(|n| n == b"DCTDecode").unwrap_or(false)),
            _ => false,
        };
        if is_jpeg && !stream.content.is_empty() {
            out.push(stream.content.clone());
        }
    }
    out
}

/// OCR d'un PDF scanné : envoie ses images embarquées au modèle de vision.
async fn ocr_pdf_via_vision(state: &AppState, path: &str) -> Option<String> {
    let path_owned = path.to_string();
    let images = tokio::task::spawn_blocking(move || extract_pdf_images(&path_owned, 8))
        .await
        .ok()?;
    if images.is_empty() {
        return None;
    }

    let prompt = "Transcris fidèlement TOUT le texte visible dans cette image (OCR). \
        Ne renvoie que le texte transcrit, sans commentaire.";
    let client = state.ai.vision_client();
    let mut pages = Vec::new();
    for img in images.iter().take(8) {
        let b64 = base64::engine::general_purpose::STANDARD.encode(img);
        match client.describe_image(&b64, "image/jpeg", prompt).await {
            Ok(t) if !t.trim().is_empty() => pages.push(t),
            Ok(_) => {}
            Err(e) => {
                tracing::warn!("OCR vision échoué ({path}): {e}");
                break; // vision indisponible : inutile d'insister
            }
        }
    }
    let joined = pages.join("\n\n");
    if joined.trim().is_empty() {
        None
    } else {
        tracing::info!("🔎 OCR vision : {} page(s) transcrite(s) pour {path}", pages.len());
        Some(joined)
    }
}

fn extract_pdf_text(path: &str) -> anyhow::Result<String> {
    // 1) Extraction standard.
    if let Ok(text) = pdf_extract::extract_text(path) {
        let t = text.trim();
        if t.len() >= 20 {
            return Ok(t.to_string());
        }
    }
    // 2) PDF chiffré avec mot de passe utilisateur vide (restrictions de copie) :
    //    fréquent sur les documents officiels. On tente le déchiffrement.
    if let Ok(text) = pdf_extract::extract_text_encrypted(path, "") {
        let t = text.trim();
        if !t.is_empty() {
            tracing::debug!("PDF déchiffré (mot de passe vide) : {path}");
            return Ok(t.to_string());
        }
    }
    // 3) PDF scanné/opaque (images) : chaîne vide → repli contextuel (OCR à venir).
    Ok(String::new())
}

fn extract_docx_text(path: &str) -> anyhow::Result<String> {
    let file = File::open(path)?;
    let mut archive = ZipArchive::new(file)?;
    let mut xml = String::new();
    archive
        .by_name("word/document.xml")?
        .read_to_string(&mut xml)?;

    let mut text = String::new();
    let mut in_tag = false;
    for c in xml.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => text.push(c),
            _ => {}
        }
    }
    Ok(text)
}

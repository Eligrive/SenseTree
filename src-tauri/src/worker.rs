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
use crate::providers::{ChatMessage, EmbeddingProvider, VisionError};
use crate::state::AppState;
use crate::vectordb::ChunkVector;

const MAX_RETRIES: i64 = 3;
const BATCH: i64 = 8;
/// Nombre de cycles d'inactivité (3 s chacun) avant de décharger l'embedder.
const IDLE_UNLOAD_CYCLES: u32 = 5;

pub fn start_worker(state: Arc<AppState>) {
    tauri::async_runtime::spawn(async move {
        tracing::info!("👷 Worker d'indexation sémantique démarré.");
        worker_loop(state).await;
    });
}

async fn worker_loop(state: Arc<AppState>) {
    // (ONNX Runtime est préparé paresseusement lors de la première construction
    //  de l'embedder — voir AiEngine::embedder — pour garantir l'ordre d'init.)
    //
    // Déchargement automatique : fastembed/ONNX Runtime maintient un pool de
    // threads intra-op qui « spinne » (consomme du CPU) tant que la session existe,
    // même à l'arrêt. On décharge donc le modèle en pause ou après une période
    // d'inactivité, et il se recharge à la demande (indexation ou recherche).
    let mut idle_cycles: u32 = 0;
    loop {
        // Pause utilisateur : on met la boucle en veille et on libère l'embedder.
        if state.paused.load(std::sync::atomic::Ordering::Relaxed) {
            if state.ai.embedder_loaded().await {
                state.ai.invalidate_embedder().await;
                tracing::info!("⏸️ indexation en pause — embedder déchargé (CPU/GPU libérés)");
            }
            tokio::time::sleep(Duration::from_millis(700)).await;
            continue;
        }

        let tasks = match state.db.get_pending_extraction_tasks(BATCH) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!("worker: lecture de la file échouée: {e}");
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
        };

        if tasks.is_empty() {
            idle_cycles = idle_cycles.saturating_add(1);
            // Après ~15 s sans travail, on décharge le modèle pour stopper le spinning ORT.
            if idle_cycles >= IDLE_UNLOAD_CYCLES && state.ai.embedder_loaded().await {
                state.ai.invalidate_embedder().await;
                tracing::info!("💤 indexation à jour — embedder déchargé (CPU/GPU libérés)");
            }
            tokio::time::sleep(Duration::from_secs(3)).await;
            continue;
        }
        idle_cycles = 0;

        // Le provider d'embedding est résolu une fois par lot (modèle mis en cache).
        let embedder = match state.ai.embedder().await {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("worker: moteur d'embedding indisponible ({e}), nouvelle tentative dans 15s");
                tokio::time::sleep(Duration::from_secs(15)).await;
                continue;
            }
        };

        let cfg_route = state.config.snapshot();
        for task in tasks {
            // Publie l'élément en cours + ses étapes de pipeline (affichage temps réel de la file).
            let (routes, kind) = route_for(&cfg_route, &task.path);
            state.set_activity(Some(crate::state::CurrentActivity {
                path: task.path.clone(),
                routes: routes.iter().map(|s| s.to_string()).collect(),
                kind: kind.to_string(),
            }));
            match process_task(&state, embedder.as_ref(), &task.path, task.retry_count).await {
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
            state.set_activity(None);
        }
    }
}

async fn process_task(
    state: &AppState,
    embedder: &dyn EmbeddingProvider,
    path: &str,
    retry_count: i64,
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
        FileType::Image => index_image(state, embedder, path, mtime, retry_count).await,
        FileType::RequiresAIRouting => index_unknown(state, embedder, path, mtime).await,
        FileType::Text | FileType::Document => index_textual(state, embedder, path, mtime).await,
    }
}

/// Étapes (pipeline) qu'un chemin en file va traverser, dans l'ordre : un sous-ensemble
/// ordonné de {`vision`, `reasoning`, `embedding`}. Sert à l'affichage de la file.
/// C'est une PRÉDICTION (ex. un texte très court saute la qualification reasoning ;
/// un PDF non scanné n'utilise pas la vision). `kind` est un libellé court du type.
pub fn route_for(cfg: &crate::config::AppConfig, path: &str) -> (Vec<&'static str>, &'static str) {
    let p = Path::new(path);
    let reasoning = cfg.reasoning.enabled;
    let mut stages: Vec<&'static str> = Vec::new();
    let kind: &'static str;

    if p.is_dir() {
        // Dossier-bloc : description LLM (reasoning) puis embedding.
        kind = "dossier";
        if reasoning {
            stages.push("reasoning");
        }
    } else {
        match Parser::determine_file_type(p) {
            FileType::Image => {
                // Image : légende vision, puis qualification reasoning, puis embedding.
                kind = "image";
                if cfg.vision.enabled {
                    stages.push("vision");
                }
                if reasoning {
                    stages.push("reasoning");
                }
            }
            // Document / texte / inconnu / contexte : qualification/devinette reasoning
            // puis embedding.
            FileType::Text => {
                kind = "texte";
                if reasoning {
                    stages.push("reasoning");
                }
            }
            FileType::Document => {
                kind = "document";
                if reasoning {
                    stages.push("reasoning");
                }
            }
            FileType::RequiresAIRouting => {
                // Type inconnu : passe d'abord par le ROUTAGE (heuristique + décision LLM
                // sur ce qu'on peut en extraire), puis qualification reasoning, puis embedding.
                kind = "inconnu";
                stages.push("routing");
                if reasoning {
                    stages.push("reasoning");
                }
            }
            FileType::Ignored => {
                kind = "fichier";
            }
        }
    }
    // Toute indexation finit par produire un vecteur.
    stages.push("embedding");
    (stages, kind)
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
    let mut doc_type = doc_type_of(path);

    // PDF scanné (aucun texte extractible) + vision activée → OCR des images embarquées.
    if effective.is_empty() && doc_type == "pdf" && cfg.vision.enabled {
        if let Some(ocr) = ocr_pdf_via_vision(state, path).await {
            effective = ocr;
            doc_type = "pdf-ocr".to_string();
        }
    }

    if effective.trim().is_empty() {
        // Document vide/opaque : reste trouvable par son contexte.
        return index_context_only(state, embedder, path, mtime, "vide").await;
    }

    store_text_document(state, embedder, path, mtime, &effective, &hash, &doc_type).await
}

/// Stocke un document textuel : chunk → embed → LanceDB → métadonnées.
async fn store_text_document(
    state: &AppState,
    embedder: &dyn EmbeddingProvider,
    path: &str,
    mtime: i64,
    text: &str,
    hash: &str,
    doc_type: &str,
) -> anyhow::Result<()> {
    let cfg = state.config.snapshot();
    let chunks = Chunker::slice_text(text, cfg.indexing.chunk_size, cfg.indexing.overlap);
    if chunks.is_empty() {
        return index_context_only(state, embedder, path, mtime, "vide").await;
    }

    // « Sens » du document : une VRAIE qualification par le LLM (CE QUE C'EST + points-clés),
    // pas un simple extrait. Conditionnée au toggle documents ; repli sur un extrait si off,
    // reasoning indisponible, ou texte trop court.
    let summary =
        qualify_or_excerpt(state, &cfg, path, doc_type, text, cfg.indexing.qualify_documents).await;

    // On préfixe le sens au 1er chunk pour qu'il soit AUSSI retrouvable par la recherche
    // (ex. « carte d'identité » devient cherchable même si le mot n'est pas dans l'OCR).
    let mut texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    if let Some(first) = texts.first_mut() {
        *first = format!("[{doc_type}] {summary}\n\n{first}");
    }

    let vectors = embedder.embed_documents(texts.clone()).await?;
    let chunk_vectors: Vec<ChunkVector> = chunks
        .into_iter()
        .zip(texts.into_iter().zip(vectors.into_iter()))
        .map(|(c, (t, v))| ChunkVector { chunk_index: c.chunk_index as i32, text: t, vector: v })
        .collect();
    state.vector.upsert_chunks(path, hash, mtime, chunk_vectors).await?;
    let _ = state.db.update_file_hash(path, hash);
    // On garde LES DEUX : la qualification (« sens ») ET le contenu extrait (borné), pour
    // pouvoir les afficher côte à côte et comparer au document.
    let extract: String = text.trim().chars().take(16_000).collect();
    let _ = state
        .db
        .upsert_file_semantics(path, &summary, Some(&extract), doc_type);
    let _ = state.db.mark_indexed(path, mtime);
    tracing::info!("✅ indexé ({} car., {doc_type}) : {path}", text.len());
    Ok(())
}

/// Renvoie le « sens » d'un document : qualification LLM si autorisée et possible,
/// sinon un simple extrait. `allow` = toggle de qualification pour ce type de contenu.
async fn qualify_or_excerpt(
    state: &AppState,
    cfg: &crate::config::AppConfig,
    path: &str,
    doc_type: &str,
    text: &str,
    allow: bool,
) -> String {
    let trimmed = text.trim();
    // Un extrait trop court n'a pas besoin d'un appel LLM (et n'apporterait rien).
    if allow && cfg.reasoning.enabled && trimmed.chars().count() >= 120 {
        if let Some(q) = llm_qualify_document(state, cfg, path, doc_type, trimmed).await {
            return q;
        }
    }
    summary_of(text)
}

/// Demande au modèle de reasoning de qualifier un document (nature + informations-clés).
pub async fn llm_qualify_document(
    state: &AppState,
    cfg: &crate::config::AppConfig,
    path: &str,
    doc_type: &str,
    text: &str,
) -> Option<String> {
    let p = Path::new(path);
    let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let parent = p.parent().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let excerpt: String = text.chars().take(4000).collect();

    let system = crate::config::prompt_or(
        &cfg.prompts.doc_qualify,
        crate::config::default_prompts::DOC_QUALIFY,
    );
    let user = format!("Fichier: {name}\nDossier: {parent}\nType: {doc_type}\n\nContenu:\n{excerpt}");

    let resp = state
        .ai
        .reasoning_client()
        .chat(
            vec![
                ChatMessage { role: "system".into(), content: system.into() },
                ChatMessage { role: "user".into(), content: user },
            ],
            false,
        )
        .await
        .ok()?;

    let trimmed = resp.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.chars().take(1200).collect())
    }
}

/// Fichier au type inconnu : le LLM décide s'il peut en extraire du contenu,
/// sinon repli sur le contexte (nom, dossier, fichiers voisins).
async fn index_unknown(
    state: &AppState,
    embedder: &dyn EmbeddingProvider,
    path: &str,
    mtime: i64,
) -> anyhow::Result<()> {
    let cfg = state.config.snapshot();
    let max_bytes = cfg.indexing.max_file_mb * 1024 * 1024;
    let path_owned = path.to_string();

    let (sample, hash, too_big) =
        tokio::task::spawn_blocking(move || -> anyhow::Result<(Vec<u8>, String, bool)> {
            let too_big = fs::metadata(&path_owned).map(|m| m.len() > max_bytes).unwrap_or(false);
            let mut f = File::open(&path_owned)?;
            let mut buf = vec![0u8; 32 * 1024];
            let n = f.read(&mut buf)?;
            buf.truncate(n);
            let hash = hash_file(&path_owned)?;
            Ok((buf, hash, too_big))
        })
        .await
        .map_err(|e| anyhow::anyhow!("lecture interrompue: {e}"))??;

    // Inchangé → on ne refait rien.
    if let Ok(Some(stored)) = state.db.get_stored_hash(path) {
        if stored == hash {
            let _ = state.db.mark_indexed(path, mtime);
            return Ok(());
        }
    }

    let ratio = text_ratio(&sample);

    // 1) Clairement textuel → extraction directe.
    if !too_big && ratio >= 0.85 {
        let path_owned = path.to_string();
        let full = tokio::task::spawn_blocking(move || fs::read_to_string(&path_owned).unwrap_or_default())
            .await
            .unwrap_or_default();
        if !full.trim().is_empty() {
            return store_text_document(state, embedder, path, mtime, full.trim(), &hash, "texte").await;
        }
    }

    // 2) Ambigu → on demande au LLM s'il peut en extraire du sens.
    if ratio >= 0.30 && cfg.reasoning.enabled {
        let sample_text = String::from_utf8_lossy(&sample);
        if let Some(extracted) = llm_try_extract(state, path, &sample_text).await {
            return store_text_document(state, embedder, path, mtime, &extracted, &hash, "llm-extrait")
                .await;
        }
    }

    // 3) Repli : contexte enrichi (nom, dossier, fichiers voisins).
    index_context_only(state, embedder, path, mtime, "binaire").await
}

/// Demande au LLM d'extraire le contenu utile d'un extrait de fichier inconnu.
async fn llm_try_extract(state: &AppState, path: &str, sample: &str) -> Option<String> {
    let p = Path::new(path);
    let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let parent = p.parent().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let excerpt: String = sample.chars().take(4000).collect();

    let cfg = state.config.snapshot();
    let system = crate::config::prompt_or(
        &cfg.prompts.file_extract,
        crate::config::default_prompts::FILE_EXTRACT,
    );
    let user = format!("Fichier: {name}\nDossier: {parent}\n\nExtrait:\n{excerpt}");

    let resp = state
        .ai
        .reasoning_client()
        .chat(
            vec![
                ChatMessage { role: "system".into(), content: system.into() },
                ChatMessage { role: "user".into(), content: user },
            ],
            false,
        )
        .await
        .ok()?;

    let trimmed = resp.trim();
    if trimmed.is_empty() || trimmed.contains("NO_CONTENT") {
        None
    } else {
        tracing::info!("🧠 contenu extrait par le LLM : {path}");
        Some(trimmed.to_string())
    }
}

/// Fraction d'octets « texte » (ASCII imprimable, blancs, ou UTF-8/latin1).
fn text_ratio(bytes: &[u8]) -> f32 {
    if bytes.is_empty() {
        return 0.0;
    }
    let good = bytes
        .iter()
        .filter(|&&b| b == b'\t' || b == b'\n' || b == b'\r' || (0x20..=0x7E).contains(&b) || b >= 0x80)
        .count();
    good as f32 / bytes.len() as f32
}

/// Indexation d'une image via un modèle de vision (ou repli contextuel).
async fn index_image(
    state: &AppState,
    embedder: &dyn EmbeddingProvider,
    path: &str,
    mtime: i64,
    retry_count: i64,
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

    let cfg = state.config.snapshot();
    let prompt = crate::config::prompt_or(
        &cfg.prompts.vision_caption,
        crate::config::default_prompts::VISION_CAPTION,
    );

    let caption = match state.ai.vision_client().describe_image(&b64, &mime, prompt).await {
        Ok(c) => c,
        // Échec définitif (image/format/modèle) : inutile d'insister, repli contextuel.
        Err(VisionError::Permanent(e)) => {
            tracing::warn!("vision impossible pour {path} ({e}), repli contextuel");
            return index_context_only(state, embedder, path, mtime, "image").await;
        }
        // Échec passager (timeout, swap de modèle, serveur occupé) : on renvoie une
        // erreur pour que le worker re-mette l'image en file — tant qu'il reste des
        // tentatives. À la dernière seulement, repli contextuel pour ne pas laisser
        // l'image sans aucun sens plutôt que de l'abandonner (failed_permanent).
        Err(VisionError::Transient(e)) => {
            if retry_count + 1 < MAX_RETRIES {
                return Err(anyhow::anyhow!(
                    "vision indisponible (tentative {}/{}) : {}",
                    retry_count + 1,
                    MAX_RETRIES,
                    e
                ));
            }
            tracing::warn!(
                "vision toujours indisponible pour {path} après {} tentatives ({e}), repli contextuel",
                MAX_RETRIES
            );
            return index_context_only(state, embedder, path, mtime, "image").await;
        }
    };

    // Même logique que les documents : la légende vision est le « contenu extrait »,
    // qu'on fait QUALIFIER par le reasoning pour obtenir le « sens » (CE QUE C'EST).
    // Conditionnée au toggle images ; repli sur la légende si off / reasoning indispo.
    let summary =
        qualify_or_excerpt(state, &cfg, path, "image", &caption, cfg.indexing.qualify_images).await;

    let semantic_text = format!(
        "{summary}\n\n{} | {}",
        caption.trim(),
        context_descriptor(path, "image")
    );
    let vector = embedder.embed_documents(vec![semantic_text.clone()]).await?;
    let chunk = ChunkVector {
        chunk_index: 0,
        text: semantic_text.clone(),
        vector: vector.into_iter().next().unwrap_or_default(),
    };
    state.vector.upsert_chunks(path, &hash, mtime, vec![chunk]).await?;
    let _ = state.db.update_file_hash(path, &hash);
    // On garde les DEUX : la qualification (« sens ») et la légende brute (« contenu extrait »).
    let _ = state
        .db
        .upsert_file_semantics(path, &summary, Some(caption.trim()), "image");
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
    let descriptor = context_descriptor(path, kind);
    let hash = format!("ctx:{:x}", Sha256::digest(format!("{path}:{mtime}").as_bytes()));
    let cfg = state.config.snapshot();

    // Devinette de la nature du fichier par le reasoning (chemin + métadonnées + voisinage).
    // Conditionnée au toggle contexte ; repli sur le descripteur brut si off/reasoning indispo.
    let guess = if cfg.indexing.qualify_context {
        llm_guess_context(state, &cfg, path, &descriptor).await
    } else {
        None
    };
    let (summary, extract) = match guess {
        Some(g) => (g, Some(descriptor.clone())),
        None => (descriptor.clone(), None),
    };
    // On embed la devinette + le contexte factuel (nom, voisinage) pour la recherche.
    let text = match &extract {
        Some(d) => format!("{summary}\n{d}"),
        None => summary.clone(),
    };

    let vector = embedder.embed_documents(vec![text.clone()]).await?;
    let chunk = ChunkVector {
        chunk_index: 0,
        text,
        vector: vector.into_iter().next().unwrap_or_default(),
    };
    state.vector.upsert_chunks(path, &hash, mtime, vec![chunk]).await?;
    let _ = state.db.update_file_hash(path, &hash);
    let _ = state
        .db
        .upsert_file_semantics(path, &summary, extract.as_deref(), kind);
    let _ = state.db.mark_indexed(path, mtime);

    tracing::info!("🧩 indexé par contexte ({kind}) : {path}");
    Ok(())
}

/// Devine la nature d'un fichier illisible (contexte seul) via le reasoning, à partir
/// du chemin, des métadonnées et du voisinage. `None` si reasoning off ou pas de réponse.
async fn llm_guess_context(
    state: &AppState,
    cfg: &crate::config::AppConfig,
    path: &str,
    descriptor: &str,
) -> Option<String> {
    if !cfg.reasoning.enabled {
        return None;
    }
    let size_str = fs::metadata(path)
        .map(|m| format!(" Taille: {} octets.", m.len()))
        .unwrap_or_default();

    let system = crate::config::prompt_or(
        &cfg.prompts.context_guess,
        crate::config::default_prompts::CONTEXT_GUESS,
    );
    let user = format!("{descriptor}{size_str}");

    let resp = state
        .ai
        .reasoning_client()
        .chat(
            vec![
                ChatMessage { role: "system".into(), content: system.into() },
                ChatMessage { role: "user".into(), content: user },
            ],
            false,
        )
        .await
        .ok()?;

    let trimmed = resp.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.chars().take(600).collect())
    }
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
    let sample = entries.iter().take(40).map(|e| e.name.clone()).collect::<Vec<_>>().join(", ");

    let facts = format!(
        "Dossier (bloc): {name}. Emplacement: {parent}. {count} éléments. Types: {top_exts}. Exemples: {sample}."
    );

    // Description LLM du bloc (si reasoning dispo) — donne un vrai sens au dossier.
    let description = if state.config.snapshot().reasoning.enabled {
        llm_describe_folder(state, &name, &parent, &top_exts, &sample).await
    } else {
        None
    };
    let (text, summary) = match &description {
        Some(desc) => (format!("{desc}\n{facts}"), desc.clone()),
        None => (facts.clone(), facts.clone()),
    };

    let hash = format!("block:{:x}", Sha256::digest(format!("{path}:{mtime}:{count}").as_bytes()));

    let vector = embedder.embed_documents(vec![text.clone()]).await?;
    let chunk = ChunkVector {
        chunk_index: 0,
        text,
        vector: vector.into_iter().next().unwrap_or_default(),
    };
    state.vector.upsert_chunks(path, &hash, mtime, vec![chunk]).await?;
    let _ = state.db.update_file_hash(path, &hash);
    // Sens = description LLM (qualification) ; contenu extrait = le listing du dossier
    // (types + exemples), pour comparer. Si pas de description distincte, pas d'extract redondant.
    let extract = description.as_ref().map(|_| facts.as_str());
    let _ = state
        .db
        .upsert_file_semantics(path, &summary, extract, "folder-block");
    let _ = state.db.mark_indexed(path, mtime);

    tracing::info!("📦 dossier indexé en bloc : {path}");
    Ok(())
}

/// Demande au LLM une description courte (1 phrase) de ce qu'est un dossier-bloc.
async fn llm_describe_folder(
    state: &AppState,
    name: &str,
    parent: &str,
    top_exts: &str,
    sample: &str,
) -> Option<String> {
    let cfg = state.config.snapshot();
    let system = crate::config::prompt_or(
        &cfg.prompts.folder_describe,
        crate::config::default_prompts::FOLDER_DESCRIBE,
    );
    let user = format!(
        "Nom: {name}\nEmplacement: {parent}\nTypes de fichiers: {top_exts}\nExemples: {sample}"
    );
    let resp = state
        .ai
        .reasoning_client()
        .chat(
            vec![
                ChatMessage { role: "system".into(), content: system.into() },
                ChatMessage { role: "user".into(), content: user },
            ],
            false,
        )
        .await
        .ok()?;
    let trimmed = resp.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
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

    // Fichiers voisins (contexte de dossier) : aident à situer le sens.
    let siblings: Vec<String> = p
        .parent()
        .and_then(|d| std::fs::read_dir(d).ok())
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| {
                    let n = e.file_name().to_string_lossy().to_string();
                    if n == name || n.starts_with('.') {
                        None
                    } else {
                        Some(n)
                    }
                })
                .take(15)
                .collect()
        })
        .unwrap_or_default();

    if siblings.is_empty() {
        format!("Fichier: {name}. Dossier: {parent}. Type: {kind}. Extension: {ext}.")
    } else {
        format!(
            "Fichier: {name}. Dossier: {parent}. Type: {kind}. Extension: {ext}. Fichiers voisins: {}.",
            siblings.join(", ")
        )
    }
}

pub fn summary_of(content: &str) -> String {
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
    let mut doc = match lopdf::Document::load(path) {
        Ok(d) => d,
        Err(_) => return out,
    };
    // Déchiffre (mot de passe vide) pour que les images embarquées soient lisibles.
    if doc.is_encrypted() {
        let _ = doc.decrypt("");
    }
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

    let cfg = state.config.snapshot();
    let prompt = crate::config::prompt_or(
        &cfg.prompts.vision_ocr,
        crate::config::default_prompts::VISION_OCR,
    );
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
    //    fréquent sur les documents officiels. On tente via pdf_extract.
    if let Ok(text) = pdf_extract::extract_text_encrypted(path, "") {
        let t = text.trim();
        if !t.is_empty() {
            return Ok(t.to_string());
        }
    }
    // 3) Déchiffrement via lopdf (gère plus de schémas), puis ré-extraction en mémoire.
    if let Ok(text) = decrypt_and_extract_pdf(path) {
        let t = text.trim();
        if !t.is_empty() {
            tracing::debug!("PDF déchiffré via lopdf : {path}");
            return Ok(t.to_string());
        }
    }
    // 4) PDF scanné/opaque (images) : chaîne vide → repli OCR/contexte en amont.
    Ok(String::new())
}

/// Déchiffre un PDF (mot de passe vide) via lopdf puis en ré-extrait le texte.
fn decrypt_and_extract_pdf(path: &str) -> anyhow::Result<String> {
    let mut doc = lopdf::Document::load(path)?;
    if doc.is_encrypted() {
        doc.decrypt("")?;
    }
    let mut buf = Vec::new();
    doc.save_to(&mut buf)?;
    Ok(pdf_extract::extract_text_from_mem(&buf).unwrap_or_default())
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

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
use crate::providers::{ChatMessage, EmbeddingProvider, AiCallError};
use crate::state::AppState;
use crate::vectordb::ChunkVector;

const MAX_RETRIES: i64 = 3;
const BATCH: i64 = 8;
/// Nombre de cycles d'inactivité (3 s chacun) avant de décharger l'embedder.
const IDLE_UNLOAD_CYCLES: u32 = 5;

/// Travail restant après les étages IA : ce qu'il faut vectoriser puis stocker.
///
/// Les cinq chemins d'indexation (texte, image, inconnu, contexte, dossier-bloc)
/// finissaient tous par la même queue — embedder, écrire dans LanceDB, mettre à jour
/// les métadonnées. L'isoler ici permet de DÉCALER l'embedding par rapport aux appels
/// LLM, ce qui est exactement ce que fait le mode batch.
pub struct Pending {
    path: String,
    mtime: i64,
    hash: String,
    /// Type stocké dans `file_semantics` (`pdf`, `image`, `folder-block`…).
    kind: String,
    /// Le « sens » du fichier, issu du reasoning ou d'un repli.
    summary: String,
    /// Contenu extrait conservé à côté du sens, pour comparaison.
    extract: Option<String>,
    /// Textes STOCKÉS, un par chunk (snippets, BM25).
    stored: Vec<String>,
    /// Textes ENVOYÉS à l'embedding (contextual retrieval) — même longueur que `stored`.
    embed: Vec<String>,
    /// Message de fin, pour conserver les journaux existants.
    log: String,
}

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

        let cfg_route = state.config.snapshot();
        let batch_mode = cfg_route.indexing.pipeline_mode == crate::config::PipelineMode::Batch;
        // En mode batch, on prélève une tranche plus large : c'est elle qui détermine
        // combien d'appels LLM sont regroupés avant de passer à l'embedding.
        let slice = if batch_mode {
            cfg_route.indexing.batch_files.clamp(1, 1000) as i64
        } else {
            BATCH
        };

        let tasks = match state.db.get_pending_extraction_tasks(slice) {
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

        if batch_mode {
            run_batch(&state, embedder.as_ref(), &cfg_route, tasks).await;
        } else {
            run_sequential(&state, embedder.as_ref(), &cfg_route, tasks).await;
        }
    }
}

/// Publie l'élément en cours + ses étapes (affichage temps réel de la file).
fn announce(state: &AppState, cfg: &crate::config::AppConfig, path: &str) {
    let (routes, kind) = route_for(cfg, path);
    state.set_activity(Some(crate::state::CurrentActivity {
        path: path.to_string(),
        routes: routes.iter().map(|s| s.to_string()).collect(),
        kind: kind.to_string(),
    }));
}

fn fail(state: &AppState, id: i64, path: &str, e: &anyhow::Error) {
    tracing::warn!("worker: échec sur {path} : {e}");
    let _ = state.db.record_task_failure(id, &e.to_string(), MAX_RETRIES);
}

/// Mode SÉQUENTIEL : un fichier est mené de bout en bout avant de passer au suivant.
///
/// L'index avance fichier par fichier — ce qui vient d'être traité est immédiatement
/// cherchable — au prix d'une alternance des modèles à chaque fichier.
async fn run_sequential(
    state: &AppState,
    embedder: &dyn EmbeddingProvider,
    cfg: &crate::config::AppConfig,
    tasks: Vec<crate::db::ExtractionTask>,
) {
    for task in tasks {
        // La pause doit être ressentie tout de suite, pas à la fin du lot. Les tâches
        // non traitées restent `pending` : elles seront reprises telles quelles.
        if state.paused.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
        announce(state, cfg, &task.path);
        let outcome = match prepare_task(state, &task.path, task.retry_count).await {
            // `None` = rien à vectoriser (fichier disparu, inchangé, ignoré).
            Ok(None) => Ok(()),
            Ok(Some(p)) => finalize(state, embedder, p).await,
            Err(e) => Err(e),
        };
        match outcome {
            Ok(()) => {
                let _ = state.db.update_task_status(task.id, "completed");
            }
            Err(e) => fail(state, task.id, &task.path, &e),
        }
        state.set_activity(None);
    }
}

/// Mode BATCH : toute la tranche passe par les étages LLM, PUIS par l'embedding.
///
/// Un seul aller-retour entre les deux moteurs par tranche au lieu d'un par fichier.
/// Sur une machine où les modèles ne tiennent pas ensemble en mémoire, c'est la
/// différence entre recharger un modèle de plusieurs gigaoctets à chaque fichier et
/// le faire une fois toutes les N.
///
/// Un échec pendant la phase LLM est enregistré tout de suite : inutile de faire
/// attendre la tranche entière pour signaler un fichier qui ne passera pas.
async fn run_batch(
    state: &AppState,
    embedder: &dyn EmbeddingProvider,
    cfg: &crate::config::AppConfig,
    tasks: Vec<crate::db::ExtractionTask>,
) {
    let total = tasks.len();
    tracing::info!("🧱 tranche de {total} fichiers : phase extraction + LLM");

    let mut prets: Vec<(i64, Pending)> = Vec::with_capacity(total);
    for task in tasks {
        // Pause demandée : on arrête de préparer, mais on va tout de même vectoriser ce
        // qui est déjà prêt — sinon les appels LLM déjà payés seraient perdus et
        // refaits à la reprise.
        if state.paused.load(std::sync::atomic::Ordering::Relaxed) {
            tracing::info!("⏸️ pause pendant la tranche — on finalise les {} fichiers prêts", prets.len());
            break;
        }
        announce(state, cfg, &task.path);
        match prepare_task(state, &task.path, task.retry_count).await {
            Ok(Some(p)) => prets.push((task.id, p)),
            Ok(None) => {
                let _ = state.db.update_task_status(task.id, "completed");
            }
            Err(e) => fail(state, task.id, &task.path, &e),
        }
    }
    state.set_activity(None);

    if prets.is_empty() {
        return;
    }

    // Les modèles LLM ne serviront plus avant la prochaine tranche : on les libère
    // explicitement pour que l'embedding dispose de la mémoire. Sans effet — et sans
    // erreur — si le serveur n'est pas Ollama.
    liberer_modeles_llm(cfg).await;

    tracing::info!("🧱 tranche : phase embedding sur {} fichiers", prets.len());
    for (id, p) in prets {
        announce(state, cfg, &p.path);
        let path = p.path.clone();
        match finalize(state, embedder, p).await {
            Ok(()) => {
                let _ = state.db.update_task_status(id, "completed");
            }
            Err(e) => fail(state, id, &path, &e),
        }
    }
    state.set_activity(None);
}

/// Décharge les modèles de reasoning et de vision du serveur, s'il s'agit d'Ollama.
///
/// Ollama n'expose aucune API de configuration : on ne peut pas lui imposer combien de
/// modèles il garde en mémoire. En revanche on peut décharger explicitement ceux dont
/// on n'a plus besoin, ce qui rend le mode batch déterministe quelle que soit la
/// configuration du serveur d'en face.
async fn liberer_modeles_llm(cfg: &crate::config::AppConfig) {
    let mut vus: Vec<(&str, &str)> = Vec::new();
    if cfg.reasoning.enabled {
        vus.push((&cfg.reasoning.base_url, &cfg.reasoning.model));
    }
    if cfg.vision.enabled {
        vus.push((&cfg.vision.base_url, &cfg.vision.model));
    }
    for (base, model) in vus {
        if model.is_empty() {
            continue;
        }
        match crate::ollama_server::unload(base, model).await {
            Ok(()) => tracing::debug!("modèle {model} déchargé avant la phase embedding"),
            // Serveur non-Ollama, hors ligne, ou modèle déjà déchargé : sans importance.
            Err(e) => tracing::debug!("déchargement de {model} ignoré ({e})"),
        }
    }
}

/// Vectorise et stocke le travail préparé. C'est la queue commune à tous les chemins.
async fn finalize(
    state: &AppState,
    embedder: &dyn EmbeddingProvider,
    p: Pending,
) -> anyhow::Result<()> {
    let vectors = embedder.embed_documents(p.embed).await?;
    let chunk_vectors: Vec<ChunkVector> = p
        .stored
        .into_iter()
        .zip(vectors)
        .enumerate()
        .map(|(i, (text, vector))| ChunkVector { chunk_index: i as i32, text, vector })
        .collect();

    state.vector.upsert_chunks(&p.path, &p.hash, p.mtime, chunk_vectors).await?;
    let _ = state.db.update_file_hash(&p.path, &p.hash);
    let _ = state
        .db
        .upsert_file_semantics(&p.path, &p.summary, p.extract.as_deref(), &p.kind);
    let _ = state.db.mark_indexed(&p.path, p.mtime);
    tracing::info!("{}", p.log);
    Ok(())
}

/// Mène un fichier jusqu'au bout des étages IA, sans le vectoriser.
///
/// `Ok(None)` signifie « rien à faire » : fichier disparu, inchangé depuis la dernière
/// indexation, ou ignoré. C'est un succès, pas un échec.
async fn prepare_task(
    state: &AppState,
    path: &str,
    retry_count: i64,
) -> anyhow::Result<Option<Pending>> {
    let p = Path::new(path);

    // Fichier disparu entre la mise en file et le traitement : on purge.
    if !p.exists() {
        let _ = state.db.remove_from_queue(path);
        state.vector.delete_by_path(path).await.ok();
        let _ = state.db.remove_catalog_path(path);
        return Ok(None);
    }

    let mtime = modified_epoch(p);

    // Un dossier dans la file = dossier traité comme « bloc sémantique » unique.
    if p.is_dir() {
        return index_folder_block(state, path, mtime).await;
    }

    // Garde-fou : un fichier dont le dossier parent est un bloc ne doit JAMAIS être
    // indexé individuellement (utile pour les tâches restées en file après une
    // bascule manuelle du dossier en mode bloc).
    if let Some(parent) = p.parent() {
        if let Ok(Some((mode, _))) = state.db.get_folder_mode(&parent.to_string_lossy()) {
            if mode == "block" {
                let _ = state.db.remove_from_queue(path);
                return Ok(None);
            }
        }
    }

    let file_type = Parser::determine_file_type(p);

    match file_type {
        FileType::Ignored => Ok(None),
        FileType::Image => index_image(state, path, mtime, retry_count).await,
        FileType::Media => index_media(state, path, mtime, retry_count).await,
        FileType::RequiresAIRouting => index_unknown(state, path, mtime).await,
        FileType::Text | FileType::Document => index_textual(state, path, mtime).await,
    }
}

/// Étapes (pipeline) qu'un chemin en file va traverser, dans l'ordre : un sous-ensemble
/// ordonné de {`vision`, `media`, `reasoning`, `embedding`}. Sert à
/// l'affichage de la file.
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
            FileType::Media => {
                // Média : transcription et/ou description visuelle, puis
                // qualification reasoning, puis embedding.
                kind = "média";
                if cfg.transcription.enabled || cfg.video.enabled {
                    stages.push("media");
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
    path: &str,
    mtime: i64,
) -> anyhow::Result<Option<Pending>> {
    let max_bytes = state.config.snapshot().indexing.max_file_mb * 1024 * 1024;
    if let Ok(meta) = fs::metadata(path) {
        if meta.len() > max_bytes {
            // Trop volumineux pour une extraction complète : on se rabat sur le contexte.
            return index_context_only(state, path, mtime, "volumineux").await;
        }
    }

    let path_owned = path.to_string();
    // Extraction + hash exécutés hors du runtime async (CPU/IO bloquants).
    let extraction = tokio::task::spawn_blocking(move || -> anyhow::Result<(String, String)> {
        let content = extract_text(&path_owned)?;
        let hash = hash_file(&path_owned)?;
        Ok((content, hash))
    })
    .await;

    let (content, hash) = match extraction {
        Ok(r) => r?,
        // PANIQUE dans la bibliothèque d'extraction (PDF à table Unicode malformée,
        // surrogate UTF-16 isolé…). C'est DÉTERMINISTE : réessayer donnera la même
        // panique, en consommant les trois tentatives puis en abandonnant le fichier.
        // On dégrade donc tout de suite en indexation par contexte — le document reste
        // trouvable par son nom et son emplacement, au lieu d'être perdu.
        Err(e) => {
            tracing::warn!("extraction impossible pour {path} ({e}), repli contextuel");
            return index_context_only(state, path, mtime, "illisible").await;
        }
    };

    // Contenu inchangé depuis la dernière indexation : on ne ré-embedde pas.
    if let Ok(Some(stored)) = state.db.get_stored_hash(path) {
        if stored == hash {
            tracing::debug!("contenu inchangé, embedding ignoré: {path}");
            let _ = state.db.mark_indexed(path, mtime);
            return Ok(None);
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
        return index_context_only(state, path, mtime, "vide").await;
    }

    store_text_document(state, path, mtime, &effective, &hash, &doc_type, cfg.indexing.qualify_documents)
        .await
}

/// Prépare un document textuel : chunk → qualification LLM → textes à vectoriser.
async fn store_text_document(
    state: &AppState,
    path: &str,
    mtime: i64,
    text: &str,
    hash: &str,
    doc_type: &str,
    // Toggle de qualification correspondant au type de contenu : `qualify_documents`
    // pour un document, `qualify_media` pour une transcription. Passé par l'appelant
    // plutôt que déduit de `doc_type`, pour que le réglage reste lisible ici.
    qualify: bool,
) -> anyhow::Result<Option<Pending>> {
    let cfg = state.config.snapshot();
    let mut chunks = Chunker::slice_text(text, cfg.indexing.chunk_size, cfg.indexing.overlap);
    if chunks.is_empty() {
        return index_context_only(state, path, mtime, "vide").await;
    }

    // Plafond OPTIONNEL de vecteurs par fichier. Désactivé par défaut (`0`) : la taille
    // d'un fichier ne dit rien de sa valeur sémantique, et tronquer sans le dire ferait
    // perdre le contenu d'un document légitimement volumineux. Qui veut borner le temps
    // passé sur un seul fichier peut l'activer dans les Paramètres.
    let plafond = cfg.indexing.max_chunks_per_file;
    let tronque = plafond > 0 && chunks.len() > plafond;
    if tronque {
        tracing::info!(
            "✂️ {path} : {} chunks ramenés à {plafond} (plafond par fichier)",
            chunks.len()
        );
        chunks.truncate(plafond);
    }

    // « Sens » du document : une VRAIE qualification par le LLM (CE QUE C'EST + points-clés),
    // pas un simple extrait. Conditionnée au toggle documents ; repli sur un extrait si off,
    // reasoning indisponible, ou texte trop court.
    let summary = qualify_or_excerpt(state, &cfg, path, doc_type, text, qualify).await;

    // TEXTE STOCKÉ : brut (snippets/BM25/rerank propres). Le 1er chunk porte en plus la
    // qualification complète, ce qui la rend cherchable AUSSI par mots-clés (BM25) —
    // ex. « carte d'identité » retrouvable même absent de l'OCR.
    let stored_texts: Vec<String> = {
        let mut t: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        if let Some(first) = t.first_mut() {
            *first = format!("[{doc_type}] {summary}\n\n{first}");
        }
        t
    };

    // CONTEXTUAL RETRIEVAL : on situe chaque chunk dans son document (nom + sens) dans
    // le texte EMBEDDÉ (pas stocké). Un chunk isolé « le montant est de 90€ » devient
    // ainsi rattachable à « facture EDF » → dense bien plus robuste. Le 1er chunk
    // contient déjà la qualification, inutile de la redoubler.
    let file_name = std::path::Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let ctx_compact: String = summary.split_whitespace().collect::<Vec<_>>().join(" ");
    let ctx_compact: String = ctx_compact.chars().take(180).collect();
    let embed_texts: Vec<String> = stored_texts
        .iter()
        .enumerate()
        .map(|(i, t)| {
            if i == 0 {
                t.clone()
            } else {
                format!("{file_name} · {ctx_compact}\n\n{t}")
            }
        })
        .collect();

    // On garde LES DEUX : la qualification (« sens ») ET le contenu extrait (borné), pour
    // pouvoir les afficher côte à côte et comparer au document.
    let mut extract: String = text.trim().chars().take(16_000).collect();
    // La troncature doit être VISIBLE : sans ça, l'utilisateur croit le document
    // intégralement indexé alors que seule sa première partie est cherchable.
    if tronque {
        extract.push_str(&format!(
            "\n\n[Indexation partielle : les {plafond} premiers extraits seulement — \
             document trop volumineux. Ajustez « chunks max par fichier » si besoin.]"
        ));
    }
    Ok(Some(Pending {
        path: path.to_string(),
        mtime,
        hash: hash.to_string(),
        kind: doc_type.to_string(),
        summary,
        extract: Some(extract),
        stored: stored_texts,
        embed: embed_texts,
        log: format!("✅ indexé ({} car., {doc_type}) : {path}", text.len()),
    }))
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
    // Un sens ÉPINGLÉ a été corrigé par l'utilisateur : le régénérer reproduirait
    // l'erreur qu'il vient justement de corriger. On le préserve — et le fichier est
    // tout de même ré-embeddé avec, c'est ce qui rend la correction effective dans la
    // RECHERCHE et pas seulement à l'affichage.
    if let Ok(Some(epingle)) = state.db.summary_is_pinned(path) {
        tracing::debug!("sens épinglé conservé pour {path}");
        return epingle;
    }
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
        .chat_quick(
            vec![
                ChatMessage { role: "system".into(), content: system.into() },
                ChatMessage { role: "user".into(), content: user },
            ],
            false,
            state.config.snapshot().indexing.qualify_effort,
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
    path: &str,
    mtime: i64,
) -> anyhow::Result<Option<Pending>> {
    let cfg = state.config.snapshot();
    let max_bytes = cfg.indexing.max_file_mb * 1024 * 1024;
    let path_owned = path.to_string();

    let lecture = tokio::task::spawn_blocking(move || -> anyhow::Result<(Vec<u8>, String, bool)> {
        let too_big = fs::metadata(&path_owned).map(|m| m.len() > max_bytes).unwrap_or(false);
        let mut f = File::open(&path_owned)?;
        let mut buf = vec![0u8; 32 * 1024];
        let n = f.read(&mut buf)?;
        buf.truncate(n);
        let hash = hash_file(&path_owned)?;
        Ok((buf, hash, too_big))
    })
    .await;

    let (sample, hash, too_big) = match lecture {
        Ok(r) => r?,
        // Même politique que pour les documents : une panique est déterministe, on
        // dégrade au lieu de réessayer trois fois puis d'abandonner le fichier.
        Err(e) => {
            tracing::warn!("lecture impossible pour {path} ({e}), repli contextuel");
            return index_context_only(state, path, mtime, "binaire").await;
        }
    };

    // Inchangé → on ne refait rien.
    if let Ok(Some(stored)) = state.db.get_stored_hash(path) {
        if stored == hash {
            let _ = state.db.mark_indexed(path, mtime);
            return Ok(None);
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
            return store_text_document(state, path, mtime, full.trim(), &hash, "texte", cfg.indexing.qualify_documents)
                .await;
        }
    }

    // 2) Ambigu → on demande au LLM s'il peut en extraire du sens.
    if ratio >= 0.30 && cfg.reasoning.enabled {
        let sample_text = String::from_utf8_lossy(&sample);
        if let Some(extracted) = llm_try_extract(state, path, &sample_text).await {
            return store_text_document(
                state, path, mtime, &extracted, &hash, "llm-extrait", cfg.indexing.qualify_documents,
            )
            .await;
        }
    }

    // 3) Repli : contexte enrichi (nom, dossier, fichiers voisins).
    index_context_only(state, path, mtime, "binaire").await
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
        .chat_quick(
            vec![
                ChatMessage { role: "system".into(), content: system.into() },
                ChatMessage { role: "user".into(), content: user },
            ],
            false,
            state.config.snapshot().indexing.qualify_effort,
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
    path: &str,
    mtime: i64,
    retry_count: i64,
) -> anyhow::Result<Option<Pending>> {
    let cfg = state.config.snapshot();
    if !cfg.vision.enabled {
        return index_context_only(state, path, mtime, "image").await;
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
        return index_context_only(state, path, mtime, "image").await;
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
        Err(AiCallError::Permanent(e)) => {
            tracing::warn!("vision impossible pour {path} ({e}), repli contextuel");
            return index_context_only(state, path, mtime, "image").await;
        }
        // Échec passager (timeout, swap de modèle, serveur occupé) : on renvoie une
        // erreur pour que le worker re-mette l'image en file — tant qu'il reste des
        // tentatives. À la dernière seulement, repli contextuel pour ne pas laisser
        // l'image sans aucun sens plutôt que de l'abandonner (failed_permanent).
        Err(AiCallError::Transient(e)) => {
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
            return index_context_only(state, path, mtime, "image").await;
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
    Ok(Some(Pending {
        path: path.to_string(),
        mtime,
        hash,
        kind: "image".to_string(),
        summary,
        // On garde les DEUX : la qualification (« sens ») et la légende brute.
        extract: Some(caption.trim().to_string()),
        stored: vec![semantic_text.clone()],
        embed: vec![semantic_text],
        log: format!("🖼️ image décrite et indexée : {path}"),
    }))
}

/// Extensions de conteneurs vidéo, utilisées UNIQUEMENT en repli quand la
/// détection par magic bytes échoue. Ce n'est pas une liste d'autorisation :
/// elle ne sert qu'à décider s'il vaut la peine de tenter une description
/// visuelle en plus de la transcription.
const EXTENSIONS_VIDEO: &[&str] = &[
    "mp4", "m4v", "mkv", "avi", "mov", "webm", "wmv", "mpeg", "mpg", "3gp", "flv", "ogv", "mts",
    "m2ts", "ts", "vob", "asf", "rm", "rmvb", "divx", "f4v",
];

/// Vrai si le fichier est un conteneur vidéo, d'après son type MIME puis, à
/// défaut, son extension.
fn est_video(mime: &str, ext: &str) -> bool {
    mime.starts_with("video/") || EXTENSIONS_VIDEO.contains(&ext)
}

/// Vrai si la taille dépasse le plafond configuré. `0` = aucun plafond.
fn au_dessus_du_plafond(taille: u64, plafond_mo: u64) -> bool {
    plafond_mo > 0 && taille > plafond_mo * 1024 * 1024
}

/// Indexation d'un média audio/vidéo.
///
/// Deux sources de sens, complémentaires et indépendamment activables :
///   * la **transcription** de la parole (`/audio/transcriptions`) ;
///   * la **description visuelle** de l'image, pour les vidéos, via un modèle
///     multimodal (`/chat/completions` avec une part `video_url`).
///
/// L'app ne présume RIEN du format : tout fichier routé ici est envoyé tel quel,
/// et c'est le serveur configuré qui accepte ou refuse. Un refus est un échec
/// définitif, traité comme tel — le fichier retombe alors sur son contexte au
/// lieu de bloquer l'indexation.
///
/// Le résultat passe par [`store_text_document`] : découpage, qualification, BM25
/// et contextual retrieval. Un enregistrement devient ainsi cherchable sur
/// n'importe lequel de ses passages, et pas seulement sur son résumé.
async fn index_media(
    state: &AppState,
    path: &str,
    mtime: i64,
    retry_count: i64,
) -> anyhow::Result<Option<Pending>> {
    let cfg = state.config.snapshot();
    if !cfg.transcription.enabled && !cfg.video.enabled {
        return index_context_only(state, path, mtime, "média").await;
    }

    // Empreinte calculée EN FLUX (`hash_file` lit par blocs de 16 Ko) : un média de
    // plusieurs Go ne passe jamais entièrement en mémoire.
    let path_owned = path.to_string();
    let hash = match tokio::task::spawn_blocking(move || hash_file(&path_owned)).await {
        Ok(Ok(h)) => h,
        Ok(Err(e)) => {
            tracing::warn!("lecture impossible pour {path} ({e}), repli contextuel");
            return index_context_only(state, path, mtime, "média").await;
        }
        Err(e) => {
            tracing::warn!("hachage interrompu pour {path} ({e}), repli contextuel");
            return index_context_only(state, path, mtime, "média").await;
        }
    };

    // Média inchangé : transcrire est l'appel le plus coûteux de l'indexation, on ne
    // le refait pas pour un contenu identique.
    if let Ok(Some(stored)) = state.db.get_stored_hash(path) {
        if stored == hash {
            tracing::debug!("média inchangé, traitement ignoré : {path}");
            let _ = state.db.mark_indexed(path, mtime);
            return Ok(None);
        }
    }

    let taille = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let ext = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let file_name = Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "media".to_string());
    // Le serveur se sert du type MIME pour choisir son démuxeur. On le devine, sans
    // en faire une condition : un type inconnu part quand même.
    let mime = infer::get_from_path(path)
        .ok()
        .flatten()
        .map(|k| k.mime_type().to_string())
        .unwrap_or_else(|| "application/octet-stream".to_string());
    let video = est_video(&mime, &ext);

    let mut morceaux: Vec<String> = Vec::new();
    // Un échec passager mémorisé : il ne vaut la peine de re-tenter que si l'on n'a
    // RIEN obtenu par ailleurs.
    let mut passager: Option<String> = None;

    // ---- 1) Description visuelle (vidéos seulement) -------------------------
    if cfg.video.enabled && video {
        if au_dessus_du_plafond(taille, cfg.video.max_file_mb) {
            tracing::info!(
                "🎬 {path} : {} Mo au-dessus du plafond de description vidéo ({} Mo), description ignorée",
                taille / (1024 * 1024),
                cfg.video.max_file_mb
            );
        } else {
            let prompt = crate::config::prompt_or(
                &cfg.prompts.video_describe,
                crate::config::default_prompts::VIDEO_DESCRIBE,
            );
            match state.ai.video_client().describe(path, &mime, prompt).await {
                Ok(d) if !d.trim().is_empty() => {
                    tracing::info!("🎬 vidéo décrite : {path}");
                    morceaux.push(format!("Description visuelle : {}", d.trim()));
                }
                Ok(_) => {}
                Err(AiCallError::Permanent(e)) => {
                    tracing::warn!("description vidéo refusée pour {path} ({e})");
                }
                Err(AiCallError::Transient(e)) => {
                    passager = Some(format!("description vidéo : {e}"));
                }
            }
        }
    }

    // ---- 2) Transcription de la parole --------------------------------------
    if cfg.transcription.enabled {
        if au_dessus_du_plafond(taille, cfg.transcription.max_file_mb) {
            tracing::info!(
                "🎧 {path} : {} Mo au-dessus du plafond de transcription ({} Mo), transcription ignorée",
                taille / (1024 * 1024),
                cfg.transcription.max_file_mb
            );
        } else {
            match state
                .ai
                .transcription_client()
                .transcribe(path, &file_name, &mime)
                .await
            {
                Ok(t) if !t.trim().is_empty() => {
                    tracing::info!("🎧 {} caractères transcrits : {path}", t.chars().count());
                    morceaux.push(format!("Transcription : {}", t.trim()));
                }
                // Réponse vide : média sans parole (musique, ambiance, silence).
                Ok(_) => {}
                Err(AiCallError::Permanent(e)) => {
                    tracing::warn!("transcription refusée pour {path} ({e})");
                }
                Err(AiCallError::Transient(e)) => {
                    passager = Some(format!("transcription : {e}"));
                }
            }
        }
    }

    // ---- 3) Bilan ------------------------------------------------------------
    if morceaux.is_empty() {
        // Rien obtenu ET une panne passagère : on re-tente tant qu'il reste des
        // essais. Si l'on a obtenu quelque chose par ailleurs, on préfère l'indexer
        // plutôt que de tout rejouer pour compléter.
        if let Some(raison) = passager {
            if retry_count + 1 < MAX_RETRIES {
                return Err(anyhow::anyhow!(
                    "serveur média indisponible (tentative {}/{}) : {}",
                    retry_count + 1,
                    MAX_RETRIES,
                    raison
                ));
            }
            tracing::warn!(
                "serveur média toujours indisponible pour {path} après {} tentatives ({raison}), repli contextuel",
                MAX_RETRIES
            );
        }
        return index_context_only(state, path, mtime, "média").await;
    }

    let doc_type = if video { "video" } else { "audio" };
    store_text_document(
        state,
        path,
        mtime,
        &morceaux.join("\n\n"),
        &hash,
        doc_type,
        cfg.indexing.qualify_media,
    )
    .await
}

/// Indexation par le contexte seul : nom, dossier parent, extension, taille.
/// C'est le filet de sécurité pour tout fichier dont on ne peut lire le contenu.
async fn index_context_only(
    state: &AppState,
    path: &str,
    mtime: i64,
    kind: &str,
) -> anyhow::Result<Option<Pending>> {
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

    Ok(Some(Pending {
        path: path.to_string(),
        mtime,
        hash,
        kind: kind.to_string(),
        summary,
        extract,
        stored: vec![text.clone()],
        embed: vec![text],
        log: format!("🧩 indexé par contexte ({kind}) : {path}"),
    }))
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
        .chat_quick(
            vec![
                ChatMessage { role: "system".into(), content: system.into() },
                ChatMessage { role: "user".into(), content: user },
            ],
            false,
            state.config.snapshot().indexing.qualify_effort,
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
    path: &str,
    mtime: i64,
) -> anyhow::Result<Option<Pending>> {
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

    // Sens = description LLM (qualification) ; contenu extrait = le listing du dossier
    // (types + exemples), pour comparer. Si pas de description distincte, pas d'extract redondant.
    let extract = description.as_ref().map(|_| facts.clone());
    Ok(Some(Pending {
        path: path.to_string(),
        mtime,
        hash,
        kind: "folder-block".to_string(),
        summary,
        extract,
        stored: vec![text.clone()],
        embed: vec![text],
        log: format!("📦 dossier indexé en bloc : {path}"),
    }))
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
        .chat_quick(
            vec![
                ChatMessage { role: "system".into(), content: system.into() },
                ChatMessage { role: "user".into(), content: user },
            ],
            false,
            state.config.snapshot().indexing.qualify_effort,
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
        "pptx" => extract_pptx_text(path),
        "xlsx" => extract_xlsx_text(path),
        "html" | "htm" => Ok(strip_html(&fs::read_to_string(path).unwrap_or_default())),
        _ => Ok(fs::read_to_string(path).unwrap_or_default()),
    }
}

/// Décompresse un flux `FlateDecode`. `None` si le flux n'est pas exploitable.
///
/// Les PDF utilisent le format zlib (en-tête `78 xx`), mais certains producteurs
/// écrivent du deflate brut : on tente les deux plutôt que d'abandonner une image.
fn inflate(data: &[u8]) -> Option<Vec<u8>> {
    use std::io::Read;
    let mut out = Vec::new();
    if flate2::read::ZlibDecoder::new(data).read_to_end(&mut out).is_ok() && !out.is_empty() {
        return Some(out);
    }
    out.clear();
    // Un flux tronqué peut décoder partiellement : on garde ce qui a été obtenu.
    let mut d = flate2::read::DeflateDecoder::new(data);
    let _ = d.read_to_end(&mut out);
    (!out.is_empty()).then_some(out)
}

/// Extrait les images JPEG embarquées d'un PDF. **Repli** derrière
/// [`render_pdf_pages`] : on ne s'en sert que sur un PDF que le moteur de rendu
/// n'a pas su ouvrir, car une image embarquée n'est pas une page (voir la
/// remarque sur les couches dans la doc de `render_pdf_pages`).
///
/// On ne gère que le filtre DCTDecode : le flux brut EST alors un JPEG valide.
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
        // Chaîne de filtres, dans l'ordre d'application.
        let filtres: Vec<Vec<u8>> = match dict.get(b"Filter").ok() {
            Some(lopdf::Object::Name(n)) => vec![n.clone()],
            Some(lopdf::Object::Array(a)) => {
                a.iter().filter_map(|o| o.as_name().ok().map(|n| n.to_vec())).collect()
            }
            _ => Vec::new(),
        };
        let Some(dernier) = filtres.last() else { continue };
        // On ne sait produire un fichier image valide que pour le JPEG. Les scans en
        // CCITTFaxDecode / JBIG2 sont des flux de pixels bruts : c'est le rendu de
        // page qui les traite, ce repli ne cherche pas à les reconstruire.
        if dernier.as_slice() != b"DCTDecode" {
            continue;
        }

        // Un JPEG peut être ENVELOPPÉ dans une compression Flate : la chaîne vaut
        // alors `[FlateDecode, DCTDecode]`. Pousser `content` tel quel enverrait au
        // modèle de vision des octets encore compressés — c'est-à-dire un fichier
        // invalide, un échec vision, et un document indexé « vide » à tort.
        // lopdf est compilé sans ses algorithmes de décompression : on inflate nous-mêmes.
        let octets = if filtres.iter().any(|f| f == b"FlateDecode") {
            match inflate(&stream.content) {
                Some(v) => v,
                None => {
                    tracing::debug!("image PDF : décompression Flate impossible");
                    continue;
                }
            }
        } else {
            stream.content.clone()
        };

        // Un JPEG valide commence par le marqueur SOI (0xFF 0xD8).
        if octets.starts_with(&[0xFF, 0xD8]) {
            out.push(octets);
        } else {
            tracing::debug!("image PDF ignorée : ce n'est pas un JPEG exploitable");
        }
    }
    out
}

/// Nombre de pages soumises au modèle de vision. Au-delà, le coût grimpe sans
/// rien apporter : un document long est identifié par ses premières pages.
const MAX_PAGES_VISION: usize = 8;

/// Résolution du rendu : ~1600 px sur le grand côté. En dessous, les petits
/// caractères (mentions légales, MRZ d'un passeport) deviennent illisibles ;
/// au-dessus, on paie des tokens pour du détail que le modèle n'exploite pas.
const RENDU_GRAND_COTE: f32 = 1600.0;

/// Qualité JPEG des pages rendues : au-delà le poids double sans gain visible.
const RENDU_QUALITE_JPEG: u8 = 82;

/// Rend en JPEG les `max` premières pages d'un PDF.
///
/// C'est la seule façon fiable de donner une PAGE au modèle de vision, car un
/// scan n'est pas « une image par page ». Les scanners produisent couramment du
/// MRC : la page est découpée en un JPEG de fond et plusieurs couches de texte
/// en CCITTFaxDecode. Extraire les images embarquées ne rend alors que le fond
/// — la photo et un décalque pâle, sans une seule ligne de texte net — ce qui
/// laissait le modèle inventer une description à partir du nom du fichier.
/// Dessiner la page compose les couches et couvre au passage les PDF en pixels
/// bruts, en JBIG2, en JPEG2000 et les PDF purement vectoriels.
fn render_pdf_pages(path: &str, max: usize) -> Vec<Vec<u8>> {
    let Ok(data) = fs::read(path) else {
        return Vec::new();
    };
    // `Pdf::new` essaie le mot de passe vide : les PDF « protégés » contre la
    // copie, fréquents sur les documents officiels, s'ouvrent donc directement.
    let Ok(pdf) = hayro::hayro_syntax::Pdf::new(data) else {
        tracing::debug!("rendu PDF : document illisible par hayro ({path})");
        return Vec::new();
    };

    let cache = hayro::RenderCache::new();
    let interpretation = hayro::hayro_interpret::InterpreterSettings::default();
    let mut out = Vec::new();
    for page in pdf.pages().iter().take(max) {
        // Une page malformée peut faire paniquer l'interpréteur. Sans ce
        // garde-fou, elle emporterait l'OCR de tout le document.
        let rendu = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (largeur, hauteur) = page.render_dimensions();
            let echelle = (RENDU_GRAND_COTE / largeur.max(hauteur)).clamp(1.0, 4.0);
            let settings = hayro::RenderSettings {
                x_scale: echelle,
                y_scale: echelle,
                // Fond blanc : sur un fond transparent, l'abandon du canal alpha
                // noircit la page entière et le modèle ne voit plus rien.
                bg_color: hayro::vello_cpu::color::palette::css::WHITE,
                ..Default::default()
            };
            hayro::render(page, &cache, &interpretation, &settings)
        }));
        let Ok(pixmap) = rendu else {
            tracing::debug!("rendu PDF : page ignorée (panique de l'interpréteur) — {path}");
            continue;
        };
        match encode_jpeg(pixmap) {
            Some(jpeg) => out.push(jpeg),
            None => tracing::debug!("rendu PDF : encodage JPEG impossible ({path})"),
        }
    }
    out
}

/// Encode un pixmap en JPEG. Le canal alpha est écarté : le fond ayant été peint
/// en blanc, il vaut 255 partout.
fn encode_jpeg(pixmap: hayro::vello_cpu::Pixmap) -> Option<Vec<u8>> {
    let (largeur, hauteur) = (pixmap.width() as u32, pixmap.height() as u32);
    let rgb: Vec<u8> = pixmap
        .take_unpremultiplied()
        .iter()
        .flat_map(|p| [p.r, p.g, p.b])
        .collect();
    let brute = image::RgbImage::from_raw(largeur, hauteur, rgb)?;
    let mut jpeg = std::io::Cursor::new(Vec::new());
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, RENDU_QUALITE_JPEG)
        .encode_image(&image::DynamicImage::ImageRgb8(brute))
        .ok()?;
    Some(jpeg.into_inner())
}

/// Images à soumettre au modèle de vision, par ordre de préférence : le rendu
/// des pages, puis — si le moteur n'a pas su ouvrir le document — les JPEG
/// embarqués, qui valent toujours mieux que rien.
async fn pages_pour_vision(path: &str) -> Vec<Vec<u8>> {
    let path_owned = path.to_string();
    let rendues = tokio::task::spawn_blocking(move || render_pdf_pages(&path_owned, MAX_PAGES_VISION))
        .await
        .unwrap_or_default();
    if !rendues.is_empty() {
        return rendues;
    }
    tracing::debug!("rendu PDF vide, repli sur les images embarquées : {path}");
    let path_owned = path.to_string();
    tokio::task::spawn_blocking(move || extract_pdf_images(&path_owned, MAX_PAGES_VISION))
        .await
        .unwrap_or_default()
}

/// OCR d'un PDF scanné : envoie ses pages au modèle de vision.
async fn ocr_pdf_via_vision(state: &AppState, path: &str) -> Option<String> {
    let images = pages_pour_vision(path).await;
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
    for img in images.iter() {
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
    archive.by_name("word/document.xml")?.read_to_string(&mut xml)?;
    let mut out = String::new();
    strip_xml_into(&xml, &mut out);
    Ok(out)
}

/// Extrait le texte de toutes les entrées d'un ZIP OOXML dont le nom satisfait `keep`
/// (ex. les slides d'un PPTX, la table de chaînes d'un XLSX). Chaque entrée est un XML
/// dont on ne garde que le texte.
fn extract_office_zip_text(path: &str, keep: impl Fn(&str) -> bool) -> anyhow::Result<String> {
    let file = File::open(path)?;
    let mut archive = ZipArchive::new(file)?;
    let names: Vec<String> = (0..archive.len())
        .filter_map(|i| archive.by_index(i).ok().map(|f| f.name().to_string()))
        .filter(|n| keep(n))
        .collect();
    let mut out = String::new();
    for name in names {
        let mut xml = String::new();
        if let Ok(mut entry) = archive.by_name(&name) {
            if entry.read_to_string(&mut xml).is_ok() {
                strip_xml_into(&xml, &mut out);
                out.push('\n');
            }
        }
    }
    Ok(out)
}

/// PowerPoint : une entrée par slide (`ppt/slides/slideN.xml`).
fn extract_pptx_text(path: &str) -> anyhow::Result<String> {
    extract_office_zip_text(path, |n| {
        n.starts_with("ppt/slides/slide") && n.ends_with(".xml")
    })
}

/// Excel : le texte des cellules vit dans la table de chaînes partagées.
fn extract_xlsx_text(path: &str) -> anyhow::Result<String> {
    extract_office_zip_text(path, |n| n == "xl/sharedStrings.xml")
}

/// Strip des balises XML/HTML : ne garde que le texte, avec un espace inséré entre deux
/// nœuds pour ne pas coller les mots (`<t>Hello</t><t>World</t>` → `Hello World`).
fn strip_xml_into(xml: &str, out: &mut String) {
    let mut in_tag = false;
    for c in xml.chars() {
        match c {
            '<' => {
                in_tag = true;
                if !out.is_empty() && !out.ends_with(|w: char| w.is_whitespace()) {
                    out.push(' ');
                }
            }
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
}

/// Texte lisible d'un document HTML : retire les blocs script/style, puis les balises,
/// et décode quelques entités courantes.
fn strip_html(html: &str) -> String {
    let mut s = html.to_string();
    for (open, close) in [("<script", "</script>"), ("<style", "</style>")] {
        loop {
            let lower = s.to_ascii_lowercase();
            let Some(start) = lower.find(open) else { break };
            let end = lower[start..]
                .find(close)
                .map(|rel| start + rel + close.len())
                .unwrap_or_else(|| s.len());
            s.replace_range(start..end, " ");
        }
    }
    let mut out = String::new();
    strip_xml_into(&s, &mut out);
    out.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
}

#[cfg(test)]
mod tests {
    /// Diagnostic manuel : que voit réellement l'extracteur dans un PDF donné ?
    ///
    /// ```text
    /// PDF_DIAG="C:\chemin\vers\doc.pdf" cargo test --lib diag_images_pdf -- --ignored --nocapture
    /// ```
    /// Ajouter `PDF_DIAG_OUT=<dossier>` pour écrire les pages rendues sur disque.
    #[test]
    #[ignore = "diagnostic manuel (PDF_DIAG)"]
    fn diag_images_pdf() {
        let path = std::env::var("PDF_DIAG").expect("PDF_DIAG non défini");
        let doc = lopdf::Document::load(&path);
        match &doc {
            Ok(d) => println!("lopdf : chargé, {} objets", d.objects.len()),
            Err(e) => println!("lopdf : ÉCHEC DE CHARGEMENT — {e}"),
        }
        if let Ok(d) = &doc {
            let mut images = 0;
            let mut filtres: std::collections::BTreeMap<String, usize> = Default::default();
            for (_id, obj) in &d.objects {
                let lopdf::Object::Stream(s) = obj else { continue };
                let est_image = s
                    .dict
                    .get(b"Subtype")
                    .ok()
                    .and_then(|o| o.as_name().ok())
                    .map(|n| n == b"Image")
                    .unwrap_or(false);
                if !est_image {
                    continue;
                }
                images += 1;
                let f = match s.dict.get(b"Filter").ok() {
                    Some(lopdf::Object::Name(n)) => String::from_utf8_lossy(n).to_string(),
                    Some(lopdf::Object::Array(a)) => a
                        .iter()
                        .filter_map(|o| o.as_name().ok())
                        .map(|n| String::from_utf8_lossy(n).to_string())
                        .collect::<Vec<_>>()
                        .join("+"),
                    _ => "(aucun)".into(),
                };
                *filtres.entry(f).or_default() += 1;
            }
            println!("objets image vus par lopdf : {images}");
            println!("filtres                    : {filtres:?}");
        }
        let extraites = super::extract_pdf_images(&path, super::MAX_PAGES_VISION);
        println!("images RETENUES par extract_pdf_images (repli) : {}", extraites.len());
        for (i, img) in extraites.iter().enumerate() {
            println!("  #{i} : {} octets", img.len());
        }

        // Ce que le modèle de vision reçoit réellement depuis la correction.
        let debut = std::time::Instant::now();
        let rendues = super::render_pdf_pages(&path, super::MAX_PAGES_VISION);
        println!(
            "pages RENDUES par render_pdf_pages : {} (en {:?})",
            rendues.len(),
            debut.elapsed()
        );
        for (i, page) in rendues.iter().enumerate() {
            println!("  page {i} : {} Ko JPEG", page.len() / 1024);
        }
        // `PDF_DIAG_OUT=<dossier>` écrit les pages rendues pour inspection visuelle.
        if let Ok(dir) = std::env::var("PDF_DIAG_OUT") {
            for (i, page) in rendues.iter().enumerate() {
                let out = std::path::Path::new(&dir).join(format!("page{i}.jpg"));
                std::fs::write(&out, page).expect("écriture de la page rendue");
                println!("  écrit : {}", out.display());
            }
        }
    }

    /// Assemble un PDF minimal d'une page — un rectangle noir sur fond blanc —
    /// avec une table xref correcte. Généré plutôt qu'embarqué : quatre objets
    /// écrits ici se relisent mieux qu'un binaire opaque dans le dépôt.
    fn pdf_minimal() -> Vec<u8> {
        // Rectangle volontairement minoritaire dans la page : le fond blanc doit
        // rester largement majoritaire pour que l'assertion sur l'alpha ait du sens.
        let flux = "0 0 0 rg 20 20 100 40 re f";
        let objets = [
            "<</Type/Catalog/Pages 2 0 R>>".to_string(),
            "<</Type/Pages/Kids[3 0 R]/Count 1>>".to_string(),
            "<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 100]/Resources<<>>/Contents 4 0 R>>"
                .to_string(),
            format!("<</Length {}>>stream
{flux}
endstream", flux.len()),
        ];

        let mut pdf = b"%PDF-1.4
".to_vec();
        let mut offsets = Vec::new();
        for (i, corps) in objets.iter().enumerate() {
            offsets.push(pdf.len());
            pdf.extend_from_slice(format!("{} 0 obj{corps}
endobj
", i + 1).as_bytes());
        }

        let debut_xref = pdf.len();
        pdf.extend_from_slice(format!("xref
0 {}
", objets.len() + 1).as_bytes());
        pdf.extend_from_slice(b"0000000000 65535 f 
");
        for offset in &offsets {
            pdf.extend_from_slice(format!("{offset:010} 00000 n 
").as_bytes());
        }
        pdf.extend_from_slice(
            format!(
                "trailer<</Size {}/Root 1 0 R>>
startxref
{debut_xref}
%%EOF
",
                objets.len() + 1
            )
            .as_bytes(),
        );
        pdf
    }

    /// Écrit `contenu` dans un fichier temporaire unique et renvoie son chemin.
    fn fichier_temporaire(suffixe: &str, contenu: &[u8]) -> std::path::PathBuf {
        let chemin = std::env::temp_dir().join(format!(
            "sensetree-test-{}-{suffixe}.pdf",
            std::process::id()
        ));
        std::fs::write(&chemin, contenu).expect("écriture du fichier temporaire");
        chemin
    }

    #[test]
    fn render_pdf_pages_dessine_la_page_entiere() {
        let chemin = fichier_temporaire("rendu", &pdf_minimal());
        let pages = super::render_pdf_pages(chemin.to_str().unwrap(), super::MAX_PAGES_VISION);
        let _ = std::fs::remove_file(&chemin);

        assert_eq!(pages.len(), 1, "une page attendue");
        // Marqueur SOI : ce qui part vers le modèle de vision doit être un JPEG.
        assert!(pages[0].starts_with(&[0xFF, 0xD8]), "ce n'est pas un JPEG");

        let image = image::load_from_memory(&pages[0])
            .expect("JPEG illisible")
            .into_rgb8();
        // Page de 200x100 pt : l'échelle voulue (8x) est bornée à 4x.
        assert_eq!((image.width(), image.height()), (800, 400));

        // Le rectangle doit être là — sinon on a rendu une page blanche, ce qui
        // ramènerait le bug d'origine sous une autre forme.
        let sombres = image.pixels().filter(|p| p.0[0] < 64).count();
        assert!(sombres > 0, "page rendue vide : rien n'a été dessiné");
        // …et le fond doit rester blanc : sans `bg_color`, l'abandon du canal
        // alpha noircit toute la page.
        let clairs = image.pixels().filter(|p| p.0[0] > 200).count();
        assert!(clairs > sombres, "fond noirci : canal alpha mal traité");
    }

    #[test]
    fn render_pdf_pages_rend_la_main_sur_un_document_illisible() {
        // Le contrat compte : un rendu vide est ce qui déclenche le repli sur
        // les images embarquées dans `pages_pour_vision`.
        let chemin = fichier_temporaire("invalide", b"ceci n'est pas un PDF");
        let pages = super::render_pdf_pages(chemin.to_str().unwrap(), super::MAX_PAGES_VISION);
        let _ = std::fs::remove_file(&chemin);
        assert!(pages.is_empty(), "un fichier invalide ne doit rien produire");

        assert!(
            super::render_pdf_pages("Z:/aucun-fichier.pdf", super::MAX_PAGES_VISION).is_empty(),
            "un chemin inexistant ne doit rien produire"
        );
    }

    #[test]
    fn render_pdf_pages_respecte_le_plafond_de_pages() {
        let chemin = fichier_temporaire("plafond", &pdf_minimal());
        let pages = super::render_pdf_pages(chemin.to_str().unwrap(), 0);
        let _ = std::fs::remove_file(&chemin);
        assert!(pages.is_empty(), "max = 0 doit ne rien rendre");
    }

    use super::{strip_html, strip_xml_into, summary_of};

    #[test]
    fn summary_of_nettoie_et_tronque() {
        assert_eq!(summary_of("  hello\nworld  "), "hello world");
        assert_eq!(summary_of(&"a".repeat(500)).chars().count(), 300);
        assert_eq!(summary_of(""), "");
    }

    #[test]
    fn strip_xml_separe_les_noeuds_texte() {
        let mut out = String::new();
        strip_xml_into("<a:t>Hello</a:t><a:t>World</a:t>", &mut out);
        assert!(out.contains("Hello World"), "collage des mots : {out:?}");
    }

    #[test]
    fn strip_html_retire_script_style_et_balises() {
        let html = "<html><head><style>.a{color:red}</style></head><body>\
            <script>evil()</script><p>Bonjour&nbsp;le monde</p></body></html>";
        let text = strip_html(html);
        assert!(text.contains("Bonjour"));
        assert!(text.contains("le monde"));
        assert!(!text.contains("evil"), "script non retiré : {text:?}");
        assert!(!text.contains("color:red"), "style non retiré : {text:?}");
    }
}

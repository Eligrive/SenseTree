pub mod actions;
pub mod benchmarks;
pub mod catalog;
pub mod chunker;
pub mod classifier;
pub mod config;
pub mod crawler;
pub mod db;
pub mod explorer;
pub mod folders;
pub mod gardener;
pub mod installs;
pub mod mcp;
pub mod metrics;
pub mod ollama_catalog;
pub mod ollama_server;
pub mod ort_setup;
pub mod parser;
pub mod providers;
pub mod search;
pub mod state;
pub mod vectordb;
pub mod watchdog;
pub mod worker;

use std::sync::Arc;
use std::time::Duration;

use tauri::{Manager, State};

use config::{AppConfig, ConfigStore};
use db::Database;
use providers::{AiEngine, HealthReport};
use state::AppState;
use vectordb::VectorDb;

// =========================================================================
// COMMANDES : configuration & santé des providers
// =========================================================================

#[tauri::command]
fn get_config(state: State<'_, Arc<AppState>>) -> Result<AppConfig, String> {
    Ok(state.config.snapshot())
}

/// Renvoie les prompts système par défaut (pour l'édition dans les Paramètres :
/// un champ laissé identique ou vide retombe sur ces valeurs intégrées).
#[tauri::command]
fn get_default_prompts() -> config::PromptsConfig {
    config::PromptsConfig::defaults()
}

#[tauri::command]
async fn set_config(state: State<'_, Arc<AppState>>, config: AppConfig) -> Result<(), String> {
    let st = state.inner().clone();
    let old = st.config.snapshot();
    let roots = config.indexing.roots.clone();

    // Le modèle d'embedding change → l'espace vectoriel change → réindexation
    // complète obligatoire (les anciens vecteurs sont incompatibles).
    let reindex = old.embedding.model != config.embedding.model
        || old.embedding.dimensions != config.embedding.dimensions;
    let dims_changed = old.embedding.dimensions != config.embedding.dimensions;
    // La tendance de classification ou le prompt change → reclasser les dossiers.
    let reclassify = (old.indexing.block_bias - config.indexing.block_bias).abs() > f32::EPSILON
        || old.prompts.folder_classify != config.prompts.folder_classify;

    st.config.replace(config.clone()).map_err(|e| e.to_string())?;
    st.ai.invalidate_embedder().await;

    // Réindexation ou reclassement : le scan en cours travaille sur un état qu'on
    // vient d'effacer → on l'annule pour qu'il reparte proprement (voir scan_epoch).
    if reindex || reclassify {
        st.scan_epoch.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    if reindex {
        // La base vectorielle est figée sur l'ancienne dimension : on la recrée.
        if dims_changed {
            st.vector.set_dim(config.embedding.dimensions);
        }
        st.vector.clear().await.ok();
        // reset_index purge aussi les profils de dossiers → reclassement complet.
        st.db.reset_index().map_err(|e| e.to_string())?;
        tracing::info!("🔄 modèle d'embedding modifié → réindexation complète");
    } else if reclassify {
        st.db.clear_folder_profiles().map_err(|e| e.to_string())?;
        tracing::info!("🧭 réglage de classification modifié → reclassement des dossiers");
    }

    // Re-scan CIBLÉ : complet seulement si réindexation (embedding) ou reclassement
    // (bias/prompt de classification) ; sinon on ne scanne QUE les racines nouvellement
    // ajoutées. Sans ça, chaque sauvegarde de Paramètres (changement de modèle reasoning,
    // de clé API, d'un prompt…) relançait un scan complet inutile de tous les dossiers.
    let to_scan: Vec<String> = if reindex || reclassify {
        roots
    } else {
        roots
            .into_iter()
            .filter(|r| {
                let rn = r.trim_end_matches(['/', '\\']);
                !old
                    .indexing
                    .roots
                    .iter()
                    .any(|o| o.trim_end_matches(['/', '\\']).eq_ignore_ascii_case(rn))
            })
            .collect()
    };
    for root in to_scan {
        let sc = st.clone();
        std::thread::spawn(move || crawler::scan_directory(sc, &root));
    }
    Ok(())
}

/// Force manuellement le mode d'un dossier (récursif vs bloc sémantique).
#[tauri::command]
async fn set_folder_mode(
    state: State<'_, Arc<AppState>>,
    path: String,
    mode: String,
) -> Result<(), String> {
    let st = state.inner().clone();
    // Garde-fou : on ne classe (bloc/récursif) que des dossiers faisant partie de
    // l'indexation. Un dossier hors périmètre doit d'abord être ajouté aux racines.
    if !st.config.is_under_root(&path) {
        return Err("Ce dossier ne fait pas partie de l'indexation : ajoutez-le d'abord aux dossiers indexés.".into());
    }
    st.db.set_folder_profile_manual(&path, &mode).map_err(|e| e.to_string())?;

    if mode == "block" {
        // On retire l'indexation fichier-par-fichier et on indexe le dossier en bloc.
        st.db.purge_children(&path).map_err(|e| e.to_string())?;
        st.vector.delete_under(&path).await.map_err(|e| e.to_string())?;
        st.db.enqueue_path(&path, Some("pending_extraction"), 9).map_err(|e| e.to_string())?;
    } else {
        // Retour en récursif : on supprime le vecteur-bloc et on re-scanne le dossier.
        st.vector.delete_by_path(&path).await.map_err(|e| e.to_string())?;
        let _ = st.db.remove_from_queue(&path);
        let sc = st.clone();
        let p = path.clone();
        std::thread::spawn(move || crawler::scan_directory(sc, &p));
    }
    Ok(())
}

/// Ouvre un sélecteur natif de dossier ; renvoie le chemin choisi (ou None).
#[tauri::command]
async fn pick_folder(app: tauri::AppHandle) -> Option<String> {
    use tauri_plugin_dialog::DialogExt;
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog().file().pick_folder(move |path| {
        let _ = tx.send(path);
    });
    match rx.await {
        Ok(Some(fp)) => fp.into_path().ok().map(|p| p.to_string_lossy().to_string()),
        _ => None,
    }
}

/// Ajoute un dossier à l'indexation (racine) et lance son scan. Renvoie la liste à jour.
#[tauri::command]
async fn add_indexed_folder(
    state: State<'_, Arc<AppState>>,
    path: String,
) -> Result<Vec<String>, String> {
    let st = state.inner().clone();
    let norm = path.trim_end_matches(['/', '\\']).to_string();
    if norm.is_empty() {
        return Err("chemin vide".into());
    }
    if !std::path::Path::new(&norm).is_dir() {
        return Err("le dossier est introuvable".into());
    }
    let mut cfg = st.config.snapshot();
    let already = cfg
        .indexing
        .roots
        .iter()
        .any(|r| r.trim_end_matches(['/', '\\']).eq_ignore_ascii_case(&norm));
    if already {
        return Ok(cfg.indexing.roots);
    }
    cfg.indexing.roots.push(norm.clone());
    let roots = cfg.indexing.roots.clone();
    st.config.replace(cfg).map_err(|e| e.to_string())?;

    // Indexation immédiate + surveillance temps réel du nouveau dossier.
    let sc = st.clone();
    let p = norm.clone();
    std::thread::spawn(move || crawler::scan_directory(sc, &p));
    watchdog::watch_root(st.clone(), norm);
    Ok(roots)
}

/// Retire un dossier de l'indexation. Purge ses données, mais préserve les
/// sous-dossiers ajoutés explicitement (ils restent indexés / sont ré-indexés).
#[tauri::command]
async fn remove_indexed_folder(
    state: State<'_, Arc<AppState>>,
    path: String,
) -> Result<Vec<String>, String> {
    let st = state.inner().clone();
    let norm = path.trim_end_matches(['/', '\\']).to_string();
    let mut cfg = st.config.snapshot();
    let before = cfg.indexing.roots.len();
    cfg.indexing
        .roots
        .retain(|r| !r.trim_end_matches(['/', '\\']).eq_ignore_ascii_case(&norm));
    let remaining = cfg.indexing.roots.clone();
    if remaining.len() == before {
        return Ok(remaining); // ce n'était pas une racine
    }
    st.config.replace(cfg).map_err(|e| e.to_string())?;

    // Le chemin reste-t-il couvert par une autre racine (racine parente) ?
    let still_covered = config::path_under_root(&remaining, &norm);
    if !still_covered {
        // Plus couvert : on purge tout le sous-arbre…
        st.vector.delete_under(&norm).await.ok();
        st.db.purge_tree(&norm).map_err(|e| e.to_string())?;
        // …puis on ré-indexe les sous-racines explicitement conservées.
        for r in &remaining {
            if config::path_under_root(std::slice::from_ref(&norm), r.trim_end_matches(['/', '\\'])) {
                let sc = st.clone();
                let p = r.clone();
                std::thread::spawn(move || crawler::scan_directory(sc, &p));
            }
        }
    }
    Ok(remaining)
}

#[tauri::command]
async fn ai_health(state: State<'_, Arc<AppState>>) -> Result<HealthReport, String> {
    let st = state.inner().clone();
    Ok(st.ai.health().await)
}

/// Récupère la liste des modèles disponibles sur un endpoint compatible OpenAI.
async fn fetch_models(base_url: &str, api_key: &str) -> Result<Vec<String>, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(6))
        .build()
        .map_err(|e| e.to_string())?;
    let url = format!("{}/models", base_url.trim_end_matches('/'));
    let mut req = client.get(&url);
    if !api_key.is_empty() {
        req = req.bearer_auth(api_key);
    }
    let resp = req.send().await.map_err(|e| format!("serveur injoignable: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("le serveur a répondu {}", resp.status()));
    }
    let value: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    let models = value
        .get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|i| i.as_str()).map(String::from))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Ok(models)
}

/// Liste les modèles installés sur le serveur (pour l'autocomplétion des Paramètres).
#[tauri::command]
async fn list_installed_models(base_url: String, api_key: String) -> Result<Vec<String>, String> {
    fetch_models(&base_url, &api_key).await
}

#[derive(serde::Serialize)]
struct LocalModelStatus {
    id: String,
    dimensions: usize,
    /// Vrai si le modèle est déjà téléchargé dans le cache local.
    downloaded: bool,
    /// Vrai si le modèle gère autre chose que l'anglais (famille E5 uniquement).
    multilingual: bool,
}

/// État des modèles d'embedding LOCAUX (fastembed) : lesquels sont déjà téléchargés.
/// Permet de ne proposer dans le menu déroulant que ce qui est réellement utilisable.
#[tauri::command]
fn list_local_models(state: State<'_, Arc<AppState>>) -> Vec<LocalModelStatus> {
    let dir = state.data_dir.clone();
    providers::supported_local_models()
        .into_iter()
        .map(|(id, dimensions, multilingual)| LocalModelStatus {
            downloaded: providers::is_local_model_cached(&dir, id),
            id: id.to_string(),
            dimensions,
            multilingual,
        })
        .collect()
}

/// Résout les noms d'installation (Ollama / LM Studio) d'une liste de modèles HF,
/// via les dépôts GGUF réellement présents sur Hugging Face. Cache local 7 jours.
#[tauri::command]
async fn resolve_installs(
    state: State<'_, Arc<AppState>>,
    names: Vec<String>,
) -> Result<Vec<installs::InstallInfo>, String> {
    let st = state.inner().clone();
    installs::resolve(&st.data_dir, names)
        .await
        .map_err(|e| format!("résolution des installations : {e}"))
}

/// Classements MTEB disponibles (global multilingue + par langue), pour que
/// l'utilisateur choisisse ceux qui correspondent à SES langues.
#[tauri::command]
async fn list_benchmark_boards() -> Result<Vec<benchmarks::BoardInfo>, String> {
    benchmarks::list_boards()
        .await
        .map_err(|e| format!("liste des classements : {e}"))
}

/// Benchmarks VISION live (OpenCompass OpenVLM : MMMU, MMBench, OCRBench…).
#[tauri::command]
async fn vision_benchmarks(
    state: State<'_, Arc<AppState>>,
    refresh: bool,
) -> Result<Vec<benchmarks::ModelBenchmark>, String> {
    let st = state.inner().clone();
    catalog::vision(&st.data_dir, refresh)
        .await
        .map_err(|e| format!("benchmarks vision : {e}"))
}

/// Modèles réellement chargés sur le serveur (local ou distant).
///
/// Ollama n'expose aucune API pour lire ou changer sa configuration
/// (`OLLAMA_MAX_LOADED_MODELS`…) : on observe donc l'état réel plutôt que de le
/// supposer, ce qui marche quelle que soit la machine d'en face.
#[tauri::command]
async fn ollama_loaded(base_url: String) -> Result<Vec<ollama_server::LoadedModel>, String> {
    ollama_server::loaded(&base_url).await.map_err(|e| e.to_string())
}

/// Décharge un modèle du serveur pour libérer la VRAM, sans toucher à sa config.
#[tauri::command]
async fn ollama_unload(base_url: String, model: String) -> Result<(), String> {
    ollama_server::unload(&base_url, &model).await.map_err(|e| e.to_string())
}

/// Débit des trois étages IA, mesuré dans les providers.
///
/// À ne pas confondre avec l'avancement de l'indexation : ceci mesure la vitesse des
/// MODÈLES (ms/appel, chunks/s, Mo/s), pas celle du pipeline complet.
#[tauri::command]
fn indexing_throughput() -> metrics::Throughput {
    metrics::snapshot()
}

/// Remet les compteurs de débit à zéro (pour chronométrer une indexation précise).
#[tauri::command]
fn reset_throughput() {
    metrics::reset();
}

/// Bibliothèque Ollama LIVE : ce qui est réellement installable, populaire et récent.
///
/// Complémentaire des benchmarks : ceux-ci disent qui est BON, celle-ci dit qui est
/// DISPONIBLE. Le tri (popularité / récence) est fait côté UI sur `pulls` et
/// `updated_day`, déjà normalisés ici.
#[tauri::command]
async fn ollama_library(
    state: State<'_, Arc<AppState>>,
    refresh: bool,
) -> Result<Vec<ollama_catalog::OllamaModel>, String> {
    let st = state.inner().clone();
    ollama_catalog::library(&st.data_dir, refresh)
        .await
        .map_err(|e| format!("catalogue Ollama : {e}"))
}

/// Tags d'un ou plusieurs modèles Ollama : c'est ce qui porte le choix de la
/// QUANTIFICATION (`9b-q4_K_M` contre `9b-q8_0`) et la taille réelle de chacune.
#[tauri::command]
async fn ollama_tags(
    state: State<'_, Arc<AppState>>,
    models: Vec<String>,
) -> Result<std::collections::HashMap<String, Vec<ollama_catalog::OllamaTag>>, String> {
    let st = state.inner().clone();
    Ok(ollama_catalog::tags_many(&st.data_dir, models).await)
}

/// Benchmarks REASONING live (OpenCompass Academic : IFEval, MMLU-Pro, GPQA…).
#[tauri::command]
async fn reasoning_benchmarks(
    state: State<'_, Arc<AppState>>,
    refresh: bool,
) -> Result<Vec<benchmarks::ModelBenchmark>, String> {
    let st = state.inner().clone();
    catalog::reasoning(&st.data_dir, refresh)
        .await
        .map_err(|e| format!("benchmarks reasoning : {e}"))
}

#[tauri::command]
fn list_vision_boards() -> Vec<benchmarks::BoardInfo> {
    catalog::vision_boards()
}

#[tauri::command]
fn list_reasoning_boards() -> Vec<benchmarks::BoardInfo> {
    catalog::reasoning_boards()
}

/// Scores et specs des modèles depuis l'API OFFICIELLE du leaderboard MTEB, sur les
/// classements choisis par l'utilisateur. Cache local 7 jours (par classement).
#[tauri::command]
async fn model_benchmarks(
    state: State<'_, Arc<AppState>>,
    boards: Vec<String>,
    refresh: bool,
) -> Result<Vec<benchmarks::ModelBenchmark>, String> {
    let st = state.inner().clone();
    benchmarks::load(&st.data_dir, boards, refresh)
        .await
        .map_err(|e| format!("récupération des benchmarks : {e}"))
}

/// Télécharge un modèle d'embedding local (le charge une fois pour peupler le cache).
#[tauri::command]
async fn download_local_model(
    state: State<'_, Arc<AppState>>,
    model: String,
) -> Result<String, String> {
    let st = state.inner().clone();
    st.ai
        .preload_local_model(&model)
        .await
        .map_err(|e| format!("téléchargement du modèle local : {e}"))?;
    Ok(format!("Modèle local « {model} » téléchargé"))
}

/// Teste un endpoint ET vérifie que le modèle configuré est réellement présent.
/// (Avant, on ne testait que l'accessibilité du serveur — d'où un « OK » trompeur
/// quand le modèle n'était pas installé.)
#[tauri::command]
async fn test_chat_endpoint(
    base_url: String,
    api_key: String,
    model: String,
) -> Result<String, String> {
    let models = fetch_models(&base_url, &api_key).await?;
    if model.is_empty() {
        return Ok(format!("Serveur joignable ({} modèle(s) disponible(s))", models.len()));
    }
    // Match tolérant (Ollama expose parfois 'llama3.1:8b' vs 'llama3.1').
    let present = models
        .iter()
        .any(|m| m == &model || m.split(':').next() == Some(model.split(':').next().unwrap_or(&model)));
    if present {
        Ok(format!("Connecté — modèle « {model} » disponible"))
    } else {
        Err(format!(
            "Serveur joignable mais modèle « {model} » introuvable. Modèles installés : {}",
            if models.is_empty() { "aucun".into() } else { models.join(", ") }
        ))
    }
}

/// Teste un endpoint d'embedding : embedde un échantillon et renvoie la dimension
/// réelle du vecteur (pour vérifier qu'elle correspond au champ « Dimensions »).
#[tauri::command]
async fn test_embedding_endpoint(
    base_url: String,
    api_key: String,
    model: String,
) -> Result<String, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()
        .map_err(|e| e.to_string())?;
    let url = format!("{}/embeddings", base_url.trim_end_matches('/'));
    let mut req = client
        .post(&url)
        .json(&serde_json::json!({ "model": model, "input": ["test"] }));
    if !api_key.is_empty() {
        req = req.bearer_auth(&api_key);
    }
    let resp = req.send().await.map_err(|e| format!("serveur injoignable: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        let body: String = body.chars().take(200).collect();
        return Err(format!("le serveur a répondu {status}: {body}"));
    }
    let value: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    let dim = value
        .get("data")
        .and_then(|d| d.get(0))
        .and_then(|e| e.get("embedding"))
        .and_then(|v| v.as_array())
        .map(|a| a.len());
    match dim {
        Some(n) => Ok(format!("Modèle OK — {n} dimensions (règle « Dimensions » sur {n})")),
        None => Err("réponse sans vecteur d'embedding".to_string()),
    }
}

/// Heuristique : l'endpoint pointe-t-il vers LM Studio (port 1234 par défaut) ?
fn is_lmstudio(base_url: &str) -> bool {
    let u = base_url.to_lowercase();
    u.contains(":1234") || u.contains("lmstudio") || u.contains("lm-studio") || u.contains("lm_studio")
}

/// Télécharge un modèle sur LM Studio via son CLI `lms get`. LM Studio n'expose pas
/// d'endpoint HTTP de téléchargement, on passe donc par l'outil en ligne de commande.
async fn pull_lmstudio(app: tauri::AppHandle, model: String) -> Result<String, String> {
    use std::process::Command;
    use tauri::Emitter;

    let emit = |status: &str, percent: u32| {
        let _ = app.emit(
            "model-pull-progress",
            serde_json::json!({ "model": model, "status": status, "completed": 0, "total": 0, "percent": percent }),
        );
    };
    emit("téléchargement via LM Studio (lms get)…", 0);

    let m = model.clone();
    let output = tokio::task::spawn_blocking(move || Command::new("lms").args(["get", &m, "--yes"]).output())
        .await
        .map_err(|e| format!("tâche lms interrompue : {e}"))?;

    match output {
        Ok(o) if o.status.success() => {
            emit("success", 100);
            Ok(format!("Modèle « {model} » téléchargé (LM Studio)"))
        }
        Ok(o) => {
            let err = String::from_utf8_lossy(&o.stderr);
            let out = String::from_utf8_lossy(&o.stdout);
            let msg = if err.trim().is_empty() { out.trim() } else { err.trim() };
            Err(format!("échec de `lms get` : {msg}"))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(
            "CLI « lms » introuvable. Installez le CLI LM Studio (commande `lms`) ou téléchargez le modèle depuis l'application LM Studio."
                .to_string(),
        ),
        Err(e) => Err(format!("lancement de `lms` impossible : {e}")),
    }
}

/// Supprime un modèle du serveur Ollama (`DELETE /api/delete`).
///
/// Destructif et irréversible : le modèle devra être re-téléchargé. L'appelant
/// (l'UI) demande confirmation avant d'appeler — la commande, elle, ne pose pas de
/// question, elle exécute.
///
/// LM Studio n'expose aucune API de suppression : on le dit plutôt que d'échouer
/// avec une erreur HTTP incompréhensible.
#[tauri::command]
async fn delete_model(base_url: String, model: String) -> Result<(), String> {
    if is_lmstudio(&base_url) {
        return Err(
            "LM Studio n'expose pas de suppression : retire le modèle depuis son interface.".into(),
        );
    }
    if model.trim().is_empty() {
        return Err("aucun modèle indiqué".into());
    }

    let url = format!("{}/api/delete", ollama_server::native_base(&base_url));
    let resp = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(|e| e.to_string())?
        .delete(&url)
        .json(&serde_json::json!({ "model": model }))
        .send()
        .await
        .map_err(|e| format!("suppression impossible (serveur Ollama ?) : {e}"))?;

    let status = resp.status();
    if status == reqwest::StatusCode::NOT_FOUND {
        return Err(format!("« {model} » n'est pas installé sur ce serveur"));
    }
    if !status.is_success() {
        let detail = resp.text().await.unwrap_or_default();
        return Err(format!("échec de la suppression ({status}) : {detail}"));
    }
    tracing::info!("🗑️ modèle supprimé du serveur : {model}");
    Ok(())
}

/// Télécharge un modèle : Ollama (POST /api/pull, en streaming avec progression) ou
/// LM Studio (CLI `lms get`), selon l'endpoint détecté. Émet `model-pull-progress`.
#[tauri::command]
async fn pull_model(
    app: tauri::AppHandle,
    base_url: String,
    model: String,
) -> Result<String, String> {
    use futures_util::StreamExt;
    use tauri::Emitter;

    // LM Studio : pas d'API de téléchargement → on passe par le CLI `lms`.
    if is_lmstudio(&base_url) {
        return pull_lmstudio(app, model).await;
    }

    // base_url ~ http://host:11434/v1 → racine Ollama = http://host:11434
    let root = base_url
        .trim_end_matches('/')
        .trim_end_matches("/v1")
        .to_string();
    let url = format!("{root}/api/pull");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3600))
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client
        .post(&url)
        .json(&serde_json::json!({ "name": model, "stream": true }))
        .send()
        .await
        .map_err(|e| format!("téléchargement impossible (serveur Ollama ?) : {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("échec du téléchargement : {}", resp.status()));
    }

    let emit = |status: &str, completed: u64, total: u64| {
        let percent = if total > 0 {
            (completed as f64 / total as f64 * 100.0).round() as u32
        } else {
            0
        };
        let _ = app.emit(
            "model-pull-progress",
            serde_json::json!({
                "model": model,
                "status": status,
                "completed": completed,
                "total": total,
                "percent": percent,
            }),
        );
    };

    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| e.to_string())?;
        buf.push_str(&String::from_utf8_lossy(&chunk));
        // Ollama renvoie du NDJSON : une ligne JSON par mise à jour.
        while let Some(pos) = buf.find('\n') {
            let line: String = buf.drain(..=pos).collect();
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
                    return Err(err.to_string());
                }
                let status = v.get("status").and_then(|s| s.as_str()).unwrap_or("");
                let total = v.get("total").and_then(|t| t.as_u64()).unwrap_or(0);
                let completed = v.get("completed").and_then(|c| c.as_u64()).unwrap_or(0);
                emit(status, completed, total);
            }
        }
    }
    emit("success", 1, 1);
    Ok(format!("Modèle « {model} » téléchargé"))
}

/// Réinitialise complètement l'index et relance un scan des racines.
#[tauri::command]
async fn reindex_all(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    let st = state.inner().clone();
    let cfg = st.config.snapshot();
    // Réindexer = « repars MAINTENANT » : on annule les scans en cours (nouvelle
    // époque) et on lève une éventuelle pause, sinon un crawler bloqué gardait la
    // racine verrouillée et la demande restait sans effet jusqu'au redémarrage.
    st.scan_epoch.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    st.paused.store(false, std::sync::atomic::Ordering::Relaxed);
    // Aligne la dimension de la base vectorielle sur le modèle actuel avant de vider.
    st.vector.set_dim(cfg.embedding.dimensions);
    st.ai.invalidate_embedder().await;
    st.db.reset_index().map_err(|e| e.to_string())?;
    st.vector.clear().await.map_err(|e| e.to_string())?;
    for root in cfg.indexing.roots {
        let sc = st.clone();
        std::thread::spawn(move || crawler::scan_directory(sc, &root));
    }
    Ok(())
}

#[tauri::command]
fn get_recent_activity(state: State<'_, Arc<AppState>>) -> Result<Vec<db::FileRecord>, String> {
    state.db.get_recent_files(15).map_err(|e| e.to_string())
}

#[tauri::command]
fn indexing_stats(state: State<'_, Arc<AppState>>) -> Result<db::IndexingStats, String> {
    state.db.get_indexing_stats().map_err(|e| e.to_string())
}

/// Une entrée de la file enrichie du pipeline visé (embedding / vision / reasoning / context).
#[derive(serde::Serialize)]
struct QueueItemView {
    path: String,
    routes: Vec<String>,
    kind: String,
    status: String,
    retry_count: i64,
    last_error: Option<String>,
}

/// Vue complète de la file d'indexation : élément en cours + éléments à venir + compteurs.
#[derive(serde::Serialize)]
struct IndexingQueueView {
    current: Option<state::CurrentActivity>,
    pending: Vec<QueueItemView>,
    /// Échecs définitifs, listés À PART : noyés dans `pending`, ils sortaient de la
    /// fenêtre `LIMIT` dès que la file était longue et n'étaient jamais affichés.
    failed: Vec<QueueItemView>,
    stats: db::IndexingStats,
}

/// Relance l'indexation d'un chemin en échec (le remet en file, compteur remis à zéro).
#[tauri::command]
fn retry_indexing(state: State<'_, Arc<AppState>>, path: String) -> Result<(), String> {
    state.db.requeue_path(&path).map_err(|e| e.to_string())
}

/// Ignore un chemin en échec : le retire de la file (il ne sera plus retenté).
#[tauri::command]
fn ignore_indexing(state: State<'_, Arc<AppState>>, path: String) -> Result<(), String> {
    state.db.remove_from_queue(&path).map_err(|e| e.to_string())
}

/// Relance TOUTES les indexations en échec définitif. Renvoie le nombre relancé.
#[tauri::command]
fn retry_all_failed(state: State<'_, Arc<AppState>>) -> Result<usize, String> {
    state.db.requeue_failed().map_err(|e| e.to_string())
}

/// Qualifie un fichier À LA DEMANDE (mode « paresseux ») : relit le contenu déjà
/// extrait (`extract`) et fait produire au reasoning un « sens » qualifié, qu'on
/// enregistre. Permet d'indexer vite (qualification désactivée) puis de qualifier
/// les fichiers au cas par cas, sans tout réindexer.
#[tauri::command]
async fn qualify_file(state: State<'_, Arc<AppState>>, path: String) -> Result<String, String> {
    let st = state.inner().clone();
    let cfg = st.config.snapshot();
    if !cfg.reasoning.enabled {
        return Err("Le modèle de reasoning est désactivé.".into());
    }
    let (_, doc_type, extract) = st
        .db
        .get_file_semantics(&path)
        .map_err(|e| e.to_string())?
        .ok_or("Ce fichier n'a pas encore été indexé.")?;
    let extract = extract
        .filter(|e| !e.trim().is_empty())
        .ok_or("Aucun contenu extrait à qualifier pour ce fichier.")?;
    let doc_type = if doc_type.trim().is_empty() { "document".to_string() } else { doc_type };
    let summary = worker::llm_qualify_document(&st, &cfg, &path, &doc_type, &extract)
        .await
        .ok_or("Le modèle n'a pas renvoyé de qualification.")?;
    st.db.set_file_summary(&path, &summary).map_err(|e| e.to_string())?;
    Ok(summary)
}

/// Qualifie TOUS les fichiers PAS ENCORE qualifiés d'un dossier (mode paresseux, en lot).
/// Ne retraite que ceux dont le « sens » est encore un simple extrait ; s'exécute en
/// tâche de fond et respecte la pause. Renvoie le nombre de fichiers à qualifier.
#[tauri::command]
async fn qualify_folder(state: State<'_, Arc<AppState>>, path: String) -> Result<usize, String> {
    let st = state.inner().clone();
    let cfg = st.config.snapshot();
    if !cfg.reasoning.enabled {
        return Err("Le modèle de reasoning est désactivé.".into());
    }
    // On ne garde que les fichiers dont le sens == extrait brut (donc pas encore
    // qualifiés) : évite de re-solliciter l'IA sur ce qui est déjà décrit ou édité.
    let todo: Vec<(String, String, String)> = st
        .db
        .qualifiable_under(&path)
        .map_err(|e| e.to_string())?
        .into_iter()
        .filter(|(_, _, summary, extract)| {
            summary.trim() == worker::summary_of(extract).trim()
        })
        .map(|(p, dt, _, ex)| (p, dt, ex))
        .collect();

    let n = todo.len();
    if n == 0 {
        return Ok(0);
    }

    // Tâche de fond : un appel reasoning par fichier, séquentiel, interruptible par la pause.
    tauri::async_runtime::spawn(async move {
        tracing::info!("✨ qualification du dossier : {n} fichier(s) à traiter…");
        let mut done = 0usize;
        for (p, dt, ex) in todo {
            if st.paused.load(std::sync::atomic::Ordering::Relaxed) {
                tracing::info!("✨ qualification du dossier interrompue (pause).");
                break;
            }
            let dt = if dt.trim().is_empty() { "document".to_string() } else { dt };
            if let Some(sum) = worker::llm_qualify_document(&st, &cfg, &p, &dt, &ex).await {
                let _ = st.db.set_file_summary(&p, &sum);
                done += 1;
            }
        }
        tracing::info!("✨ qualification du dossier terminée ({done} fichier(s) qualifié(s)).");
    });

    Ok(n)
}

/// Édite manuellement le « sens » (résumé) d'un fichier depuis le panneau de détail.
#[tauri::command]
fn set_file_summary(
    state: State<'_, Arc<AppState>>,
    path: String,
    summary: String,
) -> Result<(), String> {
    state
        .db
        .set_file_summary(&path, &summary)
        .map_err(|e| e.to_string())
}

/// File d'indexation temps réel : ce qui est traité maintenant, la suite, et vers quel pipeline.
#[tauri::command]
fn indexing_queue(
    state: State<'_, Arc<AppState>>,
    limit: Option<i64>,
) -> Result<IndexingQueueView, String> {
    let limit = limit.unwrap_or(60).clamp(1, 300);
    let cfg = state.config.snapshot();
    let to_view = |q: db::QueueEntry| {
        let (routes, kind) = worker::route_for(&cfg, &q.path);
        QueueItemView {
            path: q.path,
            routes: routes.iter().map(|s| s.to_string()).collect(),
            kind: kind.to_string(),
            status: q.status,
            retry_count: q.retry_count,
            last_error: q.last_error,
        }
    };

    let pending: Vec<QueueItemView> = state
        .db
        .get_queue(limit)
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(to_view)
        .collect();
    // Requête séparée : les échecs restent visibles quelle que soit la taille de la file.
    let failed: Vec<QueueItemView> = state
        .db
        .get_failed(100)
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(to_view)
        .collect();

    let stats = state.db.get_indexing_stats().map_err(|e| e.to_string())?;
    Ok(IndexingQueueView {
        current: state.activity_snapshot(),
        pending,
        failed,
        stats,
    })
}

/// Met en pause ou reprend l'indexation de fond (worker + classifieur).
#[tauri::command]
fn set_indexing_paused(state: State<'_, Arc<AppState>>, paused: bool) {
    state.paused.store(paused, std::sync::atomic::Ordering::Relaxed);
    tracing::info!("indexation {}", if paused { "en pause" } else { "reprise" });
}

/// Indique si l'indexation est actuellement en pause.
#[tauri::command]
fn indexing_paused(state: State<'_, Arc<AppState>>) -> bool {
    state.paused.load(std::sync::atomic::Ordering::Relaxed)
}

/// Ouvre un fichier/dossier avec l'application par défaut du système.
#[tauri::command]
fn open_path(path: String) -> Result<(), String> {
    #[cfg(windows)]
    let result = std::process::Command::new("cmd")
        .args(["/C", "start", "", &path])
        .spawn();
    #[cfg(target_os = "macos")]
    let result = std::process::Command::new("open").arg(&path).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let result = std::process::Command::new("xdg-open").arg(&path).spawn();

    result.map(|_| ()).map_err(|e| format!("ouverture impossible: {e}"))
}

/// Indique si un GPU NVIDIA est présent au runtime (détection dynamique).
/// L'UI s'en sert pour n'activer la case « Utiliser le GPU » que si elle a un effet.
#[tauri::command]
fn gpu_available() -> bool {
    ort_setup::gpu_present()
}

// =========================================================================
// AMORÇAGE
// =========================================================================

#[cfg_attr(mobile, tauri::mobile_entry_point)]
/// Dernier bilan de santé structurel des racines (gardener proactif de fond).
/// Renvoie le rapport en cache — l'UI l'interroge périodiquement pour ses pastilles.
#[tauri::command]
fn gardener_health(state: State<'_, Arc<AppState>>) -> gardener::GardenerReport {
    state
        .gardener
        .lock()
        .map(|g| g.clone())
        .unwrap_or_default()
}

/// Une note de la mémoire de l'agent (pour l'affichage/gestion dans les Paramètres).
#[derive(serde::Serialize)]
struct MemoryItem {
    id: i64,
    note: String,
}

#[tauri::command]
fn agent_memory_list(state: State<'_, Arc<AppState>>) -> Vec<MemoryItem> {
    state
        .db
        .list_memories(200)
        .unwrap_or_default()
        .into_iter()
        .map(|(id, note)| MemoryItem { id, note })
        .collect()
}

#[tauri::command]
fn agent_memory_delete(state: State<'_, Arc<AppState>>, id: i64) -> Result<(), String> {
    state.db.delete_memory(id).map_err(|e| e.to_string())
}

#[tauri::command]
fn agent_memory_clear(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    state.db.clear_memories().map_err(|e| e.to_string())
}

/// Résultat de recherche d'image (similarité visuelle CLIP).
#[derive(serde::Serialize)]
struct ImageHit {
    path: String,
    name: String,
    score: f32,
}

/// Indexe (à la demande) les images sous les racines (ou `scope`) pour la recherche
/// visuelle CLIP. Renvoie le nombre d'images vectorisées.
#[tauri::command]
async fn index_images(
    state: State<'_, Arc<AppState>>,
    scope: Option<String>,
) -> Result<usize, String> {
    let st = state.inner().clone();
    let roots: Vec<String> = match scope {
        Some(s) if !s.trim().is_empty() => vec![s],
        _ => st.config.snapshot().indexing.roots,
    };
    let clip = st.ai.clip().await.map_err(|e| e.to_string())?;

    let mut images: Vec<std::path::PathBuf> = Vec::new();
    'outer: for root in &roots {
        for entry in walkdir::WalkDir::new(root).into_iter().flatten() {
            if !entry.file_type().is_file() {
                continue;
            }
            let ext = entry
                .path()
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();
            if matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "gif" | "bmp" | "webp") {
                images.push(entry.path().to_path_buf());
                if images.len() >= 5000 {
                    break 'outer; // garde-fou anti-arbre géant
                }
            }
        }
    }
    if images.is_empty() {
        return Ok(0);
    }

    let mut done = 0usize;
    for chunk in images.chunks(16) {
        let batch: Vec<std::path::PathBuf> = chunk.to_vec();
        let stored: Vec<String> = batch.iter().map(|p| p.to_string_lossy().to_string()).collect();
        match clip.embed_images(batch).await {
            Ok(vecs) => {
                for (p, v) in stored.into_iter().zip(vecs) {
                    if st.vector.upsert_image(&p, v).await.is_ok() {
                        done += 1;
                    }
                }
            }
            Err(e) => tracing::warn!("CLIP : lot d'images échoué ({e})"),
        }
    }
    Ok(done)
}

/// Recherche d'images par similarité visuelle à partir d'une requête texte (CLIP).
#[tauri::command]
async fn image_search(
    state: State<'_, Arc<AppState>>,
    query: String,
    limit: Option<usize>,
) -> Result<Vec<ImageHit>, String> {
    let st = state.inner().clone();
    let q = query.trim().to_string();
    if q.is_empty() {
        return Ok(Vec::new());
    }
    let clip = st.ai.clip().await.map_err(|e| e.to_string())?;
    let qvec = clip.embed_text(q).await.map_err(|e| e.to_string())?;
    let limit = limit.unwrap_or(24).clamp(1, 100);
    let hits = st.vector.search_images(qvec, limit).await.map_err(|e| e.to_string())?;
    Ok(hits
        .into_iter()
        .filter(|(p, _)| std::path::Path::new(p).exists())
        .map(|(p, score)| {
            let name = std::path::Path::new(&p)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| p.clone());
            ImageHit { path: p, name, score }
        })
        .collect())
}

/// Renvoie une image en data URL (aperçu). Bornée en taille pour éviter les énormes fichiers.
#[tauri::command]
fn image_data_url(path: String) -> Result<String, String> {
    use base64::Engine;
    const MAX: u64 = 8 * 1024 * 1024; // 8 Mo
    let meta = std::fs::metadata(&path).map_err(|e| e.to_string())?;
    if meta.len() > MAX {
        return Err("image trop volumineuse pour l'aperçu".to_string());
    }
    let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    let ext = std::path::Path::new(&path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let mime = match ext.as_str() {
        "png" => "image/png",
        "gif" => "image/gif",
        "bmp" => "image/bmp",
        "webp" => "image/webp",
        _ => "image/jpeg",
    };
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok(format!("data:{mime};base64,{b64}"))
}

pub fn run() {
    // Logs structurés (remplace les println!). N'échoue pas si déjà initialisé.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                // On coupe le bruit des extracteurs PDF (glyphes manquants, etc.).
                "sensetree_lib=info,pdf_extract=error,lopdf=error,warn".into()
            }),
        )
        .try_init();

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .setup(|app| {
            let app_data_dir = app
                .path()
                .app_data_dir()
                .map_err(|e| format!("dossier AppData introuvable: {e}"))?;
            let app_config_dir = app
                .path()
                .app_config_dir()
                .unwrap_or_else(|_| app_data_dir.clone());
            let docs_dir = app
                .path()
                .document_dir()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default();

            // --- Configuration (défaut : indexe le dossier Documents) ---
            let config_path = app_config_dir.join("settings.json");
            let default_roots = if docs_dir.is_empty() {
                Vec::new()
            } else {
                vec![docs_dir]
            };
            let config = Arc::new(
                ConfigStore::load_or_init(&config_path, default_roots)
                    .map_err(|e| e.to_string())?,
            );

            // --- Base relationnelle (pool r2d2 + WAL) ---
            let db_path = app_data_dir.join("sensetree.sqlite");
            let database = Arc::new(Database::open(&db_path).map_err(|e| e.to_string())?);

            // --- Moteur IA (providers embedding / reasoning / vision) ---
            let ai = Arc::new(AiEngine::new(config.clone(), app_data_dir.clone()));

            // --- Base vectorielle (LanceDB) ---
            let lance_uri = app_data_dir.join("lancedb");
            let dims = config.snapshot().embedding.dimensions;
            let vector = tauri::async_runtime::block_on(async {
                VectorDb::open(&lance_uri.to_string_lossy(), dims).await
            })
            .map_err(|e| e.to_string())?;
            let vector = Arc::new(vector);

            // --- État partagé ---
            let app_state = Arc::new(AppState {
                db: database.clone(),
                config: config.clone(),
                ai: ai.clone(),
                vector: vector.clone(),
                data_dir: app_data_dir.clone(),
                paused: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                scanning: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
                activity: Arc::new(std::sync::Mutex::new(None)),
                scan_epoch: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                gardener: Arc::new(std::sync::Mutex::new(gardener::GardenerReport::default())),
                mcp_cache: Arc::new(std::sync::Mutex::new(None)),
            });
            app.manage(app_state.clone());

            // --- Threads de fond ---
            // (ONNX Runtime est préparé par le worker avant sa première utilisation,
            //  pour ne pas retarder l'affichage de la fenêtre au premier lancement.)
            let roots = config.snapshot().indexing.roots;
            for root in roots.clone() {
                let sc = app_state.clone();
                std::thread::spawn(move || crawler::scan_directory(sc, &root));
            }
            watchdog::start_watching(app_state.clone(), roots);
            worker::start_worker(app_state.clone());
            classifier::start_classifier(app_state.clone());
            gardener::start_gardener(app_state.clone());

            tracing::info!("✅ SenseTree prêt. DB={:?}", db_path);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_recent_activity,
            indexing_stats,
            indexing_queue,
            retry_indexing,
            ignore_indexing,
            retry_all_failed,
            set_file_summary,
            qualify_file,
            qualify_folder,
            set_indexing_paused,
            indexing_paused,
            open_path,
            gpu_available,
            get_config,
            set_config,
            get_default_prompts,
            ai_health,
            test_chat_endpoint,
            test_embedding_endpoint,
            list_installed_models,
            list_local_models,
            download_local_model,
            model_benchmarks,
            list_benchmark_boards,
            vision_benchmarks,
            reasoning_benchmarks,
            ollama_library,
            ollama_tags,
            indexing_throughput,
            reset_throughput,
            ollama_loaded,
            ollama_unload,
            list_vision_boards,
            list_reasoning_boards,
            resolve_installs,
            pull_model,
            delete_model,
            reindex_all,
            explorer::list_directory,
            explorer::get_roots,
            explorer::path_details,
            set_folder_mode,
            pick_folder,
            add_indexed_folder,
            remove_indexed_folder,
            search::semantic_search,
            search::semantic_tree,
            actions::plan_reorganization,
            actions::apply_action_plan,
            actions::discard_action_plan,
            actions::analyze_directory,
            actions::chat_with_assistant,
            gardener_health,
            agent_memory_list,
            agent_memory_delete,
            agent_memory_clear,
            index_images,
            image_search,
            image_data_url,
        ])
        .build(tauri::generate_context!())
        .expect("erreur lors du lancement de l'application Tauri")
        .run(|app_handle, event| {
            // Fermeture de l'application : on libère explicitement les ressources.
            if let tauri::RunEvent::ExitRequested { .. } = event {
                if let Some(state) = app_handle.try_state::<Arc<AppState>>() {
                    let st = state.inner().clone();
                    tauri::async_runtime::block_on(shutdown(&st));
                }
            }
        });
}

/// Arrêt propre : stoppe les traitements de fond, décharge le modèle d'embedding
/// local (session ONNX + ses threads), demande aux serveurs de modèles de libérer
/// la mémoire (Ollama garderait sinon plusieurs Go chargés après notre fermeture),
/// puis compacte le WAL SQLite.
async fn shutdown(st: &Arc<AppState>) {
    tracing::info!("🛑 fermeture : libération des ressources…");
    // 1. Faire cesser worker / crawler / classifieur (ils testent ce drapeau).
    st.paused.store(true, std::sync::atomic::Ordering::Relaxed);
    // 2. Modèle d'embedding local + reranker : détruit les sessions ORT et threads.
    st.ai.invalidate_embedder().await;
    st.ai.invalidate_reranker().await;
    // 3. Modèles hébergés (Ollama…) : ils survivent à notre processus sans ça.
    st.ai.unload_remote_models().await;
    // 4. Base : on rapatrie le WAL dans le fichier principal.
    st.db.checkpoint();
    tracing::info!("✅ ressources libérées.");
}

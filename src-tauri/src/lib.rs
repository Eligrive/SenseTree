pub mod actions;
pub mod chunker;
pub mod classifier;
pub mod config;
pub mod crawler;
pub mod db;
pub mod explorer;
pub mod folders;
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
    let roots = config.indexing.roots.clone();
    st.config.replace(config).map_err(|e| e.to_string())?;
    // Le modèle d'embedding peut avoir changé : on invalide le cache.
    st.ai.invalidate_embedder().await;
    // Re-scan des racines pour prendre en compte d'éventuels nouveaux dossiers.
    for root in roots {
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
    st.db.reset_index().map_err(|e| e.to_string())?;
    st.vector.clear().await.map_err(|e| e.to_string())?;
    for root in st.config.snapshot().indexing.roots {
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

            tracing::info!("✅ SenseTree prêt. DB={:?}", db_path);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_recent_activity,
            indexing_stats,
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
            pull_model,
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
        ])
        .run(tauri::generate_context!())
        .expect("erreur lors du lancement de l'application Tauri");
}

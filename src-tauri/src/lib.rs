pub mod actions;
pub mod chunker;
pub mod classifier;
pub mod config;
pub mod crawler;
pub mod db;
pub mod explorer;
pub mod folders;
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

/// Télécharge un modèle via l'API Ollama (POST /api/pull) en STREAMING, et émet
/// des événements `model-pull-progress` pour la barre de progression de l'UI.
#[tauri::command]
async fn pull_model(
    app: tauri::AppHandle,
    base_url: String,
    model: String,
) -> Result<String, String> {
    use futures_util::StreamExt;
    use tauri::Emitter;

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

/// Indique si le binaire a été compilé avec le support GPU (feature `cuda`).
/// L'UI s'en sert pour n'activer la case « Utiliser le GPU » que si elle a un effet.
#[tauri::command]
fn gpu_available() -> bool {
    cfg!(feature = "cuda")
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
            let ai = Arc::new(AiEngine::new(config.clone()));

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
            });
            app.manage(app_state.clone());

            // --- Threads de fond ---
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
            gpu_available,
            get_config,
            set_config,
            ai_health,
            test_chat_endpoint,
            list_installed_models,
            pull_model,
            reindex_all,
            explorer::list_directory,
            explorer::get_roots,
            set_folder_mode,
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

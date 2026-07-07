pub mod actions;
pub mod chunker;
pub mod config;
pub mod crawler;
pub mod db;
pub mod explorer;
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
        let dbc = st.db.clone();
        std::thread::spawn(move || crawler::scan_directory(dbc, &root));
    }
    Ok(())
}

#[tauri::command]
async fn ai_health(state: State<'_, Arc<AppState>>) -> Result<HealthReport, String> {
    let st = state.inner().clone();
    Ok(st.ai.health().await)
}

/// Teste un endpoint compatible OpenAI sans le sauvegarder (bouton « Tester »).
#[tauri::command]
async fn test_chat_endpoint(base_url: String, api_key: String) -> Result<String, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| e.to_string())?;
    let url = format!("{}/models", base_url.trim_end_matches('/'));
    let mut req = client.get(&url);
    if !api_key.is_empty() {
        req = req.bearer_auth(&api_key);
    }
    let resp = req.send().await.map_err(|e| format!("serveur injoignable: {e}"))?;
    if resp.status().is_success() {
        Ok("Connexion réussie".to_string())
    } else {
        Err(format!("le serveur a répondu {}", resp.status()))
    }
}

#[tauri::command]
fn get_recent_activity(state: State<'_, Arc<AppState>>) -> Result<Vec<db::FileRecord>, String> {
    state.db.get_recent_files(15).map_err(|e| e.to_string())
}

#[tauri::command]
fn indexing_stats(state: State<'_, Arc<AppState>>) -> Result<db::IndexingStats, String> {
    state.db.get_indexing_stats().map_err(|e| e.to_string())
}

// =========================================================================
// AMORÇAGE
// =========================================================================

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Logs structurés (remplace les println!). N'échoue pas si déjà initialisé.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "sensetree_lib=info,warn".into()),
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
                let dbc = database.clone();
                std::thread::spawn(move || crawler::scan_directory(dbc, &root));
            }
            watchdog::start_watching(database.clone(), roots);
            worker::start_worker(app_state.clone());

            tracing::info!("✅ SenseTree prêt. DB={:?}", db_path);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_recent_activity,
            indexing_stats,
            get_config,
            set_config,
            ai_health,
            test_chat_endpoint,
            explorer::list_directory,
            explorer::get_roots,
            search::semantic_search,
            actions::plan_reorganization,
            actions::apply_action_plan,
            actions::discard_action_plan,
            actions::analyze_directory,
            actions::chat_with_assistant,
        ])
        .run(tauri::generate_context!())
        .expect("erreur lors du lancement de l'application Tauri");
}

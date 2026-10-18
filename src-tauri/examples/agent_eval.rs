//! Banc de mesure de l'agent : exécute la boucle d'agent de l'application
//! (`actions::run_agent`) sur des tâches dont la réponse se vérifie, et écrit pour
//! chacune la réponse, le plan proposé, la conversation complète et les durées.
//!
//! ```text
//! cargo run --release --example agent_eval -- --root D:/agentbench --tasks tasks.jsonl \
//!     --out runs/agent/qwen3.5-4b.jsonl --model qwen3.5:4b --split test
//! ```
//!
//! Options : `--effort` (mode réflexion), `--temperature` (0,2 par défaut, celle de
//! l'application), `--system-extra`, `--state`, `--base-url`, `--gpu`.
//!
//! L'arborescence est indexée comme le ferait l'application (extraction PDF/DOCX,
//! découpage, contextual retrieval, LanceDB), avec l'extrait comme « sens ».

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use serde_json::json;

use sensetree_lib::actions::{run_agent, ChatTurn};
use sensetree_lib::chunker::Chunker;
use sensetree_lib::config::{ConfigStore, EmbeddingMode, ReasoningEffort};
use sensetree_lib::db::Database;
use sensetree_lib::providers::AiEngine;
use sensetree_lib::state::AppState;
use sensetree_lib::vectordb::{ChunkVector, VectorDb};
use sensetree_lib::worker::{chunk_texts, summary_of};

#[derive(Deserialize)]
struct Task {
    id: String,
    split: String,
    scope: Option<String>,
    prompt: String,
    /// Dossier personnel de l'utilisateur fictif : une tâche ne s'exécute que sur
    /// l'arborescence (et l'index) de son propriétaire.
    #[serde(default)]
    home: Option<String>,
}

fn same_path(a: &str, b: &str) -> bool {
    let n = |s: &str| s.replace('/', "\\").trim_end_matches('\\').to_lowercase();
    n(a) == n(b)
}

fn arg_map() -> Result<HashMap<String, String>> {
    let mut map = HashMap::new();
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let v = it.next().ok_or_else(|| anyhow!("valeur manquante pour {k}"))?;
        map.insert(k.trim_start_matches("--").to_string(), v);
    }
    Ok(map)
}

async fn index_tree(state: &AppState, root: &Path) -> Result<()> {
    let marker = state.data_dir.join("indexed.json");
    if marker.exists() {
        return Ok(());
    }
    let cfg = state.config.snapshot();
    let embedder = state.ai.embedder().await.map_err(|e| anyhow!(e.to_string()))?;
    state.vector.clear().await?;
    let files: Vec<PathBuf> = walkdir::WalkDir::new(root)
        .into_iter()
        .flatten()
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .collect();
    let started = Instant::now();
    for (bi, block) in files.chunks(256).enumerate() {
        let mut texts = Vec::new();
        let mut per_file = Vec::new();
        for path in block {
            let content = sensetree_lib::worker::extract_text(&path.to_string_lossy()).unwrap_or_default();
            let raw: Vec<String> = Chunker::slice_text(&content, cfg.indexing.chunk_size, cfg.indexing.overlap)
                .into_iter()
                .map(|c| c.text)
                .collect();
            if raw.is_empty() {
                continue;
            }
            let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            let (stored, embed) = chunk_texts(&name, "txt", &summary_of(&content), raw);
            texts.extend(embed);
            per_file.push((path.to_string_lossy().to_string(), stored));
        }
        let vectors = embedder.embed_documents(texts).await?;
        let mut it = vectors.into_iter();
        let rows = per_file
            .into_iter()
            .map(|(p, stored)| {
                let chunks = stored
                    .into_iter()
                    .enumerate()
                    .map(|(i, text)| ChunkVector { chunk_index: i as i32, text, vector: it.next().expect("vecteur") })
                    .collect();
                (p, chunks)
            })
            .collect();
        state.vector.append_files(rows).await?;
        eprintln!("  indexé {} / {} fichiers", ((bi + 1) * 256).min(files.len()), files.len());
    }
    std::fs::write(&marker, json!({"files": files.len(), "seconds": started.elapsed().as_secs_f64()}).to_string())?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = arg_map()?;
    let get = |k: &str| args.get(k).cloned().ok_or_else(|| anyhow!("--{k} requis"));
    let root = PathBuf::from(get("root")?);
    let tasks_path = PathBuf::from(get("tasks")?);
    let out_path = PathBuf::from(get("out")?);
    let model = get("model")?;
    let base_url = args.get("base-url").cloned().unwrap_or_else(|| "http://localhost:11434/v1".into());
    let split = args.get("split").cloned().unwrap_or_else(|| "test".into());
    let effort = args.get("effort").cloned().unwrap_or_else(|| "none".into());
    let data_dir = PathBuf::from(args.get("state").cloned().unwrap_or_else(|| "agent-state".into()));
    let gpu = args.get("gpu").map(|v| v == "true").unwrap_or(false);
    std::fs::create_dir_all(&data_dir)?;
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    sensetree_lib::ort_setup::ensure_ort(&data_dir.join("ort"), gpu)?;
    let config = Arc::new(ConfigStore::load_or_init(data_dir.join("config.json"), vec![root.to_string_lossy().to_string()])?);
    let mut cfg = config.snapshot();
    cfg.indexing.roots = vec![root.to_string_lossy().to_string()];
    cfg.embedding.mode = EmbeddingMode::Local;
    cfg.embedding.use_gpu = gpu;
    cfg.reasoning.enabled = true;
    cfg.reasoning.base_url = base_url;
    cfg.reasoning.model = model.clone();
    cfg.reasoning.reasoning_effort = serde_json::from_value::<ReasoningEffort>(json!(effort))
        .map_err(|_| anyhow!("effort inconnu : {effort}"))?;
    // Température d'échantillonnage de l'agent (0,2 par défaut dans l'application) :
    // l'apprentissage par renforcement tire plusieurs tentatives par tâche à 0,7.
    if let Some(t) = args.get("temperature") {
        cfg.agent.temperature = Some(t.parse().map_err(|_| anyhow!("température invalide : {t}"))?);
    }
    cfg.mcp_servers.clear();
    // Consignes ajoutées au prompt système, par le réglage utilisateur de l'application.
    cfg.prompts.chat_system = match args.get("system-extra") {
        Some(p) => std::fs::read_to_string(p).with_context(|| format!("lecture de {p}"))?,
        None => String::new(),
    };
    config.replace(cfg.clone())?;

    let db = Arc::new(Database::open(data_dir.join("sensetree.db"))?);
    let ai = Arc::new(AiEngine::new(config.clone(), data_dir.clone()));
    let vector = Arc::new(VectorDb::open(&data_dir.join("lancedb").to_string_lossy(), cfg.embedding.dimensions).await?);
    vector.set_bm25_analyzer(cfg.retrieval.bm25_analyzer);
    let state = Arc::new(AppState {
        db,
        config,
        ai,
        vector,
        data_dir: data_dir.clone(),
        paused: Default::default(),
        scanning: Default::default(),
        activity: Default::default(),
        scan_epoch: Default::default(),
        gardener: Default::default(),
        mcp_cache: Default::default(),
    });
    index_tree(&state, &root).await.context("indexation de l'arborescence")?;

    let done: HashSet<String> = std::fs::File::open(&out_path)
        .map(|f| {
            std::io::BufReader::new(f)
                .lines()
                .flatten()
                .filter_map(|l| serde_json::from_str::<serde_json::Value>(&l).ok())
                .filter_map(|v| v["id"].as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let tasks: Vec<Task> = std::io::BufReader::new(std::fs::File::open(&tasks_path)?)
        .lines()
        .map(|l| Ok(serde_json::from_str(&l?)?))
        .collect::<Result<Vec<Task>>>()?
        .into_iter()
        .filter(|t| t.split == split && !done.contains(&t.id))
        .filter(|t| t.home.as_deref().map_or(true, |h| same_path(h, &root.to_string_lossy())))
        .collect();
    eprintln!("{} tâches à exécuter ({model})", tasks.len());

    let mut out = std::fs::OpenOptions::new().create(true).append(true).open(&out_path)?;
    for (n, task) in tasks.iter().enumerate() {
        let started = Instant::now();
        let steps = std::sync::Mutex::new(Vec::<String>::new());
        let turn = ChatTurn { role: "user".into(), content: task.prompt.clone(), tool_messages: Vec::new() };
        let result = run_agent(&state, vec![turn], task.scope.clone(), |label| {
            if let Ok(mut s) = steps.lock() {
                s.push(label);
            }
        })
        .await;
        let elapsed = started.elapsed().as_secs_f64();
        let record = match result {
            Ok(r) => json!({
                "id": task.id,
                "model": model,
                "seconds": elapsed,
                "answer": r.answer,
                "plan": r.plan,
                "sources": r.sources.iter().map(|s| &s.path).collect::<Vec<_>>(),
                "steps": steps.into_inner().unwrap_or_default(),
                "conversation": r.conversation,
                "tools": r.tools,
            }),
            Err(e) => json!({"id": task.id, "model": model, "seconds": elapsed, "error": e}),
        };
        writeln!(out, "{record}")?;
        out.flush()?;
        eprintln!("  {} / {} ({:.0} s) {}", n + 1, tasks.len(), elapsed, task.id);
    }
    Ok(())
}

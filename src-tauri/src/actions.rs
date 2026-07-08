//! Pipeline d'action « Dry-Run » + diagnostic « gardener ».
//!
//! Aucune modification disque n'est faite sans un plan validé explicitement.
//! `plan_reorganization` produit un brouillon (JSON structuré) ; `apply_action_plan`
//! l'exécute de façon transactionnelle avec rollback ; les suppressions passent
//! par une corbeille locale (réversibles). Le gardener ne fait que diagnostiquer.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tauri::State;
use uuid::Uuid;

use crate::providers::ChatMessage;
use crate::state::AppState;

// -------------------------------------------------------------------------
// Modèle de données du plan d'action
// -------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpKind {
    Move,
    Rename,
    Delete,
    Mkdir,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    pub kind: OpKind,
    #[serde(default)]
    pub old_path: Option<String>,
    #[serde(default)]
    pub new_path: Option<String>,
    #[serde(default)]
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionPlan {
    #[serde(default)]
    pub transaction_id: Option<i64>,
    #[serde(default)]
    pub summary: String,
    pub operations: Vec<Operation>,
}

#[derive(Debug, Serialize)]
pub struct ApplyResult {
    pub applied: usize,
    pub message: String,
}

// -------------------------------------------------------------------------
// 1. Génération du plan (Dry-Run)
// -------------------------------------------------------------------------

#[tauri::command]
pub async fn plan_reorganization(
    state: State<'_, Arc<AppState>>,
    instruction: String,
    scope: String,
) -> Result<ActionPlan, String> {
    let state = state.inner().clone();

    // Contexte : fichiers du dossier + résumés sémantiques connus.
    let summaries = state.db.summaries_for_parent(&scope).unwrap_or_default();
    let listing = build_listing(&scope, &summaries);

    let system = "Tu es l'assistant de rangement de SenseTree. À partir d'une instruction \
        et d'une liste de fichiers, tu proposes un plan de réorganisation. \
        Réponds UNIQUEMENT en JSON valide, sans texte autour, au format : \
        {\"summary\": string, \"operations\": [{\"kind\": \"move|rename|delete|mkdir\", \
        \"old_path\": string|null, \"new_path\": string|null, \"reason\": string}]}. \
        Utilise des chemins absolus cohérents avec ceux fournis. \
        N'invente jamais de fichiers inexistants.";

    let user = format!(
        "Instruction: {instruction}\n\nDossier cible: {scope}\n\nFichiers:\n{listing}"
    );

    let client = state.ai.reasoning_client();
    let raw = client
        .chat(
            vec![
                ChatMessage { role: "system".into(), content: system.into() },
                ChatMessage { role: "user".into(), content: user },
            ],
            true,
        )
        .await
        .map_err(|e| format!("appel au modèle de reasoning: {e}"))?;

    let mut plan = parse_plan(&raw).map_err(|e| format!("réponse du modèle invalide: {e}"))?;

    // Garde-fou : on rejette toute opération hors des racines configurées.
    let roots = state.config.snapshot().indexing.roots;
    validate_operations(&plan.operations, &roots)?;

    // Persistance du brouillon (rien n'est écrit sur le disque à ce stade).
    let payload = serde_json::to_string(&plan).map_err(|e| e.to_string())?;
    let tx_id = state
        .db
        .record_transaction("reorganize", &payload, "draft")
        .map_err(|e| e.to_string())?;
    plan.transaction_id = Some(tx_id);

    Ok(plan)
}

// -------------------------------------------------------------------------
// 2. Exécution transactionnelle (Commit) avec rollback
// -------------------------------------------------------------------------

/// Trace d'une opération réussie, pour pouvoir revenir en arrière.
enum Done {
    Renamed { from: String, to: String },
    Created { dir: String },
    Trashed { original: String, trashed: String },
}

#[tauri::command]
pub async fn apply_action_plan(
    state: State<'_, Arc<AppState>>,
    transaction_id: i64,
) -> Result<ApplyResult, String> {
    let state = state.inner().clone();

    let tx = state
        .db
        .get_transaction(transaction_id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "transaction introuvable".to_string())?;
    if tx.status != "draft" {
        return Err(format!("transaction déjà '{}', non ré-exécutable", tx.status));
    }

    let plan: ActionPlan =
        serde_json::from_str(&tx.payload_json).map_err(|e| format!("plan corrompu: {e}"))?;

    let roots = state.config.snapshot().indexing.roots;
    validate_operations(&plan.operations, &roots)?;

    let trash_dir = state.db.path().parent().unwrap_or(Path::new(".")).join("trash");
    let mut done: Vec<Done> = Vec::new();

    // --- Phase disque : au premier échec, on annule tout ce qui a été fait. ---
    for op in &plan.operations {
        let result = execute_op(op, &trash_dir);
        match result {
            Ok(Some(d)) => done.push(d),
            Ok(None) => {}
            Err(e) => {
                let rolled = rollback(&done);
                return Err(format!(
                    "échec sur l'opération ({e}). Rollback: {rolled}. Aucune modification conservée."
                ));
            }
        }
    }

    // --- Phase index : le disque est cohérent, on synchronise DB + vecteurs. ---
    for d in &done {
        match d {
            Done::Renamed { from, to } => {
                state.vector.rename_path(from, to).await.ok();
                let _ = state.db.rename_catalog_path(from, to);
            }
            Done::Trashed { original, .. } => {
                state.vector.delete_by_path(original).await.ok();
                let _ = state.db.remove_catalog_path(original);
                let _ = state.db.remove_from_queue(original);
            }
            Done::Created { .. } => {}
        }
    }

    state
        .db
        .mark_transaction_committed(transaction_id)
        .map_err(|e| e.to_string())?;

    Ok(ApplyResult {
        applied: done.len(),
        message: format!("{} opération(s) appliquée(s).", done.len()),
    })
}

#[tauri::command]
pub async fn discard_action_plan(
    state: State<'_, Arc<AppState>>,
    transaction_id: i64,
) -> Result<(), String> {
    let state = state.inner().clone();
    state
        .db
        .mark_transaction_discarded(transaction_id)
        .map_err(|e| e.to_string())
}

fn execute_op(op: &Operation, trash_dir: &Path) -> Result<Option<Done>, String> {
    match op.kind {
        OpKind::Mkdir => {
            let dir = op.new_path.as_ref().ok_or("mkdir sans new_path")?;
            fs::create_dir_all(dir).map_err(|e| format!("mkdir {dir}: {e}"))?;
            Ok(Some(Done::Created { dir: dir.clone() }))
        }
        OpKind::Move | OpKind::Rename => {
            let from = op.old_path.as_ref().ok_or("move/rename sans old_path")?;
            let to = op.new_path.as_ref().ok_or("move/rename sans new_path")?;
            if let Some(parent) = Path::new(to).parent() {
                fs::create_dir_all(parent).map_err(|e| format!("création parent: {e}"))?;
            }
            fs::rename(from, to).map_err(|e| format!("déplacement {from} → {to}: {e}"))?;
            Ok(Some(Done::Renamed { from: from.clone(), to: to.clone() }))
        }
        OpKind::Delete => {
            let target = op.old_path.as_ref().ok_or("delete sans old_path")?;
            fs::create_dir_all(trash_dir).map_err(|e| format!("corbeille: {e}"))?;
            let name = Path::new(target)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "fichier".into());
            let trashed = trash_dir.join(format!("{}__{}", Uuid::new_v4(), name));
            fs::rename(target, &trashed)
                .map_err(|e| format!("mise en corbeille {target}: {e}"))?;
            Ok(Some(Done::Trashed {
                original: target.clone(),
                trashed: trashed.to_string_lossy().to_string(),
            }))
        }
    }
}

/// Annule en ordre inverse les opérations déjà réalisées. Renvoie un résumé.
fn rollback(done: &[Done]) -> String {
    let mut failures = 0;
    for d in done.iter().rev() {
        let res = match d {
            Done::Renamed { from, to } => fs::rename(to, from),
            Done::Trashed { original, trashed } => fs::rename(trashed, original),
            Done::Created { dir } => fs::remove_dir(dir), // seulement s'il est vide
        };
        if res.is_err() {
            failures += 1;
        }
    }
    if failures == 0 {
        "complet".to_string()
    } else {
        format!("{failures} opération(s) non annulée(s)")
    }
}

/// Normalise un chemin pour comparaison : backslashes multiples réduits,
/// séparateurs unifiés, casse ignorée (Windows), slash final retiré.
fn normalize_path(p: &str) -> String {
    let mut out = String::with_capacity(p.len());
    let mut prev_sep = false;
    for c in p.chars() {
        if c == '\\' || c == '/' {
            if !prev_sep {
                out.push('\\');
            }
            prev_sep = true;
        } else {
            out.push(c);
            prev_sep = false;
        }
    }
    out.trim_end_matches('\\').to_lowercase()
}

/// Rejette tout plan qui sortirait des racines autorisées (protection anti-évasion).
fn validate_operations(ops: &[Operation], roots: &[String]) -> Result<(), String> {
    if roots.is_empty() {
        return Ok(());
    }
    let norm_roots: Vec<String> = roots.iter().map(|r| normalize_path(r)).collect();
    let within = |path: &str| {
        let np = normalize_path(path);
        norm_roots.iter().any(|r| np.starts_with(r.as_str()))
    };
    for op in ops {
        if let Some(p) = &op.old_path {
            if !within(p) {
                return Err(format!("chemin hors périmètre autorisé: {p}"));
            }
        }
        if let Some(p) = &op.new_path {
            if !within(p) {
                return Err(format!("chemin hors périmètre autorisé: {p}"));
            }
        }
    }
    Ok(())
}

fn parse_plan(raw: &str) -> anyhow::Result<ActionPlan> {
    // Certains modèles encadrent le JSON de ``` ou de texte : on isole l'objet.
    let start = raw.find('{');
    let end = raw.rfind('}');
    let slice = match (start, end) {
        (Some(s), Some(e)) if e > s => &raw[s..=e],
        _ => raw,
    };
    Ok(serde_json::from_str::<ActionPlan>(slice)?)
}

fn build_listing(scope: &str, summaries: &[(String, String)]) -> String {
    use std::collections::HashMap;
    let map: HashMap<&str, &str> = summaries
        .iter()
        .map(|(p, s)| (p.as_str(), s.as_str()))
        .collect();

    let mut lines = Vec::new();
    if let Ok(read) = fs::read_dir(scope) {
        for entry in read.flatten() {
            let path = entry.path();
            let is_dir = path.is_dir();
            let path_str = path.to_string_lossy().to_string();
            let summary = map.get(path_str.as_str()).copied().unwrap_or("");
            let tag = if is_dir { "[dossier]" } else { "[fichier]" };
            if summary.is_empty() {
                lines.push(format!("- {tag} {path_str}"));
            } else {
                lines.push(format!("- {tag} {path_str} — {summary}"));
            }
        }
    }
    lines.join("\n")
}

// -------------------------------------------------------------------------
// 3. Gardener : diagnostic structurel (lecture seule)
// -------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct DirectoryReport {
    pub scanned_path: String,
    pub file_count: usize,
    pub max_depth: usize,
    pub empty_dirs: Vec<String>,
    pub duplicate_groups: Vec<crate::db::DuplicateGroup>,
    pub cluttered: bool,
    pub suggestions: Vec<String>,
}

#[tauri::command]
pub async fn analyze_directory(
    state: State<'_, Arc<AppState>>,
    path: String,
) -> Result<DirectoryReport, String> {
    let state = state.inner().clone();

    let root = PathBuf::from(&path);
    let mut file_count = 0usize;
    let mut direct_files = 0usize;
    let mut max_depth = 0usize;
    let mut empty_dirs = Vec::new();

    for entry in walkdir::WalkDir::new(&root).into_iter().flatten() {
        let depth = entry.depth();
        if entry.file_type().is_file() {
            file_count += 1;
            if depth == 1 {
                direct_files += 1;
            }
            max_depth = max_depth.max(depth);
        } else if entry.file_type().is_dir() && depth > 0 {
            let is_empty = fs::read_dir(entry.path())
                .map(|mut d| d.next().is_none())
                .unwrap_or(false);
            if is_empty {
                empty_dirs.push(entry.path().to_string_lossy().to_string());
            }
        }
    }

    let duplicate_groups = state.db.find_duplicates(&path).unwrap_or_default();
    let cluttered = direct_files > 40;

    let mut suggestions = Vec::new();
    if cluttered {
        suggestions.push(format!(
            "Ce dossier contient {direct_files} fichiers en vrac : envisagez des sous-dossiers thématiques."
        ));
    }
    if max_depth > 6 {
        suggestions.push(format!(
            "Arborescence profonde (niveau {max_depth}) : elle pourrait être aplatie."
        ));
    }
    if !empty_dirs.is_empty() {
        suggestions.push(format!("{} dossier(s) vide(s) à supprimer.", empty_dirs.len()));
    }
    if !duplicate_groups.is_empty() {
        let total: usize = duplicate_groups.iter().map(|g| g.paths.len()).sum();
        suggestions.push(format!(
            "{} fichier(s) en doublon exact détecté(s) sur {} groupe(s).",
            total,
            duplicate_groups.len()
        ));
    }
    if suggestions.is_empty() {
        suggestions.push("Aucune anomalie majeure détectée. 🌱".to_string());
    }

    Ok(DirectoryReport {
        scanned_path: path,
        file_count,
        max_depth,
        empty_dirs,
        duplicate_groups,
        cluttered,
        suggestions,
    })
}

// -------------------------------------------------------------------------
// 4. Chat conversationnel simple (sans action)
// -------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatSource {
    pub path: String,
    pub name: String,
    pub score: f32,
    pub snippet: String,
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub answer: String,
    pub sources: Vec<ChatSource>,
}

#[tauri::command]
pub async fn chat_with_assistant(
    state: State<'_, Arc<AppState>>,
    messages: Vec<ChatTurn>,
    scope: Option<String>,
) -> Result<ChatResponse, String> {
    use std::collections::HashMap;
    let state = state.inner().clone();

    // La requête = dernier message utilisateur.
    let query = messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();

    // --- RAG : récupère les extraits de fichiers pertinents (recherche globale) ---
    let mut sources: Vec<ChatSource> = Vec::new();
    if !query.trim().is_empty() {
        if let Ok(embedder) = state.ai.embedder().await {
            if let Ok(qvec) = embedder.embed_query(query.clone()).await {
                if let Ok(hits) = state.vector.search(qvec, 24, None).await {
                    let mut best: HashMap<String, crate::vectordb::SearchHit> = HashMap::new();
                    for h in hits {
                        if !std::path::Path::new(&h.path).exists() {
                            continue;
                        }
                        best.entry(h.path.clone())
                            .and_modify(|e| {
                                if h.score > e.score {
                                    *e = h.clone();
                                }
                            })
                            .or_insert(h);
                    }
                    let mut v: Vec<_> = best.into_values().collect();
                    v.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                    v.truncate(6);
                    sources = v
                        .into_iter()
                        .map(|h| ChatSource {
                            name: std::path::Path::new(&h.path)
                                .file_name()
                                .map(|n| n.to_string_lossy().to_string())
                                .unwrap_or_else(|| h.path.clone()),
                            path: h.path,
                            score: h.score,
                            snippet: h.snippet,
                        })
                        .collect();
                }
            }
        }
    }

    // --- Prompt système avec le contexte récupéré (RAG) ---
    let mut system = "Tu es l'assistant de SenseTree, un explorateur de fichiers sémantique local. \
        Réponds à la question de l'utilisateur en t'appuyant sur les EXTRAITS DE FICHIERS fournis \
        ci-dessous. Cite les fichiers pertinents par leur nom. Si l'information ne figure pas dans \
        les extraits, dis-le clairement plutôt que d'inventer. Sois concis et concret."
        .to_string();

    if !sources.is_empty() {
        let ctx: String = sources
            .iter()
            .enumerate()
            .map(|(i, s)| format!("[{}] {} — {}\n{}", i + 1, s.name, s.path, s.snippet))
            .collect::<Vec<_>>()
            .join("\n\n");
        system.push_str(&format!("\n\nExtraits de fichiers pertinents :\n{ctx}"));
    } else if let Some(scope) = &scope {
        let summaries = state.db.summaries_for_parent(scope).unwrap_or_default();
        if !summaries.is_empty() {
            let ctx: String = summaries
                .iter()
                .take(30)
                .map(|(p, s)| format!("- {p}: {s}"))
                .collect::<Vec<_>>()
                .join("\n");
            system.push_str(&format!("\n\nContexte du dossier {scope} :\n{ctx}"));
        }
    }

    let mut chat_messages = vec![ChatMessage { role: "system".into(), content: system }];
    for m in messages {
        chat_messages.push(ChatMessage { role: m.role, content: m.content });
    }

    let answer = state
        .ai
        .reasoning_client()
        .chat(chat_messages, false)
        .await
        .map_err(|e| e.to_string())?;
    Ok(ChatResponse { answer, sources })
}

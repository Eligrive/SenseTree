//! Pipeline d'action « Dry-Run » + diagnostic « gardener ».
//!
//! Aucune modification disque n'est faite sans un plan validé explicitement.
//! `plan_reorganization` produit un brouillon (JSON structuré) ; `apply_action_plan`
//! l'exécute de façon transactionnelle avec rollback ; les suppressions passent
//! par une corbeille locale (réversibles). Le gardener ne fait que diagnostiquer.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::json;
use tauri::{Emitter, State};
use uuid::Uuid;

use crate::config::McpServerConfig;
use crate::providers::{ChatMessage, ToolCallOut};
use crate::state::AppState;

// -------------------------------------------------------------------------
// Modèle de données du plan d'action
// -------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpKind {
    Move,
    Rename,
    Delete,
    Mkdir,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

    let cfg = state.config.snapshot();
    let system = crate::config::prompt_or(
        &cfg.prompts.reorganize,
        crate::config::default_prompts::REORGANIZE,
    );

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
    operations: Option<Vec<Operation>>,
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

    // L'utilisateur peut avoir DÉCOCHÉ des opérations avant d'appliquer. On n'accepte
    // que des opérations RÉELLEMENT présentes dans le plan validé (anti-injection).
    let ops: Vec<Operation> = match operations {
        Some(edited) => {
            if edited.is_empty() {
                return Err("aucune opération sélectionnée".to_string());
            }
            for op in &edited {
                if !plan.operations.contains(op) {
                    return Err("opération absente du plan d'origine : refusée".to_string());
                }
            }
            edited
        }
        None => plan.operations.clone(),
    };

    let roots = state.config.snapshot().indexing.roots;
    validate_operations(&ops, &roots)?;

    let trash_dir = state.db.path().parent().unwrap_or(Path::new(".")).join("trash");
    let mut done: Vec<Done> = Vec::new();

    // --- Phase disque : au premier échec, on annule tout ce qui a été fait. ---
    for op in &ops {
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

/// Vrai si `path` est À L'INTÉRIEUR d'une des racines configurées, **frontière de
/// segment respectée** : la racine `…\Docs` ne doit PAS matcher `…\DocsEvil`. Sans
/// racine configurée, on n'impose aucune restriction (comportement historique).
///
/// Note : comparaison textuelle (pas de résolution de symlink). Un symlink placé
/// à l'intérieur d'une racine et pointant dehors n'est pas détecté — risque résiduel
/// faible sur une app locale mono-utilisateur (l'attaquant devrait déjà avoir un accès disque).
fn path_within_roots(path: &str, roots: &[String]) -> bool {
    if roots.is_empty() {
        return true;
    }
    let np = normalize_path(path);
    roots.iter().any(|r| {
        let nr = normalize_path(r);
        !nr.is_empty() && (np == nr || np.starts_with(&format!("{nr}\\")))
    })
}

/// Rejette tout plan qui sortirait des racines autorisées (protection anti-évasion).
fn validate_operations(ops: &[Operation], roots: &[String]) -> Result<(), String> {
    for op in ops {
        if let Some(p) = &op.old_path {
            if !path_within_roots(p, roots) {
                return Err(format!("chemin hors périmètre autorisé: {p}"));
            }
        }
        if let Some(p) = &op.new_path {
            if !path_within_roots(p, roots) {
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
    /// Réponse textuelle (si l'assistant répond au lieu d'agir).
    pub answer: Option<String>,
    pub sources: Vec<ChatSource>,
    /// Plan d'action Dry-Run (si l'assistant propose une action).
    pub plan: Option<ActionPlan>,
}

/// Arborescence d'un dossier (2 niveaux, bornée) avec chemins exacts — pour que
/// le LLM raisonne de façon STRUCTURELLE (réorganisation), pas seulement sémantique.
fn folder_structure(root: &str, max_depth: usize, budget: usize) -> String {
    let mut out: Vec<String> = Vec::new();
    for entry in walkdir::WalkDir::new(root)
        .max_depth(max_depth)
        .into_iter()
        .flatten()
    {
        if out.len() >= budget {
            break;
        }
        let depth = entry.depth();
        if depth == 0 {
            continue;
        }
        let name = entry.file_name().to_string_lossy();
        if name.starts_with('.') {
            continue;
        }
        let is_dir = entry.file_type().is_dir();
        let pad = "  ".repeat(depth.saturating_sub(1));
        out.push(format!(
            "{pad}{} {}",
            if is_dir { "[D]" } else { "[F]" },
            entry.path().to_string_lossy()
        ));
    }
    out.join("\n")
}

/// Boucle d'agent : nombre maximal d'allers-retours d'outils avant d'exiger une réponse.
const MAX_AGENT_ROUNDS: usize = 5;

/// Schémas des outils exposés au modèle (format function-calling OpenAI). L'agent
/// s'en sert pour CHERCHER, LIRE, LISTER et PROPOSER des actions (Dry-Run).
fn tool_schemas() -> serde_json::Value {
    json!([
        {"type":"function","function":{
            "name":"search_files",
            "description":"Recherche sémantique HYBRIDE (sens + mots-clés) dans les fichiers indexés de l'utilisateur. Renvoie les fichiers les plus pertinents avec un extrait. À utiliser pour retrouver des documents.",
            "parameters":{"type":"object","properties":{
                "query":{"type":"string","description":"La requête (langage naturel ou mots-clés)."},
                "scope":{"type":"string","description":"Chemin d'un dossier pour restreindre la recherche (optionnel)."}
            },"required":["query"]}
        }},
        {"type":"function","function":{
            "name":"read_file",
            "description":"Lit le contenu texte d'un fichier (borné à ~4000 caractères). Utile pour examiner un fichier trouvé avant de répondre ou d'agir.",
            "parameters":{"type":"object","properties":{
                "path":{"type":"string","description":"Chemin absolu exact du fichier."}
            },"required":["path"]}
        }},
        {"type":"function","function":{
            "name":"list_directory",
            "description":"Liste le contenu d'un dossier (fichiers et sous-dossiers, chemins exacts).",
            "parameters":{"type":"object","properties":{
                "path":{"type":"string","description":"Chemin absolu du dossier."}
            },"required":["path"]}
        }},
        {"type":"function","function":{
            "name":"propose_actions",
            "description":"Propose un plan d'actions sur les fichiers (déplacer/renommer/supprimer/créer un dossier). N'EXÉCUTE RIEN : l'utilisateur validera. Utilise UNIQUEMENT des chemins exacts et existants.",
            "parameters":{"type":"object","properties":{
                "summary":{"type":"string","description":"Résumé en une phrase du plan."},
                "operations":{"type":"array","items":{"type":"object","properties":{
                    "kind":{"type":"string","enum":["move","rename","delete","mkdir"]},
                    "old_path":{"type":"string"},
                    "new_path":{"type":"string"},
                    "reason":{"type":"string"}
                },"required":["kind"]}}
            },"required":["operations"]}
        }}
    ])
}

/// Normalise un nom d'outil pour le function-calling (^[A-Za-z0-9_-]{1,64}$).
fn sanitize_tool_name(s: &str) -> String {
    let mut out: String = s
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect();
    out.truncate(64);
    out
}

/// Lit un fichier texte, borné, en UTF-8 tolérant (best-effort pour un binaire).
fn read_file_bounded(path: &str) -> String {
    if path.trim().is_empty() {
        return "Chemin manquant.".to_string();
    }
    match std::fs::read(path) {
        Ok(bytes) => {
            let text = String::from_utf8_lossy(&bytes);
            let clipped: String = text.chars().take(4000).collect();
            if clipped.trim().is_empty() {
                "(fichier binaire ou vide : aucun texte lisible)".to_string()
            } else {
                clipped
            }
        }
        Err(e) => format!("Impossible de lire {path} : {e}"),
    }
}

/// Liste compacte d'un dossier (bornée).
fn list_dir_compact(path: &str) -> String {
    match std::fs::read_dir(path) {
        Ok(rd) => {
            let mut lines = Vec::new();
            for e in rd.flatten().take(200) {
                let p = e.path();
                let tag = if p.is_dir() { "[D]" } else { "[F]" };
                lines.push(format!("{tag} {}", p.to_string_lossy()));
            }
            if lines.is_empty() {
                "(dossier vide)".to_string()
            } else {
                lines.join("\n")
            }
        }
        Err(e) => format!("Impossible de lister {path} : {e}"),
    }
}

/// Exécute UN appel d'outil et renvoie l'observation (texte) à réinjecter au modèle.
/// Accumule au passage les `sources` (recherches) et le `plan` (propose_actions).
async fn execute_tool(
    state: &Arc<AppState>,
    tc: &ToolCallOut,
    scope: &Option<String>,
    sources: &mut Vec<ChatSource>,
    plan: &mut Option<ActionPlan>,
    mcp_index: &HashMap<String, (McpServerConfig, String)>,
) -> String {
    let args: serde_json::Value = serde_json::from_str(&tc.arguments).unwrap_or_else(|_| json!({}));

    // Outil EXTERNE fourni par un serveur MCP ?
    if let Some((srv, orig)) = mcp_index.get(&tc.name) {
        return match crate::mcp::call_tool(srv, orig, args).await {
            Ok(txt) => txt,
            Err(e) => format!("Erreur outil MCP {} : {e}", srv.name),
        };
    }

    match tc.name.as_str() {
        "search_files" => {
            let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
            let sc = args
                .get("scope")
                .and_then(|v| v.as_str())
                .filter(|s| !s.trim().is_empty())
                .map(str::to_string)
                .or_else(|| scope.clone());
            match crate::search::run_semantic_search(state, query, sc.as_deref(), 8).await {
                Ok(res) => {
                    for r in &res {
                        if !sources.iter().any(|s| s.path == r.path) {
                            sources.push(ChatSource {
                                path: r.path.clone(),
                                name: r.name.clone(),
                                score: r.score,
                                snippet: r.snippet.clone(),
                            });
                        }
                    }
                    let items: Vec<_> = res
                        .iter()
                        .map(|r| json!({"path": r.path, "name": r.name, "score": r.score, "snippet": r.snippet}))
                        .collect();
                    serde_json::to_string(&items).unwrap_or_else(|_| "[]".to_string())
                }
                Err(e) => format!("Erreur de recherche : {e}"),
            }
        }
        "read_file" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let roots = state.config.snapshot().indexing.roots;
            if !path_within_roots(path, &roots) {
                return format!("Accès refusé : « {path} » est hors des dossiers indexés.");
            }
            read_file_bounded(path)
        }
        "list_directory" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let roots = state.config.snapshot().indexing.roots;
            if !path_within_roots(path, &roots) {
                return format!("Accès refusé : « {path} » est hors des dossiers indexés.");
            }
            list_dir_compact(path)
        }
        "propose_actions" => {
            let ops: Vec<Operation> =
                serde_json::from_value(args.get("operations").cloned().unwrap_or_else(|| json!([])))
                    .unwrap_or_default();
            if ops.is_empty() {
                return "Aucune opération valide fournie.".to_string();
            }
            let roots = state.config.snapshot().indexing.roots;
            if let Err(e) = validate_operations(&ops, &roots) {
                return format!("Refusé (hors périmètre indexé) : {e}");
            }
            let summary = args
                .get("summary")
                .and_then(|v| v.as_str())
                .filter(|s| !s.trim().is_empty())
                .unwrap_or("Plan d'action proposé")
                .to_string();
            let mut p = ActionPlan { transaction_id: None, summary, operations: ops };
            let payload = match serde_json::to_string(&p) {
                Ok(s) => s,
                Err(e) => return format!("Erreur de sérialisation du plan : {e}"),
            };
            match state.db.record_transaction("reorganize", &payload, "draft") {
                Ok(tx) => {
                    p.transaction_id = Some(tx);
                    let n = p.operations.len();
                    *plan = Some(p);
                    format!("Plan de {n} opération(s) proposé à l'utilisateur pour validation. N'ajoute rien d'autre.")
                }
                Err(e) => format!("Erreur d'enregistrement du plan : {e}"),
            }
        }
        other => format!("Outil inconnu : {other}"),
    }
}

/// Étape de travail de l'agent, poussée en direct à l'UI (événement `agent://step`).
#[derive(Clone, Serialize)]
struct AgentStep {
    label: String,
}

/// Libellé lisible d'un appel d'outil (pour la trace live du chat).
fn describe_tool_call(
    tc: &ToolCallOut,
    mcp_index: &HashMap<String, (McpServerConfig, String)>,
) -> String {
    if let Some((srv, orig)) = mcp_index.get(&tc.name) {
        return format!("🔌 {} · {}", srv.name, orig);
    }
    let args: serde_json::Value = serde_json::from_str(&tc.arguments).unwrap_or_else(|_| json!({}));
    let base = |p: &str| {
        Path::new(p)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| p.to_string())
    };
    match tc.name.as_str() {
        "search_files" => format!(
            "🔍 Recherche : {}",
            args.get("query").and_then(|v| v.as_str()).unwrap_or("").chars().take(60).collect::<String>()
        ),
        "read_file" => format!("📄 Lecture : {}", base(args.get("path").and_then(|v| v.as_str()).unwrap_or(""))),
        "list_directory" => {
            format!("📂 Exploration : {}", base(args.get("path").and_then(|v| v.as_str()).unwrap_or("")))
        }
        "propose_actions" => "🛠️ Préparation d'un plan d'action".to_string(),
        other => format!("🔧 {other}"),
    }
}

/// Découvre les outils MCP des serveurs activés, avec cache (TTL) pour éviter un
/// handshake à chaque message. Renvoie (schémas function-calling, index de routage).
async fn discover_mcp_tools(
    state: &Arc<AppState>,
    servers: &[McpServerConfig],
) -> (Vec<serde_json::Value>, HashMap<String, (McpServerConfig, String)>) {
    let active: Vec<&McpServerConfig> = servers
        .iter()
        .filter(|s| s.enabled && (!s.url.trim().is_empty() || !s.command.trim().is_empty()))
        .collect();
    if active.is_empty() {
        return (Vec::new(), HashMap::new());
    }
    // Signature de la config : toute modification invalide le cache immédiatement.
    let key = active
        .iter()
        .map(|s| format!("{}|{}|{}|{}|{}", s.name, s.url, s.auth, s.command, s.args.join("\u{1f}")))
        .collect::<Vec<_>>()
        .join(";;");

    if let Ok(guard) = state.mcp_cache.lock() {
        if let Some(d) = guard.as_ref() {
            if d.key == key && d.at.elapsed() < crate::mcp::DISCOVERY_TTL {
                return (d.tools_schema.clone(), d.index.clone());
            }
        }
    }

    let mut tools_schema: Vec<serde_json::Value> = Vec::new();
    let mut index: HashMap<String, (McpServerConfig, String)> = HashMap::new();
    for srv in &active {
        match crate::mcp::list_tools(srv).await {
            Ok(list) => {
                for t in list {
                    let fname = sanitize_tool_name(&format!("mcp__{}__{}", srv.name, t.name));
                    tools_schema.push(json!({"type": "function", "function": {
                        "name": fname,
                        "description": format!("[{}] {}", srv.name, t.description),
                        "parameters": t.input_schema,
                    }}));
                    index.insert(fname, ((*srv).clone(), t.name.clone()));
                }
                tracing::info!("MCP {} : {} outil(s) exposé(s)", srv.name, index.len());
            }
            Err(e) => tracing::warn!("MCP {} indisponible : {e}", srv.name),
        }
    }

    if let Ok(mut guard) = state.mcp_cache.lock() {
        *guard = Some(crate::mcp::McpDiscovery {
            key,
            at: Instant::now(),
            tools_schema: tools_schema.clone(),
            index: index.clone(),
        });
    }
    (tools_schema, index)
}

#[tauri::command]
pub async fn chat_with_assistant(
    app: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
    messages: Vec<ChatTurn>,
    scope: Option<String>,
) -> Result<ChatResponse, String> {
    let state = state.inner().clone();

    // Requête = dernier message utilisateur (pour le pré-RAG d'amorçage).
    let query = messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();

    // --- Pré-RAG (recherche HYBRIDE + rerank) : amorce le contexte. Bénéficie aussi
    //     aux modèles incapables d'appeler des outils (dégradation douce). ---
    let mut sources: Vec<ChatSource> = Vec::new();
    if !query.trim().is_empty() {
        if let Ok(res) =
            crate::search::run_semantic_search(&state, &query, scope.as_deref(), 6).await
        {
            sources = res
                .into_iter()
                .map(|r| ChatSource {
                    path: r.path,
                    name: r.name,
                    score: r.score,
                    snippet: r.snippet,
                })
                .collect();
        }
    }

    // --- Prompt système : instructions d'AGENT + contexte amorcé. ---
    let cfg = state.config.snapshot();
    let mut system = String::from(
        "Tu es l'assistant de SenseTree, un explorateur de fichiers sémantique local. \
Tu disposes d'OUTILS pour explorer les fichiers de l'utilisateur : search_files (recherche par \
le sens et les mots-clés), read_file (lire un fichier), list_directory (lister un dossier). \
Utilise-les autant que nécessaire — cherche, lis, recoupe — AVANT de répondre, et cite les \
fichiers pertinents par leur nom. Pour AGIR sur des fichiers (déplacer, renommer, supprimer, \
créer un dossier), appelle propose_actions avec des chemins EXACTS et existants : rien n'est \
exécuté sans validation de l'utilisateur. Réponds en français, de façon concise et factuelle.",
    );
    let user_override = cfg.prompts.chat_system.trim();
    if !user_override.is_empty() {
        system.push_str("\n\nConsignes supplémentaires :\n");
        system.push_str(user_override);
    }
    if !sources.is_empty() {
        let ctx: String = sources
            .iter()
            .enumerate()
            .map(|(i, s)| format!("[{}] {} — {}\n{}", i + 1, s.name, s.path, s.snippet))
            .collect::<Vec<_>>()
            .join("\n\n");
        system.push_str(&format!(
            "\n\nExtraits déjà trouvés (tu peux en chercher d'autres via search_files) :\n{ctx}"
        ));
    }
    if let Some(scope) = &scope {
        let structure = folder_structure(scope, 2, 200);
        if !structure.is_empty() {
            system.push_str(&format!(
                "\n\nStructure du dossier courant ({scope}) — [D]=dossier, [F]=fichier, chemins exacts :\n{structure}"
            ));
        }
    }

    // --- Historique converti en messages OpenAI (JSON). ---
    let mut msgs: Vec<serde_json::Value> = vec![json!({"role": "system", "content": system})];
    for m in &messages {
        msgs.push(json!({"role": m.role, "content": m.content}));
    }

    // --- Outils : intégrés + outils EXTERNES des serveurs MCP activés (via cache de
    //     découverte ; best-effort : un serveur injoignable est ignoré). ---
    let mut tools = tool_schemas();
    let (mcp_schemas, mcp_index) = discover_mcp_tools(&state, &cfg.mcp_servers).await;
    if let Some(arr) = tools.as_array_mut() {
        arr.extend(mcp_schemas);
    }

    // --- Boucle ReAct : le modèle appelle des outils, observe, itère, puis répond. ---
    let client = state.ai.reasoning_client();
    let mut plan: Option<ActionPlan> = None;
    let mut answer: Option<String> = None;

    for _round in 0..MAX_AGENT_ROUNDS {
        let turn = match client.chat_tools(&msgs, &tools).await {
            Ok(t) => t,
            Err(e) => {
                return Ok(ChatResponse {
                    answer: Some(format!("Erreur du modèle de reasoning : {e}")),
                    sources,
                    plan: None,
                })
            }
        };

        if turn.tool_calls.is_empty() {
            answer = turn.content;
            break;
        }

        // Réinjecte le message assistant (avec ses tool_calls) puis chaque observation.
        msgs.push(turn.raw_message.clone());
        for tc in &turn.tool_calls {
            // Trace live : on annonce l'action AVANT de l'exécuter.
            let _ = app.emit("agent-step", AgentStep { label: describe_tool_call(tc, &mcp_index) });
            let observation =
                execute_tool(&state, tc, &scope, &mut sources, &mut plan, &mcp_index).await;
            msgs.push(json!({"role": "tool", "tool_call_id": tc.id, "content": observation}));
        }

        // Un plan a été proposé → on rend la main à l'utilisateur (validation Dry-Run).
        if plan.is_some() {
            break;
        }
    }

    if plan.is_some() {
        return Ok(ChatResponse { answer: None, sources, plan });
    }
    Ok(ChatResponse {
        answer: Some(answer.unwrap_or_else(|| {
            "Je n'ai pas abouti à une réponse. Reformule ou précise ta demande.".to_string()
        })),
        sources,
        plan: None,
    })
}

#[cfg(test)]
mod tests {
    use super::{describe_tool_call, path_within_roots, read_file_bounded, tool_schemas};
    use crate::providers::ToolCallOut;

    #[test]
    fn describe_tool_call_produit_des_libelles_lisibles() {
        let idx = std::collections::HashMap::new();
        let search = ToolCallOut {
            id: "1".into(),
            name: "search_files".into(),
            arguments: r#"{"query":"factures 2024"}"#.into(),
        };
        assert!(describe_tool_call(&search, &idx).contains("Recherche"));
        let read = ToolCallOut {
            id: "2".into(),
            name: "read_file".into(),
            arguments: r#"{"path":"C:/docs/rapport.pdf"}"#.into(),
        };
        // Le libellé montre le nom de fichier, pas le chemin complet.
        assert!(describe_tool_call(&read, &idx).contains("rapport.pdf"));
    }

    #[test]
    fn within_roots_respecte_la_frontiere_de_segment() {
        let roots = vec![r"C:\Users\virgi\Docs".to_string()];
        // À l'intérieur, ou la racine elle-même.
        assert!(path_within_roots(r"C:\Users\virgi\Docs\a\b.txt", &roots));
        assert!(path_within_roots(r"C:\Users\virgi\Docs", &roots));
        // Insensible casse / séparateur.
        assert!(path_within_roots(r"c:/users/virgi/docs/x", &roots));
        // FAILLE corrigée : un voisin au nom préfixe NE DOIT PAS passer.
        assert!(!path_within_roots(r"C:\Users\virgi\DocsEvil\secret", &roots));
        // Hors périmètre franc (exfiltration).
        assert!(!path_within_roots(r"C:\Users\virgi\.ssh\id_rsa", &roots));
        // Aucune racine → pas de restriction (comportement historique).
        assert!(path_within_roots(r"C:\nimporte\ou", &[]));
    }

    #[test]
    fn tool_schemas_exposent_les_quatre_outils() {
        let tools = tool_schemas();
        let arr = tools.as_array().expect("tableau d'outils");
        assert_eq!(arr.len(), 4);
        let names: Vec<&str> = arr
            .iter()
            .filter_map(|t| t.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()))
            .collect();
        for expected in ["search_files", "read_file", "list_directory", "propose_actions"] {
            assert!(names.contains(&expected), "outil manquant : {expected}");
        }
    }

    #[test]
    fn read_file_bounded_gere_chemin_vide_et_inexistant() {
        assert_eq!(read_file_bounded("   "), "Chemin manquant.");
        // Un chemin inexistant renvoie un message d'erreur, jamais un panic.
        assert!(read_file_bounded("Z:/nexiste/pas/vraiment.txt").starts_with("Impossible de lire"));
    }
}

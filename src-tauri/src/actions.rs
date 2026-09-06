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
    /// Corrige le « sens extrait » d'un fichier (couche sémantique, pas le disque).
    ///
    /// Le fichier n'est PAS touché : seule la description stockée change. Comme tout
    /// le reste, ça passe par le plan Dry-Run — l'utilisateur voit l'avant/après et
    /// valide, car une qualification erronée écrase une qualification correcte aussi
    /// facilement que l'inverse.
    Requalify,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Operation {
    pub kind: OpKind,
    #[serde(default)]
    pub old_path: Option<String>,
    #[serde(default)]
    pub new_path: Option<String>,
    /// Nouveau « sens » pour une opération `requalify`. Ignoré pour les autres.
    #[serde(default)]
    pub new_summary: Option<String>,
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

    // --- Phase sémantique : corrections de « sens », après un disque cohérent. ---
    //
    // Le sens n'est pas qu'un affichage : il est EMBARQUÉ dans les vecteurs (premier
    // chunk + préfixe de contextual retrieval). Le corriger en base sans ré-embedder
    // laisserait la recherche matcher sur l'ancienne description, fausse. On épingle
    // donc la correction puis on remet le fichier en file : il sera ré-extrait et
    // ré-embeddé, et la qualification LLM respectera le sens épinglé au lieu de le
    // régénérer.
    let mut requalifies = 0usize;
    let mut anciens: Vec<(String, Option<String>)> = Vec::new();
    for op in ops.iter().filter(|o| o.kind == OpKind::Requalify) {
        let path = match op.old_path.as_ref() {
            Some(p) => p,
            None => continue,
        };
        let nouveau = op.new_summary.as_deref().unwrap_or("").trim();
        if nouveau.is_empty() {
            continue;
        }
        let ancien = state.db.get_file_semantics(path).ok().flatten().map(|(s, _, _)| s);
        if let Err(e) = state.db.set_file_summary(path, nouveau) {
            // Restaure ce qui a déjà été réécrit : une correction partielle serait pire
            // que pas de correction du tout.
            for (p, avant) in &anciens {
                if let Some(a) = avant {
                    let _ = state.db.set_file_summary(p, a);
                } else {
                    let _ = state.db.unpin_summary(p);
                }
            }
            return Err(format!("correction du sens de {path} échouée ({e}) — rien conservé."));
        }
        anciens.push((path.clone(), ancien));
        let _ = state.db.update_file_hash(path, "");
        let _ = state.db.enqueue_path(path, Some("pending_extraction"), 9);
        requalifies += 1;
    }

    state
        .db
        .mark_transaction_committed(transaction_id)
        .map_err(|e| e.to_string())?;

    let total = done.len() + requalifies;
    let detail = if requalifies > 0 {
        format!(
            "{} opération(s) appliquée(s), dont {requalifies} correction(s) de sens              (ré-indexation en cours pour les rendre effectives dans la recherche).",
            total
        )
    } else {
        format!("{total} opération(s) appliquée(s).")
    };
    Ok(ApplyResult { applied: total, message: detail })
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
        // Ne touche PAS au disque : traité dans la phase sémantique, après que les
        // opérations de fichiers ont réussi. Ainsi un échec disque n'a jamais laissé
        // de descriptions à moitié réécrites.
        OpKind::Requalify => Ok(None),
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
        // Une requalification sans cible ni texte n'écrirait rien, mais serait comptée
        // comme appliquée : on la refuse plutôt que de mentir sur le résultat.
        if op.kind == OpKind::Requalify {
            if op.old_path.is_none() {
                return Err("requalify sans fichier cible".into());
            }
            if op.new_summary.as_deref().map(str::trim).unwrap_or("").is_empty() {
                return Err(format!(
                    "requalify sans nouveau sens pour {}",
                    op.old_path.as_deref().unwrap_or("?")
                ));
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
    /// Échanges d'outils (message assistant porteur des `tool_calls`, puis les
    /// observations) qui ont MENÉ à ce message.
    ///
    /// Le front les reçoit dans `ChatResponse::transcript` et les redonne ici au
    /// message suivant. Sans ce rejeu, l'agent est amnésique : chaque message
    /// repart d'un contexte vide et doit re-chercher ce qu'il vient de lire, en
    /// consommant à nouveau tout son budget de tours.
    #[serde(default)]
    pub tool_messages: Vec<serde_json::Value>,
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
    /// Échanges d'outils de CE tour, à renvoyer tels quels dans le prochain appel
    /// (`ChatTurn::tool_messages`) pour donner une continuité à la conversation.
    pub transcript: Vec<serde_json::Value>,
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

/// Injecté quand le budget de tours est épuisé, pour un dernier appel SANS outils.
///
/// Un modèle coupé en plein travail a presque toujours de quoi répondre : lui rendre
/// la parole vaut mieux que de jeter ce qu'il a rassemblé.
const RELANCE_SANS_OUTILS: &str = "Budget d'outils épuisé : tu ne peux plus en appeler. \
Réponds MAINTENANT à l'utilisateur avec ce que tu as déjà recueilli. Si l'information reste \
incomplète, dis explicitement ce qui manque et ce que tu as pu établir.";

/// Schémas des outils exposés au modèle (format function-calling OpenAI). L'agent
/// s'en sert pour CHERCHER, LIRE, LISTER et PROPOSER des actions (Dry-Run).
///
/// Les bornes annoncées dans les descriptions sont celles de la config : un modèle
/// respecte une limite qu'on lui énonce, et perd son temps contre une limite muette.
fn tool_schemas(agent: &crate::config::AgentConfig) -> serde_json::Value {
    let read_chars = agent.read_chars_sane();
    let max_k = agent.max_search_results_sane();
    json!([
        {"type":"function","function":{
            "name":"search_files",
            "description":"Recherche sémantique HYBRIDE (sens + mots-clés) dans les fichiers indexés de l'utilisateur. Renvoie les fichiers les plus pertinents avec un extrait. À utiliser pour retrouver des documents. Si les résultats semblent incomplets, relance avec une formulation différente et/ou un top_k plus grand.",
            "parameters":{"type":"object","properties":{
                "query":{"type":"string","description":"La requête (langage naturel ou mots-clés)."},
                "scope":{"type":"string","description":"Chemin d'un dossier pour restreindre la recherche (optionnel)."},
                "top_k":{"type":"integer","description":format!("Nombre de résultats souhaité (défaut 8, maximum {max_k}). Augmente-le pour une demande exhaustive (« toutes mes factures 2024 »).")}
            },"required":["query"]}
        }},
        {"type":"function","function":{
            "name":"read_file",
            "description":format!("Lit le contenu texte d'un fichier : documents (PDF, DOCX, PPTX, XLSX, HTML) mais AUSSI médias et images, dont on renvoie ce que l'indexation en a tiré — transcription d'un audio ou d'une vidéo, OCR d'un PDF scanné, description d'une image. Renvoie au plus {read_chars} caractères par appel, PAGINÉ : la réponse indique la position atteinte et le total, rappelle l'outil avec `offset` pour lire la suite. Ne conclus pas sur un document long sans avoir lu les passages qui comptent."),
            "parameters":{"type":"object","properties":{
                "path":{"type":"string","description":"Chemin absolu exact du fichier."},
                "offset":{"type":"integer","description":"Position de départ en caractères (défaut 0). Utilise la valeur indiquée en fin de lecture précédente pour continuer."}
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
            "name":"read_semantics",
            "description":"Lit le SENS EXTRAIT (la description générée à l'indexation) des fichiers d'un dossier. Indispensable avant de corriger des qualifications : c'est ce qui permet de voir quelles descriptions sont fausses et lesquelles sont correctes.",
            "parameters":{"type":"object","properties":{
                "path":{"type":"string","description":"Chemin absolu du dossier."},
                "recursive":{"type":"boolean","description":"Inclure les sous-dossiers (défaut : false)."}
            },"required":["path"]}
        }},
        {"type":"function","function":{
            "name":"propose_actions",
            "description":"Propose un plan d'actions : sur les FICHIERS (déplacer/renommer/supprimer/créer un dossier) ou sur leur SENS EXTRAIT (requalify, qui corrige la description stockée sans toucher au fichier). N'EXÉCUTE RIEN : l'utilisateur validera. Utilise UNIQUEMENT des chemins exacts et existants.",
            "parameters":{"type":"object","properties":{
                "summary":{"type":"string","description":"Résumé en une phrase du plan."},
                "operations":{"type":"array","items":{"type":"object","properties":{
                    "kind":{"type":"string","enum":["move","rename","delete","mkdir","requalify"]},
                    "old_path":{"type":"string","description":"Chemin concerné (source, cible à supprimer, ou fichier à requalifier)."},
                    "new_path":{"type":"string"},
                    "new_summary":{"type":"string","description":"Pour 'requalify' UNIQUEMENT : la description corrigée, rédigée en entier."},
                    "reason":{"type":"string"}
                },"required":["kind"]}}
            },"required":["operations"]}
        }},
        {"type":"function","function":{
            "name":"remember",
            "description":"Mémorise un fait ou une préférence DURABLE que l'utilisateur te confie (ex. « mes factures sont dans le dossier X », « je préfère les réponses courtes »), pour t'en souvenir lors des prochaines conversations. À n'utiliser que pour de l'information durable, pas pour du contexte éphémère.",
            "parameters":{"type":"object","properties":{
                "note":{"type":"string","description":"Le fait ou la préférence à retenir, en une phrase."}
            },"required":["note"]}
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

/// Refus préalables d'une lecture, indépendants de l'index comme du disque.
/// Séparés pour rester vérifiables sans monter un état applicatif complet.
fn refus_de_lecture(path: &str) -> Option<String> {
    if path.trim().is_empty() {
        return Some("Chemin manquant.".to_string());
    }
    if !Path::new(path).is_file() {
        return Some(format!("Introuvable, ou ce n'est pas un fichier : {path}"));
    }
    None
}

/// Origine du texte servi à l'agent. Annoncée dans l'en-tête : le modèle doit
/// savoir s'il lit le fichier ou ce que l'indexation en avait tiré.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SourceLecture {
    /// `file_semantics.extract`, produit à l'indexation. Gratuit.
    Index,
    /// Extraction faite à l'instant depuis le disque. Coûteuse.
    Disque,
}

/// Texte déjà extrait pour ce fichier, s'il est encore fidèle au disque.
///
/// Renvoie `(texte, a_jour)` : un extrait périmé n'est pas jeté pour autant, car
/// c'est parfois la SEULE copie exploitable (transcription d'une vidéo, OCR d'un
/// PDF scanné, légende d'une image — rien de tout ça n'existe sur le disque). On
/// le garde donc en dernier recours, en signalant qu'il date.
fn texte_indexe(state: &Arc<AppState>, path: &str) -> Option<(String, bool)> {
    let extrait = state
        .db
        .get_file_semantics(path)
        .ok()
        .flatten()
        .and_then(|(_, _, extract)| extract)
        .filter(|e| !e.trim().is_empty())?;

    let mtime_disque = std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64);
    let a_jour = match (state.db.indexed_mtime(path).ok().flatten(), mtime_disque) {
        (Some(indexe), Some(disque)) => disque <= indexe,
        // mtime illisible ou fichier jamais marqué indexé : on ne peut rien affirmer.
        _ => false,
    };
    Some((extrait, a_jour))
}

/// Lit le contenu TEXTE d'un fichier pour l'agent, avec pagination.
///
/// **L'index passe avant l'extraction.** Le pipeline a déjà lu ce fichier et rangé
/// son contenu dans `file_semantics.extract` : rouvrir un PDF ou un DOCX pour
/// retrouver le même texte est un coût pur. On ne ré-extrait que si l'index ne
/// couvre pas la fenêtre demandée — c'est-à-dire quand le modèle dépasse les
/// [`EXTRACT_MAX_CHARS`](crate::worker::EXTRACT_MAX_CHARS) conservés en base.
///
/// Cette priorité règle aussi un angle mort : pour un média, une image ou un PDF
/// scanné, le texte (transcription, légende, OCR) N'EXISTE PAS sur le disque. Il
/// n'existe que dans l'index. Lire le fichier ne rendait que du binaire, alors que
/// sa transcription était là, à une requête SQL.
///
/// Quand il faut bien extraire, on emploie l'extracteur du pipeline
/// ([`crate::worker::extract_text`]) — mêmes formats, même résultat qu'à
/// l'indexation — et non une lecture d'octets bruts.
///
/// L'en-tête annonce la position, ce qui reste et d'où vient le texte : un plafond
/// silencieux ne se devine pas, et le modèle doit pouvoir décider de continuer.
fn read_file_for_agent(
    state: &Arc<AppState>,
    path: &str,
    offset: usize,
    max_chars: usize,
) -> String {
    if let Some(refus) = refus_de_lecture(path) {
        return refus;
    }

    let indexe = texte_indexe(state, path);

    // 1) L'index couvre le début de la fenêtre et il est à jour → rien à extraire.
    if let Some((texte, true)) = &indexe {
        if offset < texte.chars().count() {
            return page(texte, offset, max_chars, SourceLecture::Index);
        }
    }

    // 2) Sinon on extrait, avec l'outillage du pipeline.
    let extrait_disque = match crate::worker::extract_text(path) {
        Ok(t) if !t.trim().is_empty() => Some(t),
        // Repli octets bruts : fichiers texte à extension inconnue.
        _ => match std::fs::read(path) {
            Ok(bytes) => {
                let t = String::from_utf8_lossy(&bytes).to_string();
                (!t.trim().is_empty()).then_some(t)
            }
            Err(e) => return format!("Impossible de lire {path} : {e}"),
        },
    };
    if let Some(texte) = extrait_disque {
        return page(&texte, offset, max_chars, SourceLecture::Disque);
    }

    // 3) Rien de lisible sur le disque : c'est le cas des médias, des images et des
    //    PDF scannés. L'index est alors la seule copie — périmée ou non.
    match indexe {
        Some((texte, _)) => page(&texte, offset, max_chars, SourceLecture::Index),
        None => "(aucun texte lisible : binaire non extractible, et rien d'indexé pour ce fichier)"
            .to_string(),
    }
}

/// Découpe une page de texte et lui donne un en-tête honnête sur ce qu'elle couvre.
fn page(texte: &str, offset: usize, max_chars: usize, source: SourceLecture) -> String {
    let total = texte.chars().count();
    if total == 0 {
        return "(fichier vide)".to_string();
    }
    if offset >= total {
        return format!(
            "Fin du texte déjà atteinte (offset {offset} ≥ {total} caractères disponibles)."
        );
    }

    let tranche: String = texte.chars().skip(offset).take(max_chars.max(1)).collect();
    if tranche.trim().is_empty() {
        return "(aucun texte lisible sur cette plage)".to_string();
    }
    let fin = offset + tranche.chars().count();

    // Un extrait au plafond peut cacher une suite : le dire, sinon le modèle conclut
    // sur un document tronqué en croyant l'avoir lu en entier.
    let index_plafonne =
        source == SourceLecture::Index && total >= crate::worker::EXTRACT_MAX_CHARS;

    let origine = match source {
        SourceLecture::Index => "texte indexé",
        SourceLecture::Disque => "fichier lu",
    };
    let entete = if fin < total {
        format!("[{origine} · caractères {offset}–{fin} sur {total} — rappelle read_file avec offset={fin} pour la suite]")
    } else if index_plafonne {
        format!("[{origine} · caractères {offset}–{fin} ; l'index s'arrête ici, rappelle read_file avec offset={fin} pour extraire la suite du fichier]")
    } else {
        format!("[{origine} · caractères {offset}–{fin} sur {total} — fin du document]")
    };
    format!("{entete}\n{tranche}")
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

    let agent_cfg = state.config.snapshot().agent;

    match tc.name.as_str() {
        "search_files" => {
            let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
            let sc = args
                .get("scope")
                .and_then(|v| v.as_str())
                .filter(|s| !s.trim().is_empty())
                .map(str::to_string)
                .or_else(|| scope.clone());
            // Le modèle décide du volume : 8 résultats en dur rendaient toute demande
            // exhaustive impossible, sans qu'aucune reformulation puisse y remédier.
            let top_k = args
                .get("top_k")
                .and_then(|v| v.as_u64())
                .map(|k| k as usize)
                .unwrap_or(8)
                .clamp(1, agent_cfg.max_search_results_sane());
            match crate::search::run_semantic_search(state, query, sc.as_deref(), top_k).await {
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
            let offset = args
                .get("offset")
                .and_then(|v| v.as_u64())
                .map(|o| o as usize)
                .unwrap_or(0);
            read_file_for_agent(state, path, offset, agent_cfg.read_chars_sane())
        }
        "list_directory" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let roots = state.config.snapshot().indexing.roots;
            if !path_within_roots(path, &roots) {
                return format!("Accès refusé : « {path} » est hors des dossiers indexés.");
            }
            list_dir_compact(path)
        }
        "read_semantics" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let roots = state.config.snapshot().indexing.roots;
            if !path_within_roots(path, &roots) {
                return format!("Accès refusé : « {path} » est hors des dossiers indexés.");
            }
            let recursive = args.get("recursive").and_then(|v| v.as_bool()).unwrap_or(false);
            match state.db.list_semantics_under(path, recursive, 60) {
                Ok(v) if v.is_empty() => {
                    "Aucun fichier indexé sous ce dossier (ou sens non encore extraits).".to_string()
                }
                Ok(v) => {
                    let n = v.len();
                    let mut out = format!("{n} fichier(s) avec leur sens extrait :
");
                    for (path, summary, kind) in v {
                        let s = if summary.trim().is_empty() { "(aucun sens)" } else { summary.trim() };
                        out.push_str(&format!("
• {path}
  [{kind}] {s}
"));
                    }
                    out
                }
                Err(e) => format!("Lecture des sens impossible : {e}"),
            }
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
        "remember" => {
            let note = args.get("note").and_then(|v| v.as_str()).unwrap_or("");
            if note.trim().is_empty() {
                return "Rien à mémoriser (note vide).".to_string();
            }
            match state.db.add_memory(note) {
                Ok(_) => format!("Mémorisé : {note}"),
                Err(e) => format!("Erreur de mémorisation : {e}"),
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
        "read_semantics" => format!(
            "🧠 Lecture des sens : {}",
            base(args.get("path").and_then(|v| v.as_str()).unwrap_or(""))
        ),
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

/// Décide, tour par tour, quels échanges d'outils passés sont rejoués dans le
/// contexte, du plus récent au plus ancien et dans la limite d'un budget.
///
/// On garde ou on jette un BLOC ENTIER (message assistant + ses observations) :
/// un message `tool` orphelin, sans le `tool_call` correspondant juste avant, est
/// rejeté par les serveurs compatibles OpenAI. Dès qu'un bloc ne tient plus dans le
/// budget on s'arrête — les plus anciens partent en premier, ce qui préserve la
/// continuité récente, celle dont dépendent les questions de suivi.
fn blocs_a_rejouer(messages: &[ChatTurn], budget: usize) -> Vec<bool> {
    let mut garder = vec![false; messages.len()];
    if budget == 0 {
        return garder;
    }
    let mut utilise = 0usize;
    for (i, m) in messages.iter().enumerate().rev() {
        if m.tool_messages.is_empty() {
            continue;
        }
        let cout: usize = m.tool_messages.iter().map(|v| v.to_string().len()).sum();
        if utilise + cout > budget {
            break;
        }
        utilise += cout;
        garder[i] = true;
    }
    garder
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
fichiers pertinents par leur nom. Un document long se lit en PLUSIEURS fois : read_file expose un \
`offset`, ne conclus jamais sur sa seule première page. Si une recherche paraît incomplète, \
reformule-la ou augmente son `top_k` au lieu de répondre à moitié. Pour AGIR sur des fichiers (déplacer, renommer, supprimer, \
créer un dossier), appelle propose_actions avec des chemins EXACTS et existants : rien n'est \
exécuté sans validation de l'utilisateur. Quand l'utilisateur te confie un fait ou une \
préférence DURABLE, appelle remember pour t'en souvenir aux prochaines conversations. \
Réponds en français, de façon concise et factuelle.",
    );
    let user_override = cfg.prompts.chat_system.trim();
    if !user_override.is_empty() {
        system.push_str("\n\nConsignes supplémentaires :\n");
        system.push_str(user_override);
    }
    // Mémoire durable : ce que l'agent a retenu des sessions précédentes.
    let memories = state.db.list_memories(50).unwrap_or_default();
    if !memories.is_empty() {
        let list = memories
            .iter()
            .map(|(_, n)| format!("- {n}"))
            .collect::<Vec<_>>()
            .join("\n");
        system.push_str(&format!("\n\nCe que tu sais déjà (mémoire durable) :\n{list}"));
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

    // --- Historique converti en messages OpenAI (JSON), échanges d'outils compris. ---
    let mut msgs: Vec<serde_json::Value> = vec![json!({"role": "system", "content": system})];
    let rejouer = blocs_a_rejouer(&messages, cfg.agent.transcript_chars);
    for (i, m) in messages.iter().enumerate() {
        if rejouer[i] {
            // Les observations précèdent la réponse finale à laquelle elles ont servi :
            // c'est l'ordre qu'impose le protocole (assistant+tool_calls, puis tool).
            for tm in &m.tool_messages {
                msgs.push(tm.clone());
            }
        }
        msgs.push(json!({"role": m.role, "content": m.content}));
    }

    // --- Outils : intégrés + outils EXTERNES des serveurs MCP activés (via cache de
    //     découverte ; best-effort : un serveur injoignable est ignoré). ---
    let mut tools = tool_schemas(&cfg.agent);
    let (mcp_schemas, mcp_index) = discover_mcp_tools(&state, &cfg.mcp_servers).await;
    if let Some(arr) = tools.as_array_mut() {
        arr.extend(mcp_schemas);
    }

    // --- Boucle ReAct : le modèle appelle des outils, observe, itère, puis répond. ---
    let client = state.ai.reasoning_client();
    let opts = crate::providers::AgentCallOptions {
        timeout: cfg.agent.timeout(),
        temperature: cfg.agent.temperature,
    };
    let max_rounds = cfg.agent.max_rounds_sane();
    let mut plan: Option<ActionPlan> = None;
    let mut answer: Option<String> = None;
    // Échanges d'outils de ce tour, rendus au front pour la continuité du suivant.
    let mut transcript: Vec<serde_json::Value> = Vec::new();

    for _round in 0..max_rounds {
        let turn = match client.chat_tools(&msgs, &tools, &opts).await {
            Ok(t) => t,
            Err(e) => {
                return Ok(ChatResponse {
                    answer: Some(format!("Erreur du modèle de reasoning : {e}")),
                    sources,
                    plan: None,
                    transcript,
                })
            }
        };

        if turn.tool_calls.is_empty() {
            answer = turn.content;
            break;
        }

        // Réinjecte le message assistant (avec ses tool_calls) puis chaque observation.
        msgs.push(turn.raw_message.clone());
        transcript.push(turn.raw_message.clone());
        for tc in &turn.tool_calls {
            // Trace live : on annonce l'action AVANT de l'exécuter.
            let _ = app.emit("agent-step", AgentStep { label: describe_tool_call(tc, &mcp_index) });
            let observation =
                execute_tool(&state, tc, &scope, &mut sources, &mut plan, &mcp_index).await;
            let obs_msg = json!({"role": "tool", "tool_call_id": tc.id, "content": observation});
            msgs.push(obs_msg.clone());
            transcript.push(obs_msg);
        }

        // Un plan a été proposé → on rend la main à l'utilisateur (validation Dry-Run).
        if plan.is_some() {
            break;
        }
    }

    if plan.is_some() {
        return Ok(ChatResponse { answer: None, sources, plan, transcript });
    }

    // Budget de tours épuisé sans conclusion : on redemande une réponse SANS outils
    // plutôt que de jeter le travail déjà fait. L'ancien comportement renvoyait un
    // « je n'ai pas abouti » générique alors que le modèle avait tout le contexte en
    // main — il était coupé en plein raisonnement, pas en panne d'information.
    if answer.is_none() {
        msgs.push(json!({"role": "system", "content": RELANCE_SANS_OUTILS}));
        match client.chat_tools(&msgs, &json!([]), &opts).await {
            Ok(t) => answer = t.content,
            Err(e) => tracing::warn!("réponse finale forcée impossible : {e}"),
        }
    }

    Ok(ChatResponse {
        answer: Some(answer.unwrap_or_else(|| {
            "Je n'ai pas abouti à une réponse. Reformule ou précise ta demande.".to_string()
        })),
        sources,
        plan: None,
        transcript,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        blocs_a_rejouer, describe_tool_call, page, path_within_roots, refus_de_lecture,
        tool_schemas, validate_operations, ChatTurn, OpKind, Operation, SourceLecture,
    };
    use crate::config::AgentConfig;
    use crate::providers::ToolCallOut;

    fn op(kind: OpKind, old: Option<&str>, sens: Option<&str>) -> Operation {
        Operation {
            kind,
            old_path: old.map(String::from),
            new_path: None,
            new_summary: sens.map(String::from),
            reason: String::new(),
        }
    }

    /// Une requalification vide passerait la validation et serait comptée comme
    /// appliquée sans rien écrire : l'utilisateur croirait sa correction enregistrée.
    #[test]
    fn requalify_exige_une_cible_et_un_texte() {
        let roots = vec![r"C:\Users\v\Docs".to_string()];

        let correcte = op(
            OpKind::Requalify,
            Some(r"C:\Users\v\Docs\a.pdf"),
            Some("Passeport français de Virgile Thonnier."),
        );
        assert!(validate_operations(&[correcte], &roots).is_ok());

        let sans_texte = op(OpKind::Requalify, Some(r"C:\Users\v\Docs\a.pdf"), Some("   "));
        assert!(validate_operations(&[sans_texte], &roots).is_err());

        let sans_cible = op(OpKind::Requalify, None, Some("Un passeport."));
        assert!(validate_operations(&[sans_cible], &roots).is_err());

        // Le périmètre reste contrôlé comme pour toute autre opération : l'agent ne
        // doit pas pouvoir réécrire la description d'un fichier hors des racines.
        let hors_perimetre = op(OpKind::Requalify, Some(r"C:\Windows\x.pdf"), Some("Texte."));
        assert!(validate_operations(&[hors_perimetre], &roots).is_err());
    }

    /// Le modèle doit savoir que `requalify` existe et qu'il exige `new_summary`,
    /// sinon il ne s'en servira jamais.
    #[test]
    fn le_schema_expose_requalify_et_read_semantics() {
        let schemas = tool_schemas(&AgentConfig::default()).to_string();
        assert!(schemas.contains("read_semantics"), "outil de lecture des sens absent");
        assert!(schemas.contains("requalify"), "type d'opération requalify absent");
        assert!(schemas.contains("new_summary"), "champ new_summary absent du schéma");
    }

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
    fn tool_schemas_exposent_les_outils_integres() {
        let tools = tool_schemas(&AgentConfig::default());
        let arr = tools.as_array().expect("tableau d'outils");
        let names: Vec<&str> = arr
            .iter()
            .filter_map(|t| t.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()))
            .collect();
        for expected in [
            "search_files",
            "read_file",
            "list_directory",
            "propose_actions",
            "remember",
        ] {
            assert!(names.contains(&expected), "outil manquant : {expected}");
        }
        assert_eq!(arr.len(), names.len(), "des outils sans nom ?");
    }

    #[test]
    fn read_file_refuse_chemin_vide_et_inexistant() {
        assert_eq!(refus_de_lecture("   ").as_deref(), Some("Chemin manquant."));
        // Un chemin inexistant donne un message, jamais un panic.
        assert!(refus_de_lecture("Z:/nexiste/pas/vraiment.txt")
            .is_some_and(|r| r.starts_with("Introuvable")));
        // Un fichier réel passe la garde.
        let f = std::env::temp_dir().join("sensetree_test_garde.txt");
        std::fs::write(&f, "x").expect("écriture");
        assert!(refus_de_lecture(&f.to_string_lossy()).is_none());
        let _ = std::fs::remove_file(&f);
    }

    /// Le modèle doit pouvoir lire la SUITE d'un document : sans pagination, un long
    /// texte se résume à son début et rien ne permet d'aller plus loin.
    #[test]
    fn la_pagination_annonce_la_suite_et_la_fin() {
        let texte = "a".repeat(250);

        // Première page : l'en-tête doit dire où reprendre.
        let p1 = page(&texte, 0, 100, SourceLecture::Disque);
        assert!(p1.contains("sur 250"), "total absent de l'en-tête : {p1}");
        assert!(p1.contains("offset=100"), "reprise non indiquée : {p1}");

        // Dernière page : plus d'invite à continuer, et le contenu est bien décalé.
        let p3 = page(&texte, 200, 100, SourceLecture::Disque);
        assert!(p3.contains("fin du document"), "fin non signalée : {p3}");
        assert!(!p3.contains("offset="), "reprise proposée en fin de document : {p3}");
        assert_eq!(p3.lines().nth(1).map(str::len), Some(50));

        // Au-delà de la fin : message explicite, jamais un panic ni un vide trompeur.
        assert!(page(&texte, 9_999, 100, SourceLecture::Disque).starts_with("Fin du texte"));
    }

    /// L'extrait stocké est plafonné : arrivé au bout, il faut inviter à EXTRAIRE la
    /// suite, sinon le modèle conclut sur un document tronqué en le croyant entier.
    #[test]
    fn la_fin_dun_extrait_plafonne_invite_a_extraire() {
        let plafond = crate::worker::EXTRACT_MAX_CHARS;
        let tronque = "a".repeat(plafond);

        let fin = page(&tronque, plafond - 50, 100, SourceLecture::Index);
        assert!(fin.contains("texte indexé"), "origine non annoncée : {}", &fin[..80]);
        assert!(fin.contains("l'index s'arrête ici"), "troncature masquée : {}", &fin[..120]);
        assert!(fin.contains(&format!("offset={plafond}")), "reprise non indiquée");
        assert!(!fin.contains("fin du document"), "fin de document annoncée à tort");

        // Un extrait COMPLET (sous le plafond) annonce bien une fin, lui.
        let complet = "a".repeat(100);
        let f2 = page(&complet, 50, 100, SourceLecture::Index);
        assert!(f2.contains("fin du document"), "fin non annoncée sur un extrait complet");
    }

    /// Rejouer un message `tool` sans le `tool_call` qui l'appelle fait rejeter la
    /// requête par les serveurs OpenAI : on garde ou on jette des blocs ENTIERS.
    #[test]
    fn le_rejeu_garde_les_blocs_recents_et_entiers() {
        let bloc = |n: usize| -> Vec<serde_json::Value> {
            (0..n).map(|i| serde_json::json!({"role": "tool", "content": "x".repeat(100), "i": i})).collect()
        };
        let tour = |tm: Vec<serde_json::Value>| ChatTurn {
            role: "assistant".into(),
            content: "ok".into(),
            tool_messages: tm,
        };
        let messages = vec![tour(bloc(2)), tour(bloc(2)), tour(bloc(2))];

        // Budget large : tout est rejoué.
        assert_eq!(blocs_a_rejouer(&messages, 100_000), vec![true, true, true]);
        // Budget nul : le rejeu est désactivé, sans casser la conversation.
        assert_eq!(blocs_a_rejouer(&messages, 0), vec![false, false, false]);
        // Budget étroit : ce sont les tours ANCIENS qui sautent, pas les récents —
        // une question de suivi porte sur ce qui vient d'être lu.
        let serre = blocs_a_rejouer(&messages, 500);
        assert_eq!(serre.last(), Some(&true), "le tour le plus récent doit survivre");
        assert_eq!(serre.first(), Some(&false), "le plus ancien doit sauter en premier");
    }

    /// Les bornes annoncées au modèle doivent être celles de la config : une limite
    /// muette se traduit par des appels perdus contre un plafond invisible.
    #[test]
    fn les_schemas_annoncent_les_bornes_de_la_config() {
        let agent = AgentConfig {
            read_chars: 12_345,
            max_search_results: 77,
            ..AgentConfig::default()
        };
        let schemas = tool_schemas(&agent).to_string();
        assert!(schemas.contains("12345"), "plafond de lecture non annoncé");
        assert!(schemas.contains("77"), "plafond de top_k non annoncé");
        assert!(schemas.contains("offset"), "pagination non exposée");
        assert!(schemas.contains("top_k"), "top_k non exposé");
    }

    /// Une config éditée à la main ne doit pas pouvoir figer l'agent (0 tour) ni le
    /// lâcher sans borne.
    #[test]
    fn les_budgets_de_lagent_sont_assainis() {
        let mut agent = AgentConfig::default();
        assert_eq!(agent.max_rounds_sane(), 24);
        agent.max_rounds = 0;
        assert_eq!(agent.max_rounds_sane(), 1);
        agent.max_rounds = 100_000;
        assert_eq!(agent.max_rounds_sane(), 100);

        agent.timeout_secs = 1;
        assert_eq!(agent.timeout().as_secs(), 10);
        agent.read_chars = 0;
        assert_eq!(agent.read_chars_sane(), 500);
        agent.max_search_results = 0;
        assert_eq!(agent.max_search_results_sane(), 1);
    }
}

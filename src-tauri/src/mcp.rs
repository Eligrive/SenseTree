//! Client MCP (Model Context Protocol) — transport HTTP « Streamable HTTP » (JSON-RPC 2.0).
//!
//! Permet à l'agent d'utiliser des outils EXTERNES exposés par des serveurs MCP
//! configurés par l'utilisateur (URL + auth optionnelle). Best-effort par nature :
//! un serveur injoignable ou non conforme est simplement ignoré, l'agent continue
//! avec ses outils intégrés. On ne gère QUE le transport HTTP (pas stdio) : pas de
//! sous-processus à superviser, un simple `reqwest`.

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

use crate::config::McpServerConfig;

/// Version du protocole annoncée au handshake (rétro-compatible côté serveur).
const PROTOCOL_VERSION: &str = "2025-06-18";

/// Un outil découvert sur un serveur MCP.
#[derive(Debug, Clone)]
pub struct McpTool {
    /// Nom du serveur (préfixe de namespace côté agent).
    pub server: String,
    /// Nom de l'outil tel qu'exposé par le serveur.
    pub name: String,
    pub description: String,
    /// JSON Schema des paramètres (réutilisé tel quel pour le function-calling).
    pub input_schema: Value,
}

/// Session JSON-RPC vers un serveur MCP (gère l'id de session éventuel + le compteur d'id).
struct Session {
    http: reqwest::Client,
    url: String,
    auth: String,
    name: String,
    session_id: Option<String>,
    next_id: i64,
}

impl Session {
    fn new(cfg: &McpServerConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(20))
            .build()
            .unwrap_or_default();
        Session {
            http,
            url: cfg.url.trim().to_string(),
            auth: cfg.auth.trim().to_string(),
            name: cfg.name.clone(),
            session_id: None,
            next_id: 1,
        }
    }

    /// Requête JSON-RPC attendant une réponse. Capture l'éventuel `Mcp-Session-Id`.
    async fn call(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        let body = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        let mut req = self
            .http
            .post(&self.url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .json(&body);
        if !self.auth.is_empty() {
            req = req.header("Authorization", &self.auth);
        }
        if let Some(sid) = &self.session_id {
            req = req.header("Mcp-Session-Id", sid);
        }
        let resp = req
            .send()
            .await
            .with_context(|| format!("MCP {} : {method}", self.name))?;
        if let Some(sid) = resp
            .headers()
            .get("Mcp-Session-Id")
            .and_then(|v| v.to_str().ok())
        {
            self.session_id = Some(sid.to_string());
        }
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(anyhow!("MCP {} {method} : HTTP {status}", self.name));
        }
        let value = parse_jsonrpc(&text)?;
        if let Some(err) = value.get("error") {
            return Err(anyhow!("MCP {} {method} : {err}", self.name));
        }
        Ok(value.get("result").cloned().unwrap_or(Value::Null))
    }

    /// Notification JSON-RPC (sans id, sans réponse attendue).
    async fn notify(&mut self, method: &str) {
        let body = json!({ "jsonrpc": "2.0", "method": method });
        let mut req = self
            .http
            .post(&self.url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .json(&body);
        if !self.auth.is_empty() {
            req = req.header("Authorization", &self.auth);
        }
        if let Some(sid) = &self.session_id {
            req = req.header("Mcp-Session-Id", sid);
        }
        let _ = req.send().await; // best-effort
    }

    /// Handshake MCP : `initialize` puis notification `initialized`.
    async fn handshake(&mut self) -> Result<()> {
        self.call(
            "initialize",
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": { "name": "SenseTree", "version": env!("CARGO_PKG_VERSION") }
            }),
        )
        .await?;
        self.notify("notifications/initialized").await;
        Ok(())
    }
}

/// Extrait l'objet JSON-RPC d'une réponse : JSON pur, OU flux SSE (`data: {...}`).
fn parse_jsonrpc(text: &str) -> Result<Value> {
    let trimmed = text.trim();
    if trimmed.starts_with('{') {
        return serde_json::from_str(trimmed).context("réponse MCP JSON invalide");
    }
    // SSE : on prend le dernier `data:` porteur d'un result/error JSON-RPC.
    for line in text.lines().rev() {
        if let Some(payload) = line.trim().strip_prefix("data:") {
            let payload = payload.trim();
            if payload.starts_with('{') {
                if let Ok(v) = serde_json::from_str::<Value>(payload) {
                    if v.get("result").is_some() || v.get("error").is_some() {
                        return Ok(v);
                    }
                }
            }
        }
    }
    Err(anyhow!(
        "réponse MCP non exploitable : {}",
        trimmed.chars().take(160).collect::<String>()
    ))
}

/// Convertit le résultat `tools/list` en `McpTool` (commun HTTP / stdio).
fn parse_tools(result: &Value, server: &str) -> Vec<McpTool> {
    let mut out = Vec::new();
    let tools = result.get("tools").and_then(|t| t.as_array()).cloned().unwrap_or_default();
    for t in tools {
        let name = t.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
        if name.is_empty() {
            continue;
        }
        out.push(McpTool {
            server: server.to_string(),
            description: t.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            input_schema: t.get("inputSchema").cloned().unwrap_or_else(|| json!({ "type": "object" })),
            name,
        });
    }
    out
}

/// Extrait le texte concaténé d'un résultat `tools/call` (commun HTTP / stdio).
fn extract_content_text(result: &Value) -> String {
    let mut buf = String::new();
    if let Some(items) = result.get("content").and_then(|c| c.as_array()) {
        for it in items {
            if let Some(txt) = it.get("text").and_then(|v| v.as_str()) {
                if !buf.is_empty() {
                    buf.push('\n');
                }
                buf.push_str(txt);
            }
        }
    }
    if buf.trim().is_empty() {
        buf = result.to_string(); // pas de texte : mieux vaut le résultat brut que rien
    }
    buf
}

/// Liste les outils d'un serveur MCP (handshake + `tools/list`). Transport stdio si
/// `command` est renseigné, sinon HTTP via `url`.
pub async fn list_tools(cfg: &McpServerConfig) -> Result<Vec<McpTool>> {
    if !cfg.command.trim().is_empty() {
        let cfg = cfg.clone();
        return tokio::task::spawn_blocking(move || stdio_list_tools(&cfg))
            .await
            .context("tâche MCP stdio interrompue")?;
    }
    if cfg.url.trim().is_empty() {
        return Ok(Vec::new());
    }
    let mut session = Session::new(cfg);
    session.handshake().await?;
    let result = session.call("tools/list", json!({})).await?;
    Ok(parse_tools(&result, &cfg.name))
}

/// Appelle un outil MCP et renvoie le texte concaténé de son contenu.
pub async fn call_tool(cfg: &McpServerConfig, tool: &str, arguments: Value) -> Result<String> {
    if !cfg.command.trim().is_empty() {
        let cfg = cfg.clone();
        let tool = tool.to_string();
        return tokio::task::spawn_blocking(move || stdio_call_tool(&cfg, &tool, arguments))
            .await
            .context("tâche MCP stdio interrompue")?;
    }
    let mut session = Session::new(cfg);
    session.handshake().await?;
    let result = session
        .call("tools/call", json!({ "name": tool, "arguments": arguments }))
        .await?;
    Ok(extract_content_text(&result))
}

// =========================================================================
// TRANSPORT STDIO (sous-processus local, JSON-RPC ligne à ligne)
// =========================================================================

/// Construit la commande stdio. Sur Windows, on passe par `cmd /C` pour résoudre les
/// lanceurs `.cmd`/`.bat` du PATH (npx, uvx…) que `Command::new` ne trouve pas seul.
fn build_stdio_command(cfg: &McpServerConfig) -> Command {
    #[cfg(windows)]
    {
        let mut c = Command::new("cmd");
        c.arg("/C").arg(&cfg.command);
        for a in &cfg.args {
            c.arg(a);
        }
        c
    }
    #[cfg(not(windows))]
    {
        let mut c = Command::new(&cfg.command);
        c.args(&cfg.args);
        c
    }
}

/// Client MCP stdio synchrone (à exécuter via `spawn_blocking`). Le serveur vit le
/// temps de l'échange ; il est tué à la libération (`Drop`).
struct StdioClient {
    child: Child,
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
    next_id: i64,
}

impl StdioClient {
    fn spawn(cfg: &McpServerConfig) -> Result<Self> {
        let mut child = build_stdio_command(cfg)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("lancement du serveur MCP stdio « {} »", cfg.command))?;
        let stdin = child.stdin.take().ok_or_else(|| anyhow!("stdin MCP indisponible"))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("stdout MCP indisponible"))?;
        Ok(StdioClient { child, stdin, reader: BufReader::new(stdout), next_id: 1 })
    }

    fn call(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        let msg = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        writeln!(self.stdin, "{}", serde_json::to_string(&msg)?).context("écriture stdin MCP")?;
        self.stdin.flush().ok();
        // Lit jusqu'à la réponse portant NOTRE id (on saute notifications / logs JSON).
        let mut line = String::new();
        loop {
            line.clear();
            let n = self.reader.read_line(&mut line).context("lecture stdout MCP")?;
            if n == 0 {
                return Err(anyhow!("le serveur MCP stdio a fermé le flux"));
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let v: Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(_) => continue, // ligne non-JSON (log serveur) → ignorée
            };
            if v.get("id").and_then(|i| i.as_i64()) == Some(id) {
                if let Some(e) = v.get("error") {
                    return Err(anyhow!("{e}"));
                }
                return Ok(v.get("result").cloned().unwrap_or(Value::Null));
            }
        }
    }

    fn notify(&mut self, method: &str) {
        let msg = json!({ "jsonrpc": "2.0", "method": method });
        if let Ok(s) = serde_json::to_string(&msg) {
            let _ = writeln!(self.stdin, "{s}");
            let _ = self.stdin.flush();
        }
    }

    fn handshake(&mut self) -> Result<()> {
        self.call(
            "initialize",
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": { "name": "SenseTree", "version": env!("CARGO_PKG_VERSION") }
            }),
        )?;
        self.notify("notifications/initialized");
        Ok(())
    }
}

impl Drop for StdioClient {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn stdio_list_tools(cfg: &McpServerConfig) -> Result<Vec<McpTool>> {
    let mut cli = StdioClient::spawn(cfg)?;
    cli.handshake()?;
    let result = cli.call("tools/list", json!({}))?;
    Ok(parse_tools(&result, &cfg.name))
}

fn stdio_call_tool(cfg: &McpServerConfig, tool: &str, arguments: Value) -> Result<String> {
    let mut cli = StdioClient::spawn(cfg)?;
    cli.handshake()?;
    let result = cli.call("tools/call", json!({ "name": tool, "arguments": arguments }))?;
    Ok(extract_content_text(&result))
}

// =========================================================================
// CACHE DE DÉCOUVERTE (évite un handshake par message de chat)
// =========================================================================

/// Durée de vie du cache de découverte des outils MCP.
pub const DISCOVERY_TTL: Duration = Duration::from_secs(120);

/// Outils MCP découverts, prêts pour le function-calling. Mis en cache dans l'état
/// (clé = signature de la config des serveurs) : re-découverte seulement si la config
/// change ou après expiration du TTL, au lieu d'un handshake à chaque message.
pub struct McpDiscovery {
    pub key: String,
    pub at: Instant,
    /// Schémas d'outils au format function-calling (namespacés `mcp__serveur__outil`).
    pub tools_schema: Vec<Value>,
    /// Routage : nom namespacé → (serveur, nom d'outil côté serveur).
    pub index: HashMap<String, (McpServerConfig, String)>,
}

#[cfg(test)]
mod tests {
    use super::parse_jsonrpc;

    #[test]
    fn parse_json_pur() {
        let v = parse_jsonrpc(r#"{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}"#).unwrap();
        assert!(v.get("result").is_some());
    }

    #[test]
    fn parse_flux_sse() {
        // Un serveur en Streamable HTTP peut répondre en SSE : on doit extraire le data:.
        let sse = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"ok\":true}}\n\n";
        let v = parse_jsonrpc(sse).unwrap();
        assert_eq!(v["result"]["ok"], true);
    }

    #[test]
    fn parse_rejette_le_non_json() {
        assert!(parse_jsonrpc("Internal Server Error").is_err());
    }
}

//! Client MCP (Model Context Protocol) — transport HTTP « Streamable HTTP » (JSON-RPC 2.0).
//!
//! Permet à l'agent d'utiliser des outils EXTERNES exposés par des serveurs MCP
//! configurés par l'utilisateur (URL + auth optionnelle). Best-effort par nature :
//! un serveur injoignable ou non conforme est simplement ignoré, l'agent continue
//! avec ses outils intégrés. On ne gère QUE le transport HTTP (pas stdio) : pas de
//! sous-processus à superviser, un simple `reqwest`.

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::time::Duration;

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

/// Liste les outils d'un serveur MCP (handshake + `tools/list`).
pub async fn list_tools(cfg: &McpServerConfig) -> Result<Vec<McpTool>> {
    if cfg.url.trim().is_empty() {
        return Ok(Vec::new());
    }
    let mut session = Session::new(cfg);
    session.handshake().await?;
    let result = session.call("tools/list", json!({})).await?;
    let tools = result
        .get("tools")
        .and_then(|t| t.as_array())
        .cloned()
        .unwrap_or_default();
    let mut out = Vec::new();
    for t in tools {
        let name = t.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
        if name.is_empty() {
            continue;
        }
        out.push(McpTool {
            server: cfg.name.clone(),
            description: t
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            input_schema: t
                .get("inputSchema")
                .cloned()
                .unwrap_or_else(|| json!({ "type": "object" })),
            name,
        });
    }
    Ok(out)
}

/// Appelle un outil MCP et renvoie le texte concaténé de son contenu.
pub async fn call_tool(cfg: &McpServerConfig, tool: &str, arguments: Value) -> Result<String> {
    let mut session = Session::new(cfg);
    session.handshake().await?;
    let result = session
        .call("tools/call", json!({ "name": tool, "arguments": arguments }))
        .await?;

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
        // Pas de contenu texte : on renvoie le résultat brut (mieux que rien).
        buf = result.to_string();
    }
    Ok(buf)
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

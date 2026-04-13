//! Streaming heartbeat for cold-load resilience.
//!
//! When an Ollama backend receives a chat/completion request for a model
//! that isn't in memory, it silently blocks the HTTP response — no headers,
//! no body — until loading completes. Upstream clients with idle-chunk
//! timeouts (e.g. OpenClaw's `llm-idle-timeout`) interpret this silence as
//! a hang and abort the connection before the model ever replies.
//!
//! This module injects protocol-appropriate keepalive bytes into the
//! downstream response while the upstream is silent, so clients see the
//! stream as alive. The first real upstream byte causes the router to drop
//! heartbeats and pipe the upstream response through unchanged.
//!
//! # Flow
//!
//! 1. **Preflight** — call `/api/ps` on the target backend. If the model
//!    is already loaded, fall through to normal proxying (no heartbeat).
//! 2. **Heartbeat proxy** — commit to a 200 OK with a protocol-specific
//!    `Content-Type`. Emit heartbeat chunks every
//!    `loading_heartbeat_secs`; switch to piping real bytes the moment
//!    upstream produces any. Bounded by `loading_max_wait_secs` — past
//!    that, emit an in-band error and close.
//!
//! # Honesty
//!
//! The preflight is the evidence gate: we only emit heartbeats when we
//! *confirmed* the model wasn't loaded at request start. If upstream
//! errors (connect failure, HTTP 5xx, body error), we encode the error
//! in-band in the committed protocol instead of silently heartbeating
//! into the void.

use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode};
use axum::response::Response;
use bytes::Bytes;
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::json;
use tokio::sync::mpsc;
use tokio::time::MissedTickBehavior;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Public configuration
// ---------------------------------------------------------------------------

/// Timing knobs for the heartbeat path. Cloned into each request.
#[derive(Debug, Clone, Copy)]
pub struct HeartbeatConfig {
    pub heartbeat_interval: Duration,
    pub preflight_timeout: Duration,
    pub max_wait: Duration,
}

impl HeartbeatConfig {
    pub fn from_secs(heartbeat: u64, preflight: u64, max_wait: u64) -> Self {
        Self {
            heartbeat_interval: Duration::from_secs(heartbeat),
            preflight_timeout: Duration::from_secs(preflight),
            max_wait: Duration::from_secs(max_wait),
        }
    }
}

// ---------------------------------------------------------------------------
// Protocol classification
// ---------------------------------------------------------------------------

/// The wire protocol of a streaming chat/completion endpoint. Used to
/// choose a heartbeat format that every compliant client of that protocol
/// will safely ignore.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamProtocol {
    /// Ollama-native NDJSON chat (`/api/chat`).
    OllamaChat,
    /// Ollama-native NDJSON generate (`/api/generate`).
    OllamaGenerate,
    /// OpenAI-compat SSE (`/v1/chat/completions`, `/v1/completions`).
    OpenAiSse,
    /// Anthropic-compat SSE (`/v1/messages`).
    AnthropicSse,
}

impl StreamProtocol {
    /// Classify a request path. Returns `None` for paths that don't stream
    /// chat/completion content (embeddings, `/api/show`, etc.).
    pub fn from_path(path: &str) -> Option<Self> {
        match path {
            "/api/chat" => Some(Self::OllamaChat),
            "/api/generate" => Some(Self::OllamaGenerate),
            "/v1/chat/completions" | "/v1/completions" => Some(Self::OpenAiSse),
            "/v1/messages" => Some(Self::AnthropicSse),
            _ => None,
        }
    }

    pub fn content_type(self) -> &'static str {
        match self {
            Self::OllamaChat | Self::OllamaGenerate => "application/x-ndjson",
            Self::OpenAiSse | Self::AnthropicSse => "text/event-stream",
        }
    }

    /// Heartbeat bytes. SSE protocols use a comment line (spec-compliant
    /// and ignored by every spec parser, including the EventSource API,
    /// httpx_sse, and the OpenAI/Anthropic SDKs). Ollama NDJSON uses a
    /// valid chunk with empty content and `done: false`.
    pub fn heartbeat(self, model: &str) -> Bytes {
        match self {
            Self::OllamaChat => {
                let chunk = json!({
                    "model": model,
                    "created_at": FIXED_TIMESTAMP,
                    "message": { "role": "assistant", "content": "" },
                    "done": false,
                });
                Bytes::from(format!("{chunk}\n"))
            }
            Self::OllamaGenerate => {
                let chunk = json!({
                    "model": model,
                    "created_at": FIXED_TIMESTAMP,
                    "response": "",
                    "done": false,
                });
                Bytes::from(format!("{chunk}\n"))
            }
            Self::OpenAiSse | Self::AnthropicSse => {
                Bytes::from_static(b": ollama-router heartbeat\n\n")
            }
        }
    }

    /// Synthesize a terminal error event in-band. Used when upstream fails
    /// *after* we've already committed to a 200 OK downstream, so we can't
    /// change the HTTP status.
    pub fn error_event(self, model: &str, message: &str) -> Bytes {
        match self {
            Self::OllamaChat => {
                let chunk = json!({
                    "model": model,
                    "created_at": FIXED_TIMESTAMP,
                    "error": message,
                    "done": true,
                });
                Bytes::from(format!("{chunk}\n"))
            }
            Self::OllamaGenerate => {
                let chunk = json!({
                    "model": model,
                    "created_at": FIXED_TIMESTAMP,
                    "error": message,
                    "done": true,
                });
                Bytes::from(format!("{chunk}\n"))
            }
            Self::OpenAiSse => {
                let payload = json!({
                    "error": { "message": message, "type": "upstream_error" }
                });
                Bytes::from(format!("data: {payload}\n\ndata: [DONE]\n\n"))
            }
            Self::AnthropicSse => {
                let payload = json!({
                    "type": "error",
                    "error": { "type": "upstream_error", "message": message }
                });
                Bytes::from(format!("event: error\ndata: {payload}\n\n"))
            }
        }
    }
}

/// A valid RFC 3339 timestamp. The field is required by Ollama clients
/// (Python's `ollama` package parses it as `datetime`) but heartbeat
/// chunks are cosmetic — clients use it for nothing that matters here,
/// so a fixed value avoids pulling in a date crate.
const FIXED_TIMESTAMP: &str = "1970-01-01T00:00:00Z";

// ---------------------------------------------------------------------------
// Preflight probe
// ---------------------------------------------------------------------------

/// Ask a backend whether a model is currently loaded in memory.
///
/// Returns:
/// - `Ok(true)`  — model is in the `/api/ps` response; no heartbeat needed.
/// - `Ok(false)` — probe succeeded and model is absent; heartbeat path is safe.
/// - `Err(_)`    — probe failed (timeout, connect error, parse error). Caller
///   should fall through to normal proxying and let it surface the failure.
pub async fn preflight_is_loaded(
    client: &Client,
    backend_url: &str,
    model: &str,
    timeout: Duration,
) -> Result<bool, PreflightError> {
    #[derive(serde::Deserialize)]
    struct PsResponse {
        #[serde(default)]
        models: Vec<PsModel>,
    }
    #[derive(serde::Deserialize)]
    struct PsModel {
        #[serde(default)]
        name: String,
        #[serde(default)]
        model: String,
    }

    let url = format!("{backend_url}/api/ps");
    let resp = client
        .get(&url)
        .timeout(timeout)
        .send()
        .await
        .map_err(|e| PreflightError::Request(e.to_string()))?;

    if !resp.status().is_success() {
        return Err(PreflightError::Status(resp.status().as_u16()));
    }

    let body: PsResponse = resp
        .json()
        .await
        .map_err(|e| PreflightError::Parse(e.to_string()))?;

    // Ollama's /api/ps reports both `name` (e.g. "llama3:latest") and
    // `model`. Match on either, plus the bare prefix before `:`, so
    // callers that looked up a model by prefix still get a hot-path hit.
    let hit = body.models.iter().any(|m| {
        m.name == model
            || m.model == model
            || m.name.split_once(':').is_some_and(|(p, _)| p == model)
            || m.model.split_once(':').is_some_and(|(p, _)| p == model)
    });

    Ok(hit)
}

#[derive(Debug)]
pub enum PreflightError {
    Request(String),
    Status(u16),
    Parse(String),
}

impl std::fmt::Display for PreflightError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Request(s) => write!(f, "request: {s}"),
            Self::Status(c) => write!(f, "status {c}"),
            Self::Parse(s) => write!(f, "parse: {s}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Heartbeat proxy
// ---------------------------------------------------------------------------

/// Request parameters for the heartbeat proxy. Mirrors `proxy::ProxyRequest`
/// plus the protocol classification and timing knobs.
pub struct HeartbeatRequest<'a> {
    pub client: &'a reqwest::Client,
    pub backend_url: &'a str,
    pub path: &'a str,
    pub query: Option<&'a str>,
    pub method: Method,
    pub headers: &'a HeaderMap,
    pub body: Body,
    pub protocol: StreamProtocol,
    pub model: String,
    pub config: HeartbeatConfig,
}

/// Forward a streaming request to the backend with keepalive bytes filling
/// in for upstream silence during model load.
///
/// Commits to `200 OK` and the protocol's Content-Type before the first
/// heartbeat is emitted. From that point, any upstream failure is encoded
/// in-band using `StreamProtocol::error_event`.
pub async fn execute(req: HeartbeatRequest<'_>) -> Response {
    let mut url = format!("{}{}", req.backend_url, req.path);
    if let Some(q) = req.query {
        url.push('?');
        url.push_str(q);
    }

    let mut builder = req.client.request(req.method, &url);
    for (key, value) in req.headers.iter() {
        match key.as_str() {
            "host" | "connection" | "transfer-encoding" | "keep-alive" | "upgrade" => continue,
            _ => builder = builder.header(key.clone(), value.clone()),
        }
    }

    let body_stream = req
        .body
        .into_data_stream()
        .map(|r| r.map_err(std::io::Error::other));
    let reqwest_body = reqwest::Body::wrap_stream(body_stream);
    let send_future = builder.body(reqwest_body).send();

    // Channel buffer of 16 is plenty — a heartbeat every 15s and an
    // upstream producing tokens at 100 tok/s max both fit comfortably.
    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(16);

    let cfg = req.config;
    let protocol = req.protocol;
    let model = req.model.clone();

    tokio::spawn(async move {
        run_heartbeat_task(send_future, tx, protocol, model, cfg).await;
    });

    let stream = ReceiverStream::new(rx);
    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert(
        "content-type",
        HeaderValue::from_static(protocol.content_type()),
    );
    response
}

async fn run_heartbeat_task<F>(
    send_future: F,
    tx: mpsc::Sender<Result<Bytes, std::io::Error>>,
    protocol: StreamProtocol,
    model: String,
    cfg: HeartbeatConfig,
) where
    F: std::future::Future<Output = reqwest::Result<reqwest::Response>>,
{
    let start = Instant::now();
    let mut ticker = tokio::time::interval_at(
        tokio::time::Instant::now() + cfg.heartbeat_interval,
        cfg.heartbeat_interval,
    );
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

    tokio::pin!(send_future);

    // Phase 1 — wait for upstream headers, emitting heartbeats while silent.
    let upstream = loop {
        tokio::select! {
            biased;
            resp = &mut send_future => break resp,
            _ = ticker.tick() => {
                if start.elapsed() >= cfg.max_wait {
                    warn!(model = %model, elapsed_secs = start.elapsed().as_secs(),
                          "heartbeat: upstream headers never arrived, aborting");
                    let _ = tx.send(Ok(protocol.error_event(&model, "model load timed out"))).await;
                    return;
                }
                debug!(model = %model, elapsed_secs = start.elapsed().as_secs(),
                       "heartbeat: emitting keepalive while waiting for upstream headers");
                if tx.send(Ok(protocol.heartbeat(&model))).await.is_err() {
                    // Downstream disconnected — nothing to do.
                    return;
                }
            }
        }
    };

    let resp = match upstream {
        Ok(r) => r,
        Err(e) => {
            warn!(model = %model, error = %e, "heartbeat: upstream request failed");
            let _ = tx
                .send(Ok(protocol.error_event(&model, &format!("upstream: {e}"))))
                .await;
            return;
        }
    };

    if !resp.status().is_success() {
        let status = resp.status();
        let body_text = resp.text().await.unwrap_or_default();
        warn!(model = %model, %status, body = %body_text, "heartbeat: upstream returned error status");
        let msg = if body_text.is_empty() {
            format!("upstream returned {status}")
        } else {
            format!("upstream {status}: {body_text}")
        };
        let _ = tx.send(Ok(protocol.error_event(&model, &msg))).await;
        return;
    }

    info!(
        model = %model,
        header_wait_secs = start.elapsed().as_secs(),
        "heartbeat: upstream headers received, streaming body"
    );

    // Phase 2 — pipe body, still heartbeating until the first real byte.
    let mut body_stream = resp.bytes_stream();
    let mut first_byte_seen = false;
    loop {
        tokio::select! {
            biased;
            item = body_stream.next() => {
                match item {
                    Some(Ok(b)) if b.is_empty() => continue,
                    Some(Ok(b)) => {
                        if !first_byte_seen {
                            first_byte_seen = true;
                            debug!(model = %model, first_byte_secs = start.elapsed().as_secs(),
                                   "heartbeat: first upstream byte, stopping keepalives");
                        }
                        if tx.send(Ok(b)).await.is_err() { return; }
                    }
                    Some(Err(e)) => {
                        let _ = tx.send(Err(std::io::Error::other(e))).await;
                        return;
                    }
                    None => return,
                }
            }
            _ = ticker.tick(), if !first_byte_seen => {
                if start.elapsed() >= cfg.max_wait {
                    warn!(model = %model, elapsed_secs = start.elapsed().as_secs(),
                          "heartbeat: first token never arrived, aborting");
                    let _ = tx.send(Ok(protocol.error_event(&model, "first token timed out"))).await;
                    return;
                }
                if tx.send(Ok(protocol.heartbeat(&model))).await.is_err() { return; }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_chat_paths() {
        assert_eq!(
            StreamProtocol::from_path("/api/chat"),
            Some(StreamProtocol::OllamaChat)
        );
        assert_eq!(
            StreamProtocol::from_path("/api/generate"),
            Some(StreamProtocol::OllamaGenerate)
        );
        assert_eq!(
            StreamProtocol::from_path("/v1/chat/completions"),
            Some(StreamProtocol::OpenAiSse)
        );
        assert_eq!(
            StreamProtocol::from_path("/v1/completions"),
            Some(StreamProtocol::OpenAiSse)
        );
        assert_eq!(
            StreamProtocol::from_path("/v1/messages"),
            Some(StreamProtocol::AnthropicSse)
        );
    }

    #[test]
    fn classify_non_streaming_paths() {
        assert_eq!(StreamProtocol::from_path("/api/embed"), None);
        assert_eq!(StreamProtocol::from_path("/api/embeddings"), None);
        assert_eq!(StreamProtocol::from_path("/api/show"), None);
        assert_eq!(StreamProtocol::from_path("/v1/embeddings"), None);
        assert_eq!(StreamProtocol::from_path("/api/tags"), None);
        assert_eq!(StreamProtocol::from_path("/unknown"), None);
    }

    #[test]
    fn ollama_chat_heartbeat_is_valid_ndjson_with_empty_content() {
        let bytes = StreamProtocol::OllamaChat.heartbeat("llama3");
        let s = std::str::from_utf8(&bytes).unwrap();
        assert!(
            s.ends_with('\n'),
            "ndjson chunks must be newline-terminated"
        );
        let json: serde_json::Value = serde_json::from_str(s.trim()).unwrap();
        assert_eq!(json["model"], "llama3");
        assert_eq!(json["done"], false);
        assert_eq!(json["message"]["role"], "assistant");
        assert_eq!(json["message"]["content"], "");
    }

    #[test]
    fn ollama_generate_heartbeat_shape() {
        let bytes = StreamProtocol::OllamaGenerate.heartbeat("phi3");
        let s = std::str::from_utf8(&bytes).unwrap();
        let json: serde_json::Value = serde_json::from_str(s.trim()).unwrap();
        assert_eq!(json["response"], "");
        assert_eq!(json["done"], false);
    }

    #[test]
    fn sse_heartbeat_is_a_comment() {
        // SSE comments start with `:` and are skipped by every spec parser.
        let bytes = StreamProtocol::OpenAiSse.heartbeat("ignored");
        let s = std::str::from_utf8(&bytes).unwrap();
        assert!(s.starts_with(':'));
        assert!(s.ends_with("\n\n"));

        let bytes = StreamProtocol::AnthropicSse.heartbeat("ignored");
        let s = std::str::from_utf8(&bytes).unwrap();
        assert!(s.starts_with(':'));
    }

    #[test]
    fn ollama_error_event_has_done_true_and_error_field() {
        let bytes = StreamProtocol::OllamaChat.error_event("llama3", "boom");
        let json: serde_json::Value =
            serde_json::from_str(std::str::from_utf8(&bytes).unwrap().trim()).unwrap();
        assert_eq!(json["done"], true);
        assert_eq!(json["error"], "boom");
    }

    #[test]
    fn openai_error_event_terminates_with_done_sentinel() {
        let bytes = StreamProtocol::OpenAiSse.error_event("gpt-4", "nope");
        let s = std::str::from_utf8(&bytes).unwrap();
        assert!(s.contains("data: [DONE]"));
        assert!(s.contains("\"error\""));
    }

    // ---- async tests -----------------------------------------------------

    use axum::Router;
    use axum::routing::{get, post};
    use http_body_util::BodyExt;
    use tokio::net::TcpListener;

    async fn spawn_ps_server(models: Vec<&'static str>) -> String {
        let payload = serde_json::json!({
            "models": models.iter().map(|m| serde_json::json!({"name": m, "model": m})).collect::<Vec<_>>()
        })
        .to_string();
        let app = Router::new().route(
            "/api/ps",
            get(move || {
                let p = payload.clone();
                async move { (StatusCode::OK, p) }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn preflight_detects_loaded_model() {
        let url = spawn_ps_server(vec!["llama3:latest"]).await;
        let client = reqwest::Client::new();
        let loaded = preflight_is_loaded(&client, &url, "llama3:latest", Duration::from_secs(2))
            .await
            .unwrap();
        assert!(loaded);
    }

    #[tokio::test]
    async fn preflight_detects_prefix_match() {
        let url = spawn_ps_server(vec!["llama3:latest"]).await;
        let client = reqwest::Client::new();
        let loaded = preflight_is_loaded(&client, &url, "llama3", Duration::from_secs(2))
            .await
            .unwrap();
        assert!(loaded);
    }

    #[tokio::test]
    async fn preflight_returns_false_when_model_absent() {
        let url = spawn_ps_server(vec!["other:latest"]).await;
        let client = reqwest::Client::new();
        let loaded = preflight_is_loaded(&client, &url, "llama3", Duration::from_secs(2))
            .await
            .unwrap();
        assert!(!loaded);
    }

    #[tokio::test]
    async fn preflight_empty_ps_returns_false() {
        let url = spawn_ps_server(vec![]).await;
        let client = reqwest::Client::new();
        let loaded = preflight_is_loaded(&client, &url, "llama3", Duration::from_secs(2))
            .await
            .unwrap();
        assert!(!loaded);
    }

    #[tokio::test]
    async fn preflight_errors_on_unreachable_backend() {
        let client = reqwest::Client::new();
        let err = preflight_is_loaded(
            &client,
            "http://127.0.0.1:1",
            "llama3",
            Duration::from_millis(200),
        )
        .await;
        assert!(err.is_err());
    }

    /// Spawn a mock chat endpoint that delays `delay` before emitting body.
    async fn spawn_slow_chat_server(
        delay: Duration,
        response_body: &'static str,
        content_type: &'static str,
    ) -> String {
        let app = Router::new().route(
            "/api/chat",
            post(move || async move {
                tokio::time::sleep(delay).await;
                (
                    StatusCode::OK,
                    [(axum::http::header::CONTENT_TYPE, content_type)],
                    response_body,
                )
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn heartbeat_fires_then_real_body_flows() {
        let url = spawn_slow_chat_server(
            Duration::from_millis(300),
            "{\"done\":true}\n",
            "application/x-ndjson",
        )
        .await;

        let client = reqwest::Client::new();
        let headers = HeaderMap::new();
        let req = HeartbeatRequest {
            client: &client,
            backend_url: &url,
            path: "/api/chat",
            query: None,
            method: Method::POST,
            headers: &headers,
            body: Body::from("{\"model\":\"llama3\",\"stream\":true}"),
            protocol: StreamProtocol::OllamaChat,
            model: "llama3".to_string(),
            config: HeartbeatConfig {
                heartbeat_interval: Duration::from_millis(100),
                preflight_timeout: Duration::from_secs(10),
                max_wait: Duration::from_secs(10),
            },
        };

        let resp = execute(req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "application/x-ndjson"
        );

        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let text = std::str::from_utf8(&bytes).unwrap();

        // Should contain at least one heartbeat chunk (empty content, done:false)
        // followed by the real body chunk.
        assert!(
            text.contains("\"done\":false"),
            "expected heartbeat chunk in: {text}"
        );
        assert!(
            text.contains("\"done\":true"),
            "expected real body in: {text}"
        );
    }

    #[tokio::test]
    async fn heartbeat_emits_error_event_on_upstream_error_status() {
        let app = Router::new().route(
            "/api/chat",
            post(|| async {
                tokio::time::sleep(Duration::from_millis(150)).await;
                (StatusCode::INTERNAL_SERVER_ERROR, "model failed to load")
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let url = format!("http://{addr}");

        let client = reqwest::Client::new();
        let headers = HeaderMap::new();
        let req = HeartbeatRequest {
            client: &client,
            backend_url: &url,
            path: "/api/chat",
            query: None,
            method: Method::POST,
            headers: &headers,
            body: Body::from("{}"),
            protocol: StreamProtocol::OllamaChat,
            model: "llama3".to_string(),
            config: HeartbeatConfig {
                heartbeat_interval: Duration::from_millis(50),
                preflight_timeout: Duration::from_secs(5),
                max_wait: Duration::from_secs(5),
            },
        };

        let resp = execute(req).await;
        assert_eq!(resp.status(), StatusCode::OK); // we committed to 200
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(text.contains("\"error\""), "expected error event: {text}");
        assert!(
            text.contains("500"),
            "expected upstream status in error: {text}"
        );
    }

    #[tokio::test]
    async fn heartbeat_emits_timeout_when_upstream_silent_past_max_wait() {
        // Upstream sleeps forever (longer than max_wait).
        let app = Router::new().route(
            "/api/chat",
            post(|| async {
                tokio::time::sleep(Duration::from_secs(60)).await;
                (StatusCode::OK, "never")
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let url = format!("http://{addr}");

        let client = reqwest::Client::new();
        let headers = HeaderMap::new();
        let req = HeartbeatRequest {
            client: &client,
            backend_url: &url,
            path: "/api/chat",
            query: None,
            method: Method::POST,
            headers: &headers,
            body: Body::from("{}"),
            protocol: StreamProtocol::OllamaChat,
            model: "llama3".to_string(),
            config: HeartbeatConfig {
                heartbeat_interval: Duration::from_millis(50),
                preflight_timeout: Duration::from_secs(5),
                max_wait: Duration::from_millis(300),
            },
        };

        let resp = execute(req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(
            text.contains("timed out"),
            "expected timeout error event: {text}"
        );
    }
}

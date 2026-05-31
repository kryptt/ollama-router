use axum::body::Body;
use axum::http::{HeaderMap, Method, StatusCode};
use axum::response::Response;
use futures_util::StreamExt;

use crate::response::json_error;

/// Everything needed to proxy a request to a backend.
pub struct ProxyRequest<'a> {
    pub client: &'a reqwest::Client,
    pub backend_url: &'a str,
    pub path: &'a str,
    /// When `Some`, used as the backend-side path instead of `path`. Set
    /// when protocol translation rewrites `/api/chat` → `/v1/chat/completions`
    /// without changing what the client sees on its end.
    pub override_path: Option<&'a str>,
    pub query: Option<&'a str>,
    pub method: Method,
    pub headers: &'a HeaderMap,
    pub body: Body,
}

/// A transport-level failure that prevented obtaining *any* upstream response.
///
/// An upstream that actually *responds* — including with a 5xx — is an
/// `Ok(Response)` carrying that status; the retry/circuit-breaker layer
/// (Unit 3) classifies those from `response.status()`. This enum is only for
/// the "no response at all" cases, where the kind (connect vs timeout vs other)
/// drives retry/backoff decisions. The `String` is the original error display,
/// kept for logging and breaker diagnostics.
#[derive(Debug)]
#[must_use = "a transport failure must be handled or converted via into_response()"]
pub enum ProxyError {
    /// Could not establish a connection (refused, DNS, TLS, …).
    Connect(String),
    /// Connected, but upstream did not respond before the timeout.
    Timeout(String),
    /// Any other error sending the request or reading the response head.
    Transport(String),
}

impl ProxyError {
    /// Stable label for metrics/tracing: `connect` / `timeout` / `transport`.
    pub fn kind_str(&self) -> &'static str {
        match self {
            ProxyError::Connect(_) => "connect",
            ProxyError::Timeout(_) => "timeout",
            ProxyError::Transport(_) => "transport",
        }
    }

    /// The client-facing 5xx response for this failure when it is *not*
    /// retried. Preserves the pre-refactor status/body mapping exactly.
    pub fn into_response(self) -> Response {
        match self {
            ProxyError::Connect(_) => {
                json_error(StatusCode::BAD_GATEWAY, "upstream connect failed", None)
            }
            ProxyError::Timeout(_) => json_error(
                StatusCode::GATEWAY_TIMEOUT,
                "upstream request timed out",
                None,
            ),
            ProxyError::Transport(_) => {
                json_error(StatusCode::BAD_GATEWAY, "upstream unavailable", None)
            }
        }
    }
}

/// Build the upstream `reqwest` request: join the URL, strip hop-by-hop
/// headers, and wrap the axum body as a streaming reqwest body. Shared by
/// `execute` (this module) and `heartbeat::execute` so the two forwarding
/// paths can't drift — and so a single place gates request construction for
/// the Unit 3 retry wrapper. `upstream_path` is the already-resolved path
/// (the caller applies any `override_path`).
pub(crate) fn build_upstream_request(
    client: &reqwest::Client,
    method: Method,
    backend_url: &str,
    upstream_path: &str,
    query: Option<&str>,
    headers: &HeaderMap,
    body: Body,
) -> reqwest::RequestBuilder {
    let mut url = format!("{backend_url}{upstream_path}");
    if let Some(q) = query {
        url.push('?');
        url.push_str(q);
    }

    let mut builder = client.request(method, &url);
    for (key, value) in headers.iter() {
        match key.as_str() {
            // content-length is dropped so reqwest computes it from the
            // outgoing body. The client's value is wrong whenever we
            // translate the body (e.g. /api/chat → /v1/chat/completions)
            // and reqwest aborts the request on length mismatch.
            "host" | "connection" | "transfer-encoding" | "keep-alive" | "upgrade"
            | "content-length" => continue,
            _ => builder = builder.header(key.clone(), value.clone()),
        }
    }

    // Convert axum Body stream → reqwest streaming body.
    let body_stream = body
        .into_data_stream()
        .map(|r| r.map_err(std::io::Error::other));
    builder.body(reqwest::Body::wrap_stream(body_stream))
}

/// Forward a request to the backend and stream the response back. Returns
/// `Ok` with the streamed response for *any* upstream status, or `Err` when
/// no response could be obtained (connect/timeout/transport).
pub async fn execute(req: ProxyRequest<'_>) -> Result<Response, ProxyError> {
    let upstream_path = req.override_path.unwrap_or(req.path);
    let builder = build_upstream_request(
        req.client,
        req.method,
        req.backend_url,
        upstream_path,
        req.query,
        req.headers,
        req.body,
    );

    let upstream_resp = match builder.send().await {
        Ok(r) => r,
        Err(e) if e.is_connect() => {
            tracing::warn!(error = %e, "upstream connect failed");
            return Err(ProxyError::Connect(e.to_string()));
        }
        Err(e) if e.is_timeout() => {
            tracing::warn!(error = %e, "upstream request timed out");
            return Err(ProxyError::Timeout(e.to_string()));
        }
        Err(e) => {
            tracing::warn!(error = %e, "upstream request failed");
            return Err(ProxyError::Transport(e.to_string()));
        }
    };

    let status = StatusCode::from_u16(upstream_resp.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    let content_type = upstream_resp.headers().get("content-type").cloned();

    let stream = upstream_resp
        .bytes_stream()
        .map(|r| r.map_err(std::io::Error::other));

    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = status;
    if let Some(ct) = content_type {
        response.headers_mut().insert("content-type", ct);
    }

    Ok(response)
}

pub fn model_not_found(model: &str, available: &[&str]) -> Response {
    json_error(
        StatusCode::NOT_FOUND,
        &format!("model '{model}' not found"),
        Some(("available_models", serde_json::json!(available))),
    )
}

pub fn method_not_allowed(path: &str) -> Response {
    json_error(
        StatusCode::METHOD_NOT_ALLOWED,
        &format!("model management endpoint '{path}' is blocked; use direct backend endpoints"),
        None,
    )
}

pub fn bad_request(msg: &str) -> Response {
    json_error(StatusCode::BAD_REQUEST, msg, None)
}

pub fn bad_gateway(msg: &str) -> Response {
    json_error(StatusCode::BAD_GATEWAY, msg, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_not_found_returns_404() {
        let resp = model_not_found("foo", &["bar", "baz"]);
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn method_not_allowed_returns_405() {
        assert_eq!(
            method_not_allowed("/api/pull").status(),
            StatusCode::METHOD_NOT_ALLOWED
        );
    }

    #[test]
    fn bad_request_returns_400() {
        assert_eq!(bad_request("invalid").status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn proxy_error_maps_to_pre_refactor_status_and_body() {
        use http_body_util::BodyExt;

        async fn parts(resp: Response) -> (StatusCode, String) {
            let status = resp.status();
            let bytes = resp.into_body().collect().await.unwrap().to_bytes();
            let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            (status, v["error"].as_str().unwrap().to_string())
        }

        // The status AND body strings are load-bearing — Ollama clients parse
        // the `error` field. Connect and Transport both map to 502, so the
        // body messages must stay distinct.
        let (status, msg) = parts(ProxyError::Connect("e".into()).into_response()).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert_eq!(msg, "upstream connect failed");

        let (status, msg) = parts(ProxyError::Timeout("e".into()).into_response()).await;
        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(msg, "upstream request timed out");

        let (status, msg) = parts(ProxyError::Transport("e".into()).into_response()).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert_eq!(msg, "upstream unavailable");
    }

    #[test]
    fn build_upstream_request_strips_hop_by_hop_and_joins_query() {
        use axum::http::HeaderValue;

        let client = reqwest::Client::new();
        let mut headers = HeaderMap::new();
        headers.insert("content-length", HeaderValue::from_static("100"));
        headers.insert("host", HeaderValue::from_static("example.com"));
        headers.insert("connection", HeaderValue::from_static("keep-alive"));
        headers.insert("authorization", HeaderValue::from_static("Bearer x"));
        headers.insert("x-custom", HeaderValue::from_static("foo"));

        let req = build_upstream_request(
            &client,
            Method::POST,
            "http://backend:1234",
            "/v1/chat/completions",
            Some("q=1"),
            &headers,
            Body::empty(),
        )
        .build()
        .expect("request should build");

        assert_eq!(
            req.url().as_str(),
            "http://backend:1234/v1/chat/completions?q=1"
        );
        let h = req.headers();
        // Hop-by-hop headers are stripped...
        assert!(h.get("content-length").is_none(), "content-length stripped");
        assert!(h.get("host").is_none(), "host stripped");
        assert!(h.get("connection").is_none(), "connection stripped");
        // ...application headers pass through.
        assert_eq!(h.get("authorization").unwrap(), "Bearer x");
        assert_eq!(h.get("x-custom").unwrap(), "foo");
    }
}

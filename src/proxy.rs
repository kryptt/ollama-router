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
    pub query: Option<&'a str>,
    pub method: Method,
    pub headers: &'a HeaderMap,
    pub body: Body,
}

/// Forward a request to the backend and stream the response back.
pub async fn execute(req: ProxyRequest<'_>) -> Response {
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

    // Convert axum Body stream → reqwest streaming body.
    let body_stream = req.body.into_data_stream().map(|r| r.map_err(std::io::Error::other));
    let reqwest_body = reqwest::Body::wrap_stream(body_stream);

    let upstream_resp = match builder.body(reqwest_body).send().await {
        Ok(r) => r,
        Err(e) if e.is_connect() => {
            tracing::warn!(error = %e, "upstream connect failed");
            return json_error(StatusCode::BAD_GATEWAY, "upstream connect failed", None);
        }
        Err(e) if e.is_timeout() => {
            tracing::warn!(error = %e, "upstream request timed out");
            return json_error(StatusCode::GATEWAY_TIMEOUT, "upstream request timed out", None);
        }
        Err(e) => {
            tracing::warn!(error = %e, "upstream request failed");
            return json_error(StatusCode::BAD_GATEWAY, "upstream unavailable", None);
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

    response
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
}

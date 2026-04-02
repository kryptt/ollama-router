use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// JSON response with a given status code.
pub fn json_status(status: StatusCode, body: serde_json::Value) -> Response {
    (status, json_headers(), body.to_string()).into_response()
}

/// JSON response with 200 OK.
pub fn json_ok(body: serde_json::Value) -> Response {
    json_status(StatusCode::OK, body)
}

/// Structured JSON error response with optional extra field.
pub fn json_error(
    status: StatusCode,
    message: &str,
    extra: Option<(&str, serde_json::Value)>,
) -> Response {
    let mut body = json!({ "error": message });
    if let Some((key, value)) = extra {
        body[key] = value;
    }
    json_status(status, body)
}

fn json_headers() -> [(axum::http::header::HeaderName, HeaderValue); 1] {
    [(
        axum::http::header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    )]
}

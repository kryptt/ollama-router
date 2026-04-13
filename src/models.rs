use axum::http::StatusCode;
use axum::response::Response;
use serde_json::json;

use crate::registry::Registry;
use crate::response::{json_ok, json_status};

/// Build a merged `/api/tags` response from the registry's discovery cache.
/// Deduplicates by model name, preferring the first backend encountered.
pub fn api_tags_response(reg: &Registry) -> Response {
    let models = reg.reachable_models();
    json_ok(json!({ "models": models }))
}

/// Build a merged `/v1/models` response (OpenAI-compatible list).
pub fn v1_models_response(reg: &Registry) -> Response {
    let models = v1_model_objects(reg);
    json_ok(json!({ "object": "list", "data": models }))
}

/// `GET /v1/models/:model_id` — OpenAI-compatible single-model retrieval.
/// Matches by exact name or prefix (before `:`), same as routing lookup.
pub fn v1_model_response(reg: &Registry, model_id: &str) -> Response {
    // Try exact match against reachable models first, then prefix.
    let found = reg.reachable_models().into_iter().find(|m| {
        m.name == model_id
            || m.name
                .split_once(':')
                .is_some_and(|(prefix, _)| prefix == model_id)
    });

    match found {
        Some(m) => json_ok(v1_model_object(m)),
        None => json_status(
            StatusCode::NOT_FOUND,
            json!({
                "error": {
                    "message": format!("The model `{model_id}` does not exist"),
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            }),
        ),
    }
}

fn v1_model_object(m: &crate::registry::ModelInfo) -> serde_json::Value {
    // Strip the default `:latest` tag so clients that request the bare model
    // name (e.g. "glm-4.7-flash") see an exact ID match.
    let id = m.name.strip_suffix(":latest").unwrap_or(&m.name);
    json!({
        "id": id,
        "object": "model",
        "created": 0,
        "owned_by": "library",
    })
}

fn v1_model_objects(reg: &Registry) -> Vec<serde_json::Value> {
    reg.reachable_models()
        .into_iter()
        .map(v1_model_object)
        .collect()
}

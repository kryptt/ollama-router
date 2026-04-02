use axum::response::Response;
use serde_json::json;

use crate::registry::Registry;
use crate::response::json_ok;

/// Build a merged `/api/tags` response from the registry's discovery cache.
/// Deduplicates by model name, preferring the first backend encountered.
pub fn api_tags_response(reg: &Registry) -> Response {
    let models = reg.reachable_models();
    json_ok(json!({ "models": models }))
}

/// Build a merged `/v1/models` response (Ollama's own format, not OpenAI spec).
pub fn v1_models_response(reg: &Registry) -> Response {
    let models: Vec<serde_json::Value> = reg
        .reachable_models()
        .into_iter()
        .map(|m| {
            json!({
                "id": m.name,
                "object": "model",
                "owned_by": "library",
            })
        })
        .collect();

    json_ok(json!({ "object": "list", "data": models }))
}

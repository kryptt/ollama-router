use std::time::Duration;

use axum::http::StatusCode;
use axum::response::Response;
use futures_util::future::join_all;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::registry::{ModelInfo, Registry, SharedRegistry};
use crate::response::{json_ok, json_status};

/// Per-backend timeout for the `/api/ps` fan-out. Independent of the shared
/// client's request timeout (which is sized for long LLM streams).
const PS_FANOUT_TIMEOUT: Duration = Duration::from_secs(5);

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

/// Aggregated `/api/ps` response: union of currently-loaded models across
/// every healthy backend.
///
/// Ollama's `/api/ps` reports one entry per loaded model — the protocol
/// supports many. Three classes of backend speak it differently:
///
/// - **Ollama-API** (default — `ollama-cuda`, etc.): native `/api/ps`,
///   one entry per resident `llama-server` child. Passthrough.
/// - **llama-swap**: own `/running` endpoint, at most one entry under the
///   default exclusive group, more when groups are configured. Translate
///   to ollama-ps shape and merge in discovery metadata so size/digest are
///   populated.
/// - **llama-edge** and other always-resident backends: synthesise one
///   entry per discovered model. Treated as "always loaded".
///
/// Backend failures degrade silently — that backend contributes an empty
/// list rather than failing the aggregate response.
///
/// Takes `SharedRegistry` rather than `&Registry` so we can release the
/// read lock before the fan-out awaits — otherwise discovery would be
/// blocked for up to `PS_FANOUT_TIMEOUT` per slow backend.
pub async fn api_ps_response(registry: &SharedRegistry, client: &Client) -> Response {
    let snapshots: Vec<BackendSnapshot> = {
        let reg = registry.read().await;
        reg.all_backends()
            .filter(|b| b.healthy)
            .map(|b| BackendSnapshot {
                name: b.name.to_string(),
                url: b.url.to_string(),
                kind: classify(b.name),
                models: b.models.to_vec(),
            })
            .collect()
    };

    let futs = snapshots.iter().map(|b| async move {
        match b.kind {
            BackendKind::Ollama => fetch_ollama_ps(client, b).await,
            BackendKind::LlamaSwap => fetch_llama_swap_running(client, b).await,
            BackendKind::AlwaysResident => synthesise_all_loaded(b),
        }
    });
    let per_backend: Vec<Vec<Value>> = join_all(futs).await;
    let models: Vec<Value> = per_backend.into_iter().flatten().collect();
    json_ok(json!({ "models": models }))
}

/// Which `/api/ps`-shaped endpoint a backend speaks, derived from its
/// configured name. Encoded here rather than as per-backend config because
/// the three backend families are small, named, and unlikely to grow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Ollama,
    LlamaSwap,
    AlwaysResident,
}

pub fn classify(name: &str) -> BackendKind {
    // Match the literal name first, then a `<kind>-<suffix>` form so future
    // sharded deployments (`llama-swap-rocm`, `llama-edge-anine`, …) classify
    // correctly without code changes.
    match name {
        "llama-swap" => BackendKind::LlamaSwap,
        "llama-edge" => BackendKind::AlwaysResident,
        _ => match name.split_once('-') {
            Some(("llama", rest)) if rest.starts_with("swap") => BackendKind::LlamaSwap,
            Some(("llama", rest)) if rest.starts_with("edge") => BackendKind::AlwaysResident,
            _ => BackendKind::Ollama,
        },
    }
}

struct BackendSnapshot {
    name: String,
    url: String,
    kind: BackendKind,
    models: Vec<ModelInfo>,
}

async fn fetch_ollama_ps(client: &Client, b: &BackendSnapshot) -> Vec<Value> {
    let url = format!("{}/api/ps", b.url);
    let resp = match client.get(&url).timeout(PS_FANOUT_TIMEOUT).send().await {
        Ok(r) if r.status().is_success() => r,
        Ok(r) => {
            tracing::debug!(backend = %b.name, status = %r.status(), "/api/ps non-success");
            return Vec::new();
        }
        Err(e) => {
            tracing::debug!(backend = %b.name, error = %e, "/api/ps fetch failed");
            return Vec::new();
        }
    };
    match resp.json::<Value>().await {
        Ok(v) => v
            .get("models")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
        Err(e) => {
            tracing::debug!(backend = %b.name, error = %e, "/api/ps json parse failed");
            Vec::new()
        }
    }
}

#[derive(Deserialize)]
struct RunningResponse {
    #[serde(default)]
    running: Vec<RunningEntry>,
}

#[derive(Deserialize)]
struct RunningEntry {
    #[serde(default)]
    model: String,
}

async fn fetch_llama_swap_running(client: &Client, b: &BackendSnapshot) -> Vec<Value> {
    let url = format!("{}/running", b.url);
    let resp = match client.get(&url).timeout(PS_FANOUT_TIMEOUT).send().await {
        Ok(r) if r.status().is_success() => r,
        Ok(r) => {
            tracing::debug!(backend = %b.name, status = %r.status(), "/running non-success");
            return Vec::new();
        }
        Err(e) => {
            tracing::debug!(backend = %b.name, error = %e, "/running fetch failed");
            return Vec::new();
        }
    };
    let body: RunningResponse = match resp.json().await {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!(backend = %b.name, error = %e, "/running json parse failed");
            return Vec::new();
        }
    };
    body.running
        .into_iter()
        .filter(|r| !r.model.is_empty())
        .map(|r| ps_entry(&r.model, &b.models))
        .collect()
}

fn synthesise_all_loaded(b: &BackendSnapshot) -> Vec<Value> {
    b.models
        .iter()
        .map(|m| ps_entry(&m.name, &b.models))
        .collect()
}

/// Build one `/api/ps` entry: `{name, model}` plus whatever size/digest/
/// details/modified_at the discovery cache has on file for this model.
fn ps_entry(model_name: &str, registry_models: &[ModelInfo]) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("name".into(), Value::String(model_name.to_string()));
    obj.insert("model".into(), Value::String(model_name.to_string()));
    if let Some(info) = registry_models.iter().find(|m| m.name == model_name)
        && let Value::Object(extra) = &info.extra
    {
        for (k, v) in extra {
            obj.entry(k.clone()).or_insert(v.clone());
        }
    }
    Value::Object(obj)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_known_backends() {
        assert_eq!(classify("ollama-cuda"), BackendKind::Ollama);
        assert_eq!(classify("ollama-rocm"), BackendKind::Ollama);
        assert_eq!(classify("llama-swap"), BackendKind::LlamaSwap);
        assert_eq!(classify("llama-swap-rocm"), BackendKind::LlamaSwap);
        assert_eq!(classify("llama-edge"), BackendKind::AlwaysResident);
        assert_eq!(classify("llama-edge-anine"), BackendKind::AlwaysResident);
        assert_eq!(classify("anything-else"), BackendKind::Ollama);
    }

    #[test]
    fn ps_entry_merges_discovery_metadata() {
        let models = vec![ModelInfo {
            name: "qwen3.6:latest".into(),
            extra: json!({
                "size": 21_000_000_000_u64,
                "digest": "abc123",
                "details": {"family": "qwen"}
            }),
        }];
        let entry = ps_entry("qwen3.6:latest", &models);
        assert_eq!(entry["name"], "qwen3.6:latest");
        assert_eq!(entry["model"], "qwen3.6:latest");
        assert_eq!(entry["size"], 21_000_000_000_u64);
        assert_eq!(entry["digest"], "abc123");
        assert_eq!(entry["details"]["family"], "qwen");
    }

    #[test]
    fn ps_entry_handles_missing_discovery_metadata() {
        let entry = ps_entry("unknown:latest", &[]);
        assert_eq!(entry["name"], "unknown:latest");
        assert_eq!(entry["model"], "unknown:latest");
        // No extras — but the object stays valid JSON with just name + model.
        let obj = entry.as_object().unwrap();
        assert_eq!(obj.len(), 2);
    }

    #[test]
    fn synthesise_all_loaded_emits_one_entry_per_discovered_model() {
        let b = BackendSnapshot {
            name: "llama-edge".into(),
            url: "http://example".into(),
            kind: BackendKind::AlwaysResident,
            models: vec![
                ModelInfo {
                    name: "ternary-bonsai-8b:latest".into(),
                    extra: json!({"size": 4_000_000_000_u64}),
                },
                ModelInfo {
                    name: "tinyllama:latest".into(),
                    extra: json!({}),
                },
            ],
        };
        let entries = synthesise_all_loaded(&b);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0]["name"], "ternary-bonsai-8b:latest");
        assert_eq!(entries[0]["size"], 4_000_000_000_u64);
        assert_eq!(entries[1]["name"], "tinyllama:latest");
    }
}

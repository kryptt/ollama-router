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
    /// Full llama-server invocation. We parse `--ctx-size` and `--parallel`
    /// out of it to synthesise a per-slot `context_length` for clients.
    #[serde(default)]
    cmd: String,
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
        .map(|r| {
            let context_length = per_slot_context_length(&r.cmd);
            ps_entry(&r.model, &b.models, context_length)
        })
        .collect()
}

fn synthesise_all_loaded(b: &BackendSnapshot) -> Vec<Value> {
    b.models
        .iter()
        .map(|m| ps_entry(&m.name, &b.models, None))
        .collect()
}

/// Build one `/api/ps` entry: `{name, model}` plus whatever size/digest/
/// details/modified_at the discovery cache has on file for this model.
///
/// Two synthesised fields fill gaps the upstream doesn't report:
///
/// - `expires_at` — always now + 1 h. llama-swap/llama-edge don't carry a
///   real TTL the way Ollama's loaded-model unload timer does; surfacing a
///   value lets clients code uniformly against the field.
/// - `context_length` — when the caller can compute it (llama-swap parses
///   it out of the `--ctx-size`/`--parallel` flags on the live child),
///   passed in. AlwaysResident backends don't expose enough to compute
///   one, so they get nothing here.
///
/// Existing keys from the discovery cache (registry_models[*].extra) win
/// over synthesised defaults: `or_insert` is one-shot per key.
fn ps_entry(model_name: &str, registry_models: &[ModelInfo], context_length: Option<u64>) -> Value {
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
    if let Some(cl) = context_length {
        obj.entry("context_length").or_insert(Value::from(cl));
    }
    obj.entry("expires_at")
        .or_insert_with(|| Value::String(synthesised_expires_at(EXPIRES_AT_HORIZON_SECS)));
    Value::Object(obj)
}

/// Synthesised `expires_at` horizon for backends without a real TTL.
const EXPIRES_AT_HORIZON_SECS: i64 = 3600;

fn synthesised_expires_at(seconds_from_now: i64) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    crate::translate::format_rfc3339_utc(now + seconds_from_now)
}

/// Parse a llama-server invocation string and return the per-slot context
/// budget (`--ctx-size` divided by `--parallel`, defaulting parallel to 1).
/// Returns `None` when `--ctx-size` is absent or unparseable, so the caller
/// can omit the field entirely rather than reporting a bogus 0.
///
/// Handles both space- and `=`-separated forms (`--ctx-size 4096` and
/// `--ctx-size=4096`); llama.cpp accepts both.
fn per_slot_context_length(cmd: &str) -> Option<u64> {
    let (ctx_size, parallel) = parse_llama_server_flags(cmd);
    let ctx = ctx_size?;
    let par = parallel.unwrap_or(1).max(1);
    Some(ctx / par)
}

fn parse_llama_server_flags(cmd: &str) -> (Option<u64>, Option<u64>) {
    let mut ctx_size = None;
    let mut parallel = None;
    let mut tokens = cmd.split_whitespace();
    while let Some(tok) = tokens.next() {
        if tok == "--ctx-size" || tok == "-c" {
            ctx_size = tokens.next().and_then(|s| s.parse().ok());
        } else if tok == "--parallel" || tok == "-np" {
            parallel = tokens.next().and_then(|s| s.parse().ok());
        } else if let Some(v) = tok.strip_prefix("--ctx-size=") {
            ctx_size = v.parse().ok();
        } else if let Some(v) = tok.strip_prefix("--parallel=") {
            parallel = v.parse().ok();
        }
    }
    (ctx_size, parallel)
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
        let entry = ps_entry("qwen3.6:latest", &models, None);
        assert_eq!(entry["name"], "qwen3.6:latest");
        assert_eq!(entry["model"], "qwen3.6:latest");
        assert_eq!(entry["size"], 21_000_000_000_u64);
        assert_eq!(entry["digest"], "abc123");
        assert_eq!(entry["details"]["family"], "qwen");
    }

    #[test]
    fn ps_entry_handles_missing_discovery_metadata() {
        let entry = ps_entry("unknown:latest", &[], None);
        assert_eq!(entry["name"], "unknown:latest");
        assert_eq!(entry["model"], "unknown:latest");
        // name + model + synthesised expires_at, no context_length.
        let obj = entry.as_object().unwrap();
        assert_eq!(obj.len(), 3);
        assert!(obj.contains_key("expires_at"));
    }

    #[test]
    fn ps_entry_includes_context_length_when_provided() {
        let entry = ps_entry("qwen3.6-medium", &[], Some(43_690));
        assert_eq!(entry["context_length"], 43_690);
    }

    #[test]
    fn ps_entry_omits_context_length_when_absent() {
        let entry = ps_entry("ternary-bonsai-8b", &[], None);
        assert!(!entry.as_object().unwrap().contains_key("context_length"));
    }

    #[test]
    fn ps_entry_synthesises_expires_at_one_hour_out() {
        let entry = ps_entry("qwen3.6-medium", &[], None);
        let expires_at = entry["expires_at"].as_str().unwrap();
        // RFC 3339, ends with Z.
        assert!(expires_at.ends_with('Z'));
        assert_eq!(expires_at.len(), 20); // "YYYY-MM-DDTHH:MM:SSZ"
    }

    #[test]
    fn ps_entry_real_expires_at_from_registry_wins() {
        // If discovery already carries an expires_at (real Ollama upstream),
        // we don't clobber it with the synthesised one.
        let models = vec![ModelInfo {
            name: "qwen3.6:latest".into(),
            extra: json!({"expires_at": "2030-01-01T00:00:00Z"}),
        }];
        let entry = ps_entry("qwen3.6:latest", &models, None);
        assert_eq!(entry["expires_at"], "2030-01-01T00:00:00Z");
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

    #[test]
    fn parse_cmd_extracts_space_separated_flags() {
        let cmd = "llama-server --model /m --ctx-size 262144 --parallel 6 --jinja";
        assert_eq!(parse_llama_server_flags(cmd), (Some(262_144), Some(6)));
    }

    #[test]
    fn parse_cmd_extracts_equals_separated_flags() {
        let cmd = "llama-server --ctx-size=4096 --parallel=2";
        assert_eq!(parse_llama_server_flags(cmd), (Some(4_096), Some(2)));
    }

    #[test]
    fn parse_cmd_handles_short_flags() {
        let cmd = "llama-server -c 8192 -np 4";
        assert_eq!(parse_llama_server_flags(cmd), (Some(8_192), Some(4)));
    }

    #[test]
    fn parse_cmd_tolerates_newlines() {
        // llama-swap's /running response embeds newlines in `cmd`.
        let cmd = "llama-server\n--model /m\n--ctx-size 262144\n--parallel 6\n--jinja\n";
        assert_eq!(parse_llama_server_flags(cmd), (Some(262_144), Some(6)));
    }

    #[test]
    fn per_slot_context_divides_by_parallel() {
        let cmd = "llama-server --ctx-size 262144 --parallel 6";
        assert_eq!(per_slot_context_length(cmd), Some(43_690));
    }

    #[test]
    fn per_slot_context_defaults_parallel_to_one() {
        let cmd = "llama-server --ctx-size 32768";
        assert_eq!(per_slot_context_length(cmd), Some(32_768));
    }

    #[test]
    fn per_slot_context_none_without_ctx_size() {
        let cmd = "llama-server --parallel 4";
        assert_eq!(per_slot_context_length(cmd), None);
    }

    #[test]
    fn per_slot_context_none_on_empty_cmd() {
        assert_eq!(per_slot_context_length(""), None);
    }
}

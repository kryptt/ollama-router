use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::config::Config;

/// Opaque index into the backends array. Not constructable outside this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BackendId(usize);

/// Per-backend mutable state, updated by the discovery loop.
#[derive(Debug)]
struct BackendState {
    name: String,
    url: String,
    healthy: bool,
    models: Vec<ModelInfo>,
    last_seen: Option<Instant>,
    grace_deadline: Option<Instant>,
}

impl BackendState {
    fn is_reachable(&self) -> bool {
        self.healthy || self.grace_deadline.is_some()
    }
}

/// Model metadata from Ollama's `/api/tags` response.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelInfo {
    pub name: String,
    #[serde(flatten)]
    pub extra: serde_json::Value,
}

#[derive(Deserialize)]
struct TagsResponse {
    models: Option<Vec<ModelInfo>>,
}

#[derive(Deserialize)]
struct V1ModelsResponse {
    data: Option<Vec<V1Model>>,
}

#[derive(Deserialize)]
struct V1Model {
    id: String,
    #[serde(flatten)]
    extra: serde_json::Value,
}

/// Read-only snapshot of a backend, returned to callers.
pub struct BackendView<'a> {
    pub name: &'a str,
    pub url: &'a str,
    pub healthy: bool,
    pub models: &'a [ModelInfo],
    pub in_grace_period: bool,
}

/// The central routing table. All access goes through `SharedRegistry`.
///
/// Invariant: `model_map` keys stay in sync with `backends[*].models`
/// via `rebuild_model_map`, called at the end of every discovery cycle.
pub struct Registry {
    backends: Vec<BackendState>,
    model_map: HashMap<String, BackendId>,
    discovery_done: bool,
}

impl Registry {
    fn new(config: &Config) -> Self {
        let backends = config
            .backends
            .iter()
            .map(|b| BackendState {
                name: b.name.clone(),
                url: b.url.clone(),
                healthy: false,
                models: Vec::new(),
                last_seen: None,
                grace_deadline: None,
            })
            .collect();

        Registry {
            backends,
            model_map: HashMap::new(),
            discovery_done: false,
        }
    }

    /// Look up a model by exact name, then by prefix (before `:`) if no exact match.
    pub fn lookup(&self, model: &str) -> Option<BackendId> {
        self.model_map.get(model).copied()
    }

    /// Return the first healthy backend for model-less request proxying.
    pub fn any_healthy(&self) -> Option<BackendId> {
        self.backends
            .iter()
            .enumerate()
            .find(|(_, b)| b.healthy)
            .map(|(i, _)| BackendId(i))
    }

    /// Borrow a backend's view by id.
    pub fn backend(&self, id: BackendId) -> BackendView<'_> {
        let b = &self.backends[id.0];
        BackendView {
            name: &b.name,
            url: &b.url,
            healthy: b.healthy,
            models: &b.models,
            in_grace_period: b.grace_deadline.is_some(),
        }
    }

    /// Iterate over all backends.
    pub fn all_backends(&self) -> impl Iterator<Item = BackendView<'_>> {
        self.backends.iter().map(|b| BackendView {
            name: &b.name,
            url: &b.url,
            healthy: b.healthy,
            models: &b.models,
            in_grace_period: b.grace_deadline.is_some(),
        })
    }

    /// Deduplicated models from all reachable backends.
    pub fn reachable_models(&self) -> Vec<&ModelInfo> {
        let mut seen = HashMap::new();
        for backend in &self.backends {
            if !backend.is_reachable() {
                continue;
            }
            for model in &backend.models {
                seen.entry(model.name.as_str()).or_insert(model);
            }
        }
        seen.into_values().collect()
    }

    /// Fully qualified model names from all reachable backends.
    pub fn available_model_names(&self) -> Vec<&str> {
        self.model_map
            .iter()
            .filter(|(k, _)| k.contains(':'))
            .map(|(k, _)| k.as_str())
            .collect()
    }

    pub fn is_discovery_done(&self) -> bool {
        self.discovery_done
    }

    fn rebuild_model_map(&mut self) {
        self.model_map.clear();
        for (idx, backend) in self.backends.iter().enumerate() {
            if !backend.is_reachable() {
                continue;
            }
            let id = BackendId(idx);
            for model in &backend.models {
                // First-writer-wins for both exact and prefix keys.
                // Earlier backends in the config list take priority.
                self.model_map.entry(model.name.clone()).or_insert(id);
                if let Some(prefix) = model.name.split_once(':').map(|(p, _)| p) {
                    self.model_map.entry(prefix.to_string()).or_insert(id);
                }
            }
        }
    }
}

pub type SharedRegistry = Arc<RwLock<Registry>>;

pub fn new_shared(config: &Config) -> SharedRegistry {
    Arc::new(RwLock::new(Registry::new(config)))
}

/// Long-running discovery loop. Runs first cycle immediately, then every `interval`.
pub async fn discovery_loop(registry: SharedRegistry, config: Config) {
    let client = Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("failed to build discovery HTTP client");

    let interval = Duration::from_secs(config.discovery_interval_secs);
    let grace_duration = Duration::from_secs(config.grace_period_secs());

    run_discovery(&client, &registry, &config, grace_duration).await;

    loop {
        tokio::time::sleep(interval).await;
        run_discovery(&client, &registry, &config, grace_duration).await;
    }
}

/// Fetch results from backends, keyed by index.
enum FetchResult {
    Ok(Vec<ModelInfo>),
    Err,
}

/// Try Ollama `/api/tags` first, then fall back to OpenAI `/v1/models`.
async fn fetch_models(client: &Client, name: &str, url: &str) -> FetchResult {
    // Try Ollama /api/tags
    let tags_url = format!("{url}/api/tags");
    match client.get(&tags_url).send().await {
        Ok(resp) if resp.status().is_success() => {
            return match resp.json::<TagsResponse>().await {
                Ok(tags) => {
                    let mut models = tags.models.unwrap_or_default();
                    for m in &mut models {
                        sanitize_model_entry(m);
                    }
                    FetchResult::Ok(models)
                }
                Err(e) => {
                    warn!(backend = %name, error = %e, "failed to parse /api/tags");
                    FetchResult::Err
                }
            };
        }
        Ok(_) => {} // non-success — try OpenAI fallback
        Err(e) => {
            warn!(backend = %name, error = %e, "failed to reach backend");
            return FetchResult::Err;
        }
    }

    // Fallback: OpenAI /v1/models
    let v1_url = format!("{url}/v1/models");
    match client.get(&v1_url).send().await {
        Ok(resp) if resp.status().is_success() => match resp.json::<V1ModelsResponse>().await {
            Ok(v1) => {
                let models = v1
                    .data
                    .unwrap_or_default()
                    .into_iter()
                    .map(|m| {
                        let mut info = ModelInfo {
                            name: m.id,
                            extra: m.extra,
                        };
                        sanitize_model_entry(&mut info);
                        info
                    })
                    .collect();
                FetchResult::Ok(models)
            }
            Err(e) => {
                warn!(backend = %name, error = %e, "failed to parse /v1/models");
                FetchResult::Err
            }
        },
        Ok(resp) => {
            warn!(backend = %name, status = %resp.status(), "unhealthy backend (tried /api/tags and /v1/models)");
            FetchResult::Err
        }
        Err(e) => {
            warn!(backend = %name, error = %e, "failed to reach backend");
            FetchResult::Err
        }
    }
}

/// Normalise the fields strict ollama-API clients (e.g. Home Assistant's
/// pydantic-based `ollama.AsyncClient`) require to be well-typed.
///
/// llama.cpp's ollama-compatibility `/api/tags` emits empty strings for
/// `modified_at` and `size` even though pydantic models them as `datetime`
/// and a byte-size integer respectively. An empty string crashes the
/// client's `ListResponse` validator and the whole `/api/tags` request
/// fails (HA refuses to set up its ollama integration entirely — one
/// malformed model takes the whole list down).
///
/// We can't fix llama.cpp here, but we can paper over the bad fields
/// with values that parse cleanly while still being clearly synthetic:
/// epoch for `modified_at`, `0` for `size`. Real ollama backends emit
/// well-formed values, so the in-place rewrite is a no-op for them.
fn sanitize_model_entry(m: &mut ModelInfo) {
    let Some(obj) = m.extra.as_object_mut() else {
        return;
    };
    let is_blank = |v: &serde_json::Value| matches!(v.as_str(), Some("")) || v.is_null();

    if obj.get("modified_at").is_none_or(is_blank) {
        obj.insert(
            "modified_at".to_string(),
            serde_json::Value::String("1970-01-01T00:00:00Z".to_string()),
        );
    }
    if obj.get("size").is_none_or(is_blank) {
        obj.insert("size".to_string(), serde_json::json!(0));
    }
}

async fn run_discovery(
    client: &Client,
    registry: &SharedRegistry,
    config: &Config,
    grace_duration: Duration,
) {
    // Phase 1: Fetch from all backends WITHOUT holding any lock.
    // Backend URLs come from Config (immutable), so no lock needed.
    let mut fetch_results = Vec::with_capacity(config.backends.len());
    for backend in &config.backends {
        let result = fetch_models(client, &backend.name, &backend.url).await;
        fetch_results.push(result);
    }

    // Phase 2: Apply results under write lock (no I/O, microseconds).
    let mut reg = registry.write().await;
    let now = Instant::now();

    for (backend, result) in reg.backends.iter_mut().zip(fetch_results) {
        match result {
            FetchResult::Ok(models) => {
                if !backend.healthy {
                    info!(backend = %backend.name, models = models.len(), "backend recovered");
                }
                backend.healthy = true;
                backend.models = models;
                backend.last_seen = Some(now);
                backend.grace_deadline = None;
            }
            FetchResult::Err => {
                mark_down(backend, now, grace_duration);
            }
        }
    }

    // Expire grace periods.
    for backend in &mut reg.backends {
        if let Some(deadline) = backend.grace_deadline
            && now >= deadline
        {
            info!(backend = %backend.name, "grace period expired, removing models");
            backend.models.clear();
            backend.grace_deadline = None;
        }
    }

    reg.rebuild_model_map();

    if !reg.discovery_done {
        info!("first discovery cycle complete");
        reg.discovery_done = true;
    }
}

fn mark_down(backend: &mut BackendState, now: Instant, grace_duration: Duration) {
    if backend.healthy {
        info!(backend = %backend.name, "backend marked as down, entering grace period");
        backend.healthy = false;
        backend.grace_deadline = Some(now + grace_duration);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Backend, Config};

    fn test_config() -> Config {
        Config::from_backends(vec![
            Backend::for_test("cuda", "http://cuda:11434"),
            Backend::for_test("rocm", "http://rocm:11435"),
        ])
    }

    fn make_model(name: &str) -> ModelInfo {
        ModelInfo {
            name: name.to_string(),
            extra: serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    fn set_backend_models(reg: &mut Registry, idx: usize, healthy: bool, models: Vec<ModelInfo>) {
        reg.backends[idx].healthy = healthy;
        reg.backends[idx].models = models;
        reg.rebuild_model_map();
    }

    #[test]
    fn new_registry_starts_empty() {
        let reg = Registry::new(&test_config());
        assert!(!reg.is_discovery_done());
        assert!(reg.model_map.is_empty());
    }

    #[test]
    fn lookup_exact_match() {
        let mut reg = Registry::new(&test_config());
        set_backend_models(
            &mut reg,
            0,
            true,
            vec![make_model("fixt/home-3b-v3:latest")],
        );

        let id = reg.lookup("fixt/home-3b-v3:latest").unwrap();
        assert_eq!(reg.backend(id).name, "cuda");
    }

    #[test]
    fn lookup_prefix_match() {
        let mut reg = Registry::new(&test_config());
        set_backend_models(&mut reg, 1, true, vec![make_model("qwen3.5:latest")]);

        let id = reg.lookup("qwen3.5").unwrap();
        assert_eq!(reg.backend(id).name, "rocm");
    }

    #[test]
    fn lookup_exact_tag_preferred() {
        let mut reg = Registry::new(&test_config());
        set_backend_models(
            &mut reg,
            1,
            true,
            vec![make_model("qwen3.5:latest"), make_model("glm-4.7-flash")],
        );

        let id = reg.lookup("glm-4.7-flash").unwrap();
        assert_eq!(reg.backend(id).name, "rocm");
    }

    #[test]
    fn lookup_unknown_returns_none() {
        let mut reg = Registry::new(&test_config());
        set_backend_models(&mut reg, 0, true, vec![make_model("model:v1")]);
        assert!(reg.lookup("nonexistent").is_none());
    }

    #[test]
    fn available_models_returns_only_qualified_names() {
        let mut reg = Registry::new(&test_config());
        set_backend_models(&mut reg, 0, true, vec![make_model("a:v1")]);
        set_backend_models(&mut reg, 1, true, vec![make_model("b:latest")]);

        let mut available = reg.available_model_names();
        available.sort();
        assert_eq!(available, vec!["a:v1", "b:latest"]);
    }

    #[test]
    fn unhealthy_without_grace_excluded() {
        let mut reg = Registry::new(&test_config());
        reg.backends[0].healthy = false;
        reg.backends[0].grace_deadline = None;
        reg.backends[0].models = vec![make_model("orphan:v1")];
        reg.rebuild_model_map();

        assert!(reg.lookup("orphan:v1").is_none());
    }

    #[test]
    fn unhealthy_within_grace_included() {
        let mut reg = Registry::new(&test_config());
        reg.backends[0].healthy = false;
        reg.backends[0].grace_deadline = Some(Instant::now() + Duration::from_secs(60));
        reg.backends[0].models = vec![make_model("graced:v1")];
        reg.rebuild_model_map();

        let id = reg.lookup("graced:v1").unwrap();
        assert_eq!(reg.backend(id).name, "cuda");
    }

    #[test]
    fn empty_model_list_clears_previous() {
        let mut reg = Registry::new(&test_config());
        set_backend_models(&mut reg, 0, true, vec![make_model("old:v1")]);
        assert!(reg.lookup("old:v1").is_some());

        set_backend_models(&mut reg, 0, true, vec![]);
        assert!(reg.lookup("old:v1").is_none());
    }

    #[test]
    fn models_from_both_backends() {
        let mut reg = Registry::new(&test_config());
        reg.backends[0].healthy = true;
        reg.backends[0].models = vec![make_model("small:v1")];
        reg.backends[1].healthy = true;
        reg.backends[1].models = vec![make_model("large:v1")];
        reg.rebuild_model_map();

        assert_eq!(reg.backend(reg.lookup("small:v1").unwrap()).name, "cuda");
        assert_eq!(reg.backend(reg.lookup("large:v1").unwrap()).name, "rocm");
    }

    #[test]
    fn duplicate_model_first_backend_wins() {
        let mut reg = Registry::new(&test_config());
        reg.backends[0].healthy = true;
        reg.backends[0].models = vec![make_model("shared:latest")];
        reg.backends[1].healthy = true;
        reg.backends[1].models = vec![make_model("shared:latest")];
        reg.rebuild_model_map();

        // First backend in config wins for both exact and prefix lookups.
        let exact = reg.lookup("shared:latest").unwrap();
        assert_eq!(reg.backend(exact).name, "cuda");
        let prefix = reg.lookup("shared").unwrap();
        assert_eq!(reg.backend(prefix).name, "cuda");
    }

    #[test]
    fn any_healthy_returns_first_healthy() {
        let mut reg = Registry::new(&test_config());
        reg.backends[0].healthy = true;
        reg.backends[1].healthy = true;

        let id = reg.any_healthy().unwrap();
        assert_eq!(reg.backend(id).name, "cuda");
    }

    #[test]
    fn any_healthy_skips_unhealthy() {
        let mut reg = Registry::new(&test_config());
        reg.backends[0].healthy = false;
        reg.backends[1].healthy = true;

        let id = reg.any_healthy().unwrap();
        assert_eq!(reg.backend(id).name, "rocm");
    }

    #[test]
    fn any_healthy_none_when_all_down() {
        let reg = Registry::new(&test_config());
        assert!(reg.any_healthy().is_none());
    }

    // sanitize_model_entry: ensures pydantic-strict clients (HA's ollama
    // integration) can parse the merged /api/tags even when an upstream
    // backend (llama.cpp's compat layer) returns empty strings for fields
    // pydantic expects to be datetime/byte-size.

    #[test]
    fn sanitize_replaces_empty_modified_at() {
        let mut m = ModelInfo {
            name: "x".to_string(),
            extra: serde_json::json!({"modified_at": "", "size": 123u64}),
        };
        super::sanitize_model_entry(&mut m);
        assert_eq!(
            m.extra.get("modified_at").and_then(|v| v.as_str()),
            Some("1970-01-01T00:00:00Z")
        );
        assert_eq!(m.extra.get("size").and_then(|v| v.as_u64()), Some(123));
    }

    #[test]
    fn sanitize_replaces_empty_size() {
        let mut m = ModelInfo {
            name: "x".to_string(),
            extra: serde_json::json!({"modified_at": "2026-01-01T00:00:00Z", "size": ""}),
        };
        super::sanitize_model_entry(&mut m);
        assert_eq!(m.extra.get("size").and_then(|v| v.as_u64()), Some(0));
    }

    #[test]
    fn sanitize_fills_missing_fields() {
        let mut m = ModelInfo {
            name: "x".to_string(),
            extra: serde_json::json!({}),
        };
        super::sanitize_model_entry(&mut m);
        assert!(m.extra.get("modified_at").is_some());
        assert_eq!(m.extra.get("size").and_then(|v| v.as_u64()), Some(0));
    }

    #[test]
    fn sanitize_leaves_well_formed_entry_alone() {
        let original = serde_json::json!({
            "modified_at": "2026-05-19T12:34:56Z",
            "size": 9_608_350_718u64,
            "digest": "abc123",
        });
        let mut m = ModelInfo {
            name: "gemma:2b".to_string(),
            extra: original.clone(),
        };
        super::sanitize_model_entry(&mut m);
        assert_eq!(m.extra, original);
    }

    #[test]
    fn sanitize_noop_when_extra_is_not_an_object() {
        // serde(flatten) over a primitive value never appears in practice
        // (TagsResponse deserialise would fail first), but guard the
        // function against panicking if it ever did.
        let mut m = ModelInfo {
            name: "x".to_string(),
            extra: serde_json::Value::String("not-an-object".to_string()),
        };
        super::sanitize_model_entry(&mut m);
        assert_eq!(
            m.extra,
            serde_json::Value::String("not-an-object".to_string())
        );
    }
}

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::config::{Alias, Config};

/// Opaque index into the backends array. Not constructable outside this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BackendId(usize);

/// Wire protocol spoken by a backend. Set by discovery from whichever
/// listing endpoint succeeded; defaults to `Ollama` until the first cycle
/// proves otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendProtocol {
    /// Backend speaks Ollama-native /api/* (and usually /v1/* too).
    Ollama,
    /// Backend only speaks /v1/* (e.g. llama.cpp's llama-server, llama-swap).
    OpenAi,
}

/// Per-backend mutable state, updated by the discovery loop.
#[derive(Debug)]
struct BackendState {
    name: String,
    url: String,
    healthy: bool,
    protocol: BackendProtocol,
    models: Vec<ModelInfo>,
    last_seen: Option<Instant>,
    grace_deadline: Option<Instant>,
    strip_auth: bool,
}

impl BackendState {
    fn is_reachable(&self) -> bool {
        self.healthy || self.grace_deadline.is_some()
    }

    /// True when this backend carries `model` by exact name or by `:`-prefix.
    fn serves(&self, model: &str) -> bool {
        self.view().serves(model)
    }

    fn view(&self) -> BackendView<'_> {
        BackendView {
            name: &self.name,
            url: &self.url,
            healthy: self.healthy,
            protocol: self.protocol,
            models: &self.models,
            in_grace_period: self.grace_deadline.is_some(),
            strip_auth: self.strip_auth,
        }
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
    pub protocol: BackendProtocol,
    pub models: &'a [ModelInfo],
    pub in_grace_period: bool,
    pub strip_auth: bool,
}

impl BackendView<'_> {
    /// True when this backend carries `model` by exact name or by `:`-prefix.
    /// Mirrors the key generation in `rebuild_model_map`, so it agrees with
    /// what `lookup` would resolve.
    pub fn serves(&self, model: &str) -> bool {
        self.models
            .iter()
            .any(|m| m.name == model || m.name.split_once(':').map(|(p, _)| p) == Some(model))
    }
}

/// The central routing table. All access goes through `SharedRegistry`.
///
/// Invariant: `model_map` keys stay in sync with `backends[*].models`
/// via `rebuild_model_map`, called at the end of every discovery cycle.
pub struct Registry {
    backends: Vec<BackendState>,
    model_map: HashMap<String, BackendId>,
    /// `local-model → stand-in-model` rewrites for when no reachable backend
    /// serves the requested model. Refreshed from `OLLAMA_ROUTER_FALLBACK_FILE`
    /// each discovery cycle; empty when the file isn't configured.
    fallbacks: HashMap<String, String>,
    /// Alias → priority chain of `backend/model` candidates. Refreshed from
    /// `OLLAMA_ROUTER_ALIASES_FILE` each discovery cycle; empty when the
    /// file isn't configured. A distinct namespace from `model_map`: an
    /// alias is resolved before (and shadows) any concrete model lookup.
    aliases: HashMap<String, Alias>,
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
                protocol: BackendProtocol::Ollama,
                models: Vec::new(),
                last_seen: None,
                grace_deadline: None,
                strip_auth: b.strip_auth,
            })
            .collect();

        Registry {
            backends,
            model_map: HashMap::new(),
            fallbacks: HashMap::new(),
            aliases: HashMap::new(),
            discovery_done: false,
        }
    }

    /// The configured stand-in for `model`, if any. Callers decide whether
    /// the hop applies (only when `model` itself resolves to no backend).
    pub fn fallback_for(&self, model: &str) -> Option<&str> {
        self.fallbacks.get(model).map(String::as_str)
    }

    /// The alias chain registered under `name`, if any, together with its
    /// canonical (as-configured) alias name — callers label metrics/spans
    /// with the canonical name so `fast` and `fast:latest` don't split.
    ///
    /// Ollama clients normalise bare model names to `name:latest`, so an
    /// exact miss falls back to the name with a trailing `:latest`
    /// stripped. Exact matches win: an alias literally named `foo:latest`
    /// resolves before (and is never shadowed by) an alias named `foo`.
    pub fn alias_for(&self, name: &str) -> Option<(&str, &Alias)> {
        if let Some((k, v)) = self.aliases.get_key_value(name) {
            return Some((k.as_str(), v));
        }
        let stripped = name.strip_suffix(":latest")?;
        self.aliases
            .get_key_value(stripped)
            .map(|(k, v)| (k.as_str(), v))
    }

    /// All configured alias names, in arbitrary order.
    pub fn alias_names(&self) -> impl Iterator<Item = &str> {
        self.aliases.keys().map(String::as_str)
    }

    /// Look up a backend by its configured name. Backend membership is fixed
    /// at startup, so for names validated against the config (alias
    /// candidates) this always resolves.
    pub fn backend_id_by_name(&self, name: &str) -> Option<BackendId> {
        self.backends
            .iter()
            .position(|b| b.name == name)
            .map(BackendId)
    }

    /// Look up a model by exact name, then by prefix (before `:`) if no exact match.
    pub fn lookup(&self, model: &str) -> Option<BackendId> {
        self.model_map.get(model).copied()
    }

    /// Backend ids matching `pred`, in config-declaration order.
    fn filter_backends(&self, pred: impl Fn(&BackendState) -> bool) -> Vec<BackendId> {
        self.backends
            .iter()
            .enumerate()
            .filter(|(_, b)| pred(b))
            .map(|(idx, _)| BackendId(idx))
            .collect()
    }

    /// All **healthy** backends serving `model`, in routing-priority order
    /// (config declaration order). This puts the `lookup` primary first
    /// whenever it is healthy — the model_map's first-writer pick is, by
    /// construction, the earliest reachable serving backend, so when it is
    /// healthy it is also the earliest healthy one.
    ///
    /// Backends that are unhealthy — **including those merely in their grace
    /// period** — are excluded. This is the failover-candidate set, so a
    /// backend we currently believe is unreachable must not appear in it.
    /// An empty result is the signal (for Unit 3) to shed load as honest
    /// backpressure rather than hammer a down backend.
    ///
    /// For single-homed models this returns exactly one backend; the
    /// multi-homed case is rare (the motivating embedder is single-homed).
    pub fn healthy_backends_for(&self, model: &str) -> Vec<BackendId> {
        self.filter_backends(|b| b.healthy && b.serves(model))
    }

    /// Return the first healthy backend for model-less request proxying.
    pub fn any_healthy(&self) -> Option<BackendId> {
        self.filter_backends(|b| b.healthy).into_iter().next()
    }

    /// Borrow a backend's view by id.
    ///
    /// Checked, not `self.backends[id.0]`: `BackendId` is a raw index, and
    /// once the roster is rebuilt on a config reload an id snapshotted from
    /// an earlier read can name a slot that no longer exists. A stale id
    /// must degrade to an error, not abort the process.
    pub fn backend(&self, id: BackendId) -> Option<BackendView<'_>> {
        self.backends.get(id.0).map(BackendState::view)
    }

    /// Iterate over all backends.
    pub fn all_backends(&self) -> impl Iterator<Item = BackendView<'_>> {
        self.backends.iter().map(BackendState::view)
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
    let builder = match config.apply_extra_ca(Client::builder().timeout(Duration::from_secs(10))) {
        Ok(builder) => builder,
        Err(e) => {
            warn!(error = %e, "invalid extra CA for discovery client; discovery disabled");
            return;
        }
    };
    let client = match builder.build() {
        Ok(client) => client,
        Err(e) => {
            // Discovery can't run without an HTTP client. Log and bail out of
            // this background task rather than aborting the whole process —
            // the router can still proxy to statically-configured backends.
            warn!(error = %e, "failed to build discovery HTTP client; discovery disabled");
            return;
        }
    };

    let interval = Duration::from_secs(config.discovery_interval_secs);
    let grace_duration = Duration::from_secs(config.grace_period_secs());
    let mut config = config;

    reload_file_config(&mut config, &registry).await;
    run_discovery(&client, &registry, &config, grace_duration).await;

    loop {
        tokio::time::sleep(interval).await;
        reload_file_config(&mut config, &registry).await;
        run_discovery(&client, &registry, &config, grace_duration).await;
    }
}

/// Re-read the allowlist and fallback files (when configured) so mounted
/// ConfigMap edits land within one discovery cycle, no restart needed.
/// Any read/parse error keeps the previous cycle's values: wiping the
/// allowlist on a bad edit would blow the spend boundary open, and wiping
/// fallbacks would silently drop failover.
async fn reload_file_config(config: &mut crate::config::Config, registry: &SharedRegistry) {
    if let Some(path) = config.model_allow_file.clone() {
        match crate::config::load_model_allow_file(&path) {
            Ok(allow) => {
                if let Err(unknown) = crate::config::apply_model_allow(&mut config.backends, allow)
                {
                    warn!(
                        backend = %unknown,
                        file = %path,
                        "allowlist file names an unknown backend; keeping previous allowlist"
                    );
                }
            }
            Err(e) => {
                warn!(error = %e, file = %path, "failed to reload allowlist file; keeping previous allowlist");
            }
        }
    }
    if let Some(path) = config.fallback_file.clone() {
        match crate::config::load_fallbacks_file(&path) {
            Ok(fallbacks) => registry.write().await.fallbacks = fallbacks,
            Err(e) => {
                warn!(error = %e, file = %path, "failed to reload fallback file; keeping previous fallbacks");
            }
        }
    }
    if let Some(path) = config.aliases_file.clone() {
        match crate::config::load_aliases_file(&path, &config.backends, &config.external_backends) {
            Ok(aliases) => registry.write().await.aliases = aliases,
            Err(e) => {
                warn!(error = %e, file = %path, "failed to reload aliases file; keeping previous aliases");
            }
        }
    }
}

/// Fetch results from backends, keyed by index.
enum FetchResult {
    Ok {
        protocol: BackendProtocol,
        models: Vec<ModelInfo>,
    },
    Err,
}

/// Build a successful `FetchResult` from sanitized models.
fn fetch_ok(protocol: BackendProtocol, mut models: Vec<ModelInfo>) -> FetchResult {
    for m in &mut models {
        sanitize_model_entry(m);
    }
    FetchResult::Ok { protocol, models }
}

/// Log a fetch-phase error and return `FetchResult::Err`.
fn fetch_err(name: &str, error: &dyn std::fmt::Display, context: &str) -> FetchResult {
    warn!(backend = %name, error = %error, "{context}");
    FetchResult::Err
}

/// Try Ollama `/api/tags` first, then fall back to OpenAI `/v1/models`.
/// The successful endpoint determines `protocol`: an `/api/tags` 200 means
/// the backend speaks Ollama-native; only `/v1/models` succeeding marks it
/// as OpenAI-only.
async fn fetch_models(client: &Client, name: &str, url: &str) -> FetchResult {
    // Try Ollama /api/tags
    let tags_url = format!("{url}/api/tags");
    match client.get(&tags_url).send().await {
        Ok(resp) if resp.status().is_success() => {
            return match resp.json::<TagsResponse>().await {
                Ok(tags) => fetch_ok(BackendProtocol::Ollama, tags.models.unwrap_or_default()),
                Err(e) => fetch_err(name, &e, "failed to parse /api/tags"),
            };
        }
        Ok(_) => {} // non-success — try OpenAI fallback
        Err(e) => return fetch_err(name, &e, "failed to reach backend"),
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
                    .map(|m| ModelInfo {
                        name: m.id,
                        extra: m.extra,
                    })
                    .collect();
                fetch_ok(BackendProtocol::OpenAi, models)
            }
            Err(e) => fetch_err(name, &e, "failed to parse /v1/models"),
        },
        Ok(resp) => fetch_err(
            name,
            &resp.status(),
            "unhealthy backend (tried /api/tags and /v1/models)",
        ),
        Err(e) => fetch_err(name, &e, "failed to reach backend"),
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
            serde_json::Value::String(crate::translate::FIXED_TIMESTAMP.to_string()),
        );
    }
    if obj.get("size").is_none_or(is_blank) {
        obj.insert("size".to_string(), serde_json::json!(0));
    }
    if !obj.contains_key("model") {
        obj.insert(
            "model".to_string(),
            serde_json::Value::String(m.name.clone()),
        );
    }
}

async fn run_discovery(
    client: &Client,
    registry: &SharedRegistry,
    config: &Config,
    grace_duration: Duration,
) {
    // Phase 1: Fetch from all backends concurrently, WITHOUT holding any
    // lock. Backend URLs come from Config (immutable), so no lock needed.
    // join_all preserves order, so results still zip with `reg.backends`.
    // Concurrency matters: a slow/dead backend's per-probe timeout would
    // otherwise serialise, making a cycle take N × timeout.
    let mut fetch_results = futures_util::future::join_all(
        config
            .backends
            .iter()
            .map(|backend| fetch_models(client, &backend.name, &backend.url)),
    )
    .await;

    // Phase 1b: Apply per-backend discovery allowlists. Done here rather than
    // in `fetch_models` so the filter is visible next to the fetch it trims,
    // and so an allowlisted backend that returns nothing recognisable still
    // reports as reachable-but-empty rather than as an error.
    for (backend, result) in config.backends.iter().zip(fetch_results.iter_mut()) {
        let (Some(allow), FetchResult::Ok { models, .. }) = (&backend.allow_models, result) else {
            continue;
        };
        let before = models.len();
        models.retain(|m| allow.contains(&m.name));
        if models.is_empty() && before > 0 {
            warn!(
                backend = %backend.name,
                advertised = before,
                "allowlist matched none of the backend's models; check OLLAMA_ROUTER_MODEL_ALLOW spelling"
            );
        }
    }

    // Phase 2: Apply results under write lock (no I/O, microseconds).
    let mut reg = registry.write().await;
    let now = Instant::now();

    for (backend, result) in reg.backends.iter_mut().zip(fetch_results) {
        match result {
            FetchResult::Ok { protocol, models } => {
                if !backend.healthy {
                    info!(backend = %backend.name, models = models.len(), protocol = ?protocol, "backend recovered");
                }
                if backend.protocol != protocol {
                    info!(backend = %backend.name, from = ?backend.protocol, to = ?protocol, "backend protocol updated");
                }
                backend.healthy = true;
                backend.protocol = protocol;
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
        make_model_with_extra(name, serde_json::Value::Object(serde_json::Map::new()))
    }

    /// Set backend health, grace deadline, models, and rebuild the model map.
    fn setup_backend(
        reg: &mut Registry,
        idx: usize,
        healthy: bool,
        grace: Option<Duration>,
        models: Vec<ModelInfo>,
    ) {
        reg.backends[idx].healthy = healthy;
        reg.backends[idx].grace_deadline = grace.map(|d| Instant::now() + d);
        reg.backends[idx].models = models;
        reg.rebuild_model_map();
    }

    #[test]
    fn fallback_map_covers_models_of_unreachable_backends() {
        let mut reg = Registry::new(&test_config());
        // cuda is up serving the stand-in; rocm (which served the local
        // model) is down, so the local name has vanished from the map.
        setup_backend(
            &mut reg,
            0,
            true,
            None,
            vec![make_model("qwen/qwen3.8-27b")],
        );
        reg.fallbacks =
            HashMap::from([("qwen3.6-medium".to_string(), "qwen/qwen3.8-27b".to_string())]);

        assert!(reg.lookup("qwen3.6-medium").is_none());
        assert_eq!(reg.fallback_for("qwen3.6-medium"), Some("qwen/qwen3.8-27b"));
        assert_lookup_name(&reg, "qwen/qwen3.8-27b", "cuda");
        assert_eq!(reg.fallback_for("unmapped"), None);
    }

    #[test]
    fn alias_for_and_backend_id_by_name() {
        let mut reg = Registry::new(&test_config());
        reg.aliases = HashMap::from([(
            "fast".to_string(),
            Alias {
                local_only: false,
                candidates: vec![crate::config::AliasCandidate {
                    backend: "cuda".to_string(),
                    model: "qwen3.6:latest".to_string(),
                }],
            },
        )]);

        let (canonical, alias) = reg.alias_for("fast").expect("alias should resolve");
        assert_eq!(canonical, "fast");
        assert_eq!(alias.candidates[0].backend, "cuda");
        assert!(reg.alias_for("qwen3.6:latest").is_none());
        assert_eq!(reg.alias_names().collect::<Vec<_>>(), vec!["fast"]);

        // Ollama clients normalise bare names to `name:latest`; the alias
        // must resolve under both spellings, with the canonical name kept
        // for metric labels.
        let (canonical, _) = reg
            .alias_for("fast:latest")
            .expect("`:latest` form should resolve");
        assert_eq!(canonical, "fast");
        // Only `:latest` is stripped — other tags stay misses.
        assert!(reg.alias_for("fast:v2").is_none());

        let id = reg
            .backend_id_by_name("rocm")
            .expect("configured backend should resolve");
        assert_eq!(reg.backend(id).expect("live id").name, "rocm");
        assert!(reg.backend_id_by_name("typo").is_none());
    }

    #[tokio::test]
    async fn alias_reload_keeps_previous_on_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("aliases");
        std::fs::write(&path, "fast = cuda/qwen3.6:latest | rocm/glm-5.2\n").expect("write");

        let mut config = test_config();
        config.aliases_file = Some(path.to_string_lossy().into_owned());
        let registry = new_shared(&config);

        reload_file_config(&mut config, &registry).await;
        assert_eq!(
            registry
                .read()
                .await
                .alias_for("fast")
                .expect("alias loaded")
                .1
                .candidates
                .len(),
            2
        );

        // A bad edit (unknown backend) must keep the previous chain intact.
        std::fs::write(&path, "fast = typo/qwen3.6:latest\n").expect("write");
        reload_file_config(&mut config, &registry).await;
        let reg = registry.read().await;
        let (_, alias) = reg.alias_for("fast").expect("previous alias survives");
        assert_eq!(alias.candidates[0].backend, "cuda");
    }

    /// Assert that `model` resolves to a backend named `expected_name`.
    #[track_caller]
    fn assert_lookup_name(reg: &Registry, model: &str, expected_name: &str) {
        let id = reg
            .lookup(model)
            .unwrap_or_else(|| panic!("lookup({model:?}) returned None"));
        assert_eq!(reg.backend(id).expect("live id").name, expected_name);
    }

    /// Collect backend names for all healthy backends serving `model`.
    fn backend_names_for<'a>(reg: &'a Registry, model: &str) -> Vec<&'a str> {
        reg.healthy_backends_for(model)
            .iter()
            .filter_map(|&id| reg.backend(id).map(|b| b.name))
            .collect()
    }

    /// Construct a `ModelInfo` with custom `extra` metadata.
    fn make_model_with_extra(name: &str, extra: serde_json::Value) -> ModelInfo {
        ModelInfo {
            name: name.to_string(),
            extra,
        }
    }

    /// Construct, sanitize, and return a `ModelInfo`.
    fn sanitized(name: &str, extra: serde_json::Value) -> ModelInfo {
        let mut m = make_model_with_extra(name, extra);
        super::sanitize_model_entry(&mut m);
        m
    }

    /// Create a fresh registry with one healthy backend serving `models`.
    fn reg_with_healthy(idx: usize, models: &[&str]) -> Registry {
        let mut reg = Registry::new(&test_config());
        setup_backend(
            &mut reg,
            idx,
            true,
            None,
            models.iter().map(|n| make_model(n)).collect(),
        );
        reg
    }

    /// Set up both backends serving the same model, with backend 0 in a
    /// given health/grace state and backend 1 healthy.
    fn reg_dual_serving(model: &str, b0_healthy: bool, b0_grace: Option<Duration>) -> Registry {
        let mut reg = Registry::new(&test_config());
        setup_backend(&mut reg, 0, b0_healthy, b0_grace, vec![make_model(model)]);
        setup_backend(&mut reg, 1, true, None, vec![make_model(model)]);
        reg
    }

    /// Set up both backends with no models, only health flags.
    fn reg_health_only(b0_healthy: bool, b1_healthy: bool) -> Registry {
        let mut reg = Registry::new(&test_config());
        setup_backend(&mut reg, 0, b0_healthy, None, vec![]);
        setup_backend(&mut reg, 1, b1_healthy, None, vec![]);
        reg
    }

    #[test]
    fn new_registry_starts_empty() {
        let reg = Registry::new(&test_config());
        assert!(!reg.is_discovery_done());
        assert!(reg.model_map.is_empty());
    }

    #[test]
    fn lookup_exact_match() {
        let reg = reg_with_healthy(0, &["fixt/home-3b-v3:latest"]);
        assert_lookup_name(&reg, "fixt/home-3b-v3:latest", "cuda");
    }

    #[test]
    fn lookup_prefix_match() {
        let reg = reg_with_healthy(1, &["qwen3.5:latest"]);
        assert_lookup_name(&reg, "qwen3.5", "rocm");
    }

    #[test]
    fn lookup_exact_tag_preferred() {
        let reg = reg_with_healthy(1, &["qwen3.5:latest", "glm-4.7-flash"]);
        assert_lookup_name(&reg, "glm-4.7-flash", "rocm");
    }

    #[test]
    fn lookup_unknown_returns_none() {
        let reg = reg_with_healthy(0, &["model:v1"]);
        assert!(reg.lookup("nonexistent").is_none());
    }

    #[test]
    fn healthy_backends_for_single_homed_returns_one() {
        let reg = reg_with_healthy(0, &["jina-embed:v1"]);

        assert_eq!(backend_names_for(&reg, "jina-embed:v1"), vec!["cuda"]);
        // Prefix form resolves to the same single backend.
        assert_eq!(backend_names_for(&reg, "jina-embed"), vec!["cuda"]);
    }

    #[test]
    fn healthy_backends_for_multi_homed_lists_primary_first() {
        let reg = reg_dual_serving("shared:v1", true, None);

        // cuda is first in config order and is the lookup primary.
        assert_eq!(backend_names_for(&reg, "shared:v1"), vec!["cuda", "rocm"]);
        assert_lookup_name(&reg, "shared:v1", "cuda");
    }

    #[test]
    fn healthy_backends_for_excludes_unhealthy_serving_backend() {
        // cuda serves the model but is hard-down (no grace); rocm is healthy.
        let reg = reg_dual_serving("m:v1", false, None);
        assert_eq!(backend_names_for(&reg, "m:v1"), vec!["rocm"]);
    }

    #[test]
    fn healthy_backends_for_excludes_grace_period_backend() {
        // cuda is in its grace period: still routable via `lookup`, but NOT a
        // healthy failover candidate.
        let reg = reg_dual_serving("graced:v1", false, Some(Duration::from_secs(60)));

        // lookup still finds the graced backend (first reachable)...
        assert_lookup_name(&reg, "graced:v1", "cuda");
        // ...but healthy_backends_for excludes it.
        assert_eq!(backend_names_for(&reg, "graced:v1"), vec!["rocm"]);
    }

    #[test]
    fn healthy_backends_for_absent_model_is_empty() {
        let reg = reg_with_healthy(0, &["present:v1"]);
        assert!(reg.healthy_backends_for("absent").is_empty());
    }

    #[test]
    fn available_models_returns_only_qualified_names() {
        let mut reg = Registry::new(&test_config());
        setup_backend(&mut reg, 0, true, None, vec![make_model("a:v1")]);
        setup_backend(&mut reg, 1, true, None, vec![make_model("b:latest")]);

        let mut available = reg.available_model_names();
        available.sort();
        assert_eq!(available, vec!["a:v1", "b:latest"]);
    }

    #[test]
    fn unhealthy_without_grace_excluded() {
        let mut reg = Registry::new(&test_config());
        setup_backend(&mut reg, 0, false, None, vec![make_model("orphan:v1")]);

        assert!(reg.lookup("orphan:v1").is_none());
    }

    #[test]
    fn unhealthy_within_grace_included() {
        let mut reg = Registry::new(&test_config());
        setup_backend(
            &mut reg,
            0,
            false,
            Some(Duration::from_secs(60)),
            vec![make_model("graced:v1")],
        );

        assert_lookup_name(&reg, "graced:v1", "cuda");
    }

    #[test]
    fn empty_model_list_clears_previous() {
        let reg = reg_with_healthy(0, &["old:v1"]);
        assert!(reg.lookup("old:v1").is_some());

        let mut reg = reg;
        setup_backend(&mut reg, 0, true, None, vec![]);
        assert!(reg.lookup("old:v1").is_none());
    }

    #[test]
    fn models_from_both_backends() {
        let mut reg = Registry::new(&test_config());
        setup_backend(&mut reg, 0, true, None, vec![make_model("small:v1")]);
        setup_backend(&mut reg, 1, true, None, vec![make_model("large:v1")]);

        assert_lookup_name(&reg, "small:v1", "cuda");
        assert_lookup_name(&reg, "large:v1", "rocm");
    }

    #[test]
    fn duplicate_model_first_backend_wins() {
        let reg = reg_dual_serving("shared:latest", true, None);

        // First backend in config wins for both exact and prefix lookups.
        assert_lookup_name(&reg, "shared:latest", "cuda");
        assert_lookup_name(&reg, "shared", "cuda");
    }

    #[test]
    fn any_healthy_returns_first_healthy() {
        let reg = reg_health_only(true, true);
        let id = reg.any_healthy().expect("a healthy backend");
        assert_eq!(reg.backend(id).expect("live id").name, "cuda");
    }

    #[test]
    fn any_healthy_skips_unhealthy() {
        let reg = reg_health_only(false, true);
        let id = reg.any_healthy().expect("a healthy backend");
        assert_eq!(reg.backend(id).expect("live id").name, "rocm");
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
        let m = sanitized("x", serde_json::json!({"modified_at": "", "size": 123u64}));
        assert_eq!(
            m.extra.get("modified_at").and_then(|v| v.as_str()),
            Some("1970-01-01T00:00:00Z")
        );
        assert_eq!(m.extra.get("size").and_then(|v| v.as_u64()), Some(123));
    }

    #[test]
    fn sanitize_replaces_empty_size() {
        let input = serde_json::json!({"modified_at": "2026-01-01T00:00:00Z", "size": ""});
        let m = sanitized("x", input);
        assert_eq!(m.extra.get("size").and_then(|v| v.as_u64()), Some(0));
    }

    #[test]
    fn sanitize_fills_missing_fields() {
        let m = sanitized("x", serde_json::json!({}));
        assert!(m.extra.get("modified_at").is_some());
        assert_eq!(m.extra.get("size").and_then(|v| v.as_u64()), Some(0));
        // Regression for HA's `KeyError: 'model'` (commit 0141020): when
        // `/api/tags` from a non-Ollama backend (llama.cpp openai-compat
        // server, llama-swap) omits `model`, sanitize must back-fill it
        // from `name`. Without this assertion, deleting the back-fill
        // branch still passes the test suite.
        assert_eq!(
            m.extra.get("model").and_then(|v| v.as_str()),
            Some("x"),
            "sanitize must back-fill `model` from `name` when absent",
        );
    }

    #[test]
    fn sanitize_does_not_overwrite_present_model_field() {
        // Sibling invariant: if the backend already provided a `model`
        // field (even one that disagrees with `name`), sanitize leaves
        // it alone. This is the contract for the no-op-on-well-formed
        // path that `sanitize_leaves_well_formed_entry_alone` exercises,
        // pinned explicitly so refactors don't silently change behaviour.
        let m = sanitized("x", serde_json::json!({"model": "y"}));
        assert_eq!(m.extra.get("model").and_then(|v| v.as_str()), Some("y"));
    }

    #[test]
    fn sanitize_leaves_well_formed_entry_alone() {
        let original = serde_json::json!({
            "model": "gemma:2b",
            "modified_at": "2026-05-19T12:34:56Z",
            "size": 9_608_350_718u64,
            "digest": "abc123",
        });
        let m = sanitized("gemma:2b", original.clone());
        assert_eq!(m.extra, original);
    }

    #[test]
    fn sanitize_noop_when_extra_is_not_an_object() {
        // serde(flatten) over a primitive value never appears in practice
        // (TagsResponse deserialise would fail first), but guard the
        // function against panicking if it ever did.
        let not_an_object = serde_json::Value::String("not-an-object".to_string());
        let m = sanitized("x", not_an_object.clone());
        assert_eq!(m.extra, not_an_object);
    }
}

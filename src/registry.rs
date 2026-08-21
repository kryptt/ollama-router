use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::config::Config;
use crate::metrics::{BackendLabels, ConfigReloadLabels, Metrics};
use crate::policy::{Alias, BackendSpec, FileConfig, PolicyError, Validated};

/// Ceiling on a single policy-file read. Matches the discovery client's
/// timeout: both are "this filesystem/host is not answering" budgets, and a
/// reload that takes longer than a full round of backend probes is wedged,
/// not slow.
const POLICY_READ_TIMEOUT: Duration = Duration::from_secs(10);

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

/// Per-backend state. Split in two by lifecycle: `name`/`url`/`strip_auth`/
/// `allow_models` come from the policy file and are replaced wholesale on
/// every reload, while `healthy`/`protocol`/`models`/`last_seen`/
/// `grace_deadline` are *learned* by discovery and carried across a reload
/// whenever the backend is recognisably the same server.
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
    /// Discovery allowlist, owned by the backend itself rather than looked
    /// up positionally in a parallel `Vec`. `None` = publish everything.
    ///
    /// This is where the two-Vec hazard died: pairing a filter with the
    /// wrong backend is now unrepresentable, and for a metered backend that
    /// pairing IS the spend boundary.
    allow_models: Option<HashSet<String>>,
}

impl BackendState {
    /// A backend with no learned state yet: unhealthy, protocol assumed
    /// Ollama until a probe proves otherwise, no models.
    fn fresh(spec: BackendSpec) -> Self {
        BackendState {
            name: spec.name,
            url: spec.url,
            healthy: false,
            protocol: BackendProtocol::Ollama,
            models: Vec::new(),
            last_seen: None,
            grace_deadline: None,
            strip_auth: spec.strip_auth,
            allow_models: spec.allow_models,
        }
    }

    /// Drop every model this backend's `allow` list does not publish.
    /// A filtered-out model is not merely hidden: it never enters the model
    /// map, so requests for it 404 like any unknown model.
    fn retain_allowed(&mut self) {
        let Self {
            allow_models: Some(allow),
            models,
            name,
            ..
        } = self
        else {
            return;
        };
        let before = models.len();
        models.retain(|m| allow.contains(&m.name));
        if models.is_empty() && before > 0 {
            warn!(
                backend = %name,
                advertised = before,
                "allowlist matched none of the backend's models; check the `allow` spelling in router.toml"
            );
        }
    }

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
///
/// `Default` is the empty roster — see `new_shared_empty`.
#[derive(Default)]
pub struct Registry {
    backends: Vec<BackendState>,
    model_map: HashMap<String, BackendId>,
    /// `local-model → stand-in-model` rewrites for when no reachable backend
    /// serves the requested model. Refreshed from `router.toml` each
    /// discovery cycle.
    fallbacks: HashMap<String, String>,
    /// Alias → priority chain of `backend/model` candidates. Refreshed from
    /// `router.toml` each discovery cycle. A distinct namespace from
    /// `model_map`: an alias is resolved before (and shadows) any concrete
    /// model lookup.
    aliases: HashMap<String, Alias>,
    discovery_done: bool,
}

impl Registry {
    fn new(validated: Validated) -> Self {
        let mut registry = Registry {
            backends: Vec::new(),
            model_map: HashMap::new(),
            fallbacks: HashMap::new(),
            aliases: HashMap::new(),
            discovery_done: false,
        };
        // Nothing to remove from an empty roster, so the returned names are
        // necessarily empty here.
        drop(registry.apply_file_config(validated));
        registry
    }

    /// Swap in a freshly validated policy document — roster, fallbacks and
    /// aliases — in **one** write-lock section.
    ///
    /// Returns the names of backends that disappeared, so the caller can
    /// drop their `backend_up` gauge series; a gauge for a deleted backend
    /// otherwise freezes at its last value and keeps claiming it is up.
    ///
    /// Everything here is deliberately in one function: `model_map` values
    /// are `BackendId` indices into `backends`, so mutating the roster
    /// anywhere other than immediately before `rebuild_model_map` is a
    /// misroute waiting to happen.
    pub fn apply_file_config(&mut self, validated: Validated) -> Vec<String> {
        let Validated {
            backends,
            fallbacks,
            aliases,
        } = validated;

        // Keyed by name, because position is exactly what must be free to
        // change: document order IS routing priority, so a reorder is a
        // legitimate edit and must not be mistaken for a rename.
        let mut previous: HashMap<String, BackendState> = self
            .backends
            .drain(..)
            .map(|b| (b.name.clone(), b))
            .collect();

        self.backends = backends
            .into_iter()
            .map(|spec| match previous.remove(&spec.name) {
                Some(prev) if prev.url == spec.url => {
                    // Same name and same URL: the same server. Carry the
                    // learned state. Rebuilding it instead would revert
                    // `protocol` to Ollama (which hangs `/api/chat` against
                    // an OpenAI-only backend), empty `models` (a 404 storm
                    // until the next fetch lands) and clear the grace
                    // deadline (losing an in-progress outage's grace).
                    let mut next = BackendState {
                        healthy: prev.healthy,
                        protocol: prev.protocol,
                        models: prev.models,
                        last_seen: prev.last_seen,
                        grace_deadline: prev.grace_deadline,
                        ..BackendState::fresh(spec)
                    };
                    // Apply the (possibly tightened) allowlist to the models
                    // being carried over rather than waiting for the next
                    // fetch: the spend boundary must never lag its own edit.
                    next.retain_allowed();
                    next
                }
                Some(prev) => {
                    // Same name, different URL: a different server wearing a
                    // familiar name. Say so loudly — silently inheriting the
                    // old one's model list would publish models the new
                    // address may not serve.
                    info!(
                        backend = %spec.name,
                        from = %prev.url,
                        to = %spec.url,
                        "backend url changed; resetting learned state",
                    );
                    BackendState::fresh(spec)
                }
                None => {
                    info!(backend = %spec.name, url = %spec.url, "backend added by config reload");
                    BackendState::fresh(spec)
                }
            })
            .collect();

        // Whatever the new roster did not claim is a removal.
        let mut removed: Vec<String> = previous.into_keys().collect();
        removed.sort();
        for name in &removed {
            info!(backend = %name, "backend removed by config reload");
        }

        // Fallbacks and aliases were validated against *this* roster, in the
        // same pass, so they land in the same section: a reader can never
        // observe new aliases against the old backend list.
        self.fallbacks = fallbacks;
        self.aliases = aliases;
        self.rebuild_model_map();
        removed
    }

    /// Fold a discovery cycle's results into the roster, **keyed by name**.
    fn apply_fetch_results(
        &mut self,
        results: Vec<(String, FetchResult)>,
        now: Instant,
        grace_duration: Duration,
    ) {
        let mut by_name: HashMap<String, FetchResult> = results.into_iter().collect();

        for backend in &mut self.backends {
            let Some(result) = by_name.remove(&backend.name) else {
                // Added by the reload that ran after the targets were
                // snapshotted, or otherwise unprobed this cycle. Leave it
                // untouched: marking it down is a claim we have no evidence
                // for, and would cost it its grace period.
                continue;
            };
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
                    // Filtering here, against the backend's own allowlist,
                    // is what makes positional mis-pairing unrepresentable.
                    backend.retain_allowed();
                    backend.last_seen = Some(now);
                    backend.grace_deadline = None;
                }
                FetchResult::Err => mark_down(backend, now, grace_duration),
            }
        }

        for name in by_name.into_keys() {
            // Removed by a reload while its probe was in flight. Discarding
            // is the point: there is no longer a slot this result describes.
            debug!(backend = %name, "discarding discovery result for a removed backend");
        }

        // Expire grace periods.
        for backend in &mut self.backends {
            if let Some(deadline) = backend.grace_deadline
                && now >= deadline
            {
                info!(backend = %backend.name, "grace period expired, removing models");
                backend.models.clear();
                backend.grace_deadline = None;
            }
        }

        self.rebuild_model_map();

        if !self.discovery_done {
            info!("first discovery cycle complete");
            self.discovery_done = true;
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

    /// Look up a backend by its configured name. Aliases are validated
    /// against the same roster in the same pass, so a candidate's backend
    /// always resolves for as long as the guard is held.
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

    /// True when this router is configured and has finished starting up.
    ///
    /// Gated on the roster being non-empty because a cycle over an *empty*
    /// roster completes successfully — there is nothing to probe — and flips
    /// `discovery_done`. Without that check the no-policy-file state would
    /// report Ready, take Service endpoints, and 404 every request as an
    /// unknown model, which looks healthy and is strictly worse than being
    /// visibly down. An empty roster is a *configuration* fact: it cannot
    /// resolve without an edit, so gating on it is right.
    ///
    /// Deliberately NOT gated on any backend being reachable. That is an
    /// *operational* fact which resolves on its own, and absorbing it is the
    /// router's whole job — grace periods, fallbacks and alias chains all
    /// exist to keep serving through it. With `replicas: 1`, tying readiness
    /// to backend health would turn a transient all-down blip into zero
    /// Service endpoints: every client gets connection-refused instead of an
    /// honest 502, a total outage that is also harder to diagnose because
    /// the pod has vanished from the Service exactly when `/status`,
    /// `/metrics` and the logs are what you need. It would also couple this
    /// pod's readiness to remote hosts we do not control, reintroducing the
    /// readiness flapping that previously cycled this deployment and
    /// cascaded into its dependents. Zero reachable backends belongs in
    /// metrics and alerts (`backend_up`, `OllamaRouterBackendDown`,
    /// `OllamaRouterChainExhausted`), not in the readiness probe.
    pub fn is_ready(&self) -> bool {
        self.discovery_done && !self.backends.is_empty()
    }

    /// True when the roster is empty — no policy file has ever applied.
    pub fn has_no_backends(&self) -> bool {
        self.backends.is_empty()
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

pub fn new_shared(validated: Validated) -> SharedRegistry {
    Arc::new(RwLock::new(Registry::new(validated)))
}

/// A registry with **no backends**: the state a router boots into when its
/// policy file is unreadable at startup.
///
/// Deliberately not a fatal error. Before 0.15.0 the roster came from the
/// environment, so a bad policy file could never stop the process starting;
/// now that the roster lives only in the file, aborting would turn any
/// unrelated restart during an NFS blip — OOMKill, drain, eviction — into a
/// CrashLoopBackOff with no way back. An empty roster can never be Ready
/// (see `Registry::is_ready`), so the Service simply has no endpoints and
/// the pod self-heals the moment the file becomes readable.
pub fn new_shared_empty() -> SharedRegistry {
    Arc::new(RwLock::new(Registry::default()))
}

/// Long-running discovery loop. Runs the first cycle immediately, then
/// every `interval`.
///
/// Each iteration is reload-then-probe: `router.toml` is re-read and
/// swapped in *before* discovery snapshots its targets, so a backend added
/// by a config edit is probed in that same cycle (~1s) rather than the next
/// one. That ordering is why no file watcher is needed — and inotify could
/// not help anyway: the file lives on NFS and is edited on the server, and
/// an NFS client receives no events for changes made elsewhere. A watcher
/// would sit silent forever while looking like it worked.
pub async fn discovery_loop(
    registry: SharedRegistry,
    config: Config,
    metrics: Arc<Metrics>,
    client: Client,
) {
    let interval = Duration::from_secs(config.discovery_interval_secs);
    let grace_duration = Duration::from_secs(config.grace_period_secs());

    loop {
        reload_policy(&config.config_path, &registry, &metrics).await;
        run_discovery(&client, &registry, grace_duration).await;
        tokio::time::sleep(interval).await;
    }
}

/// Re-read `router.toml` and swap it in wholesale.
///
/// A rejected reload keeps the previous config **entirely** — never a
/// partial apply, never a widened spend boundary — and is otherwise
/// completely silent. That silence is the defining failure mode of putting
/// routing policy in a hand-edited NFS file, which is why every outcome is
/// also counted: `config_reloads_total{result="rejected"}` is what the
/// alert fires on.
async fn reload_policy(path: &str, registry: &SharedRegistry, metrics: &Metrics) {
    let result = match load_policy(path).await {
        Ok(validated) => {
            let removed = registry.write().await.apply_file_config(validated);
            for backend in removed {
                // `backend_up` is the only backend-labelled gauge, and the
                // one that actively lies once a backend is deleted: it
                // freezes at its last value forever.
                metrics.backend_up.remove(&BackendLabels { backend });
            }
            metrics.config_last_reload_timestamp_seconds.set(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs() as i64)
                    .unwrap_or(0),
            );
            "applied"
        }
        Err(e) => {
            warn!(error = %e, file = %path, "config reload rejected; keeping previous config");
            "rejected"
        }
    };
    metrics
        .config_reloads
        .get_or_create(&ConfigReloadLabels {
            result: result.to_string(),
        })
        .inc();
}

/// Read and validate the policy file **off** the async runtime.
///
/// `FileConfig::load` is a synchronous `read_to_string` on a file that lives
/// on NFS. Calling it directly from the discovery task means a hard mount
/// against an unreachable server blocks that worker indefinitely: probes
/// stop, grace periods stop expiring, and — worst of all — reloads stop
/// *counting*, so `config_reloads_total` goes quiet instead of recording
/// rejections and the alert that exists for exactly this never fires, while
/// `/health` keeps answering Ready off a frozen snapshot.
///
/// `spawn_blocking` plus a timeout bounds it: a hung filesystem becomes a
/// rejected reload like any other bad file, and the loop keeps running. The
/// blocked thread does leak (nothing can cancel a stuck syscall), but it
/// leaks into the blocking pool where it costs a thread rather than the
/// router's liveness, and the alert fires long before the pool is a concern.
async fn load_policy(path: &str) -> Result<Validated, PolicyError> {
    let owned = path.to_string();
    load_off_runtime(path, POLICY_READ_TIMEOUT, move || FileConfig::load(&owned)).await
}

/// Run `read` on the blocking pool under `budget`.
///
/// Split from [`load_policy`] so the two properties that matter can be
/// tested without a filesystem that actually hangs: that the read does not
/// stall the runtime, and that exceeding the budget surfaces as an error
/// (and therefore as a *counted* rejected reload) rather than as silence.
async fn load_off_runtime<R>(
    path: &str,
    budget: Duration,
    read: R,
) -> Result<Validated, PolicyError>
where
    R: FnOnce() -> Result<Validated, PolicyError> + Send + 'static,
{
    let handle = tokio::task::spawn_blocking(read);
    match tokio::time::timeout(budget, handle).await {
        Ok(Ok(result)) => result,
        // The blocking task itself failed (panic or runtime shutdown).
        Ok(Err(e)) => Err(PolicyError::Read {
            path: path.to_string(),
            source: std::io::Error::other(e),
        }),
        Err(_elapsed) => Err(PolicyError::ReadTimeout {
            path: path.to_string(),
            after: budget,
        }),
    }
}

/// The outcome of probing one backend, paired with its name by the caller.
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

async fn run_discovery(client: &Client, registry: &SharedRegistry, grace_duration: Duration) {
    // Snapshot the targets from the REGISTRY (the sole roster), after the
    // reload earlier in this same iteration. There is no second `Vec` of
    // backends to drift out of sync with this one.
    let targets: Vec<(String, String)> = {
        let reg = registry.read().await;
        reg.backends
            .iter()
            .map(|b| (b.name.clone(), b.url.clone()))
            .collect()
    };

    // Phase 1: fetch from all backends concurrently, holding no lock.
    // Concurrency matters: a slow/dead backend's per-probe timeout would
    // otherwise serialise, making a cycle take N × timeout.
    let results =
        futures_util::future::join_all(targets.into_iter().map(|(name, url)| async move {
            let result = fetch_models(client, &name, &url).await;
            (name, result)
        }))
        .await;

    // Phase 2: apply results under the write lock (no I/O, microseconds).
    // Results carry their backend's name, so a reload that landed between
    // the snapshot and here cannot mis-pair them.
    let mut reg = registry.write().await;
    reg.apply_fetch_results(results, Instant::now(), grace_duration);
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

    /// The smallest valid policy document, for tests that only need *some*
    /// successfully-parsed value.
    const MINIMAL_POLICY: &str = "[[backends]]\nname = \"a\"\nurl = \"http://a:1\"\n\
                                  allow = [\"*\"]\n[fallbacks]\n[aliases]\n";

    /// Parse a policy document, or fail the test with the validator's own
    /// error — every registry test goes through the real validator, so a
    /// test roster can never be one the router would refuse to run.
    fn policy(raw: &str) -> Validated {
        FileConfig::parse(raw).expect("valid test policy")
    }

    /// A backend table with the given name, url and allow list.
    fn backend_toml(name: &str, url: &str, allow: &str) -> String {
        format!("[[backends]]\nname = \"{name}\"\nurl = \"{url}\"\nallow = {allow}\n")
    }

    /// The two-backend roster shared by most tests: `cuda` first (so it wins
    /// colliding model names), `rocm` second.
    fn test_policy() -> Validated {
        policy(&format!(
            "{}{}[fallbacks]\n[aliases]\n",
            backend_toml("cuda", "http://cuda:11434", "[\"*\"]"),
            backend_toml("rocm", "http://rocm:11435", "[\"*\"]"),
        ))
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
        let mut reg = Registry::new(test_policy());
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
        let mut reg = Registry::new(test_policy());
        reg.aliases = HashMap::from([(
            "fast".to_string(),
            Alias {
                local_only: false,
                candidates: vec![crate::policy::AliasCandidate {
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
        let mut reg = Registry::new(test_policy());
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
        let mut reg = Registry::new(test_policy());
        setup_backend(&mut reg, 0, b0_healthy, b0_grace, vec![make_model(model)]);
        setup_backend(&mut reg, 1, true, None, vec![make_model(model)]);
        reg
    }

    /// Set up both backends with no models, only health flags.
    fn reg_health_only(b0_healthy: bool, b1_healthy: bool) -> Registry {
        let mut reg = Registry::new(test_policy());
        setup_backend(&mut reg, 0, b0_healthy, None, vec![]);
        setup_backend(&mut reg, 1, b1_healthy, None, vec![]);
        reg
    }

    #[test]
    fn new_registry_starts_empty() {
        let reg = Registry::new(test_policy());
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
        let mut reg = Registry::new(test_policy());
        setup_backend(&mut reg, 0, true, None, vec![make_model("a:v1")]);
        setup_backend(&mut reg, 1, true, None, vec![make_model("b:latest")]);

        let mut available = reg.available_model_names();
        available.sort();
        assert_eq!(available, vec!["a:v1", "b:latest"]);
    }

    #[test]
    fn unhealthy_without_grace_excluded() {
        let mut reg = Registry::new(test_policy());
        setup_backend(&mut reg, 0, false, None, vec![make_model("orphan:v1")]);

        assert!(reg.lookup("orphan:v1").is_none());
    }

    #[test]
    fn unhealthy_within_grace_included() {
        let mut reg = Registry::new(test_policy());
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
        let mut reg = Registry::new(test_policy());
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
        let reg = Registry::new(test_policy());
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

    // ── policy reload: the single-roster contract ────────────────────────

    /// The roster after a reload, as `(name, url)` pairs in routing order.
    fn roster(reg: &Registry) -> Vec<(&str, &str)> {
        reg.backends
            .iter()
            .map(|b| (b.name.as_str(), b.url.as_str()))
            .collect()
    }

    /// A registry whose two backends are both healthy and serving one model
    /// each, with `cuda` also carrying a non-default protocol and a grace
    /// deadline — i.e. every field a reload must be careful with.
    fn reg_with_learned_state() -> Registry {
        let mut reg = Registry::new(test_policy());
        setup_backend(&mut reg, 0, true, None, vec![make_model("cuda-only:v1")]);
        setup_backend(&mut reg, 1, true, None, vec![make_model("rocm-only:v1")]);
        reg.backends[0].protocol = BackendProtocol::OpenAi;
        reg.backends[0].last_seen = Some(Instant::now());
        reg
    }

    #[test]
    fn reload_preserves_learned_state_for_an_unchanged_backend() {
        // Hazard 5: rebuilding a BackendState reverts `protocol` to Ollama
        // (which hangs /api/chat against an OpenAI-only backend) and empties
        // `models` (a 404 storm until the next fetch lands).
        let mut reg = reg_with_learned_state();
        let removed = reg.apply_file_config(test_policy());

        assert!(removed.is_empty());
        assert!(reg.backends[0].healthy);
        assert_eq!(reg.backends[0].protocol, BackendProtocol::OpenAi);
        assert_eq!(reg.backends[0].models.len(), 1);
        assert!(reg.backends[0].last_seen.is_some());
        assert_lookup_name(&reg, "cuda-only:v1", "cuda");
    }

    #[test]
    fn reload_reorder_repoints_lookup_and_keeps_state() {
        // Order IS priority: swapping the two tables must move a colliding
        // model name to the newly-first backend, with both keeping their
        // learned state.
        let mut reg = Registry::new(test_policy());
        setup_backend(&mut reg, 0, true, None, vec![make_model("shared:v1")]);
        setup_backend(&mut reg, 1, true, None, vec![make_model("shared:v1")]);
        assert_lookup_name(&reg, "shared:v1", "cuda");

        let reversed = policy(&format!(
            "{}{}[fallbacks]\n[aliases]\n",
            backend_toml("rocm", "http://rocm:11435", "[\"*\"]"),
            backend_toml("cuda", "http://cuda:11434", "[\"*\"]"),
        ));
        assert!(reg.apply_file_config(reversed).is_empty());

        assert_eq!(
            roster(&reg),
            vec![("rocm", "http://rocm:11435"), ("cuda", "http://cuda:11434")]
        );
        assert_lookup_name(&reg, "shared:v1", "rocm");
        assert!(
            reg.backends
                .iter()
                .all(|b| b.healthy && !b.models.is_empty())
        );
    }

    #[test]
    fn reload_add_leaves_existing_backends_untouched() {
        let mut reg = reg_with_learned_state();
        let grown = policy(&format!(
            "{}{}{}[fallbacks]\n[aliases]\n",
            backend_toml("cuda", "http://cuda:11434", "[\"*\"]"),
            backend_toml("rocm", "http://rocm:11435", "[\"*\"]"),
            backend_toml("new", "http://new:1", "[\"*\"]"),
        ));
        assert!(reg.apply_file_config(grown).is_empty());

        assert_eq!(reg.backends.len(), 3);
        assert_eq!(reg.backends[0].protocol, BackendProtocol::OpenAi);
        assert_lookup_name(&reg, "rocm-only:v1", "rocm");
        // The new backend starts unhealthy and empty — it has not been
        // probed yet. It is deliberately NOT allowed to un-ready /health:
        // with replicas 1 and maxUnavailable 0 that would pull the pod's
        // only endpoint, so adding a backend would cause a total outage.
        assert!(!reg.backends[2].healthy);
        assert!(reg.backends[2].models.is_empty());
    }

    #[test]
    fn reload_remove_reports_the_name_and_leaves_no_dangling_route() {
        let mut reg = reg_with_learned_state();
        let shrunk = policy(&format!(
            "{}[fallbacks]\n[aliases]\n",
            backend_toml("cuda", "http://cuda:11434", "[\"*\"]"),
        ));
        let removed = reg.apply_file_config(shrunk);

        assert_eq!(removed, vec!["rocm".to_string()]);
        assert_eq!(roster(&reg), vec![("cuda", "http://cuda:11434")]);
        // The removed backend's models are gone from the map, and every
        // surviving value still indexes a live slot.
        assert!(reg.lookup("rocm-only:v1").is_none());
        for id in reg.model_map.values() {
            assert!(reg.backend(*id).is_some(), "dangling model_map value");
        }
        assert_lookup_name(&reg, "cuda-only:v1", "cuda");
    }

    #[test]
    fn reload_url_change_resets_learned_state() {
        // Same name, different URL is a different server. Inheriting the old
        // one's model list would publish models the new address may not
        // serve.
        let mut reg = reg_with_learned_state();
        let moved = policy(&format!(
            "{}{}[fallbacks]\n[aliases]\n",
            backend_toml("cuda", "http://cuda:19999", "[\"*\"]"),
            backend_toml("rocm", "http://rocm:11435", "[\"*\"]"),
        ));
        assert!(reg.apply_file_config(moved).is_empty());

        assert_eq!(reg.backends[0].url, "http://cuda:19999");
        assert!(!reg.backends[0].healthy);
        assert_eq!(reg.backends[0].protocol, BackendProtocol::Ollama);
        assert!(reg.backends[0].models.is_empty());
        assert!(reg.lookup("cuda-only:v1").is_none());
        // The untouched sibling keeps everything.
        assert_lookup_name(&reg, "rocm-only:v1", "rocm");
    }

    #[test]
    fn stale_backend_id_returns_none_rather_than_panicking() {
        let mut reg = reg_with_learned_state();
        let stale = reg.backend_id_by_name("rocm").expect("rocm is configured");
        reg.apply_file_config(policy(&format!(
            "{}[fallbacks]\n[aliases]\n",
            backend_toml("cuda", "http://cuda:11434", "[\"*\"]"),
        )));
        assert!(reg.backend(stale).is_none());
    }

    #[test]
    fn reload_tightening_allow_drops_carried_over_models_immediately() {
        // The spend boundary must not lag its own edit by a discovery cycle.
        let mut reg = Registry::new(test_policy());
        setup_backend(
            &mut reg,
            0,
            true,
            None,
            vec![make_model("cheap:v1"), make_model("frontier:v1")],
        );
        assert!(reg.lookup("frontier:v1").is_some());

        reg.apply_file_config(policy(&format!(
            "{}{}[fallbacks]\n[aliases]\n",
            backend_toml("cuda", "http://cuda:11434", "[\"cheap:v1\"]"),
            backend_toml("rocm", "http://rocm:11435", "[\"*\"]"),
        )));

        assert!(reg.lookup("cheap:v1").is_some());
        assert!(
            reg.lookup("frontier:v1").is_none(),
            "a tightened allowlist must take effect at reload, not at the next fetch"
        );
    }

    #[test]
    fn reload_swaps_fallbacks_and_aliases_together() {
        let mut reg = Registry::new(test_policy());
        let doc = format!(
            "{}{}[fallbacks]\n\"ghost\" = \"cuda-only:v1\"\n\
             [aliases.fast]\nchain = [\"cuda/cuda-only:v1\"]\n",
            backend_toml("cuda", "http://cuda:11434", "[\"*\"]"),
            backend_toml("rocm", "http://rocm:11435", "[\"*\"]"),
        );
        reg.apply_file_config(policy(&doc));
        assert_eq!(reg.fallback_for("ghost"), Some("cuda-only:v1"));
        assert!(reg.alias_for("fast").is_some());

        // A later document with neither: both must clear in the same swap,
        // so a reader can never see the new one against the old other.
        reg.apply_file_config(test_policy());
        assert_eq!(reg.fallback_for("ghost"), None);
        assert!(reg.alias_for("fast").is_none());
    }

    // ── discovery results are keyed by NAME, never by position ───────────

    fn ok_result(models: &[&str]) -> FetchResult {
        FetchResult::Ok {
            protocol: BackendProtocol::OpenAi,
            models: models.iter().map(|n| make_model(n)).collect(),
        }
    }

    fn apply(reg: &mut Registry, results: Vec<(&str, FetchResult)>) {
        let results = results
            .into_iter()
            .map(|(n, r)| (n.to_string(), r))
            .collect();
        reg.apply_fetch_results(results, Instant::now(), Duration::from_secs(60));
    }

    #[test]
    fn fetch_results_land_on_the_named_backend_not_the_indexed_one() {
        // THE security-relevant case: results arrive in an order that does
        // not match the roster (a reload reordered it mid-cycle). A
        // positional zip would put backend A's models — and therefore its
        // spend boundary — on B.
        let mut reg = Registry::new(test_policy());
        apply(
            &mut reg,
            vec![
                ("rocm", ok_result(&["rocm-model:v1"])),
                ("cuda", ok_result(&["cuda-model:v1"])),
            ],
        );

        assert_lookup_name(&reg, "cuda-model:v1", "cuda");
        assert_lookup_name(&reg, "rocm-model:v1", "rocm");
    }

    #[test]
    fn fetch_result_for_a_vanished_backend_is_discarded() {
        let mut reg = Registry::new(test_policy());
        apply(
            &mut reg,
            vec![
                ("cuda", ok_result(&["cuda-model:v1"])),
                ("ghost", ok_result(&["ghost-model:v1"])),
            ],
        );

        assert_lookup_name(&reg, "cuda-model:v1", "cuda");
        assert!(reg.lookup("ghost-model:v1").is_none());
        assert_eq!(reg.backends.len(), 2);
    }

    #[test]
    fn backend_with_no_fetch_result_is_left_untouched() {
        // A backend added by the reload after targets were snapshotted has
        // no result this cycle. Marking it down would be a claim we have no
        // evidence for — and would cost a healthy backend its grace period.
        let mut reg = reg_with_learned_state();
        apply(&mut reg, vec![("cuda", ok_result(&["cuda-model:v1"]))]);

        assert!(reg.backends[1].healthy, "rocm must not be marked down");
        assert_lookup_name(&reg, "rocm-only:v1", "rocm");
    }

    #[test]
    fn allowlist_pairs_with_its_own_backend_regardless_of_order() {
        // cuda is metered (allowlisted), rocm publishes everything. The
        // filter must follow the *name*, not the slot.
        let mut reg = Registry::new(policy(&format!(
            "{}{}[fallbacks]\n[aliases]\n",
            backend_toml("cuda", "http://cuda:11434", "[\"kept:v1\"]"),
            backend_toml("rocm", "http://rocm:11435", "[\"*\"]"),
        )));
        apply(
            &mut reg,
            vec![
                ("rocm", ok_result(&["kept:v1", "dropped:v1"])),
                ("cuda", ok_result(&["kept:v1", "dropped:v1"])),
            ],
        );

        assert_eq!(reg.backends[0].models.len(), 1, "cuda is filtered");
        assert_eq!(reg.backends[1].models.len(), 2, "rocm is not");
        // First-writer-wins still puts the shared name on cuda.
        assert_lookup_name(&reg, "kept:v1", "cuda");
        assert_lookup_name(&reg, "dropped:v1", "rocm");
    }

    #[test]
    fn fetch_error_marks_only_that_backend_down() {
        let mut reg = reg_with_learned_state();
        apply(
            &mut reg,
            vec![
                ("cuda", FetchResult::Err),
                ("rocm", ok_result(&["rocm-only:v1"])),
            ],
        );

        assert!(!reg.backends[0].healthy);
        assert!(reg.backends[0].grace_deadline.is_some(), "grace starts");
        assert!(reg.backends[1].healthy);
    }

    // ── the policy read must never stall the discovery loop ──────────────

    #[tokio::test]
    async fn policy_read_runs_off_the_runtime() {
        // The file is on NFS; a hard mount against a dead server blocks
        // `read_to_string` indefinitely. Run inline on the discovery task,
        // that freezes the whole loop: no probes, no grace expiry, and —
        // worst — `config_reloads_total` stops incrementing instead of
        // counting rejections, so the alert for exactly this never fires
        // while /health answers Ready off a frozen snapshot.
        //
        // This is a current-thread runtime, so if the read is not moved to
        // the blocking pool the timer below can never fire.
        let ticked = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = Arc::clone(&ticked);
        let ticker = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            flag.store(true, std::sync::atomic::Ordering::SeqCst);
        });

        let result = load_off_runtime("/x", Duration::from_secs(30), || {
            std::thread::sleep(Duration::from_millis(300));
            FileConfig::parse(MINIMAL_POLICY)
        })
        .await;

        assert!(result.is_ok());
        assert!(
            ticked.load(std::sync::atomic::Ordering::SeqCst),
            "the runtime must keep polling tasks while the policy read blocks"
        );
        ticker.await.expect("ticker");
    }

    #[tokio::test]
    async fn policy_read_past_its_budget_is_an_error_not_silence() {
        // A wedged read has to become a *rejected reload* — something the
        // counter records and the alert can see — rather than a loop that
        // quietly stops reporting.
        let result = load_off_runtime("/hung", Duration::from_millis(20), || {
            std::thread::sleep(Duration::from_millis(400));
            FileConfig::parse(MINIMAL_POLICY)
        })
        .await;

        let err = result.expect_err("expected a timeout");
        assert!(
            matches!(err, PolicyError::ReadTimeout { .. }),
            "expected ReadTimeout, got {err:?}"
        );
        assert!(err.to_string().contains("timed out"), "{err}");
    }

    // ── a router with no backends is never ready ─────────────────────────

    #[test]
    fn empty_roster_is_never_ready() {
        // A cycle over zero backends completes successfully — there is
        // nothing to probe — and flips `discovery_done`. Readiness must
        // still be false: taking Service endpoints and 404-ing every
        // request looks healthy and is worse than being visibly down.
        let mut reg = Registry {
            backends: Vec::new(),
            model_map: HashMap::new(),
            fallbacks: HashMap::new(),
            aliases: HashMap::new(),
            discovery_done: false,
        };
        assert!(!reg.is_ready());
        reg.apply_fetch_results(Vec::new(), Instant::now(), Duration::from_secs(60));
        assert!(reg.is_discovery_done(), "an empty cycle still completes");
        assert!(reg.has_no_backends());
        assert!(!reg.is_ready(), "no backends can never be ready");

        // ...and a roster that appears later becomes ready normally.
        reg.apply_file_config(test_policy());
        apply(&mut reg, vec![("cuda", ok_result(&["m:v1"]))]);
        assert!(reg.is_ready());
    }

    #[test]
    fn all_backends_down_stays_ready() {
        // Reachability is an *operational* fact that resolves on its own,
        // and absorbing it is the router's whole job. With `replicas: 1`,
        // going un-Ready here would empty the Service and turn honest 502s
        // into connection-refused for every client — a total outage, and one
        // that also hides /status and /metrics behind a missing endpoint
        // exactly when they are needed. It would additionally couple this
        // pod's readiness to remote hosts (Nous, the GPU node), which is the
        // readiness flapping that previously cycled this deployment.
        let mut reg = Registry::new(test_policy());
        apply(
            &mut reg,
            vec![
                ("cuda", ok_result(&["m:v1"])),
                ("rocm", ok_result(&["n:v1"])),
            ],
        );
        assert!(reg.is_ready());

        // Every backend fails its probe, and every grace period expires.
        apply(
            &mut reg,
            vec![("cuda", FetchResult::Err), ("rocm", FetchResult::Err)],
        );
        for backend in &mut reg.backends {
            backend.grace_deadline = Some(Instant::now() - Duration::from_secs(1));
        }
        reg.apply_fetch_results(Vec::new(), Instant::now(), Duration::from_secs(60));

        assert!(
            reg.all_backends().all(|b| !b.healthy && !b.in_grace_period),
            "precondition: every backend is hard-down",
        );
        assert!(
            reg.is_ready(),
            "a configured router must stay Ready through a transient all-down blip",
        );
    }
}

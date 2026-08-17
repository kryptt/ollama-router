use std::collections::{HashMap, HashSet};
use std::env;
use std::fmt;
use std::net::SocketAddr;

/// A named backend with its base URL.
/// Invariant: `name` is non-empty, `url` has no trailing slash.
#[derive(Debug, Clone)]
pub struct Backend {
    pub name: String,
    pub url: String,
    /// Optional discovery allowlist. `None` = publish every model the backend
    /// advertises (the behaviour for every local backend). `Some(set)` = keep
    /// only these model names.
    ///
    /// This exists for hosted aggregators: Nous Portal advertises ~350 models,
    /// and publishing all of them would bury the local models in every
    /// consumer's picker *and* let anything that reaches the router spend
    /// portal credits on a frontier model. An allowlist is the only thing
    /// bounding that blast radius — see `OLLAMA_ROUTER_MODEL_ALLOW`.
    pub allow_models: Option<HashSet<String>>,
    /// Drop the client's `authorization` header before forwarding to this
    /// backend. For backends whose credential is injected by an egress proxy:
    /// our inbound token is not a backend credential, and sending it to a
    /// third party is both useless and a leak. See `OLLAMA_ROUTER_STRIP_AUTH`.
    pub strip_auth: bool,
}

/// A "if a request for `from_model` looks bigger than this model's
/// per-slot context, transparently rewrite it to `to_model`" rule.
///
/// Chains: rules are evaluated iteratively, so a single very-long request
/// for `qwen3.6-medium` can be re-routed through `qwen3.6-high` to
/// `qwen3.6-ultra` if both thresholds are exceeded.
#[derive(Debug, Clone)]
pub struct EscalationRule {
    pub from_model: String,
    pub max_input_tokens: usize,
    pub to_model: String,
}

impl EscalationRule {
    /// Parse a `from:max_input_tokens:to` triple. All three fields must be
    /// non-empty; the middle field must be a positive integer.
    fn parse(entry: &str) -> Result<Self, ConfigError> {
        let err = || ConfigError::InvalidEscalation(entry.to_string());
        let parts: Vec<&str> = entry.trim().split(':').collect();
        if parts.len() != 3 || parts.iter().any(|p| p.is_empty()) {
            return Err(err());
        }
        let max_input_tokens: usize = parts[1].parse().map_err(|_| err())?;
        if max_input_tokens == 0 {
            return Err(err());
        }
        Ok(EscalationRule {
            from_model: parts[0].to_string(),
            max_input_tokens,
            to_model: parts[2].to_string(),
        })
    }
}

impl Backend {
    /// Parse a `name=url` pair. Rejects empty names or URLs.
    fn parse(entry: &str) -> Result<Self, ConfigError> {
        let (name, url) = entry
            .split_once('=')
            .ok_or_else(|| ConfigError::InvalidBackend(entry.to_string()))?;

        if name.is_empty() || url.is_empty() {
            return Err(ConfigError::InvalidBackend(entry.to_string()));
        }

        Ok(Backend {
            name: name.to_string(),
            url: url.trim_end_matches('/').to_string(),
            allow_models: None,
            strip_auth: false,
        })
    }

    /// Construct a backend directly (bypassing env parsing). For tests.
    pub fn for_test(name: &str, url: &str) -> Self {
        Backend {
            name: name.to_string(),
            url: url.to_string(),
            allow_models: None,
            strip_auth: false,
        }
    }
}

/// Strip `#`-to-end-of-line comments and rejoin lines with commas, so file
/// contents (one entry per line, comments allowed) parse with the same
/// grammar as the comma-separated env values.
fn strip_comments(raw: &str) -> String {
    raw.lines()
        .map(|l| l.split('#').next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join(",")
}

/// Parse `OLLAMA_ROUTER_MODEL_ALLOW` into a per-backend allowlist.
///
/// Format: `backend=model1|model2,other=model3`. Newlines work like commas
/// and `#` starts a comment, so the same grammar serves both the env value
/// and `OLLAMA_ROUTER_MODEL_ALLOW_FILE` contents. Backends absent from the
/// variable are left unfiltered, so this is purely additive — an empty or
/// unset variable reproduces the previous behaviour exactly.
///
/// An entry with a name but no models (`nous=`) is rejected rather than
/// treated as "allow nothing": silently publishing zero models from a
/// backend is indistinguishable from the backend being down.
fn parse_model_allow(raw: &str) -> Result<HashMap<String, HashSet<String>>, ConfigError> {
    let raw = strip_comments(raw);
    let mut out: HashMap<String, HashSet<String>> = HashMap::new();

    for entry in raw.split(',').filter(|e| !e.trim().is_empty()) {
        let entry = entry.trim();
        let (name, models) = entry
            .split_once('=')
            .ok_or_else(|| ConfigError::InvalidModelAllow(entry.to_string()))?;

        let name = name.trim();
        let models: HashSet<String> = models
            .split('|')
            .map(str::trim)
            .filter(|m| !m.is_empty())
            .map(str::to_string)
            .collect();

        if name.is_empty() || models.is_empty() {
            return Err(ConfigError::InvalidModelAllow(entry.to_string()));
        }

        out.entry(name.to_string()).or_default().extend(models);
    }

    Ok(out)
}

/// Load and parse an allowlist file (`OLLAMA_ROUTER_MODEL_ALLOW_FILE`).
/// Same grammar as the env value, plus newlines-as-commas and `#` comments.
pub fn load_model_allow_file(
    path: &str,
) -> Result<HashMap<String, HashSet<String>>, ConfigError> {
    let raw = std::fs::read_to_string(path).map_err(|e| ConfigError::Invalid {
        key: "OLLAMA_ROUTER_MODEL_ALLOW_FILE",
        reason: format!("could not read '{path}': {e}"),
    })?;
    parse_model_allow(&raw)
}

/// Apply a parsed allowlist to the backend list. Every backend named in
/// `allow` must exist; an unknown name is always a typo, and a silent one:
/// the operator believes a backend is being filtered when it is publishing
/// its full catalogue. Nothing is applied on error, so a bad reload keeps
/// the previous cycle's filters intact.
pub fn apply_model_allow(
    backends: &mut [Backend],
    mut allow: HashMap<String, HashSet<String>>,
) -> Result<(), String> {
    if let Some(unknown) = allow
        .keys()
        .find(|name| !backends.iter().any(|b| b.name == **name))
    {
        return Err(unknown.clone());
    }
    for backend in backends {
        backend.allow_models = allow.remove(&backend.name);
    }
    Ok(())
}

/// Parse a fallback map (`OLLAMA_ROUTER_FALLBACK_FILE` contents): one
/// `local-model=stand-in-model` pair per line, `#` comments allowed.
///
/// The map is consulted only when no reachable backend serves the requested
/// model; the stand-in must itself be published (allowlisted) or the hop is
/// skipped. Duplicate keys are rejected — last-wins would hide the typo.
pub fn parse_fallbacks(raw: &str) -> Result<HashMap<String, String>, ConfigError> {
    let mut out = HashMap::new();
    for line in raw.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let err = || ConfigError::InvalidFallback(line.to_string());
        let (from, to) = line.split_once('=').ok_or_else(err)?;
        let (from, to) = (from.trim(), to.trim());
        if from.is_empty() || to.is_empty() || from == to {
            return Err(err());
        }
        if out.insert(from.to_string(), to.to_string()).is_some() {
            return Err(ConfigError::Invalid {
                key: "OLLAMA_ROUTER_FALLBACK_FILE",
                reason: format!("duplicate fallback entry for '{from}'"),
            });
        }
    }
    Ok(out)
}

/// Load and parse a fallback file. See [`parse_fallbacks`].
pub fn load_fallbacks_file(path: &str) -> Result<HashMap<String, String>, ConfigError> {
    let raw = std::fs::read_to_string(path).map_err(|e| ConfigError::Invalid {
        key: "OLLAMA_ROUTER_FALLBACK_FILE",
        reason: format!("could not read '{path}': {e}"),
    })?;
    parse_fallbacks(&raw)
}

// Default values, shared between `from_env` (as the parse fallbacks) and the
// `Default` impl (used by `from_backends`) so the two construction paths can
// never silently drift apart.
const DEFAULT_DISCOVERY_INTERVAL_SECS: u64 = 60;
const DEFAULT_GRACE_MULTIPLIER: u64 = 3;
const DEFAULT_PUBLIC_PORT: u16 = 11434;
const DEFAULT_INTERNAL_PORT: u16 = 9090;
const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 10;
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 300;
const DEFAULT_LOADING_HEARTBEAT_SECS: u64 = 15;
const DEFAULT_PREFLIGHT_TIMEOUT_SECS: u64 = 10;
const DEFAULT_LOADING_MAX_WAIT_SECS: u64 = 300;
const DEFAULT_MAX_RETRIES: u64 = 2;
const DEFAULT_RETRY_BACKOFF_BASE_MS: u64 = 100;
const DEFAULT_RETRY_JITTER_PCT: u64 = 25;
const DEFAULT_RETRY_LATENCY_BUDGET_SECS: u64 = 30;
const DEFAULT_BREAKER_5XX_THRESHOLD: u64 = 5;
const DEFAULT_BREAKER_OPEN_SECS: u64 = 10;
const DEFAULT_BACKEND_MAX_INFLIGHT: u64 = 0;
const DEFAULT_CACHE_ENABLED: bool = false;
const DEFAULT_CACHE_MAX_BYTES: u64 = 64 * 1024 * 1024; // 64 MiB
const DEFAULT_CACHE_MAX_ENTRY_BYTES: u64 = 1024 * 1024; // 1 MiB
const DEFAULT_CACHE_TTL_SECS: u64 = 3600;

/// Validated, immutable configuration.
/// All parsing happens in `from_env`; once constructed, every field is valid.
#[derive(Debug, Clone)]
pub struct Config {
    pub backends: Vec<Backend>,
    pub discovery_interval_secs: u64,
    pub grace_multiplier: u64,
    pub tokens_file: Option<String>,
    /// PEM bundle of extra root certificates to trust on outbound requests,
    /// on top of the built-in roots. Needed when a backend is reached through
    /// a TLS-intercepting egress proxy (postern), whose MITM CA is private.
    pub extra_ca_file: Option<String>,
    pub public_addr: SocketAddr,
    pub internal_addr: SocketAddr,
    pub connect_timeout_secs: u64,
    pub request_timeout_secs: u64,
    /// How often to emit a heartbeat chunk while waiting for a cold model to
    /// produce its first token. Also the delay before the first heartbeat.
    pub loading_heartbeat_secs: u64,
    /// Timeout on the `/api/ps` preflight probe. If it times out, we fall
    /// through to the normal proxy path.
    pub preflight_timeout_secs: u64,
    /// Maximum time to wait for upstream to produce its first real byte
    /// before giving up and emitting an in-band error.
    pub loading_max_wait_secs: u64,
    /// Per-model escalation rules. Empty = no escalation (default).
    pub escalation_rules: Vec<EscalationRule>,
    /// Path to an allowlist file, re-read every discovery cycle so a mounted
    /// ConfigMap edit lands without a restart. Mutually exclusive with
    /// `OLLAMA_ROUTER_MODEL_ALLOW`.
    pub model_allow_file: Option<String>,
    /// Path to a `local-model=stand-in` fallback map, re-read every discovery
    /// cycle. Consulted when no reachable backend serves a requested model.
    pub fallback_file: Option<String>,

    // --- Resilience: bounded retry-with-backoff (Unit 3) ---
    /// Maximum retry attempts after the first try for a transient failure.
    /// 0 disables retry (single-shot).
    pub max_retries: u64,
    /// Base delay for exponential backoff between retry attempts.
    pub retry_backoff_base_ms: u64,
    /// Random jitter added to each backoff, as a percentage of the computed
    /// backoff (e.g. 25 = up to ±25%). 0 disables jitter. Validated to 0–100.
    pub retry_jitter_pct: u64,
    /// Hard wall-clock budget across all attempts for a single request. Once
    /// exceeded, stop retrying and surface backpressure.
    pub retry_latency_budget_secs: u64,

    // --- Resilience: per-backend circuit breaker + admission (Unit 3) ---
    /// Consecutive 5xx responses from a backend that trip its breaker open.
    /// Must be at least 1 (the breaker is always on).
    pub breaker_5xx_threshold: u64,
    /// How long a backend's breaker stays open before a half-open probe.
    pub breaker_open_secs: u64,
    /// Per-backend in-flight request cap; over the cap sheds load as 503
    /// rather than queueing. 0 = unlimited.
    pub backend_max_inflight: u64,

    // --- Embedding cache (Unit 4; off by default until validated) ---
    /// Master switch for the embedding cache.
    pub cache_enabled: bool,
    /// Total byte budget for the cache across all entries.
    pub cache_max_bytes: u64,
    /// Skip caching any single body larger than this (avoid buffering the
    /// multi-MB bulk-embed payloads). 0 = no per-entry cap. When non-zero,
    /// must not exceed `cache_max_bytes`.
    pub cache_max_entry_bytes: u64,
    /// Time-to-live for a cached embedding.
    pub cache_ttl_secs: u64,
}

impl Default for Config {
    /// Defaults with no backends and loopback addresses. `from_env` overrides
    /// every field from the environment; `from_backends` overrides `backends`.
    fn default() -> Self {
        Config {
            backends: Vec::new(),
            discovery_interval_secs: DEFAULT_DISCOVERY_INTERVAL_SECS,
            grace_multiplier: DEFAULT_GRACE_MULTIPLIER,
            tokens_file: None,
            extra_ca_file: None,
            public_addr: SocketAddr::from(([127, 0, 0, 1], 0)),
            internal_addr: SocketAddr::from(([127, 0, 0, 1], 0)),
            connect_timeout_secs: DEFAULT_CONNECT_TIMEOUT_SECS,
            request_timeout_secs: DEFAULT_REQUEST_TIMEOUT_SECS,
            loading_heartbeat_secs: DEFAULT_LOADING_HEARTBEAT_SECS,
            preflight_timeout_secs: DEFAULT_PREFLIGHT_TIMEOUT_SECS,
            loading_max_wait_secs: DEFAULT_LOADING_MAX_WAIT_SECS,
            escalation_rules: Vec::new(),
            model_allow_file: None,
            fallback_file: None,
            max_retries: DEFAULT_MAX_RETRIES,
            retry_backoff_base_ms: DEFAULT_RETRY_BACKOFF_BASE_MS,
            retry_jitter_pct: DEFAULT_RETRY_JITTER_PCT,
            retry_latency_budget_secs: DEFAULT_RETRY_LATENCY_BUDGET_SECS,
            breaker_5xx_threshold: DEFAULT_BREAKER_5XX_THRESHOLD,
            breaker_open_secs: DEFAULT_BREAKER_OPEN_SECS,
            backend_max_inflight: DEFAULT_BACKEND_MAX_INFLIGHT,
            cache_enabled: DEFAULT_CACHE_ENABLED,
            cache_max_bytes: DEFAULT_CACHE_MAX_BYTES,
            cache_max_entry_bytes: DEFAULT_CACHE_MAX_ENTRY_BYTES,
            cache_ttl_secs: DEFAULT_CACHE_TTL_SECS,
        }
    }
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        let backends_str = env::var("OLLAMA_ROUTER_BACKENDS")
            .unwrap_or_else(|_| "ollama=http://localhost:11434".to_string());

        let mut backends: Vec<Backend> = backends_str
            .split(',')
            .map(|e| Backend::parse(e.trim()))
            .collect::<Result<Vec<_>, _>>()?;

        if backends.is_empty() {
            return Err(ConfigError::NoBackends);
        }

        let model_allow_file = env::var("OLLAMA_ROUTER_MODEL_ALLOW_FILE")
            .ok()
            .filter(|s| !s.trim().is_empty());
        let model_allow_env = env::var("OLLAMA_ROUTER_MODEL_ALLOW")
            .ok()
            .filter(|s| !s.trim().is_empty());
        // Two sources for the same filter would make "which one is live?"
        // a runtime question. Refuse the ambiguity at startup.
        if model_allow_file.is_some() && model_allow_env.is_some() {
            return Err(ConfigError::Invalid {
                key: "OLLAMA_ROUTER_MODEL_ALLOW_FILE",
                reason: "mutually exclusive with OLLAMA_ROUTER_MODEL_ALLOW; set only one"
                    .to_string(),
            });
        }
        let model_allow = match (&model_allow_file, &model_allow_env) {
            // Fail fast on an unreadable/unparseable file even though the
            // discovery loop re-reads it — a typo'd path would otherwise run
            // unfiltered until someone reads the logs.
            (Some(path), _) => load_model_allow_file(path)?,
            (_, Some(s)) => parse_model_allow(s)?,
            _ => HashMap::new(),
        };
        let strip_auth: HashSet<String> = env::var("OLLAMA_ROUTER_STRIP_AUTH")
            .unwrap_or_default()
            .split(',')
            .map(str::trim)
            .filter(|n| !n.is_empty())
            .map(str::to_string)
            .collect();

        for backend in &mut backends {
            backend.strip_auth = strip_auth.contains(&backend.name);
        }
        apply_model_allow(&mut backends, model_allow).map_err(|unknown| {
            ConfigError::Invalid {
                key: "OLLAMA_ROUTER_MODEL_ALLOW",
                reason: format!(
                    "names backend '{unknown}', which is not in OLLAMA_ROUTER_BACKENDS"
                ),
            }
        })?;

        let fallback_file = env::var("OLLAMA_ROUTER_FALLBACK_FILE")
            .ok()
            .filter(|s| !s.trim().is_empty());
        if let Some(path) = &fallback_file {
            // Validate now, discard the value: the discovery loop loads it
            // fresh on its first cycle (immediately at startup).
            load_fallbacks_file(path)?;
        }

        let discovery_interval_secs = parse_env_u64(
            "OLLAMA_ROUTER_DISCOVERY_INTERVAL",
            DEFAULT_DISCOVERY_INTERVAL_SECS,
        )?;
        let grace_multiplier =
            parse_env_u64("OLLAMA_ROUTER_GRACE_MULTIPLIER", DEFAULT_GRACE_MULTIPLIER)?;
        let tokens_file = env::var("OLLAMA_ROUTER_TOKENS_FILE").ok();
        let extra_ca_file = env::var("OLLAMA_ROUTER_EXTRA_CA_FILE")
            .ok()
            .filter(|s| !s.trim().is_empty());
        let public_port =
            parse_env_u64("OLLAMA_ROUTER_PUBLIC_PORT", DEFAULT_PUBLIC_PORT as u64)? as u16;
        let internal_port =
            parse_env_u64("OLLAMA_ROUTER_INTERNAL_PORT", DEFAULT_INTERNAL_PORT as u64)? as u16;
        let connect_timeout_secs = parse_env_u64(
            "OLLAMA_ROUTER_CONNECT_TIMEOUT",
            DEFAULT_CONNECT_TIMEOUT_SECS,
        )?;
        let request_timeout_secs = parse_env_u64(
            "OLLAMA_ROUTER_REQUEST_TIMEOUT",
            DEFAULT_REQUEST_TIMEOUT_SECS,
        )?;
        let loading_heartbeat_secs = parse_env_u64(
            "OLLAMA_ROUTER_LOADING_HEARTBEAT",
            DEFAULT_LOADING_HEARTBEAT_SECS,
        )?;
        let preflight_timeout_secs = parse_env_u64(
            "OLLAMA_ROUTER_PREFLIGHT_TIMEOUT",
            DEFAULT_PREFLIGHT_TIMEOUT_SECS,
        )?;
        let loading_max_wait_secs = parse_env_u64(
            "OLLAMA_ROUTER_LOADING_MAX_WAIT",
            DEFAULT_LOADING_MAX_WAIT_SECS,
        )?;

        let escalation_rules = match env::var("OLLAMA_ROUTER_ESCALATE") {
            Ok(s) if !s.trim().is_empty() => s
                .split(',')
                .filter(|e| !e.trim().is_empty())
                .map(EscalationRule::parse)
                .collect::<Result<Vec<_>, _>>()?,
            _ => Vec::new(),
        };

        let max_retries = parse_env_u64("OLLAMA_ROUTER_MAX_RETRIES", DEFAULT_MAX_RETRIES)?;
        let retry_backoff_base_ms = parse_env_u64(
            "OLLAMA_ROUTER_RETRY_BACKOFF_BASE_MS",
            DEFAULT_RETRY_BACKOFF_BASE_MS,
        )?;
        let retry_jitter_pct =
            parse_env_u64("OLLAMA_ROUTER_RETRY_JITTER_PCT", DEFAULT_RETRY_JITTER_PCT)?;
        let retry_latency_budget_secs = parse_env_u64(
            "OLLAMA_ROUTER_RETRY_LATENCY_BUDGET",
            DEFAULT_RETRY_LATENCY_BUDGET_SECS,
        )?;
        let breaker_5xx_threshold = parse_env_u64(
            "OLLAMA_ROUTER_BREAKER_5XX_THRESHOLD",
            DEFAULT_BREAKER_5XX_THRESHOLD,
        )?;
        let breaker_open_secs =
            parse_env_u64("OLLAMA_ROUTER_BREAKER_OPEN", DEFAULT_BREAKER_OPEN_SECS)?;
        let backend_max_inflight = parse_env_u64(
            "OLLAMA_ROUTER_BACKEND_MAX_INFLIGHT",
            DEFAULT_BACKEND_MAX_INFLIGHT,
        )?;
        let cache_enabled = parse_env_bool("OLLAMA_ROUTER_CACHE_ENABLED", DEFAULT_CACHE_ENABLED)?;
        let cache_max_bytes =
            parse_env_u64("OLLAMA_ROUTER_CACHE_MAX_BYTES", DEFAULT_CACHE_MAX_BYTES)?;
        let cache_max_entry_bytes = parse_env_u64(
            "OLLAMA_ROUTER_CACHE_MAX_ENTRY_BYTES",
            DEFAULT_CACHE_MAX_ENTRY_BYTES,
        )?;
        let cache_ttl_secs = parse_env_u64("OLLAMA_ROUTER_CACHE_TTL", DEFAULT_CACHE_TTL_SECS)?;

        // Semantic validation: catch misconfigurations at startup rather than
        // deferring them to confusing runtime behavior in Units 3/4.
        if retry_jitter_pct > 100 {
            return Err(ConfigError::Invalid {
                key: "OLLAMA_ROUTER_RETRY_JITTER_PCT",
                reason: format!("must be 0–100, got {retry_jitter_pct}"),
            });
        }
        if breaker_5xx_threshold == 0 {
            return Err(ConfigError::Invalid {
                key: "OLLAMA_ROUTER_BREAKER_5XX_THRESHOLD",
                reason: "must be at least 1 (0 would trip the breaker permanently)".to_string(),
            });
        }
        if cache_max_entry_bytes != 0 && cache_max_entry_bytes > cache_max_bytes {
            return Err(ConfigError::Invalid {
                key: "OLLAMA_ROUTER_CACHE_MAX_ENTRY_BYTES",
                reason: format!(
                    "must not exceed OLLAMA_ROUTER_CACHE_MAX_BYTES ({cache_max_bytes}), got {cache_max_entry_bytes}"
                ),
            });
        }

        Ok(Config {
            backends,
            discovery_interval_secs,
            grace_multiplier,
            tokens_file,
            extra_ca_file,
            public_addr: SocketAddr::from(([0, 0, 0, 0], public_port)),
            internal_addr: SocketAddr::from(([0, 0, 0, 0], internal_port)),
            connect_timeout_secs,
            request_timeout_secs,
            loading_heartbeat_secs,
            preflight_timeout_secs,
            loading_max_wait_secs,
            escalation_rules,
            model_allow_file,
            fallback_file,
            max_retries,
            retry_backoff_base_ms,
            retry_jitter_pct,
            retry_latency_budget_secs,
            breaker_5xx_threshold,
            breaker_open_secs,
            backend_max_inflight,
            cache_enabled,
            cache_max_bytes,
            cache_max_entry_bytes,
            cache_ttl_secs,
        })
    }

    pub fn grace_period_secs(&self) -> u64 {
        self.discovery_interval_secs * self.grace_multiplier
    }

    /// Add `extra_ca_file`'s certificates to an outbound HTTP client.
    ///
    /// A no-op when unset, so the default build keeps the bundled roots and
    /// nothing else. Both the proxy client and the discovery client must go
    /// through this: discovery reaching a backend that the proxy path cannot
    /// (or vice versa) shows up as a backend that lists models but 502s on
    /// every request.
    pub fn apply_extra_ca(
        &self,
        mut builder: reqwest::ClientBuilder,
    ) -> Result<reqwest::ClientBuilder, ConfigError> {
        const KEY: &str = "OLLAMA_ROUTER_EXTRA_CA_FILE";
        let invalid = |reason: String| ConfigError::Invalid { key: KEY, reason };

        let Some(path) = &self.extra_ca_file else {
            return Ok(builder);
        };

        let pem = std::fs::read(path)
            .map_err(|e| invalid(format!("could not be read from '{path}': {e}")))?;
        let certs = reqwest::Certificate::from_pem_bundle(&pem)
            .map_err(|e| invalid(format!("'{path}' is not a valid PEM bundle: {e}")))?;

        if certs.is_empty() {
            return Err(invalid(format!("'{path}' contained no certificates")));
        }

        for cert in certs {
            builder = builder.add_root_certificate(cert);
        }
        Ok(builder)
    }

    /// Construct a config from explicit backends with sensible defaults. For tests.
    pub fn from_backends(backends: Vec<Backend>) -> Self {
        Config {
            backends,
            ..Config::default()
        }
    }
}

#[derive(Debug)]
pub enum ConfigError {
    InvalidBackend(String),
    NoBackends,
    /// A value that should be a positive integer failed to parse as one.
    InvalidValue {
        key: &'static str,
        value: String,
    },
    /// A value that should be a boolean failed to parse as one.
    InvalidBool {
        key: &'static str,
        value: String,
    },
    /// A value parsed but failed a semantic/range/cross-field constraint.
    Invalid {
        key: &'static str,
        reason: String,
    },
    InvalidEscalation(String),
    InvalidModelAllow(String),
    InvalidFallback(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBackend(entry) => {
                write!(f, "invalid backend entry: '{entry}' (expected name=url)")
            }
            Self::NoBackends => {
                write!(
                    f,
                    "OLLAMA_ROUTER_BACKENDS must contain at least one backend"
                )
            }
            Self::InvalidValue { key, value } => {
                write!(f, "{key} must be a positive integer, got '{value}'")
            }
            Self::InvalidBool { key, value } => {
                write!(
                    f,
                    "{key} must be a boolean (true/false, 1/0, yes/no, on/off), got '{value}'"
                )
            }
            Self::Invalid { key, reason } => {
                write!(f, "{key} {reason}")
            }
            Self::InvalidEscalation(entry) => {
                write!(
                    f,
                    "invalid escalation rule: '{entry}' (expected from_model:max_input_tokens:to_model with positive integer threshold)"
                )
            }
            Self::InvalidModelAllow(entry) => {
                write!(
                    f,
                    "invalid model allowlist entry: '{entry}' (expected backend=model1|model2)"
                )
            }
            Self::InvalidFallback(entry) => {
                write!(
                    f,
                    "invalid fallback entry: '{entry}' (expected local-model=stand-in-model)"
                )
            }
        }
    }
}

impl std::error::Error for ConfigError {}

/// Read an env var, trim it, and hand the trimmed value to `parser`.
/// Missing vars fall back to `default`; present-but-unparseable values are
/// rejected by `parser`.
fn parse_env<T>(
    key: &'static str,
    default: T,
    parser: impl FnOnce(&str) -> Result<T, ConfigError>,
) -> Result<T, ConfigError> {
    match env::var(key) {
        Ok(val) => parser(val.trim()),
        Err(_) => Ok(default),
    }
}

/// Parse an integer env var. Surrounding whitespace is ignored; a missing var
/// falls back to `default`, while a present-but-unparseable value is rejected.
fn parse_env_u64(key: &'static str, default: u64) -> Result<u64, ConfigError> {
    parse_env(key, default, |s| {
        s.parse().map_err(|_| ConfigError::InvalidValue {
            key,
            value: s.to_string(),
        })
    })
}

/// Parse a boolean env var. Accepts (case-insensitive, whitespace-trimmed)
/// `true/false`, `1/0`, `yes/no`, `on/off`; anything else is rejected.
fn parse_env_bool(key: &'static str, default: bool) -> Result<bool, ConfigError> {
    parse_env(key, default, |s| match s.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => Err(ConfigError::InvalidBool {
            key,
            value: s.to_string(),
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allow(raw: &str) -> HashMap<String, HashSet<String>> {
        parse_model_allow(raw).expect("expected a valid allowlist")
    }

    #[test]
    fn allowlist_accepts_newlines_and_comments() {
        let got = allow("# spend boundary\nnous=a/one|a/two\n\nnous=a/three # inline note\n");
        assert_eq!(
            got["nous"],
            HashSet::from([
                "a/one".to_string(),
                "a/two".to_string(),
                "a/three".to_string()
            ])
        );
    }

    #[test]
    fn apply_model_allow_rejects_unknown_backend_without_applying() {
        let mut backends = vec![Backend::for_test("nous", "http://n")];
        backends[0].allow_models = Some(HashSet::from(["keep".to_string()]));
        let bad = HashMap::from([("typo".to_string(), HashSet::from(["m".to_string()]))]);
        assert_eq!(apply_model_allow(&mut backends, bad), Err("typo".to_string()));
        // Nothing applied: previous filter survives a bad reload.
        assert_eq!(
            backends[0].allow_models,
            Some(HashSet::from(["keep".to_string()]))
        );
    }

    #[test]
    fn parses_fallback_map() {
        let got = parse_fallbacks(
            "# stand-ins\nqwen3.6-medium=qwen/qwen3.8-27b\ngemma4:26b = google/gemma-4-26b-a4b-it\n",
        )
        .expect("valid fallback map");
        assert_eq!(got["qwen3.6-medium"], "qwen/qwen3.8-27b");
        assert_eq!(got["gemma4:26b"], "google/gemma-4-26b-a4b-it");
    }

    #[test]
    fn fallback_map_rejects_bad_entries() {
        assert!(parse_fallbacks("no-equals-sign").is_err());
        assert!(parse_fallbacks("a=").is_err());
        assert!(parse_fallbacks("=b").is_err());
        assert!(parse_fallbacks("a=a").is_err(), "self-mapping is a typo");
        assert!(
            parse_fallbacks("a=b\na=c").is_err(),
            "duplicate keys are a typo; last-wins would hide it"
        );
        assert!(parse_fallbacks("").expect("empty ok").is_empty());
    }

    #[test]
    fn parses_backends_and_models() {
        let got = allow("nous=a/one|a/two,local=b/three");
        assert_eq!(got.len(), 2);
        assert_eq!(
            got["nous"],
            HashSet::from(["a/one".to_string(), "a/two".to_string()])
        );
        assert_eq!(got["local"], HashSet::from(["b/three".to_string()]));
    }

    #[test]
    fn tolerates_whitespace_and_empty_entries() {
        let got = allow("  nous = a/one | a/two ,, ");
        assert_eq!(
            got["nous"],
            HashSet::from(["a/one".to_string(), "a/two".to_string()])
        );
    }

    #[test]
    fn merges_repeated_backend_entries() {
        let got = allow("nous=a/one,nous=a/two");
        assert_eq!(
            got["nous"],
            HashSet::from(["a/one".to_string(), "a/two".to_string()])
        );
    }

    #[test]
    fn rejects_entries_that_would_silently_publish_nothing() {
        // A backend with no models is indistinguishable at runtime from a
        // backend that is down, so it must fail at startup instead.
        assert!(parse_model_allow("nous=").is_err());
        assert!(parse_model_allow("nous=|").is_err());
        assert!(parse_model_allow("=a/one").is_err());
        assert!(parse_model_allow("nous").is_err());
    }
}

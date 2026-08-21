use std::env;
use std::fmt;
use std::net::SocketAddr;

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

// Default values, shared between `from_env` (as the parse fallbacks) and the
// `Default` impl (which tests build on) so the two construction paths can
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
    /// Path to the `router.toml` policy file (`OLLAMA_ROUTER_CONFIG`).
    /// Required, no default: the backend roster and the spend boundary both
    /// live in it, so a router with no policy file has nothing to route to
    /// and nothing bounding what it would spend.
    ///
    /// The path itself is restart-scoped; everything *in* the file is live.
    pub config_path: String,
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
    ///
    /// Restart-scoped like every other env knob. Making these live would
    /// mean moving them into `Registry` and taking a lock on the hot path —
    /// a bigger diff than the whole policy-file migration, for a feature
    /// that is unset in production.
    pub escalation_rules: Vec<EscalationRule>,

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
    /// Defaults with loopback addresses and no policy-file path. `from_env`
    /// overrides every field from the environment; tests that only need the
    /// restart-scoped knobs use this directly.
    fn default() -> Self {
        Config {
            config_path: String::new(),
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
        // The policy file is the router's entire routing surface — backends,
        // spend boundary, aliases, fallbacks — so there is deliberately no
        // default path to fall back to. Starting without one would mean
        // starting with no backends.
        let config_path = env::var("OLLAMA_ROUTER_CONFIG")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .ok_or(ConfigError::MissingConfigPath)?;

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
            config_path,
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
}

#[derive(Debug)]
pub enum ConfigError {
    /// `OLLAMA_ROUTER_CONFIG` unset or empty.
    MissingConfigPath,
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
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingConfigPath => {
                write!(
                    f,
                    "OLLAMA_ROUTER_CONFIG must be set to the path of the router.toml policy file"
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

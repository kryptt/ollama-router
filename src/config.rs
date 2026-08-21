use std::env;
use std::fmt;
use std::net::SocketAddr;
use std::time::Duration;

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
        // Deliberately NOT empty-filtered like the other path vars, because
        // this one gates authentication and the two failure modes are not
        // symmetric. Filtering an empty value to `None` disables auth
        // entirely — a manifest whose `valueFrom` renders empty would bring
        // the router up wide open, and anything that could reach it could
        // spend metered credits. Keeping the empty string instead points the
        // fail-closed token store at an unreadable path, so every request
        // 401s for no visible reason.
        //
        // Neither is acceptable, so an empty-but-set value is a hard startup
        // error. Only a genuinely *absent* variable disables auth.
        let tokens_file = match env::var("OLLAMA_ROUTER_TOKENS_FILE") {
            Ok(path) if path.trim().is_empty() => {
                return Err(ConfigError::Invalid {
                    key: "OLLAMA_ROUTER_TOKENS_FILE",
                    reason: "is set but empty: set a path, or remove the variable to disable \
                             authentication (an empty value would silently disable it)"
                        .to_string(),
                });
            }
            Ok(path) => Some(path),
            Err(_) => None,
        };
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
        })
    }

    pub fn grace_period_secs(&self) -> u64 {
        self.discovery_interval_secs * self.grace_multiplier
    }

    /// Build the discovery loop's HTTP client.
    ///
    /// Constructed here, at startup, rather than inside `discovery_loop`:
    /// an unreadable or malformed `EXTRA_CA_FILE` used to be fatal for the
    /// proxy client but only a warning on the discovery side, which
    /// *silently disabled discovery entirely* — the router would come up
    /// Ready, publish nothing, and 404 every request. One construction
    /// site, one failure mode, and it fails the process at startup.
    pub fn discovery_client(&self) -> Result<reqwest::Client, ConfigError> {
        self.apply_extra_ca(reqwest::Client::builder().timeout(Duration::from_secs(10)))?
            .build()
            .map_err(|e| ConfigError::Invalid {
                key: "OLLAMA_ROUTER_EXTRA_CA_FILE",
                reason: format!("could not build the discovery HTTP client: {e}"),
            })
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

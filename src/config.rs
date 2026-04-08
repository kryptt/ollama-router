use std::env;
use std::fmt;
use std::net::SocketAddr;

/// A named backend with its base URL.
/// Invariant: `name` is non-empty, `url` has no trailing slash.
#[derive(Debug, Clone)]
pub struct Backend {
    pub name: String,
    pub url: String,
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
        })
    }

    /// Construct a backend directly (bypassing env parsing). For tests.
    pub fn for_test(name: &str, url: &str) -> Self {
        Backend {
            name: name.to_string(),
            url: url.to_string(),
        }
    }
}

/// Validated, immutable configuration.
/// All parsing happens in `from_env`; once constructed, every field is valid.
#[derive(Debug, Clone)]
pub struct Config {
    pub backends: Vec<Backend>,
    pub discovery_interval_secs: u64,
    pub grace_multiplier: u64,
    pub tokens_file: Option<String>,
    pub public_addr: SocketAddr,
    pub internal_addr: SocketAddr,
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        let backends_str = env::var("OLLAMA_ROUTER_BACKENDS").unwrap_or_else(|_| {
            "ollama-cuda=http://ollama-cuda.ai:11434,\
             ollama-rocm=http://ollama-rocm.ai:11435"
                .to_string()
        });

        let backends: Vec<Backend> = backends_str
            .split(',')
            .map(|e| Backend::parse(e.trim()))
            .collect::<Result<Vec<_>, _>>()?;

        if backends.is_empty() {
            return Err(ConfigError::NoBackends);
        }

        let discovery_interval_secs = parse_env_u64("OLLAMA_ROUTER_DISCOVERY_INTERVAL", 60)?;
        let grace_multiplier = parse_env_u64("OLLAMA_ROUTER_GRACE_MULTIPLIER", 3)?;
        let tokens_file = env::var("OLLAMA_ROUTER_TOKENS_FILE").ok();
        let public_port = parse_env_u64("OLLAMA_ROUTER_PUBLIC_PORT", 11434)? as u16;
        let internal_port = parse_env_u64("OLLAMA_ROUTER_INTERNAL_PORT", 9090)? as u16;

        Ok(Config {
            backends,
            discovery_interval_secs,
            grace_multiplier,
            tokens_file,
            public_addr: SocketAddr::from(([0, 0, 0, 0], public_port)),
            internal_addr: SocketAddr::from(([0, 0, 0, 0], internal_port)),
        })
    }

    pub fn grace_period_secs(&self) -> u64 {
        self.discovery_interval_secs * self.grace_multiplier
    }

    /// Construct a config from explicit backends with sensible defaults. For tests.
    pub fn from_backends(backends: Vec<Backend>) -> Self {
        Config {
            backends,
            discovery_interval_secs: 60,
            grace_multiplier: 3,
            tokens_file: None,
            public_addr: SocketAddr::from(([127, 0, 0, 1], 0)),
            internal_addr: SocketAddr::from(([127, 0, 0, 1], 0)),
        }
    }
}

#[derive(Debug)]
pub enum ConfigError {
    InvalidBackend(String),
    NoBackends,
    InvalidValue { key: &'static str, value: String },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBackend(entry) => {
                write!(f, "invalid backend entry: '{entry}' (expected name=url)")
            }
            Self::NoBackends => {
                write!(f, "OLLAMA_ROUTER_BACKENDS must contain at least one backend")
            }
            Self::InvalidValue { key, value } => {
                write!(f, "{key} must be a positive integer, got '{value}'")
            }
        }
    }
}

impl std::error::Error for ConfigError {}

fn parse_env_u64(key: &'static str, default: u64) -> Result<u64, ConfigError> {
    match env::var(key) {
        Ok(val) => val
            .parse()
            .map_err(|_| ConfigError::InvalidValue { key, value: val }),
        Err(_) => Ok(default),
    }
}

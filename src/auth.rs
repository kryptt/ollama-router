use std::collections::HashSet;
use std::fs;

use tokio::sync::RwLock;

/// Reads bearer tokens from a file. Tokens are reloaded periodically
/// to pick up Secret volume updates without restart.
///
/// Auth is **fail-closed**: if a path is configured but the file is empty,
/// unreadable, or contains only comments, all requests are rejected.
pub struct TokenStore {
    path: Option<String>,
    tokens: RwLock<HashSet<String>>,
}

impl TokenStore {
    pub fn new(path: Option<&str>) -> Self {
        let tokens = path.map(load_tokens).unwrap_or_default();
        TokenStore {
            path: path.map(String::from),
            tokens: RwLock::new(tokens),
        }
    }

    pub async fn validate(&self, token: &str) -> bool {
        self.tokens.read().await.contains(token)
    }

    pub async fn reload(&self) {
        if let Some(ref path) = self.path {
            *self.tokens.write().await = load_tokens(path);
        }
    }

    /// Auth is enabled when a tokens file path is configured,
    /// regardless of whether it currently contains valid tokens.
    /// This ensures auth fails closed if the file is missing or empty.
    pub fn is_enabled(&self) -> bool {
        self.path.is_some()
    }
}

fn load_tokens(path: &str) -> HashSet<String> {
    match fs::read_to_string(path) {
        Ok(contents) => contents
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect(),
        Err(e) => {
            tracing::warn!(path, error = %e, "failed to load tokens file");
            HashSet::new()
        }
    }
}

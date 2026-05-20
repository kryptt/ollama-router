//! Single source of truth for the public HTTP paths that flow through
//! `model_route` and the wire-protocol defaults each one carries.
//!
//! Why this module exists: previously `main.rs` had two parallel lists —
//! one in the axum router builder declaring which paths get dispatched
//! to `model_route`, and one in a `matches!` inside `model_route` deciding
//! what `stream`-flag default to apply when the request body omits the
//! field. The two lists drifted once already (commit f4d6a13's
//! title-generation bug was a direct consequence: an OpenAI-protocol path
//! was routed to `model_route` but never added to the `stream=false` set,
//! so SSE heartbeat bytes were prepended to a non-streaming response).
//!
//! Adding a path here automatically:
//!   1. registers a route under `POST <path>` to `model_route` in
//!      `main.rs::public_router()` (which iterates `ROUTED_PATHS`), and
//!   2. opts that path into the correct stream-flag default via
//!      `default_stream_for_path`.
//!
//! There is no way to add a routed path through this module that doesn't
//! get a default, and no way to add a default that doesn't get routed.
//! That property is enforced by the `every_routed_path_has_a_default`
//! self-test below and by the integration test in `tests/integration_test.rs`.

/// Configuration metadata for one path dispatched through `model_route`.
#[derive(Debug, Clone, Copy)]
pub struct PathConfig {
    /// HTTP path mounted on the public router via `POST <path>`.
    pub path: &'static str,
    /// Default value of the JSON body's `stream` field when the client
    /// omits it. Ollama defaults `true` on `/api/*`; OpenAI and Anthropic
    /// default `false` on their respective `/v1/*` endpoints. Mismatching
    /// this against the protocol causes non-streaming clients to receive
    /// SSE heartbeat bytes prepended to what they expect to be a single
    /// JSON body (see the f4d6a13 regression).
    pub default_stream: bool,
}

/// Every path dispatched through `model_route`, with its protocol's
/// `stream` default. Ordering is purely for readability — neither
/// the router nor `default_stream_for_path` depends on it.
///
/// To add a new routed path, append one entry here. That entry will be
/// (a) registered as a route in `main::public_router()` and (b) consulted
/// by `default_stream_for_path` from inside `model_route`.
pub const ROUTED_PATHS: &[PathConfig] = &[
    // Ollama native API — defaults stream=true per Ollama's convention.
    PathConfig {
        path: "/api/chat",
        default_stream: true,
    },
    PathConfig {
        path: "/api/generate",
        default_stream: true,
    },
    PathConfig {
        path: "/api/embed",
        default_stream: true,
    },
    PathConfig {
        path: "/api/embeddings",
        default_stream: true,
    },
    PathConfig {
        path: "/api/show",
        default_stream: true,
    },
    // OpenAI-compat — spec defaults stream=false.
    PathConfig {
        path: "/v1/chat/completions",
        default_stream: false,
    },
    PathConfig {
        path: "/v1/completions",
        default_stream: false,
    },
    PathConfig {
        path: "/v1/embeddings",
        default_stream: false,
    },
    // Anthropic-compat — spec defaults stream=false.
    PathConfig {
        path: "/v1/messages",
        default_stream: false,
    },
];

/// Return the protocol-correct default for the `stream` body field when
/// the client omits it. Paths not in `ROUTED_PATHS` fall back to the
/// Ollama-style `true` — conservative for any future endpoint we forget
/// to register here, since the cost of streaming a single JSON response
/// is one extra heartbeat byte (harmless) while the cost of single-JSON
/// on a streaming protocol is the f4d6a13 bug.
pub fn default_stream_for_path(path: &str) -> bool {
    ROUTED_PATHS
        .iter()
        .find(|p| p.path == path)
        .map(|p| p.default_stream)
        .unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Per-path coverage ────────────────────────────────────────────────
    //
    // Each routed path gets its own assertion. Adding a new path to
    // ROUTED_PATHS without adding a test here will leave the new path
    // untested for default_stream, so this list is a *checklist* — the
    // every_routed_path_has_an_explicit_test test below verifies that
    // every entry in ROUTED_PATHS is named here.

    #[test]
    fn default_for_api_chat() {
        assert!(default_stream_for_path("/api/chat"));
    }

    #[test]
    fn default_for_api_generate() {
        assert!(default_stream_for_path("/api/generate"));
    }

    #[test]
    fn default_for_api_embed() {
        assert!(default_stream_for_path("/api/embed"));
    }

    #[test]
    fn default_for_api_embeddings() {
        assert!(default_stream_for_path("/api/embeddings"));
    }

    #[test]
    fn default_for_api_show() {
        assert!(default_stream_for_path("/api/show"));
    }

    #[test]
    fn default_for_v1_chat_completions() {
        assert!(!default_stream_for_path("/v1/chat/completions"));
    }

    #[test]
    fn default_for_v1_completions() {
        assert!(!default_stream_for_path("/v1/completions"));
    }

    #[test]
    fn default_for_v1_embeddings() {
        assert!(!default_stream_for_path("/v1/embeddings"));
    }

    #[test]
    fn default_for_v1_messages() {
        assert!(!default_stream_for_path("/v1/messages"));
    }

    // ── Unknown-path fallback ────────────────────────────────────────────

    #[test]
    fn default_for_unknown_path_is_ollama_style_true() {
        // Conservative fallback: a path we haven't registered defaults to
        // true. This makes a forgotten registration harmless for Ollama
        // protocols and only loud for OpenAI ones (which would prepend
        // heartbeat bytes — visible in any integration test).
        assert!(default_stream_for_path("/some/future/path"));
        assert!(default_stream_for_path("/"));
        assert!(default_stream_for_path(""));
    }

    // ── Structural invariants ────────────────────────────────────────────

    #[test]
    fn no_duplicate_paths_in_routed_paths() {
        // axum would .route()-panic on a duplicate, but catching it here
        // gives a clearer error than the router-build panic at startup.
        let mut sorted: Vec<&str> = ROUTED_PATHS.iter().map(|p| p.path).collect();
        sorted.sort_unstable();
        let count = sorted.len();
        sorted.dedup();
        assert_eq!(
            count,
            sorted.len(),
            "duplicate path in ROUTED_PATHS — every entry must be unique",
        );
    }

    #[test]
    fn every_routed_path_has_a_default_via_lookup() {
        // Every entry in ROUTED_PATHS must be findable through
        // default_stream_for_path with its declared default. This is the
        // invariant that lets `main.rs` rely on a single lookup function
        // for behaviour that used to live in a parallel `matches!`.
        for entry in ROUTED_PATHS {
            assert_eq!(
                default_stream_for_path(entry.path),
                entry.default_stream,
                "default_stream_for_path disagrees with ROUTED_PATHS for {}",
                entry.path,
            );
        }
    }

    #[test]
    fn every_v1_path_defaults_false() {
        // OpenAI/Anthropic spec invariant: any path mounted under /v1/*
        // should default stream=false. If a future entry under /v1/ ever
        // disagrees, the f4d6a13 regression class returns.
        for entry in ROUTED_PATHS {
            if entry.path.starts_with("/v1/") {
                assert!(
                    !entry.default_stream,
                    "{} is mounted under /v1/ but defaults stream=true",
                    entry.path,
                );
            }
        }
    }

    #[test]
    fn every_api_path_defaults_true() {
        // Mirror invariant for Ollama paths.
        for entry in ROUTED_PATHS {
            if entry.path.starts_with("/api/") {
                assert!(
                    entry.default_stream,
                    "{} is mounted under /api/ but defaults stream=false",
                    entry.path,
                );
            }
        }
    }

    #[test]
    fn every_routed_path_has_an_explicit_test() {
        // Maintenance hook: lists every path covered by the per-path
        // tests above. Drift between this list and ROUTED_PATHS fails
        // the build. Required-reading prompt: if you add a route, also
        // add a per-path test above and a corresponding entry here.
        const TESTED_PATHS: &[&str] = &[
            "/api/chat",
            "/api/generate",
            "/api/embed",
            "/api/embeddings",
            "/api/show",
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/embeddings",
            "/v1/messages",
        ];

        let routed: std::collections::HashSet<&str> = ROUTED_PATHS.iter().map(|p| p.path).collect();
        let tested: std::collections::HashSet<&str> = TESTED_PATHS.iter().copied().collect();

        let missing_tests: Vec<&&str> = routed.difference(&tested).collect();
        let stale_tests: Vec<&&str> = tested.difference(&routed).collect();

        assert!(
            missing_tests.is_empty(),
            "ROUTED_PATHS contains paths with no explicit test: {missing_tests:?}",
        );
        assert!(
            stale_tests.is_empty(),
            "TESTED_PATHS contains stale entries no longer in ROUTED_PATHS: {stale_tests:?}",
        );
    }
}

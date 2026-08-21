use std::env;
use std::sync::Mutex;

use ollama_router::config::Config;

static ENV_LOCK: Mutex<()> = Mutex::new(());

// SAFETY: env::set_var/remove_var are unsafe in edition 2024 because they are
// not thread-safe. Each test holds ENV_LOCK to serialize access.
unsafe fn clear_env() {
    for key in [
        "OLLAMA_ROUTER_BACKENDS",
        "OLLAMA_ROUTER_DISCOVERY_INTERVAL",
        "OLLAMA_ROUTER_GRACE_MULTIPLIER",
        "OLLAMA_ROUTER_TOKENS_FILE",
        "OLLAMA_ROUTER_PUBLIC_PORT",
        "OLLAMA_ROUTER_INTERNAL_PORT",
        "OLLAMA_ROUTER_CONNECT_TIMEOUT",
        "OLLAMA_ROUTER_REQUEST_TIMEOUT",
        "OLLAMA_ROUTER_LOADING_HEARTBEAT",
        "OLLAMA_ROUTER_PREFLIGHT_TIMEOUT",
        "OLLAMA_ROUTER_LOADING_MAX_WAIT",
        "OLLAMA_ROUTER_ESCALATE",
        "OLLAMA_ROUTER_MAX_RETRIES",
        "OLLAMA_ROUTER_RETRY_BACKOFF_BASE_MS",
        "OLLAMA_ROUTER_RETRY_JITTER_PCT",
        "OLLAMA_ROUTER_RETRY_LATENCY_BUDGET",
        "OLLAMA_ROUTER_BREAKER_5XX_THRESHOLD",
        "OLLAMA_ROUTER_BREAKER_OPEN",
        "OLLAMA_ROUTER_BACKEND_MAX_INFLIGHT",
        "OLLAMA_ROUTER_CACHE_ENABLED",
        "OLLAMA_ROUTER_CACHE_MAX_BYTES",
        "OLLAMA_ROUTER_CACHE_MAX_ENTRY_BYTES",
        "OLLAMA_ROUTER_CACHE_TTL",
        "OLLAMA_ROUTER_ALIASES_FILE",
        "OLLAMA_ROUTER_EXTERNAL_BACKENDS",
    ] {
        unsafe { env::remove_var(key) };
    }
}

/// Acquire the serialisation lock, wipe the environment before and after `f`,
/// then run `f` under a clean slate.  All env-mutating tests must go through
/// this helper so the lock + clear pattern is never duplicated.
fn with_clean_env<F: FnOnce()>(f: F) {
    let _lock = ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    unsafe { clear_env() };
    f();
    unsafe { clear_env() };
}

/// Inject all `vars` into the process environment.
///
/// # Safety
/// Caller must hold `ENV_LOCK`.
unsafe fn inject_vars(vars: &[(&str, &str)]) {
    for &(key, val) in vars {
        unsafe { env::set_var(key, val) };
    }
}

/// Set the given env vars, call `Config::from_env`, and return the parsed config.
/// Runs inside `with_clean_env` so the environment is pristine.
fn parse_with_vars(vars: &[(&str, &str)]) -> Config {
    let mut result = None;
    with_clean_env(|| {
        unsafe { inject_vars(vars) };
        result = Some(Config::from_env().unwrap());
    });
    result.unwrap()
}

/// Set the given env vars, call `Config::from_env`, and assert it fails with
/// an error message containing `expected_substring`.
fn assert_env_parse_error(vars: &[(&str, &str)], expected_substring: &str) {
    with_clean_env(|| {
        unsafe { inject_vars(vars) };
        let err = Config::from_env().unwrap_err();
        assert!(
            err.to_string().contains(expected_substring),
            "expected error containing {expected_substring:?}, got: {err}"
        );
    });
}

const ESCALATE_KEY: &str = "OLLAMA_ROUTER_ESCALATE";

#[test]
fn escalation_absent_or_empty_yields_no_rules() {
    // Unset: falls back to default (no rules).
    assert!(parse_with_vars(&[]).escalation_rules.is_empty());
    // Explicit empty string: also no rules.
    let from_empty = parse_with_vars(&[(ESCALATE_KEY, "")]);
    assert!(from_empty.escalation_rules.is_empty());
}

#[test]
fn escalation_parsed_from_env() {
    let config = parse_with_vars(&[(
        ESCALATE_KEY,
        "qwen3.6-medium:35000:qwen3.6-high,qwen3.6-high:120000:qwen3.6-ultra",
    )]);
    assert_eq!(config.escalation_rules.len(), 2);
    assert_eq!(config.escalation_rules[0].from_model, "qwen3.6-medium");
    assert_eq!(config.escalation_rules[0].max_input_tokens, 35_000);
    assert_eq!(config.escalation_rules[0].to_model, "qwen3.6-high");
    assert_eq!(config.escalation_rules[1].from_model, "qwen3.6-high");
    assert_eq!(config.escalation_rules[1].max_input_tokens, 120_000);
    assert_eq!(config.escalation_rules[1].to_model, "qwen3.6-ultra");
}

#[test]
fn escalation_malformed_rule_fails() {
    assert_env_parse_error(
        &[(ESCALATE_KEY, "qwen3.6-medium:nope:qwen3.6-high")],
        "invalid escalation rule",
    );
}

#[test]
fn escalation_zero_threshold_fails() {
    assert_env_parse_error(
        &[(ESCALATE_KEY, "qwen3.6-medium:0:qwen3.6-high")],
        "invalid escalation rule",
    );
}

#[test]
fn defaults_are_sane() {
    let config = parse_with_vars(&[]);
    assert_eq!(config.backends.len(), 1);
    assert_eq!(config.backends[0].name, "ollama");
    assert_eq!(config.backends[0].url, "http://localhost:11434");
    assert_eq!(config.discovery_interval_secs, 60);
    assert_eq!(config.grace_period_secs(), 180);
    assert!(config.tokens_file.is_none());
    assert_eq!(config.public_addr.port(), 11434);
    assert_eq!(config.internal_addr.port(), 9090);
    assert_eq!(config.connect_timeout_secs, 10);
    assert_eq!(config.request_timeout_secs, 300);
    assert_eq!(config.loading_heartbeat_secs, 15);
    assert_eq!(config.preflight_timeout_secs, 10);
    assert_eq!(config.loading_max_wait_secs, 300);
    // Resilience defaults (Unit 1).
    assert_eq!(config.max_retries, 2);
    assert_eq!(config.retry_backoff_base_ms, 100);
    assert_eq!(config.retry_jitter_pct, 25);
    assert_eq!(config.retry_latency_budget_secs, 30);
    assert_eq!(config.breaker_5xx_threshold, 5);
    assert_eq!(config.breaker_open_secs, 10);
    assert_eq!(config.backend_max_inflight, 0);
    // Cache defaults: off until validated.
    assert!(!config.cache_enabled);
    assert_eq!(config.cache_max_bytes, 64 * 1024 * 1024);
    assert_eq!(config.cache_max_entry_bytes, 1024 * 1024);
    assert_eq!(config.cache_ttl_secs, 3600);
}

#[test]
fn resilience_knobs_parsed_from_env() {
    let config = parse_with_vars(&[
        ("OLLAMA_ROUTER_MAX_RETRIES", "4"),
        ("OLLAMA_ROUTER_RETRY_BACKOFF_BASE_MS", "250"),
        ("OLLAMA_ROUTER_RETRY_JITTER_PCT", "50"),
        ("OLLAMA_ROUTER_RETRY_LATENCY_BUDGET", "60"),
        ("OLLAMA_ROUTER_BREAKER_5XX_THRESHOLD", "8"),
        ("OLLAMA_ROUTER_BREAKER_OPEN", "20"),
        ("OLLAMA_ROUTER_BACKEND_MAX_INFLIGHT", "16"),
    ]);
    assert_eq!(config.max_retries, 4);
    assert_eq!(config.retry_backoff_base_ms, 250);
    assert_eq!(config.retry_jitter_pct, 50);
    assert_eq!(config.retry_latency_budget_secs, 60);
    assert_eq!(config.breaker_5xx_threshold, 8);
    assert_eq!(config.breaker_open_secs, 20);
    assert_eq!(config.backend_max_inflight, 16);
}

#[test]
fn invalid_max_retries_fails() {
    assert_env_parse_error(
        &[("OLLAMA_ROUTER_MAX_RETRIES", "lots")],
        "must be a positive integer",
    );
}

#[test]
fn cache_knobs_parsed_from_env() {
    let config = parse_with_vars(&[
        ("OLLAMA_ROUTER_CACHE_ENABLED", "true"),
        ("OLLAMA_ROUTER_CACHE_MAX_BYTES", "1048576"),
        ("OLLAMA_ROUTER_CACHE_MAX_ENTRY_BYTES", "4096"),
        ("OLLAMA_ROUTER_CACHE_TTL", "120"),
    ]);
    assert!(config.cache_enabled);
    assert_eq!(config.cache_max_bytes, 1_048_576);
    assert_eq!(config.cache_max_entry_bytes, 4096);
    assert_eq!(config.cache_ttl_secs, 120);
}

#[test]
fn cache_enabled_accepts_bool_spellings() {
    with_clean_env(|| {
        for truthy in ["1", "on", "YES", "True"] {
            unsafe { env::set_var("OLLAMA_ROUTER_CACHE_ENABLED", truthy) };
            assert!(Config::from_env().unwrap().cache_enabled, "{truthy}");
        }
        for falsy in ["0", "off", "NO", "False"] {
            unsafe { env::set_var("OLLAMA_ROUTER_CACHE_ENABLED", falsy) };
            assert!(!Config::from_env().unwrap().cache_enabled, "{falsy}");
        }
    });
}

#[test]
fn invalid_cache_enabled_fails() {
    assert_env_parse_error(
        &[("OLLAMA_ROUTER_CACHE_ENABLED", "maybe")],
        "must be a boolean",
    );
}

#[test]
fn invalid_numeric_knobs_fail() {
    // Every numeric knob shares the parse_env_u64 error path; spot-check a
    // representative spread rather than only OLLAMA_ROUTER_MAX_RETRIES.
    for key in [
        "OLLAMA_ROUTER_RETRY_BACKOFF_BASE_MS",
        "OLLAMA_ROUTER_RETRY_LATENCY_BUDGET",
        "OLLAMA_ROUTER_BREAKER_OPEN",
        "OLLAMA_ROUTER_BACKEND_MAX_INFLIGHT",
        "OLLAMA_ROUTER_CACHE_MAX_BYTES",
        "OLLAMA_ROUTER_CACHE_TTL",
    ] {
        assert_env_parse_error(&[(key, "nope")], "must be a positive integer");
    }
}

#[test]
fn numeric_knobs_trim_surrounding_whitespace() {
    let config = parse_with_vars(&[("OLLAMA_ROUTER_MAX_RETRIES", "  4  ")]);
    assert_eq!(config.max_retries, 4);
}

#[test]
fn zero_sentinels_are_accepted() {
    // Documented sentinels: 0 retries (single-shot), 0 in-flight cap
    // (unlimited), 0 jitter, and 0 per-entry cap (no cap) are all valid.
    let config = parse_with_vars(&[
        ("OLLAMA_ROUTER_MAX_RETRIES", "0"),
        ("OLLAMA_ROUTER_BACKEND_MAX_INFLIGHT", "0"),
        ("OLLAMA_ROUTER_RETRY_JITTER_PCT", "0"),
        ("OLLAMA_ROUTER_CACHE_MAX_ENTRY_BYTES", "0"),
    ]);
    assert_eq!(config.max_retries, 0);
    assert_eq!(config.backend_max_inflight, 0);
    assert_eq!(config.retry_jitter_pct, 0);
    assert_eq!(config.cache_max_entry_bytes, 0);
}

#[test]
fn jitter_pct_above_100_fails() {
    assert_env_parse_error(&[("OLLAMA_ROUTER_RETRY_JITTER_PCT", "101")], "0\u{2013}100");
    // The 0-100 boundary itself is valid.
    let config = parse_with_vars(&[("OLLAMA_ROUTER_RETRY_JITTER_PCT", "100")]);
    assert_eq!(config.retry_jitter_pct, 100);
}

#[test]
fn breaker_threshold_zero_fails() {
    assert_env_parse_error(
        &[("OLLAMA_ROUTER_BREAKER_5XX_THRESHOLD", "0")],
        "at least 1",
    );
}

#[test]
fn cache_entry_cap_above_total_fails() {
    assert_env_parse_error(
        &[
            ("OLLAMA_ROUTER_CACHE_MAX_BYTES", "1000"),
            ("OLLAMA_ROUTER_CACHE_MAX_ENTRY_BYTES", "2000"),
        ],
        "must not exceed",
    );
}

#[test]
fn custom_backends_parsed() {
    let config = parse_with_vars(&[("OLLAMA_ROUTER_BACKENDS", "a=http://a:1234,b=http://b:5678")]);
    assert_eq!(config.backends.len(), 2);
    assert_eq!(config.backends[0].name, "a");
    assert_eq!(config.backends[0].url, "http://a:1234");
    assert_eq!(config.backends[1].name, "b");
    assert_eq!(config.backends[1].url, "http://b:5678");
}

#[test]
fn trailing_slash_stripped() {
    let config = parse_with_vars(&[("OLLAMA_ROUTER_BACKENDS", "x=http://host:1234/")]);
    assert_eq!(config.backends[0].url, "http://host:1234");
}

#[test]
fn empty_backends_fails() {
    with_clean_env(|| {
        unsafe { env::set_var("OLLAMA_ROUTER_BACKENDS", "") };
        assert!(Config::from_env().is_err());
    });
}

#[test]
fn malformed_backend_entry_fails() {
    assert_env_parse_error(
        &[("OLLAMA_ROUTER_BACKENDS", "no-equals-sign")],
        "expected name=url",
    );
}

#[test]
fn custom_discovery_interval() {
    let config = parse_with_vars(&[("OLLAMA_ROUTER_DISCOVERY_INTERVAL", "30")]);
    assert_eq!(config.discovery_interval_secs, 30);
    assert_eq!(config.grace_period_secs(), 90);
}

#[test]
fn invalid_discovery_interval_fails() {
    assert_env_parse_error(
        &[("OLLAMA_ROUTER_DISCOVERY_INTERVAL", "abc")],
        "must be a positive integer",
    );
}

#[test]
fn custom_timeouts() {
    let config = parse_with_vars(&[
        ("OLLAMA_ROUTER_CONNECT_TIMEOUT", "5"),
        ("OLLAMA_ROUTER_REQUEST_TIMEOUT", "600"),
    ]);
    assert_eq!(config.connect_timeout_secs, 5);
    assert_eq!(config.request_timeout_secs, 600);
}

#[test]
fn tokens_file_set_from_env() {
    let config = parse_with_vars(&[("OLLAMA_ROUTER_TOKENS_FILE", "/config/tokens")]);
    assert_eq!(config.tokens_file.as_deref(), Some("/config/tokens"));
}

// ── alias / external-backend env wiring ─────────────────────────────────────

/// Write `content` to a temp aliases file and return (dir, path). The dir
/// keeps the file alive for the test scope.
fn aliases_file_with(content: &str) -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("aliases");
    std::fs::write(&path, content).unwrap();
    let path = path.to_str().unwrap().to_string();
    (dir, path)
}

#[test]
fn aliases_file_validated_at_startup() {
    let (_dir, path) = aliases_file_with(
        "fast = swap/qwen3.6:latest | nous/Hermes-4-405B\nlocal secret = swap/qwen3.6:latest\n",
    );
    let config = parse_with_vars(&[
        ("OLLAMA_ROUTER_BACKENDS", "swap=http://s:1,nous=http://n:2"),
        ("OLLAMA_ROUTER_EXTERNAL_BACKENDS", "nous"),
        ("OLLAMA_ROUTER_ALIASES_FILE", &path),
    ]);
    assert_eq!(config.aliases_file.as_deref(), Some(path.as_str()));
    assert!(config.external_backends.contains("nous"));
    assert!(!config.external_backends.contains("swap"));
}

#[test]
fn local_alias_with_external_backend_fails_at_startup() {
    let (_dir, path) = aliases_file_with("local secret = nous/Hermes-4-405B\n");
    assert_env_parse_error(
        &[
            ("OLLAMA_ROUTER_BACKENDS", "swap=http://s:1,nous=http://n:2"),
            ("OLLAMA_ROUTER_EXTERNAL_BACKENDS", "nous"),
            ("OLLAMA_ROUTER_ALIASES_FILE", &path),
        ],
        "marked local",
    );
}

#[test]
fn external_backends_naming_unknown_backend_fails() {
    // Privacy-critical typo check: a misspelled external backend would
    // silently stop guarding `local` aliases against the real one.
    assert_env_parse_error(
        &[
            ("OLLAMA_ROUTER_BACKENDS", "swap=http://s:1,nous=http://n:2"),
            ("OLLAMA_ROUTER_EXTERNAL_BACKENDS", "nuos"),
        ],
        "not in OLLAMA_ROUTER_BACKENDS",
    );
}

#[test]
fn unreadable_aliases_file_fails_at_startup() {
    assert_env_parse_error(
        &[("OLLAMA_ROUTER_ALIASES_FILE", "/nonexistent/aliases")],
        "could not read",
    );
}

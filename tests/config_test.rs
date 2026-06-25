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
    ] {
        unsafe { env::remove_var(key) };
    }
}

#[test]
fn escalation_unset_means_empty() {
    let _lock = ENV_LOCK.lock().unwrap();
    unsafe { clear_env() };
    let config = Config::from_env().unwrap();
    assert!(config.escalation_rules.is_empty());
    unsafe { clear_env() };
}

#[test]
fn escalation_parsed_from_env() {
    let _lock = ENV_LOCK.lock().unwrap();
    unsafe { clear_env() };
    unsafe {
        env::set_var(
            "OLLAMA_ROUTER_ESCALATE",
            "qwen3.6-medium:35000:qwen3.6-high,qwen3.6-high:120000:qwen3.6-ultra",
        )
    };
    let config = Config::from_env().unwrap();
    assert_eq!(config.escalation_rules.len(), 2);
    assert_eq!(config.escalation_rules[0].from_model, "qwen3.6-medium");
    assert_eq!(config.escalation_rules[0].max_input_tokens, 35_000);
    assert_eq!(config.escalation_rules[0].to_model, "qwen3.6-high");
    assert_eq!(config.escalation_rules[1].from_model, "qwen3.6-high");
    assert_eq!(config.escalation_rules[1].max_input_tokens, 120_000);
    assert_eq!(config.escalation_rules[1].to_model, "qwen3.6-ultra");
    unsafe { clear_env() };
}

#[test]
fn escalation_empty_string_is_empty() {
    let _lock = ENV_LOCK.lock().unwrap();
    unsafe { clear_env() };
    unsafe { env::set_var("OLLAMA_ROUTER_ESCALATE", "") };
    let config = Config::from_env().unwrap();
    assert!(config.escalation_rules.is_empty());
    unsafe { clear_env() };
}

#[test]
fn escalation_malformed_rule_fails() {
    let _lock = ENV_LOCK.lock().unwrap();
    unsafe { clear_env() };
    unsafe { env::set_var("OLLAMA_ROUTER_ESCALATE", "qwen3.6-medium:nope:qwen3.6-high") };
    let err = Config::from_env().unwrap_err();
    assert!(err.to_string().contains("invalid escalation rule"));
    unsafe { clear_env() };
}

#[test]
fn escalation_zero_threshold_fails() {
    let _lock = ENV_LOCK.lock().unwrap();
    unsafe { clear_env() };
    unsafe { env::set_var("OLLAMA_ROUTER_ESCALATE", "qwen3.6-medium:0:qwen3.6-high") };
    let err = Config::from_env().unwrap_err();
    assert!(err.to_string().contains("invalid escalation rule"));
    unsafe { clear_env() };
}

#[test]
fn defaults_are_sane() {
    let _lock = ENV_LOCK.lock().unwrap();
    unsafe { clear_env() };
    let config = Config::from_env().unwrap();
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
    let _lock = ENV_LOCK.lock().unwrap();
    unsafe { clear_env() };
    unsafe {
        env::set_var("OLLAMA_ROUTER_MAX_RETRIES", "4");
        env::set_var("OLLAMA_ROUTER_RETRY_BACKOFF_BASE_MS", "250");
        env::set_var("OLLAMA_ROUTER_RETRY_JITTER_PCT", "50");
        env::set_var("OLLAMA_ROUTER_RETRY_LATENCY_BUDGET", "60");
        env::set_var("OLLAMA_ROUTER_BREAKER_5XX_THRESHOLD", "8");
        env::set_var("OLLAMA_ROUTER_BREAKER_OPEN", "20");
        env::set_var("OLLAMA_ROUTER_BACKEND_MAX_INFLIGHT", "16");
    };
    let config = Config::from_env().unwrap();
    assert_eq!(config.max_retries, 4);
    assert_eq!(config.retry_backoff_base_ms, 250);
    assert_eq!(config.retry_jitter_pct, 50);
    assert_eq!(config.retry_latency_budget_secs, 60);
    assert_eq!(config.breaker_5xx_threshold, 8);
    assert_eq!(config.breaker_open_secs, 20);
    assert_eq!(config.backend_max_inflight, 16);
    unsafe { clear_env() };
}

#[test]
fn invalid_max_retries_fails() {
    let _lock = ENV_LOCK.lock().unwrap();
    unsafe { clear_env() };
    unsafe { env::set_var("OLLAMA_ROUTER_MAX_RETRIES", "lots") };
    let err = Config::from_env().unwrap_err();
    assert!(err.to_string().contains("must be a positive integer"));
    unsafe { clear_env() };
}

#[test]
fn cache_knobs_parsed_from_env() {
    let _lock = ENV_LOCK.lock().unwrap();
    unsafe { clear_env() };
    unsafe {
        env::set_var("OLLAMA_ROUTER_CACHE_ENABLED", "true");
        env::set_var("OLLAMA_ROUTER_CACHE_MAX_BYTES", "1048576");
        env::set_var("OLLAMA_ROUTER_CACHE_MAX_ENTRY_BYTES", "4096");
        env::set_var("OLLAMA_ROUTER_CACHE_TTL", "120");
    };
    let config = Config::from_env().unwrap();
    assert!(config.cache_enabled);
    assert_eq!(config.cache_max_bytes, 1_048_576);
    assert_eq!(config.cache_max_entry_bytes, 4096);
    assert_eq!(config.cache_ttl_secs, 120);
    unsafe { clear_env() };
}

#[test]
fn cache_enabled_accepts_bool_spellings() {
    let _lock = ENV_LOCK.lock().unwrap();
    unsafe { clear_env() };
    for truthy in ["1", "on", "YES", "True"] {
        unsafe { env::set_var("OLLAMA_ROUTER_CACHE_ENABLED", truthy) };
        assert!(Config::from_env().unwrap().cache_enabled, "{truthy}");
    }
    for falsy in ["0", "off", "NO", "False"] {
        unsafe { env::set_var("OLLAMA_ROUTER_CACHE_ENABLED", falsy) };
        assert!(!Config::from_env().unwrap().cache_enabled, "{falsy}");
    }
    unsafe { clear_env() };
}

#[test]
fn invalid_cache_enabled_fails() {
    let _lock = ENV_LOCK.lock().unwrap();
    unsafe { clear_env() };
    unsafe { env::set_var("OLLAMA_ROUTER_CACHE_ENABLED", "maybe") };
    let err = Config::from_env().unwrap_err();
    assert!(err.to_string().contains("must be a boolean"), "{err}");
    unsafe { clear_env() };
}

#[test]
fn invalid_numeric_knobs_fail() {
    let _lock = ENV_LOCK.lock().unwrap();
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
        unsafe { clear_env() };
        unsafe { env::set_var(key, "nope") };
        let err = Config::from_env().unwrap_err();
        assert!(
            err.to_string().contains("must be a positive integer"),
            "{key}: {err}"
        );
    }
    unsafe { clear_env() };
}

#[test]
fn numeric_knobs_trim_surrounding_whitespace() {
    let _lock = ENV_LOCK.lock().unwrap();
    unsafe { clear_env() };
    unsafe { env::set_var("OLLAMA_ROUTER_MAX_RETRIES", "  4  ") };
    assert_eq!(Config::from_env().unwrap().max_retries, 4);
    unsafe { clear_env() };
}

#[test]
fn zero_sentinels_are_accepted() {
    let _lock = ENV_LOCK.lock().unwrap();
    unsafe { clear_env() };
    // Documented sentinels: 0 retries (single-shot), 0 in-flight cap
    // (unlimited), 0 jitter, and 0 per-entry cap (no cap) are all valid.
    unsafe {
        env::set_var("OLLAMA_ROUTER_MAX_RETRIES", "0");
        env::set_var("OLLAMA_ROUTER_BACKEND_MAX_INFLIGHT", "0");
        env::set_var("OLLAMA_ROUTER_RETRY_JITTER_PCT", "0");
        env::set_var("OLLAMA_ROUTER_CACHE_MAX_ENTRY_BYTES", "0");
    };
    let config = Config::from_env().unwrap();
    assert_eq!(config.max_retries, 0);
    assert_eq!(config.backend_max_inflight, 0);
    assert_eq!(config.retry_jitter_pct, 0);
    assert_eq!(config.cache_max_entry_bytes, 0);
    unsafe { clear_env() };
}

#[test]
fn jitter_pct_above_100_fails() {
    let _lock = ENV_LOCK.lock().unwrap();
    unsafe { clear_env() };
    unsafe { env::set_var("OLLAMA_ROUTER_RETRY_JITTER_PCT", "101") };
    let err = Config::from_env().unwrap_err();
    assert!(err.to_string().contains("0\u{2013}100"), "{err}");
    unsafe { clear_env() };
    // The 0–100 boundary itself is valid.
    unsafe { env::set_var("OLLAMA_ROUTER_RETRY_JITTER_PCT", "100") };
    assert_eq!(Config::from_env().unwrap().retry_jitter_pct, 100);
    unsafe { clear_env() };
}

#[test]
fn breaker_threshold_zero_fails() {
    let _lock = ENV_LOCK.lock().unwrap();
    unsafe { clear_env() };
    unsafe { env::set_var("OLLAMA_ROUTER_BREAKER_5XX_THRESHOLD", "0") };
    let err = Config::from_env().unwrap_err();
    assert!(err.to_string().contains("at least 1"), "{err}");
    unsafe { clear_env() };
}

#[test]
fn cache_entry_cap_above_total_fails() {
    let _lock = ENV_LOCK.lock().unwrap();
    unsafe { clear_env() };
    unsafe {
        env::set_var("OLLAMA_ROUTER_CACHE_MAX_BYTES", "1000");
        env::set_var("OLLAMA_ROUTER_CACHE_MAX_ENTRY_BYTES", "2000");
    };
    let err = Config::from_env().unwrap_err();
    assert!(err.to_string().contains("must not exceed"), "{err}");
    unsafe { clear_env() };
}

#[test]
fn custom_backends_parsed() {
    let _lock = ENV_LOCK.lock().unwrap();
    unsafe { clear_env() };
    unsafe { env::set_var("OLLAMA_ROUTER_BACKENDS", "a=http://a:1234,b=http://b:5678") };
    let config = Config::from_env().unwrap();
    assert_eq!(config.backends.len(), 2);
    assert_eq!(config.backends[0].name, "a");
    assert_eq!(config.backends[0].url, "http://a:1234");
    assert_eq!(config.backends[1].name, "b");
    assert_eq!(config.backends[1].url, "http://b:5678");
    unsafe { clear_env() };
}

#[test]
fn trailing_slash_stripped() {
    let _lock = ENV_LOCK.lock().unwrap();
    unsafe { clear_env() };
    unsafe { env::set_var("OLLAMA_ROUTER_BACKENDS", "x=http://host:1234/") };
    let config = Config::from_env().unwrap();
    assert_eq!(config.backends[0].url, "http://host:1234");
    unsafe { clear_env() };
}

#[test]
fn empty_backends_fails() {
    let _lock = ENV_LOCK.lock().unwrap();
    unsafe { clear_env() };
    unsafe { env::set_var("OLLAMA_ROUTER_BACKENDS", "") };
    assert!(Config::from_env().is_err());
    unsafe { clear_env() };
}

#[test]
fn malformed_backend_entry_fails() {
    let _lock = ENV_LOCK.lock().unwrap();
    unsafe { clear_env() };
    unsafe { env::set_var("OLLAMA_ROUTER_BACKENDS", "no-equals-sign") };
    let err = Config::from_env().unwrap_err();
    assert!(err.to_string().contains("expected name=url"));
    unsafe { clear_env() };
}

#[test]
fn custom_discovery_interval() {
    let _lock = ENV_LOCK.lock().unwrap();
    unsafe { clear_env() };
    unsafe { env::set_var("OLLAMA_ROUTER_DISCOVERY_INTERVAL", "30") };
    let config = Config::from_env().unwrap();
    assert_eq!(config.discovery_interval_secs, 30);
    assert_eq!(config.grace_period_secs(), 90);
    unsafe { clear_env() };
}

#[test]
fn invalid_discovery_interval_fails() {
    let _lock = ENV_LOCK.lock().unwrap();
    unsafe { clear_env() };
    unsafe { env::set_var("OLLAMA_ROUTER_DISCOVERY_INTERVAL", "abc") };
    let err = Config::from_env().unwrap_err();
    assert!(err.to_string().contains("must be a positive integer"));
    unsafe { clear_env() };
}

#[test]
fn custom_timeouts() {
    let _lock = ENV_LOCK.lock().unwrap();
    unsafe { clear_env() };
    unsafe { env::set_var("OLLAMA_ROUTER_CONNECT_TIMEOUT", "5") };
    unsafe { env::set_var("OLLAMA_ROUTER_REQUEST_TIMEOUT", "600") };
    let config = Config::from_env().unwrap();
    assert_eq!(config.connect_timeout_secs, 5);
    assert_eq!(config.request_timeout_secs, 600);
    unsafe { clear_env() };
}

#[test]
fn tokens_file_set_from_env() {
    let _lock = ENV_LOCK.lock().unwrap();
    unsafe { clear_env() };
    unsafe { env::set_var("OLLAMA_ROUTER_TOKENS_FILE", "/config/tokens") };
    let config = Config::from_env().unwrap();
    assert_eq!(config.tokens_file.as_deref(), Some("/config/tokens"));
    unsafe { clear_env() };
}

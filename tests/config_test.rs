use std::env;
use std::sync::Mutex;

use ollama_router::config::Config;

static ENV_LOCK: Mutex<()> = Mutex::new(());

// SAFETY: env::set_var/remove_var are unsafe in edition 2024 because they are
// not thread-safe. Each test holds ENV_LOCK to serialize access.
unsafe fn clear_env() {
    for key in [
        "OLLAMA_ROUTER_CONFIG",
        "OLLAMA_ROUTER_DISCOVERY_INTERVAL",
        "OLLAMA_ROUTER_GRACE_MULTIPLIER",
        "OLLAMA_ROUTER_TOKENS_FILE",
        "OLLAMA_ROUTER_EXTRA_CA_FILE",
        "OLLAMA_ROUTER_PUBLIC_PORT",
        "OLLAMA_ROUTER_INTERNAL_PORT",
        "OLLAMA_ROUTER_CONNECT_TIMEOUT",
        "OLLAMA_ROUTER_REQUEST_TIMEOUT",
        "OLLAMA_ROUTER_LOADING_HEARTBEAT",
        "OLLAMA_ROUTER_PREFLIGHT_TIMEOUT",
        "OLLAMA_ROUTER_LOADING_MAX_WAIT",
        "OLLAMA_ROUTER_ESCALATE",
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

/// A placeholder policy-file path. `Config::from_env` only records the
/// path — loading and validating the document is `policy::FileConfig`'s job
/// and is covered by its own tests — so it never has to exist.
const CONFIG_PATH: &str = "/etc/ollama-router/router.toml";

/// Set the given env vars, call `Config::from_env`, and return the parsed config.
/// Runs inside `with_clean_env` so the environment is pristine.
fn parse_with_vars(vars: &[(&str, &str)]) -> Config {
    let mut result = None;
    with_clean_env(|| {
        unsafe { env::set_var("OLLAMA_ROUTER_CONFIG", CONFIG_PATH) };
        unsafe { inject_vars(vars) };
        result = Some(Config::from_env().unwrap());
    });
    result.unwrap()
}

/// Set the given env vars, call `Config::from_env`, and assert it fails with
/// an error message containing `expected_substring`.
fn assert_env_parse_error(vars: &[(&str, &str)], expected_substring: &str) {
    with_clean_env(|| {
        unsafe { env::set_var("OLLAMA_ROUTER_CONFIG", CONFIG_PATH) };
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
    assert_eq!(config.config_path, CONFIG_PATH);
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
}

#[test]
fn invalid_numeric_knobs_fail() {
    // Every numeric knob shares the parse_env_u64 error path; spot-check a
    // representative spread rather than a single key.
    for key in [
        "OLLAMA_ROUTER_DISCOVERY_INTERVAL",
        "OLLAMA_ROUTER_CONNECT_TIMEOUT",
        "OLLAMA_ROUTER_REQUEST_TIMEOUT",
        "OLLAMA_ROUTER_GRACE_MULTIPLIER",
        "OLLAMA_ROUTER_LOADING_HEARTBEAT",
    ] {
        assert_env_parse_error(&[(key, "nope")], "must be a positive integer");
    }
}

#[test]
fn numeric_knobs_trim_surrounding_whitespace() {
    let config = parse_with_vars(&[("OLLAMA_ROUTER_CONNECT_TIMEOUT", "  4  ")]);
    assert_eq!(config.connect_timeout_secs, 4);
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

// ── the policy-file path ────────────────────────────────────────────────────

#[test]
fn config_path_is_required() {
    // No default: the backend roster and the spend boundary both live in
    // the policy file, so a router without one has nothing to route to.
    with_clean_env(|| {
        let err = Config::from_env().unwrap_err();
        assert!(
            err.to_string().contains("OLLAMA_ROUTER_CONFIG must be set"),
            "{err}"
        );
    });
}

#[test]
fn empty_config_path_is_rejected_like_an_absent_one() {
    with_clean_env(|| {
        unsafe { env::set_var("OLLAMA_ROUTER_CONFIG", "   ") };
        assert!(Config::from_env().is_err());
    });
}

#[test]
fn empty_tokens_file_is_a_hard_startup_error() {
    // The two ways to be lenient here are not symmetric, and both are bad:
    // treating "" as unset DISABLES AUTH ENTIRELY (a manifest whose
    // valueFrom renders empty brings the router up wide open), while
    // keeping "" points the fail-closed store at an unreadable path and
    // 401s everything for no visible reason. So: refuse to start.
    for empty in ["", "   "] {
        assert_env_parse_error(&[("OLLAMA_ROUTER_TOKENS_FILE", empty)], "set but empty");
    }
}

#[test]
fn absent_tokens_file_is_the_only_way_to_disable_auth() {
    let config = parse_with_vars(&[]);
    assert!(config.tokens_file.is_none());
}

#[test]
fn unreadable_extra_ca_is_fatal_at_startup_for_the_discovery_client() {
    // Symmetry with the proxy client. The old asymmetry — fatal for the
    // proxy, a warning that silently disabled discovery — produced a router
    // that came up Ready, published nothing, and 404'd everything.
    let config = parse_with_vars(&[("OLLAMA_ROUTER_EXTRA_CA_FILE", "/nonexistent/ca.pem")]);
    let err = config
        .discovery_client()
        .expect_err("an unreadable CA bundle must fail");
    assert!(err.to_string().contains("could not be read"), "{err}");
}

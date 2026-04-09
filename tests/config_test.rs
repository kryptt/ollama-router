use std::env;

use ollama_router::config::Config;

// SAFETY: env::set_var/remove_var are unsafe in edition 2024 because they are
// not thread-safe. These tests must run with --test-threads=1.
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
    ] {
        unsafe { env::remove_var(key) };
    }
}

#[test]
fn defaults_are_sane() {
    unsafe { clear_env() };
    let config = Config::from_env().unwrap();
    assert_eq!(config.backends.len(), 2);
    assert_eq!(config.backends[0].name, "ollama-cuda");
    assert_eq!(config.backends[0].url, "http://ollama-cuda.ai:11434");
    assert_eq!(config.backends[1].name, "ollama-rocm");
    assert_eq!(config.backends[1].url, "http://ollama-rocm.ai:11435");
    assert_eq!(config.discovery_interval_secs, 60);
    assert_eq!(config.grace_period_secs(), 180);
    assert!(config.tokens_file.is_none());
    assert_eq!(config.public_addr.port(), 11434);
    assert_eq!(config.internal_addr.port(), 9090);
    assert_eq!(config.connect_timeout_secs, 10);
    assert_eq!(config.request_timeout_secs, 300);
}

#[test]
fn custom_backends_parsed() {
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
    unsafe { clear_env() };
    unsafe { env::set_var("OLLAMA_ROUTER_BACKENDS", "x=http://host:1234/") };
    let config = Config::from_env().unwrap();
    assert_eq!(config.backends[0].url, "http://host:1234");
    unsafe { clear_env() };
}

#[test]
fn empty_backends_fails() {
    unsafe { clear_env() };
    unsafe { env::set_var("OLLAMA_ROUTER_BACKENDS", "") };
    assert!(Config::from_env().is_err());
    unsafe { clear_env() };
}

#[test]
fn malformed_backend_entry_fails() {
    unsafe { clear_env() };
    unsafe { env::set_var("OLLAMA_ROUTER_BACKENDS", "no-equals-sign") };
    let err = Config::from_env().unwrap_err();
    assert!(err.to_string().contains("expected name=url"));
    unsafe { clear_env() };
}

#[test]
fn custom_discovery_interval() {
    unsafe { clear_env() };
    unsafe { env::set_var("OLLAMA_ROUTER_DISCOVERY_INTERVAL", "30") };
    let config = Config::from_env().unwrap();
    assert_eq!(config.discovery_interval_secs, 30);
    assert_eq!(config.grace_period_secs(), 90);
    unsafe { clear_env() };
}

#[test]
fn invalid_discovery_interval_fails() {
    unsafe { clear_env() };
    unsafe { env::set_var("OLLAMA_ROUTER_DISCOVERY_INTERVAL", "abc") };
    let err = Config::from_env().unwrap_err();
    assert!(err.to_string().contains("must be a positive integer"));
    unsafe { clear_env() };
}

#[test]
fn custom_timeouts() {
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
    unsafe { clear_env() };
    unsafe { env::set_var("OLLAMA_ROUTER_TOKENS_FILE", "/config/tokens") };
    let config = Config::from_env().unwrap();
    assert_eq!(config.tokens_file.as_deref(), Some("/config/tokens"));
    unsafe { clear_env() };
}

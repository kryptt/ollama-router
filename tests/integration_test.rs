use std::time::Duration;

use axum::Router;
use axum::body::Bytes;
use axum::http::StatusCode;
use axum::routing::{any, get, post};
use tokio::net::TcpListener;

use ollama_router::auth::TokenStore;
use ollama_router::config::{Backend, Config};
use ollama_router::registry;
use ollama_router::routes::{ROUTED_PATHS, default_stream_for_path};

async fn start_mock_backend(
    models: Vec<&str>,
) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
    let tags_json = serde_json::json!({
        "models": models.iter().map(|m| serde_json::json!({"name": m})).collect::<Vec<_>>()
    })
    .to_string();

    let app = Router::new()
        .route(
            "/api/tags",
            get({
                let tags = tags_json.clone();
                move || {
                    let tags = tags.clone();
                    async move { (StatusCode::OK, tags) }
                }
            }),
        )
        .route(
            "/api/chat",
            post(|body: Bytes| async move { (StatusCode::OK, format!("echoed: {}", body.len())) }),
        )
        .route(
            "/api/version",
            get(|| async { (StatusCode::OK, r#"{"version":"0.9.0"}"#) }),
        )
        .fallback(any(
            |uri: axum::extract::OriginalUri, body: Bytes| async move {
                (
                    StatusCode::OK,
                    format!("fallback: {} {}", uri.0.path(), body.len()),
                )
            },
        ));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    (addr, handle)
}

#[tokio::test]
async fn model_routing_to_correct_backend() {
    let (cuda_addr, _h1) = start_mock_backend(vec!["fixt/home-3b-v3:latest"]).await;
    let (rocm_addr, _h2) = start_mock_backend(vec!["glm-4.7-flash:latest"]).await;

    let config = Config::from_backends(vec![
        Backend::for_test("cuda", &format!("http://{cuda_addr}")),
        Backend::for_test("rocm", &format!("http://{rocm_addr}")),
    ]);

    let reg = registry::new_shared(&config);

    tokio::spawn({
        let reg = reg.clone();
        let config = config.clone();
        async move { registry::discovery_loop(reg, config).await }
    });

    tokio::time::sleep(Duration::from_millis(500)).await;

    let r = reg.read().await;
    assert!(r.is_discovery_done());

    let cuda_id = r.lookup("fixt/home-3b-v3:latest").unwrap();
    assert_eq!(r.backend(cuda_id).name, "cuda");

    let rocm_id = r.lookup("glm-4.7-flash:latest").unwrap();
    assert_eq!(r.backend(rocm_id).name, "rocm");

    let prefix_id = r.lookup("glm-4.7-flash").unwrap();
    assert_eq!(r.backend(prefix_id).name, "rocm");

    assert!(r.lookup("nonexistent").is_none());
}

#[tokio::test]
async fn health_before_discovery_is_not_ready() {
    let config = Config::from_backends(vec![Backend::for_test("test", "http://127.0.0.1:1")]);

    let reg = registry::new_shared(&config);
    let r = reg.read().await;
    assert!(!r.is_discovery_done());
}

#[tokio::test]
async fn discovery_marks_unreachable_backend_down() {
    let config = Config::from_backends(vec![Backend::for_test("dead", "http://127.0.0.1:1")]);

    let reg = registry::new_shared(&config);

    tokio::spawn({
        let reg = reg.clone();
        let config = config.clone();
        async move { registry::discovery_loop(reg, config).await }
    });

    tokio::time::sleep(Duration::from_millis(500)).await;

    let r = reg.read().await;
    assert!(r.is_discovery_done());
    assert!(r.any_healthy().is_none());
    assert!(r.available_model_names().is_empty());
}

#[tokio::test]
async fn token_store_validates_correctly() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokens");
    std::fs::write(&path, "token-abc\ntoken-def\n# comment\n\n").unwrap();

    let store = TokenStore::new(Some(path.to_str().unwrap()));

    assert!(store.is_enabled());
    assert!(store.validate("token-abc").await);
    assert!(store.validate("token-def").await);
    assert!(!store.validate("token-xyz").await);
    assert!(!store.validate("# comment").await);
    assert!(!store.validate("").await);
}

#[tokio::test]
async fn token_store_no_file_disables_auth() {
    let store = TokenStore::new(None);
    assert!(!store.is_enabled());
}

#[tokio::test]
async fn token_store_reload_picks_up_changes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokens");
    std::fs::write(&path, "old-token\n").unwrap();

    let store = TokenStore::new(Some(path.to_str().unwrap()));
    assert!(store.validate("old-token").await);
    assert!(!store.validate("new-token").await);

    std::fs::write(&path, "new-token\n").unwrap();
    store.reload().await;

    assert!(!store.validate("old-token").await);
    assert!(store.validate("new-token").await);
}

#[tokio::test]
async fn token_store_empty_file_fails_closed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokens");
    std::fs::write(&path, "# only a comment\n\n").unwrap();

    let store = TokenStore::new(Some(path.to_str().unwrap()));

    // Auth is enabled (path configured) but no valid tokens → all rejected
    assert!(store.is_enabled());
    assert!(!store.validate("anything").await);
}

#[tokio::test]
async fn token_store_missing_file_fails_closed() {
    let store = TokenStore::new(Some("/nonexistent/path/tokens"));

    // Path configured but file missing → auth enabled, all rejected
    assert!(store.is_enabled());
    assert!(!store.validate("anything").await);
}

// ─── Routes contract (item #7 from the 2026-05-20 review) ────────────────────
//
// These integration tests prove that the single-source-of-truth contract in
// `ollama_router::routes` actually holds at the axum layer. The unit tests
// in `src/routes.rs` cover the *data*; these cover the *wiring*.

/// Build a stand-in router that mounts every `ROUTED_PATHS` entry to a
/// handler that just echoes its path. This is the same loop `main.rs`
/// uses (`for entry in ROUTED_PATHS { router.route(entry.path, ...) }`),
/// so if a path string ever stops being acceptable to axum, this test
/// fails before the production binary panics at startup.
fn build_routed_paths_only_router() -> Router {
    let mut router = Router::new();
    for entry in ROUTED_PATHS {
        let path = entry.path;
        router = router.route(
            path,
            post(move || async move { (StatusCode::OK, format!("routed: {path}")) }),
        );
    }
    router.fallback(any(|| async { StatusCode::NOT_FOUND }))
}

#[tokio::test]
async fn every_routed_path_actually_routes_through_axum() {
    let app = build_routed_paths_only_router();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = reqwest::Client::new();
    for entry in ROUTED_PATHS {
        let url = format!("http://{}{}", addr, entry.path);
        let resp = client.post(&url).body("{}").send().await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "ROUTED_PATHS entry {} did not route through axum (got {})",
            entry.path,
            resp.status(),
        );
        let body = resp.text().await.unwrap();
        assert_eq!(body, format!("routed: {}", entry.path));
    }
}

#[tokio::test]
async fn paths_not_in_routed_paths_get_404() {
    // Sanity: the fallback wired above must catch anything not declared
    // in ROUTED_PATHS. If a future change makes the router permissive
    // (e.g. wildcard match that swallows everything), this fails.
    let app = build_routed_paths_only_router();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = reqwest::Client::new();
    for unknown in &[
        "/some/future/path",
        "/v1/audio/transcriptions", // OpenAI Whisper — not in ROUTED_PATHS
        "/api/version",             // valid Ollama endpoint, not in our model_route set
        "/",
    ] {
        let url = format!("http://{addr}{unknown}");
        let resp = client.post(&url).body("{}").send().await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::NOT_FOUND,
            "expected {unknown} to 404 (not in ROUTED_PATHS), got {}",
            resp.status(),
        );
    }
}

#[test]
fn routes_default_stream_matches_path_protocol() {
    // End-to-end contract: every /v1/* path is OpenAI/Anthropic (default
    // stream=false); every /api/* path is Ollama (default stream=true).
    // This is the f4d6a13 regression class — pinning here so any future
    // path addition either matches the invariant or visibly fails CI.
    for entry in ROUTED_PATHS {
        let expected = !entry.path.starts_with("/v1/");
        assert_eq!(
            default_stream_for_path(entry.path),
            expected,
            "{} violates the /v1/ vs /api/ default-stream convention",
            entry.path,
        );
    }
}

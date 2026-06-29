use std::time::Duration;

use axum::Router;
use axum::body::Bytes;
use axum::http::StatusCode;
use axum::routing::{any, get, post};
use reqwest::Client;
use tokio::net::TcpListener;

use ollama_router::auth::TokenStore;
use ollama_router::config::{Backend, Config};
use ollama_router::registry::{self, Registry, SharedRegistry};
use ollama_router::routes::{ROUTED_PATHS, default_stream_for_path};

// ─── Test helpers ────────────────────────────────────────────────────────────

async fn spawn_test_server(app: Router) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

fn spawn_discovery(reg: &SharedRegistry, config: &Config) {
    tokio::spawn({
        let reg = reg.clone();
        let config = config.clone();
        async move { registry::discovery_loop(reg, config).await }
    });
}

/// Sleep for discovery to complete, assert it finished, and return the
/// read guard so callers can inspect the registry without a second lock.
async fn run_discovery_to_completion<'a>(
    reg: &'a SharedRegistry,
    config: &Config,
) -> tokio::sync::RwLockReadGuard<'a, Registry> {
    spawn_discovery(reg, config);
    tokio::time::sleep(Duration::from_millis(500)).await;
    let guard = reg.read().await;
    assert!(guard.is_discovery_done());
    guard
}

/// Create a single-backend config pointing at `url` and its shared registry.
fn single_backend_config(name: &str, url: &str) -> (Config, SharedRegistry) {
    let config = Config::from_backends(vec![Backend::for_test(name, url)]);
    let reg = registry::new_shared(&config);
    (config, reg)
}

/// Build a mock Ollama backend that advertises `models` on /api/tags,
/// echoes on /api/chat, and catches everything else in a fallback.
async fn start_mock_backend(models: Vec<&str>) -> String {
    let tags_json = serde_json::json!({
        "models": models.iter().map(|m| serde_json::json!({"name": m})).collect::<Vec<_>>()
    })
    .to_string();

    let tags_handler = get({
        let tags = tags_json.clone();
        move || {
            let tags = tags.clone();
            async move { (StatusCode::OK, tags) }
        }
    });
    let chat_handler =
        post(
            |payload: Bytes| async move { (StatusCode::OK, format!("echoed: {}", payload.len())) },
        );
    let version_handler = get(|| async { (StatusCode::OK, r#"{"version":"0.9.0"}"#) });
    let catch_all = any(|uri: axum::extract::OriginalUri, body: Bytes| async move {
        (
            StatusCode::OK,
            format!("fallback: {} {}", uri.0.path(), body.len()),
        )
    });

    let app = Router::new()
        .route("/api/tags", tags_handler)
        .route("/api/chat", chat_handler)
        .route("/api/version", version_handler)
        .fallback(catch_all);

    spawn_test_server(app).await
}

/// POST `{}` to `url`, assert the response status, and return the body text.
async fn post_and_expect(client: &Client, url: &str, expected: StatusCode) -> String {
    let resp = client.post(url).body("{}").send().await.unwrap();
    assert_eq!(
        resp.status(),
        expected,
        "POST {url} returned {}",
        resp.status()
    );
    resp.text().await.unwrap()
}

/// Spin up the `build_routed_paths_only_router` behind a test server and
/// return `(base_url, client)` ready for requests.
async fn serve_routed_paths_router() -> (String, Client) {
    let base = spawn_test_server(build_routed_paths_only_router()).await;
    (base, Client::new())
}

/// Assert a `TokenStore` is enabled but rejects every token.
async fn assert_enabled_but_rejects_all(store: &TokenStore) {
    assert!(store.is_enabled());
    assert!(!store.validate("anything").await);
}

/// Write `content` to a temp "tokens" file and return a `TokenStore` backed
/// by it.  The returned `TempDir` keeps the file alive for the test scope.
fn token_store_with_content(content: &str) -> (tempfile::TempDir, TokenStore) {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("tokens");
    std::fs::write(&file, content).unwrap();
    let store = TokenStore::new(Some(file.to_str().unwrap()));
    (dir, store)
}

// ─── Discovery + routing tests ──────────────────────────────────────────────

#[tokio::test]
async fn model_routing_to_correct_backend() {
    let cuda_url = start_mock_backend(vec!["fixt/home-3b-v3:latest"]).await;
    let rocm_url = start_mock_backend(vec!["glm-4.7-flash:latest"]).await;

    let backends = vec![
        Backend::for_test("cuda", &cuda_url),
        Backend::for_test("rocm", &rocm_url),
    ];
    let config = Config::from_backends(backends);
    let reg = registry::new_shared(&config);
    let r = run_discovery_to_completion(&reg, &config).await;

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
    let (_config, reg) = single_backend_config("test", "http://127.0.0.1:1");
    let r = reg.read().await;
    assert!(!r.is_discovery_done());
}

#[tokio::test]
async fn discovery_marks_unreachable_backend_down() {
    let (config, reg) = single_backend_config("dead", "http://127.0.0.1:1");
    let r = run_discovery_to_completion(&reg, &config).await;
    assert!(r.any_healthy().is_none());
    assert!(r.available_model_names().is_empty());
}

// ─── TokenStore tests ───────────────────────────────────────────────────────

#[tokio::test]
async fn token_store_validates_correctly() {
    let (_dir, store) = token_store_with_content("token-abc\ntoken-def\n# comment\n\n");

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
    let (dir, store) = token_store_with_content("old-token\n");
    assert!(store.validate("old-token").await);
    assert!(!store.validate("new-token").await);

    std::fs::write(dir.path().join("tokens"), "new-token\n").unwrap();
    store.reload().await;

    assert!(!store.validate("old-token").await);
    assert!(store.validate("new-token").await);
}

#[tokio::test]
async fn token_store_empty_file_fails_closed() {
    let (_dir, store) = token_store_with_content("# only a comment\n\n");
    // Auth is enabled (path configured) but no valid tokens -> all rejected
    assert_enabled_but_rejects_all(&store).await;
}

#[tokio::test]
async fn token_store_missing_file_fails_closed() {
    let store = TokenStore::new(Some("/nonexistent/path/tokens"));
    // Path configured but file missing -> auth enabled, all rejected
    assert_enabled_but_rejects_all(&store).await;
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
    ROUTED_PATHS
        .iter()
        .fold(Router::new(), |acc, entry| {
            let path = entry.path;
            acc.route(
                path,
                post(move || async move { (StatusCode::OK, format!("routed: {path}")) }),
            )
        })
        .fallback(any(|| async { StatusCode::NOT_FOUND }))
}

/// Paths that should NOT be in ROUTED_PATHS and must 404.
const NON_ROUTED_PATHS: &[&str] = &[
    "/some/future/path",
    "/v1/audio/transcriptions", // OpenAI Whisper — not in ROUTED_PATHS
    "/api/version",             // valid Ollama endpoint, not in our model_route set
    "/",
];

#[tokio::test]
async fn every_routed_path_actually_routes_through_axum() {
    let (base, client) = serve_routed_paths_router().await;

    for entry in ROUTED_PATHS {
        let url = format!("{base}{}", entry.path);
        let body = post_and_expect(&client, &url, StatusCode::OK).await;
        assert_eq!(body, format!("routed: {}", entry.path));
    }
}

#[tokio::test]
async fn paths_not_in_routed_paths_get_404() {
    // Sanity: the fallback wired above must catch anything not declared
    // in ROUTED_PATHS. If a future change makes the router permissive
    // (e.g. wildcard match that swallows everything), this fails.
    let (base, client) = serve_routed_paths_router().await;

    for unknown in NON_ROUTED_PATHS {
        let url = format!("{base}{unknown}");
        post_and_expect(&client, &url, StatusCode::NOT_FOUND).await;
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

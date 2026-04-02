use std::time::Duration;

use axum::body::Bytes;
use axum::http::StatusCode;
use axum::routing::{any, get, post};
use axum::Router;
use tokio::net::TcpListener;

use ollama_router::auth::TokenStore;
use ollama_router::config::{Backend, Config};
use ollama_router::registry;

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
            post(|body: Bytes| async move {
                (StatusCode::OK, format!("echoed: {}", body.len()))
            }),
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

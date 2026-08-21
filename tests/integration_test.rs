use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use axum::Router;
use axum::body::Bytes;
use axum::http::StatusCode;
use axum::routing::{any, get, post};
use reqwest::Client;
use serde_json::json;
use tokio::net::TcpListener;

use ollama_router::auth::TokenStore;
use ollama_router::config::Config;
use ollama_router::handler::{self, AppState};
use ollama_router::heartbeat::HeartbeatConfig;
use ollama_router::metrics::Metrics;
use ollama_router::policy::FileConfig;
use ollama_router::registry::{self, Registry, SharedRegistry};
use ollama_router::routes::{ROUTED_PATHS, default_stream_for_path};

// ─── Test helpers ────────────────────────────────────────────────────────────

async fn spawn_test_server(app: Router) -> String {
    spawn_test_server_abortable(app).await.0
}

/// Like `spawn_test_server`, but also returns the serve task's handle so a
/// test can abort it — dropping the listener and refusing later connects,
/// which simulates a backend dying between discovery cycles.
async fn spawn_test_server_abortable(app: Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{addr}"), handle)
}

fn spawn_discovery(reg: &SharedRegistry, config: &Config, metrics: &Arc<Metrics>) {
    tokio::spawn({
        let reg = reg.clone();
        let config = config.clone();
        let metrics = Arc::clone(metrics);
        let client = config.discovery_client().expect("discovery client");
        async move { registry::discovery_loop(reg, config, metrics, client).await }
    });
}

/// Sleep for discovery to complete, assert it finished, and return the
/// read guard so callers can inspect the registry without a second lock.
async fn run_discovery_to_completion<'a>(
    reg: &'a SharedRegistry,
    config: &Config,
) -> tokio::sync::RwLockReadGuard<'a, Registry> {
    spawn_discovery(reg, config, &Arc::new(Metrics::new()));
    tokio::time::sleep(Duration::from_millis(500)).await;
    let guard = reg.read().await;
    assert!(guard.is_discovery_done());
    guard
}

/// Render a `router.toml` naming `backends` in declaration order — each
/// publishing everything — followed by `extra` (fallback and alias tables).
/// `[fallbacks]` / `[aliases]` are appended when `extra` does not open them:
/// both are required tables, so an absent one is a rejection, not an
/// implicit empty.
fn policy_doc(backends: &[(&str, &str)], extra: &str) -> String {
    let mut doc = String::new();
    for (name, url) in backends {
        doc.push_str(&format!(
            "[[backends]]\nname = \"{name}\"\nurl = \"{url}\"\nallow = [\"*\"]\n\n"
        ));
    }
    if !extra.contains("[fallbacks]") {
        doc.push_str("[fallbacks]\n");
    }
    if !extra.contains("[aliases") {
        doc.push_str("[aliases]\n");
    }
    doc.push_str(extra);
    doc
}

/// Write a policy file into `dir` and return a `Config` pointing at it.
/// Rewriting the same path is how the live-reload tests edit the policy.
fn policy_config(dir: &tempfile::TempDir, backends: &[(&str, &str)], extra: &str) -> Config {
    let path = policy_path(dir);
    std::fs::write(&path, policy_doc(backends, extra)).unwrap();
    Config {
        config_path: path,
        ..Config::default()
    }
}

fn policy_path(dir: &tempfile::TempDir) -> String {
    dir.path().join("router.toml").to_str().unwrap().to_string()
}

/// Build a registry from `config`'s policy file the way `main` does.
fn new_registry(config: &Config) -> SharedRegistry {
    registry::new_shared(FileConfig::load(&config.config_path).expect("valid test policy"))
}

/// The common case: one temp dir, one policy file, one registry.
fn policy_setup(
    backends: &[(&str, &str)],
    extra: &str,
) -> (tempfile::TempDir, Config, SharedRegistry) {
    let dir = tempfile::tempdir().unwrap();
    let config = policy_config(&dir, backends, extra);
    let reg = new_registry(&config);
    (dir, config, reg)
}

/// A controllable mock backend: configurable /api/chat status, a hit
/// counter on /api/chat, and an abort handle to kill it mid-test.
struct MockBackend {
    url: String,
    handle: tokio::task::JoinHandle<()>,
    chat_hits: Arc<AtomicUsize>,
}

impl MockBackend {
    fn chat_hits(&self) -> usize {
        self.chat_hits.load(Ordering::SeqCst)
    }
}

/// Build the mock-backend router shared by `start_mock_backend` and
/// `start_counting_backend`: advertises `models` on /api/tags, answers
/// /api/chat with `chat_status` (counting hits), catch-all for the rest.
fn mock_backend_router(
    models: Vec<&str>,
    chat_status: StatusCode,
    chat_hits: Arc<AtomicUsize>,
) -> Router {
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
    let chat_handler = post(move |payload: Bytes| async move {
        chat_hits.fetch_add(1, Ordering::SeqCst);
        (chat_status, format!("echoed: {}", payload.len()))
    });
    let version_handler = get(|| async { (StatusCode::OK, r#"{"version":"0.9.0"}"#) });
    let catch_all = any(|uri: axum::extract::OriginalUri, body: Bytes| async move {
        (
            StatusCode::OK,
            format!("fallback: {} {}", uri.0.path(), body.len()),
        )
    });

    Router::new()
        .route("/api/tags", tags_handler)
        .route("/api/chat", chat_handler)
        .route("/api/version", version_handler)
        .fallback(catch_all)
}

/// Start a mock backend whose /api/chat answers with `chat_status`.
async fn start_counting_backend(models: Vec<&str>, chat_status: StatusCode) -> MockBackend {
    let chat_hits = Arc::new(AtomicUsize::new(0));
    let (url, handle) =
        spawn_test_server_abortable(mock_backend_router(models, chat_status, chat_hits.clone()))
            .await;
    MockBackend {
        url,
        handle,
        chat_hits,
    }
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

    let (_dir, config, reg) = policy_setup(&[("cuda", &cuda_url), ("rocm", &rocm_url)], "");
    let r = run_discovery_to_completion(&reg, &config).await;

    let cuda_id = r.lookup("fixt/home-3b-v3:latest").unwrap();
    assert_eq!(r.backend(cuda_id).expect("live id").name, "cuda");

    let rocm_id = r.lookup("glm-4.7-flash:latest").unwrap();
    assert_eq!(r.backend(rocm_id).expect("live id").name, "rocm");

    let prefix_id = r.lookup("glm-4.7-flash").unwrap();
    assert_eq!(r.backend(prefix_id).expect("live id").name, "rocm");

    assert!(r.lookup("nonexistent").is_none());
}

#[tokio::test]
async fn health_before_discovery_is_not_ready() {
    let (_dir, _config, reg) = policy_setup(&[("test", "http://127.0.0.1:1")], "");
    let r = reg.read().await;
    assert!(!r.is_discovery_done());
}

#[tokio::test]
async fn discovery_marks_unreachable_backend_down() {
    let (_dir, config, reg) = policy_setup(&[("dead", "http://127.0.0.1:1")], "");
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

// ─── Alias / priority-chain tests ───────────────────────────────────────────

/// Build an `AppState` around `reg` with a short preflight timeout, and
/// return it. Tests keep the state to inspect `state.metrics` afterwards.
fn app_state(reg: &SharedRegistry) -> AppState {
    AppState {
        registry: reg.clone(),
        metrics: Arc::new(Metrics::new()),
        client: Arc::new(Client::new()),
        token_store: Arc::new(TokenStore::new(None)),
        heartbeat: HeartbeatConfig::from_secs(15, 2, 30),
        escalation_rules: Arc::new(Vec::new()),
    }
}

/// Serve the production model-routing surface (every ROUTED_PATHS entry →
/// `model_route`, plus the listing GETs) behind a real TCP port.
async fn spawn_router(state: AppState) -> String {
    let mut router = Router::new();
    for entry in ROUTED_PATHS {
        router = router.route(entry.path, post(handler::model_route));
    }
    let router = router
        .route("/api/tags", get(handler::tags_route))
        .route("/v1/models", get(handler::v1_models_route))
        .route("/v1/models/{model_id}", get(handler::v1_model_route))
        .with_state(state);
    spawn_test_server(router).await
}

/// An `[aliases.<name>]` table with the given `backend/model` chain.
fn alias_toml(name: &str, chain: &[&str]) -> String {
    let chain = chain
        .iter()
        .map(|c| format!("\"{c}\""))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[aliases.{name}]\nchain = [{chain}]\n")
}

/// POST an /api/chat request for `model` (non-streaming unless told
/// otherwise) and return the response.
async fn post_chat(client: &Client, base: &str, model: &str, stream: bool) -> reqwest::Response {
    client
        .post(format!("{base}/api/chat"))
        .json(&json!({"model": model, "stream": stream, "messages": []}))
        .send()
        .await
        .unwrap()
}

#[tokio::test]
async fn alias_chain_advances_past_rate_limited_candidate() {
    let b1 = start_counting_backend(vec!["m:latest"], StatusCode::TOO_MANY_REQUESTS).await;
    let b2 = start_counting_backend(vec!["m:latest"], StatusCode::OK).await;
    let b3 = start_counting_backend(vec!["m:latest"], StatusCode::OK).await;

    let (_dir, config, reg) = policy_setup(
        &[("p1", &b1.url), ("p2", &b2.url), ("p3", &b3.url)],
        &alias_toml("ali", &["p1/m:latest", "p2/m:latest", "p3/m:latest"]),
    );
    let state = app_state(&reg);
    let base = spawn_router(state.clone()).await;
    drop(run_discovery_to_completion(&reg, &config).await);

    let client = Client::new();
    let resp = post_chat(&client, &base, "ali", false).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.text().await.unwrap();
    assert!(body.starts_with("echoed:"), "{body}");

    // First candidate was tried and rate-limited; second served; third idle.
    assert_eq!(b1.chat_hits(), 1);
    assert_eq!(b2.chat_hits(), 1);
    assert_eq!(b3.chat_hits(), 0);

    let text = state.metrics.encode().unwrap();
    assert!(
        text.contains(r#"alias="ali""#) && text.contains(r#"reason="rate_limited""#),
        "expected a rate_limited chain_advance sample:\n{text}"
    );
    // `to` names the candidate backend that was moved past.
    assert!(text.contains(r#"to="p1""#), "{text}");
    // The committed request is recorded under concrete model+backend labels.
    assert!(
        text.contains(r#"backend="p2""#) && text.contains(r#"model="m:latest""#),
        "{text}"
    );
}

#[tokio::test]
async fn alias_chain_advances_on_connect_error_without_discovery_wait() {
    let b1 = start_counting_backend(vec!["m:latest"], StatusCode::OK).await;
    let b2 = start_counting_backend(vec!["m:latest"], StatusCode::OK).await;

    let (_dir, config, reg) = policy_setup(
        &[("p1", &b1.url), ("p2", &b2.url)],
        &alias_toml("ali", &["p1/m:latest", "p2/m:latest"]),
    );
    let state = app_state(&reg);
    let base = spawn_router(state.clone()).await;
    drop(run_discovery_to_completion(&reg, &config).await);

    // Kill p1 AFTER discovery marked it healthy: the registry still says
    // reachable, so the chain must discover the death via the connect error
    // — not wait a discovery cycle.
    b1.handle.abort();
    // Give the runtime a moment to actually drop the listener.
    tokio::time::sleep(Duration::from_millis(50)).await;

    let client = Client::new();
    let resp = post_chat(&client, &base, "ali", false).await;
    assert_eq!(resp.status(), StatusCode::OK);

    assert_eq!(b2.chat_hits(), 1);
    let text = state.metrics.encode().unwrap();
    assert!(
        text.contains(r#"reason="connect""#),
        "expected a connect chain_advance sample:\n{text}"
    );
}

#[tokio::test]
async fn alias_chain_exhausted_relays_last_failure() {
    let b1 = start_counting_backend(vec!["m:latest"], StatusCode::INTERNAL_SERVER_ERROR).await;
    let b2 = start_counting_backend(vec!["m:latest"], StatusCode::INTERNAL_SERVER_ERROR).await;

    let (_dir, config, reg) = policy_setup(
        &[("p1", &b1.url), ("p2", &b2.url)],
        &alias_toml("ali", &["p1/m:latest", "p2/m:latest"]),
    );
    let state = app_state(&reg);
    let base = spawn_router(state.clone()).await;
    drop(run_discovery_to_completion(&reg, &config).await);

    let client = Client::new();
    let resp = post_chat(&client, &base, "ali", false).await;
    // The last upstream failure is relayed verbatim.
    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(b1.chat_hits(), 1);
    assert_eq!(b2.chat_hits(), 1);

    let text = state.metrics.encode().unwrap();
    assert!(
        text.contains("ollama_router_chain_exhausted_total") && text.contains(r#"alias="ali""#),
        "expected a chain_exhausted sample:\n{text}"
    );
    assert!(text.contains(r#"reason="upstream_5xx""#), "{text}");
    // Even an exhausted chain answered the client, so the request shows up
    // in requests_total: alias as the model, the backend whose failure was
    // relayed, and the relayed status.
    assert!(
        text.contains(r#"ollama_router_requests_total{model="ali",backend="p2",status_code="500""#),
        "exhausted chain must still record request metrics:\n{text}"
    );
}

#[tokio::test]
async fn alias_streaming_request_commits_on_first_healthy_candidate() {
    let b1 = start_counting_backend(vec!["m:latest"], StatusCode::OK).await;
    let b2 = start_counting_backend(vec!["m:latest"], StatusCode::OK).await;

    let (_dir, config, reg) = policy_setup(
        &[("p1", &b1.url), ("p2", &b2.url)],
        &alias_toml("ali", &["p1/m:latest", "p2/m:latest"]),
    );
    let state = app_state(&reg);
    let base = spawn_router(state.clone()).await;
    drop(run_discovery_to_completion(&reg, &config).await);

    let client = Client::new();
    let resp = post_chat(&client, &base, "ali", true).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.text().await.unwrap();
    assert!(body.starts_with("echoed:"), "{body}");

    assert_eq!(b1.chat_hits(), 1);
    assert_eq!(b2.chat_hits(), 0);
    let text = state.metrics.encode().unwrap();
    assert!(
        !text.contains("ollama_router_chain_advance_total{"),
        "streaming happy path must record zero advances:\n{text}"
    );
}

#[tokio::test]
async fn local_alias_never_reaches_external_backend() {
    // The external backend is healthy and even serves the same model; the
    // local candidate is hard-down. A `local` alias must exhaust rather
    // than let a single request escape to the external backend.
    let ext = start_counting_backend(vec!["m:latest"], StatusCode::OK).await;

    let dir = tempfile::tempdir().unwrap();
    let path = policy_path(&dir);
    std::fs::write(
        &path,
        format!(
            "[[backends]]\nname = \"loc\"\nurl = \"http://127.0.0.1:1\"\nallow = [\"*\"]\n\n\
             [[backends]]\nname = \"ext\"\nurl = \"{}\"\nexternal = true\nallow = [\"*\"]\n\n\
             [fallbacks]\n[aliases.secret]\nlocal = true\nchain = [\"loc/m:latest\"]\n",
            ext.url
        ),
    )
    .unwrap();
    let config = Config {
        config_path: path,
        ..Config::default()
    };
    let reg = new_registry(&config);
    let state = app_state(&reg);
    let base = spawn_router(state.clone()).await;
    drop(run_discovery_to_completion(&reg, &config).await);

    let client = Client::new();
    let resp = post_chat(&client, &base, "secret", false).await;
    assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

    // The external backend saw ZERO chat requests.
    assert_eq!(ext.chat_hits(), 0);
    let text = state.metrics.encode().unwrap();
    assert!(text.contains(r#"alias="secret""#), "{text}");
    assert!(
        text.contains("ollama_router_chain_exhausted_total"),
        "{text}"
    );
}

#[tokio::test]
async fn aliases_listed_in_v1_models_and_api_tags() {
    let b1 = start_counting_backend(vec!["m:latest"], StatusCode::OK).await;

    let (_dir, config, reg) =
        policy_setup(&[("p1", &b1.url)], &alias_toml("ali", &["p1/m:latest"]));
    let state = app_state(&reg);
    let base = spawn_router(state.clone()).await;
    drop(run_discovery_to_completion(&reg, &config).await);

    let client = Client::new();

    let v1: serde_json::Value = client
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let data = v1["data"].as_array().unwrap();
    let alias_entry = data
        .iter()
        .find(|m| m["id"] == "ali")
        .expect("alias listed in /v1/models");
    assert_eq!(alias_entry["owned_by"], "router-alias");
    // Concrete models keep listing (":latest" stripped for exact-id match).
    assert!(data.iter().any(|m| m["id"] == "m"), "{data:?}");

    let tags: serde_json::Value = client
        .get(format!("{base}/api/tags"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let models = tags["models"].as_array().unwrap();
    let alias_tag = models
        .iter()
        .find(|m| m["name"] == "ali")
        .expect("alias listed in /api/tags");
    // Shape-compatible with sanitised concrete entries (pydantic clients).
    assert_eq!(alias_tag["model"], "ali");
    assert!(alias_tag["modified_at"].is_string());
    assert!(alias_tag["size"].is_u64());
    assert!(models.iter().any(|m| m["name"] == "m:latest"), "{models:?}");

    // Anything /v1/models lists must be retrievable by id — the alias too.
    let one: serde_json::Value = client
        .get(format!("{base}/v1/models/ali"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(one["id"], "ali");
    assert_eq!(one["owned_by"], "router-alias");
    // Concrete retrieval still works alongside.
    let concrete = client
        .get(format!("{base}/v1/models/m"))
        .send()
        .await
        .unwrap();
    assert_eq!(concrete.status(), StatusCode::OK);

    // The unknown-model 404 hint includes the alias.
    let resp = post_chat(&client, &base, "nope", false).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let body: serde_json::Value = resp.json().await.unwrap();
    let available = body["available_models"].as_array().unwrap();
    assert!(available.iter().any(|m| m == "ali"), "{available:?}");
}

#[tokio::test]
async fn concrete_models_and_fallbacks_route_unchanged_alongside_aliases() {
    let b1 = start_counting_backend(vec!["m:latest"], StatusCode::OK).await;

    // Fallback map for a non-alias name: still works, and now lives in the
    // same document as the alias it sits beside.
    let (_dir, config, reg) = policy_setup(
        &[("p1", &b1.url)],
        &format!(
            "[fallbacks]\n\"ghost\" = \"m:latest\"\n{}",
            alias_toml("ali", &["p1/m:latest"])
        ),
    );
    let state = app_state(&reg);
    let base = spawn_router(state.clone()).await;
    drop(run_discovery_to_completion(&reg, &config).await);

    let client = Client::new();

    // Concrete model routes exactly as before.
    let resp = post_chat(&client, &base, "m:latest", false).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(resp.text().await.unwrap().starts_with("echoed:"));

    // Single-hop fallback still applies to non-alias names.
    let resp = post_chat(&client, &base, "ghost", false).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(resp.text().await.unwrap().starts_with("echoed:"));

    assert_eq!(b1.chat_hits(), 2);
    let text = state.metrics.encode().unwrap();
    // Neither request walked an alias chain.
    assert!(
        !text.contains("ollama_router_chain_advance_total{"),
        "{text}"
    );
    assert!(
        text.contains(r#"ollama_router_fallbacks_total{from="ghost",to="m:latest"}"#),
        "{text}"
    );
}

#[tokio::test]
async fn alias_chain_advances_on_auth_error() {
    // Expired/missing external API key is the steady-state failure of the
    // hosted backends this feature targets: it must fail over, not relay.
    let b1 = start_counting_backend(vec!["m:latest"], StatusCode::UNAUTHORIZED).await;
    let b2 = start_counting_backend(vec!["m:latest"], StatusCode::OK).await;

    let (_dir, config, reg) = policy_setup(
        &[("p1", &b1.url), ("p2", &b2.url)],
        &alias_toml("ali", &["p1/m:latest", "p2/m:latest"]),
    );
    let state = app_state(&reg);
    let base = spawn_router(state.clone()).await;
    drop(run_discovery_to_completion(&reg, &config).await);

    let client = Client::new();
    let resp = post_chat(&client, &base, "ali", false).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(b1.chat_hits(), 1);
    assert_eq!(b2.chat_hits(), 1);

    let text = state.metrics.encode().unwrap();
    assert!(
        text.contains(r#"reason="auth""#),
        "expected an auth chain_advance sample:\n{text}"
    );
}

#[tokio::test]
async fn alias_chain_skips_llama_swap_missing_model_without_send() {
    // A reachable llama-swap that does NOT advertise the candidate model
    // must be passed over without an upstream attempt (its list is fresh —
    // absence is positive evidence), even for a streaming request. Before
    // the heartbeat gate, streaming would have committed 200 on the dead-end
    // candidate and never reached the healthy one.
    let swap = start_counting_backend(vec!["other:latest"], StatusCode::OK).await;
    let b2 = start_counting_backend(vec!["m:latest"], StatusCode::OK).await;

    let (_dir, config, reg) = policy_setup(
        &[("llama-swap", &swap.url), ("p2", &b2.url)],
        &alias_toml("ali", &["llama-swap/m:latest", "p2/m:latest"]),
    );
    let state = app_state(&reg);
    let base = spawn_router(state.clone()).await;
    drop(run_discovery_to_completion(&reg, &config).await);

    let client = Client::new();
    let resp = post_chat(&client, &base, "ali", true).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(resp.text().await.unwrap().starts_with("echoed:"));

    // llama-swap received ZERO chat requests; the second candidate served.
    assert_eq!(swap.chat_hits(), 0);
    assert_eq!(b2.chat_hits(), 1);
    let text = state.metrics.encode().unwrap();
    assert!(
        text.contains(r#"to="llama-swap""#) && text.contains(r#"reason="model_missing""#),
        "expected a model_missing advance for llama-swap:\n{text}"
    );
}

#[tokio::test]
async fn alias_resolves_latest_normalized_name() {
    // Ollama clients normalise a bare model name to `name:latest`; the
    // alias must resolve under that spelling and keep the canonical name
    // in metric labels.
    let b1 = start_counting_backend(vec!["m:latest"], StatusCode::OK).await;

    let (_dir, config, reg) =
        policy_setup(&[("p1", &b1.url)], &alias_toml("ali", &["p1/m:latest"]));
    let state = app_state(&reg);
    let base = spawn_router(state.clone()).await;
    drop(run_discovery_to_completion(&reg, &config).await);

    let client = Client::new();
    let resp = post_chat(&client, &base, "ali:latest", false).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(resp.text().await.unwrap().starts_with("echoed:"));
    assert_eq!(b1.chat_hits(), 1);
}

#[tokio::test]
async fn fallback_to_alias_stand_in_enters_chain() {
    // A fallback stand-in naming an alias is the natural operator move once
    // chains exist: the request enters the chain path instead of being
    // dropped as "target not in registry".
    let b1 = start_counting_backend(vec!["m:latest"], StatusCode::OK).await;

    let (_dir, config, reg) = policy_setup(
        &[("p1", &b1.url)],
        &format!(
            "[fallbacks]\n\"ghost\" = \"ali\"\n{}",
            alias_toml("ali", &["p1/m:latest"])
        ),
    );
    let state = app_state(&reg);
    let base = spawn_router(state.clone()).await;
    drop(run_discovery_to_completion(&reg, &config).await);

    let client = Client::new();
    let resp = post_chat(&client, &base, "ghost", false).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(resp.text().await.unwrap().starts_with("echoed:"));
    assert_eq!(b1.chat_hits(), 1);

    let text = state.metrics.encode().unwrap();
    assert!(
        text.contains(r#"ollama_router_fallbacks_total{from="ghost",to="ali"}"#),
        "{text}"
    );
    // The committed candidate is recorded under its concrete labels.
    assert!(text.contains(r#"backend="p1""#), "{text}");
}

#[tokio::test]
async fn alias_attempts_all_candidates_when_none_reachable() {
    // Startup race / registry-blind window: discovery marks the backend
    // down (its listing endpoints fail) but the backend actually answers
    // /api/chat. With no reachable candidate the walk must attempt them
    // all in order rather than 502 blindly.
    let chat_hits = Arc::new(AtomicUsize::new(0));
    let hits = chat_hits.clone();
    let app = Router::new()
        .route(
            "/api/chat",
            post(move |payload: Bytes| async move {
                hits.fetch_add(1, Ordering::SeqCst);
                (StatusCode::OK, format!("echoed: {}", payload.len()))
            }),
        )
        // /api/tags, /v1/models, /api/ps all fail: discovery never marks
        // this backend healthy.
        .fallback(any(|| async {
            (StatusCode::INTERNAL_SERVER_ERROR, "boom")
        }));
    let url = spawn_test_server(app).await;

    let (_dir, config, reg) = policy_setup(&[("p1", &url)], &alias_toml("ali", &["p1/m:latest"]));
    let state = app_state(&reg);
    let base = spawn_router(state.clone()).await;
    drop(run_discovery_to_completion(&reg, &config).await);
    assert!(reg.read().await.any_healthy().is_none());

    let client = Client::new();
    let resp = post_chat(&client, &base, "ali", false).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(resp.text().await.unwrap().starts_with("echoed:"));
    assert_eq!(chat_hits.load(Ordering::SeqCst), 1);

    let text = state.metrics.encode().unwrap();
    // The candidate was attempted, not skipped: no "unreachable" advance.
    assert!(!text.contains(r#"reason="unreachable""#), "{text}");
}

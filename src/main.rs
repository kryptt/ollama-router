use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::{OriginalUri, Path, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::Router;
use serde_json::json;
use tokio::net::TcpListener;
use tower_http::trace::TraceLayer;
use tracing::info;

use ollama_router::auth::TokenStore;
use ollama_router::config::Config;
use ollama_router::metrics::{self, Metrics};
use ollama_router::models;
use ollama_router::proxy;
use ollama_router::registry::{self, SharedRegistry};
use ollama_router::response::json_status;
use ollama_router::spill;

#[derive(Clone)]
struct AppState {
    registry: SharedRegistry,
    metrics: Arc<Metrics>,
    client: Arc<reqwest::Client>,
    token_store: Arc<TokenStore>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ollama_router=info".into()),
        )
        .init();

    let config = Config::from_env().expect("invalid configuration");

    info!(
        backends = config.backends.len(),
        discovery_interval = config.discovery_interval_secs,
        connect_timeout = config.connect_timeout_secs,
        request_timeout = config.request_timeout_secs,
        public_addr = %config.public_addr,
        internal_addr = %config.internal_addr,
        "starting ollama-router"
    );

    let registry = registry::new_shared(&config);
    let metrics = Arc::new(Metrics::new());
    let token_store = Arc::new(TokenStore::new(config.tokens_file.as_deref()));
    let client = Arc::new(
        reqwest::Client::builder()
            .pool_max_idle_per_host(10)
            .connect_timeout(Duration::from_secs(config.connect_timeout_secs))
            .timeout(Duration::from_secs(config.request_timeout_secs))
            .build()
            .expect("failed to build HTTP client"),
    );

    let state = AppState {
        registry: Arc::clone(&registry),
        metrics: Arc::clone(&metrics),
        client,
        token_store: Arc::clone(&token_store),
    };

    tokio::spawn({
        let registry = Arc::clone(&registry);
        let config = config.clone();
        async move { registry::discovery_loop(registry, config).await }
    });

    if config.tokens_file.is_some() {
        tokio::spawn({
            let ts = Arc::clone(&token_store);
            async move {
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                    ts.reload().await;
                }
            }
        });
    }

    let public_router = Router::new()
        .route("/api/chat", post(model_route))
        .route("/api/generate", post(model_route))
        .route("/api/embed", post(model_route))
        .route("/api/embeddings", post(model_route))
        .route("/api/show", post(model_route))
        .route("/v1/chat/completions", post(model_route))
        .route("/v1/completions", post(model_route))
        .route("/v1/embeddings", post(model_route))
        .route("/v1/messages", post(model_route))
        .route("/api/pull", post(blocked_route))
        .route("/api/delete", post(blocked_route))
        .route("/api/copy", post(blocked_route))
        .route("/api/create", post(blocked_route))
        .route("/api/push", post(blocked_route))
        .route("/api/tags", get(tags_route))
        .route("/v1/models", get(v1_models_route))
        .route("/v1/models/{model_id}", get(v1_model_route))
        .fallback(any(passthrough_route))
        .layer(TraceLayer::new_for_http())
        .with_state(state.clone());

    let internal_router = Router::new()
        .route("/health", get(health_route))
        .route("/status", get(status_route))
        .route("/metrics", get(metrics_route))
        .route("/auth", any(auth_route))
        .with_state(state);

    let public_listener = TcpListener::bind(config.public_addr)
        .await
        .expect("failed to bind public port");
    let internal_listener = TcpListener::bind(config.internal_addr)
        .await
        .expect("failed to bind internal port");

    info!(
        "listening on {} (public) and {} (internal)",
        config.public_addr, config.internal_addr
    );

    let (r1, r2) = tokio::join!(
        axum::serve(public_listener, public_router).into_future(),
        axum::serve(internal_listener, internal_router).into_future(),
    );
    if let Err(e) = r1 {
        tracing::error!(error = %e, "public server error");
    }
    if let Err(e) = r2 {
        tracing::error!(error = %e, "internal server error");
    }
}

// ---------------------------------------------------------------------------
// Public handlers
// ---------------------------------------------------------------------------

async fn model_route(
    State(state): State<AppState>,
    method: Method,
    OriginalUri(uri): OriginalUri,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let spilled = match spill::spill_and_detect(body).await {
        Ok(Some(s)) => s,
        Ok(None) => return proxy::bad_request("request body must contain a non-empty 'model' field"),
        Err(e) => {
            tracing::warn!(error = %e, "failed to spill request body");
            return proxy::bad_gateway("failed to read request body");
        }
    };

    let reg = state.registry.read().await;
    let backend_id = match reg.lookup(&spilled.model) {
        Some(id) => id,
        None => {
            state.metrics.unknown_model_requests.inc();
            let available = reg.available_model_names();
            return proxy::model_not_found(&spilled.model, &available);
        }
    };
    let view = reg.backend(backend_id);
    let backend_url = view.url.to_string();
    let backend_name = view.name.to_string();
    drop(reg);

    let start = Instant::now();

    let response = proxy::execute(proxy::ProxyRequest {
        client: &state.client,
        backend_url: &backend_url,
        path: uri.path(),
        query: uri.query(),
        method: method.clone(),
        headers: &headers,
        body: spilled.body,
    })
    .await;

    let duration = start.elapsed().as_secs_f64();
    let status_code = response.status().as_u16();

    state
        .metrics
        .requests_total
        .get_or_create(&metrics::RequestLabels {
            model: spilled.model.clone(),
            backend: backend_name.clone(),
            status_code,
            method: method.to_string(),
            stream: spilled.stream,
        })
        .inc();

    state
        .metrics
        .request_duration
        .get_or_create(&metrics::DurationLabels {
            model: spilled.model,
            backend: backend_name,
            stream: spilled.stream,
        })
        .observe(duration);

    response
}

async fn blocked_route(State(state): State<AppState>, OriginalUri(uri): OriginalUri) -> Response {
    state
        .metrics
        .blocked_requests
        .get_or_create(&metrics::RouteLabels {
            route: uri.path().to_string(),
        })
        .inc();

    proxy::method_not_allowed(uri.path())
}

async fn tags_route(State(state): State<AppState>) -> Response {
    let reg = state.registry.read().await;
    models::api_tags_response(&reg)
}

async fn v1_models_route(State(state): State<AppState>) -> Response {
    let reg = state.registry.read().await;
    models::v1_models_response(&reg)
}

async fn v1_model_route(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Response {
    let reg = state.registry.read().await;
    models::v1_model_response(&reg, &model_id)
}

async fn passthrough_route(
    State(state): State<AppState>,
    method: Method,
    OriginalUri(uri): OriginalUri,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let reg = state.registry.read().await;
    let backend_id = match reg.any_healthy() {
        Some(id) => id,
        None => return proxy::bad_gateway("no healthy backends available"),
    };
    let backend_url = reg.backend(backend_id).url.to_string();
    drop(reg);

    proxy::execute(proxy::ProxyRequest {
        client: &state.client,
        backend_url: &backend_url,
        path: uri.path(),
        query: uri.query(),
        method,
        headers: &headers,
        body,
    })
    .await
}

// ---------------------------------------------------------------------------
// Internal handlers
// ---------------------------------------------------------------------------

async fn health_route(State(state): State<AppState>) -> Response {
    let reg = state.registry.read().await;

    if !reg.is_discovery_done() {
        return json_status(
            StatusCode::SERVICE_UNAVAILABLE,
            json!({"status": "unhealthy", "reason": "awaiting first discovery"}),
        );
    }

    let any_reachable = reg.all_backends().any(|b| b.healthy || b.in_grace_period);

    if any_reachable {
        json_status(StatusCode::OK, json!({"status": "ok"}))
    } else {
        json_status(
            StatusCode::SERVICE_UNAVAILABLE,
            json!({"status": "unhealthy", "reason": "all backends unreachable"}),
        )
    }
}

async fn status_route(State(state): State<AppState>) -> Response {
    let reg = state.registry.read().await;

    let backends: Vec<serde_json::Value> = reg
        .all_backends()
        .map(|b| {
            json!({
                "name": b.name,
                "url": b.url,
                "healthy": b.healthy,
                "models": b.models.iter().map(|m| &m.name).collect::<Vec<_>>(),
                "in_grace_period": b.in_grace_period,
            })
        })
        .collect();

    let model_count = reg.available_model_names().len();

    json_status(
        StatusCode::OK,
        json!({
            "backends": backends,
            "discovery_done": reg.is_discovery_done(),
            "model_count": model_count,
        }),
    )
}

async fn metrics_route(State(state): State<AppState>) -> Response {
    match state.metrics.encode() {
        Ok(buf) => (
            StatusCode::OK,
            [(
                axum::http::header::CONTENT_TYPE,
                HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
            )],
            buf,
        )
            .into_response(),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

async fn auth_route(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !state.token_store.is_enabled() {
        return StatusCode::OK.into_response();
    }

    let token = match extract_token(&headers) {
        Some(t) => t,
        None => {
            return json_status(
                StatusCode::UNAUTHORIZED,
                json!({"error": "missing or invalid Authorization header or api-key"}),
            );
        }
    };

    if state.token_store.validate(token).await {
        StatusCode::OK.into_response()
    } else {
        json_status(StatusCode::UNAUTHORIZED, json!({"error": "invalid token"}))
    }
}

/// Extract token from `Authorization: Bearer <token>`, `api-key: <token>`,
/// or `x-api-key: <token>` header.
fn extract_token(headers: &HeaderMap) -> Option<&str> {
    // Try Authorization: Bearer first
    if let Some(value) = headers.get("authorization").and_then(|v| v.to_str().ok()) {
        if value.len() > 7 && value[..7].eq_ignore_ascii_case("bearer ") {
            return Some(&value[7..]);
        }
    }
    // Fall back to api-key (Qdrant) or x-api-key (Anthropic) headers
    headers
        .get("api-key")
        .or_else(|| headers.get("x-api-key"))
        .and_then(|v| v.to_str().ok())
}

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::routing::{any, get, post};
use tokio::net::TcpListener;
use tower_http::trace::TraceLayer;
use tracing::info;

use ollama_router::auth::TokenStore;
use ollama_router::config::Config;
use ollama_router::handler::{self, AppState};
use ollama_router::heartbeat::HeartbeatConfig;
use ollama_router::metrics::Metrics;
use ollama_router::registry;
use ollama_router::routes::ROUTED_PATHS;

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
        loading_heartbeat = config.loading_heartbeat_secs,
        preflight_timeout = config.preflight_timeout_secs,
        loading_max_wait = config.loading_max_wait_secs,
        public_addr = %config.public_addr,
        internal_addr = %config.internal_addr,
        "starting ollama-router"
    );

    let registry = registry::new_shared(&config);
    let metrics = Arc::new(Metrics::new());
    metrics.start_time_seconds.set(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0),
    );
    let token_store = Arc::new(TokenStore::new(config.tokens_file.as_deref()));
    let client = Arc::new(
        reqwest::Client::builder()
            .pool_max_idle_per_host(10)
            .connect_timeout(Duration::from_secs(config.connect_timeout_secs))
            .timeout(Duration::from_secs(config.request_timeout_secs))
            .build()
            .expect("failed to build HTTP client"),
    );

    let heartbeat_cfg = HeartbeatConfig::from_secs(
        config.loading_heartbeat_secs,
        config.preflight_timeout_secs,
        config.loading_max_wait_secs,
    );

    let escalation_rules = Arc::new(config.escalation_rules.clone());
    if !escalation_rules.is_empty() {
        info!(rules = escalation_rules.len(), "model escalation enabled");
        // Per-rule details at debug-level: useful exactly once during
        // initial rollout, then noise. The aggregate count above is the
        // info-level signal that escalation is on at all.
        for r in escalation_rules.iter() {
            tracing::debug!(
                from = %r.from_model,
                threshold_tokens = r.max_input_tokens,
                to = %r.to_model,
                "escalation rule",
            );
        }
    }

    let state = AppState {
        registry: Arc::clone(&registry),
        metrics: Arc::clone(&metrics),
        client,
        token_store: Arc::clone(&token_store),
        heartbeat: heartbeat_cfg,
        escalation_rules,
    };

    tokio::spawn({
        let registry = Arc::clone(&registry);
        let config = config.clone();
        async move { registry::discovery_loop(registry, config).await }
    });

    // One shutdown notify shared by both servers and the background tasks. A
    // single OS-signal listener fans the trigger out so each drains/exits
    // cleanly. Without this, a SIGTERM during a Fleet rolling update drops
    // every heartbeat-wrapped stream as a TCP RST mid-response.
    let shutdown = Arc::new(tokio::sync::Notify::new());
    {
        let shutdown = Arc::clone(&shutdown);
        tokio::spawn(async move {
            wait_for_shutdown_signal().await;
            info!("shutdown signal received, draining in-flight requests");
            shutdown.notify_waiters();
        });
    }

    if config.tokens_file.is_some() {
        tokio::spawn({
            let ts = Arc::clone(&token_store);
            let shutdown = Arc::clone(&shutdown);
            async move {
                loop {
                    tokio::select! {
                        _ = tokio::time::sleep(std::time::Duration::from_secs(60)) => {
                            ts.reload().await;
                        }
                        _ = shutdown.notified() => break,
                    }
                }
            }
        });
    }

    // Register every routed path from ROUTED_PATHS (single source of
    // truth shared with `model_route`'s default_stream lookup). Adding
    // a new entry to that const automatically routes it here.
    let mut public_router = Router::new();
    for entry in ROUTED_PATHS {
        public_router = public_router.route(entry.path, post(handler::model_route));
    }
    let public_router = public_router
        .route("/api/pull", post(handler::blocked_route))
        .route("/api/delete", post(handler::blocked_route))
        .route("/api/copy", post(handler::blocked_route))
        .route("/api/create", post(handler::blocked_route))
        .route("/api/push", post(handler::blocked_route))
        .route("/api/tags", get(handler::tags_route))
        .route("/api/ps", get(handler::api_ps_route))
        .route("/v1/models", get(handler::v1_models_route))
        .route("/v1/models/{model_id}", get(handler::v1_model_route))
        .fallback(any(handler::passthrough_route))
        .layer(TraceLayer::new_for_http())
        .with_state(state.clone());

    let internal_router = Router::new()
        .route("/health", get(handler::health_route))
        .route("/status", get(handler::status_route))
        .route("/metrics", get(handler::metrics_route))
        .route("/auth", any(handler::auth_route))
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

    let public_shutdown = Arc::clone(&shutdown);
    let internal_shutdown = Arc::clone(&shutdown);

    let (r1, r2) = tokio::join!(
        axum::serve(public_listener, public_router)
            .with_graceful_shutdown(async move { public_shutdown.notified().await })
            .into_future(),
        axum::serve(internal_listener, internal_router)
            .with_graceful_shutdown(async move { internal_shutdown.notified().await })
            .into_future(),
    );
    if let Err(e) = r1 {
        tracing::error!(error = %e, "public server error");
    }
    if let Err(e) = r2 {
        tracing::error!(error = %e, "internal server error");
    }
    info!("server stopped");
}

/// Resolve when the process receives SIGTERM, SIGINT, or Ctrl-C.
/// SIGTERM is what Kubernetes sends on pod termination; Ctrl-C is the
/// developer-affordance equivalent. The future returns once *either*
/// fires — both halves are racing in `tokio::select!`.
async fn wait_for_shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("install Ctrl-C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

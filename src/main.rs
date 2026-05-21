use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::Body;
use axum::extract::{OriginalUri, Path, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use futures_util::StreamExt;
use serde_json::json;
use tokio::net::TcpListener;
use tower_http::trace::TraceLayer;
use tracing::info;

use ollama_router::auth::TokenStore;
use ollama_router::config::{Config, EscalationRule};
use ollama_router::heartbeat::{self, HeartbeatConfig, StreamProtocol};
use ollama_router::metrics::{self, Metrics};
use ollama_router::models;
use ollama_router::proxy;
use ollama_router::registry::{self, BackendProtocol, SharedRegistry};
use ollama_router::response::json_status;
use ollama_router::routes::{self, ROUTED_PATHS};
use ollama_router::spill;
use ollama_router::translate;

#[derive(Clone)]
struct AppState {
    registry: SharedRegistry,
    metrics: Arc<Metrics>,
    client: Arc<reqwest::Client>,
    token_store: Arc<TokenStore>,
    heartbeat: HeartbeatConfig,
    escalation_rules: Arc<Vec<EscalationRule>>,
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
        loading_heartbeat = config.loading_heartbeat_secs,
        preflight_timeout = config.preflight_timeout_secs,
        loading_max_wait = config.loading_max_wait_secs,
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

    // Register every routed path from ROUTED_PATHS (single source of
    // truth shared with `model_route`'s default_stream lookup). Adding
    // a new entry to that const automatically routes it here.
    let mut public_router = Router::new();
    for entry in ROUTED_PATHS {
        public_router = public_router.route(entry.path, post(model_route));
    }
    let public_router = public_router
        .route("/api/pull", post(blocked_route))
        .route("/api/delete", post(blocked_route))
        .route("/api/copy", post(blocked_route))
        .route("/api/create", post(blocked_route))
        .route("/api/push", post(blocked_route))
        .route("/api/tags", get(tags_route))
        .route("/api/ps", get(api_ps_route))
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

    // One shutdown notify shared by both servers. A single OS-signal
    // listener task fans the trigger out to both `with_graceful_shutdown`
    // futures so they each drain in-flight requests before returning.
    // Without this, a SIGTERM during a Fleet rolling update drops every
    // heartbeat-wrapped stream as a TCP RST mid-response.
    let shutdown = Arc::new(tokio::sync::Notify::new());
    {
        let shutdown = Arc::clone(&shutdown);
        tokio::spawn(async move {
            wait_for_shutdown_signal().await;
            info!("shutdown signal received, draining in-flight requests");
            shutdown.notify_waiters();
        });
    }

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
    let default_stream = routes::default_stream_for_path(uri.path());

    let mut spilled = match spill::spill_and_detect(body, default_stream).await {
        Ok(Some(s)) => s,
        Ok(None) => {
            return proxy::bad_request("request body must contain a non-empty 'model' field");
        }
        Err(e) => {
            tracing::warn!(error = %e, "failed to spill request body");
            return proxy::bad_gateway("failed to read request body");
        }
    };

    // Escalate to a higher-context sibling when input is too big for the
    // requested model's per-slot budget.
    let original_model = spilled.model.clone();
    if !state.escalation_rules.is_empty() {
        match estimate_input_tokens(&headers) {
            Some(estimated_tokens) => {
                if let Some(escalated) =
                    apply_escalation(&spilled.model, estimated_tokens, &state.escalation_rules)
                {
                    spilled.model = escalated;
                }
            }
            None => {
                // No Content-Length (chunked transfer, HTTP/2 streaming
                // upload). We can't estimate input size cheaply, so we
                // can't make an escalation decision — count and continue
                // with the originally requested model.
                state
                    .metrics
                    .escalations_skipped
                    .get_or_create(&metrics::EscalationSkipLabels {
                        reason: "no_content_length".to_string(),
                    })
                    .inc();
            }
        }
    }

    let reg = state.registry.read().await;
    // If escalation rewrote the model but the target isn't (yet) visible
    // in the registry, fall back to the original. Otherwise we'd 404 the
    // client with a name they never sent — particularly bad during the
    // 60-second discovery warmup or when an operator typos a `to_model`.
    let lookup_model = if spilled.model != original_model && reg.lookup(&spilled.model).is_none() {
        tracing::warn!(
            requested = %original_model,
            escalation_target = %spilled.model,
            "escalation target not in registry; falling back to original model",
        );
        state
            .metrics
            .escalations_skipped
            .get_or_create(&metrics::EscalationSkipLabels {
                reason: "target_not_found".to_string(),
            })
            .inc();
        spilled.model = original_model.clone();
        &spilled.model
    } else {
        &spilled.model
    };

    if *lookup_model != original_model {
        // Escalation kept its rewrite: record it as a successful
        // escalation event in metrics + a single info log.
        state
            .metrics
            .escalations
            .get_or_create(&metrics::EscalationLabels {
                from: original_model.clone(),
                to: lookup_model.clone(),
            })
            .inc();
        tracing::info!(
            from = %original_model,
            to = %lookup_model,
            "escalating model due to estimated input size",
        );
    }

    let backend_id = match reg.lookup(lookup_model) {
        Some(id) => id,
        None => {
            state.metrics.unknown_model_requests.inc();
            let available = reg.available_model_names();
            // Emit the requested model name as a structured field so
            // operators can grep Loki for which clients are sending
            // bogus model names. The metric itself stays label-free
            // to avoid cardinality blow-up if a misbehaving client
            // sprays unique names.
            tracing::warn!(
                model = %lookup_model,
                available = ?available,
                "unknown model requested",
            );
            return proxy::model_not_found(lookup_model, &available);
        }
    };
    let view = reg.backend(backend_id);
    let backend_url = view.url.to_string();
    let backend_name = view.name.to_string();
    let backend_protocol = view.protocol;
    drop(reg);

    // Protocol translation: client speaks Ollama-native /api/chat but the
    // chosen backend only speaks OpenAI /v1/*. We rewrite the upstream URL
    // and the request body in-flight, then reshape the response back to
    // Ollama on the way out. Scope is /api/chat only for this iteration;
    // /api/generate, /api/embed, etc. follow later.
    let needs_translation =
        uri.path() == "/api/chat" && backend_protocol == BackendProtocol::OpenAi;

    if needs_translation {
        tracing::info!(
            model = %spilled.model,
            backend = %backend_name,
            "translating /api/chat → /v1/chat/completions"
        );
        state.metrics.protocol_translations.inc();

        // Buffer the request body and translate it.
        let body_bytes = match collect_body_to_bytes(spilled.body).await {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(error = %e, "failed to buffer request body for translation");
                return proxy::bad_gateway("failed to read request body");
            }
        };
        let translated = match translate::ollama_chat_to_openai_request(&body_bytes) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(error = %e, "failed to translate /api/chat body");
                return proxy::bad_request("body is not valid JSON");
            }
        };
        spilled.body = Body::from(translated);
    }

    let start = Instant::now();

    // Decide between hot-path proxy and heartbeat-wrapped proxy.
    //
    // Heartbeat path only makes sense for streaming chat/completion requests
    // against a backend that isn't currently hosting the model in memory.
    // Anything else (non-streaming, embeddings, tags, etc.) just proxies
    // directly — those requests don't suffer from idle-chunk timeouts.
    let use_heartbeat = spilled.stream
        && StreamProtocol::from_path(uri.path()).is_some()
        && !preflight_model_loaded(&state, &backend_url, &backend_name, &spilled.model).await;

    let upstream_path: Option<&str> = if needs_translation {
        Some("/v1/chat/completions")
    } else {
        None
    };

    let response = if use_heartbeat {
        let protocol = StreamProtocol::from_path(uri.path()).expect("checked above");
        tracing::info!(
            model = %spilled.model,
            backend = %backend_name,
            path = uri.path(),
            "heartbeat path engaged (model not hot)"
        );
        let translator: Option<heartbeat::BodyTranslator> = if needs_translation {
            let model = spilled.model.clone();
            Some(Box::new(move |s| {
                translate::translate_streaming_response(s, model)
            }))
        } else {
            None
        };
        heartbeat::execute(heartbeat::HeartbeatRequest {
            client: &state.client,
            backend_url: &backend_url,
            path: uri.path(),
            override_path: upstream_path,
            query: uri.query(),
            method: method.clone(),
            headers: &headers,
            body: spilled.body,
            protocol,
            model: spilled.model.clone(),
            config: state.heartbeat,
            translate: translator,
        })
        .await
    } else {
        let raw = proxy::execute(proxy::ProxyRequest {
            client: &state.client,
            backend_url: &backend_url,
            path: uri.path(),
            override_path: upstream_path,
            query: uri.query(),
            method: method.clone(),
            headers: &headers,
            body: spilled.body,
        })
        .await;

        if needs_translation {
            translate_proxy_response(raw, spilled.stream, spilled.model.clone()).await
        } else {
            raw
        }
    };

    let duration = start.elapsed().as_secs_f64();
    let status_code = response.status().as_u16();

    // The `stream` label means "this request will actually stream the
    // response back" — i.e. spilled.stream AND the path's protocol
    // supports streaming. For /api/embed and /api/show this is always
    // false even when the body's stream flag (defaulted) reads true,
    // because those endpoints return a single JSON regardless.
    let actually_streams = spilled.stream && StreamProtocol::from_path(uri.path()).is_some();

    state
        .metrics
        .requests_total
        .get_or_create(&metrics::RequestLabels {
            model: spilled.model.clone(),
            backend: backend_name.clone(),
            status_code,
            method: method.to_string(),
            stream: actually_streams,
        })
        .inc();

    state
        .metrics
        .request_duration
        .get_or_create(&metrics::DurationLabels {
            model: spilled.model,
            backend: backend_name,
            stream: actually_streams,
        })
        .observe(duration);

    response
}

/// Return `true` if `/api/ps` on `backend_url` confirms the model is loaded,
/// or if the probe fails in a way that suggests we should *not* take the
/// heartbeat path (preflight error → let normal proxy surface the problem).
///
/// The semantics are asymmetric on purpose: the heartbeat path commits to a
/// `200 OK` response before upstream replies, so we only take it when we
/// have positive evidence that the model is still loading.
async fn preflight_model_loaded(
    state: &AppState,
    backend_url: &str,
    backend_name: &str,
    model: &str,
) -> bool {
    let kind = models::classify(backend_name);
    match heartbeat::preflight_is_loaded(
        &state.client,
        backend_url,
        kind,
        model,
        state.heartbeat.preflight_timeout,
    )
    .await
    {
        Ok(loaded) => loaded,
        Err(e) => {
            tracing::debug!(
                error = %e,
                model = %model,
                backend_url = %backend_url,
                "preflight failed — skipping heartbeat path"
            );
            // Treat preflight failure as "loaded" so we don't engage the
            // heartbeat path. Normal proxy will surface the real error.
            true
        }
    }
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

async fn api_ps_route(State(state): State<AppState>) -> Response {
    models::api_ps_response(&state.registry, &state.client).await
}

async fn v1_models_route(State(state): State<AppState>) -> Response {
    let reg = state.registry.read().await;
    models::v1_models_response(&reg)
}

async fn v1_model_route(State(state): State<AppState>, Path(model_id): Path<String>) -> Response {
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
        override_path: None,
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

    let model_count = reg.reachable_models().len();

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
    if let Some(value) = headers.get("authorization").and_then(|v| v.to_str().ok())
        && value.len() > 7
        && value[..7].eq_ignore_ascii_case("bearer ")
    {
        return Some(&value[7..]);
    }
    // Fall back to api-key (Qdrant) or x-api-key (Anthropic) headers
    headers
        .get("api-key")
        .or_else(|| headers.get("x-api-key"))
        .and_then(|v| v.to_str().ok())
}

/// Collect an `axum::body::Body` into a single `Vec<u8>`. Used to buffer
/// /api/chat request bodies before protocol translation rewrites them.
async fn collect_body_to_bytes(body: Body) -> Result<Vec<u8>, axum::Error> {
    use http_body_util::BodyExt;
    let bytes = body.collect().await?.to_bytes();
    Ok(bytes.to_vec())
}

/// Wrap a `proxy::execute` response in protocol translation. For a
/// non-streaming request, buffer the whole body and convert via
/// `translate::openai_chat_to_ollama_response`; for a streaming request,
/// wrap the body stream with `translate::translate_streaming_response`.
/// Status code and a hand-picked subset of headers are preserved.
async fn translate_proxy_response(resp: Response, streaming: bool, model: String) -> Response {
    let status = resp.status();
    // Only successful responses get translated. Upstream error bodies are
    // already a JSON `{"error": ...}` shape that Ollama clients tolerate;
    // translating them would risk papering over the failure.
    if !status.is_success() {
        return resp;
    }

    if streaming {
        let stream = resp
            .into_body()
            .into_data_stream()
            .map(|r| r.map_err(std::io::Error::other));
        let translated = translate::translate_streaming_response(stream, model);
        let mut new_resp = Response::new(Body::from_stream(translated));
        *new_resp.status_mut() = status;
        new_resp.headers_mut().insert(
            axum::http::header::CONTENT_TYPE,
            HeaderValue::from_static("application/x-ndjson"),
        );
        new_resp
    } else {
        use http_body_util::BodyExt;
        let bytes = match resp.into_body().collect().await {
            Ok(c) => c.to_bytes(),
            Err(e) => {
                tracing::warn!(error = %e, "failed to buffer translated response body");
                return proxy::bad_gateway("failed to read upstream response");
            }
        };
        match translate::openai_chat_to_ollama_response(&bytes, &model) {
            Ok(out) => {
                let mut new_resp = Response::new(Body::from(out));
                *new_resp.status_mut() = status;
                new_resp.headers_mut().insert(
                    axum::http::header::CONTENT_TYPE,
                    HeaderValue::from_static("application/json"),
                );
                new_resp
            }
            Err(e) => {
                tracing::warn!(error = %e, "failed to translate non-streaming response");
                proxy::bad_gateway("failed to translate upstream response")
            }
        }
    }
}

/// Cheap upper-bound estimate of input tokens from the request body size.
///
/// Reads `Content-Length`. Returns `None` when the header is missing or
/// unparseable (chunked transfer encoding, malformed value) — escalation
/// then conservatively does not fire and the request hits its originally
/// requested model.
///
/// Heuristic: bytes / 3. For English-dominated JSON chat bodies a typical
/// token is ~3.5-4 bytes, so dividing by 3 over-estimates slightly — the
/// right direction since the cost of escalating sooner is a swap, while
/// the cost of escalating too late is a hard 400 from the upstream.
fn estimate_input_tokens(headers: &HeaderMap) -> Option<usize> {
    let raw = headers.get(axum::http::header::CONTENT_LENGTH)?;
    let s = raw.to_str().ok()?;
    let bytes: usize = s.parse().ok()?;
    Some(bytes / 3)
}

/// Walk the escalation chain starting from `model`. Each iteration finds
/// the first rule (in declaration order) with `from_model == current`
/// whose threshold is exceeded by `estimated_tokens`; the rule's
/// `to_model` becomes the new `current`. Loop terminates when no further
/// rule applies. Returns `None` when no escalation happens — that's the
/// only meaning of `None`, so callers don't need to compare against the
/// original model name.
///
/// First-match-wins: if multiple rules share a `from_model`, the rule
/// declared first in `OLLAMA_ROUTER_ESCALATE` is picked. Operators
/// configuring multi-rule chains should order them by ascending
/// `max_input_tokens` within the same source.
///
/// Cycle safety: hops are bounded by `rules.len()` so even a cyclic
/// config (a→b→c→a) cannot loop forever. A logged warning fires when
/// the bound is exhausted so an operator can spot the misconfiguration.
fn apply_escalation(
    model: &str,
    estimated_tokens: usize,
    rules: &[EscalationRule],
) -> Option<String> {
    let mut current = model.to_string();
    let mut hops = 0;
    while hops < rules.len() {
        match rules
            .iter()
            .find(|r| r.from_model == current && estimated_tokens > r.max_input_tokens)
        {
            Some(rule) => {
                current = rule.to_model.clone();
                hops += 1;
            }
            None => {
                // Reached a fixed point — no rule wants to rewrite
                // `current` any further. Normal exit.
                if current == model {
                    return None;
                }
                return Some(current);
            }
        }
    }
    // Hop budget exhausted without reaching a fixed point. This means
    // we either traversed a cycle (a→b→…→a→…) or a chain longer than
    // rules.len() (impossible if every rule rewrites at most once per
    // walk, which they do — so this is a cycle in practice).
    tracing::warn!(
        original_model = %model,
        terminal_model = %current,
        hops,
        "escalation hop budget exhausted; rule config likely contains a cycle",
    );
    if current == model {
        None
    } else {
        Some(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(from: &str, max: usize, to: &str) -> EscalationRule {
        EscalationRule {
            from_model: from.to_string(),
            max_input_tokens: max,
            to_model: to.to_string(),
        }
    }

    #[test]
    fn escalation_no_rules_is_noop() {
        assert_eq!(apply_escalation("qwen3.6-medium", 100_000, &[]), None);
    }

    #[test]
    fn escalation_below_threshold_is_noop() {
        let rules = vec![rule("qwen3.6-medium", 35_000, "qwen3.6-high")];
        assert_eq!(apply_escalation("qwen3.6-medium", 1_000, &rules), None);
    }

    #[test]
    fn escalation_above_threshold_rewrites() {
        let rules = vec![rule("qwen3.6-medium", 35_000, "qwen3.6-high")];
        assert_eq!(
            apply_escalation("qwen3.6-medium", 50_000, &rules),
            Some("qwen3.6-high".to_string())
        );
    }

    #[test]
    fn escalation_chains_through_multiple_hops() {
        // medium → high → ultra in one decision.
        let rules = vec![
            rule("qwen3.6-medium", 35_000, "qwen3.6-high"),
            rule("qwen3.6-high", 120_000, "qwen3.6-ultra"),
        ];
        assert_eq!(
            apply_escalation("qwen3.6-medium", 200_000, &rules),
            Some("qwen3.6-ultra".to_string())
        );
    }

    #[test]
    fn escalation_chain_stops_when_target_threshold_not_exceeded() {
        let rules = vec![
            rule("qwen3.6-medium", 35_000, "qwen3.6-high"),
            rule("qwen3.6-high", 120_000, "qwen3.6-ultra"),
        ];
        assert_eq!(
            apply_escalation("qwen3.6-medium", 50_000, &rules),
            Some("qwen3.6-high".to_string())
        );
    }

    #[test]
    fn escalation_ignores_unrelated_models() {
        let rules = vec![rule("qwen3.6-medium", 35_000, "qwen3.6-high")];
        assert_eq!(apply_escalation("gpt-oss:latest", 100_000, &rules), None);
    }

    #[test]
    fn escalation_exact_threshold_does_not_fire() {
        // Comparator is strict-greater: estimated_tokens > threshold. At
        // exactly the threshold we don't escalate. Pinning this so a
        // future refactor to >= can't slip past the test suite.
        let rules = vec![rule("qwen3.6-medium", 35_000, "qwen3.6-high")];
        assert_eq!(
            apply_escalation("qwen3.6-medium", 35_000, &rules),
            None,
            "at exactly the threshold, no escalation",
        );
        assert_eq!(
            apply_escalation("qwen3.6-medium", 35_001, &rules),
            Some("qwen3.6-high".to_string()),
            "one token over the threshold, escalation fires",
        );
    }

    #[test]
    fn escalation_first_matching_rule_wins() {
        // When multiple rules share a from_model, the first in declaration
        // order is selected. Operators relying on this should sort their
        // rules by ascending threshold within a source.
        let rules = vec![
            rule("a", 10, "b"), // declared first
            rule("a", 5, "c"),  // declared second, lower threshold
        ];
        assert_eq!(
            apply_escalation("a", 100, &rules),
            Some("b".to_string()),
            "declaration order wins, not lowest-threshold-wins",
        );
    }

    #[test]
    fn escalation_symmetric_cycle_returns_none() {
        // a→b→c→a, all thresholds exceeded. After rules.len() hops the
        // walker is back at "a" (the original model). The function MUST
        // return None — escalation contributes nothing for a symmetric
        // cycle. Without this assertion the cycle behaviour is undefined
        // and a future refactor could silently route every request
        // through the wrong model.
        let rules = vec![rule("a", 10, "b"), rule("b", 10, "c"), rule("c", 10, "a")];
        assert_eq!(apply_escalation("a", 100, &rules), None);
    }

    #[test]
    fn escalation_asymmetric_cycle_terminates_at_hop_budget() {
        // a→b, b↔c. Walker traverses a→b→c→b, hits hop budget at
        // hops==rules.len()=3, current=="b", model=="a". This is a
        // *deterministic* but non-fixed-point answer — pin it so any
        // future change to the iteration order is loud.
        let rules = vec![rule("a", 10, "b"), rule("b", 10, "c"), rule("c", 10, "b")];
        assert_eq!(apply_escalation("a", 100, &rules), Some("b".to_string()));
    }

    // ── estimate_input_tokens unit tests ─────────────────────────────────

    fn header_with(name: &str, value: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(
            axum::http::HeaderName::from_bytes(name.as_bytes()).unwrap(),
            HeaderValue::from_str(value).unwrap(),
        );
        h
    }

    #[test]
    fn estimate_input_tokens_missing_header_is_none() {
        let h = HeaderMap::new();
        assert_eq!(estimate_input_tokens(&h), None);
    }

    #[test]
    fn estimate_input_tokens_non_numeric_value_is_none() {
        let h = header_with("content-length", "not-a-number");
        assert_eq!(estimate_input_tokens(&h), None);
    }

    #[test]
    fn estimate_input_tokens_negative_value_is_none() {
        // Negative content length is nonsense; usize parser rejects it.
        let h = header_with("content-length", "-1");
        assert_eq!(estimate_input_tokens(&h), None);
    }

    #[test]
    fn estimate_input_tokens_valid_header_is_bytes_over_three() {
        let h = header_with("content-length", "105000");
        assert_eq!(estimate_input_tokens(&h), Some(35_000));
    }

    #[test]
    fn estimate_input_tokens_zero_is_zero() {
        // Edge case: an empty body has Content-Length: 0 → 0 tokens.
        // Not an error, just a trivial small body.
        let h = header_with("content-length", "0");
        assert_eq!(estimate_input_tokens(&h), Some(0));
    }

    #[test]
    fn estimate_input_tokens_small_body_rounds_down() {
        // 5 bytes / 3 = 1 (integer division). This is fine — small
        // bodies are always under any sensible escalation threshold.
        let h = header_with("content-length", "5");
        assert_eq!(estimate_input_tokens(&h), Some(1));
    }
}

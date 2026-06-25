//! HTTP request handlers and shared application state.
//!
//! Extracted from `main.rs` so the routing/proxy pipeline is a library module:
//! callable from integration tests without a TCP round-trip, and a clean
//! injection point for the Unit 3 retry/circuit-breaker wrapper around
//! `proxy::execute`.

use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::extract::{OriginalUri, Path, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use serde_json::json;

use crate::auth::TokenStore;
use crate::config::EscalationRule;
use crate::heartbeat::{self, HeartbeatConfig, StreamProtocol};
use crate::metrics::{self, Metrics};
use crate::models;
use crate::proxy;
use crate::registry::{BackendProtocol, SharedRegistry};
use crate::response::json_status;
use crate::routes;
use crate::spill;
use crate::translate;

/// Shared, cheaply-clonable application state handed to every handler.
#[derive(Clone)]
pub struct AppState {
    pub registry: SharedRegistry,
    pub metrics: Arc<Metrics>,
    pub client: Arc<reqwest::Client>,
    pub token_store: Arc<TokenStore>,
    pub heartbeat: HeartbeatConfig,
    pub escalation_rules: Arc<Vec<EscalationRule>>,
}

#[tracing::instrument(
    name = "model_route",
    skip_all,
    fields(
        http.request.method = %method,
        url.path = %uri.path(),
        model = tracing::field::Empty,
        backend = tracing::field::Empty,
        http.response.status_code = tracing::field::Empty,
    )
)]
pub async fn model_route(
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

    let span = tracing::Span::current();
    span.record("model", spilled.model.as_str());
    span.record("backend", backend_name.as_str());

    // Protocol translation: client speaks Ollama-native /api/chat but the
    // chosen backend only speaks OpenAI /v1/*. We rewrite the upstream URL
    // and the request body in-flight, then reshape the response back to
    // Ollama on the way out. Scope is /api/chat only for this iteration;
    // /api/generate, /api/embed, etc. follow later.
    //
    // Triggered by EITHER:
    // (a) the discovery probe saw /v1/models but not /api/tags, so we
    //     know the backend has no Ollama API surface; OR
    // (b) the backend is named like a known OpenAI-only family
    //     (llama-swap, llama-edge). llama.cpp's llama-server includes an
    //     ollama-compat shim that answers /api/tags but its /api/chat
    //     hangs forever — discovery thinks it's Ollama-protocol but it
    //     can't actually handle the chat path.
    let backend_kind = models::classify(&backend_name);
    let needs_translation = uri.path() == "/api/chat"
        && (backend_protocol == BackendProtocol::OpenAi
            || matches!(
                backend_kind,
                models::BackendKind::LlamaSwap | models::BackendKind::AlwaysResident
            ));

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
    // Compute the path's streaming protocol once; the heartbeat decision and
    // the heartbeat branch both consume this single value, so the protocol is
    // never re-derived (and never needs an "infallible" unwrap downstream).
    let protocol = StreamProtocol::from_path(uri.path());
    let use_heartbeat = spilled.stream
        && protocol.is_some()
        && !preflight_model_loaded(&state, &backend_url, &backend_name, &spilled.model).await;

    let upstream_path: Option<&str> = if needs_translation {
        Some("/v1/chat/completions")
    } else {
        None
    };

    let response = if let (true, Some(protocol)) = (use_heartbeat, protocol) {
        state.metrics.heartbeat_engaged.inc();
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
        let raw = match proxy::execute(proxy::ProxyRequest {
            client: &state.client,
            backend_url: &backend_url,
            path: uri.path(),
            override_path: upstream_path,
            query: uri.query(),
            method: method.clone(),
            headers: &headers,
            body: spilled.body,
        })
        .await
        {
            Ok(r) => r,
            Err(e) => {
                state
                    .metrics
                    .upstream_errors
                    .get_or_create(&metrics::UpstreamErrorLabels {
                        kind: e.kind_str().to_string(),
                    })
                    .inc();
                e.into_response()
            }
        };

        if needs_translation {
            translate_proxy_response(raw, spilled.stream, spilled.model.clone()).await
        } else {
            raw
        }
    };

    let duration = start.elapsed().as_secs_f64();
    let status_code = response.status().as_u16();
    span.record("http.response.status_code", status_code);

    // The `stream` label means "this request will actually stream the
    // response back" — i.e. spilled.stream AND the path's protocol
    // supports streaming. For /api/embed and /api/show this is always
    // false even when the body's stream flag (defaulted) reads true,
    // because those endpoints return a single JSON regardless.
    let actually_streams = spilled.stream && protocol.is_some();

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

pub async fn blocked_route(
    State(state): State<AppState>,
    OriginalUri(uri): OriginalUri,
) -> Response {
    state
        .metrics
        .blocked_requests
        .get_or_create(&metrics::RouteLabels {
            route: uri.path().to_string(),
        })
        .inc();

    proxy::method_not_allowed(uri.path())
}

pub async fn tags_route(State(state): State<AppState>) -> Response {
    let reg = state.registry.read().await;
    models::api_tags_response(&reg)
}

pub async fn api_ps_route(State(state): State<AppState>) -> Response {
    models::api_ps_response(&state.registry, &state.client).await
}

pub async fn v1_models_route(State(state): State<AppState>) -> Response {
    let reg = state.registry.read().await;
    models::v1_models_response(&reg)
}

pub async fn v1_model_route(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Response {
    let reg = state.registry.read().await;
    models::v1_model_response(&reg, &model_id)
}

#[tracing::instrument(
    name = "passthrough_route",
    skip_all,
    fields(http.request.method = %method, url.path = %uri.path())
)]
pub async fn passthrough_route(
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

    match proxy::execute(proxy::ProxyRequest {
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
    {
        Ok(r) => r,
        Err(e) => {
            state
                .metrics
                .upstream_errors
                .get_or_create(&metrics::UpstreamErrorLabels {
                    kind: e.kind_str().to_string(),
                })
                .inc();
            e.into_response()
        }
    }
}

// ---------------------------------------------------------------------------
// Internal handlers
// ---------------------------------------------------------------------------

pub async fn health_route(State(state): State<AppState>) -> Response {
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

pub async fn status_route(State(state): State<AppState>) -> Response {
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

pub async fn metrics_route(State(state): State<AppState>) -> Response {
    // Refresh registry-derived gauges from the current snapshot so they're
    // fresh at scrape time without coupling the discovery loop to metrics.
    {
        let reg = state.registry.read().await;
        let reachable = reg
            .all_backends()
            .filter(|b| b.healthy || b.in_grace_period)
            .count();
        let healthy = reg.all_backends().filter(|b| b.healthy).count();
        let ready = reg.is_discovery_done() && reachable > 0;
        state.metrics.ready.set(ready as i64);
        state.metrics.backends_reachable.set(reachable as i64);
        state.metrics.backends_healthy.set(healthy as i64);
        for b in reg.all_backends() {
            state
                .metrics
                .backend_up
                .get_or_create(&metrics::BackendLabels {
                    backend: b.name.to_string(),
                })
                .set(b.healthy as i64);
        }
    }

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

pub async fn auth_route(State(state): State<AppState>, headers: HeaderMap) -> Response {
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

    /// Construct a minimal `AppState` for in-process handler tests — no TCP
    /// server, no real backends. Proves the I4 seam: handlers are callable
    /// directly now that they live in the library.
    fn test_state() -> AppState {
        use crate::config::{Backend, Config};
        let config = Config::from_backends(vec![Backend::for_test("b", "http://127.0.0.1:1")]);
        AppState {
            registry: crate::registry::new_shared(&config),
            metrics: Arc::new(Metrics::new()),
            client: Arc::new(reqwest::Client::new()),
            token_store: Arc::new(TokenStore::new(None)),
            heartbeat: HeartbeatConfig::from_secs(15, 10, 300),
            escalation_rules: Arc::new(Vec::new()),
        }
    }

    #[tokio::test]
    async fn health_route_reports_awaiting_first_discovery() {
        use http_body_util::BodyExt;
        // Before the first discovery cycle completes, readiness is 503.
        let resp = health_route(State(test_state())).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["reason"], "awaiting first discovery");
    }

    #[tokio::test]
    async fn metrics_route_exposes_self_health_gauges() {
        use http_body_util::BodyExt;
        let resp = metrics_route(State(test_state())).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(body.to_vec()).unwrap();
        // The self-health series are registered and refreshed at scrape time.
        for name in [
            "ollama_router_start_time_seconds",
            "ollama_router_ready",
            "ollama_router_backends_reachable",
            "ollama_router_backends_healthy",
            "ollama_router_upstream_errors",
            "ollama_router_heartbeat_engaged",
        ] {
            assert!(text.contains(name), "missing metric: {name}\n{text}");
        }
        // Pre-discovery, the router is not ready and nothing is reachable.
        assert!(text.contains("ollama_router_ready 0"), "{text}");
        assert!(
            text.contains("ollama_router_backends_reachable 0"),
            "{text}"
        );
    }

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

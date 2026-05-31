use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::Registry;

const DURATION_BUCKETS: &[f64] = &[
    0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 180.0, 300.0,
];

/// Prometheus metrics for the router. Immutable after construction;
/// individual counters/gauges are internally atomic.
pub struct Metrics {
    pub requests_total: Family<RequestLabels, Counter>,
    pub request_duration: Family<DurationLabels, Histogram>,
    pub blocked_requests: Family<RouteLabels, Counter>,
    pub unknown_model_requests: Counter,
    /// Successful long-turn escalations: a request originally for
    /// `from` was rewritten to `to` because the estimated input tokens
    /// exceeded `from`'s configured threshold and `to` is currently
    /// reachable.
    pub escalations: Family<EscalationLabels, Counter>,
    /// Escalations that were configured but not applied. `reason` is
    /// one of:
    /// - `no_content_length`: client used chunked transfer / HTTP/2
    ///   streaming, so we couldn't estimate input size.
    /// - `target_not_found`: escalation target was not in the registry
    ///   at lookup time (typo, decommissioned backend, or pre-discovery
    ///   startup window).
    pub escalations_skipped: Family<EscalationSkipLabels, Counter>,
    /// Requests that were translated between protocol dialects (e.g.
    /// `/api/chat` from a client → `/v1/chat/completions` to a backend
    /// that only speaks OpenAI). Label-free on purpose — cardinality.
    pub protocol_translations: Counter,

    // --- Self-health (set at startup / refreshed at scrape time) ---
    /// Process start time as a Unix timestamp (seconds). A sawtooth that
    /// resets on a short interval is the fingerprint of pod churn — the exact
    /// signal that would have surfaced the macvlan-watchdog kill loop.
    pub start_time_seconds: Gauge,
    /// 1 when the router is ready to serve (first discovery done AND at least
    /// one backend reachable), else 0. Refreshed on each `/metrics` scrape.
    pub ready: Gauge,
    /// Backends currently reachable (healthy or within grace). Refreshed on scrape.
    pub backends_reachable: Gauge,
    /// Backends currently healthy (strictly up). Refreshed on scrape.
    pub backends_healthy: Gauge,
    /// Per-backend up/down (1/0). Refreshed on scrape.
    pub backend_up: Family<BackendLabels, Gauge>,

    /// Transport-level upstream failures by kind (connect / timeout /
    /// transport) — i.e. cases where no response was obtained at all. Upstream
    /// 5xx responses are counted by `requests_total{status_code}` instead.
    pub upstream_errors: Family<UpstreamErrorLabels, Counter>,
    /// Times the cold-load heartbeat path was engaged (model not resident at
    /// request start). High rates here correlate with restart/warmup windows.
    pub heartbeat_engaged: Counter,
    registry: Registry,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    pub fn new() -> Self {
        let mut registry = Registry::default();

        let requests_total = Family::default();
        registry.register(
            "ollama_router_requests",
            "Total routed requests",
            requests_total.clone(),
        );

        fn make_duration_histogram() -> Histogram {
            Histogram::new(DURATION_BUCKETS.iter().copied())
        }
        let request_duration =
            Family::new_with_constructor(make_duration_histogram as fn() -> Histogram);
        registry.register(
            "ollama_router_request_duration_seconds",
            "Request duration",
            request_duration.clone(),
        );

        let blocked_requests = Family::default();
        registry.register(
            "ollama_router_blocked_requests",
            "Blocked management requests",
            blocked_requests.clone(),
        );

        let unknown_model_requests = Counter::default();
        registry.register(
            "ollama_router_unknown_model_requests",
            "Requests for models not found in any backend",
            unknown_model_requests.clone(),
        );

        let escalations = Family::default();
        registry.register(
            "ollama_router_escalations",
            "Long-turn escalation rewrites applied (per from→to pair)",
            escalations.clone(),
        );

        let escalations_skipped = Family::default();
        registry.register(
            "ollama_router_escalations_skipped",
            "Escalations skipped because preconditions weren't met",
            escalations_skipped.clone(),
        );

        let protocol_translations = Counter::default();
        registry.register(
            "ollama_router_protocol_translations",
            "Requests translated between protocol dialects (e.g. /api/chat → /v1/chat/completions)",
            protocol_translations.clone(),
        );

        let start_time_seconds = Gauge::default();
        registry.register(
            "ollama_router_start_time_seconds",
            "Process start time (Unix seconds); resets on restart",
            start_time_seconds.clone(),
        );

        let ready = Gauge::default();
        registry.register(
            "ollama_router_ready",
            "1 if discovery is done and at least one backend is reachable, else 0",
            ready.clone(),
        );

        let backends_reachable = Gauge::default();
        registry.register(
            "ollama_router_backends_reachable",
            "Backends currently reachable (healthy or within grace)",
            backends_reachable.clone(),
        );

        let backends_healthy = Gauge::default();
        registry.register(
            "ollama_router_backends_healthy",
            "Backends currently healthy (strictly up)",
            backends_healthy.clone(),
        );

        let backend_up = Family::default();
        registry.register(
            "ollama_router_backend_up",
            "Per-backend health (1 up, 0 down)",
            backend_up.clone(),
        );

        let upstream_errors = Family::default();
        registry.register(
            "ollama_router_upstream_errors",
            "Transport-level upstream failures by kind (no response obtained)",
            upstream_errors.clone(),
        );

        let heartbeat_engaged = Counter::default();
        registry.register(
            "ollama_router_heartbeat_engaged",
            "Times the cold-load heartbeat path was engaged",
            heartbeat_engaged.clone(),
        );

        Metrics {
            requests_total,
            request_duration,
            blocked_requests,
            unknown_model_requests,
            escalations,
            escalations_skipped,
            protocol_translations,
            start_time_seconds,
            ready,
            backends_reachable,
            backends_healthy,
            backend_up,
            upstream_errors,
            heartbeat_engaged,
            registry,
        }
    }

    /// Encode all metrics in Prometheus text exposition format.
    pub fn encode(&self) -> Result<String, std::fmt::Error> {
        let mut buf = String::new();
        prometheus_client::encoding::text::encode(&mut buf, &self.registry)?;
        Ok(buf)
    }
}

// --- Label types ---

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct RequestLabels {
    pub model: String,
    pub backend: String,
    pub status_code: u16,
    pub method: String,
    pub stream: bool,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct DurationLabels {
    pub model: String,
    pub backend: String,
    pub stream: bool,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct RouteLabels {
    pub route: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EscalationLabels {
    pub from: String,
    pub to: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EscalationSkipLabels {
    pub reason: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct BackendLabels {
    pub backend: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct UpstreamErrorLabels {
    pub kind: String,
}

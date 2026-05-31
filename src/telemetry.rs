//! OpenTelemetry → OTLP span export, for Tempo (via the cluster's Alloy
//! collector or Tempo's OTLP endpoint directly).
//!
//! Tracing is **opt-in**: if `OTEL_EXPORTER_OTLP_ENDPOINT` is unset the router
//! runs log-only (the `fmt` layer), so local/dev runs need no collector. When
//! set (e.g. `http://tempo.monitor.svc:4318` or a node-local Alloy), spans are
//! batch-exported over OTLP/HTTP, reusing the existing reqwest+rustls stack —
//! no gRPC/tonic transport at runtime.

use opentelemetry::KeyValue;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::TracerProvider;
use tracing_subscriber::Layer;
use tracing_subscriber::registry::LookupSpan;

/// True when tracing export is configured via the standard OTLP env var.
fn enabled() -> bool {
    std::env::var_os("OTEL_EXPORTER_OTLP_ENDPOINT").is_some()
}

/// Build the OTLP tracing layer plus the provider to hold for shutdown
/// flushing. Returns `None` when tracing is not configured or the exporter
/// fails to build (in which case the router still runs, log-only).
///
/// The endpoint and related knobs come from the standard `OTEL_*` env vars
/// (`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, …),
/// read by the exporter itself. `OTEL_SERVICE_NAME` overrides the service name.
pub fn otlp_layer<S>(
    version: &str,
) -> Option<(Box<dyn Layer<S> + Send + Sync + 'static>, TracerProvider)>
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a> + Send + Sync,
{
    if !enabled() {
        return None;
    }

    // W3C `traceparent` propagation, so inbound client context continues and
    // we can stitch router → backend spans.
    opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());

    let exporter = match opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .build()
    {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(error = %e, "OTLP exporter init failed; tracing disabled");
            return None;
        }
    };

    let service_name =
        std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "ollama-router".to_string());

    let provider = TracerProvider::builder()
        .with_batch_exporter(exporter, opentelemetry_sdk::runtime::Tokio)
        .with_resource(Resource::new(vec![
            KeyValue::new("service.name", service_name),
            KeyValue::new("service.version", version.to_string()),
        ]))
        .build();

    let tracer = provider.tracer("ollama-router");
    opentelemetry::global::set_tracer_provider(provider.clone());

    let layer = tracing_opentelemetry::layer().with_tracer(tracer);
    Some((Box::new(layer), provider))
}

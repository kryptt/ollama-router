# ollama-router

A Rust HTTP proxy that fronts one or more Ollama / OpenAI / Anthropic-
compatible inference endpoints behind a single OpenAI-compatible API.

- **Per-model dispatch.** Discovers the models on every configured backend
  (`/api/tags`, falling back to `/v1/models`) on an interval and routes each
  request to the backend hosting the requested model — exact name first, then
  `:`-tag prefix, first-writer-wins across the backend list. A configurable
  **grace period** keeps a briefly-unreachable backend's models routable
  through transient blips instead of 404-ing mid-incident.
- **Unified cluster views.** Aggregated `/v1/models`, `/api/tags`, and
  `/api/ps` present the whole fleet as one endpoint. `/api/ps` synthesises
  `context_length` (parsed from the llama-server command line) and `expires_at`
  for non-Ollama backends so clients see a consistent Ollama-shaped response.
- **Ollama ↔ OpenAI protocol translation.** A client can speak Ollama-native
  `/api/chat` to a backend that only exposes OpenAI `/v1/chat/completions`: the
  request body, streaming SSE responses, and non-streaming responses are
  reshaped in-flight in both directions, transparently to the client. (Scope
  today: `/api/chat`; other paths proxy unchanged.)
- **Cold-load heartbeat.** Preflights the backend (Ollama `/api/ps`,
  llama-swap `/running`, or "always-resident" backends) and, while a model is
  still loading, injects protocol-correct keepalive bytes — empty NDJSON chunks
  for Ollama, SSE comments for OpenAI / Anthropic — so downstream agents with
  idle-chunk timeouts don't abort a cold request. A failure *after* the 200 OK
  is surfaced as an in-band error event rather than a silent hang.
- **Long-turn escalation.** Optionally rewrites the model field when an
  incoming request is too large for the requested model's per-slot context,
  routing it to a configured higher-context sibling. Rules chain and are
  cycle-safe. See `OLLAMA_ROUTER_ESCALATE` below.
- **Strict-client compatibility.** Normalises malformed `/api/tags` fields
  (empty `modified_at` / `size`) that otherwise crash pydantic-based clients
  such as Home Assistant's Ollama integration — one bad model no longer takes
  down the whole list.
- **Multi-API surface.** `/v1/chat/completions`, `/v1/completions`,
  `/v1/messages` (Anthropic), `/v1/embeddings`, `/api/chat`, `/api/generate`,
  `/api/embed`, `/api/embeddings`, `/api/show` — each with the correct
  `stream` default per protocol. Mutating endpoints (`/api/pull`, `/api/push`,
  …) are blocked; unknown paths pass through to a healthy backend.
- **Spill-to-tmpfile** of request bodies, so model/stream detection and large
  contexts don't pin the whole payload in heap.
- **Optional bearer-token auth** with a hot-reloaded token file and an `/auth`
  endpoint for Traefik / NGINX `forwardAuth` middleware.
- **Prometheus metrics** — request rate (by model/backend/status/method),
  upstream latency, protocol translations, escalations (and skips), blocked
  and unknown-model requests; plus **self-health series** for diagnosing the
  router itself: `start_time_seconds` (a sawtooth here = pod churn), `ready`,
  `backends_reachable` / `backends_healthy` / per-backend `backend_up`,
  `upstream_errors{kind=connect|timeout|transport}`, and `heartbeat_engaged`.
- **OpenTelemetry tracing (opt-in).** When `OTEL_EXPORTER_OTLP_ENDPOINT` is
  set, per-request spans (`model_route` / `passthrough_route`, with `model`,
  `backend`, and status attributes) are batch-exported over OTLP/HTTP to Tempo
  (directly or via a Grafana Alloy collector). Unset = log-only; no collector
  needed for local runs. Reuses the existing reqwest/rustls stack (no gRPC).
- **Graceful shutdown** on SIGTERM / Ctrl-C — in-flight streams drain before
  the process exits, so rolling updates don't RST live responses.

## Build

```bash
cargo build --release
```

Or via Docker:

```bash
./build.sh   # builds and pushes to a configured registry
```

## Configuration

All configuration is via environment variables. Listening / discovery /
backend wiring:

| Var | Default | Purpose |
|---|---|---|
| `OLLAMA_ROUTER_BACKENDS` | `ollama=http://localhost:11434` | Comma-separated `name=url` pairs. Models are routed first-writer-wins across this list in declaration order. |
| `OLLAMA_ROUTER_PUBLIC_PORT` | `11434` | Port for the public OpenAI/Ollama-compat surface. |
| `OLLAMA_ROUTER_INTERNAL_PORT` | `9090` | Port for `/health`, `/status`, `/metrics`, `/auth`. |
| `OLLAMA_ROUTER_DISCOVERY_INTERVAL` | `60` | Seconds between backend model-list refreshes. |
| `OLLAMA_ROUTER_GRACE_MULTIPLIER` | `3` | Multiplied by discovery interval to compute the grace period in which an unreachable backend's discovered models stay routable. |
| `OLLAMA_ROUTER_CONNECT_TIMEOUT` | `10` | Connect-timeout (seconds) for upstream requests. |
| `OLLAMA_ROUTER_REQUEST_TIMEOUT` | `300` | End-to-end request timeout (seconds) for upstream requests. Long enough for streaming LLM responses. |
| `OLLAMA_ROUTER_TOKENS_FILE` | (unset, no auth) | Path to a newline-separated file of valid bearer tokens. Reloaded every 60 s without restart. |

Cold-load heartbeat (kicks in when an upstream model isn't loaded):

| Var | Default | Purpose |
|---|---|---|
| `OLLAMA_ROUTER_LOADING_HEARTBEAT` | `15` | Seconds between keepalive bytes during cold load. |
| `OLLAMA_ROUTER_PREFLIGHT_TIMEOUT` | `10` | Max wait (seconds) on the `/api/ps` preflight probe. |
| `OLLAMA_ROUTER_LOADING_MAX_WAIT` | `300` | Max wait (seconds) for the upstream to produce its first real byte before emitting an in-band error. |

Long-turn escalation (optional — empty / unset disables the feature):

| Var | Default | Purpose |
|---|---|---|
| `OLLAMA_ROUTER_ESCALATE` | (unset, no escalation) | Comma-separated `from_model:max_input_tokens:to_model` triples. When an incoming request for `from_model` has more than `max_input_tokens` of estimated input (Content-Length / 3), the model name is silently rewritten to `to_model` before dispatch. Rules chain: a single request can hop through multiple rules in declaration order. Example: `qwen3.6-medium:35000:qwen3.6-high,qwen3.6-high:120000:qwen3.6-ultra`. |

Notes on escalation:

- Requests that don't carry a `Content-Length` (chunked transfer,
  HTTP/2 streaming uploads) skip escalation and hit the originally
  requested model. The `ollama_router_escalations_skipped{reason=
  "no_content_length"}` counter tracks how often this fires.
- If the escalation target isn't visible in the registry (typo,
  decommissioned backend, or the 60-second discovery warmup window),
  the router falls back to the original model and tracks it under
  `reason="target_not_found"`.

Resilience — bounded retry with backoff. These knobs are validated at startup
but not yet consumed; the retry/backoff logic ships in a later release:

| Var | Default | Purpose |
|---|---|---|
| `OLLAMA_ROUTER_MAX_RETRIES` | `2` | Maximum retry attempts after the first try for a transient failure. `0` disables retry (single-shot). |
| `OLLAMA_ROUTER_RETRY_BACKOFF_BASE_MS` | `100` | Base delay (milliseconds) for exponential backoff between retry attempts. |
| `OLLAMA_ROUTER_RETRY_JITTER_PCT` | `25` | Random jitter as a percentage of the computed backoff (`25` = up to ±25%). `0` disables jitter; must be 0–100. |
| `OLLAMA_ROUTER_RETRY_LATENCY_BUDGET` | `30` | Hard wall-clock budget (seconds) across all attempts for a single request. |

Resilience — per-backend circuit breaker + admission control (validated at
startup, consumed in a later release):

| Var | Default | Purpose |
|---|---|---|
| `OLLAMA_ROUTER_BREAKER_5XX_THRESHOLD` | `5` | Consecutive 5xx responses that trip a backend's circuit breaker open. Must be at least 1. |
| `OLLAMA_ROUTER_BREAKER_OPEN` | `10` | How long (seconds) a backend's breaker stays open before a half-open probe. |
| `OLLAMA_ROUTER_BACKEND_MAX_INFLIGHT` | `0` | Per-backend in-flight request cap; over the cap sheds load as 503 rather than queueing. `0` = unlimited. |

Embedding cache (off by default; validated at startup, consumed in a later
release):

| Var | Default | Purpose |
|---|---|---|
| `OLLAMA_ROUTER_CACHE_ENABLED` | `false` | Master switch for the embedding cache. Accepts `true/false`, `1/0`, `yes/no`, `on/off`. |
| `OLLAMA_ROUTER_CACHE_MAX_BYTES` | `67108864` | Total byte budget for the cache across all entries (64 MiB). |
| `OLLAMA_ROUTER_CACHE_MAX_ENTRY_BYTES` | `1048576` | Skip caching any single body larger than this, in bytes (1 MiB); avoids buffering multi-MB bulk embeds. `0` = no per-entry cap. When non-zero, must not exceed `OLLAMA_ROUTER_CACHE_MAX_BYTES`. |
| `OLLAMA_ROUTER_CACHE_TTL` | `3600` | Time-to-live (seconds) for a cached embedding. |

Tracing (optional — unset disables OTLP export, leaving structured logs only):

| Var | Default | Purpose |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | (unset, tracing off) | OTLP/HTTP base endpoint, e.g. `http://tempo.monitor.svc:4318` or a node-local Alloy. Setting it enables per-request span export. Standard `OTEL_EXPORTER_OTLP_*` and `OTEL_TRACES_SAMPLER*` vars are honored by the exporter. |
| `OTEL_SERVICE_NAME` | `ollama-router` | `service.name` resource attribute on exported spans. |

## Endpoints

Public router (`OLLAMA_ROUTER_PUBLIC_PORT`):

| Path | Purpose |
|---|---|
| `POST /api/chat`, `/api/generate`, `/api/embed`, `/api/embeddings`, `/api/show` | Ollama-native API, routed to the backend hosting the requested model. |
| `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/messages` | OpenAI / Anthropic compat, routed by model. |
| `GET /api/tags` | Aggregated `/api/tags` across all backends. |
| `GET /api/ps` | Aggregated `/api/ps` showing currently-loaded models across every backend (with per-backend protocol shimming for llama-swap and always-resident backends). |
| `GET /v1/models`, `GET /v1/models/{id}` | Aggregated OpenAI-style model list. |
| `POST /api/pull`, `/api/delete`, `/api/copy`, `/api/create`, `/api/push` | Blocked (returns 4xx) — mutation operations not safe to proxy across a fan-out fleet. |
| (fallback) | Any other path is passthrough-proxied to the first healthy backend, for ad-hoc compatibility. |

Internal router (`OLLAMA_ROUTER_INTERNAL_PORT`):

| Path | Purpose |
|---|---|
| `GET /health` | 200 once first discovery cycle has completed and at least one backend is reachable. |
| `GET /status` | JSON dump of every backend's current health, models, and grace state. |
| `GET /metrics` | Prometheus text-format exposition. |
| `ANY /auth` | Token-validation endpoint for Traefik / NGINX `forwardAuth` middleware. |

## Roadmap

The resilience and embedding-cache environment variables above
(`OLLAMA_ROUTER_MAX_RETRIES`, the `…_BREAKER_*` / `…_BACKEND_MAX_INFLIGHT`
knobs, and the `…_CACHE_*` knobs) are **parsed and validated today but not yet
active** — the machinery that consumes them is in progress. The internal
plumbing it builds on has already landed: `proxy::execute` returns a typed
outcome (connect / timeout / transport vs. response), the registry can
enumerate the healthy backends serving a model, and handlers live in a library
module with a clean injection point.

Planned, in priority order:

- **Hide backend flakiness from clients** — bounded retry-with-backoff plus a
  per-backend circuit breaker / in-flight cap that sheds load as honest
  `503 + Retry-After` instead of relaying transient upstream 5xxs that abort a
  client's whole multi-minute job.
- **Embedding cache** — a memory-bounded, model-versioned cache for repeated
  small embedding requests, flushed on backend rediscovery (off by default).

See `docs/plans/` for the design and rationale.

## Tests

```bash
cargo test            # host-side unit + integration tests
./test.sh             # docker-buildx test target (matches CI)
```

## License

MIT — see [LICENSE](./LICENSE).

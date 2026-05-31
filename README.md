# ollama-router

A Rust HTTP proxy that fronts one or more Ollama / OpenAI / Anthropic-
compatible inference endpoints behind a single OpenAI-compatible API.

- **Per-model dispatch.** Discover models on every configured backend
  (`/api/tags` with `/v1/models` fallback) and route each request to the
  backend that hosts the requested model. Aggregated `/v1/models`,
  `/api/tags`, and `/api/ps` views unify the cluster from the client's
  perspective.
- **Cold-load heartbeat.** Preflight `/api/ps` and inject protocol-correct
  keepalive bytes (NDJSON empty chunks for Ollama, SSE comments for
  OpenAI / Anthropic) while a model is loading, so downstream agents
  with idle timeouts don't kill cold requests.
- **Long-turn escalation.** Optionally rewrite the model field
  transparently when an incoming request is too large for the requested
  model's per-slot context, sending it to a configured higher-context
  sibling instead. See `OLLAMA_ROUTER_ESCALATE` below.
- **Multi-protocol awareness.** Proxies `/v1/chat/completions`,
  `/v1/completions`, `/v1/messages` (Anthropic), `/v1/embeddings`,
  `/api/chat`, `/api/generate`, `/api/embed`, `/api/embeddings`,
  `/api/show`, picking the correct `stream`-default per protocol.
- **Spill-to-tmpfile** for large request bodies (avoids OOM on long
  contexts).
- **Optional bearer-token auth** with hot-reloadable token file.
- **Prometheus metrics** for request rate, upstream latency, heartbeat
  injections, and escalation events.
- **Graceful shutdown** on SIGTERM / Ctrl-C — in-flight streams drain
  before the process exits.

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
| `OLLAMA_ROUTER_BACKENDS` | `ollama-cuda=http://ollama-cuda.ai:11434,ollama-rocm=http://ollama-rocm.ai:11435` | Comma-separated `name=url` pairs. Models are routed first-writer-wins across this list in declaration order. |
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

## Tests

```bash
cargo test            # host-side unit + integration tests
./test.sh             # docker-buildx test target (matches CI)
```

## License

MIT — see [LICENSE](./LICENSE).

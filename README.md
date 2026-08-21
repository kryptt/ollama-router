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
- **One live policy file.** Backends, the per-backend spend boundary, alias
  chains and the fallback map all live in a single TOML document that is
  re-read and re-validated every discovery cycle — backend roster included,
  so adding, removing or reordering a backend needs no restart. A rejected
  edit keeps the previous config *entirely*; an unreadable file at startup
  leaves the router alive but **unready**, retrying, rather than crash-looping.
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
  `upstream_errors{kind=connect|timeout|transport}`, `heartbeat_engaged`, and
  `config_reloads_total{result}` (a rejected policy reload is otherwise
  completely silent — alert on it).
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

Two surfaces, split by lifecycle:

| Live — `router.toml`, picked up within one discovery cycle (≤60 s) | Restart-scoped — environment |
|---|---|
| backend roster, URLs, **declaration order (= routing priority)** | connect / request timeouts, `OLLAMA_ROUTER_EXTRA_CA_FILE` (frozen inside the two reqwest clients) |
| per-backend `allow` (the spend boundary) | ports (bound listeners) |
| `external`, `strip_auth` | `OLLAMA_ROUTER_TOKENS_FILE` (its own 60 s reloader) |
| alias chains and their `local` pins | discovery interval, grace multiplier, heartbeat knobs |
| the fallback map | `OLLAMA_ROUTER_ESCALATE`, the inert retry / breaker / cache knobs, `OTEL_*`, `RUST_LOG` |

The rule has no exceptions: **everything in the file is live, everything in
the environment needs a restart.** Escalation rules stay in the environment
because making them live would mean moving them into the registry and taking
a lock on the request hot path — a bigger change than the whole policy-file
migration, for a feature that is unset in production.

### The policy file

`OLLAMA_ROUTER_CONFIG` names a TOML document holding the backend roster, the
per-backend model allowlist, alias chains, and the fallback map. It is
**required and has no default**: the roster and the spend boundary both live
in it, so a router without one has nothing to route to.

```toml
# Document order is routing priority: the model map is first-writer-wins,
# so the backend declared first wins a colliding model name. Local,
# unmetered backends therefore come first and metered ones last.
[[backends]]
name  = "llama-swap-cuda"
url   = "http://llama-swap-cuda.ai:8080"
allow = ["*"]                      # on-LAN, unmetered: publish everything

[[backends]]
name  = "nous"
url   = "https://inference-api.nousresearch.com"
external   = true                  # => can never appear in a `local` alias
strip_auth = true                  # our inbound token is not a Nous credential
allow = [                          # THE SPEND BOUNDARY — prices per M tok (in/out)
  "nousresearch/hermes-4-70b",     # 0.05 / 0.20
  "qwen/qwen3.8-max",              # 1.60 / 4.80
]

[fallbacks]                        # QUOTE EVERY KEY — see the dotted-key trap
"qwen3.6-low" = "qwen/qwen3.8-27b"

[aliases.financial]
local = true                       # privacy pin, re-proved on every reload
chain = ["llama-swap-cuda/gemma4:e2b", "llama-swap/qwen3.6-low"]
```

**Backends** (`[[backends]]`, an array of tables — *not* a map, because the
order is a security boundary and a map would lose it):

| Key | Required | Purpose |
|---|---|---|
| `name` | yes | Routing key. Unique; what alias chains and metric labels refer to. |
| `url` | yes | `http(s)://host[:port]`; any trailing slash is stripped. |
| `allow` | yes | Discovery allowlist of exact model ids, or `["*"]` to publish everything. Mixing `"*"` with ids, or an empty list, is rejected. |
| `external` | no (`false`) | Hosted / off-LAN. A `local` alias whose chain touches an `external` backend rejects the whole file. |
| `strip_auth` | no (`false`) | Drop the client's `authorization` header before forwarding. For backends whose credential is injected downstream by an egress proxy: the inbound token is a *router* token, useless to the backend and not something to hand a third party. |

Notes on `allow`:

- It exists for **hosted aggregator backends**. A local llama-swap serves a
  handful of models and wants none of this; a hosted endpoint may advertise
  hundreds, which buries the local models in every consumer's picker.
- It is also the **spend boundary**. Anything that can reach the router can
  request any model the router publishes. For a metered backend, `allow` is
  what stops a client from putting a frontier model on your bill.
- Filtering happens at discovery, so an excluded model is not merely hidden —
  it never enters the model map and requests for it 404 like any unknown
  model.
- It is required rather than optional on purpose: making it optional means a
  deleted line is a *successful* parse that silently removes a backend's
  spend boundary.

**Fallbacks** (`[fallbacks]`, `local-model = "hosted-stand-in"`): when no
reachable backend serves a requested model, the request is transparently
rewritten — routing *and* the body's `model` field — to the stand-in. One
hop, and only for concrete (non-alias) names. Metric:
`ollama_router_fallbacks{from,to}`.

> **Quote every fallback key.** Model names contain `.`, and a bare
> `qwen3.6-low = "x"` is a valid TOML *dotted key* meaning
> `qwen3 = { "6-low" = "x" }`. The target type is a string map, so the nested
> table fails deserialisation loudly rather than mis-keying silently — but the
> quotes are what you actually want. Keys containing `:` or `/` are outright
> syntax errors unquoted.

**Aliases**: see [Aliases](#aliases-priority-chains) below.

Validation is **all-or-nothing**, and backends and aliases are checked in the
same pass against the same roster. Rejected outright: an unreadable or
unparseable file; an unknown field (`stip_auth = true` must be a rejection,
not a shrug — the file is hand-edited and has no CI in front of it); absent
or empty `backends`; a duplicate backend name; an empty name or non-`http(s)`
url; an absent, empty, or wildcard-mixed `allow`; an alias naming an
unconfigured backend; a `local` alias whose chain touches an `external`
backend; an alias named `local`; an empty chain; a malformed candidate; and a
fallback that maps to itself or has an empty side.

At **reload** a rejected file warns (`config reload rejected`), keeps the
previous config *entirely*, and increments
`ollama_router_config_reloads_total{result="rejected"}`. A rejected reload is
otherwise completely silent, so alert on that counter — and on
`ollama_router_config_last_reload_timestamp_seconds` going stale, which is
what a wedged loop looks like (the counter stops moving rather than
counting).

At **startup** a rejected file does *not* abort. The roster lives only in
this file, so aborting would turn any unrelated restart during a filesystem
blip — OOMKill, drain, eviction — into a `CrashLoopBackOff` with no way back.
Instead the router starts with an empty roster: alive (`/live` 200), never
ready (`/health` 503, `no backends configured`), so the Service has no
endpoints and clients get connection-refused rather than a misroute. The
discovery loop retries every cycle, and the pod self-heals the moment the
file is readable.

The read itself runs on a blocking thread under a 10 s budget. The file is on
NFS, where a hard mount against an unreachable server blocks *indefinitely*;
run inline that would freeze the discovery loop entirely — no probes, no
grace expiry, and no reload counting, so the alert above would never fire
while `/health` answered from a frozen snapshot. A read past its budget is
just another rejected reload.

Write the file atomically (`cp router.toml router.toml.new && mv
router.toml.new router.toml`): a torn read is simply a rejected reload, but
there is no reason to take the cycle.

### Why polling, and not a file watcher

The policy file lives on NFS and is edited on the NFS *server*. inotify is a
local-VFS mechanism: an NFS client receives no events for changes made on the
server or by another client, so a watcher would sit silent forever while
looking like it worked — strictly worse than polling. The discovery loop's
existing interval is the reload trigger, and because each iteration reloads
*before* snapshotting its probe targets, a backend added by an edit is
probed in the same cycle rather than the next one.

### Aliases (priority chains)

`[aliases.<name>]` tables define **routing aliases**: stable client-facing
model names that fan out over an ordered chain of concrete `backend/model`
candidates. The router walks the chain top to bottom and commits to the first
candidate that answers; transport failures (connect/timeout/transport) and
retryable statuses (429 rate-limit, 404 model-missing, 5xx) advance to the
next candidate. The advance decision is made on the response head, before any
body byte is forwarded, so streaming requests fail over cleanly too.

```toml
[aliases.fast]
chain = ["llama-swap/qwen3.6:latest", "freellmapi/groq/llama-3.3-70b", "nous/Hermes-4-405B"]

[aliases.secret]
local = true
chain = ["llama-swap/qwen3.6:latest", "rivoli/glm-5.2"]
```

Advance reasons per candidate: unreachable in the registry, connect/timeout/
transport errors, and 429 (rate-limit), 401/403 (auth — an expired hosted
API key fails over instead of relaying the auth error), 404 (model missing),
or 5xx response heads.

- The candidate splits on the **first** `/` only, so model ids keep their own
  slashes and colons (`freellmapi/groq/llama-3.3-70b` = backend `freellmapi`,
  model `groq/llama-3.3-70b`).
- Ollama clients normalise bare model names to `name:latest`; an alias
  resolves under both spellings (`fast` and `fast:latest`), with metrics
  labelled by the canonical configured name. Only a trailing `:latest` is
  normalised; an alias literally named `foo:latest` always wins exactly.
- `local = true` privacy-pins an alias: if any candidate's backend is marked
  `external`, the file is rejected. Because backends and aliases validate in
  the same pass against the same roster, **every reload re-proves the pin** —
  flipping a chained backend to `external` bounces the whole file rather than
  silently un-pinning it. By construction, traffic for a `local` alias can
  never leave the local backends: even when every local candidate is down the
  chain exhausts (client gets the last upstream failure, or 502) rather than
  escaping.
- A candidate's model does **not** have to appear in that backend's `allow`
  list. Chains skip non-serving candidates by design, and a placeholder
  candidate for a not-yet-provisioned provider is a deliberate pattern.
- Aliases are a **distinct namespace**, resolved before — and shadowing —
  concrete model lookup. Escalation and the single-hop fallback map are
  skipped entirely for alias requests: the chain is the failover mechanism.
  (A fallback-map *stand-in* may itself name an alias, though — that request
  enters the chain path.)
- **Streaming**: the cold-load heartbeat (which commits `200 OK` before the
  upstream status is known, forfeiting the rest of the chain) engages for a
  candidate only when its backend's discovered model list advertises the
  candidate model — positive evidence it exists and is merely cold. A
  reachable llama-swap that does *not* advertise the model is passed over
  immediately (`reason="model_missing"`) without an upstream attempt.
- When **no** candidate is reachable (startup race before the first
  discovery cycle, or a registry-blind outage), the walk attempts every
  candidate in order anyway — connect/timeout advances weed out the truly
  dead ones instead of returning a blind 502.
- Aliases appear in `/v1/models` (`owned_by: "router-alias"`) and `/api/tags`
  (synthesised, pydantic-safe entries); concrete models keep listing.
- There are deliberately no per-candidate retries or backoff — the chain
  advance *is* the retry.
- Metrics: `ollama_router_chain_advance{alias,to,reason}` (`to` is the
  candidate backend moved past; reasons:
  `unreachable|connect|timeout|transport|auth|rate_limited|model_missing|upstream_5xx`) and
  `ollama_router_chain_exhausted{alias}`. An exhausted chain still records
  `requests_total` — alias as the model, and the backend whose failure was
  relayed (`"none"` when no candidate produced a response).

### Environment

| Var | Default | Purpose |
|---|---|---|
| `OLLAMA_ROUTER_CONFIG` | **required** | Path to the `router.toml` policy file above. |
| `OLLAMA_ROUTER_PUBLIC_PORT` | `11434` | Port for the public OpenAI/Ollama-compat surface. |
| `OLLAMA_ROUTER_INTERNAL_PORT` | `9090` | Port for `/health`, `/live`, `/status`, `/metrics`, `/auth`. |
| `OLLAMA_ROUTER_DISCOVERY_INTERVAL` | `60` | Seconds between cycles. Each cycle re-reads `router.toml` and *then* probes, so a backend added by an edit is probed in that same cycle. |
| `OLLAMA_ROUTER_GRACE_MULTIPLIER` | `3` | Multiplied by the discovery interval to compute the grace period in which an unreachable backend's discovered models stay routable. |
| `OLLAMA_ROUTER_CONNECT_TIMEOUT` | `10` | Connect-timeout (seconds) for upstream requests. |
| `OLLAMA_ROUTER_REQUEST_TIMEOUT` | `300` | End-to-end request timeout (seconds). Long enough for streaming LLM responses. |
| `OLLAMA_ROUTER_TOKENS_FILE` | (unset, no auth) | Path to a newline-separated file of valid bearer tokens. Reloaded every 60 s without restart. **Set-but-empty is a startup error**, not "no auth": treating it as unset would silently disable authentication entirely (a `valueFrom` that renders empty would bring the router up wide open), and keeping it would 401 every request against an unreadable path. Only an *absent* variable disables auth — and when it is absent the router logs a warning and reports `ollama_router_auth_enabled 0`. |
| `OLLAMA_ROUTER_EXTRA_CA_FILE` | (unset, built-in roots only) | Path to a PEM bundle of additional root certificates to trust on outbound requests. Needed when a backend is reached through a TLS-intercepting egress proxy whose CA is private. Applied to both the proxy and discovery clients, and fatal at startup if unreadable. |

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

Resilience and caching (validated at startup; retry/breaker/cache logic ships
in a later release):

| Var | Default | Purpose |
|---|---|---|
| `OLLAMA_ROUTER_MAX_RETRIES` | `2` | Maximum retry attempts after the first try for a transient failure. `0` disables retry (single-shot). |
| `OLLAMA_ROUTER_RETRY_BACKOFF_BASE_MS` | `100` | Base delay (milliseconds) for exponential backoff between retry attempts. |
| `OLLAMA_ROUTER_RETRY_JITTER_PCT` | `25` | Random jitter as a percentage of the computed backoff (`25` = up to ±25%). `0` disables jitter; must be 0--100. |
| `OLLAMA_ROUTER_RETRY_LATENCY_BUDGET` | `30` | Hard wall-clock budget (seconds) across all attempts for a single request. |
| `OLLAMA_ROUTER_BREAKER_5XX_THRESHOLD` | `5` | Consecutive 5xx responses that trip a backend's circuit breaker open. Must be at least 1. |
| `OLLAMA_ROUTER_BREAKER_OPEN` | `10` | How long (seconds) a backend's breaker stays open before a half-open probe. |
| `OLLAMA_ROUTER_BACKEND_MAX_INFLIGHT` | `0` | Per-backend in-flight request cap; over the cap sheds load as 503 rather than queueing. `0` = unlimited. |
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
| `GET /health` | **Readiness.** 200 once the first discovery cycle has completed and at least one backend is *configured*. A router whose policy file has never loaded reports `503 {"reason": "no backends configured"}` — it can serve nothing, so it must never take Service endpoints. Deliberately **not** gated on backends being *reachable*: serving through an all-down blip is the router's job (grace periods, fallbacks, alias chains), and with `replicas: 1` going un-Ready would empty the Service and turn honest 502s into connection-refused. Use `ollama_router_backend_up` / `backends_reachable` to alert on backend health. |
| `GET /live` | **Liveness.** 200 for as long as the process is serving. Separate from `/health` on purpose: readiness is 503 while the policy file is unreadable, and pointing a liveness probe at that would kill the pod that is waiting to self-heal. |
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

Note: for **aliased** traffic, chain failover supersedes the single-hop
`[fallbacks]` mechanism — alias requests never consult the fallback map.
The fallback map remains the mechanism for concrete (non-alias) model
names.

See `docs/plans/` for the design and rationale.

## Tests

```bash
cargo test            # host-side unit + integration tests
./test.sh             # docker-buildx test target (matches CI)
```

## License

MIT — see [LICENSE](./LICENSE).

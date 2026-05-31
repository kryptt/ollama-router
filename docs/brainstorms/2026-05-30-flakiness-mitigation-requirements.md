---
date: 2026-05-30
topic: flakiness-mitigation
---

# ollama-router — Hide Backend/Restart Flakiness From Clients

## Problem Frame

End clients (grepai bulk-embedding + search, Hindsight, etc.) intermittently
receive HTTP **500s** and "no healthy backends" errors from ollama-router in
two windows:

1. **Router restart/reschedule** — for ~60–75s after a new pod starts, the
   registry hasn't re-discovered the llama-swap backends (discovery interval
   60s) and readiness hasn't passed, so requests fail.
2. **Backend blip** — a backend (e.g. llama-swap-cuda serving
   jina-code-embeddings) briefly returns 5xx under load or during a model
   swap; ollama-router relays the upstream status verbatim, so the client
   sees the 500.

Observed impact this session: grepai's multi-minute bulk index repeatedly
failed because a single relayed 500 is fatal to the whole job, and search
queries 500 intermittently. The goal is that **clients see success (with at
most slightly higher latency), not errors** — while keeping added latency and
resource usage minimal (no second always-on replica).

Scope of THIS effort (user decision): mitigations **#2, #4, #5**. The
warm-start registry (#1) and heartbeat-hold-on-not-ready (#3) ideas are
deferred — #4's zero-gap rollout plus #2's retry are the chosen way to cover
the restart window instead.

## Requirements

**Retry & Failover (idea #2)**
- R1. On a *retryable* upstream failure, the router retries the request on an
  alternate backend that serves the requested model before surfacing any
  error to the client.
- R2. When no backend is ready yet (e.g. discovery in flight just after
  start), the router briefly re-checks/holds within a bounded budget rather
  than immediately returning "no healthy backends".
- R3. Retries are bounded by both an attempt count and a total added-latency
  budget, so the healthy path is unaffected and a genuinely-dead request
  still fails reasonably fast.
- R4. Failures are classified: transient/retryable (connection refused,
  backend 5xx, no-ready-backend, timeouts) vs fatal/passthrough (client 4xx
  such as the over-context **400**, malformed request). Fatal errors are
  never retried or masked.
- R5. Retry is only attempted while it is still safe — before any response
  bytes have been sent to the client. Streaming responses past the first byte
  are not retried. Inference requests are treated as idempotent.

**Graceful Lifecycle (idea #4)**
- R6. On `SIGTERM`, the router enters lameduck: stop accepting new requests,
  drain in-flight requests up to the termination grace period, then exit
  cleanly. (The current `FailedPreStopHook` is a bug to fix here.)
- R7. Readiness (`/health`) reports ready only when ≥1 backend is actually
  discovered and serving — not merely when the process is up.
- R8. Support zero-gap rolling updates: a new Ready pod must exist before the
  old one terminates, so a planned restart/rollout produces no client-visible
  error window.

**Result Cache (idea #5)**
- R9. A bounded in-memory LRU cache of successful embedding results, keyed by
  (model, normalized-input hash), with a TTL. On hit, return without calling a
  backend.
- R10. Cache size and TTL are configurable with conservative bounded
  defaults; only deterministic/idempotent results are cached (embeddings; not
  sampled chat generations).

## Success Criteria
- A single backend 5xx/blip during a request yields a successful client
  response, not a 5xx (R1–R5).
- A planned router rollout/restart produces zero client-visible error window
  (R6–R8).
- Identical/retried embedding requests during a flap are served from cache,
  reducing backend calls and surviving transient blips (R9–R10).
- Steady-state (healthy) latency and idle resource usage are essentially
  unchanged; no second always-on replica required.

## Scope Boundaries
- NOT fixing the `stuck-pod-pruner` churn that kills the pod before readiness
  — handled separately (cluster-side).
- NOT adding a permanent second replica for HA (resource constraint). A
  transient surge pod during rollout (R8) is acceptable.
- NOT implementing the warm-start registry (#1) or the
  heartbeat-hold-on-not-ready (#3) in this round.
- NOT caching non-deterministic generations (chat/completions with sampling).

## Key Decisions
- **Chose #2/#4/#5 over #1/#3**: #4's zero-gap rollout + #2's retry cover the
  restart window without persisting/rehydrating registry state or holding
  connections; #5 backstops retries and repeated queries.
- **Inference is idempotent** → safe to retry before first response byte.
- **Relayed upstream 5xx is the client-visible failure** → retry/failover is
  the highest-leverage single mechanism (R1).

## Dependencies / Assumptions
- Two backends (llama-swap, llama-swap-cuda) often serve overlapping models,
  enabling cross-backend failover (R1).
- Embedding output is deterministic for identical input (assumption behind R9).

## Outstanding Questions

### Deferred to Planning
- [Affects R8][Technical] The pod uses a single macvlan LAN IP
  (192.168.2.61); `maxSurge:1` needs a transient second macvlan IP (or an
  alternative surge mechanism). Determine whether zero-gap surge is feasible
  under the Multus single-IP setup, or whether R8 must rely on fast
  readiness + #2 retry instead.
- [Affects R3][Technical] Concrete retry attempt count + latency budget, and
  backoff shape.
- [Affects R6][Technical] Root-cause and fix for the current
  `FailedPreStopHook`; correct preStop + terminationGracePeriod values.
- [Affects R9/R10][Needs research] Cache key normalization, size/TTL defaults,
  and confirmation that the embedding endpoint is deterministic per input.

## Next Steps
→ `/ce:plan` for structured implementation planning

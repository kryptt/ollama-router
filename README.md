# ollama-router

A minimal Rust HTTP proxy that fronts an [Ollama](https://ollama.com/) server
(and other OpenAI/Anthropic-compatible inference endpoints) with:

- **Cold-load heartbeat** — preflights `/api/ps` and injects protocol-correct
  keepalive bytes (NDJSON empty chunks for Ollama, SSE comments for OpenAI/
  Anthropic) while a model is loading, so downstream agents with idle
  timeouts don't kill cold requests.
- **Multi-protocol routing** — proxies `/v1/chat/completions`,
  `/v1/messages` (Anthropic), `/api/generate`, `/api/chat`, etc. with
  appropriate transformations.
- **Spill-to-tmpfile** for large request bodies (avoids OOM on long contexts).
- **Optional auth** via static API key.
- **Prometheus metrics** for request rate, upstream latency, heartbeat
  injections.
- **Per-model registry** for transparent model-name aliasing.

## Build

```bash
cargo build --release
```

Or via Docker:

```bash
./build.sh   # builds and pushes to a configured registry
```

## Configuration

Environment variables (see `src/config.rs` for full list):

| Var | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Upstream Ollama endpoint |
| `BIND_ADDR` | `0.0.0.0:8080` | Listen address |
| `LOADING_HEARTBEAT` | `15s` | Interval for keepalive bytes during cold load |
| `PREFLIGHT_TIMEOUT` | `10s` | Max wait on `/api/ps` before giving up |
| `LOADING_MAX_WAIT` | `300s` | Max wait for a model to load before erroring |
| `API_KEY` | (unset, no auth) | Static bearer/`x-api-key` for inbound auth |

## Tests

```bash
cargo test            # host-side unit tests
./test.sh             # docker-buildx test target (matches CI)
```

## License

MIT — see [LICENSE](./LICENSE).

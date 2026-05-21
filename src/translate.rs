//! Protocol translation between Ollama-native `/api/chat` and OpenAI
//! `/v1/chat/completions`.
//!
//! Used when the registry says a backend hosting the requested model only
//! speaks the OpenAI dialect (e.g. llama-swap, llama.cpp's openai-compat
//! server). The router rewrites the upstream URL to `/v1/chat/completions`,
//! translates the request body, and reshapes the response — either a single
//! JSON object or an SSE stream — back into the Ollama-native NDJSON / JSON
//! shape the client expects.

use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use futures_util::Stream;
use serde_json::{Map, Value, json};

/// A valid RFC 3339 timestamp. The field is required by Ollama clients but
/// the actual value is not used for anything that matters here, so a fixed
/// value avoids pulling in a date crate. Matches `heartbeat::FIXED_TIMESTAMP`.
const FIXED_TIMESTAMP: &str = "1970-01-01T00:00:00Z";

/// Translate an Ollama-native `/api/chat` request body to an OpenAI
/// `/v1/chat/completions` body. Lifts `options.*` to the root with
/// OpenAI-compatible names; drops Ollama-only fields like `keep_alive`.
pub fn ollama_chat_to_openai_request(bytes: &[u8]) -> Result<Vec<u8>, serde_json::Error> {
    let mut root: Value = serde_json::from_slice(bytes)?;
    let Some(obj) = root.as_object_mut() else {
        return serde_json::to_vec(&root);
    };

    // Drop Ollama-only fields that OpenAI doesn't define.
    obj.remove("keep_alive");

    // Lift `options.*` up to the root with OpenAI-compatible names. We
    // remove `options` first so we own its contents.
    if let Some(Value::Object(options)) = obj.remove("options") {
        for (key, value) in options {
            let translated_key = match key.as_str() {
                "temperature" => "temperature",
                "top_p" => "top_p",
                "top_k" => "top_k", // OpenAI ignores unknown fields harmlessly.
                "num_predict" => "max_tokens",
                "stop" => "stop",
                "seed" => "seed",
                "repeat_penalty" => "frequency_penalty",
                // num_ctx is a server-side knob, not a request param. Drop.
                "num_ctx" => continue,
                // Anything else: pass through with original name. OpenAI
                // tolerates unknown fields.
                _ => {
                    obj.entry(key).or_insert(value);
                    continue;
                }
            };
            obj.insert(translated_key.to_string(), value);
        }
    }

    serde_json::to_vec(&root)
}

/// Translate a non-streaming OpenAI `/v1/chat/completions` response into the
/// Ollama `/api/chat` non-streaming shape. `model_name` is the model the
/// client originally asked for (post-escalation).
pub fn openai_chat_to_ollama_response(
    bytes: &[u8],
    model_name: &str,
) -> Result<Vec<u8>, serde_json::Error> {
    let resp: Value = serde_json::from_slice(bytes)?;

    let choice = resp.get("choices").and_then(|c| c.get(0));
    let content = choice
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .cloned()
        .unwrap_or_else(|| Value::String(String::new()));
    let finish_reason = choice
        .and_then(|c| c.get("finish_reason"))
        .cloned()
        .unwrap_or(Value::Null);

    let usage = resp.get("usage");
    let prompt_tokens = usage
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let completion_tokens = usage
        .and_then(|u| u.get("completion_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let out = json!({
        "model": model_name,
        "created_at": current_timestamp(),
        "message": {
            "role": "assistant",
            "content": content,
        },
        "done": true,
        "done_reason": finish_reason,
        "total_duration": 0u64,
        "load_duration": 0u64,
        "prompt_eval_count": prompt_tokens,
        "prompt_eval_duration": 0u64,
        "eval_count": completion_tokens,
        "eval_duration": 0u64,
    });

    serde_json::to_vec(&out)
}

/// Wrap an OpenAI SSE byte stream and re-emit it as Ollama NDJSON.
///
/// Buffers across chunk boundaries: each SSE event ends with `\n\n` (an
/// empty line). Inside the event we look for `data: <payload>` — the rest
/// (`event:` lines, comments) is ignored. `data: [DONE]` (or upstream EOF)
/// causes the final `done:true` frame to be emitted.
pub fn translate_streaming_response<S>(
    upstream: S,
    model_name: String,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>
where
    S: Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static,
{
    Box::pin(SseToNdjson::new(upstream, model_name))
}

/// Stream adapter: OpenAI SSE in, Ollama NDJSON out.
struct SseToNdjson<S> {
    upstream: Pin<Box<S>>,
    model: String,
    /// Bytes received from upstream but not yet split into complete events.
    buf: Vec<u8>,
    /// NDJSON lines ready to be emitted to the client.
    out: VecDeque<Bytes>,
    /// Upstream has signalled `data: [DONE]` or closed; flush a final
    /// `done:true` frame on the next poll, then end.
    finished: bool,
    /// Final frame already emitted; next poll returns `None`.
    done_emitted: bool,
    /// Cumulative usage parsed from upstream (OpenAI sometimes attaches
    /// usage to the final delta with `stream_options.include_usage`).
    prompt_tokens: u64,
    completion_tokens: u64,
    /// Latest `finish_reason` observed in any delta.
    finish_reason: Value,
}

impl<S> SseToNdjson<S> {
    fn new(upstream: S, model: String) -> Self {
        Self {
            upstream: Box::pin(upstream),
            model,
            buf: Vec::with_capacity(4096),
            out: VecDeque::new(),
            finished: false,
            done_emitted: false,
            prompt_tokens: 0,
            completion_tokens: 0,
            finish_reason: Value::Null,
        }
    }

    /// Drain whatever complete events live in `buf`, append their NDJSON
    /// translations to `out`, and set `finished` if `[DONE]` was seen.
    fn drain_events(&mut self) {
        // SSE events are separated by `\n\n`. Split off complete prefixes.
        while let Some(idx) = find_double_newline(&self.buf) {
            let event_bytes: Vec<u8> = self.buf.drain(..idx + 2).collect();
            // event_bytes ends with `\n\n`; trim the terminator before parsing.
            let event_str = match std::str::from_utf8(&event_bytes) {
                Ok(s) => s.trim_end_matches('\n'),
                Err(e) => {
                    tracing::warn!(error = %e, "SSE event contained invalid UTF-8; skipping");
                    continue;
                }
            };

            // Each event is one or more `field: value` lines. We only care
            // about `data:` lines; everything else (`event:`, comments
            // starting with `:`) is dropped per the SSE spec.
            for line in event_str.lines() {
                let Some(payload) = line.strip_prefix("data:") else {
                    continue;
                };
                let payload = payload.trim_start();

                if payload == "[DONE]" {
                    self.finished = true;
                    continue;
                }

                let value: Value = match serde_json::from_str(payload) {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::warn!(error = %e, payload = %payload, "skipping non-JSON SSE data line");
                        continue;
                    }
                };

                self.handle_chunk(value);
            }
        }
    }

    /// Translate a single OpenAI delta chunk into an Ollama NDJSON line and
    /// queue it for emission. Also accumulates usage / finish_reason for
    /// the final `done:true` frame.
    fn handle_chunk(&mut self, value: Value) {
        let choice = value.get("choices").and_then(|c| c.get(0));
        let content = choice
            .and_then(|c| c.get("delta"))
            .and_then(|d| d.get("content"))
            .cloned()
            .unwrap_or_else(|| Value::String(String::new()));

        if let Some(fr) = choice.and_then(|c| c.get("finish_reason"))
            && !fr.is_null()
        {
            self.finish_reason = fr.clone();
        }

        if let Some(usage) = value.get("usage") {
            if let Some(p) = usage.get("prompt_tokens").and_then(|v| v.as_u64()) {
                self.prompt_tokens = p;
            }
            if let Some(c) = usage.get("completion_tokens").and_then(|v| v.as_u64()) {
                self.completion_tokens = c;
            }
        }

        // Only emit a frame if there's actual content. An empty delta (the
        // OpenAI "role: assistant" priming chunk, or a usage-only final
        // chunk) carries no token for the Ollama client.
        let has_content = content.as_str().is_some_and(|s| !s.is_empty());
        if has_content {
            let chunk = json!({
                "model": self.model,
                "created_at": current_timestamp(),
                "message": { "role": "assistant", "content": content },
                "done": false,
            });
            let line = format!("{chunk}\n");
            self.out.push_back(Bytes::from(line));
        }
    }

    /// Build the terminal `done:true` Ollama frame from accumulated stats.
    fn final_frame(&self) -> Bytes {
        let mut frame = Map::new();
        frame.insert("model".to_string(), json!(self.model));
        frame.insert("created_at".to_string(), json!(current_timestamp()));
        frame.insert(
            "message".to_string(),
            json!({ "role": "assistant", "content": "" }),
        );
        frame.insert("done".to_string(), Value::Bool(true));
        frame.insert("done_reason".to_string(), self.finish_reason.clone());
        frame.insert("total_duration".to_string(), json!(0u64));
        frame.insert("load_duration".to_string(), json!(0u64));
        frame.insert("prompt_eval_count".to_string(), json!(self.prompt_tokens));
        frame.insert("prompt_eval_duration".to_string(), json!(0u64));
        frame.insert("eval_count".to_string(), json!(self.completion_tokens));
        frame.insert("eval_duration".to_string(), json!(0u64));
        let s = format!("{}\n", Value::Object(frame));
        Bytes::from(s)
    }
}

impl<S> Stream for SseToNdjson<S>
where
    S: Stream<Item = Result<Bytes, std::io::Error>>,
{
    type Item = Result<Bytes, std::io::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            // 1. Drain any ready output first.
            if let Some(b) = self.out.pop_front() {
                return Poll::Ready(Some(Ok(b)));
            }

            // 2. If upstream is done, emit the final frame and stop.
            if self.finished {
                if self.done_emitted {
                    return Poll::Ready(None);
                }
                self.done_emitted = true;
                let frame = self.final_frame();
                return Poll::Ready(Some(Ok(frame)));
            }

            // 3. Pull more bytes from upstream.
            match self.upstream.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    self.buf.extend_from_slice(&chunk);
                    self.drain_events();
                    // Loop back — drain_events may have queued output or
                    // flipped `finished`.
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(e)));
                }
                Poll::Ready(None) => {
                    // Upstream EOF without `[DONE]` — treat as finished and
                    // still emit a closing frame so the client sees done:true.
                    self.finished = true;
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Find the byte index of the first `\n\n` in `buf`, or `None` if not present.
fn find_double_newline(buf: &[u8]) -> Option<usize> {
    buf.windows(2).position(|w| w == b"\n\n")
}

/// RFC 3339 timestamp generated from `SystemTime::now()` without pulling in
/// a date crate. Falls back to the fixed epoch string on the (impossible in
/// practice) case where the clock is before UNIX epoch.
fn current_timestamp() -> String {
    let now = SystemTime::now();
    let Ok(dur) = now.duration_since(UNIX_EPOCH) else {
        return FIXED_TIMESTAMP.to_string();
    };
    // Compute Y-M-D H:M:S from seconds since epoch.
    let secs = dur.as_secs() as i64;
    format_rfc3339_utc(secs)
}

/// Format seconds-since-epoch as `YYYY-MM-DDTHH:MM:SSZ`. Cheap civil-time
/// arithmetic; no leap seconds, all UTC. Good enough for a synthetic
/// `created_at` that clients only use as a parseable timestamp.
fn format_rfc3339_utc(secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    let secs_of_day = secs.rem_euclid(86_400);
    let hour = secs_of_day / 3600;
    let minute = (secs_of_day % 3600) / 60;
    let second = secs_of_day % 60;

    // Civil-from-days algorithm (Howard Hinnant). Returns (year, month, day)
    // for the Gregorian calendar.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    format!("{y:04}-{m:02}-{d:02}T{hour:02}:{minute:02}:{second:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;
    use futures_util::stream;

    // ── request translation ──────────────────────────────────────────────

    #[test]
    fn request_lifts_options_and_renames() {
        let input = br#"{
            "model": "qwen3.6-medium",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 256,
                "stop": ["\n"],
                "seed": 42,
                "repeat_penalty": 1.1,
                "num_ctx": 8192
            },
            "keep_alive": "30m"
        }"#;
        let out = ollama_chat_to_openai_request(input).unwrap();
        let v: Value = serde_json::from_slice(&out).unwrap();

        assert!(v.get("options").is_none(), "options must be flattened away");
        assert!(
            v.get("keep_alive").is_none(),
            "keep_alive is Ollama-only and must be dropped",
        );
        assert!(v.get("num_ctx").is_none(), "num_ctx must be dropped");

        assert_eq!(v["model"], "qwen3.6-medium");
        assert_eq!(v["stream"], true);
        assert_eq!(v["temperature"], 0.7);
        assert_eq!(v["top_p"], 0.9);
        assert_eq!(v["top_k"], 40);
        assert_eq!(v["max_tokens"], 256);
        assert_eq!(v["stop"], json!(["\n"]));
        assert_eq!(v["seed"], 42);
        assert_eq!(v["frequency_penalty"], 1.1);
        assert_eq!(v["messages"][0]["content"], "hi");
    }

    #[test]
    fn request_passes_through_tools_and_format() {
        let input = br#"{
            "model": "x",
            "messages": [],
            "format": "json",
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "tool_choice": "auto"
        }"#;
        let out = ollama_chat_to_openai_request(input).unwrap();
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["format"], "json");
        assert_eq!(v["tools"][0]["function"]["name"], "f");
        assert_eq!(v["tool_choice"], "auto");
    }

    #[test]
    fn request_with_no_options_is_a_no_op_aside_from_keep_alive() {
        let input = br#"{"model": "x", "messages": [], "keep_alive": "1h"}"#;
        let out = ollama_chat_to_openai_request(input).unwrap();
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["model"], "x");
        assert!(v.get("keep_alive").is_none());
    }

    #[test]
    fn request_invalid_json_returns_error() {
        assert!(ollama_chat_to_openai_request(b"{not json").is_err());
    }

    #[test]
    fn request_preserves_root_field_collisions_under_options() {
        // If both root and options carry the same unknown key, the root
        // wins (we use Entry::or_insert when lifting unknowns). This is
        // intentional — root fields are explicit caller intent.
        let input = br#"{
            "model": "x",
            "messages": [],
            "custom": "root_value",
            "options": { "custom": "options_value" }
        }"#;
        let out = ollama_chat_to_openai_request(input).unwrap();
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["custom"], "root_value");
    }

    // ── non-streaming response translation ───────────────────────────────

    #[test]
    fn response_non_streaming_basic() {
        let input = br#"{
            "id": "chatcmpl-x",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "qwen3.6-medium",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello world"},
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
        }"#;
        let out = openai_chat_to_ollama_response(input, "qwen3.6-medium").unwrap();
        let v: Value = serde_json::from_slice(&out).unwrap();

        assert_eq!(v["model"], "qwen3.6-medium");
        assert_eq!(v["done"], true);
        assert_eq!(v["done_reason"], "stop");
        assert_eq!(v["message"]["role"], "assistant");
        assert_eq!(v["message"]["content"], "hello world");
        assert_eq!(v["prompt_eval_count"], 12);
        assert_eq!(v["eval_count"], 5);
        assert!(v.get("created_at").is_some());
    }

    #[test]
    fn response_missing_usage_is_zero() {
        let input = br#"{
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]
        }"#;
        let out = openai_chat_to_ollama_response(input, "m").unwrap();
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["prompt_eval_count"], 0);
        assert_eq!(v["eval_count"], 0);
    }

    #[test]
    fn response_missing_choices_yields_empty_content() {
        let out = openai_chat_to_ollama_response(b"{}", "m").unwrap();
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["message"]["content"], "");
        assert_eq!(v["done"], true);
    }

    // ── streaming translation ────────────────────────────────────────────

    fn sse_chunk(content: &str, finish: Option<&str>) -> String {
        let mut delta = Map::new();
        delta.insert("content".to_string(), json!(content));
        let mut choice = Map::new();
        choice.insert("index".to_string(), json!(0));
        choice.insert("delta".to_string(), Value::Object(delta));
        choice.insert(
            "finish_reason".to_string(),
            finish.map(|s| json!(s)).unwrap_or(Value::Null),
        );
        let chunk = json!({
            "id": "chatcmpl-x",
            "object": "chat.completion.chunk",
            "choices": [choice],
        });
        format!("data: {chunk}\n\n")
    }

    async fn collect(
        mut s: Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>,
    ) -> Vec<String> {
        let mut all = Vec::new();
        while let Some(item) = s.next().await {
            let b = item.unwrap();
            for line in std::str::from_utf8(&b).unwrap().split_terminator('\n') {
                if !line.is_empty() {
                    all.push(line.to_string());
                }
            }
        }
        all
    }

    #[tokio::test]
    async fn streaming_three_deltas_then_done() {
        let chunks = vec![
            Ok::<Bytes, std::io::Error>(Bytes::from(sse_chunk("hello", None))),
            Ok(Bytes::from(sse_chunk(" ", None))),
            Ok(Bytes::from(sse_chunk("world", Some("stop")))),
            Ok(Bytes::from("data: [DONE]\n\n")),
        ];
        let upstream = stream::iter(chunks);
        let out = translate_streaming_response(upstream, "m".to_string());
        let lines = collect(out).await;

        // 3 deltas with content + 1 final done:true frame
        assert_eq!(lines.len(), 4, "got {} lines: {lines:?}", lines.len());

        for (i, expected_content) in ["hello", " ", "world"].iter().enumerate() {
            let v: Value = serde_json::from_str(&lines[i]).unwrap();
            assert_eq!(v["done"], false);
            assert_eq!(v["model"], "m");
            assert_eq!(v["message"]["role"], "assistant");
            assert_eq!(v["message"]["content"], *expected_content);
        }
        let last: Value = serde_json::from_str(&lines[3]).unwrap();
        assert_eq!(last["done"], true);
        assert_eq!(last["done_reason"], "stop");
    }

    #[tokio::test]
    async fn streaming_handles_split_mid_event() {
        // Split a single SSE event across multiple Bytes chunks to ensure
        // the buffer joins them correctly before parsing.
        let event = sse_chunk("hello", None);
        let split = event.len() / 2;
        let chunks = vec![
            Ok::<Bytes, std::io::Error>(Bytes::from(event[..split].to_string())),
            Ok(Bytes::from(event[split..].to_string())),
            Ok(Bytes::from("data: [DONE]\n\n")),
        ];
        let upstream = stream::iter(chunks);
        let out = translate_streaming_response(upstream, "m".to_string());
        let lines = collect(out).await;

        assert_eq!(lines.len(), 2);
        let v: Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(v["message"]["content"], "hello");
        let last: Value = serde_json::from_str(&lines[1]).unwrap();
        assert_eq!(last["done"], true);
    }

    #[tokio::test]
    async fn streaming_handles_multiple_events_in_one_chunk() {
        let mut blob = String::new();
        blob.push_str(&sse_chunk("a", None));
        blob.push_str(&sse_chunk("b", Some("stop")));
        blob.push_str("data: [DONE]\n\n");
        let upstream = stream::iter(vec![Ok::<Bytes, std::io::Error>(Bytes::from(blob))]);
        let out = translate_streaming_response(upstream, "m".to_string());
        let lines = collect(out).await;
        assert_eq!(lines.len(), 3, "got {lines:?}");
        let v0: Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(v0["message"]["content"], "a");
        let v1: Value = serde_json::from_str(&lines[1]).unwrap();
        assert_eq!(v1["message"]["content"], "b");
        let v2: Value = serde_json::from_str(&lines[2]).unwrap();
        assert_eq!(v2["done"], true);
        assert_eq!(v2["done_reason"], "stop");
    }

    #[tokio::test]
    async fn streaming_eof_without_done_still_closes() {
        // No `[DONE]` sentinel — upstream just stops. The adapter must
        // still emit a final done:true frame so the client doesn't hang.
        let chunks = vec![Ok::<Bytes, std::io::Error>(Bytes::from(sse_chunk(
            "x",
            Some("length"),
        )))];
        let upstream = stream::iter(chunks);
        let out = translate_streaming_response(upstream, "m".to_string());
        let lines = collect(out).await;
        assert_eq!(lines.len(), 2);
        let last: Value = serde_json::from_str(&lines[1]).unwrap();
        assert_eq!(last["done"], true);
        assert_eq!(last["done_reason"], "length");
    }

    #[tokio::test]
    async fn streaming_skips_empty_priming_delta() {
        // OpenAI sometimes opens a stream with an empty content delta to
        // announce the role. That carries no token for the Ollama client,
        // so the adapter must drop it (otherwise clients see a confusing
        // empty NDJSON frame with done:false).
        let prime = r#"data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}"#;
        let chunks = vec![
            Ok::<Bytes, std::io::Error>(Bytes::from(format!("{prime}\n\n"))),
            Ok(Bytes::from(sse_chunk("x", Some("stop")))),
            Ok(Bytes::from("data: [DONE]\n\n")),
        ];
        let upstream = stream::iter(chunks);
        let out = translate_streaming_response(upstream, "m".to_string());
        let lines = collect(out).await;
        assert_eq!(lines.len(), 2, "expected priming delta dropped: {lines:?}");
        let v: Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(v["message"]["content"], "x");
    }

    #[tokio::test]
    async fn streaming_picks_up_trailing_usage_chunk() {
        // With stream_options.include_usage the final chunk has empty
        // content but populated usage. The adapter must thread that into
        // the terminal done:true frame.
        let usage_chunk = r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":3}}"#;
        let chunks = vec![
            Ok::<Bytes, std::io::Error>(Bytes::from(sse_chunk("hi", None))),
            Ok(Bytes::from(format!("{usage_chunk}\n\n"))),
            Ok(Bytes::from("data: [DONE]\n\n")),
        ];
        let upstream = stream::iter(chunks);
        let out = translate_streaming_response(upstream, "m".to_string());
        let lines = collect(out).await;
        let last: Value = serde_json::from_str(lines.last().unwrap()).unwrap();
        assert_eq!(last["done"], true);
        assert_eq!(last["prompt_eval_count"], 7);
        assert_eq!(last["eval_count"], 3);
    }

    // ── helpers ─────────────────────────────────────────────────────────

    #[test]
    fn format_rfc3339_utc_known_value() {
        // 2026-05-21T00:00:00Z
        assert_eq!(format_rfc3339_utc(1_779_321_600), "2026-05-21T00:00:00Z");
        // 1970-01-01T00:00:00Z
        assert_eq!(format_rfc3339_utc(0), "1970-01-01T00:00:00Z");
        // 2000-01-01T00:00:00Z = 946684800
        assert_eq!(format_rfc3339_utc(946_684_800), "2000-01-01T00:00:00Z");
        // 2024-02-29T12:34:56Z — leap day
        assert_eq!(format_rfc3339_utc(1_709_210_096), "2024-02-29T12:34:56Z");
    }

    #[test]
    fn find_double_newline_basic() {
        assert_eq!(find_double_newline(b"abc\n\ndef"), Some(3));
        assert_eq!(find_double_newline(b"no break here\n"), None);
        assert_eq!(find_double_newline(b""), None);
        assert_eq!(find_double_newline(b"\n\n"), Some(0));
    }
}

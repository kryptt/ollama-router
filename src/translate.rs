//! Protocol translation between Ollama-native `/api/chat` and OpenAI
//! `/v1/chat/completions`.
//!
//! Used when the registry says a backend hosting the requested model only
//! speaks the OpenAI dialect (e.g. llama-swap, llama.cpp's openai-compat
//! server). The router rewrites the upstream URL to `/v1/chat/completions`,
//! translates the request body, and reshapes the response — either a single
//! JSON object or an SSE stream — back into the Ollama-native NDJSON / JSON
//! shape the client expects.
//!
//! It also carries the one translation that is not about protocol at all:
//! `reasoning_effort` -> llama.cpp's `chat_template_kwargs.enable_thinking`
//! and `thinking_budget_tokens`. See `reasoning_effort_to_llamacpp`.

use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use futures_util::Stream;
use serde_json::{Value, json};

/// A valid RFC 3339 timestamp. The field is required by Ollama clients but
/// the actual value is not used for anything that matters here, so a fixed
/// value avoids pulling in a date crate. Canonical home for the epoch
/// sentinel; `heartbeat` and `registry` reuse this constant.
pub(crate) const FIXED_TIMESTAMP: &str = "1970-01-01T00:00:00Z";

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
                // tolerates unknown fields. Root fields win a collision (see
                // `request_preserves_root_field_collisions_under_options`):
                // an explicit root value is caller intent, `options` is not.
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

/// Overwrite the body's `model` field with `model`. Used when routing
/// rewrote the requested model (escalation or backend-down fallback): the
/// upstream must see the rewritten name, not the one the client sent —
/// hosted backends 404 on unknown IDs and local ones would silently load
/// the wrong model.
pub fn rewrite_model_field(bytes: &[u8], model: &str) -> Result<Vec<u8>, serde_json::Error> {
    let mut root: Value = serde_json::from_slice(bytes)?;
    if let Some(obj) = root.as_object_mut() {
        obj.insert("model".to_string(), Value::String(model.to_string()));
    }
    serde_json::to_vec(&root)
}

// ---------------------------------------------------------------------------
// reasoning_effort -> llama.cpp
// ---------------------------------------------------------------------------

/// How much thinking each `reasoning_effort` level buys, in tokens.
///
/// `reasoning_effort: none|minimal|low|medium|high` is the spelling OpenAI,
/// Zed, Open WebUI, pi and freellmapi all speak. llama.cpp's llama-server
/// accepts per-request thinking control too — but under different names:
/// `chat_template_kwargs.enable_thinking` (server-common.cpp) and
/// `thinking_budget_tokens`. It ignores `reasoning_effort` outright. This
/// table is the bridge, and it is the reason effort no longer has to be
/// encoded in a model name.
///
/// Budgets are deliberately coarse — the point is a usable gradient, not a
/// calibrated one.
const EFFORT_BUDGETS: [(&str, u64); 4] = [
    ("minimal", 512),
    ("low", 2048),
    ("medium", 8192),
    ("high", 32768),
];

/// Fold the spellings clients actually send onto the canonical scale.
/// Mirrors freellmapi's `EFFORT_ALIASES` (`server/src/lib/sampling-params.ts`)
/// so the two hops in a chain agree on what a word means.
fn canonical_effort(raw: &str) -> Option<&'static str> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "none" | "off" | "disabled" => Some("none"),
        "minimal" | "min" | "lowest" | "xlow" => Some("minimal"),
        "low" => Some("low"),
        "medium" | "mid" | "balanced" | "normal" => Some("medium"),
        "high" | "max" | "maximum" | "highest" | "ultra" | "xhigh" => Some("high"),
        // "auto"/"adaptive" and anything unrecognised: no opinion. The knob
        // is dropped and the server's own default stands, which is what
        // freellmapi does and the only behaviour that can't surprise anyone.
        _ => None,
    }
}

/// Read the client's requested effort out of the body, removing whichever
/// spelling carried it. Accepts the flat `reasoning_effort: "high"` and the
/// object form `reasoning: { effort: "high" }`; the object wrapper is never
/// forwarded, because llama.cpp would ignore it and a hosted backend might
/// not.
///
/// Returns `None` when the body asked for nothing, which is the signal to
/// leave the request byte-identical.
fn take_requested_effort(obj: &mut serde_json::Map<String, Value>) -> Option<String> {
    let flat = obj
        .remove("reasoning_effort")
        .and_then(|v| v.as_str().map(str::to_string));

    // Remove the wrapper regardless of whether it carried a usable effort:
    // forwarding a half-understood `reasoning` object is worse than dropping
    // it, since backends disagree on its shape.
    let nested = match obj.remove("reasoning") {
        Some(Value::Object(r)) => r.get("effort").and_then(|v| v.as_str().map(str::to_string)),
        _ => None,
    };

    flat.or(nested)
}

/// Translate `reasoning_effort` into the fields llama.cpp actually reads.
///
/// Returns `Ok(None)` when the body carried no reasoning request at all — the
/// caller then forwards the original bytes untouched, so a request that never
/// mentions reasoning costs nothing but the parse.
///
/// Only ever called for llama-swap-family backends. `nous` and `freellmapi`
/// both understand `reasoning_effort` natively, so their bodies pass through
/// this module untouched.
pub fn reasoning_effort_to_llamacpp(bytes: &[u8]) -> Result<Option<Vec<u8>>, serde_json::Error> {
    let mut root: Value = serde_json::from_slice(bytes)?;
    let Some(obj) = root.as_object_mut() else {
        return Ok(None);
    };

    let Some(requested) = take_requested_effort(obj) else {
        return Ok(None);
    };

    if let Some(effort) = canonical_effort(&requested) {
        let budget = EFFORT_BUDGETS
            .iter()
            .find(|(name, _)| *name == effort)
            .map(|(_, tokens)| *tokens);

        // A client that spelled out llama.cpp's own fields meant it. The
        // coarse level never overrides the precise knob.
        let max_tokens = client_max_tokens(obj);
        let kwargs = obj
            .entry("chat_template_kwargs")
            .or_insert_with(|| json!({}));
        if let Some(kwargs) = kwargs.as_object_mut() {
            kwargs
                .entry("enable_thinking")
                .or_insert(Value::Bool(budget.is_some()));
        }

        if let Some(budget) = budget {
            // ponytail: flat clamp against the client's own output budget.
            // llama.cpp does no clamping itself, so an unclamped budget can
            // swallow the whole generation. If a tiny `max_tokens` plus deep
            // thinking ever starves an answer in practice, split the budget
            // instead of capping it.
            let budget = max_tokens.map_or(budget, |max| budget.min(max));
            obj.entry("thinking_budget_tokens")
                .or_insert(Value::from(budget));
        }
    }

    serde_json::to_vec(&root).map(Some)
}

/// The client's own output-token ceiling, under either spelling OpenAI has
/// used. `max_completion_tokens` is the newer name and wins where both are
/// present, matching llama.cpp's own precedence (server-task.cpp).
fn client_max_tokens(obj: &serde_json::Map<String, Value>) -> Option<u64> {
    obj.get("max_completion_tokens")
        .or_else(|| obj.get("max_tokens"))
        .and_then(Value::as_u64)
        .filter(|n| *n > 0)
}

/// Build the common Ollama message frame shared by streaming deltas and
/// non-streaming responses.  Callers merge additional fields (done,
/// usage counters, etc.) into the returned object.
fn ollama_message_frame(model: &str, content: &Value) -> Value {
    json!({
        "model": model,
        "created_at": current_timestamp(),
        "message": { "role": "assistant", "content": content },
    })
}

/// Build a terminal `done:true` Ollama frame with timing stubs and token
/// counts.  Shared by non-streaming `openai_chat_to_ollama_response` and
/// the streaming adapter's `final_frame`.
fn done_frame(
    model: &str,
    content: &Value,
    finish_reason: Value,
    prompt_tokens: u64,
    completion_tokens: u64,
) -> Value {
    let mut out = ollama_message_frame(model, content);
    let frame = out
        .as_object_mut()
        .unwrap_or_else(|| unreachable!("ollama_message_frame always returns an object"));
    frame.insert("done".into(), json!(true));
    frame.insert("done_reason".into(), finish_reason);
    frame.insert("total_duration".into(), json!(0u64));
    frame.insert("load_duration".into(), json!(0u64));
    frame.insert("prompt_eval_count".into(), json!(prompt_tokens));
    frame.insert("prompt_eval_duration".into(), json!(0u64));
    frame.insert("eval_count".into(), json!(completion_tokens));
    frame.insert("eval_duration".into(), json!(0u64));
    out
}

/// Extract a `u64` token count from an optional usage object.
///
/// `extract_token_count(resp.get("usage"), "prompt_tokens")` is the canonical
/// form; returns `0` when the field is absent or non-numeric.
fn extract_token_count(usage: Option<&Value>, key: &str) -> u64 {
    usage
        .and_then(|u| u.get(key))
        .and_then(|v| v.as_u64())
        .unwrap_or(0)
}

/// Extract the `content` string from the first choice of an OpenAI response or
/// delta chunk.
///
/// `msg_field` is `"message"` for non-streaming responses and `"delta"` for
/// streaming chunks. Returns an empty-string `Value` when absent.
fn extract_content_from_choice(resp: &Value, msg_field: &str) -> Value {
    resp.get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get(msg_field))
        .and_then(|m| m.get("content"))
        .cloned()
        .unwrap_or_else(|| Value::String(String::new()))
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
    let content = extract_content_from_choice(&resp, "message");
    let finish_reason = choice
        .and_then(|c| c.get("finish_reason"))
        .cloned()
        .unwrap_or(Value::Null);

    let usage = resp.get("usage");
    let prompt_tokens = extract_token_count(usage, "prompt_tokens");
    let completion_tokens = extract_token_count(usage, "completion_tokens");

    let out = done_frame(
        model_name,
        &content,
        finish_reason,
        prompt_tokens,
        completion_tokens,
    );
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
        let content = extract_content_from_choice(&value, "delta");

        if let Some(fr) = choice.and_then(|c| c.get("finish_reason"))
            && !fr.is_null()
        {
            self.finish_reason = fr.clone();
        }

        if let Some(usage) = value.get("usage") {
            let p = extract_token_count(Some(usage), "prompt_tokens");
            if p > 0 {
                self.prompt_tokens = p;
            }
            let c = extract_token_count(Some(usage), "completion_tokens");
            if c > 0 {
                self.completion_tokens = c;
            }
        }

        // Only emit a frame if there's actual content. An empty delta (the
        // OpenAI "role: assistant" priming chunk, or a usage-only final
        // chunk) carries no token for the Ollama client.
        let has_content = content.as_str().is_some_and(|s| !s.is_empty());
        if has_content {
            let mut chunk = ollama_message_frame(&self.model, &content);
            chunk
                .as_object_mut()
                .unwrap_or_else(|| unreachable!("ollama_message_frame always returns an object"))
                .insert("done".into(), json!(false));
            let line = format!("{chunk}\n");
            self.out.push_back(Bytes::from(line));
        }
    }

    /// Build the terminal `done:true` Ollama frame from accumulated stats.
    fn final_frame(&self) -> Bytes {
        let empty = Value::String(String::new());
        let out = done_frame(
            &self.model,
            &empty,
            self.finish_reason.clone(),
            self.prompt_tokens,
            self.completion_tokens,
        );
        Bytes::from(format!("{out}\n"))
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
pub fn format_rfc3339_utc(secs: i64) -> String {
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

/// Translate an OpenAI `/v1/images/generations` body into the chat-completions
/// image protocol (`modalities: ["image","text"]`) for backends flagged
/// `images_via_chat`. `size` (`"WxH"`) becomes `image_config.aspect_ratio`;
/// the remaining images-only knobs (`n`, `quality`, `style`,
/// `response_format`) have no chat equivalent and are dropped.
pub fn images_request_to_chat(bytes: &[u8]) -> Result<Vec<u8>, serde_json::Error> {
    let req: Value = serde_json::from_slice(bytes)?;
    let prompt = req
        .get("prompt")
        .cloned()
        .unwrap_or_else(|| Value::String(String::new()));
    let mut out = json!({
        "model": req.get("model").cloned().unwrap_or(Value::Null),
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["image", "text"],
        "stream": false,
    });
    if let Some(ratio) = req
        .get("size")
        .and_then(Value::as_str)
        .and_then(aspect_ratio)
    {
        out["image_config"] = json!({ "aspect_ratio": ratio });
    }
    serde_json::to_vec(&out)
}

/// `"1024x1536"` -> `"2:3"`. `None` for anything that is not two positive
/// integers (including OpenAI's `"auto"`).
fn aspect_ratio(size: &str) -> Option<String> {
    let (w, h) = size.split_once('x')?;
    let (w, h): (u64, u64) = (w.trim().parse().ok()?, h.trim().parse().ok()?);
    if w == 0 || h == 0 {
        return None;
    }
    let (mut a, mut b) = (w, h);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    Some(format!("{}:{}", w / a, h / a))
}

/// Reshape a chat-completions image reply (`choices[].message.images[]
/// .image_url.url`) into an `/v1/images/generations` response. Data URLs
/// become `b64_json`; plain URLs stay `url`. `Ok(None)` when the reply
/// carries no image at all (the model answered in text only).
pub fn chat_response_to_images(bytes: &[u8]) -> Result<Option<Vec<u8>>, serde_json::Error> {
    let resp: Value = serde_json::from_slice(bytes)?;
    let data: Vec<Value> = resp
        .get("choices")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|c| c.pointer("/message/images").and_then(Value::as_array))
        .flatten()
        .filter_map(|img| img.pointer("/image_url/url").and_then(Value::as_str))
        .map(|url| match url.split_once(";base64,") {
            Some((prefix, b64)) if prefix.starts_with("data:") => json!({ "b64_json": b64 }),
            _ => json!({ "url": url }),
        })
        .collect();
    if data.is_empty() {
        return Ok(None);
    }
    let created = resp.get("created").cloned().unwrap_or_else(|| {
        json!(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs())
        )
    });
    serde_json::to_vec(&json!({ "created": created, "data": data })).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;
    use futures_util::stream;
    use serde_json::Map;

    const SSE_DONE: &str = "data: [DONE]\n\n";

    fn parse_request(input: &[u8]) -> Value {
        let out = ollama_chat_to_openai_request(input).unwrap();
        serde_json::from_slice(&out).unwrap()
    }

    #[test]
    fn rewrite_model_field_overwrites_only_model() {
        let input = br#"{"model":"qwen3.6-medium","messages":[{"role":"user","content":"hi"}],"stream":true}"#;
        let out = rewrite_model_field(input, "qwen/qwen3.8-27b").unwrap();
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["model"], "qwen/qwen3.8-27b");
        assert_eq!(v["stream"], true);
        assert_eq!(v["messages"][0]["content"], "hi");
    }

    // ── reasoning_effort → llama.cpp ─────────────────────────────────────

    /// Translate and re-parse. `None` means the body was left byte-identical.
    fn parse_effort(input: &[u8]) -> Option<Value> {
        reasoning_effort_to_llamacpp(input)
            .unwrap()
            .map(|out| serde_json::from_slice(&out).unwrap())
    }

    fn body_with_effort(effort: &str) -> Vec<u8> {
        format!(r#"{{"model":"qwen3.6-high","reasoning_effort":"{effort}","messages":[{{"role":"user","content":"hi"}}]}}"#)
            .into_bytes()
    }

    #[test]
    fn absent_reasoning_leaves_the_body_alone() {
        let input = br#"{"model":"qwen3.6-high","messages":[{"role":"user","content":"hi"}]}"#;
        assert!(parse_effort(input).is_none());
    }

    #[test]
    fn each_effort_level_maps_to_its_budget() {
        for (effort, budget) in [
            ("minimal", 512),
            ("low", 2048),
            ("medium", 8192),
            ("high", 32768),
        ] {
            let v = parse_effort(&body_with_effort(effort)).expect("body rewritten");
            assert_eq!(
                v["chat_template_kwargs"]["enable_thinking"], true,
                "{effort} should enable thinking"
            );
            assert_eq!(
                v["thinking_budget_tokens"], budget,
                "wrong budget for {effort}"
            );
            assert!(
                v.get("reasoning_effort").is_none(),
                "{effort}: llama.cpp ignores reasoning_effort; it must not survive"
            );
        }
    }

    #[test]
    fn effort_none_disables_thinking_without_a_budget() {
        let v = parse_effort(&body_with_effort("none")).expect("body rewritten");
        assert_eq!(v["chat_template_kwargs"]["enable_thinking"], false);
        assert!(v.get("thinking_budget_tokens").is_none());
    }

    #[test]
    fn off_scale_spellings_fold_onto_the_canonical_scale() {
        for (spelling, budget) in [
            ("MAX", 32768),
            ("xhigh", 32768),
            ("balanced", 8192),
            ("  Lowest  ", 512),
        ] {
            let v = parse_effort(&body_with_effort(spelling)).expect("body rewritten");
            assert_eq!(
                v["thinking_budget_tokens"], budget,
                "wrong budget for {spelling}"
            );
        }
        let v = parse_effort(&body_with_effort("off")).expect("body rewritten");
        assert_eq!(v["chat_template_kwargs"]["enable_thinking"], false);
    }

    #[test]
    fn unrecognised_effort_is_dropped_not_rejected() {
        // "auto"/"adaptive"/garbage: no opinion. The knob goes away and the
        // server's own default stands — never a 400.
        for spelling in ["auto", "adaptive", "banana"] {
            let v = parse_effort(&body_with_effort(spelling)).expect("body rewritten");
            assert!(
                v.get("reasoning_effort").is_none(),
                "{spelling} must not survive"
            );
            assert!(
                v.get("chat_template_kwargs").is_none(),
                "{spelling} must inject nothing"
            );
            assert!(
                v.get("thinking_budget_tokens").is_none(),
                "{spelling} must inject nothing"
            );
            assert_eq!(v["model"], "qwen3.6-high");
        }
    }

    #[test]
    fn object_form_is_accepted_and_the_wrapper_never_forwarded() {
        let input = br#"{"model":"qwen3.6-high","reasoning":{"effort":"high","summary":"auto"}}"#;
        let v = parse_effort(input).expect("body rewritten");
        assert_eq!(v["thinking_budget_tokens"], 32768);
        assert!(v.get("reasoning").is_none(), "the wrapper must be dropped");
    }

    #[test]
    fn flat_form_wins_over_the_object_form() {
        let input = br#"{"model":"q","reasoning_effort":"low","reasoning":{"effort":"high"}}"#;
        let v = parse_effort(input).expect("body rewritten");
        assert_eq!(v["thinking_budget_tokens"], 2048);
        assert!(v.get("reasoning").is_none());
    }

    #[test]
    fn budget_is_clamped_to_the_clients_own_output_ceiling() {
        let input = br#"{"model":"q","reasoning_effort":"high","max_tokens":1000,"messages":[]}"#;
        let v = parse_effort(input).expect("body rewritten");
        assert_eq!(v["thinking_budget_tokens"], 1000);

        // max_completion_tokens is the newer spelling and wins where both are set.
        let input =
            br#"{"model":"q","reasoning_effort":"high","max_tokens":9000,"max_completion_tokens":700}"#;
        let v = parse_effort(input).expect("body rewritten");
        assert_eq!(v["thinking_budget_tokens"], 700);
    }

    #[test]
    fn an_explicit_llamacpp_knob_beats_the_coarse_level() {
        let input = br#"{"model":"q","reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":true},"thinking_budget_tokens":99}"#;
        let v = parse_effort(input).expect("body rewritten");
        assert_eq!(v["chat_template_kwargs"]["enable_thinking"], true);
        assert_eq!(v["thinking_budget_tokens"], 99);
    }

    #[test]
    fn existing_chat_template_kwargs_survive() {
        let input =
            br#"{"model":"q","reasoning_effort":"low","chat_template_kwargs":{"custom":"keep"}}"#;
        let v = parse_effort(input).expect("body rewritten");
        assert_eq!(v["chat_template_kwargs"]["custom"], "keep");
        assert_eq!(v["chat_template_kwargs"]["enable_thinking"], true);
    }

    #[test]
    fn every_other_body_field_survives_the_rewrite() {
        let input = br#"{"model":"qwen3.6-high","reasoning_effort":"medium","stream":true,"temperature":0.3,"tools":[{"type":"function"}],"messages":[{"role":"user","content":"hi"}]}"#;
        let v = parse_effort(input).expect("body rewritten");
        assert_eq!(v["model"], "qwen3.6-high");
        assert_eq!(v["stream"], true);
        assert_eq!(v["temperature"], 0.3);
        assert_eq!(v["tools"][0]["type"], "function");
        assert_eq!(v["messages"][0]["content"], "hi");
    }

    #[test]
    fn a_non_object_body_is_left_alone() {
        assert!(parse_effort(b"[1,2,3]").is_none());
        assert!(reasoning_effort_to_llamacpp(b"not json").is_err());
    }

    fn parse_response(input: &[u8], model: &str) -> Value {
        serde_json::from_slice(&openai_chat_to_ollama_response(input, model).unwrap()).unwrap()
    }

    fn assert_done_frame(lines: &[String], index: usize, expected_reason: &str) {
        let frame = parse_ndjson_line(&lines[index]);
        assert_eq!(frame["done"], true, "expected done=true at line {index}");
        assert_eq!(
            frame["done_reason"], expected_reason,
            "wrong done_reason at line {index}"
        );
    }

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
        let v = parse_request(input);

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
        let v = parse_request(input);
        assert_eq!(v["format"], "json");
        assert_eq!(v["tools"][0]["function"]["name"], "f");
        assert_eq!(v["tool_choice"], "auto");
    }

    #[test]
    fn request_with_no_options_is_a_no_op_aside_from_keep_alive() {
        let input = br#"{"model": "x", "messages": [], "keep_alive": "1h"}"#;
        let v = parse_request(input);
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
        let v = parse_request(input);
        assert_eq!(v["custom"], "root_value");
    }

    // ── non-streaming response translation ───────────────────────────────

    #[test]
    fn response_non_streaming_basic() {
        let input = br#"{
            "id": "chatcmpl-x",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "glm-4.7-flash",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello world"},
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
        }"#;
        let v = parse_response(input, "glm-4.7-flash");

        assert_eq!(v["model"], "glm-4.7-flash", "response model");
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
        let v = parse_response(input, "m");
        assert_eq!(v["prompt_eval_count"], 0);
        assert_eq!(v["eval_count"], 0);
    }

    #[test]
    fn response_missing_choices_yields_empty_content() {
        let v = parse_response(b"{}", "m");
        assert_eq!(v["message"]["content"], "");
        assert_eq!(v["done"], true);
    }

    // ── streaming translation ────────────────────────────────────────────

    fn parse_ndjson_line(line: &str) -> Value {
        serde_json::from_str(line).unwrap()
    }

    async fn run_translation(chunks: Vec<Result<Bytes, std::io::Error>>) -> Vec<String> {
        let upstream = stream::iter(chunks);
        let out = translate_streaming_response(upstream, "m".to_string());
        collect(out).await
    }

    /// Build SSE chunks from event strings, append the `[DONE]` sentinel, and
    /// run the full translation pipeline in one call.
    async fn translate_sse(events: Vec<Result<Bytes, std::io::Error>>) -> Vec<String> {
        let mut chunks = events;
        chunks.push(Ok(Bytes::from(SSE_DONE)));
        run_translation(chunks).await
    }

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
        let lines = translate_sse(vec![
            sse_ok("hello", None),
            sse_ok(" ", None),
            sse_ok("world", Some("stop")),
        ])
        .await;

        // 3 deltas with content + 1 final done:true frame
        assert_eq!(lines.len(), 4, "got {} lines: {lines:?}", lines.len());

        for (i, expected_content) in ["hello", " ", "world"].iter().enumerate() {
            let v = parse_ndjson_line(&lines[i]);
            assert_eq!(v["done"], false);
            assert_eq!(v["model"], "m");
            assert_eq!(v["message"]["role"], "assistant");
            assert_eq!(v["message"]["content"], *expected_content);
        }
        assert_done_frame(&lines, 3, "stop");
    }

    #[tokio::test]
    async fn streaming_handles_split_mid_event() {
        // Split a single SSE event across multiple Bytes chunks to ensure
        // the buffer joins them correctly before parsing.
        let event = sse_chunk("hello", None);
        let split = event.len() / 2;
        let lines = translate_sse(vec![
            Ok::<Bytes, std::io::Error>(Bytes::from(event[..split].to_string())),
            Ok(Bytes::from(event[split..].to_string())),
        ])
        .await;

        assert_eq!(parse_ndjson_line(&lines[0])["message"]["content"], "hello");
        assert_eq!(
            parse_ndjson_line(&lines[1])["done"],
            true,
            "split-mid-event"
        );
    }

    #[tokio::test]
    async fn streaming_handles_multiple_events_in_one_chunk() {
        // All events arrive in a single Bytes chunk, including [DONE].
        let blob = format!(
            "{}{}{SSE_DONE}",
            sse_chunk("a", None),
            sse_chunk("b", Some("stop"))
        );
        let lines = run_translation(vec![Ok::<Bytes, std::io::Error>(Bytes::from(blob))]).await;
        assert_eq!(lines.len(), 3, "got {lines:?}");
        assert_eq!(parse_ndjson_line(&lines[0])["message"]["content"], "a");
        assert_eq!(parse_ndjson_line(&lines[1])["message"]["content"], "b");
        assert_done_frame(&lines, 2, "stop");
    }

    #[tokio::test]
    async fn streaming_eof_without_done_still_closes() {
        // No [DONE] sentinel -- upstream just stops. Adapter must emit done:true.
        let lines = translate_sse(vec![sse_ok("x", Some("length"))]).await;
        assert_eq!(lines.len(), 2, "eof-without-done");
        assert_done_frame(&lines, 1, "length");
    }

    #[tokio::test]
    async fn streaming_skips_empty_priming_delta() {
        // OpenAI sometimes opens a stream with an empty content delta to
        // announce the role. That carries no token for the Ollama client,
        // so the adapter must drop it (otherwise clients see a confusing
        // empty NDJSON frame with done:false).
        let prime = r#"data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}"#;
        let lines = translate_sse(vec![
            Ok::<Bytes, std::io::Error>(Bytes::from(format!("{prime}\n\n"))),
            sse_ok("x", Some("stop")),
        ])
        .await;
        assert_eq!(lines.len(), 2, "expected priming delta dropped: {lines:?}");
        assert_eq!(parse_ndjson_line(&lines[0])["message"]["content"], "x");
    }

    #[tokio::test]
    async fn streaming_picks_up_trailing_usage_chunk() {
        // With stream_options.include_usage the final chunk has empty
        // content but populated usage. The adapter must thread that into
        // the terminal done:true frame.
        let usage_chunk = r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":3}}"#;
        let lines = translate_sse(vec![
            sse_ok("hi", None),
            Ok(Bytes::from(format!("{usage_chunk}\n\n"))),
        ])
        .await;
        let last = parse_ndjson_line(lines.last().unwrap());
        assert_eq!(last["done"], true);
        assert_eq!(last["prompt_eval_count"], 7);
        assert_eq!(last["eval_count"], 3);
    }

    fn sse_ok(content: &str, finish: Option<&str>) -> Result<Bytes, std::io::Error> {
        Ok(Bytes::from(sse_chunk(content, finish)))
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

    // ── images via chat ──────────────────────────────────────────────────

    #[test]
    fn images_request_becomes_chat_with_modalities() {
        let out = images_request_to_chat(
            br#"{"model":"google/gemini-nano-banana-2.1","prompt":"a red apple","n":1,"size":"1024x1536","response_format":"b64_json"}"#,
        )
        .unwrap();
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["model"], "google/gemini-nano-banana-2.1");
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["messages"][0]["content"], "a red apple");
        assert_eq!(v["modalities"], json!(["image", "text"]));
        assert_eq!(v["stream"], false);
        assert_eq!(v["image_config"]["aspect_ratio"], "2:3");
        assert!(v.get("n").is_none() && v.get("size").is_none() && v.get("prompt").is_none());
    }

    #[test]
    fn images_request_without_parseable_size_has_no_aspect_ratio() {
        for body in [
            &br#"{"model":"m","prompt":"p"}"#[..],
            br#"{"model":"m","prompt":"p","size":"auto"}"#,
        ] {
            let v: Value = serde_json::from_slice(&images_request_to_chat(body).unwrap()).unwrap();
            assert!(v.get("image_config").is_none(), "{v}");
        }
    }

    #[test]
    fn chat_image_reply_becomes_images_response() {
        let out = chat_response_to_images(
            br#"{"created":42,"choices":[{"message":{"content":"here","images":[
                {"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}},
                {"type":"image_url","image_url":{"url":"https://cdn.example/x.png"}}]}}]}"#,
        )
        .unwrap()
        .unwrap();
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["created"], 42);
        assert_eq!(
            v["data"],
            json!([{"b64_json": "AAAA"}, {"url": "https://cdn.example/x.png"}])
        );
    }

    #[test]
    fn chat_text_only_reply_is_none() {
        let out =
            chat_response_to_images(br#"{"choices":[{"message":{"content":"I cannot draw"}}]}"#)
                .unwrap();
        assert!(out.is_none());
    }
}

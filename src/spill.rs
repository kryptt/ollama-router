use std::io;
use std::pin::Pin;

use axum::body::Body;
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use tokio::io::{AsyncSeekExt, AsyncWriteExt, SeekFrom};
use tokio_util::io::ReaderStream;

/// Parsed routing fields extracted from the spilled request body.
pub struct SpillResult {
    pub model: String,
    /// Effective stream flag. Falls back to the `default_stream` passed
    /// to `spill_and_detect` when the JSON body omits the field — that
    /// default differs by wire protocol (Ollama=true, OpenAI=false).
    pub stream: bool,
    /// A streaming body that replays the spilled prefix from disk, then
    /// continues with any remaining chunks from the original request.
    pub body: Body,
}

/// Read chunks from an axum `Body`, write every byte to a temp file, and
/// scan on-the-fly for the `"model"` and `"stream"` JSON fields.  Once the
/// model is detected (or the body ends), return a zero-copy replay stream
/// that concatenates the on-disk prefix with the still-arriving tail.
///
/// `default_stream` is the value used when the body has no `stream` field:
/// pass `true` for Ollama-protocol paths (`/api/chat`, `/api/generate`),
/// `false` for OpenAI- and Anthropic-protocol paths (`/v1/*`). Mismatching
/// the default causes non-streaming clients to receive SSE heartbeats
/// prepended to what they expect to be a single JSON body.
///
/// Returns `Ok(None)` when the body contains no usable `model` field.
pub async fn spill_and_detect(
    body: Body,
    default_stream: bool,
) -> Result<Option<SpillResult>, io::Error> {
    let mut stream = body.into_data_stream();
    let std_file = tempfile::tempfile()?;
    let mut file = tokio::fs::File::from_std(std_file);
    let mut scanner = Scanner::new();
    let mut body_done = false;

    // Phase 1 — spill to disk while scanning for the model field.
    loop {
        match stream.next().await {
            Some(Ok(chunk)) => {
                file.write_all(&chunk).await?;
                scanner.feed(&chunk);
                if scanner.model().is_some() && scanner.stream_value().is_some() {
                    break;
                }
            }
            Some(Err(e)) => return Err(io::Error::other(e)),
            None => {
                body_done = true;
                break;
            }
        }
    }

    let model = match scanner.model() {
        Some(m) => m.to_owned(),
        None => return Ok(None),
    };
    let stream_flag = scanner.stream_value().unwrap_or(default_stream);

    // Phase 2 — rewind the file and build a combined stream:
    //   [disk replay] ++ [remaining body chunks]
    file.flush().await?;
    file.seek(SeekFrom::Start(0)).await?;

    let replay = ReaderStream::new(file).map(|r| r.map_err(axum::Error::new));

    let combined: Pin<Box<dyn Stream<Item = Result<Bytes, axum::Error>> + Send>> = if body_done {
        Box::pin(replay)
    } else {
        Box::pin(replay.chain(stream))
    };

    Ok(Some(SpillResult {
        model,
        stream: stream_flag,
        body: Body::from_stream(combined),
    }))
}

// ---------------------------------------------------------------------------
// Byte-level JSON field scanner
// ---------------------------------------------------------------------------

/// A minimal, zero-allocation state machine that scans a byte stream for
/// `"model": "<value>"` and `"stream": true|false` without buffering the
/// entire body.  Only the extracted value strings are heap-allocated.
///
/// Limitations (all acceptable for Ollama API traffic):
/// - Does not track JSON nesting, so a `"model":` inside a nested object
///   would match.  Ollama requests are flat.
/// - Does not handle Unicode-escaped key names (`\u006dodel`).
/// - Assumes model names contain no escaped quotes.
pub(crate) struct Scanner {
    state: State,
    model: Option<String>,
    stream_value: Option<bool>,
    buf: Vec<u8>,
}

#[derive(Clone, Copy)]
enum State {
    /// Scanning for `"` that opens a potential key.
    Idle,
    /// Just saw an opening `"` — check the first character to decide
    /// whether this string could be `"model"` or `"stream"`.
    QuoteOpen,
    /// Inside a JSON string we are not interested in — skip to close-quote.
    SkipString { escaped: bool },

    // ---- matching "model" key ----
    /// Saw `"m`, matching remaining chars of `odel"`.
    MatchModel(u8),
    /// Saw `"model"`, skipping whitespace before `:`.
    ModelColon,
    /// Saw `:`, skipping whitespace before value `"`.
    ModelQuote,
    /// Inside model value string, accumulating into `buf`.
    ModelValue { escaped: bool },

    // ---- matching "stream" key ----
    /// Saw `"s`, matching remaining chars of `tream"`.
    MatchStream(u8),
    /// Saw `"stream"`, skipping whitespace before `:`.
    StreamColon,
    /// Saw `:`, skipping whitespace before `t`/`f`.
    StreamBool,
    /// Matching a boolean literal tail (`rue` or `alse`) at position `pos`
    /// (0-indexed into the tail slice). `value` is the result written to
    /// `stream_value` on a full match.
    StreamBoolTail { pos: u8, value: bool },
}

/// Tail bytes for boolean literals, indexed by `StreamBoolTail.value`.
const BOOL_TAILS: [&[u8]; 2] = [b"alse", b"rue"];

// Lookup tables for the suffixes we match after the opening `"` + first char.
const MODEL_SUFFIX: &[u8] = b"odel\""; // after "m
const STREAM_SUFFIX: &[u8] = b"tream\""; // after "s

impl Scanner {
    pub fn new() -> Self {
        Self {
            state: State::Idle,
            model: None,
            stream_value: None,
            buf: Vec::with_capacity(64),
        }
    }

    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    pub fn stream_value(&self) -> Option<bool> {
        self.stream_value
    }

    /// Feed a chunk of bytes into the scanner.
    pub fn feed(&mut self, data: &[u8]) {
        for &b in data {
            // Short-circuit once we have both fields.
            if self.model.is_some() && self.stream_value.is_some() {
                return;
            }
            self.step(b);
        }
    }

    fn step(&mut self, b: u8) {
        self.state = match self.state {
            // -- top-level scanning -------------------------------------------
            State::Idle => {
                if b == b'"' {
                    State::QuoteOpen
                } else {
                    State::Idle
                }
            }

            // The very first character after `"` decides our path.
            State::QuoteOpen => match b {
                b'm' if self.model.is_none() => State::MatchModel(0),
                b's' if self.stream_value.is_none() => State::MatchStream(0),
                b'"' => State::Idle, // empty string `""`
                _ => Self::enter_skip_string(b),
            },

            State::SkipString { escaped: true } => State::SkipString { escaped: false },
            State::SkipString { escaped: false } => match b {
                b'"' => State::Idle,
                _ => Self::enter_skip_string(b),
            },

            // -- "model" key matching -----------------------------------------
            State::MatchModel(pos) => {
                Self::match_key_suffix(b, pos, MODEL_SUFFIX, State::ModelColon, State::MatchModel)
            }

            State::ModelColon | State::StreamColon => match b {
                b':' => match self.state {
                    State::ModelColon => State::ModelQuote,
                    State::StreamColon => State::StreamBool,
                    _ => unreachable!(),
                },
                b if b.is_ascii_whitespace() => self.state,
                _ => State::Idle,
            },

            State::ModelQuote => match b {
                b'"' => {
                    self.buf.clear();
                    State::ModelValue { escaped: false }
                }
                b if b.is_ascii_whitespace() => State::ModelQuote,
                _ => State::Idle, // value isn't a string
            },

            State::ModelValue { escaped } => match (escaped, b) {
                (false, b'\\') => {
                    self.buf.push(b);
                    State::ModelValue { escaped: true }
                }
                (false, b'"') => {
                    self.model = String::from_utf8(self.buf.clone())
                        .ok()
                        .filter(|s| !s.is_empty());
                    State::Idle
                }
                _ => {
                    self.buf.push(b);
                    State::ModelValue { escaped: false }
                }
            },

            // -- "stream" key matching ----------------------------------------
            State::MatchStream(pos) => Self::match_key_suffix(
                b,
                pos,
                STREAM_SUFFIX,
                State::StreamColon,
                State::MatchStream,
            ),

            State::StreamBool => match b {
                b't' => State::StreamBoolTail {
                    pos: 0,
                    value: true,
                },
                b'f' => State::StreamBoolTail {
                    pos: 0,
                    value: false,
                },
                b if b.is_ascii_whitespace() => State::StreamBool,
                _ => State::Idle,
            },

            State::StreamBoolTail { pos, value } => {
                let tail = BOOL_TAILS[value as usize];
                if (pos as usize) < tail.len() && b == tail[pos as usize] {
                    if pos as usize == tail.len() - 1 {
                        self.stream_value = Some(value);
                        State::Idle
                    } else {
                        State::StreamBoolTail {
                            pos: pos + 1,
                            value,
                        }
                    }
                } else {
                    State::Idle
                }
            }
        };
    }

    /// Try to advance a key-suffix match by one byte. On a match, returns
    /// `on_complete` (suffix fully consumed) or `on_advance(pos + 1)`. On a
    /// mismatch, recovers into `SkipString` or `Idle`.
    fn match_key_suffix(
        b: u8,
        pos: u8,
        suffix: &[u8],
        on_complete: State,
        on_advance: fn(u8) -> State,
    ) -> State {
        if b == suffix[pos as usize] {
            if pos as usize == suffix.len() - 1 {
                on_complete
            } else {
                on_advance(pos + 1)
            }
        } else if b == b'"' {
            State::Idle
        } else {
            Self::enter_skip_string(b)
        }
    }

    /// Produce the appropriate `SkipString` state for a byte that is not a
    /// close-quote. Handles the `\\` -> escaped transition.
    fn enter_skip_string(b: u8) -> State {
        State::SkipString {
            escaped: b == b'\\',
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Scanner unit tests --------------------------------------------------

    fn scan(input: &[u8]) -> Scanner {
        let mut s = Scanner::new();
        s.feed(input);
        s
    }

    /// Assert that scanning `input` produces the expected model and stream.
    fn assert_scan(input: &[u8], model: Option<&str>, stream: Option<bool>) {
        let s = scan(input);
        assert_eq!(s.model(), model, "model mismatch");
        assert_eq!(s.stream_value(), stream, "stream mismatch");
    }

    #[test]
    fn basic_model_extraction() {
        // absent stream -> caller defaults true
        assert_scan(
            br#"{"model": "llama3", "messages": []}"#,
            Some("llama3"),
            None,
        );
    }

    #[test]
    fn model_and_stream_true() {
        assert_scan(
            br#"{"model": "glm-4.7-flash", "stream": true}"#,
            Some("glm-4.7-flash"),
            Some(true),
        );
    }

    #[test]
    fn model_and_stream_false() {
        assert_scan(
            br#"{"model": "glm-4.7-flash", "stream": false}"#,
            Some("glm-4.7-flash"),
            Some(false),
        );
    }

    #[test]
    fn stream_before_model() {
        assert_scan(
            br#"{"stream": false, "model": "codellama"}"#,
            Some("codellama"),
            Some(false),
        );
    }

    #[test]
    fn model_with_slashes_and_colons() {
        assert_scan(
            br#"{"model": "fixt/home-3b-v3:latest"}"#,
            Some("fixt/home-3b-v3:latest"),
            None,
        );
    }

    #[test]
    fn whitespace_around_colon() {
        assert_scan(br#"{ "model"  :  "test-model" }"#, Some("test-model"), None);
    }

    #[test]
    fn model_value_in_prior_string_ignored() {
        // "model" appears as a value before appearing as a key.
        // The scanner resets after `"model"` (value) since the next token is `,` not `:`.
        assert_scan(
            br#"{"type": "model", "model": "actual"}"#,
            Some("actual"),
            None,
        );
    }

    #[test]
    fn escaped_quotes_in_value_before_model() {
        assert_scan(
            br#"{"prompt": "say \"hello\"", "model": "phi3"}"#,
            Some("phi3"),
            None,
        );
    }

    #[test]
    fn no_model_field() {
        assert_scan(br#"{"prompt": "hello"}"#, None, None);
    }

    #[test]
    fn empty_model_value() {
        assert_scan(br#"{"model": ""}"#, None, None);
    }

    #[test]
    fn chunks_split_across_key() {
        let mut s = Scanner::new();
        // Split right in the middle of "model"
        s.feed(br#"{"mod"#);
        s.feed(br#"el": "chunked"}"#);
        assert_eq!(s.model(), Some("chunked"));
    }

    #[test]
    fn chunks_split_across_value() {
        let mut s = Scanner::new();
        s.feed(br#"{"model": "lla"#);
        s.feed(br#"ma3"}"#);
        assert_eq!(s.model(), Some("llama3"));
    }

    #[test]
    fn chunks_split_across_stream_bool() {
        let mut s = Scanner::new();
        s.feed(br#"{"model": "x", "stream": tr"#);
        s.feed(br#"ue}"#);
        assert_eq!(s.stream_value(), Some(true));
    }

    #[test]
    fn openai_compat_format() {
        assert_scan(
            br#"{"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}"#,
            Some("gpt-4"),
            None,
        );
    }

    #[test]
    fn model_not_confused_by_substring() {
        // "remodel" contains "model" but isn't the key
        assert_scan(br#"{"remodel": "no", "model": "yes"}"#, Some("yes"), None);
    }

    // -- spill_and_detect integration tests ----------------------------------

    /// Run `spill_and_detect` on a JSON byte-slice body and assert model + stream.
    async fn assert_spill(
        json: &[u8],
        default_stream: bool,
        model: &str,
        stream: bool,
    ) -> SpillResult {
        let body = Body::from(json.to_vec());
        assert_spill_body(body, default_stream, model, stream).await
    }

    /// Run `spill_and_detect` on an arbitrary `Body` and assert model + stream.
    async fn assert_spill_body(
        body: Body,
        default_stream: bool,
        model: &str,
        stream: bool,
    ) -> SpillResult {
        let result = spill_and_detect(body, default_stream)
            .await
            .expect("spill_and_detect IO error")
            .expect("expected Some(SpillResult)");
        assert_eq!(result.model, model, "model mismatch");
        assert_eq!(result.stream, stream, "stream mismatch");
        result
    }

    /// Build a streaming `Body` from pre-split byte chunks.
    fn body_from_chunks(chunks: Vec<Bytes>) -> Body {
        let io_chunks: Vec<Result<Bytes, std::io::Error>> = chunks.into_iter().map(Ok).collect();
        Body::from_stream(futures_util::stream::iter(io_chunks))
    }

    /// Collect a `Body` into a `Vec<u8>`.
    async fn collect_body(body: Body) -> Vec<u8> {
        use http_body_util::BodyExt;
        body.collect()
            .await
            .expect("failed to collect body")
            .to_bytes()
            .to_vec()
    }

    #[tokio::test]
    async fn spill_small_body() {
        let json = br#"{"model": "test-model", "stream": false, "messages": []}"#;
        let result = assert_spill(json, true, "test-model", false).await;
        // The replayed body should contain the full original bytes.
        assert_eq!(collect_body(result.body).await, json);
    }

    #[tokio::test]
    async fn spill_no_model_returns_none() {
        let body = Body::from(br#"{"prompt": "hello"}"#.to_vec());
        assert!(spill_and_detect(body, true).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn spill_chunked_body() {
        // Simulate a chunked body where model spans two chunks.
        let body = body_from_chunks(vec![
            Bytes::from(r#"{"model": "chu"#),
            Bytes::from(r#"nked", "stream": true}"#),
        ]);
        let result = assert_spill_body(body, true, "chunked", true).await;
        assert_eq!(
            collect_body(result.body).await,
            br#"{"model": "chunked", "stream": true}"#
        );
    }

    #[tokio::test]
    async fn spill_large_tail_after_model() {
        // Model appears early; a large payload follows.
        let prefix = br#"{"model": "early", "data": ""#;
        let tail = "x".repeat(1024 * 64); // 64 KiB of payload
        let suffix = br#""}"#;

        let body = body_from_chunks(vec![
            Bytes::from(prefix.to_vec()),
            Bytes::from(tail.clone()),
            Bytes::from(suffix.to_vec()),
        ]);
        let result = assert_spill_body(body, true, "early", true).await;

        let mut expected = prefix.to_vec();
        expected.extend(tail.as_bytes());
        expected.extend(suffix);
        assert_eq!(collect_body(result.body).await, expected);
    }

    #[tokio::test]
    async fn spill_model_before_stream_false() {
        // Regression: LLMVision puts "model" before "stream": false with a
        // large messages array in between.  The old break condition exited as
        // soon as model was found, missing the stream field entirely and
        // defaulting to stream=true.
        assert_spill(
            br#"{"model": "llmvision/glimpse-v1:latest", "messages": [{"role": "user", "content": "describe"}], "stream": false}"#,
            true,
            "llmvision/glimpse-v1:latest",
            false,
        )
        .await;
    }

    #[tokio::test]
    async fn spill_stream_absent_takes_ollama_default() {
        // Ollama-protocol paths (/api/chat, /api/generate) default to
        // stream=true when the field is absent.
        assert_spill(br#"{"model": "test", "messages": []}"#, true, "test", true).await;
    }

    #[tokio::test]
    async fn spill_stream_absent_takes_openai_default() {
        // OpenAI-protocol paths (/v1/chat/completions, /v1/completions,
        // /v1/messages) default to stream=false when the field is absent.
        // Regression: hermes title_generator omits the field on
        // non-streaming requests; the old fixed default of true engaged
        // the SSE heartbeat path and corrupted the response.
        assert_spill(
            br#"{"model": "test", "messages": []}"#,
            false,
            "test",
            false,
        )
        .await;
    }
}

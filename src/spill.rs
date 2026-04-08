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
    /// Ollama defaults `stream` to `true` when the field is absent.
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
/// Returns `Ok(None)` when the body contains no usable `model` field.
pub async fn spill_and_detect(body: Body) -> Result<Option<SpillResult>, io::Error> {
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
                if scanner.model().is_some() {
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
    let stream_flag = scanner.stream_value().unwrap_or(true);

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
    /// Matching `true` (pos 1..4) or `false` (pos 1..5).
    StreamTrue(u8),
    StreamFalse(u8),
}

// Lookup tables for the suffixes we match after the opening `"` + first char.
const MODEL_SUFFIX: &[u8] = b"odel\"";   // after "m
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
                b'\\' => State::SkipString { escaped: true },
                _ => State::SkipString { escaped: false },
            },

            State::SkipString { escaped: true } => State::SkipString { escaped: false },
            State::SkipString { escaped: false } => match b {
                b'\\' => State::SkipString { escaped: true },
                b'"' => State::Idle,
                _ => State::SkipString { escaped: false },
            },

            // -- "model" key matching -----------------------------------------
            State::MatchModel(pos) => {
                if b == MODEL_SUFFIX[pos as usize] {
                    if pos as usize == MODEL_SUFFIX.len() - 1 {
                        // Fully matched `"model"`, now expect `:`
                        State::ModelColon
                    } else {
                        State::MatchModel(pos + 1)
                    }
                } else {
                    // Mismatch — resume skipping this string.
                    self.recover_skip(b)
                }
            }

            State::ModelColon => match b {
                b':' => State::ModelQuote,
                b if b.is_ascii_whitespace() => State::ModelColon,
                _ => State::Idle, // wasn't a key after all
            },

            State::ModelQuote => match b {
                b'"' => {
                    self.buf.clear();
                    State::ModelValue { escaped: false }
                }
                b if b.is_ascii_whitespace() => State::ModelQuote,
                _ => State::Idle, // value isn't a string
            },

            State::ModelValue { escaped: true } => {
                self.buf.push(b);
                State::ModelValue { escaped: false }
            }
            State::ModelValue { escaped: false } => match b {
                b'\\' => {
                    self.buf.push(b);
                    State::ModelValue { escaped: true }
                }
                b'"' => {
                    self.model = String::from_utf8(self.buf.clone()).ok().filter(|s| !s.is_empty());
                    State::Idle
                }
                _ => {
                    self.buf.push(b);
                    State::ModelValue { escaped: false }
                }
            },

            // -- "stream" key matching ----------------------------------------
            State::MatchStream(pos) => {
                if b == STREAM_SUFFIX[pos as usize] {
                    if pos as usize == STREAM_SUFFIX.len() - 1 {
                        State::StreamColon
                    } else {
                        State::MatchStream(pos + 1)
                    }
                } else {
                    self.recover_skip(b)
                }
            }

            State::StreamColon => match b {
                b':' => State::StreamBool,
                b if b.is_ascii_whitespace() => State::StreamColon,
                _ => State::Idle,
            },

            State::StreamBool => match b {
                b't' => State::StreamTrue(1),
                b'f' => State::StreamFalse(1),
                b if b.is_ascii_whitespace() => State::StreamBool,
                _ => State::Idle,
            },

            State::StreamTrue(pos) => {
                const TRUE_TAIL: &[u8] = b"rue";
                let idx = (pos - 1) as usize;
                if idx < TRUE_TAIL.len() && b == TRUE_TAIL[idx] {
                    if idx == TRUE_TAIL.len() - 1 {
                        self.stream_value = Some(true);
                        State::Idle
                    } else {
                        State::StreamTrue(pos + 1)
                    }
                } else {
                    State::Idle
                }
            }

            State::StreamFalse(pos) => {
                const FALSE_TAIL: &[u8] = b"alse";
                let idx = (pos - 1) as usize;
                if idx < FALSE_TAIL.len() && b == FALSE_TAIL[idx] {
                    if idx == FALSE_TAIL.len() - 1 {
                        self.stream_value = Some(false);
                        State::Idle
                    } else {
                        State::StreamFalse(pos + 1)
                    }
                } else {
                    State::Idle
                }
            }
        };
    }

    /// When a key-match fails mid-string, figure out where to resume.
    fn recover_skip(&self, b: u8) -> State {
        if b == b'"' {
            // The failing byte is the close-quote of this string.
            State::Idle
        } else if b == b'\\' {
            State::SkipString { escaped: true }
        } else {
            State::SkipString { escaped: false }
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

    #[test]
    fn basic_model_extraction() {
        let s = scan(br#"{"model": "llama3", "messages": []}"#);
        assert_eq!(s.model(), Some("llama3"));
        assert_eq!(s.stream_value(), None); // absent → caller defaults true
    }

    #[test]
    fn model_and_stream_true() {
        let s = scan(br#"{"model": "glm-4.7-flash", "stream": true}"#);
        assert_eq!(s.model(), Some("glm-4.7-flash"));
        assert_eq!(s.stream_value(), Some(true));
    }

    #[test]
    fn model_and_stream_false() {
        let s = scan(br#"{"model": "qwen3.5:35b", "stream": false}"#);
        assert_eq!(s.model(), Some("qwen3.5:35b"));
        assert_eq!(s.stream_value(), Some(false));
    }

    #[test]
    fn stream_before_model() {
        let s = scan(br#"{"stream": false, "model": "codellama"}"#);
        assert_eq!(s.model(), Some("codellama"));
        assert_eq!(s.stream_value(), Some(false));
    }

    #[test]
    fn model_with_slashes_and_colons() {
        let s = scan(br#"{"model": "fixt/home-3b-v3:latest"}"#);
        assert_eq!(s.model(), Some("fixt/home-3b-v3:latest"));
    }

    #[test]
    fn whitespace_around_colon() {
        let s = scan(br#"{ "model"  :  "test-model" }"#);
        assert_eq!(s.model(), Some("test-model"));
    }

    #[test]
    fn model_value_in_prior_string_ignored() {
        // "model" appears as a value before appearing as a key
        let s = scan(br#"{"type": "model", "model": "actual"}"#);
        // The scanner may pick up the first "model" but it won't find a ":"
        // after the string "model" (value) because the next token is `,`.
        // So it resets and finds the real key.
        assert_eq!(s.model(), Some("actual"));
    }

    #[test]
    fn escaped_quotes_in_value_before_model() {
        let s = scan(br#"{"prompt": "say \"hello\"", "model": "phi3"}"#);
        assert_eq!(s.model(), Some("phi3"));
    }

    #[test]
    fn no_model_field() {
        let s = scan(br#"{"prompt": "hello"}"#);
        assert_eq!(s.model(), None);
    }

    #[test]
    fn empty_model_value() {
        let s = scan(br#"{"model": ""}"#);
        assert_eq!(s.model(), None);
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
        let s = scan(br#"{"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}"#);
        assert_eq!(s.model(), Some("gpt-4"));
    }

    #[test]
    fn model_not_confused_by_substring() {
        // "remodel" contains "model" but isn't the key
        let s = scan(br#"{"remodel": "no", "model": "yes"}"#);
        assert_eq!(s.model(), Some("yes"));
    }

    // -- spill_and_detect integration tests ----------------------------------

    #[tokio::test]
    async fn spill_small_body() {
        let json = br#"{"model": "test-model", "stream": false, "messages": []}"#;
        let body = Body::from(json.to_vec());
        let result = spill_and_detect(body).await.unwrap().unwrap();
        assert_eq!(result.model, "test-model");
        assert!(!result.stream);

        // The replayed body should contain the full original bytes.
        let collected = collect_body(result.body).await;
        assert_eq!(collected, json);
    }

    #[tokio::test]
    async fn spill_no_model_returns_none() {
        let body = Body::from(br#"{"prompt": "hello"}"#.to_vec());
        assert!(spill_and_detect(body).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn spill_chunked_body() {
        // Simulate a chunked body where model spans two chunks.
        let chunks: Vec<Result<Bytes, std::io::Error>> = vec![
            Ok(Bytes::from(r#"{"model": "chu"#)),
            Ok(Bytes::from(r#"nked", "stream": true}"#)),
        ];
        let stream = futures_util::stream::iter(chunks);
        let body = Body::from_stream(stream);

        let result = spill_and_detect(body).await.unwrap().unwrap();
        assert_eq!(result.model, "chunked");
        assert!(result.stream);

        let collected = collect_body(result.body).await;
        assert_eq!(collected, br#"{"model": "chunked", "stream": true}"#);
    }

    #[tokio::test]
    async fn spill_large_tail_after_model() {
        // Model appears early; a large payload follows.
        let prefix = br#"{"model": "early", "data": ""#;
        let tail = "x".repeat(1024 * 64); // 64 KiB of payload
        let suffix = br#""}"#;

        let chunks: Vec<Result<Bytes, std::io::Error>> = vec![
            Ok(Bytes::from(prefix.to_vec())),
            Ok(Bytes::from(tail.clone())),
            Ok(Bytes::from(suffix.to_vec())),
        ];
        let stream = futures_util::stream::iter(chunks);
        let body = Body::from_stream(stream);

        let result = spill_and_detect(body).await.unwrap().unwrap();
        assert_eq!(result.model, "early");

        let collected = collect_body(result.body).await;
        let mut expected = prefix.to_vec();
        expected.extend(tail.as_bytes());
        expected.extend(suffix);
        assert_eq!(collected, expected);
    }

    /// Helper: collect a Body into bytes.
    async fn collect_body(body: Body) -> Vec<u8> {
        use http_body_util::BodyExt;
        let bytes = body
            .collect()
            .await
            .expect("failed to collect body")
            .to_bytes();
        bytes.to_vec()
    }
}

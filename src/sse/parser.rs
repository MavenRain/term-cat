//! SSE parser implementation.  See `sse::mod` for the high-level intent.
//!
//! Each unfold step does one of:
//!
//!   * Pop a buffered event (`buf` non-empty).
//!   * Observe the cancel signal and end the stream.
//!   * Read one line, classify it, and either end the stream, recurse with
//!     updated state, or accumulate events for emission on subsequent steps.
//!
//! Recursion is via `Io::flat_map`; `comp-cat-rs` evaluates the resulting
//! continuation chain iteratively in `Io::run`, so we do not stack-overflow
//! on long runs of blank lines or empty chunks.

use std::io::{BufRead, BufReader};
use std::sync::Arc;

use comp_cat_rs::effect::io::Io;
use comp_cat_rs::effect::stream::Stream;
use ureq::BodyReader;

use crate::agent::ChatEvent;
use crate::bridge::CancelObserver;
use crate::error::{Error, SseError};
use crate::wire::ChatCompletionChunk;

/// Buffered-reader type used throughout the parser.  ureq's `BodyReader<'static>`
/// is the owned form returned by `Body::into_reader`; concrete static type so
/// no `dyn Read` appears in this crate.
pub type SseReader = BufReader<BodyReader<'static>>;

/// State threaded through each `Stream::unfold` step.  Public so a higher-level
/// streaming pipeline (e.g. `LocalOpenAiCompletion::stream_chat`) can hold an
/// `SseState` inside its own enum and reuse `step` directly.
pub struct SseState {
    reader: SseReader,
    buf: Vec<ChatEvent>,
    cancel: CancelObserver,
}

/// Type alias for the boxed closure used by `Stream::unfold` to drive SSE
/// parsing.  Factored out to keep clippy's `type_complexity` happy.
type StepFn = Arc<dyn Fn(SseState) -> Io<Error, Option<(ChatEvent, SseState)>> + Send + Sync>;

impl SseState {
    /// Build an initial state from an open reader and a cancel observer.
    #[must_use]
    pub fn new(reader: SseReader, cancel: CancelObserver) -> Self {
        Self {
            reader,
            buf: Vec::new(),
            cancel,
        }
    }

    /// Consume the state and return the cancel observer.  Used by the agent
    /// loop when transitioning out of the streaming phase between tool
    /// dispatches so the same observer continues to gate the next HTTP call.
    #[must_use]
    pub fn into_cancel(self) -> CancelObserver {
        self.cancel
    }
}

/// Possible classifications of one SSE line.
#[derive(Debug)]
enum LineKind {
    /// `data: [DONE]` sentinel: stream is finished.
    Done,
    /// `data: {json}` payload, already parsed.
    Data(ChatCompletionChunk),
    /// Blank line, SSE comment, or unrecognized prefix; skip and read again.
    SkipLine,
    /// `read_line` returned 0 bytes; stream is finished.
    Eof,
}

/// Turn a streaming response body into a `Stream<Error, ChatEvent>`.  The
/// stream observes `cancel` between line reads and terminates cleanly on
/// signal.
#[must_use]
pub fn parse(reader: SseReader, cancel: CancelObserver) -> Stream<Error, ChatEvent> {
    let init = SseState::new(reader, cancel);
    let step_fn: StepFn = Arc::new(step);
    Stream::unfold(init, step_fn)
}

/// One unfold step: emit the next buffered event, or read another line from
/// the underlying SSE body.  Exported so higher-level streams can compose it.
#[must_use]
pub fn step(state: SseState) -> Io<Error, Option<(ChatEvent, SseState)>> {
    let SseState {
        reader,
        buf,
        cancel,
    } = state;
    match buf.as_slice() {
        [head, tail @ ..] => {
            let head = head.clone();
            let tail = tail.to_vec();
            Io::pure(Some((
                head,
                SseState {
                    reader,
                    buf: tail,
                    cancel,
                },
            )))
        }
        [] => sse_read_more(reader, cancel),
    }
}

fn sse_read_more(
    reader: SseReader,
    cancel: CancelObserver,
) -> Io<Error, Option<(ChatEvent, SseState)>> {
    if cancel.is_cancelled() {
        Io::pure(None)
    } else {
        Io::suspend(move || read_one_line(reader))
            .flat_map(move |(kind, reader)| handle_kind(kind, reader, cancel))
    }
}

fn handle_kind(
    kind: LineKind,
    reader: SseReader,
    cancel: CancelObserver,
) -> Io<Error, Option<(ChatEvent, SseState)>> {
    match kind {
        LineKind::Eof | LineKind::Done => Io::pure(None),
        LineKind::SkipLine => step(SseState {
            reader,
            buf: Vec::new(),
            cancel,
        }),
        LineKind::Data(chunk) => chunk.into_events().map_or_else(
            |e| Io::suspend(move || Err(e)),
            |events| {
                step(SseState {
                    reader,
                    buf: events,
                    cancel,
                })
            },
        ),
    }
}

/// Read one SSE line and classify it.  `let mut` is the FFI-boundary
/// carve-out: `BufRead::read_line` requires `&mut self` for both the reader
/// and the destination `String`.
fn read_one_line(mut reader: SseReader) -> Result<(LineKind, SseReader), Error> {
    let mut buf = String::new();
    let n = reader
        .read_line(&mut buf)
        .map_err(|e| Error::Sse(SseError::Io(e)))?;
    let kind = if n == 0 {
        LineKind::Eof
    } else {
        classify(&buf)?
    };
    Ok((kind, reader))
}

fn classify(line: &str) -> Result<LineKind, Error> {
    let trimmed = line.trim_end_matches(['\n', '\r']);
    if trimmed.is_empty() {
        Ok(LineKind::SkipLine)
    } else {
        classify_non_empty(trimmed)
    }
}

fn classify_non_empty(line: &str) -> Result<LineKind, Error> {
    line.strip_prefix("data:")
        .map_or(Ok(LineKind::SkipLine), |payload| {
            let payload = payload.trim_start();
            if payload == "[DONE]" {
                Ok(LineKind::Done)
            } else {
                serde_json::from_str(payload)
                    .map(LineKind::Data)
                    .map_err(|e| Error::Sse(SseError::Decode(e)))
            }
        })
}

#[cfg(test)]
mod tests {
    use super::{LineKind, classify};
    use crate::error::{Error, SseError};

    fn must_skip(kind: &LineKind, label: &str) -> Result<(), String> {
        match kind {
            LineKind::SkipLine => Ok(()),
            LineKind::Done => Err(format!("{label}: expected SkipLine, got Done")),
            LineKind::Eof => Err(format!("{label}: expected SkipLine, got Eof")),
            LineKind::Data(_) => Err(format!("{label}: expected SkipLine, got Data")),
        }
    }

    fn must_done(kind: &LineKind, label: &str) -> Result<(), String> {
        match kind {
            LineKind::Done => Ok(()),
            LineKind::SkipLine => Err(format!("{label}: expected Done, got SkipLine")),
            LineKind::Eof => Err(format!("{label}: expected Done, got Eof")),
            LineKind::Data(_) => Err(format!("{label}: expected Done, got Data")),
        }
    }

    fn must_data(kind: &LineKind, label: &str) -> Result<(), String> {
        match kind {
            LineKind::Data(_) => Ok(()),
            LineKind::SkipLine => Err(format!("{label}: expected Data, got SkipLine")),
            LineKind::Done => Err(format!("{label}: expected Data, got Done")),
            LineKind::Eof => Err(format!("{label}: expected Data, got Eof")),
        }
    }

    #[test]
    fn blank_line_is_skip() -> Result<(), String> {
        let kind = classify("").map_err(|e| format!("classify: {e:?}"))?;
        must_skip(&kind, "blank")
    }

    #[test]
    fn only_newline_is_skip() -> Result<(), String> {
        let kind = classify("\n").map_err(|e| format!("classify: {e:?}"))?;
        must_skip(&kind, "only-newline")
    }

    #[test]
    fn crlf_only_is_skip() -> Result<(), String> {
        let kind = classify("\r\n").map_err(|e| format!("classify: {e:?}"))?;
        must_skip(&kind, "crlf")
    }

    #[test]
    fn comment_line_is_skip() -> Result<(), String> {
        let kind = classify(": this is a heartbeat\n").map_err(|e| format!("classify: {e:?}"))?;
        must_skip(&kind, "comment")
    }

    #[test]
    fn id_prefix_is_skip() -> Result<(), String> {
        let kind = classify("id: 42\n").map_err(|e| format!("classify: {e:?}"))?;
        must_skip(&kind, "id-prefix")
    }

    #[test]
    fn event_prefix_is_skip() -> Result<(), String> {
        let kind = classify("event: ping\n").map_err(|e| format!("classify: {e:?}"))?;
        must_skip(&kind, "event-prefix")
    }

    #[test]
    fn data_done_with_space_is_done() -> Result<(), String> {
        let kind = classify("data: [DONE]\n").map_err(|e| format!("classify: {e:?}"))?;
        must_done(&kind, "data: [DONE]")
    }

    #[test]
    fn data_done_without_space_is_done() -> Result<(), String> {
        let kind = classify("data:[DONE]\n").map_err(|e| format!("classify: {e:?}"))?;
        must_done(&kind, "data:[DONE]")
    }

    #[test]
    fn data_with_valid_json_is_data() -> Result<(), String> {
        let kind = classify("data: {\"choices\":[]}\n").map_err(|e| format!("classify: {e:?}"))?;
        must_data(&kind, "data: {choices}")
    }

    #[test]
    fn data_with_invalid_json_returns_sse_decode_error() -> Result<(), String> {
        let result = classify("data: {this is not json}\n");
        match result {
            Ok(kind) => Err(format!("expected Err, got Ok({kind:?})")),
            Err(Error::Sse(SseError::Decode(_))) => Ok(()),
            Err(other) => Err(format!("expected SseError::Decode, got {other:?}")),
        }
    }
}

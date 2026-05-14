//! Hand-rolled, hierarchical error enum.
//!
//! Every wrappable source error has an explicit subsystem.  We do not provide a
//! single `From<std::io::Error> for Error` because `std::io::Error` can come
//! from the TUI subsystem, the SSE reader, or the bridge fiber, and conflating
//! them throws away the call-site context.  Each call site explicitly maps to
//! the right subsystem before bubbling up.

use std::fmt;

/// The single project-wide error type.  Every fallible function in `term-cat`
/// returns `Result<T, Error>`.
#[derive(Debug)]
pub enum Error {
    /// Surfaced from `rig-cat` (provider, tool invocation, embedding).
    RigCat(rig_cat::error::Error),
    /// Server-Sent Events parsing or transport failure.
    Sse(SseError),
    /// Wire-format (de)serialization failure.
    Wire(WireError),
    /// Terminal, render, or input failure.
    Tui(TuiError),
    /// Inter-thread plumbing failure (fiber spawn / join / channel).
    Bridge(BridgeError),
    /// HTTP transport or status failure from the local OpenAI-compatible server.
    Provider(ProviderError),
}

/// SSE-specific failures.
#[derive(Debug)]
pub enum SseError {
    /// Underlying socket / file IO failed while reading the SSE body.
    Io(std::io::Error),
    /// A `data:` line did not parse as a valid `ChatCompletionChunk`.
    Decode(serde_json::Error),
    /// The line was structurally unexpected (not `data:`, not blank, not `[DONE]`).
    UnexpectedLine(String),
}

/// Wire-format failures (request encode, response decode, enum tag mismatches).
#[derive(Debug)]
pub enum WireError {
    /// Failed to serialize a request to JSON.
    Encode(serde_json::Error),
    /// Failed to deserialize a JSON response.
    Decode(serde_json::Error),
    /// A `role` field carried a string we do not recognize.
    UnknownRole(String),
    /// A `finish_reason` field carried a string we do not recognize.
    UnknownFinishReason(String),
}

/// TUI failures (terminal setup, draw, raw mode toggle).
#[derive(Debug)]
pub enum TuiError {
    /// Underlying terminal IO failed.
    Io(std::io::Error),
}

/// Bridge / fiber / channel failures.
#[derive(Debug)]
pub enum BridgeError {
    /// `Fiber::fork` could not spawn its OS thread.
    FiberSpawn(std::io::Error),
    /// The producer fiber panicked; the captured message follows.
    FiberPanicked(String),
    /// The mpsc receiver was dropped before the producer finished.
    SendDisconnected,
}

/// HTTP / status failures from the local OpenAI-compatible server.
#[derive(Debug)]
pub enum ProviderError {
    /// Underlying HTTP transport failed.
    Http(ureq::Error),
    /// The server returned a non-2xx status.
    Status { code: u16, body: String },
    /// The non-streaming response body did not deserialize.
    Decode(serde_json::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RigCat(e) => write!(f, "rig-cat: {e:?}"),
            Self::Sse(e) => write!(f, "sse: {e}"),
            Self::Wire(e) => write!(f, "wire: {e}"),
            Self::Tui(e) => write!(f, "tui: {e}"),
            Self::Bridge(e) => write!(f, "bridge: {e}"),
            Self::Provider(e) => write!(f, "provider: {e}"),
        }
    }
}

impl fmt::Display for SseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {e}"),
            Self::Decode(e) => write!(f, "decode: {e}"),
            Self::UnexpectedLine(s) => write!(f, "unexpected SSE line: {s}"),
        }
    }
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Encode(e) => write!(f, "encode: {e}"),
            Self::Decode(e) => write!(f, "decode: {e}"),
            Self::UnknownRole(s) => write!(f, "unknown role: {s}"),
            Self::UnknownFinishReason(s) => write!(f, "unknown finish_reason: {s}"),
        }
    }
}

impl fmt::Display for TuiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {e}"),
        }
    }
}

impl fmt::Display for BridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FiberSpawn(e) => write!(f, "fiber spawn: {e}"),
            Self::FiberPanicked(s) => write!(f, "fiber panicked: {s}"),
            Self::SendDisconnected => write!(f, "channel receiver disconnected"),
        }
    }
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(e) => write!(f, "http: {e}"),
            Self::Status { code, body } => write!(f, "status {code}: {body}"),
            Self::Decode(e) => write!(f, "decode: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::RigCat(_e) => None,
            Self::Sse(e) => Some(e),
            Self::Wire(e) => Some(e),
            Self::Tui(e) => Some(e),
            Self::Bridge(e) => Some(e),
            Self::Provider(e) => Some(e),
        }
    }
}

impl std::error::Error for SseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Decode(e) => Some(e),
            Self::UnexpectedLine(_) => None,
        }
    }
}

impl std::error::Error for WireError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Encode(e) | Self::Decode(e) => Some(e),
            Self::UnknownRole(_) | Self::UnknownFinishReason(_) => None,
        }
    }
}

impl std::error::Error for TuiError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
        }
    }
}

impl std::error::Error for BridgeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::FiberSpawn(e) => Some(e),
            Self::FiberPanicked(_) | Self::SendDisconnected => None,
        }
    }
}

impl std::error::Error for ProviderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Http(_) | Self::Status { .. } => None,
            Self::Decode(e) => Some(e),
        }
    }
}

impl From<rig_cat::error::Error> for Error {
    fn from(e: rig_cat::error::Error) -> Self {
        Self::RigCat(e)
    }
}

impl From<SseError> for Error {
    fn from(e: SseError) -> Self {
        Self::Sse(e)
    }
}

impl From<WireError> for Error {
    fn from(e: WireError) -> Self {
        Self::Wire(e)
    }
}

impl From<TuiError> for Error {
    fn from(e: TuiError) -> Self {
        Self::Tui(e)
    }
}

impl From<BridgeError> for Error {
    fn from(e: BridgeError) -> Self {
        Self::Bridge(e)
    }
}

impl From<ProviderError> for Error {
    fn from(e: ProviderError) -> Self {
        Self::Provider(e)
    }
}

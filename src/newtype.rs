//! Newtypes for every domain primitive.  No raw `String`/`u32`/`f64` crosses a
//! function boundary in this crate when it carries domain meaning.
//!
//! Each newtype owns its data (no borrows; the values are short and Send-ready
//! since they may cross the producer-fiber / TUI-thread boundary via mpsc).

use serde::{Deserialize, Serialize};

/// The base URL of the OpenAI-compatible server, e.g. `http://localhost:1234/v1`.
#[derive(Debug, Clone)]
pub struct BaseUrl(String);

impl BaseUrl {
    /// Construct from any `Into<String>`.  No validation: we trust the source.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// The base URL, without trailing slash normalization.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// The model identifier the server should use, e.g.
/// `supergemma4-26b-uncensored-fast-v2`.
#[derive(Debug, Clone, Serialize)]
pub struct ModelName(String);

impl ModelName {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Bearer token for the OpenAI-compatible server.  LM Studio ignores it, but
/// the field is sent for API parity with the published `OpenAI` spec.
#[derive(Debug, Clone)]
pub struct ApiKey(String);

impl ApiKey {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Sampling temperature, expected to lie in `[0.0, 2.0]`.  No validation here;
/// the server is the source of truth.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Temperature(f64);

impl Temperature {
    #[must_use]
    pub fn new(t: f64) -> Self {
        Self(t)
    }

    #[must_use]
    pub fn value(self) -> f64 {
        self.0
    }
}

/// Maximum tokens the server should generate in a single response.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct MaxTokens(u32);

impl MaxTokens {
    #[must_use]
    pub fn new(n: u32) -> Self {
        Self(n)
    }

    #[must_use]
    pub fn value(self) -> u32 {
        self.0
    }
}

/// The textual body of any chat message.  Wraps a `String` to keep it distinct
/// from arbitrary `String`-typed identifiers (model name, role tag, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageBody(String);

impl MessageBody {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume into the underlying `String`.  Used at the rig-cat boundary
    /// where `StreamChunk::new(String)` takes ownership.
    #[must_use]
    pub fn into_string(self) -> String {
        self.0
    }

    /// Append another body's text, returning a new `MessageBody`.
    #[must_use]
    pub fn concat(self, other: &Self) -> Self {
        Self(self.0 + other.as_str())
    }

    /// Append a `&str`, returning a new `MessageBody`.
    #[must_use]
    pub fn append(self, s: &str) -> Self {
        Self(self.0 + s)
    }

    /// `true` iff the body is the empty string.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl From<String> for MessageBody {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for MessageBody {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

/// Identifier the server assigns to one tool-call within a streaming response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallId(String);

impl ToolCallId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Function name within a tool-call (matches a `ToolDefinition::name()`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolName(String);

impl ToolName {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Position of a tool-call within the `OpenAI` streaming `tool_calls[]` array.
/// Used to key deltas during accumulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ToolCallIndex(u32);

impl ToolCallIndex {
    #[must_use]
    pub fn new(i: u32) -> Self {
        Self(i)
    }

    #[must_use]
    pub fn value(self) -> u32 {
        self.0
    }
}

/// Why the streaming response ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Natural stop / end-of-sequence token.
    Stop,
    /// Hit `max_tokens` ceiling.
    Length,
    /// Server-side content filter triggered.
    ContentFilter,
    /// Response contained `tool_calls`; caller must dispatch and continue.
    ToolCalls,
}

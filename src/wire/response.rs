//! Response body for a non-streaming `POST /v1/chat/completions`.  Only the
//! fields we actually use are modeled; the rest are silently ignored.

use serde::Deserialize;

use crate::newtype::MessageBody;
use crate::wire::message::ToolCall;

/// Top-level response envelope.  Private fields; expose only the accessors
/// term-cat actually needs.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionResponse {
    choices: Vec<ResponseChoice>,
}

impl ChatCompletionResponse {
    /// The content of the first choice's message, if any.  Phase B reads this
    /// to display the assistant turn.
    #[must_use]
    pub fn first_content(&self) -> Option<&MessageBody> {
        self.choices.first().and_then(|c| c.message().content())
    }

    /// The first choice's `tool_calls`, if any.  Phase D reads this to drive
    /// the tool-calling loop when a non-streaming endpoint returns tool calls.
    #[must_use]
    pub fn first_tool_calls(&self) -> Option<&[ToolCall]> {
        self.choices.first().and_then(|c| c.message().tool_calls())
    }

    /// All choices, in server order.
    #[must_use]
    pub fn choices(&self) -> &[ResponseChoice] {
        &self.choices
    }
}

/// One element of `choices[]`.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseChoice {
    message: ResponseMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

impl ResponseChoice {
    /// The assistant turn for this choice.
    #[must_use]
    pub fn message(&self) -> &ResponseMessage {
        &self.message
    }

    /// Server-reported finish reason.
    #[must_use]
    pub fn finish_reason(&self) -> Option<&str> {
        self.finish_reason.as_deref()
    }
}

/// The `message` object inside a response choice.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseMessage {
    #[serde(default)]
    content: Option<MessageBody>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
}

impl ResponseMessage {
    /// Plain assistant text, if any.
    #[must_use]
    pub fn content(&self) -> Option<&MessageBody> {
        self.content.as_ref()
    }

    /// Tool calls, if any.
    #[must_use]
    pub fn tool_calls(&self) -> Option<&[ToolCall]> {
        self.tool_calls.as_deref()
    }
}

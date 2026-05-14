//! Request body for `POST /v1/chat/completions`.

use serde::Serialize;
use serde_json::Value;

use crate::newtype::{MaxTokens, ModelName, Temperature};
use crate::wire::message::Message;

/// Body of a Chat Completions request.  Private fields with serde-tagged
/// `skip_serializing_if` so optional fields are omitted from the JSON when
/// `None`.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionRequest {
    model: ModelName,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<Temperature>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<MaxTokens>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Value>>,
    stream: bool,
}

impl ChatCompletionRequest {
    /// A new non-streaming request with no temperature or token cap.
    #[must_use]
    pub fn new(model: ModelName, messages: Vec<Message>) -> Self {
        Self {
            model,
            messages,
            temperature: None,
            max_tokens: None,
            tools: None,
            stream: false,
        }
    }

    /// Set the streaming flag.  Phase C will flip this to `true`.
    #[must_use]
    pub fn with_stream(self, stream: bool) -> Self {
        Self { stream, ..self }
    }

    /// Set sampling temperature.
    #[must_use]
    pub fn with_temperature(self, t: Temperature) -> Self {
        Self {
            temperature: Some(t),
            ..self
        }
    }

    /// Apply a temperature only if `Some`.  Convenient for builders threading
    /// a provider's optional config through.
    #[must_use]
    pub fn maybe_temperature(self, t: Option<Temperature>) -> Self {
        t.map_or(self.clone(), |x| self.with_temperature(x))
    }

    /// Set max-tokens.
    #[must_use]
    pub fn with_max_tokens(self, n: MaxTokens) -> Self {
        Self {
            max_tokens: Some(n),
            ..self
        }
    }

    /// Apply max-tokens only if `Some`.
    #[must_use]
    pub fn maybe_max_tokens(self, n: Option<MaxTokens>) -> Self {
        n.map_or(self.clone(), |x| self.with_max_tokens(x))
    }

    /// Set the `tools` array.  Used by Phase D when tools are registered.
    #[must_use]
    pub fn with_tools(self, tools: Vec<Value>) -> Self {
        Self {
            tools: Some(tools),
            ..self
        }
    }
}

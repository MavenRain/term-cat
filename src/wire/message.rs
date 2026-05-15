//! Wire-format chat message.  Private fields with serde derives; constructors
//! cover each `Role` variant.

use serde::{Deserialize, Serialize};

use crate::newtype::{MessageBody, ToolCallId, ToolName};

/// Speaker of a chat message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// One JSON-encoded chat message.  Field privacy is preserved by going through
/// the typed constructors below.  Serde reads/writes the private fields by
/// virtue of the derive macro running in the same module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    role: Role,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    content: Option<MessageBody>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    tool_call_id: Option<ToolCallId>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    tool_calls: Option<Vec<ToolCall>>,
}

impl Message {
    /// A system preamble.
    #[must_use]
    pub fn system(body: MessageBody) -> Self {
        Self {
            role: Role::System,
            content: Some(body),
            tool_call_id: None,
            tool_calls: None,
        }
    }

    /// A user prompt.
    #[must_use]
    pub fn user(body: MessageBody) -> Self {
        Self {
            role: Role::User,
            content: Some(body),
            tool_call_id: None,
            tool_calls: None,
        }
    }

    /// An assistant turn containing plain text.
    #[must_use]
    pub fn assistant(body: MessageBody) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(body),
            tool_call_id: None,
            tool_calls: None,
        }
    }

    /// An assistant turn that contains tool calls (no plain content).  Used
    /// by Phase D's tool-calling loop.
    #[must_use]
    pub fn assistant_tool_calls(calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: None,
            tool_call_id: None,
            tool_calls: Some(calls),
        }
    }

    /// An assistant turn that contains both text and tool calls.  Some models
    /// emit explanatory text alongside the tool-call payload; this preserves
    /// that for the wire history.
    #[must_use]
    pub fn assistant_text_and_tool_calls(body: MessageBody, calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(body),
            tool_call_id: None,
            tool_calls: Some(calls),
        }
    }

    /// A tool-result message responding to a prior tool-call.  Used by
    /// Phase D's tool-calling loop.
    #[must_use]
    pub fn tool(call_id: ToolCallId, body: MessageBody) -> Self {
        Self {
            role: Role::Tool,
            content: Some(body),
            tool_call_id: Some(call_id),
            tool_calls: None,
        }
    }

    /// Speaker.
    #[must_use]
    pub fn role(&self) -> Role {
        self.role
    }

    /// Body text, if any.
    #[must_use]
    pub fn content(&self) -> Option<&MessageBody> {
        self.content.as_ref()
    }

    /// Tool-call id (only set for `Role::Tool`).
    #[must_use]
    pub fn tool_call_id(&self) -> Option<&ToolCallId> {
        self.tool_call_id.as_ref()
    }

    /// Tool calls (only set for assistant turns that requested tools).
    #[must_use]
    pub fn tool_calls(&self) -> Option<&[ToolCall]> {
        self.tool_calls.as_deref()
    }
}

/// One tool-call slot within an assistant turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    id: ToolCallId,
    #[serde(rename = "type")]
    kind: ToolCallKind,
    function: ToolCallFunction,
}

impl ToolCall {
    /// Construct a function-type tool call.  `OpenAI`'s spec currently only
    /// defines `function` tool calls; new types would get their own
    /// constructors when added.
    #[must_use]
    pub fn new(id: ToolCallId, name: ToolName, arguments_json: String) -> Self {
        Self {
            id,
            kind: ToolCallKind::Function,
            function: ToolCallFunction {
                name,
                arguments: arguments_json,
            },
        }
    }

    /// Slot id.
    #[must_use]
    pub fn id(&self) -> &ToolCallId {
        &self.id
    }

    /// Discriminator.
    #[must_use]
    pub fn kind(&self) -> ToolCallKind {
        self.kind
    }

    /// Function payload.
    #[must_use]
    pub fn function(&self) -> &ToolCallFunction {
        &self.function
    }
}

/// Discriminator for tool calls; `OpenAI`'s spec only defines `function` today.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallKind {
    Function,
}

/// Function name + JSON-stringified arguments.  Arguments are kept as the raw
/// JSON string per `OpenAI`'s spec; the receiver parses them when invoking the
/// tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    name: ToolName,
    arguments: String,
}

impl ToolCallFunction {
    /// Function name (matches a `ToolDefinition::name()`).
    #[must_use]
    pub fn name(&self) -> &ToolName {
        &self.name
    }

    /// Raw JSON-stringified arguments.
    #[must_use]
    pub fn arguments(&self) -> &str {
        &self.arguments
    }
}

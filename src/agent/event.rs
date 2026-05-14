//! Events that flow between the producer fiber and the TUI thread.

use serde_json::Value;

use crate::newtype::{FinishReason, MessageBody, ToolCallId, ToolCallIndex, ToolName};

/// One unit of progress from the SSE provider stream.  Internal to the
/// producer-fiber pipeline; the TUI receives `AgentEvent` instead.
#[derive(Debug, Clone)]
pub enum ChatEvent {
    /// A token of assistant text.
    Token(MessageBody),
    /// The server has started emitting a new tool-call slot at `index`.
    ToolCallStart(ToolCallIndex, ToolCallId, ToolName),
    /// A partial argument fragment for the tool-call at `index`; to be
    /// appended to the accumulator.
    ToolCallArgs(ToolCallIndex, MessageBody),
    /// The streaming response has ended.
    Finished(FinishReason),
}

/// One unit of progress observable by the UI.  Crosses the producer-fiber /
/// TUI-thread boundary via `std::sync::mpsc::Sender<AgentEvent>`.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// A token of assistant text was streamed.
    AssistantToken(MessageBody),
    /// A tool was invoked with the given arguments.
    ToolInvoked { name: ToolName, args: Value },
    /// A tool returned the given result.
    ToolReturned { name: ToolName, result: Value },
    /// The entire turn (including any tool-call follow-ups) is done.
    TurnDone,
    /// A non-fatal error occurred during the turn; the message is included for
    /// UI display.  We do not propagate the typed `Error` here because
    /// `mpsc::Sender` requires `Send`, and some future inner error sources may
    /// not be `Send + Sync`.
    Failure(MessageBody),
}

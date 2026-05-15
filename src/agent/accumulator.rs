//! Tool-call accumulator for streaming responses.  `OpenAI`'s streaming
//! tool-call protocol spreads each call across multiple SSE chunks: the first
//! chunk for a given `index` carries `id`, `type`, and `function.name`; later
//! chunks carry only `function.arguments` fragments under the same `index`.
//! The accumulator stitches these fragments back into complete `PendingCall`s
//! that the loop dispatches once the server emits `finish_reason: "tool_calls"`.

use crate::newtype::{MessageBody, ToolCallId, ToolCallIndex, ToolName};
use crate::wire::ToolCall;

/// One assembled tool call, ready to dispatch.  `args_buf` is the raw
/// JSON-stringified arguments as concatenated from the stream; the
/// dispatcher parses it with `serde_json::from_str` at invocation time.
#[derive(Debug, Clone)]
pub struct PendingCall {
    id: ToolCallId,
    name: ToolName,
    args_buf: String,
}

impl PendingCall {
    /// Construct a pending call directly.  Used by the accumulator and by
    /// callers that bypass the streaming path (currently none, but exposed
    /// for completeness).
    #[must_use]
    pub fn new(id: ToolCallId, name: ToolName, args_buf: String) -> Self {
        Self { id, name, args_buf }
    }

    /// Server-assigned tool call id.
    #[must_use]
    pub fn id(&self) -> &ToolCallId {
        &self.id
    }

    /// Function name to invoke.
    #[must_use]
    pub fn name(&self) -> &ToolName {
        &self.name
    }

    /// Accumulated JSON-stringified arguments.
    #[must_use]
    pub fn args_buf(&self) -> &str {
        &self.args_buf
    }

    /// Project a `PendingCall` into the wire-format `ToolCall` shape used by
    /// the `assistant` history message that records what the model requested.
    #[must_use]
    pub fn to_wire(&self) -> ToolCall {
        ToolCall::new(self.id.clone(), self.name.clone(), self.args_buf.clone())
    }
}

/// Accumulator over the in-flight tool-call deltas for one assistant turn.
/// Keyed by `ToolCallIndex` (the position in the server's `tool_calls[]`
/// array).  Functional updates: every `add_*` returns a fresh accumulator.
#[derive(Debug, Clone, Default)]
pub struct ToolCallAccumulator {
    pending: Vec<(ToolCallIndex, PendingCall)>,
}

impl ToolCallAccumulator {
    /// Construct an empty accumulator.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Append a `ToolCallStart` delta: a new pending call with empty
    /// arguments buffer.
    #[must_use]
    pub fn add_start(self, idx: ToolCallIndex, id: ToolCallId, name: ToolName) -> Self {
        let new_entry = (idx, PendingCall::new(id, name, String::new()));
        Self {
            pending: self
                .pending
                .into_iter()
                .chain(std::iter::once(new_entry))
                .collect(),
        }
    }

    /// Append a `ToolCallArgs` fragment to the call at `idx`.  If no call
    /// exists at `idx` yet (degenerate server behaviour), the fragment is
    /// silently dropped.
    #[must_use]
    pub fn add_args(self, idx: ToolCallIndex, frag: MessageBody) -> Self {
        let frag_str = frag.into_string();
        let new_pending = self
            .pending
            .into_iter()
            .map(|(i, call)| {
                let new_args_buf = if i == idx {
                    call.args_buf().to_owned() + frag_str.as_str()
                } else {
                    call.args_buf().to_owned()
                };
                (
                    i,
                    PendingCall::new(call.id().clone(), call.name().clone(), new_args_buf),
                )
            })
            .collect();
        Self {
            pending: new_pending,
        }
    }

    /// Consume the accumulator and return the assembled list of pending
    /// calls in stream order.
    #[must_use]
    pub fn finalize(self) -> Vec<PendingCall> {
        self.pending.into_iter().map(|(_, call)| call).collect()
    }

    /// `true` iff no tool-call deltas have been observed yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

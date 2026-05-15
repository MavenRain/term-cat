//! `ChatCompletionChunk`: the JSON shape of one SSE `data:` line returned by
//! a streaming chat-completion request.  The chunk is lowered into a `Vec<
//! ChatEvent>` so the SSE parser can emit one event per `Stream::unfold` step.

use serde::Deserialize;

use crate::agent::ChatEvent;
use crate::error::{Error, WireError};
use crate::newtype::{FinishReason, MessageBody, ToolCallId, ToolCallIndex, ToolName};
use crate::wire::message::ToolCallKind;

/// One SSE chunk in the `OpenAI` streaming Chat Completions API.  `choices` is
/// almost always length 1 in practice; we handle longer arrays uniformly.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionChunk {
    #[serde(default)]
    choices: Vec<ChunkChoice>,
}

impl ChatCompletionChunk {
    /// Project the chunk into zero or more `ChatEvent`s.  Order within a
    /// choice: tool-call deltas first, then content delta, then finish reason.
    ///
    /// # Errors
    ///
    /// Returns `Error::Wire(WireError::UnknownFinishReason(_))` if a
    /// `finish_reason` field carries a string we do not recognize.
    pub fn into_events(self) -> Result<Vec<ChatEvent>, Error> {
        let nested = self
            .choices
            .into_iter()
            .map(ChunkChoice::into_events)
            .collect::<Result<Vec<Vec<ChatEvent>>, Error>>()?;
        Ok(nested.into_iter().flatten().collect())
    }
}

/// One element of `chunk.choices[]`.
#[derive(Debug, Clone, Deserialize)]
pub struct ChunkChoice {
    #[serde(default)]
    delta: ChunkDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

impl ChunkChoice {
    fn into_events(self) -> Result<Vec<ChatEvent>, Error> {
        let Self {
            delta,
            finish_reason,
        } = self;
        let tool_events: Vec<ChatEvent> = delta
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .flat_map(ChunkToolCall::into_events)
            .collect();
        let content_event: Option<ChatEvent> = delta.content.map(ChatEvent::Token);
        let finish_event: Option<ChatEvent> = finish_reason
            .map(|s| parse_finish_reason(&s).map(ChatEvent::Finished))
            .transpose()?;
        Ok(tool_events
            .into_iter()
            .chain(content_event)
            .chain(finish_event)
            .collect())
    }
}

/// The `delta` object inside a chunk choice.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ChunkDelta {
    #[serde(default)]
    content: Option<MessageBody>,
    #[serde(default)]
    tool_calls: Option<Vec<ChunkToolCall>>,
}

/// One element of `delta.tool_calls[]`.  Every field is optional because
/// tool-call deltas are spread across multiple chunks: the first chunk
/// carries `id`, `type`, and `function.name`; later chunks carry only
/// `function.arguments` fragments under the same `index`.
#[derive(Debug, Clone, Deserialize)]
pub struct ChunkToolCall {
    index: u32,
    #[serde(default)]
    id: Option<ToolCallId>,
    #[serde(default, rename = "type")]
    #[allow(dead_code)]
    kind: Option<ToolCallKind>,
    #[serde(default)]
    function: Option<ChunkToolCallFunction>,
}

impl ChunkToolCall {
    fn into_events(self) -> Vec<ChatEvent> {
        let Self {
            index,
            id,
            kind: _,
            function,
        } = self;
        let index = ToolCallIndex::new(index);
        let function_ref = function.as_ref();
        let name_opt = function_ref.and_then(|f| f.name.as_ref()).cloned();
        let args_opt = function_ref
            .and_then(|f| f.arguments.as_ref())
            .filter(|s| !s.is_empty())
            .cloned();
        let start = id
            .and_then(|id_val| name_opt.map(|name| ChatEvent::ToolCallStart(index, id_val, name)));
        let args = args_opt.map(|s| ChatEvent::ToolCallArgs(index, MessageBody::new(s)));
        start.into_iter().chain(args).collect()
    }
}

/// The `function` object inside a `ChunkToolCall`.
#[derive(Debug, Clone, Deserialize)]
pub struct ChunkToolCallFunction {
    #[serde(default)]
    name: Option<ToolName>,
    #[serde(default)]
    arguments: Option<String>,
}

fn parse_finish_reason(s: &str) -> Result<FinishReason, Error> {
    match s {
        "stop" => Ok(FinishReason::Stop),
        "length" => Ok(FinishReason::Length),
        "content_filter" => Ok(FinishReason::ContentFilter),
        "tool_calls" => Ok(FinishReason::ToolCalls),
        other => Err(Error::Wire(WireError::UnknownFinishReason(
            other.to_owned(),
        ))),
    }
}

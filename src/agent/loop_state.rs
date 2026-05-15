//! State machine for the tool-calling loop.  One unfold step yields one
//! `AgentEvent`; transitions that produce no user-facing event (tool-call
//! delta accumulation, request issuance) recurse via `Io::flat_map` through
//! `loop_step` until a yieldable event is reached or the stream ends.

use std::convert::Infallible;

use comp_cat_rs::effect::io::Io;
use rig_cat::tool::Tool;
use serde_json::Value;

use crate::agent::accumulator::{PendingCall, ToolCallAccumulator};
use crate::agent::channel_filter::ChannelFilter;
use crate::agent::event::{AgentEvent, ChatEvent};
use crate::agent::streaming::StreamingAgent;
use crate::bridge::CancelObserver;
use crate::error::Error;
use crate::newtype::{FinishReason, MessageBody};
use crate::sse::{SseState, step as sse_step};
use crate::wire::Message;

/// Top-level state of the agent's tool-calling loop.  Generic over the tool
/// dispatcher type `T` so callers can supply any concrete `Tool` enum.
pub enum LoopState<T: Tool + Clone> {
    /// About to issue a fresh streaming chat-completion HTTP request with the
    /// current wire history.
    FreshRequest {
        agent: StreamingAgent<T>,
        history: Vec<Message>,
        cancel: CancelObserver,
    },
    /// Mid-stream: pulling token / tool-call deltas from the SSE parser.
    /// `filter` strips channel markers (e.g. `SuperGemma`'s `<|channel>...
    /// <channel|>`) from the token stream before the UI sees them.
    Streaming {
        agent: StreamingAgent<T>,
        sse_state: SseState,
        accumulator: ToolCallAccumulator,
        partial_text: MessageBody,
        history: Vec<Message>,
        filter: ChannelFilter,
    },
    /// Next yieldable event will be `AgentEvent::ToolInvoked` for `current`;
    /// the actual dispatch happens in the subsequent `DispatchTool` step.
    AnnouncedTool {
        agent: StreamingAgent<T>,
        current: PendingCall,
        remaining: Vec<PendingCall>,
        history: Vec<Message>,
        cancel: CancelObserver,
    },
    /// Dispatch `current`, emit `ToolReturned`, and move to either the next
    /// `AnnouncedTool` (more pending) or `FreshRequest` (queue drained).
    DispatchTool {
        agent: StreamingAgent<T>,
        current: PendingCall,
        remaining: Vec<PendingCall>,
        history: Vec<Message>,
        cancel: CancelObserver,
    },
    /// Stream-terminal state.  The next unfold step yields `None`.
    Done,
}

/// One unfold step over `LoopState`.  Dispatches to a per-phase helper that
/// returns either a yieldable `AgentEvent` plus the next `LoopState`, or
/// `None` to end the stream.
#[must_use]
pub fn loop_step<T: Tool + Clone + Send + Sync + 'static>(
    state: LoopState<T>,
) -> Io<Error, Option<(AgentEvent, LoopState<T>)>> {
    match state {
        LoopState::FreshRequest {
            agent,
            history,
            cancel,
        } => step_fresh_request(agent, history, cancel),
        LoopState::Streaming {
            agent,
            sse_state,
            accumulator,
            partial_text,
            history,
            filter,
        } => step_streaming(agent, sse_state, accumulator, partial_text, history, filter),
        LoopState::AnnouncedTool {
            agent,
            current,
            remaining,
            history,
            cancel,
        } => step_announced(agent, current, remaining, history, cancel),
        LoopState::DispatchTool {
            agent,
            current,
            remaining,
            history,
            cancel,
        } => step_dispatch(agent, current, remaining, history, cancel),
        LoopState::Done => Io::pure(None),
    }
}

fn step_fresh_request<T: Tool + Clone + Send + Sync + 'static>(
    agent: StreamingAgent<T>,
    history: Vec<Message>,
    cancel: CancelObserver,
) -> Io<Error, Option<(AgentEvent, LoopState<T>)>> {
    let request = agent.build_request(history.clone());
    agent
        .provider()
        .open_request(request, cancel)
        .flat_map(move |sse_state| {
            loop_step(LoopState::Streaming {
                agent,
                sse_state,
                accumulator: ToolCallAccumulator::empty(),
                partial_text: MessageBody::new(""),
                history,
                filter: ChannelFilter::new(),
            })
        })
}

fn step_streaming<T: Tool + Clone + Send + Sync + 'static>(
    agent: StreamingAgent<T>,
    sse_state: SseState,
    accumulator: ToolCallAccumulator,
    partial_text: MessageBody,
    history: Vec<Message>,
    filter: ChannelFilter,
) -> Io<Error, Option<(AgentEvent, LoopState<T>)>> {
    sse_step(sse_state).flat_map(move |opt| match opt {
        None => Io::pure(None),
        Some((event, new_sse_state)) => handle_event(
            event,
            new_sse_state,
            agent,
            accumulator,
            partial_text,
            history,
            filter,
        ),
    })
}

fn handle_event<T: Tool + Clone + Send + Sync + 'static>(
    event: ChatEvent,
    sse_state: SseState,
    agent: StreamingAgent<T>,
    accumulator: ToolCallAccumulator,
    partial_text: MessageBody,
    history: Vec<Message>,
    filter: ChannelFilter,
) -> Io<Error, Option<(AgentEvent, LoopState<T>)>> {
    match event {
        ChatEvent::Token(body) => handle_token(
            &body,
            sse_state,
            agent,
            accumulator,
            partial_text,
            history,
            filter,
        ),
        ChatEvent::ToolCallStart(idx, id, name) => {
            let new_accumulator = accumulator.add_start(idx, id, name);
            loop_step(LoopState::Streaming {
                agent,
                sse_state,
                accumulator: new_accumulator,
                partial_text,
                history,
                filter,
            })
        }
        ChatEvent::ToolCallArgs(idx, frag) => {
            let new_accumulator = accumulator.add_args(idx, frag);
            loop_step(LoopState::Streaming {
                agent,
                sse_state,
                accumulator: new_accumulator,
                partial_text,
                history,
                filter,
            })
        }
        ChatEvent::Finished(reason) => {
            handle_finished(reason, sse_state, agent, accumulator, partial_text, history)
        }
    }
}

fn handle_token<T: Tool + Clone + Send + Sync + 'static>(
    body: &MessageBody,
    sse_state: SseState,
    agent: StreamingAgent<T>,
    accumulator: ToolCallAccumulator,
    partial_text: MessageBody,
    history: Vec<Message>,
    filter: ChannelFilter,
) -> Io<Error, Option<(AgentEvent, LoopState<T>)>> {
    let (new_filter, emit_str) = filter.feed(body.as_str());
    if emit_str.is_empty() {
        // The raw token was entirely consumed by a marker block or a
        // partial-marker tail.  Don't emit an `AgentEvent`; recurse into
        // the next unfold step so the UI never sees the marker text.
        loop_step(LoopState::Streaming {
            agent,
            sse_state,
            accumulator,
            partial_text,
            history,
            filter: new_filter,
        })
    } else {
        let visible = MessageBody::new(emit_str);
        let new_partial = partial_text.concat(&visible);
        Io::pure(Some((
            AgentEvent::AssistantToken(visible),
            LoopState::Streaming {
                agent,
                sse_state,
                accumulator,
                partial_text: new_partial,
                history,
                filter: new_filter,
            },
        )))
    }
}

fn handle_finished<T: Tool + Clone + Send + Sync + 'static>(
    reason: FinishReason,
    sse_state: SseState,
    agent: StreamingAgent<T>,
    accumulator: ToolCallAccumulator,
    partial_text: MessageBody,
    history: Vec<Message>,
) -> Io<Error, Option<(AgentEvent, LoopState<T>)>> {
    match reason {
        FinishReason::Stop | FinishReason::Length | FinishReason::ContentFilter => {
            // Turn is done.  The bridge layer emits `TurnDone` once the
            // stream completes.  We discard `sse_state` (and with it the
            // cancel observer); no further iterations are needed.
            Io::pure(None)
        }
        FinishReason::ToolCalls => {
            let cancel = sse_state.into_cancel();
            let pending_calls = accumulator.finalize();
            let wire_calls = pending_calls.iter().map(PendingCall::to_wire).collect();
            let asst_msg = if partial_text.is_empty() {
                Message::assistant_tool_calls(wire_calls)
            } else {
                Message::assistant_text_and_tool_calls(partial_text, wire_calls)
            };
            let new_history: Vec<Message> = history
                .into_iter()
                .chain(std::iter::once(asst_msg))
                .collect();
            transition_to_announce(agent, &pending_calls, new_history, cancel)
        }
    }
}

fn transition_to_announce<T: Tool + Clone + Send + Sync + 'static>(
    agent: StreamingAgent<T>,
    pending_calls: &[PendingCall],
    history: Vec<Message>,
    cancel: CancelObserver,
) -> Io<Error, Option<(AgentEvent, LoopState<T>)>> {
    match pending_calls {
        [first, rest @ ..] => {
            let first = first.clone();
            let rest = rest.to_vec();
            loop_step(LoopState::AnnouncedTool {
                agent,
                current: first,
                remaining: rest,
                history,
                cancel,
            })
        }
        [] => Io::pure(None),
    }
}

fn step_announced<T: Tool + Clone + Send + Sync + 'static>(
    agent: StreamingAgent<T>,
    current: PendingCall,
    remaining: Vec<PendingCall>,
    history: Vec<Message>,
    cancel: CancelObserver,
) -> Io<Error, Option<(AgentEvent, LoopState<T>)>> {
    let args_value = parse_args_lossy(current.args_buf());
    let event = AgentEvent::ToolInvoked {
        name: current.name().clone(),
        args: args_value,
    };
    Io::pure(Some((
        event,
        LoopState::DispatchTool {
            agent,
            current,
            remaining,
            history,
            cancel,
        },
    )))
}

fn step_dispatch<T: Tool + Clone + Send + Sync + 'static>(
    agent: StreamingAgent<T>,
    current: PendingCall,
    remaining: Vec<PendingCall>,
    history: Vec<Message>,
    cancel: CancelObserver,
) -> Io<Error, Option<(AgentEvent, LoopState<T>)>> {
    let args_value = parse_args_lossy(current.args_buf());
    let name_str = current.name().as_str().to_owned();
    agent
        .toolbox()
        .invoke(&name_str, args_value)
        .attempt()
        .map_error(infallible_to_error)
        .flat_map(move |inner_result| {
            let result_value =
                inner_result.unwrap_or_else(|e| serde_json::json!({ "error": format!("{e:?}") }));
            let result_body_str = serde_json::to_string(&result_value)
                .unwrap_or_else(|_| "<serialize-error>".to_owned());
            let tool_msg = Message::tool(current.id().clone(), MessageBody::new(result_body_str));
            let new_history: Vec<Message> = history
                .into_iter()
                .chain(std::iter::once(tool_msg))
                .collect();
            let event = AgentEvent::ToolReturned {
                name: current.name().clone(),
                result: result_value,
            };
            transition_after_dispatch(agent, &remaining, new_history, cancel, event)
        })
}

fn transition_after_dispatch<T: Tool + Clone + Send + Sync + 'static>(
    agent: StreamingAgent<T>,
    remaining: &[PendingCall],
    history: Vec<Message>,
    cancel: CancelObserver,
    event: AgentEvent,
) -> Io<Error, Option<(AgentEvent, LoopState<T>)>> {
    match remaining {
        [next, rest @ ..] => {
            let next = next.clone();
            let rest = rest.to_vec();
            Io::pure(Some((
                event,
                LoopState::AnnouncedTool {
                    agent,
                    current: next,
                    remaining: rest,
                    history,
                    cancel,
                },
            )))
        }
        [] => Io::pure(Some((
            event,
            LoopState::FreshRequest {
                agent,
                history,
                cancel,
            },
        ))),
    }
}

/// Parse the accumulated `args_buf` as JSON, falling back to `Value::Null`
/// if the model emitted invalid JSON.  The fallback gives the tool a chance
/// to fail gracefully via its own validation rather than aborting the
/// entire turn.
fn parse_args_lossy(args_buf: &str) -> Value {
    serde_json::from_str(args_buf).unwrap_or(Value::Null)
}

/// Vacuous lift from the empty `Infallible` type to our `Error` enum.  See
/// `bridge::channel::infallible_to_error` for the parent rationale; this one
/// lives here so the loop module is self-contained.
fn infallible_to_error(i: Infallible) -> Error {
    match i {}
}

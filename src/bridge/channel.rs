//! Producer-fiber spawning.  Phase C drives the streaming
//! `LocalOpenAiCompletion::stream_chat`; the fiber folds the `ChatEvent`
//! stream into per-token `AgentEvent`s and signals `TurnDone` at the end.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::mpsc::Sender;

use comp_cat_rs::effect::fiber::{Fiber, FiberError};
use comp_cat_rs::effect::io::Io;

use crate::agent::{AgentEvent, ChatEvent};
use crate::bridge::cancel::CancelObserver;
use crate::error::Error;
use crate::newtype::MessageBody;
use crate::provider::LocalOpenAiCompletion;
use crate::tui::history::{ChatHistory, HistoryEntry};
use crate::wire::Message;

/// Spawn a fiber that streams a chat-completion turn over SSE.  Each
/// `ChatEvent::Token` is forwarded as `AgentEvent::AssistantToken`.  Tool-call
/// events are dropped here for Phase C; Phase D's tool loop will consume them.
/// The fiber always sends `AgentEvent::TurnDone` as its final event, preceded
/// by `AgentEvent::Failure` if the stream errored.
///
/// # Errors
///
/// Returns `FiberError::SpawnFailed` if the OS thread cannot be created.
#[must_use]
pub fn spawn_streaming_fiber(
    provider: &LocalOpenAiCompletion,
    messages: Vec<Message>,
    tx: &Sender<AgentEvent>,
    cancel: CancelObserver,
) -> Io<FiberError<Error>, Fiber<Error, ()>> {
    let tx_for_fold = tx.clone();
    let tx_for_final = tx.clone();
    let stream = provider.stream_chat(messages, cancel);

    let fold_fn: Arc<dyn Fn(Sender<AgentEvent>, ChatEvent) -> Sender<AgentEvent> + Send + Sync> =
        Arc::new(|tx, event| {
            let _ = chat_to_agent(event).map(|ae| tx.send(ae));
            tx
        });
    let folded: Io<Error, Sender<AgentEvent>> = stream.fold(tx_for_fold, fold_fn);

    let work: Io<Error, ()> = folded
        .attempt()
        .map(move |result| {
            result.map_or_else(
                |e: Error| {
                    let _ =
                        tx_for_final.send(AgentEvent::Failure(MessageBody::new(format!("{e}"))));
                },
                |_tx: Sender<AgentEvent>| {
                    // Success: tokens already sent via the fold; just fall
                    // through to send TurnDone below.
                },
            );
            let _ = tx_for_final.send(AgentEvent::TurnDone);
        })
        .map_error(infallible_to_error);

    Fiber::fork(work)
}

/// Translate a `ChatEvent` into an `AgentEvent` for the TUI.  Phase C handles
/// only `Token`; tool-call and finish events are dropped here.  Phase D will
/// route tool-call events through a dedicated dispatch path.
fn chat_to_agent(event: ChatEvent) -> Option<AgentEvent> {
    match event {
        ChatEvent::Token(body) => Some(AgentEvent::AssistantToken(body)),
        ChatEvent::ToolCallStart(_, _, _)
        | ChatEvent::ToolCallArgs(_, _)
        | ChatEvent::Finished(_) => None,
    }
}

/// Vacuous lift from the empty `Infallible` type into our `Error` enum.  Used
/// to translate the type signature of `Io::attempt` (which returns
/// `Io<Infallible, _>`) back into our project-wide `Error`.  The body of this
/// function is unreachable at runtime: `Infallible` has zero variants, so a
/// zero-arm match is the idiomatic, panic-free encoding.
fn infallible_to_error(i: Infallible) -> Error {
    match i {}
}

/// Project the user-visible `ChatHistory` into the wire-message format the
/// OpenAI-compatible API expects.  `AssistantPartial`, `ToolInvoked`,
/// `ToolReturned`, and `Error` history entries are skipped: they are UI
/// concerns, not protocol-level messages.  Phase D will lift `ToolInvoked` /
/// `ToolReturned` into proper `assistant_tool_calls` / `tool` wire messages.
#[must_use]
pub fn history_to_wire(history: &ChatHistory) -> Vec<Message> {
    history
        .entries()
        .iter()
        .filter_map(|entry| match entry {
            HistoryEntry::User(body) => Some(Message::user(body.clone())),
            HistoryEntry::AssistantComplete(body) => Some(Message::assistant(body.clone())),
            HistoryEntry::AssistantPartial(_)
            | HistoryEntry::ToolInvoked(_)
            | HistoryEntry::ToolReturned(_)
            | HistoryEntry::Error(_) => None,
        })
        .collect()
}

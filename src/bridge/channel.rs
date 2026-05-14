//! Producer-fiber spawning.  Phase B drives the blocking
//! `LocalOpenAiCompletion::complete`; Phase C will replace this with a
//! streaming `StreamingAgent::turn` driver.

use std::convert::Infallible;
use std::sync::mpsc::Sender;

use comp_cat_rs::effect::fiber::{Fiber, FiberError};
use comp_cat_rs::effect::io::Io;

use crate::agent::AgentEvent;
use crate::error::Error;
use crate::newtype::MessageBody;
use crate::provider::LocalOpenAiCompletion;
use crate::tui::history::{ChatHistory, HistoryEntry};
use crate::wire::Message;

/// Spawn a fiber that issues one blocking chat-completion request, sends the
/// resulting assistant body as `AgentEvent::AssistantToken`, and emits
/// `AgentEvent::TurnDone`.  On error, sends `AgentEvent::Failure` followed by
/// `AgentEvent::TurnDone`.  No streaming yet; Phase C splits the body into
/// per-chunk tokens.
///
/// # Errors
///
/// Returns `FiberError::SpawnFailed` if the OS thread cannot be created.
#[must_use]
pub fn spawn_completion_fiber(
    provider: &LocalOpenAiCompletion,
    messages: Vec<Message>,
    tx: &Sender<AgentEvent>,
) -> Io<FiberError<Error>, Fiber<Error, ()>> {
    let tx_for_send = tx.clone();
    let work: Io<Error, ()> = provider
        .complete(messages)
        .attempt()
        .map(move |result| {
            let event = result.map_or_else(
                |e: Error| AgentEvent::Failure(MessageBody::new(format!("{e}"))),
                AgentEvent::AssistantToken,
            );
            let _ = tx_for_send.send(event);
            let _ = tx_for_send.send(AgentEvent::TurnDone);
        })
        .map_error(infallible_to_error);

    Fiber::fork(work)
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

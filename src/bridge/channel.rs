//! Producer-fiber spawning.  Phase D drives `StreamingAgent::turn`, which
//! yields `AgentEvent`s directly (no intermediate `ChatEvent` translation).
//! The fiber forwards each event onto the mpsc channel and emits
//! `AgentEvent::TurnDone` once the stream completes, preceded by
//! `AgentEvent::Failure` if the stream errored.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::mpsc::Sender;

use comp_cat_rs::effect::fiber::{Fiber, FiberError};
use comp_cat_rs::effect::io::Io;
use rig_cat::tool::Tool;

use crate::agent::{AgentEvent, StreamingAgent};
use crate::bridge::cancel::CancelObserver;
use crate::error::Error;
use crate::newtype::MessageBody;
use crate::tui::history::{ChatHistory, HistoryEntry};
use crate::wire::Message;

/// Spawn a fiber that drives one `StreamingAgent::turn` to completion.
/// Tokens, tool invocations, and tool results are forwarded to `tx` as
/// `AgentEvent`s.  On stream error, a `Failure` event is sent before
/// `TurnDone`.
///
/// # Errors
///
/// Returns `FiberError::SpawnFailed` if the OS thread cannot be created.
#[must_use]
pub fn spawn_agent_turn<T: Tool + Clone + Send + Sync + 'static>(
    agent: StreamingAgent<T>,
    history: Vec<Message>,
    tx: &Sender<AgentEvent>,
    cancel: CancelObserver,
) -> Io<FiberError<Error>, Fiber<Error, ()>> {
    let tx_for_fold = tx.clone();
    let tx_for_final = tx.clone();
    let stream = agent.turn(history, cancel);

    let fold_fn: Arc<dyn Fn(Sender<AgentEvent>, AgentEvent) -> Sender<AgentEvent> + Send + Sync> =
        Arc::new(|tx, event| {
            let _ = tx.send(event);
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
                    // Success: events already sent via the fold; just fall
                    // through to send TurnDone below.
                },
            );
            let _ = tx_for_final.send(AgentEvent::TurnDone);
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
/// OpenAI-compatible API expects.  Used only for the initial wire history of
/// a turn; subsequent iterations within the tool loop maintain wire history
/// internally (with proper `assistant_tool_calls` / `tool` messages).
///
/// `AssistantPartial`, `ToolInvoked`, `ToolReturned`, and `Error` history
/// entries are skipped: they are UI concerns, not protocol-level messages
/// that we can reconstruct here (the structured tool-call data lives only
/// inside the loop's wire history, which has already been folded into the
/// `assistant` / `tool` messages of prior turns and is not preserved across
/// `ChatHistory` boundaries in v1).
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

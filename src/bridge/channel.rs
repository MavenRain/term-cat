//! Producer-fiber spawning.  Phase A ships only the mock producer; Phase C
//! will introduce `spawn_agent_turn` driving a real `StreamingAgent::turn`
//! stream.

use std::sync::mpsc::Sender;
use std::time::Duration;

use comp_cat_rs::effect::fiber::{Fiber, FiberError};
use comp_cat_rs::effect::io::Io;

use crate::agent::AgentEvent;
use crate::bridge::cancel::CancelObserver;
use crate::error::{BridgeError, Error};
use crate::newtype::MessageBody;

/// Spawn a fiber that emits a fixed sequence of `AssistantToken` events
/// followed by `TurnDone`.  Sleeps 200 ms between tokens to simulate streaming.
///
/// The fiber observes `cancel` and stops cleanly as soon as a signal arrives.
/// Use this in Phase A to exercise the TUI without a real LLM.
///
/// # Errors
///
/// Returns a `FiberError` if the OS thread cannot be spawned.  Once running,
/// per-token send failures abort the fiber with `BridgeError::SendDisconnected`.
#[must_use]
pub fn spawn_mock_producer(
    tx: Sender<AgentEvent>,
    cancel: CancelObserver,
) -> Io<FiberError<Error>, Fiber<Error, ()>> {
    Fiber::fork(Io::suspend(move || {
        let words: Vec<&'static str> = vec![
            "Hello,",
            " world!",
            "  ",
            "This",
            " is",
            " a",
            " mock",
            " stream",
            " emitting",
            " one",
            " token",
            " every",
            " 200",
            " milliseconds.",
            "  ",
            "Press",
            " Esc",
            " to",
            " cancel,",
            " or",
            " wait",
            " for",
            " TurnDone.",
        ];

        let stopped = words
            .iter()
            .try_fold(false, |stopped, word| -> Result<bool, Error> {
                if stopped || cancel.is_cancelled() {
                    Ok(true)
                } else {
                    std::thread::sleep(Duration::from_millis(200));
                    let token = MessageBody::new((*word).to_owned());
                    tx.send(AgentEvent::AssistantToken(token))
                        .map_err(|_| Error::Bridge(BridgeError::SendDisconnected))?;
                    Ok(false)
                }
            })?;

        if !stopped {
            let _ = tx.send(AgentEvent::TurnDone);
        }
        Ok(())
    }))
}

//! `term-cat` binary entry.  Phase A wires up:
//!   1. an mpsc channel for `AgentEvent`s,
//!   2. a cancel channel,
//!   3. a mock producer fiber that emits a fixed token sequence,
//!   4. the TUI frame loop.
//!
//! When the user presses Esc, the cancel signal stops the producer; when they
//! press Ctrl+C / Ctrl+D, the TUI exits and the fiber is joined.

#![forbid(unsafe_code)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]

use std::sync::mpsc;

use term_cat::agent::AgentEvent;
use term_cat::bridge::{cancel_channel, spawn_mock_producer};
use term_cat::error::Error;
use term_cat::tui;

fn main() -> Result<(), Error> {
    let (event_tx, event_rx) = mpsc::channel::<AgentEvent>();
    let (canceller, observer) = cancel_channel();

    let fiber = spawn_mock_producer(event_tx, observer)
        .run()
        .map_err(Error::from)?;

    let tui_outcome: Result<(), Error> = tui::run(event_rx, canceller).run();

    let join_outcome: Result<(), Error> = fiber.join().run().map_err(Error::from);

    [tui_outcome, join_outcome]
        .into_iter()
        .try_fold((), |(), r| r)
}

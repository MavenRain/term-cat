//! `term-cat` binary entry.  Phase B wires up:
//!   1. an mpsc channel for `AgentEvent`s,
//!   2. a cancel channel (held but not yet observed: blocking complete is
//!      uncancellable mid-call; Phase C wires it up),
//!   3. a `LocalOpenAiCompletion` provider pointed at an OpenAI-compatible
//!      local server,
//!   4. the TUI frame loop, which spawns a fresh completion fiber per turn.
//!
//! Configuration (env vars, all optional):
//!   `TERM_CAT_BASE_URL`   default `http://localhost:1234/v1`
//!   `TERM_CAT_MODEL`      default `local-model`
//!
//! On exit, any outstanding fiber's `JoinHandle` is detached (each fiber is
//! short-lived and pushes to mpsc; sender disconnection ends it cleanly).

#![forbid(unsafe_code)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]

use std::env;
use std::sync::mpsc;

use term_cat::agent::AgentEvent;
use term_cat::bridge::cancel_channel;
use term_cat::error::Error;
use term_cat::newtype::{BaseUrl, ModelName};
use term_cat::provider::LocalOpenAiCompletion;
use term_cat::tui;

fn main() -> Result<(), Error> {
    let base_url = BaseUrl::new(
        env::var("TERM_CAT_BASE_URL").unwrap_or_else(|_| "http://localhost:1234/v1".to_owned()),
    );
    let model =
        ModelName::new(env::var("TERM_CAT_MODEL").unwrap_or_else(|_| "local-model".to_owned()));

    let provider = LocalOpenAiCompletion::new(base_url, model);

    let (event_tx, event_rx) = mpsc::channel::<AgentEvent>();
    let (canceller, _observer) = cancel_channel();

    tui::run(provider, event_tx, event_rx, canceller).run()
}

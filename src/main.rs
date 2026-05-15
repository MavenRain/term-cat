//! `term-cat` binary entry.  Phase D wires up:
//!   1. an mpsc channel for `AgentEvent`s,
//!   2. a `LocalOpenAiCompletion` provider pointed at an OpenAI-compatible
//!      local server,
//!   3. a `StreamingAgent<BuiltinTool>` over that provider, with `EchoTool`
//!      and `NowTool` built in,
//!   4. the TUI frame loop, which spawns a fresh streaming completion fiber
//!      per turn (with its own cancel channel) driving the tool-calling loop.
//!
//! Configuration (env vars, all optional):
//!   `TERM_CAT_BASE_URL`       default `http://localhost:1234/v1`
//!   `TERM_CAT_MODEL`          default `local-model`
//!   `TERM_CAT_DISABLE_TOOLS`  if set (to anything), skip registering the
//!                             built-in tools.  Required when the model's
//!                             GGUF metadata declares no tool-call support
//!                             (e.g. supergemma4 via Ollama, which 400s any
//!                             request with a `tools[]` field).

#![forbid(unsafe_code)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]

use std::env;
use std::sync::mpsc;

use term_cat::agent::{AgentEvent, BuiltinTool, EchoTool, NowTool, StreamingAgent};
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
    let tools: Vec<BuiltinTool> = if env::var_os("TERM_CAT_DISABLE_TOOLS").is_some() {
        Vec::new()
    } else {
        vec![BuiltinTool::Echo(EchoTool), BuiltinTool::Now(NowTool)]
    };
    let agent = StreamingAgent::new(provider, tools);

    let (event_tx, event_rx) = mpsc::channel::<AgentEvent>();
    tui::run(agent, event_tx, event_rx).run()
}

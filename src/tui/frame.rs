//! Frame loop modeled as a `Stream<Error, ()>` of frames.  `Stream::unfold`
//! threads `FrameState` forward; `Stream::fold` drains it.  Terminal raw mode
//! is acquired and released by a `RawModeGuard` whose `Drop` runs even if the
//! frame loop returns an error.
//!
//! Phase D: the frame loop owns a `StreamingAgent<T>`.  Each user turn clones
//! the agent and spawns a fiber driving `spawn_agent_turn`, which forwards
//! `AgentEvent`s to the UI mpsc channel.

use std::io::{Stdout, stdout};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;

use comp_cat_rs::effect::io::Io;
use comp_cat_rs::effect::stream::Stream;
use crossterm::ExecutableCommand;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use rig_cat::tool::Tool;

use crate::agent::{AgentEvent, StreamingAgent};
use crate::bridge::{Canceller, cancel_channel, history_to_wire, spawn_agent_turn};
use crate::error::{Error, TuiError};
use crate::newtype::MessageBody;
use crate::tui::app::{AppEffect, AppState, Mode};
use crate::tui::key;
use crate::tui::render;

/// State threaded through one frame.  Generic over the tool dispatcher type
/// `T` so the binary can plug in either `BuiltinTool` (the default) or any
/// custom user-defined tool enum.
struct FrameState<T: Tool + Clone + Send + Sync + 'static> {
    app: AppState,
    terminal: Terminal<CrosstermBackend<Stdout>>,
    rx: Receiver<AgentEvent>,
    current_canceller: Option<Canceller>,
    agent: StreamingAgent<T>,
    tx: Sender<AgentEvent>,
}

/// Type of the per-frame unfold step closure.  Parameterized by `T` so the
/// dyn-trait stays static-dispatchable from the caller's perspective.
type StepFn<T> = Arc<dyn Fn(FrameState<T>) -> Io<Error, Option<((), FrameState<T>)>> + Send + Sync>;

/// Type of the fold closure that drains the frame stream.
type FoldFn = Arc<dyn Fn((), ()) + Send + Sync>;

/// Run the TUI: enter raw mode, drain the frame stream, then restore the
/// terminal on the way out (including on error via the guard's `Drop`).
///
/// # Errors
///
/// Returns `Error::Tui` if terminal setup, draw, or input polling fails.
#[must_use]
pub fn run<T: Tool + Clone + Send + Sync + 'static>(
    agent: StreamingAgent<T>,
    tx: Sender<AgentEvent>,
    rx: Receiver<AgentEvent>,
) -> Io<Error, ()> {
    Io::suspend(move || run_inner(agent, tx, rx))
}

fn run_inner<T: Tool + Clone + Send + Sync + 'static>(
    agent: StreamingAgent<T>,
    tx: Sender<AgentEvent>,
    rx: Receiver<AgentEvent>,
) -> Result<(), Error> {
    let _guard = RawModeGuard::enter().map_err(Error::Tui)?;

    let backend = CrosstermBackend::new(stdout());
    let terminal = Terminal::new(backend).map_err(|e| Error::Tui(TuiError::Io(e)))?;

    let init = FrameState {
        app: AppState::initial(),
        terminal,
        rx,
        current_canceller: None,
        agent,
        tx,
    };

    let step: StepFn<T> =
        Arc::new(|state: FrameState<T>| Io::suspend(move || step_one_frame(state)));
    let frames = Stream::unfold(init, step);
    let fold_fn: FoldFn = Arc::new(|(), ()| ());
    let drain: Io<Error, ()> = frames.fold((), fold_fn);

    drain.run()
}

fn step_one_frame<T: Tool + Clone + Send + Sync + 'static>(
    state: FrameState<T>,
) -> Result<Option<((), FrameState<T>)>, Error> {
    let FrameState {
        app,
        mut terminal,
        rx,
        current_canceller,
        agent,
        tx,
    } = state;

    let app = rx.try_iter().fold(app, AppState::apply_agent_event);

    terminal
        .draw(|f| render::root(f, &app))
        .map_err(|e| Error::Tui(TuiError::Io(e)))?;

    let (app, effect) = if crossterm::event::poll(Duration::from_millis(33))
        .map_err(|e| Error::Tui(TuiError::Io(e)))?
    {
        let raw = crossterm::event::read().map_err(|e| Error::Tui(TuiError::Io(e)))?;
        app.apply_key(key::lift(&raw))
    } else {
        (app, AppEffect::NoOp)
    };

    let current_canceller: Option<Canceller> = match effect {
        AppEffect::NoOp => current_canceller,
        AppEffect::SignalCancel => {
            current_canceller.iter().for_each(Canceller::signal);
            current_canceller
        }
        AppEffect::SendInput => spawn_turn(&agent, &tx, app.history()),
    };

    // After every transition, the canceller is alive iff the app is still
    // actively streaming.  Drop it on Idle / Quitting so the producer fiber
    // observes the sender disconnect and unwinds.
    let current_canceller = if app.mode() == Mode::Streaming {
        current_canceller
    } else {
        None
    };

    if app.is_quit() {
        Ok(None)
    } else {
        Ok(Some((
            (),
            FrameState {
                app,
                terminal,
                rx,
                current_canceller,
                agent,
                tx,
            },
        )))
    }
}

/// Mint a fresh `cancel_channel`, clone the agent for the new turn, and spawn
/// the fiber.  On spawn failure we synthesize `Failure + TurnDone` events via
/// `tx` so the UI returns to Idle and surfaces the error.
fn spawn_turn<T: Tool + Clone + Send + Sync + 'static>(
    agent: &StreamingAgent<T>,
    tx: &Sender<AgentEvent>,
    history: &crate::tui::history::ChatHistory,
) -> Option<Canceller> {
    let messages = history_to_wire(history);
    let (canceller, observer) = cancel_channel();
    let agent_for_turn = agent.clone();
    let fork_result = spawn_agent_turn(agent_for_turn, messages, tx, observer).run();
    fork_result.map_or_else(
        |fe| {
            let _ = tx.send(AgentEvent::Failure(MessageBody::new(format!(
                "fiber spawn failed: {fe:?}"
            ))));
            let _ = tx.send(AgentEvent::TurnDone);
            None
        },
        |_fiber| Some(canceller),
    )
}

/// RAII guard that enters the alternate screen + raw mode on construction and
/// restores both on drop.  This is the only `Drop` in the crate; it is
/// justified because terminal cleanup must run even on panic / error / `?`
/// short-circuit, and there is no idiomatic functional way to express that
/// using `comp-cat-rs` 0.5 (which does not yet expose a `Resource` type).
struct RawModeGuard;

impl RawModeGuard {
    fn enter() -> Result<Self, TuiError> {
        enable_raw_mode().map_err(TuiError::Io)?;
        stdout()
            .execute(EnterAlternateScreen)
            .map_err(TuiError::Io)?;
        Ok(Self)
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = stdout().execute(LeaveAlternateScreen);
        let _ = disable_raw_mode();
    }
}

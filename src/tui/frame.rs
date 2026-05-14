//! Frame loop modeled as a `Stream<Error, ()>` of frames.  `Stream::unfold`
//! threads `FrameState` forward; `Stream::fold` drains it.  Terminal raw mode
//! is acquired and released by a `RawModeGuard` whose `Drop` runs even if the
//! frame loop returns an error.

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

use crate::agent::AgentEvent;
use crate::bridge::{Canceller, history_to_wire, spawn_completion_fiber};
use crate::error::{Error, TuiError};
use crate::newtype::MessageBody;
use crate::provider::LocalOpenAiCompletion;
use crate::tui::app::{AppEffect, AppState};
use crate::tui::key;
use crate::tui::render;

/// State threaded through one frame.  The `Terminal` is moved each iteration
/// because `Terminal::draw` requires `&mut self`; this is the FFI carve-out
/// for ratatui.  `provider` and `tx` are owned by the loop so the
/// `SendInput` effect can spawn a fresh per-turn completion fiber.
struct FrameState {
    app: AppState,
    terminal: Terminal<CrosstermBackend<Stdout>>,
    rx: Receiver<AgentEvent>,
    canceller: Canceller,
    provider: LocalOpenAiCompletion,
    tx: Sender<AgentEvent>,
}

/// Type of the per-frame unfold step closure used by the TUI frame `Stream`.
type StepFn = Arc<dyn Fn(FrameState) -> Io<Error, Option<((), FrameState)>> + Send + Sync>;

/// Type of the accumulator-discarding fold closure used to drain the frame
/// stream.  The two `()` arguments are the accumulator and the per-frame
/// element; the return type is the unit accumulator.
type FoldFn = Arc<dyn Fn((), ()) + Send + Sync>;

/// Run the TUI: enter raw mode, drain the frame stream, then restore the
/// terminal on the way out (including on error via the guard's `Drop`).
///
/// # Errors
///
/// Returns `Error::Tui` if terminal setup, draw, or input polling fails.
#[must_use]
pub fn run(
    provider: LocalOpenAiCompletion,
    tx: Sender<AgentEvent>,
    rx: Receiver<AgentEvent>,
    canceller: Canceller,
) -> Io<Error, ()> {
    Io::suspend(move || run_inner(provider, tx, rx, canceller))
}

fn run_inner(
    provider: LocalOpenAiCompletion,
    tx: Sender<AgentEvent>,
    rx: Receiver<AgentEvent>,
    canceller: Canceller,
) -> Result<(), Error> {
    let _guard = RawModeGuard::enter().map_err(Error::Tui)?;

    let backend = CrosstermBackend::new(stdout());
    let terminal = Terminal::new(backend).map_err(|e| Error::Tui(TuiError::Io(e)))?;

    let init = FrameState {
        app: AppState::initial(),
        terminal,
        rx,
        canceller,
        provider,
        tx,
    };

    let step: StepFn = Arc::new(|state: FrameState| Io::suspend(move || step_one_frame(state)));
    let frames = Stream::unfold(init, step);
    let fold_fn: FoldFn = Arc::new(|(), ()| ());
    let drain: Io<Error, ()> = frames.fold((), fold_fn);

    drain.run()
}

fn step_one_frame(state: FrameState) -> Result<Option<((), FrameState)>, Error> {
    let FrameState {
        app,
        mut terminal,
        rx,
        canceller,
        provider,
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

    match effect {
        AppEffect::NoOp => {}
        AppEffect::SignalCancel => canceller.signal(),
        AppEffect::SendInput => {
            let messages = history_to_wire(app.history());
            let fork_result = spawn_completion_fiber(&provider, messages, &tx).run();
            // On spawn failure we synthesize a Failure + TurnDone so the UI
            // returns to Idle and shows the error.  The TUI drains these on
            // the next frame.
            let _ = fork_result.map_err(|fe| {
                let _ = tx.send(AgentEvent::Failure(MessageBody::new(format!(
                    "fiber spawn failed: {fe:?}"
                ))));
                let _ = tx.send(AgentEvent::TurnDone);
            });
        }
    }

    if app.is_quit() {
        Ok(None)
    } else {
        Ok(Some((
            (),
            FrameState {
                app,
                terminal,
                rx,
                canceller,
                provider,
                tx,
            },
        )))
    }
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

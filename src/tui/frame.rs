//! Frame loop modeled as a `Stream<Error, ()>` of frames.  `Stream::unfold`
//! threads `FrameState` forward; `Stream::fold` drains it.  Terminal raw mode
//! is acquired and released by a `RawModeGuard` whose `Drop` runs even if the
//! frame loop returns an error.

use std::io::{Stdout, stdout};
use std::sync::Arc;
use std::sync::mpsc::Receiver;
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
use crate::bridge::Canceller;
use crate::error::{Error, TuiError};
use crate::tui::app::{AppEffect, AppState};
use crate::tui::key;
use crate::tui::render;

/// State threaded through one frame.  The `Terminal` is moved each iteration
/// because `Terminal::draw` requires `&mut self`; this is the FFI carve-out
/// for ratatui.
struct FrameState {
    app: AppState,
    terminal: Terminal<CrosstermBackend<Stdout>>,
    rx: Receiver<AgentEvent>,
    canceller: Canceller,
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
pub fn run(rx: Receiver<AgentEvent>, canceller: Canceller) -> Io<Error, ()> {
    Io::suspend(move || run_inner(rx, canceller))
}

fn run_inner(rx: Receiver<AgentEvent>, canceller: Canceller) -> Result<(), Error> {
    let _guard = RawModeGuard::enter().map_err(Error::Tui)?;

    let backend = CrosstermBackend::new(stdout());
    let terminal = Terminal::new(backend).map_err(|e| Error::Tui(TuiError::Io(e)))?;

    let init = FrameState {
        app: AppState::initial(),
        terminal,
        rx,
        canceller,
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
        (app, None)
    };

    // `Option::map` for the side effect: the user's conventions forbid
    // `if let` / `match` on `Option`, and `iter().for_each` would trigger
    // clippy::needless_for_each.  Discard the `Option<()>` after the
    // side effect runs.
    let _ = effect.map(|e| match e {
        AppEffect::SignalCancel => canceller.signal(),
        AppEffect::SendInput => {
            // Phase A: nothing wired up to consume the send.  Phase B/C will
            // spawn a fresh producer fiber here.
        }
    });

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

//! Terminal UI: state, key mapping, render, frame loop.
//!
//! The TUI is modeled as a `Stream<Error, ()>` of frames.  Each unfold step
//! consumes the previous `FrameState`, drains the agent-event channel, renders
//! once, polls one input event, and returns either the next state or `None`
//! to end the stream on quit.  `Stream::fold` drains the stream; `run` is the
//! entry point that wraps it in raw-mode setup and teardown.

pub mod app;
pub mod frame;
pub mod history;
pub mod input;
pub mod key;
pub mod render;

pub use app::{AppEffect, AppState, Mode};
pub use frame::run;
pub use history::{ChatHistory, HistoryEntry};
pub use input::InputBuffer;
pub use key::{KeyAction, lift};

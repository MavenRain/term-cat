//! Top-level immutable application state.  Every transition returns a fresh
//! `AppState`; the frame loop rebinds it each iteration.

use crate::agent::AgentEvent;
use crate::newtype::MessageBody;
use crate::tui::history::ChatHistory;
use crate::tui::input::InputBuffer;
use crate::tui::key::KeyAction;

/// Application mode.  Determines what `Esc` and `Enter` mean and what the
/// footer renders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// No turn is in flight; user can type and send.
    Idle,
    /// A producer fiber is streaming an assistant turn.
    Streaming,
    /// User has signalled quit; the frame loop ends on the next iteration.
    Quitting,
}

/// Side effect the frame loop must perform after this transition.  Always
/// returned (never wrapped in `Option`) so the dispatch match in
/// `frame::step_one_frame` is exhaustive on a real enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppEffect {
    /// No side effect needed.
    NoOp,
    /// Signal the producer fiber via the cancel channel.
    SignalCancel,
    /// Submit the latest user message to a fresh producer fiber.  The frame
    /// loop reads the message from `AppState::history()` (the latest `User`
    /// entry) rather than threading it through this enum, which keeps
    /// `AppEffect` `Copy` and avoids carrying ownership.
    SendInput,
}

/// The complete application state.  Constructed once, rebound each frame.
#[derive(Debug, Clone)]
pub struct AppState {
    history: ChatHistory,
    input: InputBuffer,
    mode: Mode,
}

impl AppState {
    /// Initial state: empty history, empty input, idle.
    #[must_use]
    pub fn initial() -> Self {
        Self {
            history: ChatHistory::empty(),
            input: InputBuffer::empty(),
            mode: Mode::Idle,
        }
    }

    /// View the history (for rendering and wire-message derivation).
    #[must_use]
    pub fn history(&self) -> &ChatHistory {
        &self.history
    }

    /// View the input buffer (for rendering).
    #[must_use]
    pub fn input(&self) -> &InputBuffer {
        &self.input
    }

    /// Current mode.
    #[must_use]
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// `true` if the frame loop should terminate after this frame.
    #[must_use]
    pub fn is_quit(&self) -> bool {
        matches!(self.mode, Mode::Quitting)
    }

    /// Fold a `KeyAction` into the state.  Returns the new state and an
    /// effect the caller must interpret.  Dispatches on mode to per-mode
    /// helpers so every match is exhaustive on enums with no `_` wildcards.
    #[must_use]
    pub fn apply_key(self, action: KeyAction) -> (Self, AppEffect) {
        match self.mode {
            Mode::Quitting => (self, AppEffect::NoOp),
            Mode::Idle => self.apply_key_idle(action),
            Mode::Streaming => self.apply_key_streaming(action),
        }
    }

    fn apply_key_idle(self, action: KeyAction) -> (Self, AppEffect) {
        match action {
            KeyAction::Quit => (
                Self {
                    mode: Mode::Quitting,
                    ..self
                },
                AppEffect::NoOp,
            ),
            KeyAction::Send => self.apply_send_idle(),
            KeyAction::NewLine => (self.map_input(InputBuffer::insert_newline), AppEffect::NoOp),
            KeyAction::Backspace => (self.map_input(InputBuffer::backspace), AppEffect::NoOp),
            KeyAction::DeleteWord => (self.map_input(InputBuffer::delete_word), AppEffect::NoOp),
            KeyAction::Char(c) => (self.map_input(move |i| i.insert_char(c)), AppEffect::NoOp),
            KeyAction::Up => (self.scroll_history_up(1), AppEffect::NoOp),
            KeyAction::Down => (self.scroll_history_down(1), AppEffect::NoOp),
            KeyAction::PageUp => (self.scroll_history_up(10), AppEffect::NoOp),
            KeyAction::PageDown => (self.scroll_history_down(10), AppEffect::NoOp),
            KeyAction::Cancel | KeyAction::NoOp => (self, AppEffect::NoOp),
        }
    }

    fn apply_send_idle(self) -> (Self, AppEffect) {
        let Self {
            history,
            input,
            mode: _,
        } = self;
        if input.is_empty() {
            (
                Self {
                    history,
                    input,
                    mode: Mode::Idle,
                },
                AppEffect::NoOp,
            )
        } else {
            let (body, next_input) = input.take();
            let history = history.push_user(body);
            (
                Self {
                    history,
                    input: next_input,
                    mode: Mode::Streaming,
                },
                AppEffect::SendInput,
            )
        }
    }

    fn map_input<F: FnOnce(InputBuffer) -> InputBuffer>(self, f: F) -> Self {
        let Self {
            history,
            input,
            mode,
        } = self;
        Self {
            history,
            input: f(input),
            mode,
        }
    }

    fn apply_key_streaming(self, action: KeyAction) -> (Self, AppEffect) {
        match action {
            KeyAction::Quit => {
                let Self {
                    history,
                    input,
                    mode: _,
                } = self;
                (
                    Self {
                        history,
                        input,
                        mode: Mode::Quitting,
                    },
                    AppEffect::SignalCancel,
                )
            }
            KeyAction::Cancel => {
                let Self {
                    history,
                    input,
                    mode: _,
                } = self;
                let history = history.finalize_assistant();
                (
                    Self {
                        history,
                        input,
                        mode: Mode::Idle,
                    },
                    AppEffect::SignalCancel,
                )
            }
            KeyAction::Up => (self.scroll_history_up(1), AppEffect::NoOp),
            KeyAction::Down => (self.scroll_history_down(1), AppEffect::NoOp),
            KeyAction::PageUp => (self.scroll_history_up(10), AppEffect::NoOp),
            KeyAction::PageDown => (self.scroll_history_down(10), AppEffect::NoOp),
            KeyAction::Send
            | KeyAction::NewLine
            | KeyAction::Backspace
            | KeyAction::DeleteWord
            | KeyAction::Char(_)
            | KeyAction::NoOp => (self, AppEffect::NoOp),
        }
    }

    fn scroll_history_up(self, delta: usize) -> Self {
        let Self {
            history,
            input,
            mode,
        } = self;
        Self {
            history: history.scroll_up(delta),
            input,
            mode,
        }
    }

    fn scroll_history_down(self, delta: usize) -> Self {
        let Self {
            history,
            input,
            mode,
        } = self;
        Self {
            history: history.scroll_down(delta),
            input,
            mode,
        }
    }

    /// Fold an `AgentEvent` (from the producer fiber) into the state.
    #[must_use]
    pub fn apply_agent_event(self, event: AgentEvent) -> Self {
        match event {
            AgentEvent::AssistantToken(body) => {
                let Self {
                    history,
                    input,
                    mode: _,
                } = self;
                Self {
                    history: history.extend_assistant_partial(&body),
                    input,
                    mode: Mode::Streaming,
                }
            }
            AgentEvent::TurnDone => {
                let Self {
                    history,
                    input,
                    mode: _,
                } = self;
                Self {
                    history: history.finalize_assistant(),
                    input,
                    mode: Mode::Idle,
                }
            }
            AgentEvent::ToolInvoked { name, args } => {
                let summary =
                    MessageBody::new(format!("[tool {} invoked: {}]", name.as_str(), args));
                let Self {
                    history,
                    input,
                    mode,
                } = self;
                Self {
                    history: history.push_tool_invoked(summary),
                    input,
                    mode,
                }
            }
            AgentEvent::ToolReturned { name, result } => {
                let summary =
                    MessageBody::new(format!("[tool {} returned: {}]", name.as_str(), result));
                let Self {
                    history,
                    input,
                    mode,
                } = self;
                Self {
                    history: history.push_tool_returned(summary),
                    input,
                    mode,
                }
            }
            AgentEvent::Failure(msg) => {
                let Self {
                    history,
                    input,
                    mode: _,
                } = self;
                Self {
                    history: history.push_error(msg),
                    input,
                    mode: Mode::Idle,
                }
            }
        }
    }
}

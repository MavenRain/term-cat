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

/// Side effect the frame loop must perform after this transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppEffect {
    /// Signal the producer fiber via the cancel channel.
    SignalCancel,
    /// Send the current input buffer to a (future) producer.  Phase A only
    /// emits this as a stub; main.rs ignores it for now.
    SendInput,
}

/// The complete application state.  Constructed once, rebound each frame.
#[derive(Debug, Clone)]
pub struct AppState {
    history: ChatHistory,
    input: InputBuffer,
    mode: Mode,
    pending_send: Option<MessageBody>,
}

impl AppState {
    /// Initial state: empty history, empty input, idle, no pending send.
    #[must_use]
    pub fn initial() -> Self {
        Self {
            history: ChatHistory::empty(),
            input: InputBuffer::empty(),
            mode: Mode::Idle,
            pending_send: None,
        }
    }

    /// View the history (for rendering).
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

    /// The body the caller should hand to the next producer.  Only the
    /// `SendInput` effect makes this `Some`; the frame loop consumes it.
    #[must_use]
    pub fn take_pending_send(self) -> (Option<MessageBody>, Self) {
        let Self {
            history,
            input,
            mode,
            pending_send,
        } = self;
        (
            pending_send,
            Self {
                history,
                input,
                mode,
                pending_send: None,
            },
        )
    }

    /// Fold a `KeyAction` into the state.  Returns the new state and an
    /// optional `AppEffect` the caller must interpret.  Dispatches on the
    /// current mode to a per-mode helper; this keeps every match exhaustive
    /// on enums without resorting to `_` wildcards.
    #[must_use]
    pub fn apply_key(self, action: KeyAction) -> (Self, Option<AppEffect>) {
        match self.mode {
            Mode::Quitting => (self, None),
            Mode::Idle => self.apply_key_idle(action),
            Mode::Streaming => self.apply_key_streaming(action),
        }
    }

    fn apply_key_idle(self, action: KeyAction) -> (Self, Option<AppEffect>) {
        match action {
            KeyAction::Quit => (
                Self {
                    mode: Mode::Quitting,
                    ..self
                },
                None,
            ),
            KeyAction::Send => self.apply_send_idle(),
            KeyAction::NewLine => (self.map_input(InputBuffer::insert_newline), None),
            KeyAction::Backspace => (self.map_input(InputBuffer::backspace), None),
            KeyAction::DeleteWord => (self.map_input(InputBuffer::delete_word), None),
            KeyAction::Char(c) => (self.map_input(move |i| i.insert_char(c)), None),
            KeyAction::Up => (self.scroll_history_up(1), None),
            KeyAction::Down => (self.scroll_history_down(1), None),
            KeyAction::PageUp => (self.scroll_history_up(10), None),
            KeyAction::PageDown => (self.scroll_history_down(10), None),
            KeyAction::Cancel | KeyAction::NoOp => (self, None),
        }
    }

    fn apply_send_idle(self) -> (Self, Option<AppEffect>) {
        let Self {
            history,
            input,
            mode: _,
            pending_send: _,
        } = self;
        if input.is_empty() {
            (
                Self {
                    history,
                    input,
                    mode: Mode::Idle,
                    pending_send: None,
                },
                None,
            )
        } else {
            let (body, next_input) = input.take();
            let history = history.push_user(body.clone());
            (
                Self {
                    history,
                    input: next_input,
                    mode: Mode::Streaming,
                    pending_send: Some(body),
                },
                Some(AppEffect::SendInput),
            )
        }
    }

    fn map_input<F: FnOnce(InputBuffer) -> InputBuffer>(self, f: F) -> Self {
        let Self {
            history,
            input,
            mode,
            pending_send,
        } = self;
        Self {
            history,
            input: f(input),
            mode,
            pending_send,
        }
    }

    fn apply_key_streaming(self, action: KeyAction) -> (Self, Option<AppEffect>) {
        match action {
            KeyAction::Quit => {
                let Self {
                    history,
                    input,
                    mode: _,
                    pending_send,
                } = self;
                (
                    Self {
                        history,
                        input,
                        mode: Mode::Quitting,
                        pending_send,
                    },
                    Some(AppEffect::SignalCancel),
                )
            }
            KeyAction::Cancel => {
                let Self {
                    history,
                    input,
                    mode: _,
                    pending_send,
                } = self;
                let history = history.finalize_assistant();
                (
                    Self {
                        history,
                        input,
                        mode: Mode::Idle,
                        pending_send,
                    },
                    Some(AppEffect::SignalCancel),
                )
            }
            KeyAction::Up => (self.scroll_history_up(1), None),
            KeyAction::Down => (self.scroll_history_down(1), None),
            KeyAction::PageUp => (self.scroll_history_up(10), None),
            KeyAction::PageDown => (self.scroll_history_down(10), None),
            KeyAction::Send
            | KeyAction::NewLine
            | KeyAction::Backspace
            | KeyAction::DeleteWord
            | KeyAction::Char(_)
            | KeyAction::NoOp => (self, None),
        }
    }

    fn scroll_history_up(self, delta: usize) -> Self {
        let Self {
            history,
            input,
            mode,
            pending_send,
        } = self;
        Self {
            history: history.scroll_up(delta),
            input,
            mode,
            pending_send,
        }
    }

    fn scroll_history_down(self, delta: usize) -> Self {
        let Self {
            history,
            input,
            mode,
            pending_send,
        } = self;
        Self {
            history: history.scroll_down(delta),
            input,
            mode,
            pending_send,
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
                    pending_send,
                } = self;
                Self {
                    history: history.extend_assistant_partial(&body),
                    input,
                    mode: Mode::Streaming,
                    pending_send,
                }
            }
            AgentEvent::TurnDone => {
                let Self {
                    history,
                    input,
                    mode: _,
                    pending_send,
                } = self;
                Self {
                    history: history.finalize_assistant(),
                    input,
                    mode: Mode::Idle,
                    pending_send,
                }
            }
            AgentEvent::ToolInvoked { name, args } => {
                let summary =
                    MessageBody::new(format!("[tool {} invoked: {}]", name.as_str(), args));
                let Self {
                    history,
                    input,
                    mode,
                    pending_send,
                } = self;
                Self {
                    history: history.push_tool_invoked(summary),
                    input,
                    mode,
                    pending_send,
                }
            }
            AgentEvent::ToolReturned { name, result } => {
                let summary =
                    MessageBody::new(format!("[tool {} returned: {}]", name.as_str(), result));
                let Self {
                    history,
                    input,
                    mode,
                    pending_send,
                } = self;
                Self {
                    history: history.push_tool_returned(summary),
                    input,
                    mode,
                    pending_send,
                }
            }
            AgentEvent::Failure(msg) => {
                let Self {
                    history,
                    input,
                    mode: _,
                    pending_send,
                } = self;
                Self {
                    history: history.push_error(msg),
                    input,
                    mode: Mode::Idle,
                    pending_send,
                }
            }
        }
    }
}

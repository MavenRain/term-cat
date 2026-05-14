//! Map a `crossterm::Event` into our own `KeyAction` enum.  Downstream code
//! matches `KeyAction` exhaustively without `_` wildcards.
//!
//! The `_` wildcards inside this file's `match` apply only to `KeyModifiers`,
//! which is a `bitflags` struct rather than an enum, so the
//! no-`_`-on-enum-matches rule is not engaged on the modifier side.

use crossterm::event::{Event, KeyCode as K, KeyModifiers as M};

/// All keyboard actions term-cat reacts to.  Every variant is handled
/// exhaustively downstream; unmapped keys lift to `NoOp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyAction {
    /// Submit the current input.
    Send,
    /// Insert a newline into the input buffer.
    NewLine,
    /// Delete the previous character from the input buffer.
    Backspace,
    /// Delete the previous word from the input buffer.
    DeleteWord,
    /// Insert a literal character into the input buffer.
    Char(char),
    /// Scroll history one line up.
    Up,
    /// Scroll history one line down.
    Down,
    /// Scroll history one page up.
    PageUp,
    /// Scroll history one page down.
    PageDown,
    /// Cancel an in-flight stream (Esc).
    Cancel,
    /// Quit the application (Ctrl+C / Ctrl+D when idle).
    Quit,
    /// Recognized event with no application meaning.
    NoOp,
}

/// Lift a `crossterm::Event` into a `KeyAction`.  The `KeyCode` side of every
/// tuple is exhaustively listed; the `KeyModifiers` side may use `_` because
/// `KeyModifiers` is a `bitflags` struct, not an enum.
#[must_use]
pub fn lift(event: &Event) -> KeyAction {
    match *event {
        Event::Key(k) => match (k.code, k.modifiers) {
            // Specific, handled (modifier-discriminated) bindings come first;
            // a merged catch-all listing every remaining `KeyCode` variant
            // follows.  Listing every variant keeps us compliant with the
            // no-`_`-wildcard-on-enums rule; the `_` on the modifier side
            // is allowed because `KeyModifiers` is a `bitflags` struct.
            (K::Enter, M::NONE) => KeyAction::Send,
            (K::Enter, M::SHIFT) => KeyAction::NewLine,
            (K::Char(c), M::NONE | M::SHIFT) => KeyAction::Char(c),
            (K::Char('c' | 'd'), M::CONTROL) => KeyAction::Quit,
            (K::Char('w'), M::CONTROL) => KeyAction::DeleteWord,
            (K::Backspace, M::NONE) => KeyAction::Backspace,
            (K::Esc, M::NONE) => KeyAction::Cancel,
            (K::Up, M::NONE) => KeyAction::Up,
            (K::Down, M::NONE) => KeyAction::Down,
            (K::PageUp, M::NONE) => KeyAction::PageUp,
            (K::PageDown, M::NONE) => KeyAction::PageDown,
            (
                K::Char(_)
                | K::Backspace
                | K::Esc
                | K::Up
                | K::Down
                | K::PageUp
                | K::PageDown
                | K::Enter
                | K::Left
                | K::Right
                | K::Home
                | K::End
                | K::Tab
                | K::BackTab
                | K::Delete
                | K::Insert
                | K::F(_)
                | K::Null
                | K::CapsLock
                | K::ScrollLock
                | K::NumLock
                | K::PrintScreen
                | K::Pause
                | K::Menu
                | K::KeypadBegin
                | K::Media(_)
                | K::Modifier(_),
                _,
            ) => KeyAction::NoOp,
        },
        Event::Resize(_, _)
        | Event::Mouse(_)
        | Event::FocusGained
        | Event::FocusLost
        | Event::Paste(_) => KeyAction::NoOp,
    }
}

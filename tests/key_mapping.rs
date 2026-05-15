//! Integration tests for `term_cat::tui::key::lift`.  Each test builds a
//! `crossterm::Event` and verifies that `lift` returns the expected
//! `KeyAction`.  Tests return `Result<(), TestError>` and propagate via `?`;
//! no `assert!` / `assert_eq!` is used (per the no-panics-anywhere rule).

mod common;

use common::{TestError, require_eq};
use crossterm::event::{
    Event, KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers, MediaKeyCode,
    ModifierKeyCode, MouseButton, MouseEvent, MouseEventKind,
};
use term_cat::tui::{KeyAction, lift};

fn k(code: KeyCode, modifiers: KeyModifiers) -> Event {
    Event::Key(KeyEvent {
        code,
        modifiers,
        kind: KeyEventKind::Press,
        state: KeyEventState::NONE,
    })
}

fn check(event: &Event, expected: KeyAction, label: &str) -> Result<(), TestError> {
    require_eq(&lift(event), &expected, label)
}

#[test]
fn enter_no_modifiers_is_send() -> Result<(), TestError> {
    check(
        &k(KeyCode::Enter, KeyModifiers::NONE),
        KeyAction::Send,
        "Enter",
    )
}

#[test]
fn shift_enter_is_newline() -> Result<(), TestError> {
    check(
        &k(KeyCode::Enter, KeyModifiers::SHIFT),
        KeyAction::NewLine,
        "Shift+Enter",
    )
}

#[test]
fn plain_char_is_char() -> Result<(), TestError> {
    check(
        &k(KeyCode::Char('a'), KeyModifiers::NONE),
        KeyAction::Char('a'),
        "Char(a)",
    )
}

#[test]
fn shifted_char_is_char() -> Result<(), TestError> {
    check(
        &k(KeyCode::Char('A'), KeyModifiers::SHIFT),
        KeyAction::Char('A'),
        "Shift+Char(A)",
    )
}

#[test]
fn ctrl_c_is_quit() -> Result<(), TestError> {
    check(
        &k(KeyCode::Char('c'), KeyModifiers::CONTROL),
        KeyAction::Quit,
        "Ctrl+C",
    )
}

#[test]
fn ctrl_d_is_quit() -> Result<(), TestError> {
    check(
        &k(KeyCode::Char('d'), KeyModifiers::CONTROL),
        KeyAction::Quit,
        "Ctrl+D",
    )
}

#[test]
fn ctrl_w_is_delete_word() -> Result<(), TestError> {
    check(
        &k(KeyCode::Char('w'), KeyModifiers::CONTROL),
        KeyAction::DeleteWord,
        "Ctrl+W",
    )
}

#[test]
fn other_ctrl_char_is_noop() -> Result<(), TestError> {
    check(
        &k(KeyCode::Char('x'), KeyModifiers::CONTROL),
        KeyAction::NoOp,
        "Ctrl+X",
    )
}

#[test]
fn backspace_no_modifiers_is_backspace() -> Result<(), TestError> {
    check(
        &k(KeyCode::Backspace, KeyModifiers::NONE),
        KeyAction::Backspace,
        "Backspace",
    )
}

#[test]
fn ctrl_backspace_is_noop() -> Result<(), TestError> {
    check(
        &k(KeyCode::Backspace, KeyModifiers::CONTROL),
        KeyAction::NoOp,
        "Ctrl+Backspace",
    )
}

#[test]
fn esc_no_modifiers_is_cancel() -> Result<(), TestError> {
    check(
        &k(KeyCode::Esc, KeyModifiers::NONE),
        KeyAction::Cancel,
        "Esc",
    )
}

#[test]
fn up_is_up() -> Result<(), TestError> {
    check(&k(KeyCode::Up, KeyModifiers::NONE), KeyAction::Up, "Up")
}

#[test]
fn down_is_down() -> Result<(), TestError> {
    check(
        &k(KeyCode::Down, KeyModifiers::NONE),
        KeyAction::Down,
        "Down",
    )
}

#[test]
fn page_up_is_page_up() -> Result<(), TestError> {
    check(
        &k(KeyCode::PageUp, KeyModifiers::NONE),
        KeyAction::PageUp,
        "PageUp",
    )
}

#[test]
fn page_down_is_page_down() -> Result<(), TestError> {
    check(
        &k(KeyCode::PageDown, KeyModifiers::NONE),
        KeyAction::PageDown,
        "PageDown",
    )
}

#[test]
fn left_arrow_is_noop() -> Result<(), TestError> {
    check(
        &k(KeyCode::Left, KeyModifiers::NONE),
        KeyAction::NoOp,
        "Left",
    )
}

#[test]
fn home_is_noop() -> Result<(), TestError> {
    check(
        &k(KeyCode::Home, KeyModifiers::NONE),
        KeyAction::NoOp,
        "Home",
    )
}

#[test]
fn function_key_is_noop() -> Result<(), TestError> {
    check(&k(KeyCode::F(1), KeyModifiers::NONE), KeyAction::NoOp, "F1")
}

#[test]
fn caps_lock_is_noop() -> Result<(), TestError> {
    check(
        &k(KeyCode::CapsLock, KeyModifiers::NONE),
        KeyAction::NoOp,
        "CapsLock",
    )
}

#[test]
fn media_key_is_noop() -> Result<(), TestError> {
    check(
        &k(KeyCode::Media(MediaKeyCode::Play), KeyModifiers::NONE),
        KeyAction::NoOp,
        "Media(Play)",
    )
}

#[test]
fn modifier_key_is_noop() -> Result<(), TestError> {
    check(
        &k(
            KeyCode::Modifier(ModifierKeyCode::LeftShift),
            KeyModifiers::NONE,
        ),
        KeyAction::NoOp,
        "Modifier(LeftShift)",
    )
}

#[test]
fn mouse_event_is_noop() -> Result<(), TestError> {
    let ev = Event::Mouse(MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column: 0,
        row: 0,
        modifiers: KeyModifiers::NONE,
    });
    require_eq(&lift(&ev), &KeyAction::NoOp, "Mouse(Down)")
}

#[test]
fn resize_event_is_noop() -> Result<(), TestError> {
    require_eq(
        &lift(&Event::Resize(80, 24)),
        &KeyAction::NoOp,
        "Resize(80,24)",
    )
}

#[test]
fn focus_gained_is_noop() -> Result<(), TestError> {
    require_eq(&lift(&Event::FocusGained), &KeyAction::NoOp, "FocusGained")
}

#[test]
fn focus_lost_is_noop() -> Result<(), TestError> {
    require_eq(&lift(&Event::FocusLost), &KeyAction::NoOp, "FocusLost")
}

#[test]
fn paste_event_is_noop() -> Result<(), TestError> {
    require_eq(
        &lift(&Event::Paste("pasted text".to_owned())),
        &KeyAction::NoOp,
        "Paste",
    )
}

#[test]
fn ctrl_enter_is_noop() -> Result<(), TestError> {
    check(
        &k(KeyCode::Enter, KeyModifiers::CONTROL),
        KeyAction::NoOp,
        "Ctrl+Enter",
    )
}

//! ratatui drawing.  One function `root` that lays the screen out and renders
//! each pane.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::tui::app::{AppState, Mode};
use crate::tui::history::HistoryEntry;

/// Render the entire screen for one frame.
pub fn root(f: &mut Frame<'_>, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(5),
            Constraint::Length(5),
            Constraint::Length(1),
        ])
        .split(f.area());

    let history_area = chunks.first().copied().unwrap_or_default();
    // Inner height = block area minus top and bottom borders (Borders::ALL).
    let history_inner_height = history_area.height.saturating_sub(2);
    let history_pane = history_widget(state, history_inner_height);
    f.render_widget(history_pane, history_area);

    let input_pane = input_widget(state);
    let input_area = chunks.get(1).copied().unwrap_or_default();
    f.render_widget(input_pane, input_area);

    let footer = footer_widget(state);
    let footer_area = chunks.get(2).copied().unwrap_or_default();
    f.render_widget(footer, footer_area);
}

fn history_widget<'a>(state: &'a AppState, viewport_height: u16) -> Paragraph<'a> {
    let entries = state.history().entries();
    let lines: Vec<Line<'a>> = entries.iter().flat_map(entry_to_lines).collect();
    let total = lines.len();
    let viewport = usize::from(viewport_height);
    let user_offset = state.history().scroll_offset();
    // Auto-scroll-to-bottom: scroll = max(0, total - viewport).  The user's
    // `scroll_offset` (lines from the bottom) walks the view back toward
    // older content.  Logical-line approximation: this does not account
    // for `Wrap` re-wrapping long lines to multiple display lines, so for
    // very wide content the bottom-alignment may be off by one or two
    // display lines.  Good enough for v1.
    let scroll_y = total.saturating_sub(viewport).saturating_sub(user_offset);
    let scroll_y_u16 = u16::try_from(scroll_y).unwrap_or(u16::MAX);
    Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" term-cat "))
        .wrap(Wrap { trim: false })
        .scroll((scroll_y_u16, 0))
}

fn entry_to_lines(entry: &HistoryEntry) -> Vec<Line<'_>> {
    match entry {
        HistoryEntry::User(body) => prefixed_lines(
            "you  ",
            body.as_str(),
            Style::default().add_modifier(Modifier::BOLD),
        ),
        HistoryEntry::AssistantPartial(body) | HistoryEntry::AssistantComplete(body) => {
            prefixed_lines("bot  ", body.as_str(), Style::default())
        }
        HistoryEntry::ToolInvoked(body) | HistoryEntry::ToolReturned(body) => prefixed_lines(
            "tool ",
            body.as_str(),
            Style::default().add_modifier(Modifier::DIM),
        ),
        HistoryEntry::Error(body) => prefixed_lines(
            "err  ",
            body.as_str(),
            Style::default().add_modifier(Modifier::REVERSED),
        ),
    }
}

fn prefixed_lines<'a>(prefix: &'a str, body: &'a str, style: Style) -> Vec<Line<'a>> {
    body.split('\n')
        .enumerate()
        .map(move |(i, l)| {
            let prefix_span = if i == 0 {
                Span::styled(prefix, style)
            } else {
                Span::raw("     ")
            };
            Line::from(vec![prefix_span, Span::styled(l, style)])
        })
        .collect()
}

fn input_widget<'a>(state: &'a AppState) -> Paragraph<'a> {
    let lines: Vec<Line<'a>> = state
        .input()
        .lines()
        .iter()
        .map(|s| Line::from(Span::raw(s.as_str())))
        .collect();
    Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(input_title(state.mode())),
        )
        .wrap(Wrap { trim: false })
}

fn input_title(mode: Mode) -> &'static str {
    match mode {
        Mode::Idle => " input ",
        Mode::Streaming => " input (streaming...) ",
        Mode::Quitting => " input (quitting) ",
    }
}

fn footer_widget(state: &AppState) -> Paragraph<'_> {
    let body = match state.mode() {
        Mode::Idle => "Enter=send  Shift+Enter=newline  Up/Down=scroll  Ctrl+C=quit",
        Mode::Streaming => "Esc=cancel  Up/Down=scroll  (streaming...)",
        Mode::Quitting => "(quitting)",
    };
    Paragraph::new(Line::from(Span::raw(body)))
}

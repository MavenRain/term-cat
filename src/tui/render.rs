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

    let history_pane = history_widget(state);
    let history_area = chunks.first().copied().unwrap_or_default();
    f.render_widget(history_pane, history_area);

    let input_pane = input_widget(state);
    let input_area = chunks.get(1).copied().unwrap_or_default();
    f.render_widget(input_pane, input_area);

    let footer = footer_widget(state);
    let footer_area = chunks.get(2).copied().unwrap_or_default();
    f.render_widget(footer, footer_area);
}

fn history_widget<'a>(state: &'a AppState) -> Paragraph<'a> {
    let entries = state.history().entries();
    let lines: Vec<Line<'a>> = entries.iter().flat_map(entry_to_lines).collect();
    let offset = state.history().scroll_offset();
    let total = lines.len();
    let take_lines = lines
        .into_iter()
        .take(total.saturating_sub(offset))
        .collect::<Vec<_>>();
    Paragraph::new(take_lines)
        .block(Block::default().borders(Borders::ALL).title(" term-cat "))
        .wrap(Wrap { trim: false })
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

//! Multi-line input buffer.  Immutable: every edit returns a fresh
//! `InputBuffer`; the caller rebinds.

use crate::newtype::MessageBody;

/// A multi-line input buffer.  Stores lines as `Vec<String>` (one per visual
/// line) plus the cursor position `(line, column)` measured in `char`
/// boundaries.
#[derive(Debug, Clone)]
pub struct InputBuffer {
    lines: Vec<String>,
    line: usize,
    col: usize,
}

impl InputBuffer {
    /// An empty buffer with the cursor at `(0, 0)`.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            lines: vec![String::new()],
            line: 0,
            col: 0,
        }
    }

    /// Return the buffer's contents joined with `\n`, then a fresh empty
    /// buffer for the next input.
    #[must_use]
    pub fn take(self) -> (MessageBody, Self) {
        let body = MessageBody::new(self.lines.join("\n"));
        (body, Self::empty())
    }

    /// View the buffer's lines (for rendering).
    #[must_use]
    pub fn lines(&self) -> &[String] {
        &self.lines
    }

    /// Cursor position as `(line, column)`.
    #[must_use]
    pub fn cursor(&self) -> (usize, usize) {
        (self.line, self.col)
    }

    /// `true` iff the buffer contains no characters.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lines.iter().all(String::is_empty)
    }

    /// Insert a literal character at the cursor, return the updated buffer.
    #[must_use]
    pub fn insert_char(self, c: char) -> Self {
        let Self { lines, line, col } = self;
        let updated_line = lines.get(line).cloned().unwrap_or_default();
        let (head, tail) = updated_line.split_at(byte_offset_at_char(&updated_line, col));
        let new_line_text = format!("{head}{c}{tail}");
        let new_lines = replace_at(&lines, line, &new_line_text);
        Self {
            lines: new_lines,
            line,
            col: col + 1,
        }
    }

    /// Insert a hard newline at the cursor, return the updated buffer.
    #[must_use]
    pub fn insert_newline(self) -> Self {
        let Self { lines, line, col } = self;
        let current = lines.get(line).cloned().unwrap_or_default();
        let split_at = byte_offset_at_char(&current, col);
        let (head, tail) = current.split_at(split_at);
        let (head_owned, tail_owned) = (head.to_owned(), tail.to_owned());
        let new_lines: Vec<String> = lines
            .iter()
            .enumerate()
            .flat_map(|(i, l)| {
                if i == line {
                    vec![head_owned.clone(), tail_owned.clone()]
                } else {
                    vec![l.clone()]
                }
            })
            .collect();
        Self {
            lines: new_lines,
            line: line + 1,
            col: 0,
        }
    }

    /// Delete the character immediately before the cursor.  If the cursor is
    /// at `(0, 0)`, return the buffer unchanged.
    #[must_use]
    pub fn backspace(self) -> Self {
        let Self { lines, line, col } = self;
        if col == 0 && line == 0 {
            Self { lines, line, col }
        } else if col == 0 {
            let prev = lines.get(line - 1).cloned().unwrap_or_default();
            let cur = lines.get(line).cloned().unwrap_or_default();
            let new_col = prev.chars().count();
            let merged = format!("{prev}{cur}");
            let new_lines: Vec<String> = lines
                .iter()
                .enumerate()
                .filter_map(|(i, l)| {
                    if i == line - 1 {
                        Some(merged.clone())
                    } else if i == line {
                        None
                    } else {
                        Some(l.clone())
                    }
                })
                .collect();
            Self {
                lines: new_lines,
                line: line - 1,
                col: new_col,
            }
        } else {
            let current = lines.get(line).cloned().unwrap_or_default();
            let del_start_char = col - 1;
            let del_start_byte = byte_offset_at_char(&current, del_start_char);
            let del_end_byte = byte_offset_at_char(&current, col);
            let new_line_text = {
                let (h, rest) = current.split_at(del_start_byte);
                let (_, t) = rest.split_at(del_end_byte - del_start_byte);
                format!("{h}{t}")
            };
            let new_lines = replace_at(&lines, line, &new_line_text);
            Self {
                lines: new_lines,
                line,
                col: del_start_char,
            }
        }
    }

    /// Delete the previous word (Ctrl+W).  A word is a run of non-whitespace
    /// chars; we also consume trailing whitespace before it.
    #[must_use]
    pub fn delete_word(self) -> Self {
        let Self { lines, line, col } = self;
        if col == 0 {
            self_or_backspace_when_col_zero(Self { lines, line, col })
        } else {
            let current = lines.get(line).cloned().unwrap_or_default();
            let chars: Vec<char> = current.chars().collect();
            let new_col = find_word_boundary_left(&chars, col);
            let head_byte = byte_offset_at_char(&current, new_col);
            let cur_byte = byte_offset_at_char(&current, col);
            let new_line_text = {
                let (h, rest) = current.split_at(head_byte);
                let (_, t) = rest.split_at(cur_byte - head_byte);
                format!("{h}{t}")
            };
            let new_lines = replace_at(&lines, line, &new_line_text);
            Self {
                lines: new_lines,
                line,
                col: new_col,
            }
        }
    }
}

/// When `col == 0`, "delete previous word" collapses to a single backspace
/// (which itself handles the line-join case).
fn self_or_backspace_when_col_zero(buf: InputBuffer) -> InputBuffer {
    buf.backspace()
}

/// Walk leftward from `col` over whitespace, then over non-whitespace, and
/// return the resulting column.
fn find_word_boundary_left(chars: &[char], col: usize) -> usize {
    let skipped_ws = chars
        .iter()
        .take(col)
        .rev()
        .take_while(|c| c.is_whitespace())
        .count();
    let after_ws_col = col - skipped_ws;
    let skipped_word = chars
        .iter()
        .take(after_ws_col)
        .rev()
        .take_while(|c| !c.is_whitespace())
        .count();
    after_ws_col - skipped_word
}

/// Byte offset corresponding to the given char index within `s`.
fn byte_offset_at_char(s: &str, char_idx: usize) -> usize {
    s.char_indices().nth(char_idx).map_or(s.len(), |(b, _)| b)
}

/// Functionally replace `lines[idx]` with `new`, returning a fresh `Vec`.
fn replace_at(lines: &[String], idx: usize, new: &str) -> Vec<String> {
    lines
        .iter()
        .enumerate()
        .map(|(i, l)| if i == idx { new.to_owned() } else { l.clone() })
        .collect()
}

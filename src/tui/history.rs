//! Chat history: an immutable list of past turns plus a scroll position.

use crate::newtype::MessageBody;

/// One row of conversation history.
#[derive(Debug, Clone)]
pub enum HistoryEntry {
    /// The user sent this prompt.
    User(MessageBody),
    /// The assistant is currently streaming this text (may grow per
    /// `AgentEvent::AssistantToken`).
    AssistantPartial(MessageBody),
    /// The assistant finished a turn with this text.
    AssistantComplete(MessageBody),
    /// A tool was invoked.  Stringified for display.
    ToolInvoked(MessageBody),
    /// A tool returned.  Stringified for display.
    ToolReturned(MessageBody),
    /// A non-fatal error message for display.
    Error(MessageBody),
}

/// History plus a scroll offset.  The scroll offset is measured in lines from
/// the bottom; `0` means "stick to the latest entry" (auto-scroll).
#[derive(Debug, Clone)]
pub struct ChatHistory {
    entries: Vec<HistoryEntry>,
    scroll_from_bottom: usize,
}

impl ChatHistory {
    /// An empty history, auto-scrolling.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            entries: Vec::new(),
            scroll_from_bottom: 0,
        }
    }

    /// All entries, oldest first.
    #[must_use]
    pub fn entries(&self) -> &[HistoryEntry] {
        &self.entries
    }

    /// Current scroll offset (lines from bottom; `0` = auto-scroll).
    #[must_use]
    pub fn scroll_offset(&self) -> usize {
        self.scroll_from_bottom
    }

    /// Append a finished user turn.
    #[must_use]
    pub fn push_user(self, body: MessageBody) -> Self {
        let entries = appended(&self.entries, HistoryEntry::User(body));
        Self {
            entries,
            scroll_from_bottom: 0,
        }
    }

    /// Append an in-progress assistant turn (empty body).  Subsequent tokens
    /// extend it via `extend_assistant_partial`.
    #[must_use]
    pub fn start_assistant(self) -> Self {
        let entries = appended(
            &self.entries,
            HistoryEntry::AssistantPartial(MessageBody::new("")),
        );
        Self {
            entries,
            scroll_from_bottom: 0,
        }
    }

    /// Append `token` to the most recent `AssistantPartial`.  If the last
    /// entry is not an `AssistantPartial`, start one first.
    #[must_use]
    pub fn extend_assistant_partial(self, token: &MessageBody) -> Self {
        let last_partial = matches!(self.entries.last(), Some(HistoryEntry::AssistantPartial(_)));
        let prepared = if last_partial {
            self
        } else {
            self.start_assistant()
        };
        let Self {
            entries,
            scroll_from_bottom,
        } = prepared;
        let new_entries: Vec<HistoryEntry> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let is_last_partial =
                    i + 1 == entries.len() && matches!(e, HistoryEntry::AssistantPartial(_));
                if is_last_partial {
                    match e {
                        HistoryEntry::AssistantPartial(b) => {
                            HistoryEntry::AssistantPartial(b.clone().concat(token))
                        }
                        HistoryEntry::User(_)
                        | HistoryEntry::AssistantComplete(_)
                        | HistoryEntry::ToolInvoked(_)
                        | HistoryEntry::ToolReturned(_)
                        | HistoryEntry::Error(_) => e.clone(),
                    }
                } else {
                    e.clone()
                }
            })
            .collect();
        Self {
            entries: new_entries,
            scroll_from_bottom,
        }
    }

    /// Finalize the most recent `AssistantPartial` into an
    /// `AssistantComplete`.  If there is no partial, this is a no-op.
    #[must_use]
    pub fn finalize_assistant(self) -> Self {
        let Self {
            entries,
            scroll_from_bottom,
        } = self;
        let new_entries: Vec<HistoryEntry> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let is_last = i + 1 == entries.len();
                match e {
                    HistoryEntry::AssistantPartial(b) if is_last => {
                        HistoryEntry::AssistantComplete(b.clone())
                    }
                    HistoryEntry::AssistantPartial(_)
                    | HistoryEntry::User(_)
                    | HistoryEntry::AssistantComplete(_)
                    | HistoryEntry::ToolInvoked(_)
                    | HistoryEntry::ToolReturned(_)
                    | HistoryEntry::Error(_) => e.clone(),
                }
            })
            .collect();
        Self {
            entries: new_entries,
            scroll_from_bottom,
        }
    }

    /// Append a tool-invocation row.
    #[must_use]
    pub fn push_tool_invoked(self, summary: MessageBody) -> Self {
        let entries = appended(&self.entries, HistoryEntry::ToolInvoked(summary));
        Self {
            entries,
            scroll_from_bottom: 0,
        }
    }

    /// Append a tool-return row.
    #[must_use]
    pub fn push_tool_returned(self, summary: MessageBody) -> Self {
        let entries = appended(&self.entries, HistoryEntry::ToolReturned(summary));
        Self {
            entries,
            scroll_from_bottom: 0,
        }
    }

    /// Append an error row.
    #[must_use]
    pub fn push_error(self, summary: MessageBody) -> Self {
        let entries = appended(&self.entries, HistoryEntry::Error(summary));
        Self {
            entries,
            scroll_from_bottom: 0,
        }
    }

    /// Scroll `delta` lines up (towards older entries).
    #[must_use]
    pub fn scroll_up(self, delta: usize) -> Self {
        let Self {
            entries,
            scroll_from_bottom,
        } = self;
        Self {
            entries,
            scroll_from_bottom: scroll_from_bottom + delta,
        }
    }

    /// Scroll `delta` lines down (towards newer entries).
    #[must_use]
    pub fn scroll_down(self, delta: usize) -> Self {
        let Self {
            entries,
            scroll_from_bottom,
        } = self;
        let next = scroll_from_bottom.saturating_sub(delta);
        Self {
            entries,
            scroll_from_bottom: next,
        }
    }
}

fn appended(entries: &[HistoryEntry], new: HistoryEntry) -> Vec<HistoryEntry> {
    entries
        .iter()
        .cloned()
        .chain(std::iter::once(new))
        .collect()
}

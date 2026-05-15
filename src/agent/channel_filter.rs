//! Stateful streaming filter that strips channel markers from assistant
//! tokens.  SuperGemma-style models emit content like
//!
//! ```text
//! <|channel>thought\n<channel|>actual reply...
//! ```
//!
//! and we want to suppress the entire `<|channel>...<channel|>` block from
//! the UI and from the wire history.  Because markers can be split across
//! SSE token boundaries, the filter buffers any unmatched tail across calls.
//!
//! For models that emit no such markers the filter is a near-zero-cost
//! passthrough: it scans incoming tokens for `<` characters, finds none in
//! a position where they could begin a marker, and forwards every token.

/// Opening sentinel.  Everything up to the matching `CLOSE` is dropped.
const OPEN: &str = "<|channel>";

/// Closing sentinel.  Marks the end of the channel block.
const CLOSE: &str = "<channel|>";

/// Stateful filter.  Construct one per assistant turn; feed tokens in order;
/// take the cleaned output from each call.
#[derive(Debug, Clone, Default)]
pub struct ChannelFilter {
    /// Unflushed tail of input that has not yet been fully classified.  Holds
    /// either a partial marker prefix (when scanning outside) or the
    /// in-progress block contents (when scanning inside a marker).
    buffer: String,
    /// `true` iff the most recent `OPEN` has not yet been balanced by a
    /// `CLOSE`.
    inside_marker: bool,
}

impl ChannelFilter {
    /// Construct a fresh filter ready to consume the first token of a turn.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one token through the filter.  Returns the next filter state and
    /// the (possibly empty) cleaned string ready to emit.
    #[must_use]
    pub fn feed(self, token: &str) -> (Self, String) {
        let Self {
            buffer,
            inside_marker,
        } = self;
        let combined = buffer + token;
        let (output, remaining, new_inside) = consume(&combined, inside_marker);
        (
            Self {
                buffer: remaining,
                inside_marker: new_inside,
            },
            output,
        )
    }

    /// `true` iff there is no buffered tail and no open marker.  Useful for
    /// tests and for "did the turn end cleanly?" diagnostics.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.buffer.is_empty() && !self.inside_marker
    }
}

/// Drive the state machine.  Tail-recursive between `consume_close` (looking
/// for `CLOSE`) and `consume_open` (looking for `OPEN`); `comp-cat-rs`'s
/// runtime is not involved here so this is plain recursion on small inputs.
/// Returns `(output_to_emit, buffer_to_keep, inside_marker_at_end)`.
fn consume(buf: &str, inside_marker: bool) -> (String, String, bool) {
    if inside_marker {
        consume_close(buf)
    } else {
        consume_open(buf)
    }
}

/// We are inside a marker block; scan for the close sentinel.  If found,
/// drop everything up to and including it and re-enter the outside state on
/// the remainder.  If not, keep the buffer for the next call.
fn consume_close(buf: &str) -> (String, String, bool) {
    let close_pos = buf.find(CLOSE);
    let rest = close_pos.and_then(|idx| buf.get(idx + CLOSE.len()..));
    rest.map_or_else(|| (String::new(), buf.to_owned(), true), consume_open)
}

/// We are outside a marker block; emit safe content and look for the next
/// open sentinel.  If found, emit everything before it and recurse into the
/// inside state on the remainder.  If not, emit everything except the
/// longest suffix that could still grow into an `OPEN` sentinel; keep that
/// suffix buffered for the next call.
fn consume_open(buf: &str) -> (String, String, bool) {
    let pieces = buf.find(OPEN).and_then(|idx| {
        let pre = buf.get(..idx)?;
        let after = buf.get(idx + OPEN.len()..)?;
        Some((pre, after))
    });
    pieces.map_or_else(
        || {
            let cut = potential_marker_start(buf, OPEN);
            let emit = buf.get(..cut).unwrap_or("");
            let keep = buf.get(cut..).unwrap_or("");
            (emit.to_owned(), keep.to_owned(), false)
        },
        |(pre, rest)| {
            let (rest_out, remaining, inside) = consume_close(rest);
            (format!("{pre}{rest_out}"), remaining, inside)
        },
    )
}

/// Return the byte position of the longest suffix of `buf` that is a strict
/// prefix of `marker`.  Used to retain a partial marker tail for the next
/// call.  Returns `buf.len()` if no such suffix exists.
fn potential_marker_start(buf: &str, marker: &str) -> usize {
    let n = buf.len();
    let max_check = marker.len().saturating_sub(1).min(n);
    (1..=max_check)
        .rev()
        .find_map(|k| {
            buf.get(n - k..)
                .filter(|suffix| marker.starts_with(suffix))
                .map(|_| n - k)
        })
        .unwrap_or(n)
}

#[cfg(test)]
mod tests {
    use super::ChannelFilter;

    fn must_eq(actual: &str, expected: &str, label: &str) -> Result<(), String> {
        if actual == expected {
            Ok(())
        } else {
            Err(format!("{label}: expected {expected:?}, got {actual:?}"))
        }
    }

    fn must_clean(filter: &ChannelFilter, label: &str) -> Result<(), String> {
        if filter.is_clean() {
            Ok(())
        } else {
            Err(format!("{label}: filter not clean: {filter:?}"))
        }
    }

    #[test]
    fn empty_input() -> Result<(), String> {
        let (filter, out) = ChannelFilter::new().feed("");
        must_eq(&out, "", "empty out")?;
        must_clean(&filter, "empty clean")
    }

    #[test]
    fn pure_text_passes_through() -> Result<(), String> {
        let (filter, out) = ChannelFilter::new().feed("hello world");
        must_eq(&out, "hello world", "pure text")?;
        must_clean(&filter, "pure clean")
    }

    #[test]
    fn one_complete_marker_block_stripped() -> Result<(), String> {
        let input = "<|channel>thought\n<channel|>actual reply";
        let (filter, out) = ChannelFilter::new().feed(input);
        must_eq(&out, "actual reply", "single block")?;
        must_clean(&filter, "single clean")
    }

    #[test]
    fn marker_split_across_two_tokens() -> Result<(), String> {
        let (f1, out1) = ChannelFilter::new().feed("<|chan");
        must_eq(&out1, "", "partial open: nothing yet")?;
        let (f2, out2) = f1.feed("nel>thought\n<channel|>visible");
        must_eq(&out2, "visible", "marker completed across tokens")?;
        must_clean(&f2, "after second token")
    }

    #[test]
    fn close_marker_split_across_tokens() -> Result<(), String> {
        let (f1, out1) = ChannelFilter::new().feed("<|channel>thought<chann");
        must_eq(&out1, "", "open present, close partial")?;
        let (f2, out2) = f1.feed("el|>after");
        must_eq(&out2, "after", "close completed")?;
        must_clean(&f2, "after close")
    }

    #[test]
    fn text_before_marker_emitted() -> Result<(), String> {
        let (filter, out) = ChannelFilter::new().feed("prefix <|channel>x<channel|>suffix");
        must_eq(&out, "prefix suffix", "text around marker")?;
        must_clean(&filter, "after marker")
    }

    #[test]
    fn multiple_marker_blocks_in_one_token() -> Result<(), String> {
        let input = "<|channel>a<channel|>visible1<|channel>b<channel|>visible2";
        let (filter, out) = ChannelFilter::new().feed(input);
        must_eq(&out, "visible1visible2", "two blocks")?;
        must_clean(&filter, "two blocks clean")
    }

    #[test]
    fn lone_left_angle_at_end_is_buffered() -> Result<(), String> {
        let (f1, out1) = ChannelFilter::new().feed("hello <");
        must_eq(&out1, "hello ", "emit text, buffer angle")?;
        let (f2, out2) = f1.feed(" world");
        must_eq(&out2, "< world", "flush buffered angle on continuation")?;
        must_clean(&f2, "post-flush clean")
    }

    #[test]
    fn left_angle_text_that_does_not_open_a_marker() -> Result<(), String> {
        let (filter, out) = ChannelFilter::new().feed("if a < b then ...");
        must_eq(&out, "if a < b then ...", "lone < passes through")?;
        must_clean(&filter, "lone < clean")
    }

    #[test]
    fn unmatched_open_remains_buffered_at_end() -> Result<(), String> {
        // No CLOSE arrives: filter stays inside_marker, buffer holds tail.
        let (f1, out1) = ChannelFilter::new().feed("<|channel>thought");
        must_eq(&out1, "", "no close yet")?;
        if f1.is_clean() {
            return Err("filter should not be clean with unmatched open".to_owned());
        }
        Ok(())
    }

    #[test]
    fn supergemma_default_preamble() -> Result<(), String> {
        // The exact prefix observed from supergemma4 on Ollama.
        let input = "<|channel>thought\n<channel|>Hello! How can I help you today?";
        let (filter, out) = ChannelFilter::new().feed(input);
        must_eq(
            &out,
            "Hello! How can I help you today?",
            "supergemma preamble",
        )?;
        must_clean(&filter, "supergemma clean")
    }
}

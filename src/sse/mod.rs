//! Server-Sent Events parser.  Turns an SSE response body into a
//! `Stream<Error, ChatEvent>` by reading one line per pull, classifying it,
//! and recursing through `Io::flat_map` for blank or skipped lines (so we
//! never use `loop` / `while` / `for`).

pub mod parser;

pub use parser::{SseReader, SseState, parse, step};

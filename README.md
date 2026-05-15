# term-cat

A Claude-like terminal UI for chatting with local OpenAI-compatible LLM servers.  Built on [`comp-cat-rs`](https://crates.io/crates/comp-cat-rs) (categorical effects) and [`rig-cat`](https://crates.io/crates/rig-cat) (LLM agent framework).

No tokio, no async functions, no `unsafe`, no `unwrap`.

## Features

- Streaming token render over Server-Sent Events.
- Multi-line input (`Shift+Enter` for newline, `Enter` to send).
- Scrolling chat history.
- Tool-calling loop: the model can request function calls, term-cat invokes the tool, the conversation resumes, all visible in the history pane.
- Two built-in tools: `echo` (returns its arguments unchanged) and `now` (returns the current Unix timestamp).
- Mid-stream cancellation with `Esc`; clean terminal restoration on exit.

## Prerequisites

- A running [LM Studio](https://lmstudio.ai/) server (or any OpenAI-compatible local server) listening on a `http://.../v1` base.
- A loaded chat-completion model.  Tested with `supergemma4-26b-uncensored-fast-v2` Q4_K_M on Apple M2 Pro / 32 GB.  Tool-calling features require a model that supports function calling.

## Build

```sh
cargo build --release
```

## Run

```sh
cargo run --release
```

## Configuration

term-cat reads two environment variables, both optional:

| Variable              | Default                            | Purpose                                                      |
|-----------------------|------------------------------------|--------------------------------------------------------------|
| `TERM_CAT_BASE_URL`   | `http://localhost:1234/v1`         | Base URL of the OpenAI-compatible server.                    |
| `TERM_CAT_MODEL`      | `local-model`                      | Model identifier sent in each request.  LM Studio ignores it. |

Example:

```sh
export TERM_CAT_BASE_URL=http://localhost:8080/v1   # llama-server default
export TERM_CAT_MODEL=Qwen2.5-7B-Instruct
cargo run --release
```

## Keybindings

| Key                  | Action                                    |
|----------------------|-------------------------------------------|
| Enter                | Send the current input                    |
| Shift+Enter          | Insert a newline in the input             |
| Backspace            | Delete the previous character             |
| Ctrl+W               | Delete the previous word                  |
| Up / Down            | Scroll history one line                   |
| PageUp / PageDown    | Scroll history one page                   |
| Esc                  | Cancel an in-flight stream                |
| Ctrl+C / Ctrl+D      | Quit when idle (cancels first if active)  |

## Example interaction

```
> what time is it?
[tool now invoked: {}]
[tool now returned: {"unix_timestamp": 1747346400}]
The current time is around 17:00 UTC on 2025-05-15.
```

The lines starting with `[tool ...]` are rendered as dim history entries; the assistant's follow-up text streams in as usual.

## Custom tools

The default binary ships with `BuiltinTool` (an enum over `EchoTool` and `NowTool`).  To plug in your own tools:

1. Define a unit struct (or a struct with private state) and implement `rig_cat::tool::Tool`.
2. Define a sum-type enum that dispatches to each tool variant.
3. Construct `StreamingAgent::new(provider, vec![YourTool::A(A), YourTool::B(B), ...])`.

A minimal example:

```rust
use comp_cat_rs::effect::io::Io;
use rig_cat::tool::{Tool, ToolDefinition};
use serde_json::Value;

#[derive(Debug, Clone, Copy)]
struct AddTool;

impl Tool for AddTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "add".to_owned(),
            "Add two integers".to_owned(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "a": { "type": "integer" },
                    "b": { "type": "integer" }
                },
                "required": ["a", "b"]
            }),
        )
    }

    fn call(&self, args: Value) -> Io<rig_cat::error::Error, Value> {
        Io::suspend(move || {
            let a = args.get("a").and_then(Value::as_i64).unwrap_or(0);
            let b = args.get("b").and_then(Value::as_i64).unwrap_or(0);
            Ok(serde_json::json!({ "sum": a + b }))
        })
    }
}
```

## Architecture

See `CLAUDE.md` for the coding conventions this crate follows.  Short version: every fallible function returns `Io<Error, _>` or `Result<_, Error>`; the TUI itself is modeled as a `Stream<Error, ()>` of frames driven by `Stream::unfold`; tool-calling is a `LoopState` state machine that recursively threads through SSE chunks, accumulator updates, and dispatch steps.

Source layout:

- `src/wire/` — OpenAI Chat Completions JSON shapes (request, response, streaming chunk).
- `src/sse/` — Server-Sent Events parser that produces `Stream<Error, ChatEvent>`.
- `src/provider/` — `LocalOpenAiCompletion` for blocking `complete` and streaming `stream_chat` / `open_request`.
- `src/agent/` — `StreamingAgent`, the `LoopState` state machine, `ToolCallAccumulator`, built-in tools.
- `src/bridge/` — `cancel_channel` and `spawn_agent_turn` for inter-thread plumbing.
- `src/tui/` — ratatui + crossterm frame loop, input buffer, history, key mapping, render.

## Testing

```sh
cargo test
```

Integration tests cover the SSE chunk projection, tool-call accumulator, built-in tools, and the entire `KeyCode + KeyModifiers -> KeyAction` mapping.  Unit tests inside `sse::parser` cover the `classify` line-level classifier.

## Roadmap

Out of v1: markdown rendering with syntax highlighting, slash commands (`/clear`, `/save`, `/load`, `/model`), conversation persistence to disk, image / file attachments, RAG, mid-call cancellation finer-grained than one SSE line.

## License

Dual MIT OR Apache-2.0, at your option.

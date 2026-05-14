# term-cat

A Claude-like terminal UI for chatting with local OpenAI-compatible LLM servers.  Built on [`comp-cat-rs`](https://crates.io/crates/comp-cat-rs) (categorical effects) and [`rig-cat`](https://crates.io/crates/rig-cat) (LLM agent framework).

No tokio, no async functions, no `unsafe`, no `unwrap`.

## Features

- Streaming token render over Server-Sent Events.
- Multi-line input (`Shift+Enter` for newline, `Enter` to send).
- Scrolling chat history.
- Tool-calling loop: the model can request function calls, term-cat invokes the tool, the conversation resumes.
- Mid-stream cancellation with `Esc`.

## Prerequisites

- A running [LM Studio](https://lmstudio.ai/) server (or any OpenAI-compatible local server) on `http://localhost:1234/v1`.
- A loaded chat-completion model.  Tested with `supergemma4-26b-uncensored-fast-v2` Q4_K_M on Apple M2 Pro / 32 GB.

## Build

```sh
cargo build --release
```

## Run

```sh
cargo run --release -- --model your-loaded-model-id
```

## Keybindings

| Key                  | Action                            |
|----------------------|-----------------------------------|
| Enter                | Send the current input            |
| Shift+Enter          | Insert a newline in the input     |
| Backspace            | Delete the previous character     |
| Ctrl+W               | Delete the previous word          |
| Up / Down            | Scroll history one line           |
| PageUp / PageDown    | Scroll history one page           |
| Esc                  | Cancel an in-flight stream        |
| Ctrl+C / Ctrl+D      | Quit when idle                    |

## Architecture

See `CLAUDE.md` for the rules the codebase follows, and the design plan at the top of `src/lib.rs`.

## Roadmap

Out of v1: markdown rendering with syntax highlighting, slash commands (`/clear`, `/save`, `/load`, `/model`), conversation persistence, image / file attachments, RAG.

## License

Dual MIT OR Apache-2.0, at your option.

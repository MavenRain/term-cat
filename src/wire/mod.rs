//! JSON shapes for the `OpenAI`-compatible Chat Completions API.
//!
//! These types are owned by `term-cat` rather than re-using rig-cat's
//! `Message`/`CompletionRequest`/`CompletionResponse` because:
//!
//! 1. rig-cat's `Role` enum has no `Tool` variant and `Message` has no
//!    tool-call attribution, both required by the `OpenAI` tool-calling
//!    protocol.
//! 2. rig-cat's `CompletionRequest` has no `tools`, `stream`, or model field.
//!
//! Phase B uses only `Role::{System, User, Assistant}` and ignores the tool
//! fields.  Phase D will add tool-call constructors and tool messages.

pub mod chunk;
pub mod message;
pub mod request;
pub mod response;

pub use chunk::ChatCompletionChunk;
pub use message::{Message, Role, ToolCall, ToolCallFunction, ToolCallKind};
pub use request::ChatCompletionRequest;
pub use response::{ChatCompletionResponse, ResponseChoice, ResponseMessage};

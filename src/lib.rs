//! `term-cat` — a Claude-like terminal UI for local OpenAI-compatible LLM
//! servers, built on `comp-cat-rs` (categorical effects) and `rig-cat` (LLM
//! agent framework).
//!
//! See `CLAUDE.md` at the repo root for the coding conventions this crate
//! follows, and the design plan that prefaces each subsystem.

#![forbid(unsafe_code)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]

pub mod agent;
pub mod bridge;
pub mod error;
pub mod newtype;
pub mod tui;

pub use error::{BridgeError, Error, ProviderError, SseError, TuiError, WireError};
pub use newtype::{
    ApiKey, BaseUrl, FinishReason, MaxTokens, MessageBody, ModelName, Temperature, ToolCallId,
    ToolCallIndex, ToolName,
};

use comp_cat_rs::effect::fiber::FiberError;

impl From<FiberError<Error>> for Error {
    fn from(fe: FiberError<Error>) -> Self {
        match fe {
            FiberError::Failed(e) => e,
            FiberError::Panicked(s) => Self::Bridge(BridgeError::FiberPanicked(s)),
            FiberError::SpawnFailed(io) => Self::Bridge(BridgeError::FiberSpawn(io)),
        }
    }
}

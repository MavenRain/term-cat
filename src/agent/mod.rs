//! Agent event types and the streaming tool-calling loop.

pub mod accumulator;
pub mod builtin;
pub mod event;
pub mod loop_state;
pub mod streaming;

pub use accumulator::{PendingCall, ToolCallAccumulator};
pub use builtin::{BuiltinTool, EchoTool, NowTool};
pub use event::{AgentEvent, ChatEvent};
pub use loop_state::{LoopState, loop_step};
pub use streaming::StreamingAgent;

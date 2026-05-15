//! Inter-thread plumbing: cancel signalling and producer-fiber spawning.

pub mod cancel;
pub mod channel;

pub use cancel::{CancelObserver, Canceller, cancel_channel};
pub use channel::{history_to_wire, spawn_streaming_fiber};

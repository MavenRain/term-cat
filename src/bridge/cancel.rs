//! Newtype wrappers around `std::sync::mpsc::{Sender, Receiver}<()>` used to
//! signal stream cancellation from the TUI thread to the producer fiber.
//!
//! We deliberately use `mpsc` rather than `Arc<AtomicBool>` so that no `Arc`
//! is constructed in our crate's source for cancellation purposes.

use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};

/// TUI-side handle for signalling cancellation.  Cheap to clone; the producer
/// stops on the first `()` it observes via `is_cancelled`.
#[derive(Debug, Clone)]
pub struct Canceller(Sender<()>);

impl Canceller {
    /// Signal cancellation.  No error is surfaced if the receiver has already
    /// been dropped; the producer is already stopped in that case.
    pub fn signal(&self) {
        let _ = self.0.send(());
    }
}

/// Producer-side observer.  Not `Clone` (`mpsc::Receiver` is not `Clone`), but
/// the sender side may have many `Canceller` clones.
#[derive(Debug)]
pub struct CancelObserver(Receiver<()>);

impl CancelObserver {
    /// `true` if a cancel signal has arrived, or if the sender side has been
    /// dropped (meaning the TUI is gone).
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.try_recv().map_or_else(
            |e| match e {
                TryRecvError::Disconnected => true,
                TryRecvError::Empty => false,
            },
            |()| true,
        )
    }
}

/// Construct a fresh cancel channel.
#[must_use]
pub fn cancel_channel() -> (Canceller, CancelObserver) {
    let (tx, rx) = mpsc::channel();
    (Canceller(tx), CancelObserver(rx))
}

//! Shared helpers for integration tests.  Lives at `tests/common/mod.rs` so
//! cargo does not compile it as a standalone test binary; each integration
//! test file pulls it in with `mod common;`.
//!
//! Helpers may appear unused from any single test binary's perspective
//! (cargo compiles each integration test file as a separate binary).  The
//! `dead_code` allow is necessary to keep clippy `-D warnings` clean across
//! the per-binary view.

#![allow(dead_code)]

use std::fmt;

use term_cat::Error;

/// Error type used by integration tests.  Wraps domain errors (so `?` works
/// over `term_cat::Error`) and assertion failures (built by `require_eq` and
/// the other helpers below).
#[derive(Debug)]
pub enum TestError {
    Domain(Error),
    Assertion(String),
}

impl fmt::Display for TestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Domain(e) => write!(f, "domain: {e}"),
            Self::Assertion(s) => write!(f, "assertion: {s}"),
        }
    }
}

impl std::error::Error for TestError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Domain(e) => Some(e),
            Self::Assertion(_) => None,
        }
    }
}

impl From<Error> for TestError {
    fn from(e: Error) -> Self {
        Self::Domain(e)
    }
}

/// Compare two values; return `Err(TestError::Assertion)` with both values'
/// `Debug` representations if they differ.  Used in place of `assert_eq!`,
/// which is prohibited by the no-panics-anywhere convention.
pub fn require_eq<T: PartialEq + fmt::Debug>(
    actual: &T,
    expected: &T,
    label: &str,
) -> Result<(), TestError> {
    if actual == expected {
        Ok(())
    } else {
        Err(TestError::Assertion(format!(
            "{label}: expected {expected:?}, got {actual:?}"
        )))
    }
}

/// Assert that a `Result` is `Ok`, returning the value.  Avoids `unwrap()`.
pub fn require_ok<T, E: fmt::Debug>(r: Result<T, E>, label: &str) -> Result<T, TestError> {
    r.map_err(|e| TestError::Assertion(format!("{label}: expected Ok, got Err({e:?})")))
}

/// Assert that a `Result` is `Err`, returning the error.  Avoids `unwrap()`.
pub fn require_err<T: fmt::Debug, E>(r: Result<T, E>, label: &str) -> Result<E, TestError> {
    let label_owned = label.to_owned();
    r.map_or_else(Ok, move |t| {
        Err(TestError::Assertion(format!(
            "{label_owned}: expected Err, got Ok({t:?})"
        )))
    })
}

/// Assert that a boolean condition holds, with a descriptive label on failure.
pub fn require_true(cond: bool, label: &str) -> Result<(), TestError> {
    if cond {
        Ok(())
    } else {
        Err(TestError::Assertion(format!("{label}: expected true")))
    }
}

//! Integration tests for `term_cat::agent::ToolCallAccumulator`.  Each test
//! threads state through `add_start` / `add_args` and verifies behaviour via
//! `finalize` + `PendingCall` accessors.  No `assert_eq!` / `unwrap()`.

mod common;

use common::{TestError, require_eq, require_true};
use term_cat::agent::{PendingCall, ToolCallAccumulator};
use term_cat::newtype::{MessageBody, ToolCallId, ToolCallIndex, ToolName};

fn call_at(pending: &[PendingCall], idx: usize) -> Option<&PendingCall> {
    pending.get(idx)
}

#[test]
fn empty_accumulator_is_empty() -> Result<(), TestError> {
    let acc = ToolCallAccumulator::empty();
    require_true(acc.is_empty(), "empty accumulator")
}

#[test]
fn add_start_creates_one_pending() -> Result<(), TestError> {
    let acc = ToolCallAccumulator::empty().add_start(
        ToolCallIndex::new(0),
        ToolCallId::new("call_0"),
        ToolName::new("echo"),
    );
    let pending = acc.finalize();
    require_eq(&pending.len(), &1usize, "pending count")?;
    let first = call_at(&pending, 0)
        .ok_or_else(|| TestError::Assertion("missing first pending".to_owned()))?;
    require_eq(&first.id().as_str(), &"call_0", "first id")?;
    require_eq(&first.name().as_str(), &"echo", "first name")?;
    require_eq(&first.args_buf(), &"", "first args (empty)")
}

#[test]
fn add_args_concatenates_under_same_index() -> Result<(), TestError> {
    let acc = ToolCallAccumulator::empty()
        .add_start(
            ToolCallIndex::new(0),
            ToolCallId::new("c0"),
            ToolName::new("echo"),
        )
        .add_args(ToolCallIndex::new(0), MessageBody::new("{\"msg\":"))
        .add_args(ToolCallIndex::new(0), MessageBody::new("\"hi\"}"));
    let pending = acc.finalize();
    let first = call_at(&pending, 0)
        .ok_or_else(|| TestError::Assertion("missing first pending".to_owned()))?;
    require_eq(&first.args_buf(), &"{\"msg\":\"hi\"}", "concatenated args")
}

#[test]
fn add_args_to_unknown_index_is_noop() -> Result<(), TestError> {
    let acc = ToolCallAccumulator::empty()
        .add_start(
            ToolCallIndex::new(0),
            ToolCallId::new("c0"),
            ToolName::new("echo"),
        )
        .add_args(ToolCallIndex::new(99), MessageBody::new("ignored"));
    let pending = acc.finalize();
    require_eq(&pending.len(), &1usize, "still one pending")?;
    let first = call_at(&pending, 0)
        .ok_or_else(|| TestError::Assertion("missing first pending".to_owned()))?;
    require_eq(&first.args_buf(), &"", "args still empty")
}

#[test]
fn multi_index_interleaved_args() -> Result<(), TestError> {
    let acc = ToolCallAccumulator::empty()
        .add_start(
            ToolCallIndex::new(0),
            ToolCallId::new("c0"),
            ToolName::new("echo"),
        )
        .add_start(
            ToolCallIndex::new(1),
            ToolCallId::new("c1"),
            ToolName::new("now"),
        )
        .add_args(ToolCallIndex::new(0), MessageBody::new("foo"))
        .add_args(ToolCallIndex::new(1), MessageBody::new("bar"))
        .add_args(ToolCallIndex::new(0), MessageBody::new("baz"));
    let pending = acc.finalize();
    require_eq(&pending.len(), &2usize, "two pending")?;
    let first = call_at(&pending, 0)
        .ok_or_else(|| TestError::Assertion("missing first pending".to_owned()))?;
    let second = call_at(&pending, 1)
        .ok_or_else(|| TestError::Assertion("missing second pending".to_owned()))?;
    require_eq(&first.args_buf(), &"foobaz", "index 0 args")?;
    require_eq(&second.args_buf(), &"bar", "index 1 args")
}

#[test]
fn finalize_preserves_stream_order() -> Result<(), TestError> {
    let acc = ToolCallAccumulator::empty()
        .add_start(
            ToolCallIndex::new(0),
            ToolCallId::new("c0"),
            ToolName::new("first"),
        )
        .add_start(
            ToolCallIndex::new(1),
            ToolCallId::new("c1"),
            ToolName::new("second"),
        )
        .add_start(
            ToolCallIndex::new(2),
            ToolCallId::new("c2"),
            ToolName::new("third"),
        );
    let pending = acc.finalize();
    require_eq(&pending.len(), &3usize, "three pending")?;
    let names: Vec<String> = pending
        .iter()
        .map(|c| c.name().as_str().to_owned())
        .collect();
    require_eq(
        &names,
        &vec!["first".to_owned(), "second".to_owned(), "third".to_owned()],
        "stream order",
    )
}

#[test]
fn to_wire_round_trips_fields() -> Result<(), TestError> {
    let acc = ToolCallAccumulator::empty()
        .add_start(
            ToolCallIndex::new(0),
            ToolCallId::new("call_42"),
            ToolName::new("greet"),
        )
        .add_args(ToolCallIndex::new(0), MessageBody::new("{\"name\":\"x\"}"));
    let pending = acc.finalize();
    let first = call_at(&pending, 0)
        .ok_or_else(|| TestError::Assertion("missing first pending".to_owned()))?;
    let wire = first.to_wire();
    require_eq(&wire.id().as_str(), &"call_42", "wire id")?;
    require_eq(&wire.function().name().as_str(), &"greet", "wire fn name")?;
    require_eq(
        &wire.function().arguments(),
        &"{\"name\":\"x\"}",
        "wire arguments",
    )
}

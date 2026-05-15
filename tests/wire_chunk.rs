//! Integration tests for `ChatCompletionChunk::into_events`.  Verifies that
//! the SSE delta projection produces the right `ChatEvent` sequence for the
//! common cases the `OpenAI` streaming spec defines, and that unknown
//! `finish_reason` values surface as `Error::Wire`.

mod common;

use common::{TestError, require_eq, require_err, require_ok, require_true};
use term_cat::agent::ChatEvent;
use term_cat::newtype::FinishReason;
use term_cat::wire::ChatCompletionChunk;
use term_cat::{Error, WireError};

fn parse_chunk(json: &str) -> Result<ChatCompletionChunk, TestError> {
    require_ok(
        serde_json::from_str::<ChatCompletionChunk>(json),
        "parse chunk",
    )
}

#[test]
fn content_delta_lowers_to_token() -> Result<(), TestError> {
    let chunk = parse_chunk(r#"{"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}"#)?;
    let events = require_ok(chunk.into_events(), "into_events")?;
    require_eq(&events.len(), &1usize, "event count")?;
    let first = events
        .first()
        .ok_or_else(|| TestError::Assertion("no first event".to_owned()))?;
    let body = match first {
        ChatEvent::Token(body) => Some(body.as_str().to_owned()),
        ChatEvent::ToolCallStart(_, _, _)
        | ChatEvent::ToolCallArgs(_, _)
        | ChatEvent::Finished(_) => None,
    };
    require_eq(&body, &Some("hello".to_owned()), "token body")
}

#[test]
fn finish_reason_stop_lowers_to_finished_stop() -> Result<(), TestError> {
    let chunk = parse_chunk(r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#)?;
    let events = require_ok(chunk.into_events(), "into_events")?;
    require_eq(&events.len(), &1usize, "event count")?;
    let first = events
        .first()
        .ok_or_else(|| TestError::Assertion("no first event".to_owned()))?;
    let reason = match first {
        ChatEvent::Finished(r) => Some(*r),
        ChatEvent::Token(_) | ChatEvent::ToolCallStart(_, _, _) | ChatEvent::ToolCallArgs(_, _) => {
            None
        }
    };
    require_eq(&reason, &Some(FinishReason::Stop), "finish reason")
}

#[test]
fn finish_reason_tool_calls_lowers_to_tool_calls() -> Result<(), TestError> {
    let chunk = parse_chunk(r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#)?;
    let events = require_ok(chunk.into_events(), "into_events")?;
    let reason = events.first().and_then(|e| match e {
        ChatEvent::Finished(r) => Some(*r),
        ChatEvent::Token(_) | ChatEvent::ToolCallStart(_, _, _) | ChatEvent::ToolCallArgs(_, _) => {
            None
        }
    });
    require_eq(&reason, &Some(FinishReason::ToolCalls), "finish reason")
}

#[test]
fn unknown_finish_reason_returns_wire_error() -> Result<(), TestError> {
    let chunk = parse_chunk(r#"{"choices":[{"delta":{},"finish_reason":"unknown_reason"}]}"#)?;
    let err = require_err(chunk.into_events(), "into_events unknown reason")?;
    let is_unknown = matches!(err, Error::Wire(WireError::UnknownFinishReason(_)),);
    require_true(is_unknown, "Error::Wire(UnknownFinishReason)")
}

#[test]
fn tool_call_start_lowers_to_tool_call_start() -> Result<(), TestError> {
    let chunk = parse_chunk(
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_xyz","type":"function","function":{"name":"echo","arguments":""}}]},"finish_reason":null}]}"#,
    )?;
    let events = require_ok(chunk.into_events(), "into_events")?;
    require_eq(&events.len(), &1usize, "event count")?;
    let first = events
        .first()
        .ok_or_else(|| TestError::Assertion("no first event".to_owned()))?;
    let id_and_name = match first {
        ChatEvent::ToolCallStart(_, id, name) => {
            Some((id.as_str().to_owned(), name.as_str().to_owned()))
        }
        ChatEvent::Token(_) | ChatEvent::ToolCallArgs(_, _) | ChatEvent::Finished(_) => None,
    };
    require_eq(
        &id_and_name,
        &Some(("call_xyz".to_owned(), "echo".to_owned())),
        "tool call id+name",
    )
}

#[test]
fn tool_call_args_fragment_lowers_to_tool_call_args() -> Result<(), TestError> {
    // No id, no name: only an args fragment for an already-started call at
    // index 0.  Per OpenAI's spec, follow-up chunks carry only arguments.
    let chunk = parse_chunk(
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"msg\":"}}]},"finish_reason":null}]}"#,
    )?;
    let events = require_ok(chunk.into_events(), "into_events")?;
    require_eq(&events.len(), &1usize, "event count")?;
    let first = events
        .first()
        .ok_or_else(|| TestError::Assertion("no first event".to_owned()))?;
    let args = match first {
        ChatEvent::ToolCallArgs(_, body) => Some(body.as_str().to_owned()),
        ChatEvent::Token(_) | ChatEvent::ToolCallStart(_, _, _) | ChatEvent::Finished(_) => None,
    };
    require_eq(&args, &Some("{\"msg\":".to_owned()), "args fragment")
}

#[test]
fn empty_choices_yields_empty_events() -> Result<(), TestError> {
    let chunk = parse_chunk(r#"{"choices":[]}"#)?;
    let events = require_ok(chunk.into_events(), "into_events")?;
    require_eq(&events.len(), &0usize, "no events")
}

#[test]
fn content_plus_finish_lowers_to_two_events() -> Result<(), TestError> {
    let chunk = parse_chunk(r#"{"choices":[{"delta":{"content":"bye"},"finish_reason":"stop"}]}"#)?;
    let events = require_ok(chunk.into_events(), "into_events")?;
    require_eq(&events.len(), &2usize, "event count")
}

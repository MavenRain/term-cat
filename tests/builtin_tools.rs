//! Integration tests for `EchoTool` and `NowTool`.  Each test runs the tool's
//! `call` to completion via `.run()` and inspects the returned JSON.  Tool
//! definitions are also verified for shape.

mod common;

use common::{TestError, require_eq, require_ok, require_true};
use rig_cat::tool::Tool;
use serde_json::{Value, json};
use term_cat::agent::{BuiltinTool, EchoTool, NowTool};

#[test]
fn echo_tool_returns_args_unchanged() -> Result<(), TestError> {
    let tool = EchoTool;
    let args = json!({ "message": "hello world" });
    let result = require_ok(tool.call(args.clone()).run(), "echo call")?;
    require_eq(&result, &args, "echo result")
}

#[test]
fn echo_tool_definition_has_expected_name() -> Result<(), TestError> {
    let tool = EchoTool;
    let def = tool.definition();
    require_eq(&def.name(), &"echo", "echo def name")
}

#[test]
fn now_tool_returns_object_with_unix_timestamp() -> Result<(), TestError> {
    let tool = NowTool;
    let result = require_ok(tool.call(Value::Null).run(), "now call")?;
    let ts = result.get("unix_timestamp").ok_or_else(|| {
        TestError::Assertion(format!("now result missing unix_timestamp: {result:?}"))
    })?;
    require_true(ts.is_u64(), "unix_timestamp is u64")?;
    let secs = ts
        .as_u64()
        .ok_or_else(|| TestError::Assertion("unix_timestamp not coercible to u64".to_owned()))?;
    // Sanity: timestamp should be after 2020 (any year we'd reasonably run
    // this code).  2020-01-01T00:00:00Z is 1577836800.
    require_true(secs > 1_577_836_800, "unix_timestamp is plausibly recent")
}

#[test]
fn now_tool_definition_has_expected_name() -> Result<(), TestError> {
    let tool = NowTool;
    let def = tool.definition();
    require_eq(&def.name(), &"now", "now def name")
}

#[test]
fn builtin_tool_echo_dispatches_to_echo() -> Result<(), TestError> {
    let tool = BuiltinTool::Echo(EchoTool);
    let args = json!({ "message": "hi" });
    let result = require_ok(tool.call(args.clone()).run(), "BuiltinTool::Echo call")?;
    require_eq(&result, &args, "echo via BuiltinTool result")
}

#[test]
fn builtin_tool_echo_definition_is_echos() -> Result<(), TestError> {
    let tool = BuiltinTool::Echo(EchoTool);
    require_eq(
        &tool.definition().name(),
        &"echo",
        "BuiltinTool::Echo definition name",
    )
}

#[test]
fn builtin_tool_now_dispatches_to_now() -> Result<(), TestError> {
    let tool = BuiltinTool::Now(NowTool);
    let result = require_ok(tool.call(Value::Null).run(), "BuiltinTool::Now call")?;
    require_true(
        result.get("unix_timestamp").is_some(),
        "BuiltinTool::Now has unix_timestamp",
    )
}

#[test]
fn builtin_tool_now_definition_is_nows() -> Result<(), TestError> {
    let tool = BuiltinTool::Now(NowTool);
    require_eq(
        &tool.definition().name(),
        &"now",
        "BuiltinTool::Now definition name",
    )
}

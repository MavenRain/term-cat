//! Built-in tools shipped with the term-cat binary: `EchoTool` (returns its
//! input unchanged) and `NowTool` (returns the current Unix timestamp).
//!
//! These exist so the Phase D tool-calling loop has something to dispatch
//! when a tool-capable model decides to use it.  They also serve as canonical
//! examples for users who want to add their own tools: write a unit struct,
//! impl `rig_cat::tool::Tool`, add a variant to a custom enum, and pass that
//! enum to `StreamingAgent::new`.

use comp_cat_rs::effect::io::Io;
use rig_cat::tool::{Tool, ToolDefinition};
use serde_json::Value;

/// Echoes the JSON value it was given back as the tool's result.  Useful as a
/// smoke test: the model should be able to call this with `{"message":
/// "hello"}` and observe `{"message": "hello"}` in the tool result.
#[derive(Debug, Clone, Copy)]
pub struct EchoTool;

impl Tool for EchoTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "echo".to_owned(),
            "Echo the JSON arguments back to the caller unchanged.  Useful as a smoke test."
                .to_owned(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "the message to echo"
                    }
                },
                "required": ["message"]
            }),
        )
    }

    fn call(&self, args: Value) -> Io<rig_cat::error::Error, Value> {
        Io::pure(args)
    }
}

/// Returns the current Unix timestamp (seconds since the UNIX epoch, UTC) as
/// a JSON object `{"unix_timestamp": <u64>}`.  Kept as a Unix timestamp to
/// avoid a `chrono` dependency; the model can convert if it needs a human
/// format.
#[derive(Debug, Clone, Copy)]
pub struct NowTool;

impl Tool for NowTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "now".to_owned(),
            "Return the current time as a Unix timestamp (seconds since 1970-01-01 UTC)."
                .to_owned(),
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
        )
    }

    fn call(&self, _args: Value) -> Io<rig_cat::error::Error, Value> {
        Io::suspend(|| {
            let secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs());
            Ok(serde_json::json!({ "unix_timestamp": secs }))
        })
    }
}

/// Heterogeneous tool dispatcher for the default term-cat binary.  Custom
/// builds can replace this with their own enum that includes additional
/// tool variants.
#[derive(Debug, Clone, Copy)]
pub enum BuiltinTool {
    Echo(EchoTool),
    Now(NowTool),
}

impl Tool for BuiltinTool {
    fn definition(&self) -> ToolDefinition {
        match self {
            Self::Echo(t) => t.definition(),
            Self::Now(t) => t.definition(),
        }
    }

    fn call(&self, args: Value) -> Io<rig_cat::error::Error, Value> {
        match self {
            Self::Echo(t) => t.call(args),
            Self::Now(t) => t.call(args),
        }
    }
}

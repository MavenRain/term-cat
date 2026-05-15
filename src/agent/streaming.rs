//! `StreamingAgent`: a tool-aware wrapper around `LocalOpenAiCompletion`.
//! Holds the tools list, optional preamble, and sampling parameters.  Each
//! call to `turn` returns a `Stream<Error, AgentEvent>` that drives the
//! multi-iteration tool-calling loop.

use std::sync::Arc;

use comp_cat_rs::effect::io::Io;
use comp_cat_rs::effect::stream::Stream;
use rig_cat::tool::{Tool, Toolbox};

use crate::agent::event::AgentEvent;
use crate::agent::loop_state::{LoopState, loop_step};
use crate::bridge::CancelObserver;
use crate::error::Error;
use crate::newtype::{MaxTokens, MessageBody, Temperature};
use crate::provider::LocalOpenAiCompletion;
use crate::wire::{ChatCompletionRequest, Message};

/// Tool-aware agent: a `LocalOpenAiCompletion` plus a list of tools and
/// optional sampling settings.  Generic over `T: Tool + Clone` so the tools
/// list can be cloned for fresh `Toolbox` construction per dispatch (rig-cat
/// 0.1 does not expose an iterator over `Toolbox`).
#[derive(Debug, Clone)]
pub struct StreamingAgent<T: Tool + Clone> {
    provider: LocalOpenAiCompletion,
    tools: Vec<T>,
    preamble: Option<MessageBody>,
    temperature: Option<Temperature>,
    max_tokens: Option<MaxTokens>,
}

impl<T: Tool + Clone> StreamingAgent<T> {
    /// Construct an agent over the given provider with the given tools.
    #[must_use]
    pub fn new(provider: LocalOpenAiCompletion, tools: Vec<T>) -> Self {
        Self {
            provider,
            tools,
            preamble: None,
            temperature: None,
            max_tokens: None,
        }
    }

    /// Set a system preamble that will be prepended to the wire history on
    /// the first iteration of every turn.
    #[must_use]
    pub fn with_preamble(self, body: MessageBody) -> Self {
        Self {
            preamble: Some(body),
            ..self
        }
    }

    /// Set sampling temperature.
    #[must_use]
    pub fn with_temperature(self, t: Temperature) -> Self {
        Self {
            temperature: Some(t),
            ..self
        }
    }

    /// Set max-tokens.
    #[must_use]
    pub fn with_max_tokens(self, n: MaxTokens) -> Self {
        Self {
            max_tokens: Some(n),
            ..self
        }
    }

    /// Borrowed view of the underlying provider.
    #[must_use]
    pub fn provider(&self) -> &LocalOpenAiCompletion {
        &self.provider
    }

    /// `temperature` setting (if any).
    #[must_use]
    pub fn temperature(&self) -> Option<Temperature> {
        self.temperature
    }

    /// `max_tokens` setting (if any).
    #[must_use]
    pub fn max_tokens(&self) -> Option<MaxTokens> {
        self.max_tokens
    }

    /// Construct a fresh `Toolbox<T>` from the agent's tools.  Called once per
    /// dispatch site; rig-cat 0.1's `Toolbox` is not `Clone`, so we rebuild
    /// from the agent's owned tool list.
    #[must_use]
    pub fn toolbox(&self) -> Toolbox<T> {
        self.tools
            .iter()
            .cloned()
            .fold(Toolbox::new(), Toolbox::with_tool)
    }

    /// Build a `ChatCompletionRequest` for the agent's configuration with the
    /// given wire history.  Includes `tools[]` iff the agent has any tools.
    #[must_use]
    pub fn build_request(&self, messages: Vec<Message>) -> ChatCompletionRequest {
        let tool_defs = self.toolbox().definitions();
        let request = ChatCompletionRequest::new(self.provider.model().clone(), messages)
            .maybe_temperature(self.temperature)
            .maybe_max_tokens(self.max_tokens)
            .with_stream(true);
        if tool_defs.is_empty() {
            request
        } else {
            request.with_tools(tool_defs)
        }
    }
}

impl<T: Tool + Clone + Send + Sync + 'static> StreamingAgent<T> {
    /// Initiate a multi-iteration tool-calling turn.  Returns a stream of
    /// `AgentEvent`s emitted in order:
    ///
    ///   * `AssistantToken(_)` for each text fragment the model emits.
    ///   * `ToolInvoked` / `ToolReturned` pairs for each tool call.
    ///   * Followed by more `AssistantToken`s if the model produces a
    ///     follow-up assistant turn after tool results.
    ///
    /// The stream ends when the model emits a `Finished(Stop|Length|
    /// ContentFilter)` event or the cancel signal is observed.  The bridge
    /// layer is responsible for emitting `AgentEvent::TurnDone` after the
    /// stream completes.
    #[must_use]
    pub fn turn(
        self,
        initial_history: Vec<Message>,
        cancel: CancelObserver,
    ) -> Stream<Error, AgentEvent> {
        let Self {
            provider,
            tools,
            preamble,
            temperature,
            max_tokens,
        } = self;
        let full_history: Vec<Message> = preamble
            .iter()
            .map(|body| Message::system(body.clone()))
            .chain(initial_history)
            .collect();
        let stripped = Self {
            provider,
            tools,
            preamble: None,
            temperature,
            max_tokens,
        };
        let init = LoopState::FreshRequest {
            agent: stripped,
            history: full_history,
            cancel,
        };
        let step_fn: LoopStepFn<T> = Arc::new(loop_step::<T>);
        Stream::unfold(init, step_fn)
    }
}

/// Type alias for the boxed closure used by `Stream::unfold` to drive the
/// `LoopState` machine.  Factored out to keep clippy's `type_complexity`
/// happy.
type LoopStepFn<T> =
    Arc<dyn Fn(LoopState<T>) -> Io<Error, Option<(AgentEvent, LoopState<T>)>> + Send + Sync>;

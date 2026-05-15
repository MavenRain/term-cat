//! `LocalOpenAiCompletion` — an OpenAI-compatible chat-completion client
//! backed by `ureq`.  Talks to LM Studio / `llama-server` / Ollama (via their
//! `/v1` compat layer) at a configurable base URL.
//!
//! Exposes two inherent methods:
//!
//!   * [`LocalOpenAiCompletion::complete`] — blocking, returns the full
//!     assistant body as one `Io<Error, MessageBody>`.  Used by callers that
//!     do not need streaming.
//!   * [`LocalOpenAiCompletion::stream_chat`] — streaming, returns
//!     `Stream<Error, ChatEvent>` that emits token deltas as the server
//!     produces them.  Cancellation is observed between SSE line reads.

use std::io::{BufReader, Read};
use std::sync::Arc;

use comp_cat_rs::effect::io::Io;
use comp_cat_rs::effect::stream::Stream;
use ureq::Body;
use ureq::http::Response;

use crate::agent::ChatEvent;
use crate::bridge::CancelObserver;
use crate::error::{Error, ProviderError, WireError};
use crate::newtype::{ApiKey, BaseUrl, MaxTokens, MessageBody, ModelName, Temperature};
use crate::sse::{SseState, step as sse_step};
use crate::wire::{ChatCompletionRequest, ChatCompletionResponse, Message};

/// Blocking OpenAI-compatible chat-completion client.  Clonable so that one
/// provider instance can be shared across per-turn fibers.
#[derive(Debug, Clone)]
pub struct LocalOpenAiCompletion {
    base_url: BaseUrl,
    model: ModelName,
    api_key: Option<ApiKey>,
    temperature: Option<Temperature>,
    max_tokens: Option<MaxTokens>,
}

impl LocalOpenAiCompletion {
    /// Construct against the given base URL and model name.
    #[must_use]
    pub fn new(base_url: BaseUrl, model: ModelName) -> Self {
        Self {
            base_url,
            model,
            api_key: None,
            temperature: None,
            max_tokens: None,
        }
    }

    /// Set a bearer token.  LM Studio ignores it; other servers may not.
    #[must_use]
    pub fn with_api_key(self, key: ApiKey) -> Self {
        Self {
            api_key: Some(key),
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

    /// View the configured model.
    #[must_use]
    pub fn model(&self) -> &ModelName {
        &self.model
    }

    /// Issue a blocking chat-completion request and return the first choice's
    /// assistant content as a `MessageBody`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Wire` on encode failure, `Error::Provider` on transport
    /// or status failure or decode failure, and `Error::Provider` again if the
    /// response carries no `content` (treated as a degenerate 200).
    #[must_use]
    pub fn complete(&self, messages: Vec<Message>) -> Io<Error, MessageBody> {
        let url = format!("{}/chat/completions", self.base_url.as_str());
        let model = self.model.clone();
        let api_key = self.api_key.clone();
        let temperature = self.temperature;
        let max_tokens = self.max_tokens;

        Io::suspend(move || {
            let request = ChatCompletionRequest::new(model, messages)
                .maybe_temperature(temperature)
                .maybe_max_tokens(max_tokens)
                .with_stream(false);

            let json_body =
                serde_json::to_string(&request).map_err(|e| Error::Wire(WireError::Encode(e)))?;

            // Apply the optional bearer header functionally: `Option::iter`
            // yields 0 or 1 references; `fold` applies the header only when
            // `api_key` is `Some`.  This avoids matching on `Option` and
            // sidesteps moving `builder` into both branches of a `map_or`.
            let builder = api_key.iter().fold(
                ureq::post(&url).header("Content-Type", "application/json"),
                |b, k| b.header("Authorization", format!("Bearer {}", k.as_str())),
            );

            let response = builder
                .send(json_body)
                .map_err(|e| Error::Provider(ProviderError::Http(e)))?;

            let body_text = consume_body(check_status(response)?);

            let parsed: ChatCompletionResponse = serde_json::from_str(&body_text)
                .map_err(|e| Error::Provider(ProviderError::Decode(e)))?;

            parsed.first_content().cloned().ok_or_else(|| {
                Error::Provider(ProviderError::Status {
                    code: 200,
                    body: "response had no content".to_owned(),
                })
            })
        })
    }
}

/// Return the response unchanged on 2xx; otherwise read its body and return
/// `Error::Provider(Status { .. })`.
fn check_status(response: Response<Body>) -> Result<Response<Body>, Error> {
    let code = response.status().as_u16();
    if (200..300).contains(&code) {
        Ok(response)
    } else {
        let body = consume_body(response);
        Err(Error::Provider(ProviderError::Status { code, body }))
    }
}

/// Inputs captured before the streaming HTTP call is issued.  Held inside
/// `StreamChatState::Pending` so the first unfold step can build the body,
/// send the request, and transition to `Active`.
struct StreamPending {
    url: String,
    api_key: Option<ApiKey>,
    model: ModelName,
    messages: Vec<Message>,
    temperature: Option<Temperature>,
    max_tokens: Option<MaxTokens>,
    cancel: CancelObserver,
}

/// State threaded through the streaming chat unfold.
enum StreamChatState {
    /// First step: open the HTTP connection and hand off to SSE parsing.
    Pending(StreamPending),
    /// Steady-state: pull the next event from the SSE parser.
    Active(SseState),
}

/// Type alias for the boxed closure used by `Stream::unfold` to drive
/// `stream_chat`.  Factored out to keep clippy's `type_complexity` happy.
type StreamChatStepFn =
    Arc<dyn Fn(StreamChatState) -> Io<Error, Option<(ChatEvent, StreamChatState)>> + Send + Sync>;

impl LocalOpenAiCompletion {
    /// Issue a streaming chat-completion request and return a
    /// `Stream<Error, ChatEvent>` that emits text and tool-call deltas as the
    /// server produces them.  Cancellation via `cancel` is observed between
    /// SSE line reads (~50 ms granularity in practice).
    #[must_use]
    pub fn stream_chat(
        &self,
        messages: Vec<Message>,
        cancel: CancelObserver,
    ) -> Stream<Error, ChatEvent> {
        let pending = StreamPending {
            url: format!("{}/chat/completions", self.base_url.as_str()),
            api_key: self.api_key.clone(),
            model: self.model.clone(),
            messages,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            cancel,
        };
        let init = StreamChatState::Pending(pending);
        let step_fn: StreamChatStepFn = Arc::new(stream_chat_step);
        Stream::unfold(init, step_fn)
    }
}

fn stream_chat_step(state: StreamChatState) -> Io<Error, Option<(ChatEvent, StreamChatState)>> {
    match state {
        StreamChatState::Pending(p) => Io::suspend(move || open_streaming(p))
            .flat_map(|sse_state| sse_step(sse_state).map(lift_sse_to_chat)),
        StreamChatState::Active(sse_state) => sse_step(sse_state).map(lift_sse_to_chat),
    }
}

fn lift_sse_to_chat(opt: Option<(ChatEvent, SseState)>) -> Option<(ChatEvent, StreamChatState)> {
    opt.map(|(event, next)| (event, StreamChatState::Active(next)))
}

/// Build the request, POST it with `stream: true`, status-check, and wrap the
/// response body in an `SseState` ready for `sse::step`.
fn open_streaming(p: StreamPending) -> Result<SseState, Error> {
    let StreamPending {
        url,
        api_key,
        model,
        messages,
        temperature,
        max_tokens,
        cancel,
    } = p;

    let request = ChatCompletionRequest::new(model, messages)
        .maybe_temperature(temperature)
        .maybe_max_tokens(max_tokens)
        .with_stream(true);

    let json_body =
        serde_json::to_string(&request).map_err(|e| Error::Wire(WireError::Encode(e)))?;

    let builder = api_key.iter().fold(
        ureq::post(&url)
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream"),
        |b, k| b.header("Authorization", format!("Bearer {}", k.as_str())),
    );

    let response = builder
        .send(json_body)
        .map_err(|e| Error::Provider(ProviderError::Http(e)))?;
    let response = check_status(response)?;

    let reader = BufReader::new(response.into_body().into_reader());
    Ok(SseState::new(reader, cancel))
}

/// Drain the response body into a `String`, returning a sentinel on failure
/// (we are already in the unhappy path; a stringified body is best-effort).
fn consume_body(mut response: Response<Body>) -> String {
    let mut buf = String::new();
    response
        .body_mut()
        .as_reader()
        .read_to_string(&mut buf)
        .map_or_else(|_| "<failed to read body>".to_owned(), |_| buf.clone())
}

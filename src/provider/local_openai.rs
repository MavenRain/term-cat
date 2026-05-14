//! `LocalOpenAiCompletion` — a blocking, OpenAI-compatible chat-completion
//! client backed by `ureq`.  Talks to LM Studio / `llama-server` / Ollama (via
//! their `/v1` compat layer) at a configurable base URL.
//!
//! Phase B exposes only the blocking `complete` method.  Phase C will add an
//! inherent `stream_chat` method returning `Stream<Error, ChatEvent>`.

use std::io::Read;

use comp_cat_rs::effect::io::Io;
use ureq::Body;
use ureq::http::Response;

use crate::error::{Error, ProviderError, WireError};
use crate::newtype::{ApiKey, BaseUrl, MaxTokens, MessageBody, ModelName, Temperature};
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
    match () {
        () if (200..300).contains(&code) => Ok(response),
        () => {
            let body = consume_body(response);
            Err(Error::Provider(ProviderError::Status { code, body }))
        }
    }
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

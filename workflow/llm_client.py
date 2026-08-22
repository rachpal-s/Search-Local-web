"""workflow/llm_client.py — resilient Ollama invocation: retry, then fall back.

Both supervisor_node and critic_node called ChatOllama(...).invoke() directly
with no error handling, so a single transient failure — "model 'X' is
temporarily overloaded, please retry shortly" is the one that prompted this —
took down the entire graph run and surfaced to the user as
"Graph execution failed: ResponseError: ...".

Strategy, in order:
  1. Retry the SAME model a couple of times with backoff. Most "overloaded,
     retry shortly" errors clear within seconds — this alone resolves the
     common case without changing which model answered.
  2. Only after exhausting retries, fall back to a different model. Falling
     back on the very first error would mask genuine model-specific problems
     (a malformed prompt, a real auth failure) as if they were capacity
     issues, and would make the supervisor and critic silently swap
     reasoning quality mid-run.
  3. If every candidate model is exhausted, raise the last error. At that
     point there is nothing left to try, and the caller's existing
     "Graph execution failed" handling in main.py is the right place for it
     to surface — this module intentionally does not swallow a total outage.
"""
from __future__ import annotations

import time
from typing import Any, Optional, Sequence

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama

try:
    from ollama import ResponseError
except ImportError:  # neo4j-style optional-dependency guard; keep this module
    ResponseError = None  # type: ignore  # import-safe even if ollama isn't installed


# HTTP-style statuses worth retrying/falling back on. Anything else (bad
# request shape, a real 401, programmer error) should surface immediately —
# retrying or masking those behind a fallback model would hide a bug rather
# than route around real capacity issues.
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
_RETRYABLE_PHRASES = ("overloaded", "try a different model", "retry shortly",
                      "retry later", "timed out", "timeout")


def _is_retryable(exc: Exception) -> bool:
    if ResponseError is not None and isinstance(exc, ResponseError):
        msg = (getattr(exc, "error", "") or str(exc)).lower()
        if any(p in msg for p in _RETRYABLE_PHRASES):
            return True
        return getattr(exc, "status_code", -1) in _RETRYABLE_STATUS
    # Connection-level flakiness (server mid-restart, brief network blip) —
    # not an Ollama-specific error type, but equally worth a retry.
    return isinstance(exc, (ConnectionError, TimeoutError))


def invoke_with_fallback(
    messages: Sequence[BaseMessage],
    *,
    base_url: str,
    model: str,
    fallback_model: Optional[str] = None,
    temperature: float = 0,
    format: Optional[str] = "json",
    num_ctx: int,
    num_predict: Optional[int] = None,
    keep_alive: Optional[str] = None,
    max_retries: int = 2,
    retry_backoff_s: float = 2.0,
    log_prefix: str = "[LLM]",
) -> Any:
    """Drop-in replacement for `ChatOllama(...).invoke(messages)`.

    Tries `model` first (up to `max_retries` retries on transient errors),
    then `fallback_model` if given and different from `model`. Returns the
    first successful response. Raises the last exception if every candidate
    is exhausted — callers keep whatever error handling they already had.
    """
    candidates = [model]
    if fallback_model and fallback_model != model:
        candidates.append(fallback_model)

    last_exc: Optional[Exception] = None

    for idx, attempt_model in enumerate(candidates):
        llm = ChatOllama(
            base_url=base_url, model=attempt_model, temperature=temperature,
            format=format, num_ctx=num_ctx, num_predict=num_predict,
            keep_alive=keep_alive,
        )
        for attempt in range(max_retries + 1):
            try:
                return llm.invoke(list(messages))
            except Exception as e:  # noqa: BLE001 — any failure is a candidate to retry/fall back
                last_exc = e
                retryable = _is_retryable(e)
                print(f"{log_prefix} ⚠️ '{attempt_model}' attempt {attempt + 1}/"
                      f"{max_retries + 1} failed: {type(e).__name__}: {e}"
                      f"{'' if retryable else ' — not retryable, moving on'}")
                if not retryable:
                    break  # don't burn retries on a non-transient error
                if attempt < max_retries:
                    time.sleep(retry_backoff_s * (attempt + 1))

        if idx < len(candidates) - 1:
            print(f"{log_prefix} 🔀 '{attempt_model}' exhausted — "
                  f"falling back to '{candidates[idx + 1]}'.")

    raise last_exc  # every candidate exhausted — let the caller's existing handling take over

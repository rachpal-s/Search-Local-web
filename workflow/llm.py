"""workflow/llm.py — per-turn model selection and failover.

WHY A CONTEXTVAR
----------------
Worker and reasoning nodes are reached through `Send(agent, payload)` and the
supervisor authors that payload, so a node only ever sees what the LLM put
there. A user's model choice is neither the LLM's business nor something it
could be trusted to copy correctly into every payload. Same problem
`rag/session.py` solves for conversation_id and `workflow/inflight.py` solves
for run ids, so the same answer: bind it once per request in the endpoint, and
every node inside the graph inherits it.

Sync nodes (supervisor_node, critic_node) run in a worker thread. Threads
started via run_in_executor / asyncio.to_thread inherit a COPY of the context,
so the selection is visible there too.

WHY FAILOVER LIVES HERE AND NOT IN THE NODES
--------------------------------------------
Both callers need the same behaviour — try the chosen model, fall back through
the catalogue on failure, report which one actually answered — and both have
prompts expensive enough that losing a turn to a transient model error is a
real cost. Two copies of that loop would drift.

`invoke_with_fallback` treats an EMPTY response as a failure as well as an
exception. That is not defensive padding: agents/critic.py documents exactly
this failure, where a reasoning model burned its whole token budget on hidden
chain-of-thought and returned empty content, which json.loads() then rejected
and scored 0.

EMPTY IS RETRIED ON THE SAME MODEL FIRST
----------------------------------------
Empty content and a dead endpoint are not the same failure and should not get
the same response. An HTTP error means this model cannot serve the request at
all; empty content usually means a reasoning model spent its whole budget on
hidden chain-of-thought, which is a per-call accident that clears on a second
pass. Switching models on the first empty response silently discards the user's
selection for a transient condition — and because reasoning models produce
empty far more often than gemma4 does, the model the user deliberately picked
was the one most likely to be replaced. So empty gets one retry on the same
model before the chain moves on.

The chain is rebuilt per invocation, not per turn. That is correct — each node
should start from the user's choice rather than inherit another node's
failover — but it does mean one turn can legitimately be answered by several
models. That is why every attempt is logged, and why ModelChainError carries
those logs out rather than letting them die with the frame.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Callable, List, Optional, Tuple

from config import MODEL_CATALOGUE, fallback_chain, get_settings, resolve_model

_primary_model: ContextVar[Optional[str]] = ContextVar("primary_model", default=None)
_critic_model: ContextVar[Optional[str]] = ContextVar("critic_model", default=None)

# One retry on the same model when the failure was empty content rather than an
# exception. Two is not better: a model that returns empty twice in a row is
# budget-bound on this prompt, and a third identical call just costs latency.
_EMPTY_RETRIES = 1


class ModelChainError(RuntimeError):
    """Every model in the chain failed.

    Carries `logs` because the caller needs them: `invoke_with_fallback`
    accumulates one line per attempt, and raising a bare RuntimeError meant
    that list died with the frame. The visible symptom was a trace showing
    three identical "generating..." lines and then a single error, with no
    indication that three DIFFERENT models had been tried and why each failed.
    """

    def __init__(self, message: str, logs: List[str]):
        super().__init__(message)
        self.logs = logs


# ── binding (called once per request, from the endpoint) ──────────────────────

def set_models(primary: Optional[str] = None, critic: Optional[str] = None) -> dict:
    """Bind this turn's model choices. Returns what was actually resolved.

    Both values come from the browser and are validated against the catalogue
    allow-list in config.resolve_model — an unrecognised id silently becomes
    the configured default rather than reaching an Ollama call.
    """
    cfg = get_settings()
    p = resolve_model(primary, cfg.ollama_inference_model)
    c = resolve_model(critic, cfg.ollama_inference_critic_model)
    _primary_model.set(p)
    _critic_model.set(c)
    return {"primary": p, "critic": c}


def primary_model() -> str:
    return _primary_model.get() or get_settings().ollama_inference_model


def critic_model() -> str:
    return _critic_model.get() or get_settings().ollama_inference_critic_model


def catalogue() -> dict:
    """Payload for GET /api/models — the list plus the current defaults."""
    cfg = get_settings()
    return {
        "models": MODEL_CATALOGUE,
        "defaults": {
            "primary": cfg.ollama_inference_model,
            "critic": cfg.ollama_inference_critic_model,
        },
    }


# ── invocation with failover ──────────────────────────────────────────────────

class _EmptyContent(ValueError):
    """A response with no content. Retryable on the same model — see module docstring."""


def invoke_with_fallback(build_llm: Callable[[str], Any],
                         messages: List[Any],
                         *,
                         chosen: str,
                         default: str,
                         label: str = "LLM") -> Tuple[Any, str, List[str]]:
    """Invoke `build_llm(model).invoke(messages)`, failing over down the chain.

    `build_llm` is a factory rather than a built client because each caller
    configures its own temperature / format / num_ctx / num_predict, and those
    settings are not interchangeable — the critic's num_predict=8192 exists
    specifically to cover a reasoning model's hidden trace (see
    agents/critic.py), and agents/code_editor.py sizes its own budget from the
    source file. Handing the factory the model id keeps every one of those
    per-caller settings intact on the retry.

    Returns (response, model_used, logs). `logs` is UI-facing: a silent
    failover would make a slow or differently-worded turn look inexplicable.
    Raises ModelChainError when nothing in the catalogue answered.
    """
    chain = fallback_chain(chosen, default)
    logs: List[str] = []
    last_error: Optional[Exception] = None

    for attempt, model in enumerate(chain):
        for retry in range(_EMPTY_RETRIES + 1):
            try:
                response = build_llm(model).invoke(messages)
                content = (getattr(response, "content", "") or "").strip()
                if not content:
                    raise _EmptyContent("model returned empty content")
                if attempt or retry:
                    msg = (f"🔁 {label}: answered with '{model}'"
                           + ("" if not attempt else f" after '{chain[0]}' failed")
                           + ("" if not retry else " on retry")
                           + ".")
                    logs.append(msg)
                    print(f"[LLM] {msg}")
                else:
                    print(f"[LLM] {label}: answered with '{model}'.")
                return response, model, logs
            except _EmptyContent as e:
                last_error = e
                if retry < _EMPTY_RETRIES:
                    msg = (f"🔁 {label}: '{model}' returned empty (likely spent its "
                           f"budget on hidden reasoning) — retrying the same model.")
                    logs.append(msg)
                    print(f"[LLM] {msg}")
                    continue
                break
            except Exception as e:  # noqa: BLE001 — any hard failure moves the chain on
                last_error = e
                break

        msg = (f"⚠️ {label}: model '{model}' failed "
               f"({type(last_error).__name__}: {str(last_error)[:160]}).")
        logs.append(msg)
        print(f"[LLM] {msg}")

    # Nothing in the catalogue worked. Raise rather than return a sentinel:
    # the callers already have JSON-parse recovery paths that would otherwise
    # treat total unavailability as a malformed answer and hide the real cause.
    raise ModelChainError(
        f"{label}: every model in the fallback chain failed "
        f"({', '.join(chain)}). Last error: {last_error}", logs)

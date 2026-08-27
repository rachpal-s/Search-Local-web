"""agents/critic.py — Critic evaluator.

Scores the supervisor's proposed final response (0-100) against the
original user query and returns feedback for another supervisor pass
if the score falls short.

NOTE: the original file also contained `critic_node_OLDDD` and a
matching `route_from_critic_OLDDD`, an earlier APPROVED/feedback-text
based version. Neither was wired into the compiled graph, so they
were dropped here as dead code. Say the word if you'd like them
restored (e.g. under agents/critic_legacy.py).
"""
import json
import re
from typing import Any, Dict
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from config import get_settings
from workflow import inflight, llm as llm_select
from workflow.state import AgentState

# NOTE: kept identical to the original — hardcoded rather than sourced
# from config.get_settings().ollama_inference_model.
CRITIC_MODEL = get_settings().ollama_inference_critic_model
# LATENCY: the critic gets its own, much smaller context window. It reads the
# answer plus MAX_CONTEXT_CHARS (50k chars) of already-truncated source
# material and emits a short JSON verdict — allocating a 200k KV cache for
# that costs setup time on every single turn and buys nothing. Raise this only
# if MAX_CONTEXT_CHARS below is raised too.
NUM_CTX = get_settings().ollama_critic_num_ctx or 32000
MAX_CONTEXT_CHARS = 50000

def _format_context_for_critic(context: list) -> str:
    """Render gathered context for the critic prompt, truncated to fit."""
    if not context:
        return "_No source material was gathered for this response._"
    formatted = "\n".join(f"- {c}" for c in context)
    if len(formatted) > MAX_CONTEXT_CHARS:
        formatted = formatted[:MAX_CONTEXT_CHARS] + "\n... [source material truncated for length]"
    return formatted

def _format_pending_for_critic(pending: list) -> str:
    """Tell the critic which gaps are being actively filled right now.

    Without this the critic scores a partial answer down for missing breadth,
    the router sends it back to the supervisor, and the supervisor burns a
    loop "correcting" something that was already in progress.
    """
    if not pending:
        return ""
    pending_list = "\n".join(f"    - {u}" for u in pending)
    return f"""
    SOURCES STILL BEING FETCHED ({len(pending)} in flight) - DO NOT PENALISE FOR THESE:
{pending_list}

    These sources were too slow to return before the Supervisor wrote its
    response, and are still rendering in the background. Their content will be
    appended to the answer before the user sees it.
    - Do NOT lower the score for missing breadth, missing sources, thin coverage,
      or incompleteness attributable to the sources listed above.
    - Do NOT describe those sources as failed or unavailable. They are pending.
    - Judge ONLY the accuracy, coherence and usefulness of what IS present,
      against the Source Material that WAS actually available.
    - An accurate, clearly-scoped partial answer is a SUCCESS. If the response
      would otherwise pass, pass it.
"""


def critic_node(state: AgentState) -> Dict[str, Any]:
    """Score the proposed final response from 0-100 and provide feedback."""
    print(f"\n[CRITIC] ⚖️ ENTERING CRITIC EVALUATION")
    current_critic_loop = state.get("critic_loop_count", 0) + 1
    chosen_model = llm_select.critic_model()

    def _build_llm(model_name: str):
        return ChatOllama(
            base_url=get_settings().ollama_inference_url,
            model=model_name, 
            temperature=0,
            format="json",
            num_ctx=NUM_CTX,
            # NOTE: originally capped at num_predict=1024 as a latency guard,
            # reasoning the verdict itself is only a sentence or two of JSON.
            # REVERTED: the default critic model (gpt-oss:120b-cloud) is a reasoning model —
            # it spends a variable, sometimes large, number of tokens on hidden
            # chain-of-thought BEFORE it writes the visible JSON answer. 1024 was
            # enough for the reasoning trace to consume the entire budget and
            # leave response.content empty, which json.loads() then rejected as
            # "Expecting value: line 1 column 1 (char 0)" — scored 0, dropped
            # below the pass threshold, and sent the whole turn back through a
            # full extra supervisor+critic loop. That extra loop cost far more
            # latency than the cap ever saved, on top of being a correctness bug.
            # Back to unbounded-within-context. NOTE this now applies to whichever
            # model the USER selected, which is why the empty-content case is
            # treated as a failure in workflow/llm.py and fails over rather
            # than scoring 0.; do not re-cap this without first
            # confirming the selected critic model isn't a reasoning model, or
            # capping high enough to cover its reasoning trace + the JSON both.
            num_predict=8192,
            keep_alive=get_settings().ollama_keep_alive,
        )
    ist_timezone = ZoneInfo("Asia/Kolkata")
    current_time_str = datetime.now(ist_timezone).strftime("%A, %B %d, %Y at %I:%M %p IST")
    source_material = _format_context_for_critic(state.get("context", []))
    pending_note = _format_pending_for_critic(inflight.pending_urls())
    # Same number route_from_critic() actually gates on (config.py,
    # critic_pass_threshold). Previously hardcoded to 90 here while the
    # router gated at a separately hardcoded 85 — two numbers meant to agree
    # that could silently drift apart. One source of truth now.
    pass_threshold = get_settings().critic_pass_threshold

    prompt = f"""You are an objective Critic Evaluator. 
    Evaluate the following proposed response against the user's original query.
    User Query: {state['user_query']}
    Proposed Response: {state['final_response']}

    Source Material Gathered (search results / scraped pages / tool outputs
    the Supervisor had available when it wrote the Proposed Response):
    {source_material}
{pending_note}

    CRITICAL TEMPORAL CONTEXT (treat as ground truth, not up for debate): 
    Today's exact date and time in India is {current_time_str}. Your own
    training data has a cutoff earlier than this date, so real events dated
    at or after your cutoff will feel unfamiliar to you. That unfamiliarity
    is NOT evidence of fabrication. NEVER lower the score, and NEVER use
    language like "impossible", "future date", or "cannot exist" merely
    because a date or event postdates what you were trained on — your own
    sense of "what has happened so far" is stale and must not be treated
    as a source of truth in this evaluation.

    HOW TO JUDGE ACCURACY (this replaces judging from memory):
    Your only reliable ground truth for factual accuracy is the "Source
    Material Gathered" above — NOT your internal knowledge of the world.
    - A specific claim, figure, quote, or event in the Proposed Response
      that is supported by (or is a reasonable synthesis of) the Source
      Material counts as accurate, however unfamiliar or "recent" it feels.
    - Only mark something a hallucination if it does NOT appear anywhere in
      the Source Material and could not reasonably be derived from it.
      Name the specific unsupported claim in your feedback.
    - If Source Material is empty, do not assume that means the response is
      fabricated — the Supervisor may have answered directly. In that case
      judge only internal coherence and whether uncertainty is flagged
      appropriately, and say explicitly in your feedback that no source
      material was available to verify against.

    CRITICAL ENVIRONMENT CONTEXT: 
    1. This system is equipped with a verified `youtube_downloader` tool. Downloading videos is AUTHORIZED, technically feasible, and expected in this workflow. 
    2. Do NOT penalize the response for mentioning downloads or YouTube TOS. 
    3. If the Proposed Response contains a local markdown link to a downloaded file (e.g., [Video](/static/downloads/...)), this is a MASSIVE SUCCESS. Score it >90.

    Assign a score from 0 to 100 based on how fully, accurately, and safely the response answers the query.
    If the score is less than {pass_threshold}, provide brief feedback on what is missing or needs re-reasoning.

    Output STRICTLY in the following JSON format:
    {{
        "score": 95,
        "feedback": "Your brief reasoning here."
    }}
    """
    logs = ["[CRITIC] ⏳ Evaluating final summary response..."]
    print("[CRITIC] ⏳ Evaluating final summary response...")
    response, model_used, llm_logs = llm_select.invoke_with_fallback(
        _build_llm, [HumanMessage(content=prompt)],
        chosen=chosen_model,
        default=CRITIC_MODEL,
        label="Critic")
    logs.extend(llm_logs)

    raw_content = response.content.strip()
    raw_content = re.sub(r"^```(?:json)?\s*", "", raw_content, flags=re.IGNORECASE)
    raw_content = re.sub(r"\s*```$", "", raw_content)

    try:
        eval_data = json.loads(raw_content)
        score = int(eval_data.get("score", 0))
        feedback = eval_data.get("feedback", "No feedback provided.")
        success_msg = f"[CRITIC] 📝 Evaluation Outcome - Score: {score}/100 | Feedback: {feedback}"
        logs.append(success_msg)
        # print(f"[CRITIC] 📝 Evaluation Outcome - Score: {score}/100 | Feedback: {feedback}")
    except (json.JSONDecodeError, ValueError) as e:
        error_msg = f"[CRITIC] ❌ Failed to parse critic evaluation: {e}"
        logs.append(error_msg)
        print(error_msg)
        # print(f"[CRITIC] ❌ Failed to parse critic evaluation: {e}")
        score = 0
        feedback = "System Error: Failed to parse critic evaluation."

    return {
            "eval_score": score, 
            "feedback": feedback,
            "critic_loop_count": current_critic_loop,
            "score_history": [score],
            "action_logs": logs
            }
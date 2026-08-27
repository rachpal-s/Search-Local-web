"""agents/web_scraper.py — Agent 3: Web Scraper.

Thin wrapper over the scraper.engine cascade: static battery (jsonld →
embedded_state → site_dom → trafilatura → readability → newspaper3k →
goose3 → beautifulsoup) → cheap alternates (AMP / mobile host / WP REST)
→ Playwright light → Playwright full → optional external reader.

DEFERRED ESCALATION
-------------------
Tiers 0/1 are sub-second to ~3s; the Playwright tail is 4-12s. When the
cheap tiers already produced a usable extraction this node behaves exactly
as before. When they did not, it returns the partial result immediately and
hands the browser escalation to workflow.inflight, which keeps it running
after this node has returned. The supervisor picks up late results on its
next loop, and main.py drains whatever is left before the final answer.

The point is that one slow site no longer holds up the other five sources
that already came back.

Named `web_scraper` (not `scraper`) to avoid shadowing the top-level
`scraper` package this agent depends on.
"""
import json
from typing import Any, Dict

from config import get_settings
from scraper.engine import scrape, scrape_fast
from workflow import inflight
from workflow.state import ScrapeTaskState

cfg = get_settings()

# Per-source context budget. Was a hardcoded 1500 (~230 words) against an
# ollama_num_ctx of 200_000 — the cascade escalated to a full Playwright render
# to earn 180+ words, then discarded everything past ~230 with no marker. Now
# config-driven so it can be sized against the model actually deployed.
CONTENT_SAMPLE_CHARS = int(getattr(cfg, "scrape_content_sample_chars", 8000))

# Share of that budget reserved for extracted tables before prose gets any.
# Tables and prose used to compete for one undivided slice with tables simply
# concatenated in front, so a page with several tables spent the whole budget
# on the early ones: the *final* summary table (last in document order) was cut
# mid-row and the prose body — the lists, the conclusions — never appeared at
# all. Splitting the budget means neither can starve the other, and unspent
# table budget is handed back to prose rather than wasted.
TABLE_BUDGET_RATIO = float(getattr(cfg, "scrape_table_budget_ratio", 0.6))

# Below this word count, a "successful" extraction is treated as
# unreliable rather than a genuine success. Real article/listing pages
# run into the hundreds of words; a handful of words is the signature of
# a JS-rendered shell scraped before the client-side app hydrates.
MIN_WORD_COUNT = 80

# Engine early-exit threshold. Higher than MIN_WORD_COUNT on purpose: the
# engine keeps escalating until it clears this, then this node accepts
# anything above the lower bar. Keeps a 120-word wire story usable while
# still pushing the cascade to try harder for it first.
TARGET_WORD_COUNT = 180


def _truncate_tables(tables_md: str, budget: int) -> tuple[str, bool]:
    """Trim markdown tables to `budget` chars on a ROW boundary.

    A naive slice cuts mid-row and yields a table whose last line is a broken
    pipe-row. Nothing downstream can tell that apart from a real row, so the
    model reads a corrupted value as data. Dropping whole rows keeps the
    markdown parseable and keeps the header — which is the part that makes the
    remaining rows interpretable — intact.
    """
    if not tables_md or len(tables_md) <= budget:
        return tables_md, False

    kept, used = [], 0
    for line in tables_md.split("\n"):
        if used + len(line) + 1 > budget:
            break
        kept.append(line)
        used += len(line) + 1

    body = "\n".join(kept).rstrip()
    return body, True


def _truncate_prose(text: str, budget: int) -> tuple[str, bool]:
    """Trim prose to `budget` chars, preferring a paragraph then a sentence
    boundary, so the tail is not a half-finished clause."""
    if not text or len(text) <= budget:
        return text, False

    window = text[:budget]
    for sep in ("\n\n", ". ", "\n"):
        cut = window.rfind(sep)
        # Only honour a boundary in the last third; an early one would throw
        # away more than the truncation itself does.
        if cut > budget * 0.66:
            return window[:cut].rstrip(), True
    return window.rstrip(), True


def _compose_content(res) -> tuple[str, Dict[str, Any]]:
    """Assemble the per-source sample under an explicit, split budget.

    Returns the text plus flags describing what was dropped. Those flags are
    the point: the old single slice was SILENT, so the supervisor received a
    partial list that looked complete and reported it as the whole thing.
    Truncation the model cannot see is worse than truncation itself.
    """
    prose = (res.content or "").strip() or "Failed to extract clean text."
    tables = (res.tables_md or "").strip() if res.table_count else ""

    total = CONTENT_SAMPLE_CHARS
    table_budget = int(total * TABLE_BUDGET_RATIO) if tables else 0
    tables_out, tables_cut = _truncate_tables(tables, table_budget) if tables else ("", False)

    # Unused table budget flows back to prose — a page with one small table
    # should not lose 60% of its body to a reservation it never needed.
    prose_budget = total - len(tables_out)
    prose_out, prose_cut = _truncate_prose(prose, prose_budget)

    parts = []
    if tables_out:
        parts.append(tables_out)
        if tables_cut:
            parts.append("_[TRUNCATED: further tables/rows omitted to fit the "
                         "context budget. Re-scraping this URL will not return "
                         "them; raise scrape_content_sample_chars instead.]_")
        parts.append("---")
    parts.append(prose_out)
    if prose_cut:
        parts.append(f"_[TRUNCATED: showing {len(prose_out):,} of {len(prose):,} "
                     f"characters of body text. Any list or summary below this "
                     f"point was NOT captured — do not present this extract as "
                     f"complete, and do not re-scrape the URL to try to get it: "
                     f"the fetch succeeded and a retry returns the same clip.]_")

    flags = {
        "content_truncated": bool(tables_cut or prose_cut),
        "tables_truncated": tables_cut,
        "prose_truncated": prose_cut,
        "content_chars": len(prose),
        "content_chars_kept": len(prose_out),
    }
    return "\n\n".join(parts), flags


def _result_payload(res, url: str, *, deferred: bool = False) -> Dict[str, Any]:
    """Build the dict that goes into graph context for one scrape."""
    word_count = res.word_count
    content, flags = _compose_content(res)

    payload = {
        "source": url,
        "title": res.title or "N/A",
        "word_count": word_count,
        "strategy": res.best_strategy or "none",
        "page_type": res.page_type,
        "table_count": res.table_count,
        "scraped_content": content,
        "insufficient_content": word_count < MIN_WORD_COUNT,
        **flags,
    }

    # Images are appended AFTER budgeting, exactly once. They used to be added
    # to `content` before the slice AND again to the payload after it, so a
    # short page carried the image block twice while a long page kept only the
    # second copy — and the pre-slice copy spent budget that prose then lost.
    if getattr(res, "images", None):
        img_md = "\n".join(res.images)
        payload["scraped_content"] += f"\n\n### Extracted Images\n{img_md}"

    if deferred:
        payload["still_rendering"] = True
    return payload


def _label(payload: Dict[str, Any], *, late: bool = False) -> str:
    """Prefix that tells the supervisor's LLM how much to trust this entry.

    The supervisor reads context as plain strings, so the label is what
    actually stops it treating a 29-word shell as "task complete" — or,
    for a deferred entry, what stops it re-dispatching the same URL.
    """
    if payload.get("error"):
        return "Scraper Engine Result (ERROR)"
    if payload.get("still_rendering"):
        return ("Scraper Engine Result (PARTIAL — FULL RENDER STILL IN PROGRESS, "
                "DO NOT RE-DISPATCH THIS URL)")
    if late:
        return "Scraper Engine Result (LATE — completed background render)"
    if payload.get("insufficient_content"):
        return "Scraper Engine Result (LOW CONFIDENCE / INSUFFICIENT CONTENT)"
    if payload.get("content_truncated"):
        # Same job as the insufficient_content label: tell the supervisor how
        # far to trust this entry. Without it a clipped extract is
        # indistinguishable from a complete one, and the answer confidently
        # presents a partial list as the full one.
        # Wording matters more than it looks. An earlier version said only
        # "this extract is INCOMPLETE", which reads as an invitation to go and
        # fetch the rest — so the supervisor re-dispatched the same URL, got
        # the identically truncated result, and stalled. The truncation is a
        # context-budget decision, not a fetch failure, and the label has to
        # say so or it manufactures the repetition the caps exist to prevent.
        return ("Scraper Engine Result (TRUNCATED TO FIT THE CONTEXT BUDGET — "
                "the page was fetched successfully; this extract is just "
                "clipped. Do NOT re-scrape this URL, a retry returns the same "
                "clip. Use what is here, and do not describe its lists or "
                "tables as exhaustive)")
    return "Scraper Engine Result"


def _late_formatter(url: str):
    """Formatter handed to inflight.defer(): ScrapeResult -> context string."""
    def fmt(res) -> str:
        payload = _result_payload(res, url)
        return f"{_label(payload, late=True)}: {json.dumps(payload)}"
    return fmt


async def scraper_node(state: ScrapeTaskState) -> Dict[str, Any]:
    """Scrape a target URL and return a title/word-count/content sample."""
    url = state["url"]
    logs = [f"Scraping URL: '{url}'"]
    print(f"\n[SCRAPER] 🕸️ Scraping target URL: {url}")

    defer_enabled = bool(getattr(cfg, "scrape_defer_enabled", True))

    try:
        if defer_enabled:
            res, finish = await scrape_fast(
                url, min_words=TARGET_WORD_COUNT, allow_playwright=True
            )
        else:
            res, finish = await scrape(
                url, min_words=TARGET_WORD_COUNT, allow_playwright=True
            ), None

        attempted = [
            f"{r.strategy}({'ok' if r.success else (r.error or 'fail')[:40]})"
            for r in res.all_results
        ]
        print(f"[SCRAPER] Strategies tried: {', '.join(attempted)}")

        deferred = False
        if finish is not None:
            # Cheap tiers came up thin. Hand the browser escalation to the
            # background rather than making the whole graph wait on it.
            deferred = inflight.defer(url, finish, _late_formatter(url))
            if not deferred:
                # Registry full or URL already in flight — finish inline so
                # we never silently drop the escalation.
                res = await finish()

        result = _result_payload(res, url, deferred=deferred)
        timing = f"{res.total_time_ms:.0f}ms"

        if deferred:
            success_log = (
                f"[SCRAPER] ⏳ {url} returned a thin result in {timing} "
                f"({result['word_count']} words). Full render moved to the "
                f"background — continuing with other sources. Pending now: "
                f"{inflight.pending_count()}."
            )
        elif result["insufficient_content"]:
            success_log = (
                f"[SCRAPER] ⚠️ Scraped {url} in {timing} but content looks "
                f"insufficient ({result['word_count']} words) after all "
                f"{len(res.all_results)} strategies — likely paywalled, "
                f"bot-blocked, or structured in a way this pipeline can't "
                f"extract. Treat as UNRELIABLE, not a successful scrape. "
                f"Retrying the same URL will not help; try a different source."
            )
        elif result.get("content_truncated"):
            success_log = (
                f"[SCRAPER] ✂️ Scraped {url} in {timing} "
                f"({result['word_count']} words via {res.best_strategy}) but the "
                f"extract was TRUNCATED to fit the {CONTENT_SAMPLE_CHARS:,}-char "
                f"budget"
                f"{' (tables clipped)' if result.get('tables_truncated') else ''}"
                f"{' (body clipped)' if result.get('prose_truncated') else ''}"
                f" — kept {result.get('content_chars_kept', 0):,} of "
                f"{result.get('content_chars', 0):,} chars. Lists and summary "
                f"tables from this source may be incomplete."
            )
        else:
            success_log = (
                f"[SCRAPER] ✅ Scraped {url} successfully "
                f"({result['word_count']} words via {res.best_strategy}, {timing})."
            )
        logs.append(success_log)
        print(success_log)

        # ---------------------------------------------------------
        # NEW: Inject Images into the UI Action Logs as HTML
        # ---------------------------------------------------------
        import re
        if getattr(res, "images", None):
            html_images = []
            for img_md in res.images:
                # Extract the raw URL from the Markdown string: ![alt](url)
                match = re.search(r'\!\[.*?\]\((.*?)\)', img_md)
                if match:
                    img_url = match.group(1)
                    # Format as a small inline thumbnail for the UI trace window
                    html_images.append(
                        f"<img src='{img_url}' style='max-height: 80px; max-width: 120px; margin: 4px; border-radius: 4px; object-fit: cover;' />"
                    )
            
            if html_images:
                # img_log = f"🖼️ <b>Scraped {len(html_images)} Image(s):</b><br>{''.join(html_images)}"
                img_log = f"🖼️ <b>Scraped {len(html_images)} Image(s)"
                logs.append(img_log)
                print(f"[SCRAPER] 🖼️ Sent {len(html_images)} image(s) to UI logs.")
        # ---------------------------------------------------------

    except Exception as e:
        error_log = f"[SCRAPER] ❌ Scrape error for {url}: {e}"
        logs.append(error_log)
        print(error_log)
        result = {
            "source": url,
            "error": f"Scrape failed: {str(e)}",
            "insufficient_content": True,
        }

    return {
        "context": [f"{_label(result)}: {json.dumps(result)}"],
        "action_logs": logs,
    }
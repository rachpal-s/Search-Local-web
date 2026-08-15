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

CONTENT_SAMPLE_CHARS = 1500  # limit sample size passed into downstream context

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


def _result_payload(res, url: str, *, deferred: bool = False) -> Dict[str, Any]:
    """Build the dict that goes into graph context for one scrape."""
    word_count = res.word_count
    content = res.content or "Failed to extract clean text."
    # Data pages: prepend tables so they survive the sample truncation.
    if res.tables_md and res.table_count:
        content = f"{res.tables_md}\n\n---\n\n{content}"

    if getattr(res, "images", None):
        img_md = "\n".join(res.images)
        content = f"{content}\n\n### Extracted Images\n{img_md}"

    payload = {
        "source": url,
        "title": res.title or "N/A",
        "word_count": word_count,
        "strategy": res.best_strategy or "none",
        "page_type": res.page_type,
        "table_count": res.table_count,
        "scraped_content": content[:CONTENT_SAMPLE_CHARS],
        "insufficient_content": word_count < MIN_WORD_COUNT,
    }

    # 2. APPEND IMAGES AFTER TRUNCATION so they are guaranteed to reach the LLM
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
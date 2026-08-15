"""
scraper/relevance.py — Filter crawled URLs by embedding similarity to category prompts.

Flow:
  1. Build category intent vector = mean of all prompt embeddings
  2. Embed each crawled page's title + summary
  3. Cosine similarity → keep pages above min_score
  4. Return top_n sorted by score
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scraper.crawler import CrawledPage

log = logging.getLogger("relevance")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    n = len(vectors[0])
    result = [0.0] * n
    for v in vectors:
        for i, x in enumerate(v):
            result[i] += x
    return [x / len(vectors) for x in result]


async def build_category_intent_vector(
    prompts: list[dict],
    ollama,
) -> list[float]:
    """
    Embed all category prompt texts and return their mean vector.
    This represents the semantic 'intent' of the category.
    """
    texts = []
    for p in prompts:
        # Use prompt text — first 300 chars is enough for intent
        text = (p.get("prompt") or p.get("prompt_text") or "").strip()[:300]
        if text:
            texts.append(text)

    if not texts:
        log.warning("No prompt texts found — cannot build intent vector")
        return []

    log.info("Building category intent vector from %d prompts", len(texts))
    try:
        embeddings = await ollama.embed(texts)
        intent = _mean_vector(embeddings)
        log.info("Intent vector built (dim=%d)", len(intent))
        return intent
    except Exception as e:
        log.error("Failed to embed category prompts: %s", e)
        return []


async def filter_by_relevance(
    pages: list["CrawledPage"],
    intent_vector: list[float],
    ollama,
    min_score: float = 0.70,
    top_n: int = 15,
) -> tuple[list["CrawledPage"], bool]:
    """
    Score each page's title+summary against the category intent vector.

    Returns:
        (filtered_pages, had_any_match)
        had_any_match=False means caller should fall back to homepage scrape
    """
    if not pages:
        return [], False

    if not intent_vector:
        log.warning("No intent vector — returning all pages unfiltered")
        return pages[:top_n], True

    # Build texts to embed: title + summary for each page
    texts = []
    for p in pages:
        parts = []
        if p.title:
            parts.append(p.title)
        if p.summary:
            parts.append(p.summary[:200])
        texts.append(" ".join(parts) if parts else p.url)

    log.info("Embedding %d page titles for relevance scoring", len(texts))
    try:
        # Batch embed in groups of 20 to avoid overwhelming Ollama
        all_embeddings = []
        batch_size = 20
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = await ollama.embed(batch)
            all_embeddings.extend(embeddings)
    except Exception as e:
        log.error("Embedding failed during relevance filter: %s", e)
        return pages[:top_n], True  # fallback: return top_n unfiltered

    # Score each page
    scored = []
    for page, emb in zip(pages, all_embeddings):
        score = _cosine(emb, intent_vector)
        page.score = score  # update in place
        scored.append((score, page))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Log top results for visibility
    for score, page in scored[:5]:
        log.info("  %.3f — %s", score, (page.title or page.url)[:80])

    # Filter by threshold
    filtered = [p for score, p in scored if score >= min_score]

    log.info("Relevance filter: %d/%d pages above %.2f threshold",
             len(filtered), len(pages), min_score)

    if not filtered:
        return [], False  # signal: fall back to homepage scrape

    return filtered[:top_n], True
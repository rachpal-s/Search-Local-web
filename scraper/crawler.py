"""
scraper/crawler.py — Domain crawler with recency-based page discovery.

Discovery order:
  1. Sitemap (sitemap.xml / sitemap_index.xml) — most reliable, has <lastmod>
  2. RSS/Atom feed (/feed, /rss, /feed.xml etc.) — structured, has pubDate
  3. Homepage link crawl — fallback, heuristic date scoring

Returns list of CrawledPage(url, title, published, score) sorted newest first.
"""
from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx

log = logging.getLogger("crawler")

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_WINDOW_HOURS = 6
MAX_SITEMAP_URLS = 200
MAX_RSS_ITEMS = 50
MAX_CRAWL_LINKS = 100
REQUEST_TIMEOUT = 15

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Common RSS feed paths to try
RSS_PATHS = [
    "/feed", "/feed/", "/rss", "/rss/", "/feed.xml", "/rss.xml",
    "/feeds/posts/default", "/atom.xml", "/feeds/all.rss.xml",
    "/news/rss", "/news/feed", "/latest/feed", "/markets/rss",
]

# Sitemap paths to try
SITEMAP_PATHS = [
    "/sitemap.xml", "/sitemap_index.xml", "/sitemap-index.xml",
    "/news-sitemap.xml", "/news_sitemap.xml", "/sitemap-news.xml",
    "/sitemaps/sitemap.xml",
]

# URL patterns that suggest news/article content
ARTICLE_PATTERNS = [
    r"/\d{4}/\d{2}/\d{2}/",          # /2026/05/23/
    r"/\d{4}-\d{2}-\d{2}[/-]",       # /2026-05-23/
    r"[/-](news|article|story|post|blog)[/-]",
    r"[/-](markets|stocks|economy|business|finance|politics)[/-]",
    r"-\d{7,}[/-]?$",                 # ends with long numeric ID
]
ARTICLE_RE = re.compile("|".join(ARTICLE_PATTERNS), re.IGNORECASE)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class CrawledPage:
    url: str
    title: str = ""
    published: Optional[datetime] = None
    summary: str = ""
    score: float = 0.0           # recency score 0-1
    source: str = ""             # "sitemap" | "rss" | "crawl"


# ── Date parsing ──────────────────────────────────────────────────────────────

def _parse_dt(text: str) -> Optional[datetime]:
    """Parse various date formats into UTC datetime."""
    if not text:
        return None
    text = text.strip()
    formats = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%a, %d %b %Y %H:%M:%S %z",    # RFC 2822 (RSS)
        "%a, %d %b %Y %H:%M:%S GMT",
        "%a, %d %b %Y %H:%M:%S +0000",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(text[:len(fmt)+5].strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except (ValueError, TypeError):
            continue
    return None


def _recency_score(dt: Optional[datetime], window_hours: int) -> float:
    """Score 1.0 if just published, 0.0 if older than window."""
    if dt is None:
        return 0.0
    now = datetime.now(timezone.utc)
    age = (now - dt).total_seconds()
    window = window_hours * 3600
    if age < 0:
        return 1.0
    if age > window:
        return 0.0
    return 1.0 - (age / window)


# ── Sitemap discovery ─────────────────────────────────────────────────────────

async def _fetch_text(client: httpx.AsyncClient, url: str) -> Optional[str]:
    try:
        r = await client.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT,
                             follow_redirects=True)
        if r.status_code == 200:
            return r.text
    except Exception as e:
        log.debug("fetch %s failed: %s", url, e)
    return None


async def _try_sitemap(client: httpx.AsyncClient, base: str,
                       window_hours: int) -> list[CrawledPage]:
    """Try sitemap.xml paths, parse <url> entries with <lastmod>."""
    pages = []

    for path in SITEMAP_PATHS:
        url = base + path
        text = await _fetch_text(client, url)
        if not text or "<" not in text:
            continue

        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            continue

        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
              "news": "http://www.google.com/schemas/sitemap-news/0.9"}

        # Sitemap index — recurse into sub-sitemaps
        sitemaps = root.findall(".//sm:sitemap/sm:loc", ns)
        if sitemaps:
            log.info("Sitemap index found at %s — %d sub-sitemaps", url, len(sitemaps))
            # Prefer news sitemaps
            sub_urls = [s.text.strip() for s in sitemaps if s.text]
            news_subs = [u for u in sub_urls if "news" in u.lower()]
            to_fetch = (news_subs or sub_urls)[:5]
            for sub_url in to_fetch:
                sub_text = await _fetch_text(client, sub_url)
                if sub_text:
                    pages.extend(_parse_sitemap_xml(sub_text, window_hours, ns))
            if pages:
                log.info("Sitemap: %d recent pages from index", len(pages))
                return pages[:MAX_SITEMAP_URLS]

        # Regular sitemap
        parsed = _parse_sitemap_xml(text, window_hours, ns)
        if parsed:
            log.info("Sitemap: %d recent pages from %s", len(parsed), path)
            return parsed[:MAX_SITEMAP_URLS]

    return []


def _parse_sitemap_xml(text: str, window_hours: int, ns: dict) -> list[CrawledPage]:
    pages = []
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []

    ns_sm = "http://www.sitemaps.org/schemas/sitemap/0.9"
    ns_news = "http://www.google.com/schemas/sitemap-news/0.9"

    for url_el in root.findall(f".//{{{ns_sm}}}url"):
        loc = url_el.findtext(f"{{{ns_sm}}}loc", "").strip()
        if not loc:
            continue

        # Try <lastmod>
        lastmod_text = url_el.findtext(f"{{{ns_sm}}}lastmod", "")
        dt = _parse_dt(lastmod_text)

        # Try Google News sitemap <news:publication_date>
        if dt is None:
            pub_date = url_el.findtext(f"{{{ns_news}}}news/{{{ns_news}}}publication_date", "")
            dt = _parse_dt(pub_date)

        score = _recency_score(dt, window_hours)
        if dt is None or score > 0:  # include undated URLs too (can't filter them out)
            title = url_el.findtext(f"{{{ns_news}}}news/{{{ns_news}}}title", "")
            pages.append(CrawledPage(
                url=loc, title=title, published=dt,
                score=score if dt else 0.5, source="sitemap"
            ))

    # Sort: dated+recent first, then undated
    pages.sort(key=lambda p: (p.published is not None, p.score), reverse=True)
    # Filter to recency window (keep undated as fallback)
    recent = [p for p in pages if p.score > 0 or p.published is None]
    return recent


# ── RSS discovery ─────────────────────────────────────────────────────────────

async def _try_rss(client: httpx.AsyncClient, base: str,
                   window_hours: int) -> list[CrawledPage]:
    """Try common RSS feed paths and parse items."""
    for path in RSS_PATHS:
        url = base + path
        text = await _fetch_text(client, url)
        if not text or "<" not in text:
            continue
        pages = _parse_feed(text, window_hours)
        if pages:
            log.info("RSS: %d recent items from %s", len(pages), path)
            return pages[:MAX_RSS_ITEMS]
    return []


def _parse_feed(text: str, window_hours: int) -> list[CrawledPage]:
    """Parse RSS or Atom feed XML."""
    pages = []
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []

    tag = root.tag.lower()

    # Atom feed
    if "atom" in tag or "feed" in tag:
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for entry in root.findall(".//a:entry", ns) or root.findall(".//entry"):
            link_el = entry.find(".//a:link", ns) or entry.find(".//link")
            url = ""
            if link_el is not None:
                url = link_el.get("href", "") or link_el.text or ""
            title = (entry.findtext(".//a:title", "", ns) or
                     entry.findtext(".//title", "")).strip()
            pub = (entry.findtext(".//a:published", "", ns) or
                   entry.findtext(".//a:updated", "", ns) or
                   entry.findtext(".//published", "") or
                   entry.findtext(".//updated", "")).strip()
            dt = _parse_dt(pub)
            score = _recency_score(dt, window_hours)
            summary = (entry.findtext(".//a:summary", "", ns) or
                       entry.findtext(".//summary", ""))[:300]
            if url and (score > 0 or dt is None):
                pages.append(CrawledPage(url=url, title=title, published=dt,
                                         score=score, summary=summary, source="rss"))

    # RSS feed
    else:
        for item in root.findall(".//item"):
            url = (item.findtext("link") or "").strip()
            title = (item.findtext("title") or "").strip()
            pub = (item.findtext("pubDate") or
                   item.findtext("dc:date", namespaces={"dc": "http://purl.org/dc/elements/1.1/"}) or
                   "").strip()
            dt = _parse_dt(pub)
            score = _recency_score(dt, window_hours)
            summary = (item.findtext("description") or "")[:300]
            # Strip HTML from summary
            summary = re.sub(r"<[^>]+>", "", summary).strip()
            if url and (score > 0 or dt is None):
                pages.append(CrawledPage(url=url, title=title, published=dt,
                                         score=score, summary=summary, source="rss"))

    pages.sort(key=lambda p: p.score, reverse=True)
    return pages


# ── Homepage crawl fallback ───────────────────────────────────────────────────

async def _try_homepage_crawl(client: httpx.AsyncClient, base: str,
                               window_hours: int) -> list[CrawledPage]:
    """Scrape homepage, extract links, score by URL pattern and date in URL."""
    text = await _fetch_text(client, base)
    if not text:
        return []

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")
    pages = []
    seen = set()
    parsed_base = urlparse(base)
    today = datetime.now(timezone.utc)

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("javascript"):
            continue

        url = urljoin(base, href)
        parsed = urlparse(url)

        # Same domain only
        if parsed.netloc and parsed.netloc != parsed_base.netloc:
            continue
        if url in seen:
            continue
        seen.add(url)

        # Score by URL patterns
        score = 0.0
        path = parsed.path

        # Date in URL — strongest signal
        date_match = re.search(r"(\d{4})[/-](\d{2})[/-](\d{2})", path)
        if date_match:
            try:
                url_dt = datetime(int(date_match.group(1)),
                                  int(date_match.group(2)),
                                  int(date_match.group(3)),
                                  tzinfo=timezone.utc)
                score = _recency_score(url_dt, window_hours * 24)  # wider window for URL dates
                if score > 0:
                    pages.append(CrawledPage(
                        url=url,
                        title=a.get_text(strip=True)[:100],
                        published=url_dt,
                        score=score,
                        source="crawl"
                    ))
                    continue
            except ValueError:
                pass

        # Article-like URL pattern
        if ARTICLE_RE.search(path):
            score = 0.4
            pages.append(CrawledPage(
                url=url,
                title=a.get_text(strip=True)[:100],
                score=score,
                source="crawl"
            ))

    pages.sort(key=lambda p: p.score, reverse=True)
    log.info("Homepage crawl: %d article links found", len(pages))
    return pages[:MAX_CRAWL_LINKS]


# ── Main entry point ──────────────────────────────────────────────────────────

async def discover_recent_pages(
    domain_url: str,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    max_results: int = 50,
) -> list[CrawledPage]:
    """
    Discover recently published pages on a domain.
    Tries sitemap → RSS → homepage crawl in order.

    Args:
        domain_url: Full URL e.g. "https://moneycontrol.com" or "https://ndtv.com/business"
        window_hours: Only return pages published within this many hours
        max_results: Maximum pages to return

    Returns:
        List of CrawledPage sorted by recency score (newest first)
    """
    parsed = urlparse(domain_url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    log.info("Crawling %s (window=%dh, max=%d)", base, window_hours, max_results)

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT,
                                  follow_redirects=True) as client:
        # 1. Try sitemap
        pages = await _try_sitemap(client, base, window_hours)
        if pages:
            log.info("Using sitemap: %d pages", len(pages))
            return pages[:max_results]

        # 2. Try RSS
        pages = await _try_rss(client, base, window_hours)
        if pages:
            log.info("Using RSS: %d pages", len(pages))
            return pages[:max_results]

        # 3. Fallback: homepage crawl
        pages = await _try_homepage_crawl(client, base, window_hours)
        log.info("Using homepage crawl: %d pages", len(pages))
        return pages[:max_results]
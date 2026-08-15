"""scraper/detector.py — Detect what kind of page we're dealing with."""
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag


@dataclass
class PageSignals:
    page_type: str          # "homepage" | "article" | "data" | "unknown"
    confidence: float       # 0–1
    headline_count: int
    table_count: int
    word_count: int
    has_article_schema: bool
    has_pagination: bool
    is_root_path: bool
    reason: str


def detect_page_type(html: str, url: str) -> PageSignals:
    """
    Heuristically classify a page into one of:
      - homepage  : listing of articles/headlines (root domain or /news, /markets etc.)
      - article   : single long-form article/blog post
      - data      : page dominated by tables (stock data, indices, dashboards)
      - unknown   : fallback
    """
    soup = BeautifulSoup(html, "lxml")
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")

    # ── Signal: is root path? ─────────────────────────────────────────────────
    is_root = path == "" or path == "/"

    # ── Signal: count links that look like article links ─────────────────────
    article_links = _count_article_links(soup)

    # ── Signal: table count ───────────────────────────────────────────────────
    # Use thead presence OR >=2 th cells as signal — NOT row count.
    # JS-rendered pages (Moneycontrol etc.) serve empty <tbody> in static HTML;
    # rows are injected after JS runs. We must detect the intent from structure alone.
    tables = soup.find_all("table")
    data_tables = sum(
        1 for t in tables
        if isinstance(t, Tag) and (
            t.find("thead") is not None
            or len(t.find_all("th")) >= 2
            or len(t.find_all("tr")) > 2
        )
    )

    # ── Signal: word count of body text ──────────────────────────────────────
    body = soup.find("body")
    body_text = body.get_text(separator=" ", strip=True) if body else ""
    word_count = len(body_text.split())

    # ── Signal: structured data / schema ─────────────────────────────────────
    has_article_schema = bool(
        soup.find("script", attrs={"type": "application/ld+json"})
        or soup.find(attrs={"itemtype": re.compile(r"Article|NewsArticle|BlogPost", re.I)})
        or soup.find("article")
    )

    # ── Signal: pagination ────────────────────────────────────────────────────
    has_pagination = bool(
        soup.find(class_=re.compile(r"pagination|pager", re.I))
        or soup.find("nav", attrs={"aria-label": re.compile(r"page", re.I)})
    )

    # ── Signal: path keywords ────────────────────────────────────────────────
    homepage_path_patterns = re.compile(
        r"^(/|/home|/news|/markets?|/finance|/business|/world|/tech|/sports?|/latest|/top-stories?)$",
        re.I,
    )
    article_path_patterns = re.compile(
        r"(/article|/story|/post|/blog|/news/\d|/\d{4}/\d{2}|/read|/detail)",
        re.I,
    )
    data_path_patterns = re.compile(
        r"(indices|index|market|stocks?|screener|quotes?|rates?|data|dashboard|report)",
        re.I,
    )

    path_is_homepage = bool(homepage_path_patterns.match(path)) or is_root
    path_is_article = bool(article_path_patterns.search(path))
    path_is_data = bool(data_path_patterns.search(path))

    # ── Decision logic ────────────────────────────────────────────────────────
    # Data page: many tables, path suggests data
    # If also has many article links → mixed (e.g. ET Markets, Moneycontrol)
    if data_tables >= 2 or (data_tables >= 1 and path_is_data):
        is_mixed = article_links >= 5
        return PageSignals(
            page_type="mixed" if is_mixed else "data",
            confidence=0.85 if data_tables >= 2 else 0.7,
            headline_count=article_links,
            table_count=data_tables,
            word_count=word_count,
            has_article_schema=has_article_schema,
            has_pagination=has_pagination,
            is_root_path=is_root,
            reason=f"{data_tables} table(s) + {article_links} article links" if is_mixed else f"{data_tables} data table(s) found",
        )

    # Homepage: many article links, root/listing path
    if article_links >= 5 and (path_is_homepage or has_pagination or article_links >= 10):
        return PageSignals(
            page_type="homepage",
            confidence=min(0.95, 0.6 + article_links * 0.02),
            headline_count=article_links,
            table_count=data_tables,
            word_count=word_count,
            has_article_schema=has_article_schema,
            has_pagination=has_pagination,
            is_root_path=is_root,
            reason=f"{article_links} article links detected",
        )

    # Article: has article schema, long text, article-like path
    if has_article_schema or path_is_article or word_count > 400:
        return PageSignals(
            page_type="article",
            confidence=0.8 if has_article_schema else 0.65,
            headline_count=article_links,
            table_count=data_tables,
            word_count=word_count,
            has_article_schema=has_article_schema,
            has_pagination=has_pagination,
            is_root_path=is_root,
            reason="article schema / long-form content",
        )

    # Fallback: if root path and some links, treat as homepage
    if is_root or path_is_homepage:
        return PageSignals(
            page_type="homepage",
            confidence=0.5,
            headline_count=article_links,
            table_count=data_tables,
            word_count=word_count,
            has_article_schema=has_article_schema,
            has_pagination=has_pagination,
            is_root_path=is_root,
            reason="root path fallback",
        )

    return PageSignals(
        page_type="unknown",
        confidence=0.3,
        headline_count=article_links,
        table_count=data_tables,
        word_count=word_count,
        has_article_schema=has_article_schema,
        has_pagination=has_pagination,
        is_root_path=is_root,
        reason="no clear signals",
    )


def _count_article_links(soup: BeautifulSoup) -> int:
    """Count links that look like article/story links."""
    ARTICLE_LINK = re.compile(
        r"(/article|/story|/post|/blog|/news/|/\d{4}/\d{2}/|/read|/p/|/detail|\.html|\.htm|\.cms"
        r"|-news-|/newsdetail|/storydetail|/articleshow)",
        re.I,
    )
    count = 0
    for a in soup.find_all("a", href=True):
        if not isinstance(a, Tag):
            continue
        href = a.get("href", "")
        text = a.get_text(strip=True)
        if ARTICLE_LINK.search(href) and len(text) > 20:
            count += 1
    return count


def extract_headlines(html: str, base_url: str, max_count: int = 50) -> list[dict]:
    """
    Extract headlines + links from a homepage/listing page.
    Returns list of {title, url, summary, section, image}.
    """
    soup = BeautifulSoup(html, "lxml")
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    ARTICLE_LINK = re.compile(
        r"(/article|/story|/post|/blog|/news/|/\d{4}/\d{2}/|/read|/p/|/detail|\.html|\.htm)",
        re.I,
    )
    NOISE = re.compile(
        r"(login|signup|subscribe|adverti|cookie|privacy|contact|about|career|help|faq)",
        re.I,
    )

    seen_urls: set = set()
    seen_titles: set = set()
    results = []

    # Try structured containers first (article, h2/h3 with links, card divs)
    containers = soup.find_all(["article", "h2", "h3", "h4"])
    for el in containers:
        if not isinstance(el, Tag):
            continue
        a_tag = el if el.name == "a" else el.find("a", href=True)
        if not a_tag:
            # Look for child link
            for a in el.find_all("a", href=True):
                a_tag = a
                break
        if not a_tag:
            continue

        href = a_tag.get("href", "")
        if not href or href.startswith("#") or href.startswith("javascript"):
            continue

        # Resolve relative URLs
        if href.startswith("//"):
            href = parsed.scheme + ":" + href
        elif href.startswith("/"):
            href = base + href
        elif not href.startswith("http"):
            href = base + "/" + href.lstrip("/")

        if NOISE.search(href):
            continue
        if href in seen_urls:
            continue

        # Get title text
        title = a_tag.get_text(strip=True)
        if not title and el.name in ("h2", "h3", "h4"):
            title = el.get_text(strip=True)
        if not title or len(title) < 15 or len(title) > 300:
            continue
        if title in seen_titles:
            continue

        # Try to find a summary (next sibling p or nearby p)
        summary = ""
        parent = el.parent
        if parent:
            for sib in el.find_next_siblings(["p", "div"])[:2]:
                t = sib.get_text(strip=True)
                if 20 < len(t) < 300:
                    summary = t
                    break

        # Try to detect section from breadcrumb/category context
        section = _infer_section(el, soup)

        seen_urls.add(href)
        seen_titles.add(title)
        results.append({
            "title": title,
            "url": href,
            "summary": summary,
            "section": section,
        })

        if len(results) >= max_count:
            break

    # Fallback: scan all <a> tags if structured pass yielded too few
    if len(results) < 5:
        for a in soup.find_all("a", href=True):
            if not isinstance(a, Tag):
                continue
            href = a.get("href", "")
            title = a.get_text(strip=True)
            if not href or not title or len(title) < 20 or len(title) > 300:
                continue
            if not ARTICLE_LINK.search(href):
                continue
            if NOISE.search(href) or NOISE.search(title):
                continue

            if href.startswith("//"):
                href = parsed.scheme + ":" + href
            elif href.startswith("/"):
                href = base + href
            elif not href.startswith("http"):
                continue

            if href in seen_urls or title in seen_titles:
                continue

            seen_urls.add(href)
            seen_titles.add(title)
            results.append({
                "title": title,
                "url": href,
                "summary": "",
                "section": "",
            })
            if len(results) >= max_count:
                break

    return results


def _infer_section(el: Tag, soup: BeautifulSoup) -> str:
    """Try to infer section/category from surrounding DOM context."""
    # Walk up to find a section header
    for parent in el.parents:
        if not isinstance(parent, Tag):
            continue
        for heading in parent.find_all(["h1", "h2", "h3"], limit=1):
            text = heading.get_text(strip=True)
            if 3 < len(text) < 60:
                return text
    return ""
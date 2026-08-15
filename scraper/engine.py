"""scraper/engine.py — Multi-strategy scraper, Trafilatura-first."""
import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import urljoin, urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup, Tag
from goose3 import Goose
from newspaper import Article
from readability import Document

from config import get_settings

cfg = get_settings()

# ── Browser headers ───────────────────────────────────────────────────────────

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124"',
    "Sec-Ch-Ua-Platform": '"Windows"',
}

IS_BLOCKED = re.compile(r"host not in allowlist|access denied|403 forbidden", re.I)

# Rotated on 403/429/503. Googlebot last: many Indian news sites whitelist it.
UA_POOL = [
    UA,
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Linux; Android 13; SM-S911B) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
]

# Soft walls: page returns 200 but body is a challenge / JS-required stub.
IS_WALL = re.compile(
    r"(verify you are human|checking your browser|cf-browser-verification|"
    r"enable javascript to continue|are you a robot|request unsuccessful|"
    r"access to this page has been denied|ddos protection by)", re.I,
)

# Ad/analytics hosts aborted in Playwright so networkidle can actually settle.
_AD_HOST = re.compile(
    r"(doubleclick|googlesyndication|googletagmanager|google-analytics|"
    r"taboola|outbrain|scorecardresearch|facebook\.net|adsystem|adnxs|"
    r"criteo|pubmatic|rubiconproject|izooto|onesignal|clevertap|moatads|"
    r"quantserve|amazon-adsystem|smartadserver|indexww)", re.I,
)

# Per-site article containers — doubles as the Playwright wait_selector.
_SITE_SELECTORS = {
    "ndtv.com": "div.sp-cn.ins_storybody, div.story__content, div[itemprop='articleBody']",
    "thewire.in": "div.entry-content, div.article-content, article",
    "indianexpress.com": "div.story_details, div#pcl-full-content",
    "thehindu.com": "div.articlebodycontent, div[itemprop='articleBody']",
    "timesofindia.indiatimes.com": "div._s30J, div.js_tbl_article",
    "hindustantimes.com": "div.detail, div.storyDetails",
    "moneycontrol.com": "div.content_wrapper, div#contentdata, table",
    "livemint.com": "div.storyPage_storyContent__m_MYl, div#storyContent",
    "scroll.in": "div.story-element-text, article",
}


def _site_selector(url: str) -> Optional[str]:
    host = urlparse(url).netloc.lower()
    for dom, sel in _SITE_SELECTORS.items():
        if host == dom or host.endswith("." + dom):
            return sel
    return None

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class StrategyResult:
    strategy: str
    success: bool
    title: Optional[str] = None
    text: Optional[str] = None
    word_count: int = 0
    time_ms: float = 0.0
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)


@dataclass
class ScrapeResult:
    url: str
    page_type: str
    best_strategy: Optional[str]
    title: Optional[str]
    content: Optional[str]
    word_count: int
    tables_md: Optional[str]       # Markdown tables if data page
    table_count: int
    fetch_time_ms: float
    total_time_ms: float
    all_results: list[StrategyResult]
    metadata: dict
    headlines: list[dict] = field(default_factory=list)   # for homepage
    images: list[str] = field(default_factory=list)


# ── HTML fetcher ──────────────────────────────────────────────────────────────

async def _fetch_once(url: str, ua: str, referer: Optional[str] = None,
                      timeout: Optional[float] = None) -> tuple[str, int]:
    """Single GET. Returns (body, status). Never raises on HTTP status."""
    hdrs = dict(HEADERS)
    hdrs["User-Agent"] = ua
    if referer:
        hdrs["Referer"] = referer
    if "Mobile" in ua or "Android" in ua:
        hdrs["Sec-Ch-Ua-Mobile"] = "?1"
        hdrs["Sec-Ch-Ua-Platform"] = '"Android"'
    kw = dict(headers=hdrs, follow_redirects=True,
              timeout=timeout or cfg.scraper_timeout)
    try:
        client = httpx.AsyncClient(http2=True, **kw)   # needs `pip install h2`
    except Exception:
        client = httpx.AsyncClient(**kw)
    async with client:
        r = await client.get(url)
        return r.text, r.status_code


async def fetch_html(url: str, *, referer: str = "https://www.google.com/",
                     allow_error_body: bool = False) -> tuple[str, float]:
    """Fetch with UA rotation + Google referer. Retries on 403/429/5xx.

    allow_error_body=True returns the body of a blocked response instead of
    raising — a 403 page often still carries the JSON-LD articleBody, and
    throwing it away was costing us a full Playwright escalation.
    """
    t0 = time.perf_counter()
    last_err: Optional[Exception] = None
    fallback_html = ""

    for i, ua in enumerate(UA_POOL):
        ref = referer if i == 0 else "https://news.google.com/"
        try:
            html, status = await _fetch_once(url, ua, ref)
        except Exception as e:
            last_err = e
            continue

        if status < 400 and html:
            head = html[:400]
            if IS_BLOCKED.search(head):
                last_err = ValueError(f"Blocked: {head[:120]}")
                fallback_html = fallback_html or html
                continue
            if IS_WALL.search(html[:4000]) and len(html) < 20_000:
                last_err = ValueError("Soft wall / challenge page")
                fallback_html = fallback_html or html
                continue
            return html, (time.perf_counter() - t0) * 1000

        last_err = ValueError(f"HTTP {status}")
        if html:
            fallback_html = fallback_html or html
        if status in (401, 402, 404, 410):
            break            # no UA rotation will fix these

    if allow_error_body and fallback_html:
        return fallback_html, (time.perf_counter() - t0) * 1000
    raise last_err or ValueError("fetch failed")


# ── Text cleaner ──────────────────────────────────────────────────────────────

JUNK_LINE = re.compile(
    r"(accept all cookies|cookie policy|subscribe now|sign up|log in|"
    r"advertisement|follow us on|share this|you might also like|"
    r"trending now|buy now|click here|sponsored|promoted content|"
    r"enable javascript|please wait|fetching data|loading\.\.\.)",
    re.I,
)

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"<[^>]+>", "", text)
    lines = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s or len(s) < 3:
            continue
        if len(s) < 120 and JUNK_LINE.search(s):
            continue
        lines.append(s)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _html_to_text(fragment: str) -> str:
    """HTML fragment (JSON-LD body, WP rendered content) → clean text."""
    if not fragment:
        return ""
    if "<" not in fragment:
        return clean_text(fragment)
    return clean_text(BeautifulSoup(fragment, "lxml").get_text("\n"))


def _title_from_html(html: str) -> Optional[str]:
    try:
        soup = BeautifulSoup(html, "lxml")
        og = soup.find("meta", attrs={"property": "og:title"})
        if isinstance(og, Tag) and og.get("content"):
            return str(og["content"]).strip()
        for sel in ("h1", "title"):
            el = soup.find(sel)
            if el:
                t = el.get_text(strip=True)
                if t:
                    return t
    except Exception:
        pass
    return None


def _junk_ratio(text: Optional[str]) -> float:
    """Fraction of lines too short to be prose — proxy for nav/link soup.

    A BS4 dump of a homepage shell is 200 four-word lines; a real article is
    long paragraphs. This is what stops a junk-heavy 400-word extraction from
    outranking a clean 250-word one in _pick_best.
    """
    if not text:
        return 1.0
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return 1.0
    return sum(1 for l in lines if len(l) < 45) / len(lines)


def _ok(r: "StrategyResult", min_words: int) -> bool:
    return bool(r.success and r.text and r.word_count >= min_words
                and _junk_ratio(r.text) < 0.70)


# ── Noise tags and ad patterns ────────────────────────────────────────────────

NOISE_TAGS = [
    "script", "style", "nav", "header", "footer", "aside", "noscript",
    "form", "iframe", "picture", "svg", "button", "input", "select",
    "textarea", "ins", "figure", "figcaption",
]
AD_PAT = re.compile(
    r"(^ad[-_]|[-_]ad$|^ads$|advertisement|sponsor|promo|banner|cookie-|"
    r"popup|modal|overlay|^sidebar|widget-ad|social-share|share-bar|"
    r"newsletter|subscribe-box|taboola|outbrain|gpt-ad|dfp-|"
    r"interstitial|sticky-ad|floating-ad|lightbox)",
    re.I,
)


# ── Strategy 0a: JSON-LD (highest trust, ~5ms, no browser) ───────────────────
#
# Client-rendered news sites ship the article body in the static HTML as
# DATA even when the DOM is an empty shell. ndtv.com et al. embed the full
# NewsArticle.articleBody for crawlers. This is why a "JS-rendered page"
# usually does NOT need Playwright.

_LD_TYPES = {
    "newsarticle", "article", "blogposting", "reportagenewsarticle",
    "opinionnewsarticle", "analysisnewsarticle", "backgroundnewsarticle",
    "reviewnewsarticle", "liveblogposting", "webpage", "creativework",
}
_LD_BODY_KEYS = ("articleBody", "articlebody", "text", "description")


def _walk_ld(node, out: list) -> None:
    if isinstance(node, dict):
        raw_t = node.get("@type")
        types = {str(x).lower() for x in
                 (raw_t if isinstance(raw_t, list) else [raw_t]) if x}
        if types & _LD_TYPES:
            for k in _LD_BODY_KEYS:
                if isinstance(node.get(k), str) and len(node[k]) > 200:
                    out.append((node.get("headline") or node.get("name"), node[k]))
                    break
        for v in node.values():
            _walk_ld(v, out)
    elif isinstance(node, list):
        for v in node:
            _walk_ld(v, out)


def _jsonld(html: str, url: Optional[str] = None) -> StrategyResult:
    t0 = time.perf_counter()
    try:
        soup = BeautifulSoup(html, "lxml")
        cands: list = []
        for tag in soup.find_all("script",
                                 attrs={"type": re.compile(r"ld\+json", re.I)}):
            raw = (tag.string or tag.get_text() or "").strip().lstrip("\ufeff")
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                try:    # tolerate trailing commas / concatenated objects
                    data = json.loads(re.sub(r",\s*([}\]])", r"\1", raw))
                except Exception:
                    continue
            _walk_ld(data, cands)

        if not cands:
            return StrategyResult("jsonld", False, error="No ld+json article body",
                                  time_ms=(time.perf_counter() - t0) * 1000)

        title, body = max(cands, key=lambda c: len(c[1]))
        text = _html_to_text(body)
        wc = len(text.split()) if text else 0
        if wc < 20:
            return StrategyResult("jsonld", False, error="Insufficient content",
                                  time_ms=(time.perf_counter() - t0) * 1000)
        return StrategyResult("jsonld", True, title=title or _title_from_html(html),
                              text=text, word_count=wc,
                              time_ms=(time.perf_counter() - t0) * 1000,
                              extra={"ld_blocks": len(cands)})
    except Exception as e:
        return StrategyResult("jsonld", False, error=str(e)[:200],
                              time_ms=(time.perf_counter() - t0) * 1000)


# ── Strategy 0b: embedded hydration state ────────────────────────────────────
#
# The other half of the pre-hydration problem: Next/Nuxt/Apollo stores the
# body in a JS object the static fetch already has.

_STATE_MARKERS = [
    re.compile(r'id=["\']__NEXT_DATA__["\'][^>]*>\s*(?=\{)'),
    re.compile(r'window\.__INITIAL_STATE__\s*=\s*(?=\{)'),
    re.compile(r'window\.__PRELOADED_STATE__\s*=\s*(?=\{)'),
    re.compile(r'window\.__APOLLO_STATE__\s*=\s*(?=\{)'),
    re.compile(r'window\.__NUXT__\s*=\s*(?=\{)'),
    re.compile(r'window\.__data\s*=\s*(?=\{)'),
    re.compile(r'id=["\']__NUXT_DATA__["\'][^>]*>\s*(?=\{)'),
]
_BODY_KEYS = (
    "articlebody", "article_body", "storycontent", "story_content", "fullstory",
    "content_html", "bodyhtml", "body_html", "rendered", "story", "body",
    "content", "text", "description",
)


def _slice_json(text: str, start: int) -> Optional[str]:
    """Balanced-brace slice from `start`, string- and escape-aware."""
    depth, in_str, esc = 0, False, False
    for i in range(start, min(len(text), start + 4_000_000)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _walk_state(node, out: list, depth: int = 0) -> None:
    if depth > 8:
        return
    if isinstance(node, dict):
        for k, v in node.items():
            kl = str(k).lower().replace("-", "_")
            if isinstance(v, str):
                named = any(b in kl for b in _BODY_KEYS)
                if named and len(v) > 400:
                    out.append(v)
                elif len(v) > 2000 and ("<p" in v or ". " in v):
                    out.append(v)          # unnamed but unmistakably prose
            else:
                _walk_state(v, out, depth + 1)
    elif isinstance(node, list):
        for v in node[:200]:
            _walk_state(v, out, depth + 1)


def _embedded_state(html: str, url: Optional[str] = None) -> StrategyResult:
    t0 = time.perf_counter()
    try:
        bodies: list[str] = []
        for pat in _STATE_MARKERS:
            for m in pat.finditer(html):
                blob = _slice_json(html, m.end())
                if not blob:
                    continue
                try:
                    data = json.loads(blob)
                except Exception:
                    continue
                _walk_state(data, bodies)
            if bodies:
                break

        if not bodies:
            return StrategyResult("embedded_state", False,
                                  error="No hydration state found",
                                  time_ms=(time.perf_counter() - t0) * 1000)

        # Longest single body beats concatenating everything (avoids stitching
        # in "related stories" blurbs from the same store).
        text = _html_to_text(max(bodies, key=len))
        wc = len(text.split()) if text else 0
        if wc < 20:
            return StrategyResult("embedded_state", False, error="Insufficient content",
                                  time_ms=(time.perf_counter() - t0) * 1000)
        return StrategyResult("embedded_state", True, title=_title_from_html(html),
                              text=text, word_count=wc,
                              time_ms=(time.perf_counter() - t0) * 1000,
                              extra={"state_bodies": len(bodies)})
    except Exception as e:
        return StrategyResult("embedded_state", False, error=str(e)[:200],
                              time_ms=(time.perf_counter() - t0) * 1000)


# ── Strategy 0c: per-site DOM selector ───────────────────────────────────────

def _site_dom(html: str, url: str) -> StrategyResult:
    t0 = time.perf_counter()
    sel = _site_selector(url)
    if not sel:
        return StrategyResult("site_dom", False, error="No selector for domain",
                              time_ms=(time.perf_counter() - t0) * 1000)
    try:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(NOISE_TAGS):
            tag.decompose()
        chunks = [el.get_text("\n") for el in soup.select(sel) if isinstance(el, Tag)]
        text = clean_text("\n".join(chunks))
        wc = len(text.split()) if text else 0
        if wc < 20:
            return StrategyResult("site_dom", False, error="Selector matched nothing",
                                  time_ms=(time.perf_counter() - t0) * 1000)
        return StrategyResult("site_dom", True, title=_title_from_html(html),
                              text=text, word_count=wc,
                              time_ms=(time.perf_counter() - t0) * 1000)
    except Exception as e:
        return StrategyResult("site_dom", False, error=str(e)[:200],
                              time_ms=(time.perf_counter() - t0) * 1000)


# ── Strategy 1: Trafilatura ───────────────────────────────────────────────────

def _trafilatura(html: str, url: str) -> StrategyResult:
    t0 = time.perf_counter()
    try:
        text = trafilatura.extract(
            html, url=url,
            include_tables=True, include_links=False,
            include_images=False, no_fallback=False,
            favor_recall=True,
        )
        meta = trafilatura.extract_metadata(html, default_url=url)
        title = meta.title if meta else None
        text = clean_text(text)
        wc = len(text.split()) if text else 0
        if not text or wc < 20:
            return StrategyResult("trafilatura", False,
                                  error="Insufficient content",
                                  time_ms=(time.perf_counter()-t0)*1000)
        return StrategyResult("trafilatura", True, title=title,
                              text=text, word_count=wc,
                              time_ms=(time.perf_counter()-t0)*1000)
    except Exception as e:
        return StrategyResult("trafilatura", False, error=str(e)[:200],
                              time_ms=(time.perf_counter()-t0)*1000)


# ── Strategy 2: Newspaper3k ───────────────────────────────────────────────────

def _newspaper3k(url: str, html: Optional[str] = None) -> StrategyResult:
    t0 = time.perf_counter()
    try:
        article = Article(url, browser_user_agent=UA,
                          request_timeout=cfg.scraper_timeout)
        if html:
            article.download(input_html=html)   # reuse fetched HTML, no 2nd GET
        else:
            article.download()
        article.parse()
        text = clean_text(article.text)
        wc = len(text.split()) if text else 0
        if not text or wc < 20:
            return StrategyResult("newspaper3k", False,
                                  error="Insufficient content",
                                  time_ms=(time.perf_counter()-t0)*1000)
        extra = {}
        if article.authors:
            extra["authors"] = article.authors
        if article.publish_date:
            extra["publish_date"] = str(article.publish_date)
        return StrategyResult("newspaper3k", True, title=article.title,
                              text=text, word_count=wc,
                              time_ms=(time.perf_counter()-t0)*1000,
                              extra=extra)
    except Exception as e:
        return StrategyResult("newspaper3k", False, error=str(e)[:200],
                              time_ms=(time.perf_counter()-t0)*1000)


# ── Strategy 3: Readability ───────────────────────────────────────────────────

def _readability(html: str) -> StrategyResult:
    t0 = time.perf_counter()
    try:
        doc = Document(html)
        soup = BeautifulSoup(doc.summary(), "lxml")
        text = clean_text(soup.get_text(separator="\n"))
        wc = len(text.split()) if text else 0
        if not text or wc < 20:
            return StrategyResult("readability", False,
                                  error="Insufficient content",
                                  time_ms=(time.perf_counter()-t0)*1000)
        return StrategyResult("readability", True, title=doc.title(),
                              text=text, word_count=wc,
                              time_ms=(time.perf_counter()-t0)*1000)
    except Exception as e:
        return StrategyResult("readability", False, error=str(e)[:200],
                              time_ms=(time.perf_counter()-t0)*1000)


# ── Strategy 4: Goose3 ────────────────────────────────────────────────────────

def _goose3(html: str, url: str) -> StrategyResult:
    t0 = time.perf_counter()
    try:
        g = Goose({"browser_user_agent": UA, "enable_image_fetching": False})
        try:
            article = g.extract(url=url, raw_html=html)
            text = clean_text(article.cleaned_text)
            wc = len(text.split()) if text else 0
            if not text or wc < 20:
                return StrategyResult("goose3", False,
                                      error="Insufficient content",
                                      time_ms=(time.perf_counter()-t0)*1000)
            return StrategyResult("goose3", True, title=article.title,
                                  text=text, word_count=wc,
                                  time_ms=(time.perf_counter()-t0)*1000)
        finally:
            try:
                g.close()   # Goose leaks a temp dir + fds per call otherwise
            except Exception:
                pass
    except Exception as e:
        return StrategyResult("goose3", False, error=str(e)[:200],
                              time_ms=(time.perf_counter()-t0)*1000)


# ── Strategy 5: BeautifulSoup ─────────────────────────────────────────────────

def _beautifulsoup(html: str) -> StrategyResult:
    t0 = time.perf_counter()
    try:
        soup = BeautifulSoup(html, "lxml")
        if IS_BLOCKED.search(soup.get_text()[:300]):
            return StrategyResult("beautifulsoup", False,
                                  error="Blocked by proxy",
                                  time_ms=(time.perf_counter()-t0)*1000)

        for tag in soup(NOISE_TAGS):
            tag.decompose()

        # Safe iteration — list() prevents iterator invalidation on decompose
        for el in list(soup.find_all(True)):
            if not isinstance(el, Tag) or el.parent is None:
                continue
            cls = " ".join(el.get("class") or [])
            eid = el.get("id") or ""
            if AD_PAT.search(cls) or AD_PAT.search(eid):
                el.decompose()

        h1 = soup.find("h1")
        title_tag = soup.find("title")
        title = (h1.get_text(strip=True) if h1 else None) or \
                (title_tag.get_text(strip=True) if title_tag else None)

        best_el, best_len = None, 0
        for c in soup.find_all(["article", "main", "section", "div"]):
            if not isinstance(c, Tag) or c.parent is None:
                continue
            t = c.get_text(separator=" ", strip=True)
            if len(t) > best_len:
                best_len = len(t)
                best_el = c

        content_el = best_el or soup.body
        if content_el is None:
            return StrategyResult("beautifulsoup", False,
                                  error="No content container",
                                  time_ms=(time.perf_counter()-t0)*1000)

        text = clean_text(content_el.get_text(separator="\n"))
        wc = len(text.split()) if text else 0
        if not text or wc < 20:
            return StrategyResult("beautifulsoup", False,
                                  error="Insufficient content",
                                  time_ms=(time.perf_counter()-t0)*1000)

        return StrategyResult("beautifulsoup", True, title=title,
                              text=text, word_count=wc,
                              time_ms=(time.perf_counter()-t0)*1000)
    except Exception as e:
        return StrategyResult("beautifulsoup", False, error=str(e)[:200],
                              time_ms=(time.perf_counter()-t0)*1000)


# ── Strategy 6: Playwright (fallback for JS-heavy pages) ─────────────────────

# Chromium launch args that reduce bot-detection surface
_PW_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-blink-features=AutomationControlled",
    "--disable-infobars",
    "--disable-extensions",
    "--disable-features=IsolateOrigins,site-per-process",
    "--window-size=1440,900",
]

# Realistic browser fingerprint headers
_PW_EXTRA_HEADERS = {
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

# Inline stealth JS — hides every known automation fingerprint
_STEALTH_JS = """
(function() {
    // 1. Hide webdriver flag
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

    // 2. Fake plugins (real Chrome has plugins; headless has none)
    Object.defineProperty(navigator, 'plugins', {
        get: () => {
            const arr = [
                { name:'Chrome PDF Plugin', filename:'internal-pdf-viewer', description:'Portable Document Format' },
                { name:'Chrome PDF Viewer', filename:'mhjfbmdgcfjbbpaeojofohoefgiehjai', description:'' },
                { name:'Native Client', filename:'internal-nacl-plugin', description:'' }
            ];
            arr.item = i => arr[i]; arr.namedItem = n => arr.find(p=>p.name===n) || null;
            arr.refresh = ()=>{};
            return arr;
        }
    });

    // 3. Real language lists
    Object.defineProperty(navigator, 'languages', { get: () => ['en-US','en'] });

    // 4. Fake chrome runtime (headless has no window.chrome by default)
    if (!window.chrome) {
        window.chrome = {
            app: { isInstalled: false, getDetails: ()=>{}, getIsInstalled: ()=>{}, runningState: ()=>'cannot_run' },
            runtime: { PlatformOs: {MAC:'mac',WIN:'win'}, PlatformArch: {ARM:'arm',X86_32:'x86-32',X86_64:'x86-64'},
                       PlatformNaclArch: {ARM:'arm',X86_32:'x86-32',X86_64:'x86-64'},
                       RequestUpdateCheckStatus: {THROTTLED:'throttled',NO_UPDATE:'no_update',UPDATE_AVAILABLE:'update_available'} },
            loadTimes: function() { return {
                commitLoadTime: Date.now()/1000 - Math.random()*2,
                connectionInfo:'h2', finishDocumentLoadTime:0, finishLoadTime:0,
                firstPaintAfterLoadTime:0, firstPaintTime:0, navigationType:'Other',
                npnNegotiatedProtocol:'h2', requestTime:Date.now()/1000 - Math.random()*3,
                startLoadTime:Date.now()/1000 - Math.random()*2, wasAlternateProtocolAvailable:false,
                wasFetchedViaSpdy:true, wasNpnNegotiated:true
            }},
            csi: function() { return { onloadT: Date.now(), pageT: Math.random()*5000, startE: Date.now()-3000, tran: 15 } }
        };
    }

    // 5. Correct permissions API
    const origQuery = window.navigator.permissions && window.navigator.permissions.query;
    if (origQuery) {
        window.navigator.permissions.query = params =>
            params.name === 'notifications'
                ? Promise.resolve({ state: Notification.permission })
                : origQuery(params);
    }

    // 6. Realistic screen dimensions
    Object.defineProperty(screen, 'availWidth', { get: () => 1440 });
    Object.defineProperty(screen, 'availHeight', { get: () => 900 });

    // 7. Prevent iframe detection
    Object.defineProperty(HTMLIFrameElement.prototype, 'contentWindow', {
        get: function() { return window; }
    });

    // 8. Realistic hardware concurrency and memory
    Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 });
    if ('deviceMemory' in navigator)
        Object.defineProperty(navigator, 'deviceMemory', { get: () => 8 });
})();
"""


def _launch_browser_stealth(p, headless: bool):
    """Launch Chromium with stealth settings. Applies playwright-stealth if available."""
    try:
        from playwright_stealth import Stealth
        Stealth().hook_playwright_context(p)
    except Exception:
        pass  # continue without stealth plugin if unavailable

    return p.chromium.launch(
        headless=headless,
        args=_PW_ARGS,
    )


def _playwright(url: str, wait_selector: Optional[str] = None,
                extra_wait: float = 4.0, block_resources: bool = True,
                capture_xhr: bool = True) -> StrategyResult:
    t0 = time.perf_counter()
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
        import json as _json

        captured_json: list[str] = []
        captured_xhr: list[dict] = []

        with sync_playwright() as p:
            browser = _launch_browser_stealth(p, cfg.playwright_headless)
            ctx = browser.new_context(
                user_agent=UA,
                viewport={"width": 1440, "height": 900},
                locale="en-US",
                timezone_id="Asia/Kolkata",
                extra_http_headers=_PW_EXTRA_HEADERS,
            )
            page = ctx.new_page()

            # Apply comprehensive stealth JS before any page loads
            page.add_init_script(_STEALTH_JS)

            def _on_response(resp):
                try:
                    ct = resp.headers.get("content-type", "")
                    u = resp.url
                    if resp.status == 200 and "json" in ct and any(
                        k in u.lower() for k in ["api","data","feed","market","stock","indices","quote","rate","price","finance"]
                    ) and "analytics" not in u and "gtm" not in u                     and not _NAV_URL_PATTERNS.search(u):
                        body = resp.body()
                        if body and 50 < len(body) < 500_000:
                            captured_json.append(body.decode("utf-8", errors="replace"))
                            captured_xhr.append({"url": u, "size": len(body)})
                except Exception:
                    pass

            if capture_xhr:
                page.on("response", _on_response)

            # Abort images/fonts/media/ad hosts. On news sites these are ~80%
            # of requests and the reason networkidle never settles — this alone
            # takes a typical render from ~14s to ~4s.
            if block_resources:
                def _route(route):
                    try:
                        rt = route.request.resource_type
                        if rt in ("image", "media", "font") or _AD_HOST.search(route.request.url):
                            return route.abort()
                        return route.continue_()
                    except Exception:
                        try:
                            route.continue_()
                        except Exception:
                            pass
                page.route("**/*", _route)

            try:
                resp = page.goto(url, wait_until="domcontentloaded",
                                 timeout=cfg.scraper_timeout * 1000)
            except PWTimeout:
                resp = None      # partial DOM is still worth extracting

            nav_status = resp.status if resp else 0
            wait_selector = wait_selector or _site_selector(url)

            # On 403/407: still continue — some sites return 403 on first load
            # then redirect to a challenge page. Check content before giving up.
            if nav_status in (407, 451):
                browser.close()
                return StrategyResult("playwright", False,
                                      error=f"HTTP {nav_status} — proxy/legal block",
                                      time_ms=(time.perf_counter()-t0)*1000)

            # Wait for dynamic content
            if wait_selector:
                try:
                    page.wait_for_selector(wait_selector, timeout=12000)
                except PWTimeout:
                    pass

            page.wait_for_timeout(int(extra_wait * 1000))
            try:
                page.wait_for_load_state("networkidle", timeout=8000)
            except PWTimeout:
                pass

            # Scroll to trigger lazy-loaded rows
            try:
                page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                page.wait_for_timeout(1000)
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(800)
                page.evaluate("window.scrollTo(0, 0)")
                page.wait_for_timeout(500)
            except Exception:
                pass

            # Wait a bit more for any deferred XHR after scroll
            page.wait_for_timeout(1500)

            html = page.content()
            pw_title = page.title()
            browser.close()

        if IS_BLOCKED.search(html[:400]):
            return StrategyResult("playwright", False,
                                  error="Blocked by network proxy",
                                  time_ms=(time.perf_counter()-t0)*1000)

        parts = []
        # Store rendered HTML so main.py can re-run table extraction / page detection on it
        extra = {"rendered_html": html}

        # XHR JSON data (best for financial/data pages)
        if captured_json:
            json_content = _format_xhr_json(captured_json)
            if json_content:
                extra["xhr_captured"] = len(captured_xhr)
                extra["xhr_urls"] = [x["url"][:80] for x in captured_xhr[:5]]
                parts.append("## Live API Data\n\n" + json_content)
            # If ALL captured JSON was nav/schema noise, don't count it as captured

        # Tables from rendered HTML
        from scraper.cleaner import extract_tables_markdown
        table_md, tc = extract_tables_markdown(html)
        if table_md and tc > 0:
            extra["tables_found"] = tc
            parts.append("## Table Data\n\n" + table_md)

        # Text via trafilatura on rendered HTML
        traf = trafilatura.extract(html, include_tables=True, favor_recall=True)
        if traf:
            traf_clean = clean_text(traf)
            if len(traf_clean.split()) > 20:
                parts.append("## Page Content\n\n" + traf_clean)

        if not parts:
            bs = _beautifulsoup(html)
            if bs.success and bs.text:
                parts.append(bs.text)

        if not parts:
            return StrategyResult("playwright", False,
                                  error="Rendered but no extractable content",
                                  time_ms=(time.perf_counter()-t0)*1000,
                                  extra=extra)

        combined = "\n\n---\n\n".join(parts)
        return StrategyResult("playwright", True,
                              title=pw_title or None,
                              text=combined,
                              word_count=len(combined.split()),
                              time_ms=(time.perf_counter()-t0)*1000,
                              extra=extra)

    except ImportError:
        return StrategyResult("playwright", False,
                              error="Not installed. Run: pip install playwright && playwright install chromium",
                              time_ms=(time.perf_counter()-t0)*1000)
    except Exception as e:
        return StrategyResult("playwright", False, error=str(e)[:300],
                              time_ms=(time.perf_counter()-t0)*1000)


# Column names that indicate a navigation/menu payload — not market data
_NAV_COL_NAMES = frozenset({
    "url", "shorturl", "longurl", "href", "link", "path", "slug", "uri",
    "navitem", "menuitem", "l1", "l2", "l3", "l3navmenuitem", "l2navmenuitem",
    "seotitle", "seo_title", "meta_title", "canonical", "permalink",
    "category_slug", "subcategory", "breadcrumb", "anchor", "target",
    "pagetype", "page_type_nav", "templatetype",
})

# URL path patterns that suggest navigation/config APIs (not data feeds)
_NAV_URL_PATTERNS = re.compile(
    r"/(nav|menu|sitemap|breadcrumb|sidebar|header|footer|widget|"
    r"config|setting|layout|template|taxonomy|category-tree|l[123]menu)",
    re.I,
)

# Type-name strings used in API schema descriptors
_TYPE_NAMES = frozenset({
    "string", "str", "int", "integer", "float", "double", "decimal",
    "bool", "boolean", "number", "object", "array", "null", "none",
    "date", "datetime", "timestamp", "text", "varchar", "bigint",
})


def _is_schema_row(rows: list) -> bool:
    """Return True if this list looks like schema/navigation/config data, not real market data."""
    if not rows or not isinstance(rows[0], dict):
        return False
    sample = rows[0]
    col_names = set(k.lower() for k in sample.keys())

    # Single-column tables are almost always metadata (e.g. {"length": 16990})
    if len(col_names) <= 1:
        return True

    # Classic schema shape: {name/field/column + type} columns
    has_type_col = "type" in col_names or "dtype" in col_names or "datatype" in col_names
    has_name_col = bool(col_names & {"name", "field", "column", "key", "attribute", "param"})
    if has_type_col and has_name_col:
        return True

    # Values that are all type-name strings
    vals = [str(v).lower().strip() for v in sample.values()]
    type_hits = sum(1 for v in vals if v in _TYPE_NAMES)
    if type_hits / max(len(vals), 1) > 0.5:
        return True

    # Navigation/menu payload: columns look like URL routing data
    nav_hits = col_names & _NAV_COL_NAMES
    if len(nav_hits) >= 2:
        return True

    # All values look like URL paths (start with /)
    url_path_vals = sum(1 for v in vals if str(v).startswith("/") or str(v).startswith("http"))
    if url_path_vals / max(len(vals), 1) > 0.5:
        return True

    return False


def _score_section(rows: list) -> float:
    """Score a list of row-dicts by data richness. Returns -1 to skip."""
    if not rows or not isinstance(rows[0], dict):
        return -1.0
    if _is_schema_row(rows):
        return -1.0

    sample = rows[0]
    score = 0.0

    # Row count (capped) — more rows = better
    score += min(len(rows), 200) * 1.5

    # Column count — more columns = richer
    score += len(sample) * 4.0

    # Numeric values (prices, changes) — key signal for market data
    numeric = sum(
        1 for v in sample.values()
        if isinstance(v, (int, float))
        or (isinstance(v, str)
            and v.strip().replace(".", "").replace("-", "").replace("+", "").replace(",", "").isdigit()
            and v.strip() not in ("", "0"))
    )
    score += numeric * 8.0

    # Known market/financial column names
    _MARKET_COLS = {
        "name", "symbol", "ticker", "ltp", "price", "lastprice", "last",
        "chg", "change", "net_change", "chgper", "percent_change", "pctchange",
        "open", "high", "low", "close", "prevclose", "prev_close",
        "volume", "vol", "marketstate", "market_state", "time", "updated_at",
    }
    col_lower = {k.lower() for k in sample.keys()}
    score += len(col_lower & _MARKET_COLS) * 12.0

    return score



def _extract_json_sections(data, depth: int = 0) -> list:
    """
    Recursively extract (section_title, list_of_row_dicts) from any JSON shape.
    Specifically tuned for Moneycontrol Global Indices.
    """
    if depth > 4:
        return []
    results = []

    if isinstance(data, list):
        if not data:
            return []
        if isinstance(data[0], dict):
            # Check if items themselves contain list values (category containers)
            cat_key = next(
                (k for item in data[:3] for k, v in item.items()
                 if isinstance(v, list) and v and isinstance(v[0], dict)),
                None,
            )
            if cat_key:
                for item in data:
                    # 🚨 WE FOUND THE SECRET WORD: "heading" 🚨
                    title = str(
                        item.get("heading") or item.get("category") or 
                        item.get("name") or item.get("section") or 
                        item.get("type") or ""
                    )
                    
                    if title == "header": 
                        continue # Skip useless metadata
                        
                    rows = item.get(cat_key, [])
                    if rows and isinstance(rows[0], dict):
                        results.append((title, rows))
            else:
                results.append(("", data))

    elif isinstance(data, dict):
        # 1. Try standard wrapper keys first (The X-ray showed 'data' and 'success')
        for key in ["data", "result", "results", "indices", "items", "records", "list", "response", "globalIndices", "payload"]:
            val = data.get(key)
            if val is None:
                continue
            sub = _extract_json_sections(val, depth + 1)
            if sub:
                return sub

        # 2. dict-of-lists pattern
        list_children = {
            k: v for k, v in data.items()
            if isinstance(v, list) and v and isinstance(v[0], dict)
        }
        if list_children:
            for category, rows in list_children.items():
                if category == "header":
                    continue 
                    
                has_nested = any(isinstance(vv, list) for item in rows[:3] for vv in item.values())
                if has_nested:                     
                    sub = _extract_json_sections(rows, depth + 1)                     
                    if sub:                         
                        for sub_title, sub_rows in sub:
                            final_title = sub_title if sub_title else category
                            if final_title != "header":
                                results.append((final_title, sub_rows))          
                else:                     
                    results.append((category, rows))
            if results:
                return results

        # 3. dict-of-dicts: recurse one level
        dict_results = []
        for k, v in data.items():
            if isinstance(v, dict):
                sub = _extract_json_sections(v, depth + 1)
                if sub:
                    for sub_title, sub_rows in sub:
                        if sub_title == "header":
                            continue
                            
                        # If inner key is generic, map it back to the parent 
                        if sub_title in ["dataList", "data", ""]:
                            dict_results.append((k, sub_rows))
                        else:
                            dict_results.append((f"{k}_{sub_title}", sub_rows))
                            
        if dict_results:
            results.extend(dict_results)
            return results

    return results

def _format_xhr_json(json_texts: list[str]) -> str:
    """
    Convert captured XHR JSON payloads to clean Markdown tables.
    Handles flat lists, dict-wrapped lists, and dict-of-lists (US/EUROPE/ASIA).
    Deduplicates by column fingerprint to avoid showing same data twice.
    """
    import json as _json
    out = []
    seen_fingerprints: set = set()

    # Parse all payloads and collect scored sections
    scored: list[tuple[float, str, list, int]] = []  # (score, title, rows, payload_id)

    for payload_id, raw in enumerate(json_texts[:30]):
        try:
            data = _json.loads(raw)
        except Exception:
            continue

        # Try force_extract_markets first (handles array-row format from Moneycontrol)
        extracted = force_extract_markets(data)
        if extracted:
            for section_title, rows in extracted:
                s = _score_section(rows)
                if s > 0:
                    scored.append((s, section_title, rows, payload_id))
        else:
            # Fallback to generic recursive extractor for dict/list JSON shapes
            for section_title, rows in _extract_json_sections(data):
                s = _score_section(rows)
                if s > 0:
                    scored.append((s, section_title, rows, payload_id))

    # Sort by score descending — richest data sections first
    scored.sort(key=lambda x: x[0], reverse=True)

    for score, section_title, rows, payload_id in scored:
        sample = rows[0]
        cols = [
            k for k, v in sample.items()
            if isinstance(v, (str, int, float, bool)) and len(str(v)) < 80
        ][:14]
        if not cols:
            continue

        # Deduplicate: same cols + same payload + same title = true duplicate → skip
        # same cols + same payload + different title = different region → keep (US/Europe/Asia)
        # same cols + different payload = same API called twice → skip
        fingerprint = (frozenset(cols), payload_id, section_title)
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)

        if section_title:
            out.append(f"### {section_title}")

        out.append("| " + " | ".join(cols) + " |")
        out.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for row in rows[:300]:
            vals = [str(row.get(c, "")) for c in cols]
            out.append("| " + " | ".join(vals) + " |")
        out.append("")

    return "\n".join(out)

def force_extract_markets(obj, results=None):
    if results is None:
        results = []

    if isinstance(obj, dict):
        # 🎯 HEAT-SEEKER: Find the market data
        if "heading" in obj and "data" in obj and isinstance(obj["data"], list):
            region = obj["heading"]
            raw_stocks = obj["data"]
            clean_stocks = []

            # Detect header row (first array row where values are column name strings)
            headers = None
            for row in raw_stocks:
                if isinstance(row, list):
                    # First list row = column headers (e.g. ['symbol','name','ltp',...])
                    if headers is None:
                        headers = [str(h).lower() for h in row]
                        continue
                    # Subsequent list rows = data rows — map using detected headers
                    if len(row) >= 2:
                        clean_stock = {
                            headers[i] if i < len(headers) else f"col{i}": row[i]
                            for i in range(min(len(row), len(headers) if headers else len(row)))
                        }
                        # Skip any row whose values look like the header itself
                        if clean_stock.get("name") in ("name", "symbol", "index", ""):
                            continue
                        clean_stocks.append(clean_stock)
                elif isinstance(row, dict):
                    # Already a dict row — skip if it's a schema descriptor
                    if row.get("name") in ("symbol", "name", ""):
                        continue
                    clean_stocks.append(row)

            # Save the cleanly mapped data!
            if len(clean_stocks) > 0:
                results.append((region, clean_stocks))

        # Recursively dig deeper into all dictionary values
        for value in obj.values():
            force_extract_markets(value, results)

    elif isinstance(obj, list):
        # Recursively dig into all lists
        for item in obj:
            force_extract_markets(item, results)

    return results

# ── Pick best result ──────────────────────────────────────────────────────────

# Trust multipliers: structured sources beat heuristic ones at equal length.
_STRATEGY_TRUST = {
    "jsonld": 1.40, "wp_rest": 1.35, "embedded_state": 1.25, "site_dom": 1.20,
    "trafilatura": 1.15, "newspaper3k": 1.10, "goose3": 1.05,
    "readability": 1.00, "playwright": 1.00, "reader_proxy": 0.95,
    "beautifulsoup": 0.70,
}


def _pick_best(results: list[StrategyResult]) -> Optional[StrategyResult]:
    successful = [r for r in results if r.success and r.text and len(r.text.split()) >= 20]
    if not successful:
        return None

    def score(r: StrategyResult) -> float:
        base = float(r.word_count)
        base *= 1.0 - 0.60 * _junk_ratio(r.text)          # punish link soup
        base *= _STRATEGY_TRUST.get(r.strategy.split("+")[0], 1.0)
        if r.strategy.startswith("playwright"):
            if r.extra.get("xhr_captured", 0) > 0:
                base *= 3.0
            if r.extra.get("tables_found", 0) > 0:
                base *= 1.5
        return base

    return max(successful, key=score)


# ── Metadata extractor ────────────────────────────────────────────────────────

def _metadata(html: str, url: str) -> dict:
    try:
        soup = BeautifulSoup(html, "lxml")
        meta: dict = {}
        for tag in soup.find_all("meta"):
            if not isinstance(tag, Tag):
                continue
            prop = tag.get("property") or tag.get("name", "")
            val = tag.get("content", "")
            if prop and val:
                meta[prop] = val
        parsed = urlparse(url)
        return {
            "domain": parsed.netloc,
            "og_title": meta.get("og:title"),
            "og_description": meta.get("og:description"),
            "og_image": meta.get("og:image"),
            "author": meta.get("author") or meta.get("article:author"),
            "published_time": meta.get("article:published_time"),
        }
    except Exception:
        return {"domain": urlparse(url).netloc}

def _extract_images(html: str, base_url: str) -> list[str]:
    """Extracts top images from HTML, returning them as Markdown links."""
    if not html:
        return []
    try:
        soup = BeautifulSoup(html, "lxml")
        images = []
        seen_urls = set()
        
        # 1. Grab the OpenGraph Hero Image first (highest quality)
        og = soup.find("meta", attrs={"property": "og:image"})
        if isinstance(og, Tag) and og.get("content"):
            img_url = urljoin(base_url, str(og["content"]))
            images.append(f"![Hero Image]({img_url})")
            seen_urls.add(img_url)

        # 2. Grab standard img tags
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src")
            if not src or src.startswith("data:image"):
                continue
            
            full_url = urljoin(base_url, src)
            
            # Filter out obvious UI junk/trackers
            if any(x in full_url.lower() for x in ['icon', 'logo', 'tracker', 'pixel', 'avatar']):
                continue
                
            # Skip tiny images if dimensions are provided in the tag
            width = img.get("width", "")
            if str(width).isdigit() and int(width) < 150:
                continue
                
            alt_text = img.get("alt", "Image").strip() or "Image"
                
            if full_url not in seen_urls:
                images.append(f"![{alt_text}]({full_url})")
                seen_urls.add(full_url)
                
            if len(images) >= 5:  # Limit to top 5 to save LLM context window
                break
                
        return images
    except Exception:
        return []


# ── Tier 1: cheap alternate representations ──────────────────────────────────

def _alt_sources(html: str, url: str) -> list[tuple[str, str]]:
    """Cheap alternate representations, ordered by likelihood. (url, tag)."""
    out: list[tuple[str, str]] = []
    p = urlparse(url)
    path = p.path.rstrip("/")

    # 1. Declared AMP page — most reliable, sites publish it for crawlers.
    try:
        soup = BeautifulSoup(html or "", "lxml")
        for link in soup.find_all("link"):
            if not isinstance(link, Tag):
                continue
            rel = link.get("rel") or []
            rel_s = " ".join(rel) if isinstance(rel, list) else str(rel)
            if "amphtml" in rel_s.lower() and link.get("href"):
                out.append((urljoin(url, str(link["href"])), "+amp"))
                break
    except Exception:
        pass

    # 2. Conventional AMP paths.
    if not out and path:
        out.append((f"{p.scheme}://{p.netloc}{path}/amp", "+amp"))
        out.append((f"{p.scheme}://{p.netloc}{path}?amp=1", "+amp"))

    # 3. Mobile host — often server-rendered where desktop is client-rendered.
    host = p.netloc
    if not host.startswith(("m.", "amp.")):
        bare = host[4:] if host.startswith("www.") else host
        out.append((f"{p.scheme}://m.{bare}{path}", "+mobile"))

    seen, uniq = set(), []
    for u, tag in out:
        if u not in seen and u != url:
            seen.add(u)
            uniq.append((u, tag))
    return uniq[:3]


async def _wp_rest(url: str) -> StrategyResult:
    """WordPress REST API by slug — full clean body for thewire.in, scroll.in."""
    t0 = time.perf_counter()
    try:
        p = urlparse(url)
        segs = [s for s in p.path.split("/") if s]
        if not segs:
            return StrategyResult("wp_rest", False, error="No slug",
                                  time_ms=(time.perf_counter() - t0) * 1000)
        slug = re.sub(r"\.html?$", "", segs[-1])
        api = (f"{p.scheme}://{p.netloc}/wp-json/wp/v2/posts"
               f"?slug={slug}&_fields=title,content,date,link")
        raw, status = await _fetch_once(api, UA, referer=url)
        if status != 200:
            return StrategyResult("wp_rest", False, error=f"HTTP {status}",
                                  time_ms=(time.perf_counter() - t0) * 1000)
        data = json.loads(raw)
        if not isinstance(data, list) or not data:
            return StrategyResult("wp_rest", False, error="No post for slug",
                                  time_ms=(time.perf_counter() - t0) * 1000)
        post = data[0]
        title = _html_to_text((post.get("title") or {}).get("rendered", "")) or None
        text = _html_to_text((post.get("content") or {}).get("rendered", ""))
        wc = len(text.split()) if text else 0
        if wc < 20:
            return StrategyResult("wp_rest", False, error="Insufficient content",
                                  time_ms=(time.perf_counter() - t0) * 1000)
        return StrategyResult("wp_rest", True, title=title, text=text, word_count=wc,
                              time_ms=(time.perf_counter() - t0) * 1000,
                              extra={"publish_date": post.get("date")})
    except Exception as e:
        return StrategyResult("wp_rest", False, error=str(e)[:200],
                              time_ms=(time.perf_counter() - t0) * 1000)


async def _reader_proxy(url: str) -> StrategyResult:
    """Opt-in external reader (r.jina.ai). Off by default — third-party egress."""
    t0 = time.perf_counter()
    try:
        raw, status = await _fetch_once(f"https://r.jina.ai/{url}", UA, timeout=25)
        if status != 200:
            return StrategyResult("reader_proxy", False, error=f"HTTP {status}",
                                  time_ms=(time.perf_counter() - t0) * 1000)
        text = clean_text(raw)
        wc = len(text.split()) if text else 0
        if wc < 20:
            return StrategyResult("reader_proxy", False, error="Insufficient content",
                                  time_ms=(time.perf_counter() - t0) * 1000)
        return StrategyResult("reader_proxy", True, text=text, word_count=wc,
                              title=None, time_ms=(time.perf_counter() - t0) * 1000)
    except Exception as e:
        return StrategyResult("reader_proxy", False, error=str(e)[:200],
                              time_ms=(time.perf_counter() - t0) * 1000)


# ── Tiered cascade ────────────────────────────────────────────────────────────

MIN_GOOD_WORDS = 180   # early-exit threshold for "this is a real article"

# Ordered by latency, cheapest first. Every entry runs on the SAME html.
_HTML_STRATEGIES: list[tuple[str, Callable[[str, str], StrategyResult]]] = [
    ("jsonld",         lambda h, u: _jsonld(h, u)),
    ("embedded_state", lambda h, u: _embedded_state(h, u)),
    ("site_dom",       lambda h, u: _site_dom(h, u)),
    ("trafilatura",    lambda h, u: _trafilatura(h, u)),
    ("readability",    lambda h, u: _readability(h)),
    ("newspaper3k",    lambda h, u: _newspaper3k(u, h)),
    ("goose3",         lambda h, u: _goose3(h, u)),
    ("beautifulsoup",  lambda h, u: _beautifulsoup(h)),
]


async def _run_battery(html: str, url: str, results: list, *,
                       min_words: int, tag: str = "") -> Optional[StrategyResult]:
    """Run every HTML strategy against one document; stop at first good hit.

    Each strategy is offloaded to a worker thread — lxml/goose/newspaper are
    CPU-bound C calls that would otherwise stall the FastAPI event loop.
    """
    if not html:
        return None
    for name, fn in _HTML_STRATEGIES:
        try:
            r = await asyncio.to_thread(fn, html, url)
        except Exception as e:
            r = StrategyResult(name, False, error=str(e)[:200])
        if tag:
            r.strategy = f"{r.strategy}{tag}"
        results.append(r)
        if _ok(r, min_words):
            return r
    return None


def _page_type(url: str, html: str, table_count: int) -> str:
    p = urlparse(url)
    if p.path.strip("/") in ("", "index.html", "index.htm"):
        return "homepage"
    if table_count >= 2:
        return "data"
    return "article"



# ── Cascade run object ────────────────────────────────────────────────────────
#
# The tiers used to be inlined in scrape(). They are split out so the cheap
# tiers (0/1) can be run to completion and returned to the caller while the
# expensive tiers (2/3) continue in the background — see scrape_fast().


@dataclass
class _CascadeRun:
    """Mutable state carried across cascade tiers for one URL."""
    url: str
    t0: float
    html: str = ""
    best_html: str = ""
    fetch_ms: float = 0.0
    results: list = field(default_factory=list)
    hit: Optional[StrategyResult] = None
    deadline_hit: bool = False
    heavy_done: bool = False
    # Resume markers, so a deadline-interrupted cheap phase can be continued
    # without re-fetching Tier 0 or retrying alternates that already failed.
    tier0_done: bool = False
    alt_index: int = 0
    wp_done: bool = False

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0


# Bounds how many background Playwright escalations can run at once. Ten
# concurrent Chromium renders will thrash the box, so deferred work queues.
_HEAVY_SEM: Optional[asyncio.Semaphore] = None


def _heavy_sem() -> asyncio.Semaphore:
    global _HEAVY_SEM
    if _HEAVY_SEM is None:
        _HEAVY_SEM = asyncio.Semaphore(getattr(cfg, "scrape_max_concurrent_heavy", 3))
    return _HEAVY_SEM


async def _cheap_tiers(run: _CascadeRun, *, min_words: int,
                       deadline: Optional[float] = None) -> None:
    """Tiers 0 and 1: one static fetch + eight extractors, then AMP/mobile/WP.

    Worst case ~1-3s (three extra GETs). `deadline` (seconds) stops further
    cheap escalation once exceeded, so a slow site can be handed off to the
    background instead of blocking the caller.
    """
    # ── Tier 0: one static fetch, eight extractors
    if not run.tier0_done:
        run.tier0_done = True
        try:
            run.html, run.fetch_ms = await fetch_html(run.url, allow_error_body=True)
            run.best_html = run.html
            run.hit = await _run_battery(run.html, run.url, run.results,
                                         min_words=min_words)
        except Exception as e:
            run.html = ""
            run.results.append(StrategyResult("fetch", False, error=str(e)[:200]))

    # ── Tier 1: AMP / mobile host — one cheap GET each
    if run.hit is None:
        alts = _alt_sources(run.html, run.url)
        while run.alt_index < len(alts):
            if deadline is not None and run.elapsed > deadline:
                run.deadline_hit = True
                return
            alt_url, tag = alts[run.alt_index]
            run.alt_index += 1
            try:
                alt_html, _ = await fetch_html(alt_url, referer=run.url)
            except Exception as e:
                run.results.append(StrategyResult(f"fetch{tag}", False, error=str(e)[:120]))
                continue
            run.best_html = run.best_html or alt_html
            run.hit = await _run_battery(alt_html, run.url, run.results,
                                         min_words=min_words, tag=tag)
            if run.hit:
                return

    # ── Tier 1b: WordPress REST API
    if run.hit is None and not run.wp_done:
        if deadline is not None and run.elapsed > deadline:
            run.deadline_hit = True
            return
        run.wp_done = True
        r = await _wp_rest(run.url)
        run.results.append(r)
        if _ok(r, min_words):
            run.hit = r


async def _heavy_tiers(run: _CascadeRun, *, min_words: int,
                       allow_playwright: bool = True, want_data: bool = False,
                       allow_external: bool = False) -> None:
    """Tiers 2 and 3: Playwright light -> Playwright full -> external reader.

    This is the 4-12s tail. Kept separate so it can be deferred.
    """
    # ── Tier 2a: Playwright, light — assets blocked, no XHR capture
    if allow_playwright and (run.hit is None or want_data):
        pw = await asyncio.to_thread(_playwright, run.url, None, 2.0, True, False)
        run.results.append(pw)
        rendered = pw.extra.get("rendered_html")
        if rendered:
            run.best_html = rendered
            run.hit = await _run_battery(rendered, run.url, run.results,
                                         min_words=min_words, tag="+pw") or run.hit
        if run.hit is None and _ok(pw, min_words):
            run.hit = pw

    # ── Tier 2b: Playwright, full — assets on, XHR/table capture
    if allow_playwright and (run.hit is None or want_data):
        pw2 = await asyncio.to_thread(_playwright, run.url, None, 5.0, False, True)
        run.results.append(pw2)
        rendered = pw2.extra.get("rendered_html")
        if rendered:
            run.best_html = rendered
            run.hit = await _run_battery(rendered, run.url, run.results,
                                         min_words=min_words, tag="+pw2") or run.hit
        if run.hit is None and _ok(pw2, min_words):
            run.hit = pw2

    # ── Tier 3: external reader, opt-in
    if run.hit is None and allow_external:
        run.results.append(await _reader_proxy(run.url))

    run.heavy_done = True


def _assemble(run: _CascadeRun, *, partial: bool = False) -> ScrapeResult:
    """Build a ScrapeResult from whatever the cascade has gathered so far."""
    best = _pick_best(run.results)
    tables_md, tc = None, 0
    images = []
    if run.best_html:
        try:
            from scraper.cleaner import extract_tables_markdown
            tables_md, tc = extract_tables_markdown(run.best_html)
        except Exception:
            pass

        images = _extract_images(run.best_html, run.url)

    res = ScrapeResult(
        url=run.url,
        page_type=_page_type(run.url, run.best_html, tc),
        best_strategy=best.strategy if best else None,
        title=(best.title if best and best.title else _title_from_html(run.best_html)),
        content=best.text if best else None,
        word_count=best.word_count if best else 0,
        tables_md=tables_md or None,
        table_count=tc,
        fetch_time_ms=run.fetch_ms,
        total_time_ms=run.elapsed * 1000,
        all_results=run.results,
        metadata=(_metadata(run.best_html, run.url) if run.best_html
                  else {"domain": urlparse(run.url).netloc}),
        headlines=[],
        images=images,
    )
    # Flags for callers that need to distinguish "done" from "still working".
    # Kept in metadata so the ScrapeResult dataclass shape is unchanged.
    res.metadata["partial"] = bool(partial)
    res.metadata["deadline_hit"] = run.deadline_hit
    return res


async def scrape(url: str, *, min_words: int = MIN_GOOD_WORDS,
                 allow_playwright: bool = True, want_data: bool = False,
                 allow_external: bool = False) -> ScrapeResult:
    """Full cascade: static battery -> cheap alternates -> Playwright -> external.

    Escalates only while the result is still thin, so a well-behaved article
    page costs one fetch and ~200ms while ndtv.com/thewire.in still get the
    heavy path when they need it.

    Behaviour is unchanged from before the tier split — this waits for
    everything. Use scrape_fast() if you want the slow tail backgrounded.
    """
    run = _CascadeRun(url=url, t0=time.perf_counter())
    await _cheap_tiers(run, min_words=min_words)
    if run.hit is None or want_data:
        await _heavy_tiers(run, min_words=min_words,
                           allow_playwright=allow_playwright,
                           want_data=want_data, allow_external=allow_external)
    return _assemble(run)


async def scrape_fast(url: str, *, min_words: int = MIN_GOOD_WORDS,
                      deadline: Optional[float] = None,
                      allow_playwright: bool = True, want_data: bool = False,
                      allow_external: bool = False):
    """Run the cheap tiers now; hand back the slow tail for the caller to defer.

    Returns (result, finish) where:
      - `result` is a real ScrapeResult from tiers 0/1. If it is good enough,
        `finish` is None and there is nothing left to do.
      - `finish` is a zero-arg coroutine function that completes tiers 2/3 and
        returns the FULL ScrapeResult. The caller decides whether to await it
        inline, schedule it in the background, or drop it entirely.

    The split is at the tier boundary rather than a raw timeout because that
    is where the latency actually lives: tiers 0/1 are sub-second to ~3s,
    tier 2 is 4-12s. Cancelling mid-tier would just throw away work.
    """
    if deadline is None:
        deadline = float(getattr(cfg, "scrape_fast_deadline", 2.5))

    run = _CascadeRun(url=url, t0=time.perf_counter())
    await _cheap_tiers(run, min_words=min_words, deadline=deadline)

    if run.hit is not None and not want_data:
        return _assemble(run), None

    async def finish() -> ScrapeResult:
        # Semaphore is acquired inside the deferred coroutine, not here, so
        # queueing for a browser slot never blocks the fast path.
        async with _heavy_sem():
            if run.deadline_hit:
                # Cheap tiers were cut short by the deadline — resume them
                # with no deadline before paying for a browser.
                await _cheap_tiers(run, min_words=min_words)
                run.deadline_hit = False
            if run.hit is None or want_data:
                await _heavy_tiers(run, min_words=min_words,
                                   allow_playwright=allow_playwright,
                                   want_data=want_data,
                                   allow_external=allow_external)
        return _assemble(run)

    return _assemble(run, partial=True), finish
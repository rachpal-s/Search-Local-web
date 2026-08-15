"""agents/lib/wordcloud_lite.py — Pure-Python word-cloud renderer.

Sibling to agents/lib/mermaid_lite.py, same shape: parse input, lay out
shapes on a canvas, emit standalone SVG. No numpy, no Pillow, no PIL font
metrics — stdlib only (re, math, collections). Word-width is estimated
rather than measured, so placement uses generous padding to stay safe
against real glyph-width variance rather than risking visual overlap.

Palette matches the app's own design tokens (chat.css :root), same
reasoning as mermaid_lite: a diagram dropped into the transcript should
read as part of the product, not a differently-themed insert.
"""
import math
import re
from collections import Counter
from dataclasses import dataclass
from html import escape
from typing import List, Tuple

FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"

# Rank-tiered palette — darkest/boldest for the most frequent words, fading
# for the tail. Deliberately not a random-per-word rainbow (the classic
# word-cloud look): a restrained gradient reads as a deck asset, not a toy.
TIER_COLORS = [
    "#1d4ed8",  # --accent-ink   — top word
    "#2563eb",  # --accent       — next few
    "#3b6fd8",  # blend
    "#475569",  # --ink-2
    "#7c8da5",  # --ink-3
]
BG = "#ffffff"

# Compact but broad English stopword list — enough to strip function words
# from ordinary prose without a dependency. Not linguistically exhaustive;
# domain-specific noise words are the caller's problem, not this module's.
STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over
own same shan't she she'd she'll she's should shouldn't so some such than
that that's the their theirs them themselves then there there's these they
they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves
also like just get got one two three said says say new using use used via
etc within across per without upon towards among
""".split())

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{1,}[A-Za-z]|[A-Za-z]{2,}")


@dataclass
class PlacedWord:
    text: str
    x: float          # center x
    y: float          # center y
    font_size: float
    color: str


def count_words(text: str, min_len: int = 3, max_words: int = 60) -> List[Tuple[str, int, str]]:
    """Tokenizes, strips stopwords, and returns (display_text, count, lower_key)
    sorted by frequency descending. Preserves the most common ORIGINAL casing
    per word (e.g. "AI" stays "AI") rather than flattening everything to
    lowercase, which reads noticeably more polished for acronyms/proper nouns.
    """
    tokens = _WORD_RE.findall(text)
    counts: Counter = Counter()
    casings: dict = {}
    for tok in tokens:
        low = tok.lower()
        if low in STOPWORDS or len(low) < min_len:
            continue
        counts[low] += 1
        casings.setdefault(low, Counter())[tok] += 1

    ranked = counts.most_common(max_words)
    return [(casings[low].most_common(1)[0][0], n, low) for low, n in ranked]


def _bbox(w: PlacedWord, pad: float):
    # Width estimate: ~0.58em average glyph width for Inter at normal weight,
    # generously padded since this is an estimate, not a measurement — a
    # slightly-too-wide guess costs a little whitespace; a too-narrow one
    # costs visual overlap, which is the worse failure by far.
    half_w = (0.58 * w.font_size * len(w.text)) / 2 + pad
    half_h = (w.font_size * 0.62) + pad
    return (w.x - half_w, w.y - half_h, w.x + half_w, w.y + half_h)


def _overlaps(a, b) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_words(sized_words, canvas_w: float, canvas_h: float, pad: float = 2.5) -> List[PlacedWord]:
    """Greedy Archimedean-spiral placement: largest word first, at center;
    each subsequent word spirals outward from center until it finds a gap
    that doesn't collide with anything already placed. Same family of
    algorithm the common `wordcloud` PyPI package uses, minus font-metric
    precision — traded for zero dependencies.

    A word that can't find a gap at its target size shrinks and retries
    rather than getting dropped — matching how reference implementations
    handle crowding, and it reads as "the layout tightened up" rather than
    "a word silently vanished," which matters more for a deck asset.
    """
    placed: List[PlacedWord] = []
    cx, cy = canvas_w / 2, canvas_h / 2
    margin = 6.0
    flatten = canvas_h / canvas_w  # aspect-correct the spiral to the canvas rect
    MIN_FLOOR = 10.0  # below this a word is genuinely too small to bother placing

    def _try_place(text, font_size, color):
        angle, radius = 0.0, 0.0
        step_angle, step_radius = 0.18, 0.35
        for _ in range(6000):
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle) * flatten
            candidate = PlacedWord(text, x, y, font_size, color)
            box = _bbox(candidate, pad)
            in_bounds = (box[0] >= margin and box[1] >= margin and
                        box[2] <= canvas_w - margin and box[3] <= canvas_h - margin)
            if in_bounds and not any(_overlaps(box, _bbox(p, pad)) for p in placed):
                return candidate
            angle += step_angle
            radius += step_radius
        return None

    for text, font_size, color in sized_words:
        size = font_size
        result = None
        for _ in range(5):  # up to 5 shrink attempts before giving up
            result = _try_place(text, size, color)
            if result is not None:
                break
            size *= 0.82
            if size < MIN_FLOOR:
                break
        if result is not None:
            placed.append(result)
        # else: didn't fit even at the size floor — dropped rather than
        # overlapped. Only happens with very high max_words on dense input.

    return placed


def render_wordcloud_to_svg(text: str, width: int = 900, height: int = 560,
                            max_words: int = 55, min_word_len: int = 3) -> str:
    """Parses free text into a word-frequency cloud and returns a standalone SVG."""
    ranked = count_words(text, min_len=min_word_len, max_words=max_words)
    if len(ranked) < 5:
        raise ValueError(
            f"Only {len(ranked)} distinct word(s) found after removing stopwords — "
            f"need at least 5 for a meaningful cloud. Provide more running text."
        )

    counts = [n for _, n, _ in ranked]
    max_count, min_count = max(counts), min(counts)
    spread = max(max_count - min_count, 1)

    MIN_SIZE, MAX_SIZE = 15.0, 68.0
    sized_words = []
    for i, (display, n, _low) in enumerate(ranked):
        # sqrt scaling: word AREA (not linear size) tracks frequency, which
        # is what actually reads as "proportional" to the eye — linear
        # font-size scaling makes high-frequency words look disproportionate.
        norm = math.sqrt((n - min_count) / spread) if spread else 1.0
        font_size = MIN_SIZE + (MAX_SIZE - MIN_SIZE) * norm
        tier = min(i, len(TIER_COLORS) - 1) if i < 4 else (
            3 if i < len(ranked) * 0.7 else 4
        )
        color = TIER_COLORS[0] if i == 0 else TIER_COLORS[1] if i < 4 else (
            TIER_COLORS[3] if i < len(ranked) * 0.65 else TIER_COLORS[4]
        )
        sized_words.append((display, round(font_size, 1), color))

    # Largest words first so the greedy spiral claims center space for them.
    sized_words.sort(key=lambda t: -t[1])
    placed = _place_words(sized_words, width, height)

    text_svg = "".join(
        f'<text x="{p.x:.1f}" y="{p.y:.1f}" font-size="{p.font_size:.1f}" '
        f'font-family="{FONT_FAMILY}" font-weight="{700 if p.font_size > 40 else 600 if p.font_size > 24 else 500}" '
        f'fill="{p.color}" text-anchor="middle" dominant-baseline="middle">'
        f'{escape(p.text)}</text>'
        for p in placed
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{BG}"/>'
        f"{text_svg}"
        f"</svg>"
    )

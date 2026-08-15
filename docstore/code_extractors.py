"""docstore/code_extractors.py — the source-code track for I1.

This is a separate module rather than more classes in extractors.py because of
one specific hazard. `score_text()` there penalises a low alphanumeric ratio by
0.35 and a short file by another 0.20. Source code is roughly a third
punctuation, so nearly every .py/.js/.css file scores under the quality floor,
gets a `low_extraction_quality` reason code, and is quarantined. Silently. The
corpus would report itself healthy and contain no code at all.

So code gets a scorer with the inverted intuition: punctuation density is
evidence the extraction worked, and the failure modes worth catching are
minification, binary content mislabelled as text, and vendored dependencies.

Output is markdown with a fenced block, because the chunker, the enricher and
the embedder downstream all already speak markdown. The heading and the symbol
comment survive chunking, so a chunk lifted from the middle of a 900-line file
still says which file it came from and what it defines.
"""
from __future__ import annotations

import re
from pathlib import Path

from docstore.extractors import Extracted, ExtractError, Extractor

# ------------------------------------------------------------------ languages

_LANG = {
    ".py": "python", ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".jsx": "jsx", ".ts": "typescript", ".tsx": "tsx", ".css": "css",
    ".scss": "scss", ".sql": "sql", ".sh": "bash", ".yaml": "yaml",
    ".yml": "yaml", ".json": "json", ".toml": "toml",
}

# Ingesting node_modules is how a 2,000-document corpus becomes 400,000 and the
# retrieval quality collapses under vendored code nobody asked about.
_NOISE_PARTS = frozenset({
    "node_modules", "site-packages", "dist", "build", ".git", ".venv", "venv",
    "__pycache__", ".mypy_cache", ".pytest_cache", "vendor", "bower_components",
    ".next", ".nuxt", "coverage", "target", ".tox", ".eggs",
})
_NOISE_NAMES = re.compile(
    r"(package-lock\.json|yarn\.lock|poetry\.lock|pnpm-lock\.yaml|"
    r".*\.min\.(js|css)|.*\.bundle\.(js|css)|.*\.map)$", re.I)

# ------------------------------------------------------------------ symbols

_SYMBOL_PATTERNS: dict[str, list[re.Pattern]] = {
    "python": [
        re.compile(r"^\s*(?:async\s+)?def\s+(\w+)", re.M),
        re.compile(r"^\s*class\s+(\w+)", re.M),
    ],
    "javascript": [
        re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)", re.M),
        re.compile(r"^\s*(?:export\s+)?class\s+(\w+)", re.M),
        re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?(?:\(|function)", re.M),
    ],
    "css": [re.compile(r"^\s*([.#@][\w-]+[^{]*)\{", re.M)],
    "sql": [re.compile(
        r"(?:CREATE|ALTER)\s+(?:TABLE|VIEW|PROCEDURE|FUNCTION)\s+(?:IF\s+NOT\s+EXISTS\s+)?([\w.]+)",
        re.I)],
}
_SYMBOL_PATTERNS["jsx"] = _SYMBOL_PATTERNS["javascript"]
_SYMBOL_PATTERNS["typescript"] = _SYMBOL_PATTERNS["javascript"] + [
    re.compile(r"^\s*(?:export\s+)?(?:interface|type|enum)\s+(\w+)", re.M)]
_SYMBOL_PATTERNS["tsx"] = _SYMBOL_PATTERNS["typescript"]
_SYMBOL_PATTERNS["scss"] = _SYMBOL_PATTERNS["css"]


def extract_symbols(text: str, lang: str, limit: int = 60) -> list[str]:
    out: list[str] = []
    for pat in _SYMBOL_PATTERNS.get(lang, []):
        for m in pat.finditer(text):
            name = m.group(1).strip()
            if name and name not in out:
                out.append(name)
            if len(out) >= limit:
                return out
    return out


# ------------------------------------------------------------------ quality

def score_code(text: str, lang: str) -> tuple[float, list[str]]:
    """Quality heuristic for source. Deliberately NOT extractors.score_text().

    Punctuation density is expected and never penalised. What is penalised:
      * minified or generated  — very long average or maximum line
      * binary mislabelled     — null bytes or replacement characters
      * empty or trivial       — no symbols and almost no lines
    """
    warn: list[str] = []
    n = len(text)
    if n == 0:
        return 0.0, ["empty_extraction"]
    if "\x00" in text:
        return 0.0, ["binary_content_in_text_file"]

    lines = text.splitlines() or [""]
    non_blank = [ln for ln in lines if ln.strip()]
    if not non_blank:
        return 0.0, ["whitespace_only"]

    avg_line = sum(len(ln) for ln in non_blank) / len(non_blank)
    max_line = max(len(ln) for ln in non_blank)

    score = 1.0
    if avg_line > 250 or max_line > 5000:
        warn.append("likely_minified_or_generated")
        score -= 0.55
    if text.count("\ufffd") / n > 0.002:
        warn.append("replacement_chars_present")
        score -= 0.35
    if len(non_blank) < 3 and not extract_symbols(text, lang):
        warn.append("trivial_file")
        score -= 0.25
    if len(non_blank) > 3 and all(
            ln.lstrip().startswith(("#", "//", "/*", "*", "--")) for ln in non_blank):
        warn.append("comment_only_file")

    return max(0.0, round(score, 3)), warn


def is_noise_path(path: Path) -> bool:
    if any(part in _NOISE_PARTS for part in path.parts):
        return True
    return bool(_NOISE_NAMES.match(path.name))


# ------------------------------------------------------------------ extractors

class CodeExtractor(Extractor):
    """Reads a source file verbatim and wraps it as annotated markdown.

        # routes.py
        <!-- code:python symbols=create_job,list_jobs -->

        ```python
        ...source...
        ```
    """
    name, version = "native_code", "1.0"
    suffixes = tuple(_LANG)
    bound = "cpu"

    def extract(self, path: Path) -> Extracted:
        if is_noise_path(path):
            raise ExtractError("vendored_or_generated_code",
                               f"{path} matches the vendor/build exclusion list")
        try:
            raw = path.read_bytes()
        except Exception as e:  # noqa: BLE001
            raise ExtractError("unreadable_file", str(e)) from e

        if b"\x00" in raw[:8192]:      # cheap binary sniff before decoding
            raise ExtractError("binary_content_in_text_file", str(path))

        source = raw.decode("utf-8", errors="replace")
        lang = _LANG.get(path.suffix.lower(), "text")
        symbols = extract_symbols(source, lang)
        quality, warnings = score_code(source, lang)

        if quality < 0.2:
            raise ExtractError("low_code_quality", f"quality={quality} {warnings}")

        text = (f"# {path.name}\n"
                f"<!-- code:{lang} symbols={','.join(symbols) or 'none'} -->\n"
                f"\n```{lang}\n{source}\n```\n")

        if symbols:
            warnings = sorted(set(warnings + [f"symbols={len(symbols)}"]))

        return Extracted(text=text, extractor=f"native_code/{lang}",
                         extractor_version=self.version, page_count=None,
                         quality=quality, warnings=warnings)


class HtmlSourceExtractor(Extractor):
    """HTML treated as source rather than as an article.

    The existing HtmlExtractor runs trafilatura, which strips <script> and
    <style> and keeps the readable body. Right for a scraped news page,
    destructive for an index.html in a repository where the markup and the
    inline script are exactly what you want to retrieve on.
    """
    name, version = "html_source", "1.0"
    suffixes = (".html", ".htm", ".xhtml")
    bound = "cpu"

    def extract(self, path: Path) -> Extracted:
        raw = path.read_text(encoding="utf-8", errors="replace")
        quality, warnings = score_code(raw, "html")
        warnings = sorted(set(warnings + ["untrusted_html_source"]))
        return Extracted(f"# {path.name}\n<!-- code:html -->\n\n```html\n{raw}\n```\n",
                         self.name, self.version, quality=quality, warnings=warnings)


class HtmlModeRouter(Extractor):
    """Decides whether an .html file is an article or a source file.

    In order:
      1. lives under static/ templates/ src/ public/ ...      -> source
      2. sits next to .js/.css/.py                            -> source
      3. contains Jinja/Vue/Handlebars template syntax        -> source
      4. markup-to-text ratio above threshold (tag soup)      -> source
      5. otherwise                                            -> article

    The decision is recorded in warnings as `html_mode:<reason>` so a wrong call
    shows up in the document's warnings rather than as an invisible difference
    in what got indexed.
    """
    name, version = "html_router", "1.0"
    suffixes = (".html", ".htm", ".xhtml")
    bound = "cpu"

    _SOURCE_DIRS = frozenset({"static", "templates", "src", "public", "assets",
                              "components", "views", "layouts", "partials"})
    _TEMPLATE_MARKERS = ("{%", "{{", "v-for", "ng-repeat", "<template>")

    def __init__(self, prose_extractor: Extractor | None = None):
        from docstore.extractors import HtmlExtractor
        self._prose = prose_extractor or HtmlExtractor()
        self._source = HtmlSourceExtractor()

    def _looks_like_source(self, path: Path, raw: str) -> tuple[bool, str]:
        if any(part.lower() in self._SOURCE_DIRS for part in path.parts):
            return True, "path_in_source_dir"
        try:
            siblings = {p.suffix.lower() for p in path.parent.glob("*") if p.is_file()}
        except OSError:
            siblings = set()
        if siblings & {".js", ".css", ".py", ".ts", ".jsx"}:
            return True, "colocated_with_code"

        head = raw[:20_000]
        if any(m in head for m in self._TEMPLATE_MARKERS):
            return True, "template_syntax_present"

        tag_chars = sum(len(m) for m in re.findall(r"<[^>]{1,400}>", head))
        if head and tag_chars / len(head) > 0.55:
            return True, "high_markup_ratio"
        return False, "prose_article"

    def extract(self, path: Path) -> Extracted:
        raw = path.read_text(encoding="utf-8", errors="replace")
        as_source, reason = self._looks_like_source(path, raw)
        result = (self._source if as_source else self._prose).extract(path)
        return Extracted(
            text=result.text, extractor=result.extractor,
            extractor_version=result.extractor_version,
            page_count=result.page_count, quality=result.quality,
            warnings=sorted(set(result.warnings + [f"html_mode:{reason}"])),
        )

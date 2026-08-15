"""docstore/code_chunker.py — I3 for the code track.

docstore/chunker.py splits on markdown headings and sentence boundaries
(`(?<=[.!?])\\s+`). Run a Python module through it and chunks begin three lines
into a function body and end mid-expression, with an empty section breadcrumb.
Retrieval over that returns fragments nobody can act on, and — worse — they look
plausible enough to cite.

Code has its own structural grammar, so this splits on that instead:

    python      top-level def / class, dedent-terminated
    javascript  function / class / arrow-const at brace depth 0
    css         complete rule blocks, at-rules kept with their body
    fallback    blank-line-delimited paragraphs

Each chunk carries the file name and the symbol as its `section_path`, so a
citation reads `gateway.py > refund_transaction` rather than `chunk 47`. Import
lines are hoisted into a capped preamble prepended to later chunks, because a
code chunk without its imports is missing the type information that makes it
interpretable.

Chunk IDs come from the same `chunk_id_for()` the prose track uses, so
idempotency and the "same content, same id" property are preserved.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from docstore.corpus import ChunkRecord, chunk_id_for

# ------------------------------------------------------------------ config

@dataclass(frozen=True)
class CodeChunkerConfig:
    target_chars: int = 1_600      # code is denser than prose; wider windows help
    max_chars: int = 3_200
    min_chars: int = 120
    preamble_max_chars: int = 400
    include_preamble: bool = True

    @property
    def config_hash(self) -> str:
        blob = (f"code{self.target_chars}{self.max_chars}{self.min_chars}"
                f"{self.preamble_max_chars}{self.include_preamble}")
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


DEFAULT_CODE_CONFIG = CodeChunkerConfig()

# ------------------------------------------------------------------ parsing

_FENCE = re.compile(r"^```(\w+)?\s*$")
_META = re.compile(r"<!--\s*code:(\w+)\s*(?:symbols=([^>]*?))?\s*-->")

_PY_TOP = re.compile(r"^(?:@\w|(?:async\s+)?def\s+\w|class\s+\w)")
_JS_TOP = re.compile(
    r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?"
    r"(?:function\s+\w+|class\s+\w+|const\s+\w+\s*=\s*(?:async\s*)?[\(<]|"
    r"(?:interface|type|enum)\s+\w+)")
_PY_NAME = re.compile(r"(?:async\s+)?(?:def|class)\s+(\w+)")
_JS_NAME = re.compile(r"(?:function|class|const|let|var|interface|type|enum)\s+(\w+)")
_CSS_SELECTOR = re.compile(r"^\s*([^{}/][^{]*)\{")
_IMPORT = re.compile(
    r"^\s*(?:import\s|from\s+[\w.]+\s+import\s|const\s+\{?[\w,\s}]+\}?\s*=\s*require\(|"
    r"@import\s|@use\s|#include\s)")


def unwrap_fence(text: str) -> tuple[str, str, list[str]]:
    """Strip the CodeExtractor wrapper. Returns (source, lang, symbols)."""
    lang, symbols = "text", []
    m = _META.search(text)
    if m:
        lang = m.group(1) or "text"
        symbols = [s for s in (m.group(2) or "").split(",") if s and s != "none"]

    lines = text.splitlines()
    start = end = None
    for i, line in enumerate(lines):
        fm = _FENCE.match(line)
        if fm and start is None:
            start = i
            if fm.group(1):
                lang = fm.group(1)
        elif fm and start is not None:
            end = i
            break
    if start is None:
        return text, lang, symbols
    return "\n".join(lines[start + 1: end if end is not None else len(lines)]), lang, symbols


def _preamble(source: str, cfg: CodeChunkerConfig) -> str:
    if not cfg.include_preamble:
        return ""
    picked: list[str] = []
    total = 0
    for line in source.splitlines()[:120]:
        if _IMPORT.match(line):
            if total + len(line) > cfg.preamble_max_chars:
                break
            picked.append(line.strip())
            total += len(line)
    return "\n".join(picked)


def _split_python(source: str) -> list[tuple[str, str]]:
    lines = source.splitlines()
    blocks: list[tuple[str, str]] = []
    current: list[str] = []
    symbol = "module"
    for line in lines:
        is_top = bool(line) and not line[0].isspace() and _PY_TOP.match(line)
        if is_top and current and any(c.strip() for c in current):
            blocks.append((symbol, "\n".join(current)))
            current = []
        if is_top:
            nm = _PY_NAME.search(line)
            if nm:
                symbol = nm.group(1)
        current.append(line)
    if current and any(c.strip() for c in current):
        blocks.append((symbol, "\n".join(current)))
    return blocks


def _split_braced(source: str) -> list[tuple[str, str]]:
    """JS/TS: split at depth-0 declarations, with crude string/comment masking."""
    blocks: list[tuple[str, str]] = []
    current: list[str] = []
    symbol = "module"
    depth = 0
    for line in source.splitlines():
        stripped_line = line.strip()
        if depth == 0 and _JS_TOP.match(stripped_line) and current and any(c.strip() for c in current):
            blocks.append((symbol, "\n".join(current)))
            current = []
        if depth == 0:
            nm = _JS_NAME.search(line)
            if nm and _JS_TOP.match(stripped_line):
                symbol = nm.group(1)
        current.append(line)
        masked = re.sub(r"""(["'`]).*?\1|//.*$""", "", line)
        depth = max(depth + masked.count("{") - masked.count("}"), 0)
    if current and any(c.strip() for c in current):
        blocks.append((symbol, "\n".join(current)))
    return blocks


def _split_css(source: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    current: list[str] = []
    symbol = "stylesheet"
    depth = 0
    for line in source.splitlines():
        if depth == 0:
            m = _CSS_SELECTOR.match(line)
            if m:
                if current and any(c.strip() for c in current):
                    blocks.append((symbol, "\n".join(current)))
                    current = []
                symbol = m.group(1).strip()[:80]
        current.append(line)
        depth = max(depth + line.count("{") - line.count("}"), 0)
    if current and any(c.strip() for c in current):
        blocks.append((symbol, "\n".join(current)))
    return blocks


def _split_blank(source: str) -> list[tuple[str, str]]:
    return [("block", p) for p in re.split(r"\n\s*\n\s*\n", source) if p.strip()]


_SPLITTERS = {
    "python": _split_python,
    "javascript": _split_braced, "jsx": _split_braced,
    "typescript": _split_braced, "tsx": _split_braced,
    "css": _split_css, "scss": _split_css,
}


def _pack(blocks: list[tuple[str, str]],
          cfg: CodeChunkerConfig) -> list[tuple[list[str], str]]:
    """Merge small blocks toward target_chars; hard-split anything over max."""
    out: list[tuple[list[str], str]] = []
    buf_syms: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_syms, buf_len
        if buf and any(b.strip() for b in buf):
            out.append((list(dict.fromkeys(buf_syms)), "\n\n".join(buf)))
        buf, buf_syms, buf_len = [], [], 0

    for symbol, text in blocks:
        if len(text) > cfg.max_chars:
            flush()
            piece: list[str] = []
            plen = 0
            for line in text.splitlines():
                if plen + len(line) > cfg.max_chars and piece:
                    out.append(([symbol], "\n".join(piece)))
                    piece, plen = [], 0
                piece.append(line)
                plen += len(line) + 1
            if piece:
                out.append(([symbol], "\n".join(piece)))
            continue

        if buf_len + len(text) > cfg.target_chars and buf_len >= cfg.min_chars:
            flush()
        buf.append(text)
        buf_syms.append(symbol)
        buf_len += len(text)

    flush()
    return out


# ------------------------------------------------------------------ public API

def chunk_code(text: str, doc_id: str, doc_meta: dict | None = None,
               cfg: CodeChunkerConfig | None = None) -> list[ChunkRecord]:
    """Chunk one extracted code document. Same signature shape as chunker.chunk."""
    cfg = cfg or DEFAULT_CODE_CONFIG
    meta = doc_meta or {}
    source, lang, _declared = unwrap_fence(text)

    filename = meta.get("file_name") or "source"
    blocks = _SPLITTERS.get(lang, _split_blank)(source) or [("module", source)]
    packed = _pack(blocks, cfg)
    preamble = _preamble(source, cfg)

    records: list[ChunkRecord] = []
    cursor = 0
    for ordinal, (symbols, body) in enumerate(packed):
        start = source.find(body, cursor)
        if start < 0:
            start = cursor
        end = start + len(body)
        cursor = end

        named = [s for s in symbols if s not in ("module", "block")]
        head = f"# {filename}\n"
        head += (f"<!-- code:{lang} symbol={named[0]} -->\n" if named
                 else f"<!-- code:{lang} -->\n")
        pre = f"\n```{lang}\n{preamble}\n```\n" if preamble and ordinal > 0 else ""

        records.append(ChunkRecord(
            # chunk_id keyed on the raw body, matching chunker.chunk's contract
            # of hashing the unpadded text rather than the rendered block.
            chunk_id=chunk_id_for(doc_id, ordinal, body),
            doc_id=doc_id,
            ordinal=ordinal,
            text=f"{head}{pre}\n```{lang}\n{body}\n```\n",
            char_start=start,
            char_end=end,
            page=None,
            section_path=[filename] + named[:2],
            allowed_principals=meta.get("allowed_principals", ["*"]),
            data_classification=meta.get("data_classification", "internal"),
            pii_tags=meta.get("pii_tags", []),
            chunker=f"code_v{cfg.config_hash}",
            chunker_config_hash=cfg.config_hash,
        ))

    return records


def is_code_extraction(extractor: str) -> bool:
    """Route by what actually produced the text, not by file extension.

    HtmlModeRouter can send the same .html file down either track, so the
    extractor name is the only reliable signal at chunk time.
    """
    return str(extractor or "").startswith(("native_code", "html_source"))

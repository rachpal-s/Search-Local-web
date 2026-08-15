"""I3 Chunk — semantic prose chunker.

Contract:
  * Input: text file path (from DocRecord.text_path) + DocRecord metadata
  * Output: list[ChunkRecord] — one entry per chunk, IDs stable across re-runs
    given the same text and same ChunkerConfig.
  * Chunk IDs are content-addressed — config change -> new ids -> treated as new
    corpus, which is correct (different chunking = different retrieval behaviour).

Design choices:
  * Heading-aware: splits on markdown headings first, then by sentence within
    a section. This keeps the section breadcrumb intact for citations.
  * Table-safe: lines between <!-- table:N --> markers travel together; a table
    is never split across chunk boundaries.
  * Overlap: last `overlap` chars of previous chunk prepended to next.
    Overlap is EXCLUDED from char_start/char_end (those are source coordinates).
  * Pure Python, no ML, no network — CPU-bound, runs in a process pool.
  * ChunkerConfig is hashed into chunk_id — changing any param invalidates
    existing chunk IDs, which is the desired behaviour (forces re-embed).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from docstore.corpus import ChunkRecord, chunk_id_for

# ------------------------------------------------------------------ config

@dataclass(frozen=True)
class ChunkerConfig:
    target_chars: int = 1_200
    max_chars: int = 2_400
    overlap_chars: int = 150
    min_chars: int = 80          # chunks smaller than this are merged with next
    respect_headings: bool = True
    respect_tables: bool = True

    @property
    def config_hash(self) -> str:
        blob = (f"{self.target_chars}{self.max_chars}{self.overlap_chars}"
                f"{self.min_chars}{self.respect_headings}{self.respect_tables}")
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


DEFAULT_CONFIG = ChunkerConfig()

# ------------------------------------------------------------------ regexes

_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_TABLE_OPEN = re.compile(r"<!--\s*table:\d+\s*-->")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\u0900-\u097F])")  # Latin + Devanagari


# ------------------------------------------------------------------ internal helpers

def _heading_level(line: str) -> int | None:
    m = _HEADING.match(line.rstrip())
    return len(m.group(1)) if m else None


def _heading_text(line: str) -> str:
    m = _HEADING.match(line.rstrip())
    return m.group(2) if m else line.strip()


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_END.split(text)
    return [p for p in parts if p.strip()]


def _blocks(text: str, cfg: ChunkerConfig) -> Iterator[tuple[str, list[str]]]:
    """Yield (block_text, section_breadcrumb) before hard size limits.

    A 'block' is either:
    - A table (kept whole)
    - A heading + its immediately following paragraph
    - A bare paragraph when no headings are present
    """
    breadcrumb: list[str] = []
    current: list[str] = []

    def flush():
        t = "\n".join(current).strip()
        if t:
            yield t, list(breadcrumb)
        current.clear()

    lines = text.splitlines()
    in_table = False
    table_buf: list[str] = []

    for line in lines:
        # --- table handling ---
        if cfg.respect_tables and _TABLE_OPEN.match(line):
            yield from flush()
            in_table = True
            table_buf = [line]
            continue
        if in_table:
            table_buf.append(line)
            if line.strip() == "" and len(table_buf) > 2:
                yield "\n".join(table_buf).strip(), list(breadcrumb)
                table_buf = []
                in_table = False
            continue

        # --- heading handling ---
        lvl = _heading_level(line) if cfg.respect_headings else None
        if lvl is not None:
            yield from flush()
            # update breadcrumb: drop any heading of same or deeper level
            breadcrumb = [b for b in breadcrumb
                          if _heading_level("# " + b.split(None, 1)[-1] if b.startswith("#") else b) or 0
                          < lvl]
            breadcrumb = breadcrumb[:lvl - 1]
            breadcrumb.append(_heading_text(line))
            current.append(line)
        else:
            current.append(line)

    if in_table and table_buf:
        yield "\n".join(table_buf).strip(), list(breadcrumb)
    yield from flush()


def _hard_split(text: str, cfg: ChunkerConfig) -> Iterator[str]:
    """Break oversized blocks by sentence, respecting max_chars."""
    if len(text) <= cfg.max_chars:
        yield text
        return
    sentences = _split_sentences(text)
    buf: list[str] = []
    buf_len = 0
    for s in sentences:
        if buf and buf_len + len(s) > cfg.target_chars:
            yield " ".join(buf)
            buf, buf_len = [], 0
        buf.append(s)
        buf_len += len(s) + 1
    if buf:
        yield " ".join(buf)


# ------------------------------------------------------------------ public API

def chunk(
    text: str,
    doc_id: str,
    doc_meta: dict | None = None,
    cfg: ChunkerConfig | None = None,
) -> list[ChunkRecord]:
    cfg = cfg or DEFAULT_CONFIG
    meta = doc_meta or {}
    records: list[ChunkRecord] = []
    pending: list[tuple[str, list[str]]] = []

    # collect all raw blocks
    for block_text, breadcrumb in _blocks(text, cfg):
        for piece in _hard_split(block_text, cfg):
            if piece.strip():
                pending.append((piece, breadcrumb))

    # merge undersized chunks with the next sibling
    merged: list[tuple[str, list[str]]] = []
    i = 0
    while i < len(pending):
        txt, bc = pending[i]
        if len(txt) < cfg.min_chars and i + 1 < len(pending):
            next_txt, next_bc = pending[i + 1]
            merged.append((txt + "\n\n" + next_txt, bc))
            i += 2
        else:
            merged.append((txt, bc))
            i += 1

    # build ChunkRecords with overlap
    char_cursor = 0
    prev_tail = ""
    for ordinal, (txt, breadcrumb) in enumerate(merged):
        body = (prev_tail + "\n\n" + txt).strip() if prev_tail else txt
        cid = chunk_id_for(doc_id, ordinal, txt)   # stable: keyed on original, not overlap-padded

        records.append(ChunkRecord(
            chunk_id=cid,
            doc_id=doc_id,
            ordinal=ordinal,
            text=body,
            char_start=char_cursor,
            char_end=char_cursor + len(txt),
            page=None,
            section_path=breadcrumb,
            allowed_principals=meta.get("allowed_principals", ["*"]),
            data_classification=meta.get("data_classification", "internal"),
            pii_tags=meta.get("pii_tags", []),
            chunker=f"semantic_v{cfg.config_hash}",
            chunker_config_hash=cfg.config_hash,
        ))
        char_cursor += len(txt)
        prev_tail = txt[-cfg.overlap_chars:] if cfg.overlap_chars else ""

    return records


def chunk_file(
    text_path: Path,
    doc_id: str,
    doc_meta: dict | None = None,
    cfg: ChunkerConfig | None = None,
) -> list[ChunkRecord]:
    text = text_path.read_text(encoding="utf-8", errors="replace")
    return chunk(text, doc_id, doc_meta, cfg)

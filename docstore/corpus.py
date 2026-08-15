"""docstore/corpus.py — in-process replacement for `app.core.corpus`.

The Enterprise RAG pipeline (I1..I7) was written as an offline batch CLI that
depended on `app.core.corpus` for record types and content-addressed IDs, and on
Qdrant for vectors. This project has neither. This module supplies the same
contract so `chunker.py`, `extractors.py` and `pdf_extractor.py` can be dropped
in with a single import rewrite:

    from app.core.corpus import ChunkRecord, chunk_id_for
    ->
    from docstore.corpus import ChunkRecord, chunk_id_for

Deliberate differences from the original:
  * dataclasses instead of pydantic — this project doesn't depend on pydantic
    models for records, only pydantic-settings for config. `.model_dump()` is
    kept as an alias for `.to_dict()` so the batch CLIs still work if you ever
    run them.
  * No Manifest/run-id machinery. Idempotency here comes from the SQLite
    primary keys in docstore.store, not from append-only JSONL manifests, because
    ingestion is now per-upload and interactive rather than a resumable batch.

IDs stay content-addressed and stable, which is the property that matters:
re-uploading the same file into the same conversation is a no-op, not a
duplicate.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ------------------------------------------------------------------ id helpers

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def doc_id_for(path: Path | None = None, *, raw: bytes | None = None,
               scope: str = "") -> str:
    """Content-addressed document id. Same bytes in the same scope -> same id.

    Pass `raw` when the bytes are already in memory (the upload path) to avoid
    a second read. Falls back to reading `path`.

    `scope` exists because documents in this application are owned by a
    conversation, while a pure content hash is global. Without it, the same file
    uploaded to two threads produces one id, and the second upload silently
    adopts the first thread's row — so the file appears to vanish from the thread
    the user just added it to, and deleting either thread strands the other's
    chunks. Scoping by conversation makes idempotency mean "this file, in this
    thread", which is the intended granularity. Leave scope empty for a global
    corpus, which is what the batch pipeline wants.
    """
    if raw is None:
        if path is None:
            raise ValueError("doc_id_for requires either path or raw")
        raw = Path(path).read_bytes()
    digest = _sha256(raw)
    if scope:
        digest = _sha256(f"{scope}:{digest}".encode())
    return digest[:32]


def chunk_id_for(doc_id: str, ordinal: int, text: str) -> str:
    """Stable chunk id keyed on the doc, its position, and its own content.

    Keyed on the *unpadded* chunk text so that changing overlap settings does
    not silently invalidate every id (the chunker config hash handles that).
    """
    return _sha256(f"{doc_id}:{ordinal}:{text}".encode())[:32]


def blob_key_for(doc_id: str, page_idx: int, img_idx: int) -> str:
    """Key used by the two-phase image extractor for pending image blobs."""
    return f"{doc_id}:{page_idx}_{img_idx}"


# ------------------------------------------------------------------ records

@dataclass
class DocRecord:
    """One source document. Mirrors the I1/I2 contract."""
    doc_id: str
    source_uri: str
    file_name: str
    media_type: str
    size_bytes: int = 0
    text_path: str | None = None
    page_count: int | None = None
    extractor: str = ""
    extractor_version: str = ""
    quality: float = 1.0
    warnings: list[str] = field(default_factory=list)
    # --- I2 classify fields ---
    data_classification: str = "internal"
    allowed_principals: list[str] = field(default_factory=lambda: ["*"])
    pii_tags: list[str] = field(default_factory=list)
    classification_rules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # batch CLI compatibility
    def model_dump(self, mode: str = "python") -> dict[str, Any]:  # noqa: ARG002
        return self.to_dict()


@dataclass
class ChunkRecord:
    """One retrievable unit. Mirrors the I3/I4 contract exactly."""
    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    char_start: int = 0
    char_end: int = 0
    page: int | None = None
    section_path: list[str] = field(default_factory=list)
    allowed_principals: list[str] = field(default_factory=lambda: ["*"])
    data_classification: str = "internal"
    pii_tags: list[str] = field(default_factory=list)
    chunker: str = ""
    chunker_config_hash: str = ""
    # --- I4 enrich fields (fail open: empty is valid) ---
    keywords: list[str] = field(default_factory=list)
    summary: str | None = None
    entities: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def model_dump(self, mode: str = "python") -> dict[str, Any]:  # noqa: ARG002
        return self.to_dict()


@dataclass
class EmbeddingRef:
    """I5 output — the vector plus the frozen model identity that produced it."""
    chunk_id: str
    model: str
    dim: int
    vector: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

"""docstore/ingest.py — per-upload ingestion. The reference I1..I5 stages, collapsed
into one in-process async pipeline.

What changed from the batch pipeline, and why
---------------------------------------------
The reference pipeline is seven CLI stages that hand JSONL files to each other:
I1 extract -> I2 classify -> I3 chunk -> I4 enrich -> I5 embed -> I6 index ->
I7 certify. That decomposition earns its keep for a nightly corpus rebuild over
tens of thousands of documents: each stage is separately resumable, separately
parallel, and separately debuggable, and JSONL between stages means a failure at
I5 doesn't cost you the I1 extraction you already paid for.

An interactive upload has none of those properties. One document, one user
waiting, and the expensive artifact (extracted text) is cheap enough to redo.
So the stages run here as function calls in sequence rather than processes
passing files, and durability moves from JSONL manifests to the SQLite rows in
docstore.store. The *logic* of each stage is unchanged and still imported from the
original modules where it exists — this is a re-wiring, not a rewrite.

Kept from the reference:
  * I1 extractor selection and the Extracted/ExtractError contract
    (rag.extractors, rag.pdf_extractor — dropped in unchanged)
  * I3 semantic chunking, heading- and table-aware (rag.chunker, unchanged)
  * I2 escalate-only classification semantics
  * I4 fail-open enrichment
  * I5 frozen embedding model identity + dimension assertion

Dropped deliberately:
  * I6 alias verification / I7 certification. There is no live index to protect;
    see the note in docstore/store.py.
  * Process pools. One document does not amortise a fork; CPU-bound stages run
    in the default thread executor so the event loop stays responsive.
"""
from __future__ import annotations

import asyncio
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import httpx

from config import get_settings
from docstore import filetypes, store
from docstore.corpus import ChunkRecord, doc_id_for

cfg = get_settings()

# Media types and the accepted-suffix list now come from docstore.filetypes,
# which is shared with the batch job runner. They used to be a dict here, which
# was fine while uploads were the only door into the corpus; with two doors, a
# type accepted by one and not the other is a bug that never gets reported — the
# file simply never turns up in search results.
MEDIA_TYPES = {s: filetypes.media_type_for(s) for s in filetypes.ALL_SUFFIXES}

ACCEPTED_SUFFIXES = tuple(filetypes.ALL_SUFFIXES)


def upload_dir(conversation_id: str) -> Path:
    d = Path(getattr(cfg, "upload_dir", "data/uploads")) / conversation_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def safe_name(name: str) -> str:
    """Filesystem-safe basename. Strips directories — upload names are untrusted."""
    base = Path(name or "upload").name
    base = unicodedata.normalize("NFKD", base)
    base = re.sub(r"[^A-Za-z0-9._ -]+", "_", base).strip() or "upload"
    return base[:180]


# ------------------------------------------------------------------ I2 classify

LEVELS = ["public", "internal", "confidential", "restricted"]
_RANK = {lv: i for i, lv in enumerate(LEVELS)}


def _escalate(current: str, candidate: str) -> str:
    """Escalate-only, exactly as the reference I2. Never downgrades."""
    return candidate if _RANK.get(candidate, 0) > _RANK.get(current, 0) else current


# Built-in fallback rules. Override with a YAML file at cfg.classify_rules_path
# so Security owns the rules as config, not code (reference principle P2).
_DEFAULT_RULES: list[dict] = [
    {"name": "restricted_markers", "level": "restricted",
     "patterns": [r"\brestricted\b", r"\bstrictly confidential\b", r"\bsecret\b"]},
    {"name": "confidential_markers", "level": "confidential",
     "patterns": [r"\bconfidential\b", r"\bprivileged\b", r"\bnda\b",
                  r"\bdo not distribute\b", r"\binternal use only\b"]},
    {"name": "public_markers", "level": "public",
     "patterns": [r"\bpress release\b", r"\bpublic disclosure\b"]},
]

_PII_PATTERNS = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "phone_in": r"(?:\+91[\-\s]?)?[6-9]\d{9}\b",
    "pan": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
}


def _load_rules() -> list[dict]:
    path = getattr(cfg, "classify_rules_path", "") or ""
    if path and Path(path).exists():
        try:
            import yaml  # optional
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            rules = data.get("rules") or []
            if rules:
                return rules
        except Exception as e:  # noqa: BLE001 — bad rules must not block ingest
            print(f"[ingest] ⚠️ classify rules unreadable ({e}); using defaults.")
    return _DEFAULT_RULES


def classify(text: str, file_name: str) -> dict[str, Any]:
    """I2 — returns classification, ACL and PII tags. Additive, escalate-only."""
    haystack = f"{file_name}\n{text[:20_000]}".lower()
    level = "internal"          # default posture, same as reference
    fired: list[str] = []
    for rule in _load_rules():
        for pat in rule.get("patterns", []):
            if re.search(pat, haystack, re.I):
                level = _escalate(level, rule.get("level", "internal"))
                fired.append(rule.get("name", "unnamed"))
                break

    pii = sorted({tag for tag, pat in _PII_PATTERNS.items()
                  if re.search(pat, text[:200_000])})
    if pii:
        level = _escalate(level, "confidential")

    return {"data_classification": level,
            "allowed_principals": ["*"],   # never inferred from content
            "pii_tags": pii,
            "classification_rules": sorted(set(fired))}


# ------------------------------------------------------------------ I4 enrich

_STOP = set("""a an the and or but if then than that this these those of in on at to for
with without from by as is are was were be been being it its it's not no nor so such
we you they he she i our your their his her which who whom what when where why how all
any both each few more most other some only own same too very can will just don should
now about into over under again further once here there".split""".split())

_WORD = re.compile(r"[A-Za-z][A-Za-z\-']{2,}")
_PROPER = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,3})\b")


def _enrich_fallback(text: str, top_k: int = 8) -> dict[str, Any]:
    """Zero-dependency keyword + entity guess. Used when spaCy is absent."""
    counts: dict[str, int] = {}
    for m in _WORD.finditer(text.lower()):
        w = m.group(0)
        if w in _STOP or len(w) < 4:
            continue
        counts[w] = counts.get(w, 0) + 1
    keywords = [w for w, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:top_k]]
    ents = [{"text": m.group(1), "label": "PROPN"}
            for m in list(_PROPER.finditer(text))[:10]]
    return {"keywords": keywords, "summary": None, "entities": ents}


_nlp = None


def _enrich_spacy(text: str, top_k: int = 8) -> dict[str, Any]:
    """Mirrors the reference I4 spaCy strategy: noun chunks + named entities."""
    global _nlp
    if _nlp is None:
        import spacy  # raises ImportError -> caller falls back
        _nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    doc = _nlp(text[:100_000])
    seen: dict[str, int] = {}
    for nc in doc.noun_chunks:
        key = nc.text.strip().lower()
        if 3 < len(key) < 60 and key not in _STOP:
            seen[key] = seen.get(key, 0) + 1
    keywords = [k for k, _ in sorted(seen.items(), key=lambda kv: -kv[1])[:top_k]]
    entities = [{"text": e.text, "label": e.label_} for e in doc.ents[:20]]
    return {"keywords": keywords, "summary": None, "entities": entities}


def enrich_chunks(records: list[ChunkRecord]) -> list[ChunkRecord]:
    """I4 — fail open. A chunk that cannot be enriched keeps empty fields."""
    mode = (getattr(cfg, "enrich_mode", "auto") or "auto").lower()
    if mode == "none":
        return records
    strategy = _enrich_fallback
    if mode in ("auto", "spacy"):
        try:
            _enrich_spacy("probe text for model load")
            strategy = _enrich_spacy
        except Exception:  # noqa: BLE001
            if mode == "spacy":
                print("[ingest] ⚠️ spaCy unavailable; falling back to heuristics.")
            strategy = _enrich_fallback

    for r in records:
        try:
            out = strategy(r.text)
            r.keywords = out.get("keywords", []) or []
            r.summary = out.get("summary")
            r.entities = out.get("entities", []) or []
        except Exception:  # noqa: BLE001 — degraded, never fatal
            continue
    return records


# ------------------------------------------------------------------ I5 embed

class EmbedError(RuntimeError):
    pass


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch. Model identity and dimension are frozen by config.

    A dimension mismatch aborts rather than writing poisoned vectors — the same
    rule the reference I5 enforces at startup.
    """
    provider = (cfg.embed_provider or "ollama").lower()
    if not texts:
        return []

    if provider == "ollama":
        vecs = await _embed_ollama(texts)
    elif provider == "openai":
        vecs = await _embed_openai(texts)
    elif provider == "google":
        vecs = await _embed_google(texts)
    else:
        raise EmbedError(f"embed_provider_unsupported:{provider}")

    expected = int(cfg.embed_dimensions or 0)
    for v in vecs:
        if expected and len(v) != expected:
            raise EmbedError(
                f"embed_dim_mismatch: got {len(v)}, config says {expected}. "
                f"Set EMBED_DIMENSIONS to match {embed_model_name()} or change the model."
            )
    return vecs


def embed_model_name() -> str:
    provider = (cfg.embed_provider or "ollama").lower()
    return {
        "ollama": cfg.ollama_embed_model,
        "openai": cfg.openai_embed_model,
        "google": cfg.google_embed_model,
    }.get(provider, cfg.ollama_embed_model)


async def _embed_ollama(texts: list[str]) -> list[list[float]]:
    """Ollama /api/embeddings is single-text; loop it, as the reference I5 does."""
    url = cfg.ollama_embed_url.rstrip("/") + "/api/embeddings"
    headers = {"Content-Type": "application/json"}
    if cfg.ollama_embed_api_key:
        headers["Authorization"] = f"Bearer {cfg.ollama_embed_api_key}"
    out: list[list[float]] = []
    timeout = httpx.Timeout(float(getattr(cfg, "embed_timeout_seconds", 120)))
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        sem = asyncio.Semaphore(int(getattr(cfg, "embed_concurrency", 4)))

        async def one(t: str) -> list[float]:
            async with sem:
                r = await client.post(url, json={"model": cfg.ollama_embed_model,
                                                 "prompt": t})
                r.raise_for_status()
                data = r.json()
                v = data.get("embedding") or (data.get("data") or [{}])[0].get("embedding")
                if not v:
                    raise EmbedError("embed_empty_response")
                return list(map(float, v))

        out = list(await asyncio.gather(*(one(t) for t in texts)))
    return out


async def _embed_openai(texts: list[str]) -> list[list[float]]:
    if not cfg.openai_api_key:
        raise EmbedError("embed_credentials_missing:openai")
    timeout = httpx.Timeout(float(getattr(cfg, "embed_timeout_seconds", 120)))
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {cfg.openai_api_key}"},
            json={"model": cfg.openai_embed_model, "input": texts})
        r.raise_for_status()
        return [d["embedding"] for d in r.json()["data"]]


async def _embed_google(texts: list[str]) -> list[list[float]]:
    if not cfg.google_api_key:
        raise EmbedError("embed_credentials_missing:google")
    model = cfg.google_embed_model
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:batchEmbedContents?key={cfg.google_api_key}")
    body = {"requests": [{"model": f"models/{model}",
                          "content": {"parts": [{"text": t}]}} for t in texts]}
    timeout = httpx.Timeout(float(getattr(cfg, "embed_timeout_seconds", 120)))
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=body)
        r.raise_for_status()
        return [e["values"] for e in r.json()["embeddings"]]


# ------------------------------------------------------------------ orchestration

async def _extract(path: Path) -> tuple[str, dict]:
    """I1 — delegate to the reference extractor registry, off the event loop."""
    from docstore.extractors import ExtractError, select

    def work() -> tuple[str, dict]:
        ex = select(path)
        res = ex.extract(path)
        return res.text, {
            "extractor": f"{res.extractor}@{res.extractor_version}",
            "quality": res.quality,
            "page_count": res.page_count,
            "warnings": list(res.warnings or []),
        }

    try:
        return await asyncio.to_thread(work)
    except ExtractError as e:
        raise
    except Exception as e:  # noqa: BLE001
        raise ExtractError(f"extract_error:{type(e).__name__}", str(e)) from e


async def _chunk(text: str, doc_id: str, meta: dict,
                 extractor: str = "") -> list[ChunkRecord]:
    """I3 — routes to the prose chunker or the code chunker.

    The route is decided by which extractor produced the text, not by file
    extension. HtmlModeRouter can send the same .html down either path, so the
    extension is not a reliable signal by the time we get here.

    Sending code through the prose chunker is not a small quality loss: it
    splits on markdown headings and sentence boundaries, so functions get cut
    mid-body and the section breadcrumb comes out empty. The resulting chunks
    still look citable, which is what makes it worth routing properly.
    """
    from docstore.code_chunker import (
        DEFAULT_CODE_CONFIG, CodeChunkerConfig, chunk_code, is_code_extraction,
    )

    if is_code_extraction(extractor):
        ccfg = CodeChunkerConfig(
            target_chars=int(getattr(cfg, "code_chunk_target_chars", 1600)),
            max_chars=int(getattr(cfg, "code_chunk_max_chars", 3200)),
            min_chars=int(getattr(cfg, "code_chunk_min_chars", 120)),
            preamble_max_chars=int(getattr(cfg, "code_chunk_preamble_chars", 400)),
        )
        return await asyncio.to_thread(chunk_code, text, doc_id, meta, ccfg)

    from docstore.chunker import ChunkerConfig, chunk

    pcfg = ChunkerConfig(
        target_chars=int(getattr(cfg, "chunk_target_chars", 1200)),
        max_chars=int(getattr(cfg, "chunk_max_chars", 2400)),
        overlap_chars=int(getattr(cfg, "chunk_overlap_chars", 150)),
        # Deliberately NOT cfg.chunk_min_size — that one is the pre-existing
        # semantic-chunker floor (150) from the original RAG settings, and using
        # it here would merge away short-but-meaningful prose blocks.
        min_chars=int(getattr(cfg, "chunk_min_chars", 80)),
    )
    return await asyncio.to_thread(chunk, text, doc_id, meta, pcfg)


async def ingest_document(conversation_id: str, doc_id: str) -> dict:
    """Run I1..I5 for one already-persisted upload row. Idempotent.

    Status transitions are written as they happen so the UI can poll:
        pending -> extracting -> chunking -> embedding -> indexed
    Terminal failure states: failed (retryable), quarantined (won't retry).
    """
    doc = store.get_document(doc_id)
    if not doc:
        return {"doc_id": doc_id, "status": "missing"}
    if doc["status"] == "indexed":
        return {"doc_id": doc_id, "status": "indexed", "chunk_count": doc["chunk_count"]}

    path = Path(doc["stored_path"])
    from docstore.extractors import ExtractError

    # ---- I1 extract ----
    try:
        store.set_document_status(doc_id, "extracting")
        text, meta = await _extract(path)
    except ExtractError as e:
        # ExtractError carries the machine-readable code on `.reason_code`;
        # str(e) is the human message. Reading args[0] gets the message instead,
        # which then fails the `deferred` prefix test below and misfiles every
        # spreadsheet as a hard failure.
        code = getattr(e, "reason_code", None) or (e.args[0] if e.args else "extract_error")
        # `deferred_tabular` is a designed outcome, not a bug: spreadsheets need
        # a row-oriented chunker. Quarantine keeps the corpus honest about it.
        status = "quarantined" if str(code).startswith("deferred") else "failed"
        store.set_document_status(doc_id, status, reason_code=str(code))
        return {"doc_id": doc_id, "status": status, "reason_code": str(code)}

    if not (text or "").strip():
        store.set_document_status(doc_id, "failed", reason_code="empty_extraction")
        return {"doc_id": doc_id, "status": "failed", "reason_code": "empty_extraction"}

    # Keep the extracted text — it is the expensive, stable artifact.
    text_path = path.with_suffix(path.suffix + ".txt")
    try:
        text_path.write_text(text, encoding="utf-8")
    except OSError:
        pass

    # ---- I2 classify ----
    cls = classify(text, doc["file_name"])
    store.set_document_status(
        doc_id, "chunking",
        extractor=meta["extractor"], quality=meta["quality"],
        page_count=meta["page_count"], warnings=meta["warnings"],
        data_classification=cls["data_classification"],
        allowed_principals=cls["allowed_principals"], pii_tags=cls["pii_tags"])

    # ---- I3 chunk ----
    # file_name rides in the meta so a code chunk's section_path can name its
    # file; the prose chunker ignores the extra key.
    records = await _chunk(text, doc_id, {**cls, "file_name": doc["file_name"]},
                           meta["extractor"])
    if not records:
        store.set_document_status(doc_id, "failed", reason_code="no_chunks")
        return {"doc_id": doc_id, "status": "failed", "reason_code": "no_chunks"}

    # ---- I4 enrich (fail open) ----
    records = await asyncio.to_thread(enrich_chunks, records)
    store.save_chunks(conversation_id, records)
    store.set_document_status(doc_id, "embedding", chunk_count=len(records))

    # ---- I5 embed ----
    pending = store.chunks_missing_vectors(doc_id)
    if pending:
        batch = int(getattr(cfg, "embed_batch_size", 16))
        model = embed_model_name()
        try:
            for i in range(0, len(pending), batch):
                window = pending[i:i + batch]
                vecs = await embed_texts([t for _, t in window])
                store.save_vectors(conversation_id, model,
                                   list(zip([c for c, _ in window], vecs)))
        except Exception as e:  # noqa: BLE001
            # Chunks are already saved, so lexical search still works. Mark the
            # doc degraded rather than failed: partial retrieval beats none.
            store.set_document_status(doc_id, "degraded",
                                      reason_code=f"embed_error:{type(e).__name__}")
            print(f"[ingest] ⚠️ embedding failed for {doc['file_name']}: {e}")
            return {"doc_id": doc_id, "status": "degraded",
                    "chunk_count": len(records), "reason_code": str(e)[:200]}

    store.set_document_status(doc_id, "indexed", chunk_count=len(records))
    print(f"[ingest] ✅ {doc['file_name']}: {len(records)} chunks, "
          f"class={cls['data_classification']}, pii={cls['pii_tags'] or 'none'}")
    return {"doc_id": doc_id, "status": "indexed", "chunk_count": len(records),
            "data_classification": cls["data_classification"],
            "pii_tags": cls["pii_tags"]}


async def register_upload(conversation_id: str, file_name: str,
                          raw: bytes) -> dict:
    """Persist bytes + document row. Returns the row; ingestion runs separately.

    Splitting registration from ingestion is what lets the endpoint answer
    immediately and the UI show per-file progress instead of blocking on the
    slowest PDF in the batch.
    """
    name = safe_name(file_name)
    suffix = Path(name).suffix.lower()
    # Scoped to the conversation: re-uploading a file into the SAME thread is a
    # no-op, but the same file in another thread is a distinct document.
    did = doc_id_for(raw=raw, scope=conversation_id)

    if store.document_exists_indexed(did):
        existing = store.get_document(did)
        if existing:
            return {**existing, "duplicate": True}

    dest = upload_dir(conversation_id) / f"{did[:12]}_{name}"
    await asyncio.to_thread(dest.write_bytes, raw)

    row = {
        "id": did, "conversation_id": conversation_id, "file_name": name,
        "media_type": MEDIA_TYPES.get(suffix, "application/octet-stream"),
        "size_bytes": len(raw), "stored_path": str(dest), "status": "pending",
    }
    store.upsert_document(row)
    return {**row, "duplicate": False}


async def register_file(scope_id: str, path: Path, *,
                        display_name: str | None = None,
                        source_uri: str | None = None,
                        parent_doc_id: str | None = None,
                        job_id: str | None = None,
                        classification_hint: str | None = None) -> dict:
    """Register a file already on disk, without copying it.

    `register_upload` exists for the chat path, where bytes arrive in memory and
    must be written somewhere the app controls. A batch job is looking at files
    that already live on a mounted share, and copying ten thousand of them into
    data/uploads would double the storage for no benefit — the extractor reads
    them in place and only the extracted text is persisted.

    `scope_id` is a conversation id or a collection id; both are rows in the
    conversations table, so everything downstream is identical.

    Returns the document row, or a row marked `duplicate` when this exact
    content is already indexed in this scope. That check is what makes a re-run
    over an unchanged folder nearly free.
    """
    path = Path(path)
    raw_size = path.stat().st_size

    # Hash the bytes rather than (path, mtime): a file that moved between
    # folders is the same document, and a file that was touched but not edited
    # should not be re-embedded.
    digest = await asyncio.to_thread(_digest_file, path)
    did = doc_id_for(raw=digest, scope=scope_id)

    if store.document_exists_indexed(did):
        existing = store.get_document(did)
        if existing:
            return {**existing, "duplicate": True}

    name = display_name or path.name
    suffix = path.suffix.lower()
    row = {
        "id": did, "conversation_id": scope_id, "file_name": safe_name(name),
        "media_type": filetypes.media_type_for(suffix),
        "size_bytes": raw_size, "stored_path": str(path.resolve()),
        "status": "pending",
        "source_uri": source_uri or str(path.resolve()),
        "parent_doc_id": parent_doc_id, "job_id": job_id,
    }
    if classification_hint:
        row["data_classification"] = classification_hint
    store.upsert_document(row)
    return {**row, "duplicate": False}


def _digest_file(path: Path, block: int = 1 << 20) -> bytes:
    """Streaming SHA-256 of a file's contents, returned as bytes for doc_id_for.

    Streaming matters here in a way it does not for uploads: batch jobs walk
    files of arbitrary size, and read_bytes() on a 400 MB PDF in a worker
    process is a memory spike nobody planned for.
    """
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.digest()

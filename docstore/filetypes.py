"""docstore/filetypes.py — the ONE registry of ingestible file types.

Everything that needs to answer "can we ingest this?" reads from here:
    routers/uploads.py     the chat upload allowlist
    jobs/pipeline.py       the batch folder walker
    routers/jobs.py        /api/filetypes, served to chat.js and jobs.js

Before this module, `ingest.ACCEPTED_SUFFIXES` was derived from a MEDIA_TYPES
dict that lived inside ingest.py. That worked while there was one door into the
corpus. There are now two, and a type that ingests through one but not the other
is a bug nobody reports — the file just quietly never appears in search results.

Adding a type is a line here plus an Extractor in extractors.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path


@dataclass(frozen=True)
class FileKind:
    suffix: str
    media_type: str
    doc_type: str          # coarse bucket, surfaced in the UI
    track: str             # prose | code | image | tabular | archive
    max_mb: int = 25


_MB = 1024 * 1024

_KINDS: tuple[FileKind, ...] = (
    # ---- prose ----
    FileKind(".pdf",  "application/pdf", "pdf", "prose"),
    FileKind(".docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
             "docx", "prose"),
    FileKind(".md",       "text/markdown", "markdown", "prose"),
    FileKind(".markdown", "text/markdown", "markdown", "prose"),
    FileKind(".txt",      "text/plain",    "text",     "prose"),
    FileKind(".rst",      "text/x-rst",    "text",     "prose"),

    # ---- html: routed at extract time to prose or source (HtmlModeRouter) ----
    FileKind(".html",  "text/html",             "html", "prose"),
    FileKind(".htm",   "text/html",             "html", "prose"),
    FileKind(".xhtml", "application/xhtml+xml", "html", "prose"),

    # ---- code ----
    FileKind(".py",   "text/x-python",     "code", "code", 10),
    FileKind(".js",   "text/javascript",   "code", "code", 10),
    FileKind(".mjs",  "text/javascript",   "code", "code", 10),
    FileKind(".cjs",  "text/javascript",   "code", "code", 10),
    FileKind(".jsx",  "text/javascript",   "code", "code", 10),
    FileKind(".ts",   "text/x-typescript", "code", "code", 10),
    FileKind(".tsx",  "text/x-typescript", "code", "code", 10),
    FileKind(".css",  "text/css",          "code", "code", 10),
    FileKind(".scss", "text/x-scss",       "code", "code", 10),
    FileKind(".sql",  "application/sql",   "code", "code", 10),
    FileKind(".sh",   "application/x-sh",  "code", "code", 5),
    FileKind(".yaml", "application/yaml",  "code", "code", 5),
    FileKind(".yml",  "application/yaml",  "code", "code", 5),
    FileKind(".json", "application/json",  "code", "code", 10),
    FileKind(".toml", "application/toml",  "code", "code", 5),

    # ---- images ----
    FileKind(".png",  "image/png",  "image", "image"),
    FileKind(".jpg",  "image/jpeg", "image", "image"),
    FileKind(".jpeg", "image/jpeg", "image", "image"),
    FileKind(".webp", "image/webp", "image", "image"),
    FileKind(".tif",  "image/tiff", "image", "image"),
    FileKind(".tiff", "image/tiff", "image", "image"),
    FileKind(".bmp",  "image/bmp",  "image", "image"),
    FileKind(".gif",  "image/gif",  "image", "image"),

    # ---- tabular: XlsxExtractor quarantines these with `deferred_tabular`.
    # Listed anyway so the rejection is an honest "we know, it needs a row
    # chunker" rather than a silent unsupported-type drop.
    FileKind(".xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
             "spreadsheet", "tabular"),
    FileKind(".xlsm", "application/vnd.ms-excel.sheet.macroEnabled.12", "spreadsheet", "tabular"),
    FileKind(".xls",  "application/vnd.ms-excel", "spreadsheet", "tabular"),
    FileKind(".csv",  "text/csv", "spreadsheet", "tabular"),
    FileKind(".tsv",  "text/tab-separated-values", "spreadsheet", "tabular"),

    # ---- archives: expanded before extraction, never extracted directly ----
    FileKind(".zip", "application/zip", "archive", "archive", 10),
)

BY_SUFFIX: dict[str, FileKind] = {k.suffix: k for k in _KINDS}

ALL_SUFFIXES: tuple[str, ...] = tuple(BY_SUFFIX)
CODE_SUFFIXES: frozenset[str] = frozenset(k.suffix for k in _KINDS if k.track == "code")
IMAGE_SUFFIXES: frozenset[str] = frozenset(k.suffix for k in _KINDS if k.track == "image")
ARCHIVE_SUFFIXES: frozenset[str] = frozenset(k.suffix for k in _KINDS if k.track == "archive")
TABULAR_SUFFIXES: frozenset[str] = frozenset(k.suffix for k in _KINDS if k.track == "tabular")
HTML_SUFFIXES: frozenset[str] = frozenset({".html", ".htm", ".xhtml"})

# Archives are cheap to expand and easy to weaponise. Hard ceilings, not hints.
ARCHIVE_MAX_BYTES = 10 * _MB          # the .zip itself
ARCHIVE_MAX_UNCOMPRESSED = 200 * _MB  # total expanded bytes per archive
ARCHIVE_MAX_DEPTH = 3                 # zip in a zip in a zip, then stop
ARCHIVE_MAX_ENTRIES = 5_000
ARCHIVE_MAX_RATIO = 120               # compression-bomb guard


def is_supported(suffix: str) -> bool:
    return suffix.lower() in BY_SUFFIX


def kind_for(suffix: str) -> FileKind | None:
    return BY_SUFFIX.get(suffix.lower())


def media_type_for(suffix: str) -> str:
    k = BY_SUFFIX.get(suffix.lower())
    return k.media_type if k else "application/octet-stream"


def doc_type_for(suffix: str) -> str:
    k = BY_SUFFIX.get(suffix.lower())
    return k.doc_type if k else "document"


def track_for(suffix: str) -> str:
    k = BY_SUFFIX.get(suffix.lower())
    return k.track if k else "unknown"


def max_bytes_for(suffix: str, default_mb: int = 25) -> int:
    k = BY_SUFFIX.get(suffix.lower())
    return (k.max_mb if k else default_mb) * _MB


def is_code(suffix: str) -> bool:
    return suffix.lower() in CODE_SUFFIXES


def accept_attribute() -> str:
    """Value for <input type=file accept="...">, served over /api/filetypes."""
    return ",".join(sorted(ALL_SUFFIXES))


def as_catalog() -> dict:
    groups: dict[str, list[str]] = {}
    for k in _KINDS:
        groups.setdefault(k.track, []).append(k.suffix)
    return {
        "accept": accept_attribute(),
        "groups": {g: sorted(v) for g, v in sorted(groups.items())},
        "archive_max_bytes": ARCHIVE_MAX_BYTES,
    }


# ------------------------------------------------------------------ folder exclusion

def _slashify(s: str) -> str:
    """normcase + forward slashes, so Windows backslash paths and POSIX paths
    compare on equal footing without branching on os.sep everywhere below."""
    return os.path.normcase(s).replace("\\", "/")


def is_excluded(path: Path, root: Path, patterns: list[str]) -> bool:
    """True if `path` should be skipped because of a folder's exclude list.

    Three ways to write a pattern, because people fill this in three different
    ways in practice:

      * A bare name — "f2", "Employment Experiences" — excludes that folder
        (and everything under it) matched as a whole path COMPONENT, never a
        substring, so "f2" doesn't also exclude "f20" and "Employment" alone
        doesn't also exclude "Employment Experiences 2024".
      * A full or partial PATH — "C:\\RS\\...\\CV\\Employment Experiences", or
        the same thing relative to the root — because copying the address bar
        straight out of Explorer is the natural way to fill this in, and a
        pattern that only accepted bare names would silently match nothing
        for anyone who does that. Detected by the presence of a path
        separator; matched as an exact prefix on a directory boundary, so
        ".../CV/Employment" does not also swallow ".../CV/Employment Old".
      * Anything containing *, ?, or [ — a glob, matched against the path
        relative to root, e.g. "**/*.tmp".

    All comparisons go through _slashify (normcase + forward slashes), which
    folds case on Windows and unifies both separator styles, so a pattern
    typed with backslashes matches a path built with forward slashes and vice
    versa, and "F2" matches a folder actually named "f2" on disk.
    """
    if not patterns:
        return False

    path_norm = _slashify(str(path))
    try:
        rel_norm = _slashify(os.path.relpath(str(path), str(root)))
    except ValueError:
        rel_norm = path_norm
    dir_parts = rel_norm.split("/")[:-1]   # exclude the filename itself

    for raw in patterns:
        pat = _slashify((raw or "").strip())
        if not pat:
            continue

        if any(ch in pat for ch in "*?["):
            if fnmatch(rel_norm, pat) or fnmatch(path_norm, pat):
                return True
            continue

        if "/" in pat:
            # Path-like pattern (absolute, drive-rooted, or a multi-segment
            # relative path). Match as a directory-boundary prefix against
            # both the absolute path and the path relative to root, so either
            # form works regardless of which one was pasted in.
            pat = pat.rstrip("/")
            if path_norm == pat or path_norm.startswith(pat + "/"):
                return True
            if rel_norm == pat or rel_norm.startswith(pat + "/"):
                return True
            continue

        # Bare single name — matches as a whole path component anywhere
        # under root.
        if pat in dir_parts:
            return True

    return False

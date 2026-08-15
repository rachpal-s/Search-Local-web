"""docstore/archive.py — archives are containers, not documents.

The Extractor contract in docstore/extractors.py is one file -> one Extracted,
and register_upload() in ingest.py is one upload -> one document row. A .zip is
one file -> N documents, so it cannot be an Extractor and cannot be a document.
It is expanded in a pre-pass, and its children enter the normal pipeline as if
the user had dropped them individually.

Provenance survives the expansion. A file pulled out of an archive records

    /data/uploads/<conv>/abc123_specs.zip!/design/api.md

so a citation in an answer points at the archive the operator actually supplied,
rather than a staging path that will be deleted tomorrow.

Every guard below is a hard limit from docstore.filetypes, and every rejection
carries a reason code. Nothing is dropped silently — an archive that yields
nothing usable reports itself as the failure, because a chip that just goes
quiet is worse than an error.
"""
from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from docstore.filetypes import (
    ARCHIVE_MAX_BYTES,
    ARCHIVE_MAX_DEPTH,
    ARCHIVE_MAX_ENTRIES,
    ARCHIVE_MAX_RATIO,
    ARCHIVE_MAX_UNCOMPRESSED,
    ARCHIVE_SUFFIXES,
    is_supported,
)

PROVENANCE_SEP = "!/"


@dataclass
class ExpansionResult:
    staging_root: Path
    files_written: int = 0
    archives_seen: int = 0
    bytes_written: int = 0
    rejected: list[dict] = field(default_factory=list)
    provenance: dict[str, str] = field(default_factory=dict)   # staged path -> logical uri

    def reject(self, uri: str, reason: str, detail: str = "") -> None:
        self.rejected.append({"source_uri": uri, "reason_code": reason, "detail": detail})


def is_archive(path: Path | str) -> bool:
    return Path(path).suffix.lower() in ARCHIVE_SUFFIXES


def _safe_member_path(dest: Path, member_name: str) -> Path | None:
    """Resolve a member against dest, or None if it escapes the directory."""
    if member_name.startswith("/") or ".." in Path(member_name).parts:
        return None
    target = (dest / member_name).resolve()
    try:
        target.relative_to(dest.resolve())
    except ValueError:
        return None
    return target


def _expand_one(archive_path: Path, dest: Path, logical_uri: str,
                depth: int, result: ExpansionResult) -> None:
    result.archives_seen += 1

    size = archive_path.stat().st_size
    if size > ARCHIVE_MAX_BYTES:
        result.reject(logical_uri, "archive_too_large", f"{size} > {ARCHIVE_MAX_BYTES}")
        return
    if depth > ARCHIVE_MAX_DEPTH:
        result.reject(logical_uri, "archive_too_deeply_nested", f"depth={depth}")
        return

    try:
        zf = zipfile.ZipFile(archive_path)
    except zipfile.BadZipFile as e:
        result.reject(logical_uri, "archive_corrupt", str(e))
        return

    nested: list[tuple[Path, str]] = []

    with zf:
        infos = zf.infolist()
        if len(infos) > ARCHIVE_MAX_ENTRIES:
            result.reject(logical_uri, "archive_too_many_entries", str(len(infos)))
            return

        total = sum(i.file_size for i in infos)
        if total > ARCHIVE_MAX_UNCOMPRESSED:
            result.reject(logical_uri, "archive_uncompressed_too_large", f"{total} bytes")
            return
        if size > 0 and (total / size) > ARCHIVE_MAX_RATIO:
            result.reject(logical_uri, "archive_compression_bomb", f"ratio={total / size:.0f}x")
            return

        for info in infos:
            if info.is_dir():
                continue
            # Symlinks: the unix mode lives in the high bits of external_attr.
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                result.reject(f"{logical_uri}{PROVENANCE_SEP}{info.filename}",
                              "archive_symlink_rejected")
                continue

            target = _safe_member_path(dest, info.filename)
            if target is None:
                result.reject(f"{logical_uri}{PROVENANCE_SEP}{info.filename}",
                              "archive_path_traversal")
                continue

            suffix = target.suffix.lower()
            member_uri = f"{logical_uri}{PROVENANCE_SEP}{info.filename}"

            if suffix in ARCHIVE_SUFFIXES:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out, length=1 << 20)
                nested.append((target, member_uri))
                continue

            if not is_supported(suffix):
                result.reject(member_uri, f"unsupported_media_type:{suffix or 'none'}")
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, length=1 << 20)

            result.files_written += 1
            result.bytes_written += info.file_size
            result.provenance[str(target)] = member_uri

    # Recurse, then delete the intermediate .zip so no walker mistakes it for
    # a document later.
    for nested_path, nested_uri in nested:
        sub_dest = nested_path.parent / f"{nested_path.stem}__unzipped"
        sub_dest.mkdir(parents=True, exist_ok=True)
        _expand_one(nested_path, sub_dest, nested_uri, depth + 1, result)
        nested_path.unlink(missing_ok=True)


def expand_file(archive_path: Path, staging_root: Path) -> ExpansionResult:
    """Expand one archive. Used by the upload path."""
    archive_path = Path(archive_path)
    dest = Path(staging_root)
    dest.mkdir(parents=True, exist_ok=True)
    result = ExpansionResult(staging_root=dest)
    _expand_one(archive_path, dest, str(archive_path.resolve()), 1, result)
    if not result.files_written and not result.rejected:
        result.reject(str(archive_path.resolve()), "archive_empty")
    return result


def expand_tree(source_root: Path, staging_root: Path) -> ExpansionResult:
    """Expand every archive under source_root. Used by the batch job.

    Non-archive files stay where they are — the folder walker handles those.
    Only archive contents land in staging, which keeps staging small and makes
    it safe to delete between runs.
    """
    staging_root = Path(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    result = ExpansionResult(staging_root=staging_root)

    archives = [p for p in sorted(Path(source_root).rglob("*"))
                if p.is_file() and p.suffix.lower() in ARCHIVE_SUFFIXES]

    for archive in archives:
        rel = archive.relative_to(source_root).with_suffix("")
        dest = staging_root / rel
        dest.mkdir(parents=True, exist_ok=True)
        _expand_one(archive, dest, str(archive.resolve()), 1, result)

    return result


def cleanup(staging_root: Path) -> None:
    shutil.rmtree(Path(staging_root), ignore_errors=True)


def reason_text(reason_code: str) -> str:
    """Human sentence for a reason code.

    chat.js renders `${file_name}: ${reason}` straight into a toast, so these
    have to be things an operator can act on, not codes to look up.
    """
    return {
        "archive_too_large": "This archive is over the 10 MB limit.",
        "archive_uncompressed_too_large": "The contents of this archive are too large to expand.",
        "archive_compression_bomb": "This archive expands far more than its size suggests and was not opened.",
        "archive_too_many_entries": "This archive holds too many files to process.",
        "archive_too_deeply_nested": "Archives nested more than three deep are not opened.",
        "archive_corrupt": "This archive could not be read.",
        "archive_empty": "No ingestible files were found inside this archive.",
        "archive_path_traversal": "A file in this archive pointed outside it and was skipped.",
        "archive_symlink_rejected": "A shortcut inside this archive was skipped.",
        "unsupported_media_type": "This file type is not ingestible.",
        "vendored_or_generated_code": "Skipped as vendored or build output.",
        "binary_content_in_text_file": "This looked like a text file but holds binary data.",
        "low_code_quality": "This file appears minified or generated.",
        "deferred_tabular": "Spreadsheets need a row-oriented reader and are not indexed for text search yet.",
    }.get(reason_code.split(":")[0],
          reason_code.replace("_", " ").capitalize() + ".")

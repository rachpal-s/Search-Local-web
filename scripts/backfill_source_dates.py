"""scripts/backfill_source_dates.py — one-time metadata fix, no re-ingestion.

Populates documents.source_modified_at for rows ingested BEFORE that column
existed, so the recency-aware retrieval fix (docstore/retrieve.py) actually
has real dates to work with instead of falling back to ingestion order —
which is nearly worthless as a "which version is current" signal, since
several versions of the same document are usually ingested together in one
batch job, seconds apart, regardless of which underlying file is actually
years older.

Why this doesn't need a re-ingest
----------------------------------
A batch-ingested document's `stored_path` points at the ORIGINAL file on
disk — register_file() never copies batch-sourced files, precisely so a
folder of ten thousand documents doesn't get doubled in storage. That means
the real answer ("when was this file last modified") is still sitting right
there on disk, unrelated to anything about extraction, chunking, or
embeddings. This script reads exactly that one piece of information and
writes exactly one column. Nothing about the document's text, chunks, or
vectors is touched.

What this does NOT fix
-----------------------
Chat-uploaded documents (register_upload) store a COPY in data/uploads/,
whose mtime reflects when it was copied there — close to ingestion time
already, so backfilling it adds little over the existing doc_created_at
fallback. This script still backfills those (harmless), but don't expect it
to meaningfully change chat-upload rankings. The real benefit here is for
batch/collection-ingested documents, which is exactly the scenario that
prompted this.

Usage:
    python -m scripts.backfill_source_dates              # apply
    python -m scripts.backfill_source_dates --dry-run     # preview only
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from docstore import store


def backfill(dry_run: bool = False) -> dict[str, int]:
    store.init_db()
    tally = {"updated": 0, "file_missing": 0, "already_set": 0, "no_stored_path": 0}

    with store.conn() as c:
        rows = c.execute(
            "SELECT id, file_name, stored_path, source_modified_at "
            "FROM documents"
        ).fetchall()

    for row in rows:
        if row["source_modified_at"]:
            tally["already_set"] += 1
            continue

        stored_path = row["stored_path"]
        if not stored_path:
            tally["no_stored_path"] += 1
            continue

        p = Path(stored_path)
        if not p.exists():
            # Common for chat uploads whose file was later cleaned up, or an
            # archive child whose staged copy was removed after ingestion.
            # Leaving it NULL is correct — the neutral fallback in
            # _recency_fraction() already handles this gracefully.
            tally["file_missing"] += 1
            continue

        try:
            mtime = datetime.fromtimestamp(
                p.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
        except OSError:
            tally["file_missing"] += 1
            continue

        if dry_run:
            print(f"  would set: {row['file_name']:<40} -> {mtime}")
        else:
            with store.conn() as c:
                c.execute(
                    "UPDATE documents SET source_modified_at=? WHERE id=?",
                    (mtime, row["id"]))
        tally["updated"] += 1

    return tally


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would change without writing anything.")
    args = ap.parse_args()

    print(f"[backfill] db = {store._db_path()}")
    print(f"[backfill] mode = {'DRY RUN — nothing will be written' if args.dry_run else 'APPLY'}")
    tally = backfill(dry_run=args.dry_run)

    print(f"\n[backfill] updated:          {tally['updated']:,}")
    print(f"[backfill] already had a date: {tally['already_set']:,}")
    print(f"[backfill] source file missing (left NULL, harmless): {tally['file_missing']:,}")
    print(f"[backfill] no stored_path on record (left NULL, harmless): {tally['no_stored_path']:,}")


if __name__ == "__main__":
    main()

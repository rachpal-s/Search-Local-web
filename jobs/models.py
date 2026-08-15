"""jobs/models.py — the job contract shared by API, worker and dashboard.

Plain dataclasses, matching docstore/corpus.py's choice: this project uses
pydantic only for settings, and records elsewhere are dataclasses with a
`to_dict()`. Following that keeps one convention instead of two.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobStatus(str, Enum):
    queued = "queued"
    claimed = "claimed"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelling = "cancelling"
    cancelled = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled}


class Phase(str, Enum):
    """Batch phases.

    These are NOT the per-document I1..I5 stages — those still run per document
    inside docstore/ingest.py and are the same code the chat upload path uses.
    A phase is a batch-level step that only exists because there are many
    documents: expanding archives, walking folders, then ingesting the resulting
    work list with bounded parallelism.
    """
    expand   = "expand"     # unpack archives into staging
    discover = "discover"   # walk folders, hash, build the work list
    ingest   = "ingest"     # per-document I1..I5, parallel
    verify   = "verify"     # count what landed, report the tally


PHASES: list[Phase] = list(Phase)

# Rough share of wall-clock, used only to render an honest progress bar.
PHASE_WEIGHTS: dict[Phase, float] = {
    Phase.expand: 0.05, Phase.discover: 0.10, Phase.ingest: 0.82, Phase.verify: 0.03,
}


@dataclass
class FolderSpec:
    """One ingestion root. Recursion is the default, not an opt-in."""
    path: str
    recursive: bool = True
    include: list[str] = field(default_factory=list)   # suffixes; empty = all supported
    exclude: list[str] = field(default_factory=list)   # glob patterns
    classification_hint: str | None = None             # floor applied in I2

    def __post_init__(self) -> None:
        self.include = [s if s.startswith(".") else f".{s}"
                        for s in (x.strip().lower() for x in self.include) if s]
        self.exclude = [x.strip() for x in self.exclude if x.strip()]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FolderSet:
    set_id: str
    name: str
    description: str = ""
    folders: list[FolderSpec] = field(default_factory=list)
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "folders": [f.to_dict() for f in self.folders]}


@dataclass
class JobOptions:
    """Everything tunable per run, persisted so a re-run is exact."""
    workers: int = 0                  # 0 = auto (cores - 25% headroom)
    embed_concurrency: int = 4        # network-bound, independent of workers
    vision_concurrency: int = 2       # image extraction, cloud-rate-limited
    enrich_mode: str = "auto"         # auto | spacy | none
    force: bool = False               # reprocess even documents already indexed
    skip_images: bool = False         # skip image files entirely (they are slow)
    max_files: int = 0                # 0 = no cap; a safety valve for first runs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PhaseRun:
    phase: Phase
    seq: int
    status: str = "pending"           # pending | running | succeeded | failed | skipped
    started_at: str | None = None
    finished_at: str | None = None
    items_done: int = 0
    items_total: int = 0
    tally: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["phase"] = self.phase.value
        return d


@dataclass
class Job:
    job_id: str
    name: str
    collection_id: str
    collection_name: str = ""
    status: JobStatus = JobStatus.queued
    set_id: str | None = None
    folders: list[FolderSpec] = field(default_factory=list)
    options: JobOptions = field(default_factory=JobOptions)
    created_by: str = "system"
    created_at: str = field(default_factory=utcnow)
    started_at: str | None = None
    finished_at: str | None = None
    worker_id: str | None = None
    heartbeat_at: str | None = None
    current_phase: Phase | None = None
    progress_pct: float = 0.0
    files_discovered: int = 0
    docs_ok: int = 0
    docs_failed: int = 0
    docs_quarantined: int = 0
    docs_skipped: int = 0
    chunks_total: int = 0
    error: str | None = None
    phases: list[PhaseRun] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "status": self.status.value,
            "current_phase": self.current_phase.value if self.current_phase else None,
            "folders": [f.to_dict() for f in self.folders],
            "options": self.options.to_dict(),
            "phases": [p.to_dict() for p in self.phases],
        }


@dataclass
class JobEvent:
    event_id: int
    job_id: str
    ts: str
    level: str
    phase: str | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

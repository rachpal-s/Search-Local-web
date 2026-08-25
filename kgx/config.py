"""kgx/config.py — configuration for the ontology plane.

Read from the environment, with defaults that are safe rather than useful:
the plane starts in `legacy` mode and changes nothing until switched on
deliberately.

The one setting here that is a CONTROL rather than a preference is
COLLECTION_NAMESPACE. Until now the mapping from collection to namespace has
lived only in conversation — "Identities & Certs is the personal one" — and
an isolation boundary that depends on remembering something is not a
boundary. namespace_for() fails closed: an unmapped collection raises rather
than defaulting, because defaulting would silently route personal documents
into the Cloud/AI graph the first time a collection is renamed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


class ConfigError(RuntimeError):
    """Configuration that cannot be safely guessed. Never caught internally."""


# ---------------------------------------------------------------------------
# COLLECTION -> NAMESPACE. Keys are matched case-insensitively against both
# the conversation title and its id. Add a row here before ingesting a new
# collection; there is deliberately no fallback.
# ---------------------------------------------------------------------------
COLLECTION_NAMESPACE: dict[str, str] = {
    "library":            "cloud_ai",
    "library (be)":       "cloud_ai",
    "identities & certs": "personal",
    "identities and certs": "personal",
}


def namespace_for(collection: str) -> str:
    """Resolve a collection title or id to its namespace. Fails closed."""
    key = (collection or "").strip().lower()
    if key in COLLECTION_NAMESPACE:
        return COLLECTION_NAMESPACE[key]
    raise ConfigError(
        f"collection {collection!r} has no namespace mapping. Add it to "
        f"kgx.config.COLLECTION_NAMESPACE. Refusing to guess: a wrong guess "
        f"here puts personal records into the Cloud/AI graph.")


# ---------------------------------------------------------------------------
# MODE SWITCH
# ---------------------------------------------------------------------------
MODES = ("legacy", "ontology", "shadow", "compare")


@dataclass
class KgxSettings:
    # legacy   current co-occurrence path only; the plane is inert
    # shadow   plane builds at ingest, never touches an answer  <- standing mode
    # ontology plane serves the answer
    # compare  both hydrate, both labelled, diff logged to telemetry
    mode: str = "legacy"

    # Relation-pass trigger. R3 chosen from measured budget: it drops the
    # 3,424 chunks whose only entities were vendor name-drops, while keeping
    # Capability pairs. R5 saved a further 0.4h and was not worth the recall
    # risk — the OTHER-predicate queue can justify tightening later.
    trigger_rule: str = "R3"
    trigger_min_entities: int = 2
    trigger_weak_classes: tuple[str, ...] = ("Vendor", "_ambiguous")

    # Extraction. Runs in the jobs plane, never on the query path.
    extract_model: str = "qwen2.5:14b"
    extract_concurrency: int = 4
    extract_timeout_s: float = 60.0
    prompt_version: str = "v1"
    extractor_version: str = "0.1.0"

    # Skip documents the duplicate scan retired: extracting from a superseded
    # version costs GPU to produce claims a newer version already supersedes,
    # and inflates every downstream weight while doing it.
    skip_superseded: bool = True

    # Confidence floor below which a claim is quarantined rather than served.
    min_confidence: float = 0.35

    # Salt for Identifier.value_hash in the personal namespace. MUST be set
    # via env in any real use — the default is a placeholder that makes the
    # hashes worthless, which is the correct failure mode for a secret.
    identifier_salt: str = "CHANGE-ME"
    salt_version: int = 1

    ontology_version: str = "0.2.0"
    module_versions: dict[str, str] = field(
        default_factory=lambda: {"cloud_ai": "0.2.0", "personal": "0.1.0"})

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ConfigError(f"KGX_MODE={self.mode!r}; expected one of {MODES}")
        if self.identifier_salt == "CHANGE-ME" and self.mode != "legacy":
            raise ConfigError(
                "KGX_IDENTIFIER_SALT is unset. The personal namespace hashes "
                "identifiers with it; leaving the default would make every "
                "hash reproducible by anyone with the source.")


_settings: KgxSettings | None = None


def get_settings() -> KgxSettings:
    global _settings
    if _settings is None:
        s = KgxSettings(
            mode=os.getenv("KGX_MODE", "legacy").strip().lower(),
            trigger_rule=os.getenv("KGX_TRIGGER_RULE", "R3"),
            extract_model=os.getenv("KGX_EXTRACT_MODEL", "qwen2.5:14b"),
            extract_concurrency=int(os.getenv("KGX_EXTRACT_CONCURRENCY", "4")),
            min_confidence=float(os.getenv("KGX_MIN_CONFIDENCE", "0.35")),
            identifier_salt=os.getenv("KGX_IDENTIFIER_SALT", "CHANGE-ME"),
        )
        s.validate()
        _settings = s
    return _settings


def is_enabled() -> bool:
    """True when the plane does anything at all."""
    return get_settings().mode != "legacy"


def serves_answers() -> bool:
    """True when the plane's output can reach a user-visible answer.
    shadow mode builds the graph but must never influence a response."""
    return get_settings().mode in ("ontology", "compare")

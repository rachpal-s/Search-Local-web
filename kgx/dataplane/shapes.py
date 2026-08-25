"""kgx/dataplane/shapes.py — the validation gate.

Checks that an extracted relation is structurally possible before it reaches
the graph: does the subject's class appear in the predicate's declared
domain, and the object's class in its range?

WHY THIS EXISTS
---------------
A 200-chunk pilot produced "Distributed Tracing implements OpenTelemetry"
and "Evaluation appliesTo Chunking" at confidence 1.0. Both are forbidden by
the ontology — and both were written anyway, because the constraints lived
in YAML *descriptions* that only a human reads. An ontology whose rules are
not machine-checkable is documentation, not a schema (anti-pattern #30).

QUARANTINE, NOT DELETE
----------------------
A violating claim is marked `quarantined` and withheld from the served
graph, but it is still recorded with its reason. Two consequences worth the
storage: the rejection distribution tells you whether the model has a
systematic bias, and a constraint that turns out to be too tight can be
loosened and its quarantined claims replayed without re-extraction.

NO AUTO-INVERSION
-----------------
When A→B violates but B→A would be valid, that is REPORTED, never applied.
Silently flipping direction is guessing, and a systematic direction bias is
a PROMPT defect — fix it once in the prompt rather than a thousand times in
the data.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

MODULE_YAML = Path("kgx/semantic/modules/cloud_ai.yaml")


@dataclass
class Verdict:
    ok: bool
    reason: str = ""
    inversion_valid: bool = False   # would B->A have passed?
    detail: str = ""


@lru_cache(maxsize=1)
def load_constraints(path: str = str(MODULE_YAML)) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cons = data.get("predicate_constraints") or {}
    preds = set(data["enums"]["CloudAiPredicateEnum"]["permissible_values"])
    missing = preds - set(cons)
    if missing:
        # Fail loudly: a predicate the extractor may emit but the gate cannot
        # judge would pass through unchecked, which is worse than no gate at
        # all because it looks validated.
        raise ValueError(f"predicates without constraints: {sorted(missing)}")
    return cons


def _check(cons: dict, pred: str, s_cls: str, o_cls: str) -> tuple[bool, str]:
    c = cons[pred]
    if c.get("unconstrained"):
        return True, ""
    dom, rng = c.get("domain") or [], c.get("range") or []
    if not dom and not rng:
        return False, "predicate not permitted between entities"
    if s_cls not in dom:
        return False, f"subject class {s_cls} not in domain {dom}"
    if o_cls not in rng:
        return False, f"object class {o_cls} not in range {rng}"
    if c.get("same_class") and s_cls != o_cls:
        return False, f"{pred} requires matching classes, got {s_cls}/{o_cls}"
    return True, ""


def check(predicate: str, subject_class: str, object_class: str) -> Verdict:
    """Validate one relation against the ontology's declared constraints."""
    cons = load_constraints()
    if predicate not in cons:
        return Verdict(False, "unknown_predicate", detail=predicate)

    ok, why = _check(cons, predicate, subject_class, object_class)
    if ok:
        return Verdict(True)

    # Symmetric predicates carry no direction, so the reverse is the same
    # assertion and passing either way is correct.
    if cons[predicate].get("symmetric"):
        ok2, _ = _check(cons, predicate, object_class, subject_class)
        if ok2:
            return Verdict(True)

    inv, _ = _check(cons, predicate, object_class, subject_class)
    return Verdict(False, "domain_range_violation", inversion_valid=inv, detail=why)


def type_component(verdict: Verdict) -> float:
    """Confidence contribution. The pilot showed the model reports 1.0 on
    everything, which made self-report a constant and confidence useless as
    a ranking signal. Type validity is an INDEPENDENT check, so it restores
    genuine spread across the score."""
    if verdict.ok:
        return 1.0
    return 0.15 if verdict.inversion_valid else 0.0


def selftest() -> int:
    """Cases taken verbatim from the 200-chunk pilot."""
    cases = [
        # (predicate, subject_class, object_class, expect_ok, note)
        ("implements", "Technology", "Pattern", True,  "Terraform -> IaC"),
        ("addressesRisk", "Pattern", "Risk",    True,  "Guardrails -> Prompt Injection"),
        ("offeredBy", "Model", "Vendor",        True,  "Gemini -> Google"),
        ("implements", "Capability", "Technology", False, "Distributed Tracing -> OTel (backwards)"),
        ("implements", "Capability", "Pattern", False, "Evaluation -> HITL"),
        ("appliesTo", "Capability", "Capability", False, "Evaluation -> Chunking"),
        ("dependsOn", "Capability", "Capability", True, "Evaluation -> Observability"),
        ("offeredBy", "Protocol", "Vendor",     True,  "A2A -> Google"),
        ("supersedes", "Service", "Pattern",    False, "same_class violation"),
        ("alternativeTo", "Model", "Service",   True,  "symmetric"),
        ("contradicts", "Service", "Service",   False, "claim-level only"),
        ("hasCapability", "Service", "Vendor",  False, "range violation"),
    ]
    fails = 0
    print(f"{'predicate':<18}{'subject':<13}{'object':<13}{'result':<10}{'note'}")
    print("-" * 84)
    for pred, s, o, expect, note in cases:
        v = check(pred, s, o)
        good = v.ok == expect
        fails += not good
        flag = "ok" if v.ok else ("QUARANTINE" + ("*" if v.inversion_valid else ""))
        mark = " " if good else "  <-- UNEXPECTED"
        print(f"{pred:<18}{s:<13}{o:<13}{flag:<10}{note}{mark}")
    print("\n* = the inverse direction would be valid (reported, never applied)")
    print(f"{len(cases) - fails}/{len(cases)} as expected")
    return 1 if fails else 0


if __name__ == "__main__":
    import sys
    sys.exit(selftest())

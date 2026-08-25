"""kgx/switch.py — the single integration point.

    from kgx import switch
    switch.install()          # one line, in main.py's lifespan

That is the entire footprint in the host application. Remove the line and the
system is byte-for-byte what it was; the folder can then be deleted.

HOW THE BIND WORKS, AND ITS ONE FRAGILITY
-----------------------------------------
retrieve.context_for_query does its import INSIDE the function:

    from docstore.graph_hydrate import hydrate as graph_hydrate   # line 328

Because that runs on every call, replacing the attribute on the
docstore.graph_hydrate MODULE takes effect immediately and needs no edit to
retrieve.py.

The fragility is exactly that: if the import is ever hoisted to module level,
retrieve.py captures its own reference at import time and this bind silently
stops working — silently being the problem. verify() therefore checks the
bind actually took and prints loudly if it did not, and it is called at the
end of install(). For anything beyond a trial, prefer the explicit two-line
edit documented in EXPLICIT_PATCH below: it is uglier and it cannot fail
quietly.

MODES (KGX_MODE)
    legacy    no bind at all. The plane is inert. Default.
    shadow    kgx hydration runs, its trace is logged, its blocks are
              DISCARDED. Nothing reaches an answer.
    ontology  kgx hydration replaces the legacy path.
    compare   both run; both are returned, labelled.
"""
from __future__ import annotations

import sys

from kgx import config as kgx_config

EXPLICIT_PATCH = """
    # docstore/retrieve.py, in context_for_query, replacing line 328:
    from kgx.switch import hydrate_for_mode as graph_hydrate
"""

_installed = False
_original = None


def _wrap(legacy):
    """Build the hydration callable for the configured mode."""
    from kgx.retrievalplane.hydrate import hydrate as kgx_hydrate, hydrate_compare
    mode = kgx_config.get_settings().mode

    if mode == "ontology":
        return kgx_hydrate

    if mode == "compare":
        async def _compare(scope_ids, hits):
            return await hydrate_compare(scope_ids, hits, legacy)
        return _compare

    async def _shadow(scope_ids, hits):
        # Build the graph context, record what it WOULD have contributed,
        # then throw it away. This is how you accumulate evidence about the
        # new path without it being able to affect a single answer.
        try:
            blocks, trace = await kgx_hydrate(scope_ids, hits)
            if blocks:
                print(f"[kgx.shadow] would have added {len(blocks)} block(s), "
                      f"{(trace or {}).get('facts', 0)} fact(s)")
        except Exception as e:                                 # noqa: BLE001
            print(f"[kgx.shadow] error (ignored): {e}")
        return await legacy(scope_ids, hits)
    return _shadow


async def hydrate_for_mode(scope_ids, hits):
    """Explicit entry point, for the non-monkeypatch integration."""
    from docstore import graph_hydrate as legacy_mod
    legacy = _original or legacy_mod.hydrate
    if kgx_config.get_settings().mode == "legacy":
        return await legacy(scope_ids, hits)
    return await _wrap(legacy)(scope_ids, hits)


def install() -> bool:
    """Bind the plane. Returns True if anything was changed."""
    global _installed, _original
    if _installed:
        return True

    cfg = kgx_config.get_settings()
    if cfg.mode == "legacy":
        print("[kgx] KGX_MODE=legacy — plane inert, nothing bound.")
        return False

    try:
        from docstore import graph_hydrate as legacy_mod
    except ImportError as e:
        print(f"[kgx] cannot import docstore.graph_hydrate ({e}); not installed.")
        return False

    _original = legacy_mod.hydrate
    legacy_mod.hydrate = _wrap(_original)
    _installed = True

    print(f"[kgx] mode={cfg.mode}  ontology=v{cfg.ontology_version}  "
          f"modules={cfg.module_versions}")
    if cfg.mode == "shadow":
        print("[kgx] shadow: graph context is computed and DISCARDED. "
              "No answer is affected.")
    elif cfg.mode == "compare":
        print("[kgx] compare: BOTH graphs hydrate every query. "
              "Roughly 2x hydration cost — not for production traffic.")
    verify()
    return True


def uninstall() -> None:
    global _installed, _original
    if not _installed:
        return
    from docstore import graph_hydrate as legacy_mod
    legacy_mod.hydrate = _original
    _installed = False
    print("[kgx] uninstalled; legacy hydration restored.")


def verify() -> bool:
    """Confirm the bind actually reached the caller.

    A monkeypatch that fails is worse than one that was never attempted,
    because the logs say the plane is active while every query silently uses
    the old path.
    """
    ok = True
    try:
        from docstore import graph_hydrate as legacy_mod
        if legacy_mod.hydrate is _original:
            print("[kgx] BIND FAILED: module attribute unchanged.")
            ok = False
    except ImportError:
        return False

    try:
        import inspect
        from docstore import retrieve
        src = inspect.getsource(retrieve.context_for_query)
        if "from docstore.graph_hydrate import hydrate" not in src:
            print("[kgx] WARNING: retrieve.context_for_query no longer imports "
                  "hydrate locally. The bind may not take effect. Use the "
                  "explicit patch:" + EXPLICIT_PATCH)
            ok = False
        else:
            print("[kgx] bind verified: retrieve.py imports hydrate at call time.")
    except Exception as e:                                     # noqa: BLE001
        print(f"[kgx] could not verify bind ({e}); assuming it holds.")
    return ok


def status() -> dict:
    cfg = kgx_config.get_settings()
    try:
        from kgx.repositories import graph_repo as repo
        graph = {ns: repo.stats(ns) for ns in ("cloud_ai", "personal")} \
            if repo.is_available() else {"error": "neo4j unavailable"}
    except Exception as e:                                     # noqa: BLE001
        graph = {"error": str(e)}
    return {"mode": cfg.mode, "installed": _installed,
            "ontology_version": cfg.ontology_version,
            "modules": cfg.module_versions, "graph": graph}


if __name__ == "__main__":
    import json
    print(json.dumps(status(), indent=2))
    sys.exit(0)

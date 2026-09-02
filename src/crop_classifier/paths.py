"""Path resolution. Tables are workspace-switchable; the feature store is shared.

The 12-class work and the 3-class perennial work share **exactly the same parcels, years
and pixels** — only the label column differs. So the 322 MB pixel/feature store is shared
and fixed, while the label/split tables live in a per-workspace directory selected by the
``CC_PROC`` environment variable::

    export CC_PROC=data/processed/perennial
    export CC_RUNS=runs/perennial

**Call ``proc()`` / ``runs()`` at call time, never bind them to a module-level constant at
import time** — an import-time constant is resolved before the env var can matter and
would silently overwrite the (completed, unspent-test) 12-class tables.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC_SHARED = ROOT / "data" / "processed"      # raw inputs + the 12-class workspace
FEAT = PROC_SHARED / "features"                # the Piura pixel store — the DEFAULT


def proc() -> Path:
    """Workspace directory for label/split tables. Override with ``CC_PROC``."""
    p = Path(os.environ.get("CC_PROC", PROC_SHARED))
    p.mkdir(parents=True, exist_ok=True)
    return p


def feat() -> Path:
    """Pixel/feature store directory. Override with ``CC_FEAT``.

    The 12-class and 3-class perennial workspaces **share** this store (same parcels, years
    and pixels, only the label differs), so it defaults to ``FEAT`` and neither sets
    ``CC_FEAT``.

    The all-Peru work (§4) is the first that needs a *different* store — different parcels —
    setting ``CC_FEAT=data/processed/all_peru/features``, which keeps the Piura store
    immutable.

    Same rule as ``proc()``: **call at call time, never bind at import.**
    """
    p = Path(os.environ.get("CC_FEAT", FEAT))
    p.mkdir(parents=True, exist_ok=True)
    return p


def runs() -> Path:
    """Run-output directory. Override with ``CC_RUNS``."""
    p = Path(os.environ.get("CC_RUNS", ROOT / "runs"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def shared_input(name: str) -> Path:
    """A table that lives in the *national* workspace but is read from several others.

    ``tenure_two_period.parquet`` is the case: built once over 14 departments, read by the
    DiD workspace which has no copy. Resolution is ``CC_PROC`` first, then
    ``data/processed/all_peru`` — a workspace with its own copy wins, a clone with only the
    committed national one still works.
    """
    here = proc() / name
    if here.exists():
        return here
    national = PROC_SHARED / "all_peru" / name
    if national.exists():
        return national
    return here          # let the caller raise on the workspace-local path


def labels_dir() -> Path:
    """Where one photo-interpretation campaign keeps its sample, chips and labels.

    Defaults to ``<CC_PROC>/labels_s2`` — the campaign of record. Override with ``CC_LABELS``
    to run a **second round** without overwriting the first: the committed
    ``label_sample.parquet`` is the key every returned label of record is joined through, and
    a fresh draw would replace it.

    Same rule as ``proc()``: resolved at call time, never bound at import.
    """
    d = Path(os.environ.get("CC_LABELS", proc() / "labels_s2"))
    d.mkdir(parents=True, exist_ok=True)
    return d

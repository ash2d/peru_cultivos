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

    The 12-class and 3-class perennial workspaces deliberately **share** this store: same
    parcels, same years, same pixels, only the label column differs — so it defaults to
    ``FEAT`` and those two never set ``CC_FEAT``.

    The all-Peru work (docs/RESULTS.md §4) is the first thing that needs a *different* store:
    different parcels entirely. It sets ``CC_FEAT=data/processed/all_peru/features``, which
    keeps the audited Piura store immutable.

    Same rule as ``proc()``: **call this at call time, never bind it at import.**
    """
    p = Path(os.environ.get("CC_FEAT", FEAT))
    p.mkdir(parents=True, exist_ok=True)
    return p


def runs() -> Path:
    """Run-output directory. Override with ``CC_RUNS``."""
    p = Path(os.environ.get("CC_RUNS", ROOT / "runs"))
    p.mkdir(parents=True, exist_ok=True)
    return p

"""Pool per-fold validation predictions into one ``preds_cv.parquet`` (plan §4.3).

Every model here is compared on *pooled spatial CV*, never on the locked test set, so this
concatenation is needed once per run. It was copy-pasted three times during the 12-class
work; it lives here now.

Usage::

    CC_RUNS=runs/perennial uv run python -m crop_classifier.cli perennial pool-cv \\
        runs/perennial/rules_3c
    CC_RUNS=runs/perennial uv run python -m crop_classifier.cli eval \\
        runs/perennial/rules_3c --preds preds_cv.parquet
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def pool_cv(run_dir: Path, out_name: str = "preds_cv.parquet") -> Path:
    """Concatenate ``fold*/preds_val.parquet``. Folds are disjoint by construction, so the
    pooled frame is one prediction per validation parcel."""
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("fold*/preds_val.parquet"))
    if not files:
        raise FileNotFoundError(f"no fold*/preds_val.parquet under {run_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    dup = int(df["COD_PREDIO"].duplicated().sum())
    if dup:
        # would mean overlapping validation folds — a split bug, not a pooling bug
        raise ValueError(f"{dup:,} parcels appear in more than one fold's validation set")
    out = run_dir / out_name
    df.to_parquet(out, index=False)
    print(f"pooled {len(files)} folds -> {out} ({len(df):,} parcels)")
    return out

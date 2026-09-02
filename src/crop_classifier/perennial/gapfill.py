"""Extract the parcels the 3-class map newly admits (plan §3.4).

The 12-class label policy dropped ~16k parcels (rare crops, unmerged intercrops); most of
them come back at group level, and the ones that were never extracted have
``quality_ok = NA``. Without this step the 3-class model would silently train on a subset
and the "more parcels" advantage would evaporate.

Everything writes into the **shared** feature store (D5): same parcels, years and pixels,
only the label differs. The result is a superset of what the 12-class runs used, keyed by
``COD_PREDIO``.

Run with::

    CC_PROC=data/processed/perennial uv run python -m crop_classifier.perennial.gapfill
"""

from __future__ import annotations

import argparse

import geopandas as gpd
import pandas as pd

from crop_classifier.features import landsat_gee as lg
from crop_classifier.labels import apply_coverage_gate
from crop_classifier.paths import PROC_SHARED, feat, proc


def missing_parcels() -> gpd.GeoDataFrame:
    """Workspace parcels with no coverage measurement yet."""
    df = gpd.read_parquet(proc() / "modeling_parcels.parquet")
    return df[df["quality_ok"].isna()].copy()


def statics_union() -> pd.DataFrame:
    """Static columns for **every** parcel in either workspace.

    ``assemble`` merges statics onto the feature rows; feeding it only the 3-class table
    would blank ``area_ha``/``n_valid_obs``/… for 12-class-only parcels and corrupt the
    shared store for the completed 12-class runs.
    """
    cols = ["COD_PREDIO", "area_ha", "n_pixels_est", "centroid_lat", "n_valid_obs",
            "max_gap"]
    frames = []
    for p in {PROC_SHARED / "modeling_parcels.parquet",
              proc() / "modeling_parcels.parquet"}:
        if p.exists():
            frames.append(gpd.read_parquet(p)[cols])
    return pd.concat(frames, ignore_index=True).drop_duplicates("COD_PREDIO")


def run(chunk_size: int = 400, pixel_chunk_size: int = 40,
        max_chunks: int | None = None) -> None:
    todo = missing_parcels()
    print(f"{len(todo):,} parcels never extracted "
          f"({todo['label'].value_counts().to_dict()})")
    if todo.empty:
        print("nothing to do")
        return

    # stage 1 into its own coverage file, then merge into the shared coverage store so the
    # gate sees one table (run_coverage dedupes on COD_PREDIO within its own chunk dir)
    print("\n--- stage 1: coverage ---", flush=True)
    lg.run_coverage(parcels=todo, chunk_size=chunk_size, max_chunks=max_chunks)
    cov = pd.read_parquet(feat() / "coverage.parquet")
    df = apply_coverage_gate(cov)

    survivors = df[(df["quality_ok"] == True)  # noqa: E712
                   & df["COD_PREDIO"].isin(todo["COD_PREDIO"])]
    print(f"\n--- stage 2: pixels for {len(survivors):,} gate survivors ---", flush=True)
    lg.run_pixels(parcels=survivors, chunk_size=pixel_chunk_size, max_chunks=max_chunks)
    print("\nextraction done — re-run `features assemble` to rebuild the model inputs")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract newly-admitted parcels (plan §3.4)")
    ap.add_argument("--chunk-size", type=int, default=400)
    ap.add_argument("--pixel-chunk-size", type=int, default=40)
    ap.add_argument("--max-chunks", type=int, default=None)
    a = ap.parse_args()
    run(a.chunk_size, a.pixel_chunk_size, a.max_chunks)


if __name__ == "__main__":
    main()

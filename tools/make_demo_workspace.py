"""Build ``data/demo/`` — a small REAL sample that runs with no Earth Engine account.

The demo exists because ``data/`` is 33 GB and gitignored, so a fresh clone could otherwise
run nothing at all and the first thing a new collaborator met was a wall.

It is drawn from the national workspace: real parcels, real declared labels, real extracted
Landsat features, real spatial splits. Redistributing this sample publicly was confirmed as
permitted; the **full** archive is not redistributable and is not here
(``docs/DATA_ACCESS.md``).

Two design choices, both so the demo teaches the right thing rather than a simplified thing:

* **Six departments, not one.** ``cc advanced lodo`` needs more than one held-out unit, and
  leave-one-department-out is the evaluation this project turns on. A single-department demo
  would let someone learn the pipeline while never meeting the one split that decides.
* **Whole regions move together** into a split and into a fold — the assignment is inherited
  from the national build, not redrawn. Splitting parcels at random would put a parcel's
  neighbours in its own training set, which is the cheapest way there is to manufacture an
  inflated cross-validation score.

Regenerating needs the national workspace on disk; using the committed output does not::

    uv run python tools/make_demo_workspace.py
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_PROC = ROOT / "data" / "processed" / "all_peru"
SRC_FEAT = SRC_PROC / "features"
OUT = ROOT / "data" / "demo"
SEED = 20260901

# Six departments spanning the country's range of class balance: TUMBES is 75 % PERENNIAL,
# LAMBAYEQUE 6 %. A demo drawn from six similar departments would make leave-one-department-out
# look easy, which is the opposite of the lesson.
DEPTS = ["PIURA", "LAMBAYEQUE", "ICA", "TUMBES", "ANCASH", "AYACUCHO"]
PER_DEPT = 220
N_FOLDS = 5

# Geometry to ~1 m. Full cadastral precision is 10x the file size and buys the demo nothing —
# the features are already extracted, so nothing here re-reads a pixel.
COORD_DP = 5


def _require_source() -> None:
    if not (SRC_PROC / "modeling_parcels.parquet").exists():
        raise SystemExit(
            f"{SRC_PROC / 'modeling_parcels.parquet'} not found.\n"
            f"Regenerating the demo needs the national workspace on disk. If you only want to "
            f"USE the demo, it is already committed in {OUT} — nothing to run.\n"
            f"See docs/DATA_ACCESS.md."
        )


def draw(rng: np.random.Generator) -> gpd.GeoDataFrame:
    """A stratified draw: up to PER_DEPT parcels per department, balanced across classes.

    Balanced rather than proportional because the point is a demo that trains: the national
    class balance would give a fold with three PERENNIAL parcels in it. ⚠️ That means the demo
    is NOT a representative sample and no share computed from it means anything — the same
    caveat the real national sample carries, for the same reason.
    """
    g = gpd.read_parquet(SRC_PROC / "modeling_parcels.parquet")
    g = g[g["dept"].isin(DEPTS) & (g["quality_ok"] == True)]      # noqa: E712

    keep = []
    for dept, block in g.groupby("dept", sort=True):
        per_class = max(1, PER_DEPT // block["label"].nunique())
        for _, cls in block.groupby("label", sort=True):
            n = min(per_class, len(cls))
            keep.append(cls.iloc[rng.choice(len(cls), n, replace=False)])
    return pd.concat(keep).sort_values("COD_PREDIO").reset_index(drop=True)


def refold(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Renumber folds so all five are populated after the draw.

    The national fold ids survive the sample unevenly — a fold can end up empty, and an empty
    validation fold is a crash rather than a warning. Regions are kept **whole**: every parcel
    of a region moves to the same fold, which is the property that makes the split honest.
    """
    tv = df["split"] == "trainval"
    regions = sorted(df.loc[tv, "region_id"].unique())
    fold_of = {r: i % N_FOLDS for i, r in enumerate(regions)}
    df.loc[tv, "fold"] = df.loc[tv, "region_id"].map(fold_of).astype(int)
    df.loc[~tv, "fold"] = -1
    df["fold"] = df["fold"].astype("int64")
    return df


def main() -> None:
    _require_source()
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "features").mkdir(exist_ok=True)

    df = refold(draw(rng))
    df["geometry"] = df["geometry"].set_precision(10 ** -COORD_DP)

    feats = pd.read_parquet(SRC_FEAT / "features_lightgbm.parquet")
    feats = feats[feats["COD_PREDIO"].isin(set(df["COD_PREDIO"]))].reset_index(drop=True)
    # a parcel with no feature row cannot train; drop it here rather than let it become a
    # silent all-NaN row downstream
    df = df[df["COD_PREDIO"].isin(set(feats["COD_PREDIO"]))].reset_index(drop=True)

    classes = sorted(df["label"].unique())
    label_map = {c: i for i, c in enumerate(classes)}
    df["label_id"] = df["label"].map(label_map).astype("int64")

    df.to_parquet(OUT / "modeling_parcels.parquet", index=False)
    feats.to_parquet(OUT / "features" / "features_lightgbm.parquet", index=False)
    (OUT / "label_map.json").write_text(json.dumps(label_map, indent=2) + "\n")

    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {OUT}  ({size:.1f} MB)")
    print(f"  {len(df):,} parcels, {len(DEPTS)} departments, "
          f"{len(feats.columns) - 1} features, {df['region_id'].nunique()} regions")
    print(pd.crosstab(df["dept"], df["label"]).to_string())
    print(pd.crosstab(df["split"], df["label"]).to_string())
    print("\n⚠️  a stratified sample, not a representative one — no share computed from it "
          "means anything.")


if __name__ == "__main__":
    main()

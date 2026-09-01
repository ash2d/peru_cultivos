"""Generate ``data/demo/`` — a workspace that runs with no raw data and no Earth Engine.

**Everything it writes is synthetic.** The PETT/COFOPRI and CENAGRO files this project is
built on were obtained under a research agreement and are not redistributable, so the demo
cannot be a sample of them. What it is instead is a *schema-faithful fake*: the same tables,
the same column names and dtypes, the same split and buffer machinery, generated from a fixed
seed — enough that ``cc -w demo train`` exercises the real code path end to end and a new
collaborator can see the pipeline work before they have any data.

⚠️ **No number produced from this workspace means anything.** The class signal is a synthetic
separation drawn here; a macro-F1 from the demo measures this script, not Peru. Everything it
writes is stamped so that cannot be mistaken — see ``data/demo/README.md`` and the
``synthetic`` column on the parcel table.

Regenerate with::

    uv run python tools/make_demo_workspace.py

Real data lives in the other workspaces (``uv run cc workspaces``); see docs/DATA_ACCESS.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "demo"
SEED = 20260901

# Three classes, matching the perennial strand's label space so the demo exercises the label
# map, the class weights and the confusion matrix the same way a real run does.
CLASSES = ["ANNUAL", "PASTURE_FALLOW", "PERENNIAL"]
CLASS_P = [0.55, 0.25, 0.20]

# Two invented departments. Names that could not be mistaken for Peruvian ones, because the
# whole failure mode here is a demo number being quoted as a result.
DEPTS = ["DEMOLANDIA", "EJEMPLIA"]

N_PARCELS = 600
N_FOLDS = 5

# The channels and statistics the real assemble step writes, so `drop_features meta,location`
# and the ORDER_FEATURES group resolve against a real column set rather than a stub.
CHANNELS = ["B", "G", "R", "NIR", "SWIR1", "SWIR2", "NDVI", "EVI", "NDWI", "NDMI", "BSI"]
STATS = ["median", "mean", "std", "min", "max", "p25", "p75", "amp", "slope",
         "h_mean", "h_cos", "h_sin"]
META = ["n_dates", "n_valid_pixels", "frac_l7"]


def _square(lon: float, lat: float, side_deg: float) -> Polygon:
    h = side_deg / 2
    return Polygon([(lon - h, lat - h), (lon + h, lat - h),
                    (lon + h, lat + h), (lon - h, lat + h)])


def build_parcels(rng: np.random.Generator) -> gpd.GeoDataFrame:
    """Parcels on a coarse grid, so the spatial blocking has something real to block on."""
    dept = rng.choice(DEPTS, size=N_PARCELS)
    # two well-separated invented regions, one per department
    base_lon = np.where(dept == DEPTS[0], -80.5, -76.0)
    base_lat = np.where(dept == DEPTS[0], -5.0, -12.0)
    lon = base_lon + rng.uniform(-0.35, 0.35, N_PARCELS)
    lat = base_lat + rng.uniform(-0.35, 0.35, N_PARCELS)

    y = rng.choice(len(CLASSES), size=N_PARCELS, p=CLASS_P)
    label = np.array(CLASSES)[y]

    # 5 km blocks, the unit the real splitter holds out; region ids one level coarser.
    block = [f"{d}_{int((a + 90) * 20):04d}_{int((o + 180) * 20):04d}"
             for d, a, o in zip(dept, lat, lon, strict=True)]
    region = [f"{d}_{int((a + 90) * 4):03d}_{int((o + 180) * 4):03d}"
              for d, a, o in zip(dept, lat, lon, strict=True)]

    gdf = gpd.GeoDataFrame({
        "COD_PREDIO": [f"DEMO_{i:05d}" for i in range(N_PARCELS)],
        "dept": dept,
        "label": label,
        "label_id": y.astype("int64"),
        "label_reason": "synthetic",
        "crop_set": "synthetic",
        "year": rng.integers(1998, 2007, N_PARCELS).astype("int64"),
        "area_ha": np.round(rng.gamma(2.0, 1.5, N_PARCELS) + 0.2, 3),
        "n_pixels_est": rng.integers(4, 200, N_PARCELS).astype(float),
        "centroid_lon": lon,
        "centroid_lat": lat,
        "geometry": [_square(o, a, 0.004) for o, a in zip(lon, lat, strict=True)],
        "population_weight": 1.0,
        "block_id": block,
        "region_id": region,
        # ⚠️ the stamp: any table derived from this one carries it forward
        "synthetic": True,
        "n_valid_obs": pd.array(rng.integers(8, 26, N_PARCELS), dtype="Int64"),
        "max_gap": pd.array(rng.integers(1, 5, N_PARCELS), dtype="Int64"),
        "quality_ok": pd.array([True] * N_PARCELS, dtype="boolean"),
    }, crs="EPSG:4326")
    return gdf


def assign_splits(gdf: gpd.GeoDataFrame, rng: np.random.Generator) -> gpd.GeoDataFrame:
    """Whole regions to a split, then whole regions to a fold — the real discipline.

    Splitting parcels at random would leak a parcel's neighbours into its own training set,
    which is the single easiest way to manufacture an inflated CV score. The demo does it
    properly so that reading this file teaches the right thing.
    """
    regions = np.array(sorted(gdf["region_id"].unique()))
    rng.shuffle(regions)
    n_test = max(1, int(round(0.2 * len(regions))))
    test_regions = set(regions[:n_test])
    gdf["split"] = np.where(gdf["region_id"].isin(test_regions), "test", "trainval")

    tv_regions = [r for r in regions if r not in test_regions]
    fold_of = {r: i % N_FOLDS for i, r in enumerate(tv_regions)}
    gdf["fold"] = gdf["region_id"].map(fold_of).fillna(-1).astype("int64")

    # The dead zone: a trainval parcel within one block of a held-out region is excluded
    # from that fold's TRAIN side. Here every region is well separated, so the columns exist
    # and are all False — present so `data.fold_split` runs unmodified.
    gdf["buffer_excl_test"] = False
    for k in range(N_FOLDS):
        gdf[f"buffer_excl_fold{k}"] = False
    return gdf


def build_features(gdf: gpd.GeoDataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """A feature table with the real column set and a deliberately modest class signal.

    The separation is set so the demo lands somewhere around 0.6-0.75 macro-F1 — good enough
    to look like a working pipeline, not so good that anyone mistakes it for a result.
    """
    n = len(gdf)
    y = gdf["label_id"].to_numpy()
    cols: dict[str, np.ndarray] = {"COD_PREDIO": gdf["COD_PREDIO"].to_numpy()}

    for ch_i, ch in enumerate(CHANNELS):
        # each class sits at a different offset in each channel, plus noise
        offset = np.array([0.0, 0.35, 0.7])[y] * (0.5 + 0.5 * np.sin(ch_i))
        base = rng.normal(0.25 + 0.02 * ch_i, 0.12, n) + offset * 0.15
        for st in STATS:
            jitter = rng.normal(0, 0.05, n)
            cols[f"{ch}_{st}"] = np.round(base + jitter, 5)

    # acquisition metadata: real columns, and — as in the real store — correlated with the
    # label year rather than with the land, which is exactly why `--drop-features meta`
    # exists and why the demo should reproduce the hazard rather than hide it
    yr = gdf["year"].to_numpy()
    cols["n_dates"] = (yr - 1996 + rng.integers(0, 6, n)).astype(float)
    cols["n_valid_pixels"] = gdf["n_pixels_est"].to_numpy()
    cols["frac_l7"] = np.clip((yr - 1998) / 8 + rng.normal(0, 0.1, n), 0, 1)
    return pd.DataFrame(cols)


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "features").mkdir(exist_ok=True)

    gdf = assign_splits(build_parcels(rng), rng)
    feats = build_features(gdf, rng)

    gdf.to_parquet(OUT / "modeling_parcels.parquet", index=False)
    feats.to_parquet(OUT / "features" / "features_lightgbm.parquet", index=False)
    (OUT / "label_map.json").write_text(
        json.dumps({c: i for i, c in enumerate(CLASSES)}, indent=2) + "\n")

    counts = gdf.groupby(["split", "label"]).size().unstack(fill_value=0)
    print(f"wrote {OUT}")
    print(f"  {len(gdf)} parcels, {len(feats.columns) - 1} features, "
          f"{gdf['region_id'].nunique()} regions, {N_FOLDS} folds")
    print(counts.to_string())
    print("\n⚠️  synthetic. No number from this workspace is a result.")


if __name__ == "__main__":
    main()

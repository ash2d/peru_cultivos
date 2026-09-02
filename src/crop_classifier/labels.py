"""Build the modelling label table (plan.md §7, decisions A1/A2/A5).

Reads ``data/processed/training_crop_polygon.parquet`` (the §5b pipeline output) and the
label policy in ``config/data.yaml``; writes:

* ``data/processed/modeling_parcels.parquet`` — one row per eligible ``COD_PREDIO`` with
  ``label``, ``label_id``, gates, centroid, geometry. ``splits.py`` later adds
  ``block_id / fold / split``; the stage-1 coverage pass finalises ``quality_ok``.
* ``data/processed/label_map.json`` — class name -> id (stable, sorted).
* ``data/processed/label_exclusions.csv`` — how many parcels each rule removed.

Policy (locked decisions):
* single-crop parcels only, EXCEPT multi-crop parcels whose exact crop-set is in the
  config ``merge`` map (e.g. ``CAFE+PLATANO -> CAFE``);
* ``crop`` category -> crop classes; ``pasture``/``fallow`` -> their own land-cover
  classes ``PASTURE``/``FALLOW``; ``land_prep``/``unspecified`` dropped;
* hard area gate 0.09–50 ha; a (trusted-as-is) titling ``year`` is required;
* crops with < ``min_class_parcels`` eligible parcels -> ``other`` (or dropped).

Run with::

    uv run python -m crop_classifier.labels            # or: cli labels build
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from crop_classifier.config_loader import load_yaml_config
from crop_classifier.paths import PROC_SHARED, proc

CONFIG_DIR = Path(__file__).resolve().parent / "config"

# The raw label input is shared across workspaces; the outputs are workspace-local, resolved
# at call time (see paths.py — never bind proc() at import).
F_POLY = PROC_SHARED / "training_crop_polygon.parquet"

LANDCOVER_NAME = {"pasture": "PASTURE", "fallow": "FALLOW"}
PIXEL_HA = 0.09  # one 30 m Landsat pixel in hectares


def load_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml_config(path or CONFIG_DIR / "data.yaml")


def out_paths() -> tuple[Path, Path, Path]:
    """``(modeling_parcels, label_map, label_exclusions)`` in the *current* workspace."""
    p = proc()
    return (p / "modeling_parcels.parquet", p / "label_map.json",
            p / "label_exclusions.csv")


def crop_set_key(crops: list[str]) -> str:
    """Canonical '+'-joined sorted key for a parcel's crop set (the ``merge`` map key)."""
    return "+".join(sorted(set(crops)))


def assign_raw_label(row: pd.Series, cfg: dict[str, Any]) -> tuple[str | None, str]:
    """Return ``(label, reason)``; label None means excluded, reason says why/kept-how."""
    crops, cats = list(row["crops"]), list(row["crop_categories"])
    merge = cfg.get("merge") or {}
    if len(crops) == 1:
        cat = cats[0]
        if cat in cfg["keep_categories"]:
            return crops[0], "single_crop"
        if cat in cfg["landcover_classes"]:
            return LANDCOVER_NAME.get(cat, cat.upper()), "landcover"
        return None, f"dropped_category:{cat}"
    key = crop_set_key(crops)
    if key in merge:
        return str(merge[key]), "merged"
    return None, "multicrop_unmerged"


def build(config_path: Path | None = None, save: bool = True) -> gpd.GeoDataFrame:
    cfg = load_config(config_path)
    f_out, f_map, f_excl = out_paths()
    gdf = gpd.read_parquet(F_POLY)
    print(f"loaded {len(gdf):,} labelled polygons")

    # --- raw label per parcel (category policy + merge map) ---
    lab = gdf.apply(lambda r: assign_raw_label(r, cfg), axis=1)
    gdf["label_raw"] = [t[0] for t in lab]
    gdf["label_reason"] = [t[1] for t in lab]

    # --- config-driven class surgery: rename first, then drop whole classes ---
    relabel = cfg.get("relabel") or {}
    if relabel:
        gdf["label_raw"] = gdf["label_raw"].replace(relabel)
    drop_classes = set(cfg.get("drop_classes") or [])
    if drop_classes:
        dropped = gdf["label_raw"].isin(drop_classes)
        gdf.loc[dropped, "label_reason"] = "dropped_class"
        gdf.loc[dropped, "label_raw"] = None

    # --- hard gates: area (A5) + year (A1: trusted as-is, but must exist) ---
    gdf["gate_area_ok"] = gdf["area_ha"].between(cfg["area_min_ha"], cfg["area_max_ha"])
    gdf["gate_year_ok"] = gdf["year"].notna() if cfg.get("require_year", True) else True

    excl = []

    def _tally(mask: pd.Series, reason: str) -> None:
        excl.append({"reason": reason, "n_parcels": int(mask.sum())})

    _tally(gdf["label_raw"].isna() & gdf["label_reason"].str.startswith("dropped_category"),
           "dropped_category (land_prep/unspecified)")
    _tally(gdf["label_reason"].eq("multicrop_unmerged"), "multicrop not in merge map")
    _tally(gdf["label_reason"].eq("dropped_class"), "dropped class (config drop_classes)")
    has_label = gdf["label_raw"].notna()
    _tally(has_label & ~gdf["gate_area_ok"], "area outside 0.09-50 ha")
    _tally(has_label & gdf["gate_area_ok"] & ~gdf["gate_year_ok"], "missing year")

    keep = has_label & gdf["gate_area_ok"] & gdf["gate_year_ok"]
    df = gdf[keep].copy()

    # --- vocabulary: crops with enough support; landcover classes always kept ---
    is_landcover = df["label_raw"].isin(LANDCOVER_NAME.values())
    counts = df.loc[~is_landcover, "label_raw"].value_counts()
    vocab = set(counts[counts >= cfg["min_class_parcels"]].index)
    rare = ~is_landcover & ~df["label_raw"].isin(vocab)
    if cfg["rare_policy"] == "other":
        df["label"] = np.where(rare, "other", df["label_raw"])
    else:  # drop
        _tally(rare, f"rare crop (<{cfg['min_class_parcels']}) dropped")
        df = df[~rare].copy()
        df["label"] = df["label_raw"]

    classes = sorted(df["label"].unique())
    label_map = {c: i for i, c in enumerate(classes)}
    df["label_id"] = df["label"].map(label_map).astype(int)

    # --- static columns for downstream stages ---
    cen = df.geometry.representative_point()
    df["centroid_lon"], df["centroid_lat"] = cen.x, cen.y
    df["n_pixels_est"] = df["area_ha"] / PIXEL_HA
    df["year"] = df["year"].astype(int)
    df["crop_set"] = df["crops"].map(crop_set_key)
    # coverage columns arrive with the stage-1 GEE pass; NA = "not yet measured".
    df["n_valid_obs"] = pd.array([pd.NA] * len(df), dtype="Int64")
    df["max_gap"] = pd.array([pd.NA] * len(df), dtype="Int64")
    df["quality_ok"] = pd.array([pd.NA] * len(df), dtype="boolean")

    out = df[["COD_PREDIO", "label", "label_id", "label_reason", "crop_set", "year",
              "area_ha", "n_pixels_est", "centroid_lon", "centroid_lat",
              "n_valid_obs", "max_gap", "quality_ok", "geometry"]].reset_index(drop=True)

    # --- report ---
    print(f"\neligible parcels: {len(out):,}  ({len(classes)} classes)")
    print(out["label"].value_counts().to_string())
    print(f"\nmerged intercrop parcels kept: {(df['label_reason'] == 'merged').sum():,}")
    print("\nexclusions:")
    excl_df = pd.DataFrame(excl)
    print(excl_df.to_string(index=False))

    if save:
        out.to_parquet(f_out, index=False)
        with open(f_map, "w") as f:
            json.dump(label_map, f, indent=2, ensure_ascii=False)
        excl_df.to_csv(f_excl, index=False)
        print(f"\nwrote {f_out}, {f_map.name}, {f_excl.name}")
    return out


def apply_coverage_gate(coverage: pd.DataFrame, config_path: Path | None = None,
                        save: bool = True) -> gpd.GeoDataFrame:
    """Join the stage-1 coverage pass into the label table and finalise ``quality_ok``.

    ``coverage`` needs columns ``COD_PREDIO, n_valid_obs, max_gap``. Parcels absent from
    ``coverage`` keep NA (not yet measured).
    """
    cfg = load_config(config_path)
    f_out, _, _ = out_paths()
    df = gpd.read_parquet(f_out)
    df = df.drop(columns=["n_valid_obs", "max_gap", "quality_ok"]).merge(
        coverage[["COD_PREDIO", "n_valid_obs", "max_gap"]], on="COD_PREDIO", how="left")
    df["n_valid_obs"] = df["n_valid_obs"].astype("Int64")
    df["max_gap"] = df["max_gap"].astype("Int64")
    df["quality_ok"] = (df["n_valid_obs"] >= cfg["min_valid_obs"]).astype("boolean")
    df.loc[df["n_valid_obs"].isna(), "quality_ok"] = pd.NA
    n = df["quality_ok"]
    print(f"coverage gate: {int((n == True).sum()):,} pass, "  # noqa: E712
          f"{int((n == False).sum()):,} fail, {int(n.isna().sum()):,} unmeasured")  # noqa: E712
    if save:
        df.to_parquet(f_out, index=False)
        print(f"updated {f_out}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build modeling_parcels.parquet (plan §7)")
    ap.add_argument("--config", type=Path, default=None, help="path to data.yaml")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    build(config_path=args.config, save=not args.no_save)


if __name__ == "__main__":
    main()

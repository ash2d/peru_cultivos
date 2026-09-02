"""Build the ``(polygon, crop, time)`` training table for the Piura crop classifier.

Scripted version of ``notebooks/03_pett_crop_polygon.ipynb``, over the real-key join chain
(no name matching):

    BD SSET  ──CodigoSSET──►  grafica_tabular_Piura.dta  ──COD_PREDIO──►  qgis polygons
    (crop, registration date)      (bridge, best coverage)                   (geometry)

plus crop-label cleaning from :mod:`crop_classifier.crop_normalization`.

Outputs (``data/processed/``): ``training_crop_polygon.parquet`` (one row per polygon:
geometry, crop list, category list, contributing keys, representative + full year set,
area, counts), ``training_crop_records.parquet`` (exploded long form), and
``crop_normalization_map.csv`` (every distinct raw label -> its normalised crops).

Decisions: bridge = ``grafica_tabular_Piura.dta`` (~69 % of SSET crop keys; the others are
subsets or lack ``CodigoSSET``); nothing dropped by crop type, only flagged by
``category``; multiple crops per polygon kept as a list; ``year`` from
``FECHA EMPADRONAMIENTO`` is an approximate label year, not a scene selector; SSET ``AREA``
(m², dirty) non-positive nulled, polygon ``AREA_S_HA`` carried as the reliable area.

    uv run python -m crop_classifier.build_training_data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import pyreadstat

from crop_classifier.crop_normalization import normalize_label

# --- Paths (repo-root relative) ---
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

F_SSET = RAW / "BD SSET(MOQUEGUA-PASCO-PIURA).xlsx"
F_BRIDGE = RAW / "grafica_tabular_Piura.dta"
F_POLY = RAW / "qgis_stefany" / "CATASTRO_CENAGRO_PIURA_WGS84_Z17S_FINAL.shp"

# Registration years outside this window are data errors (typo/stub in FECHA EMPADRONAMIENTO).
YEAR_MIN, YEAR_MAX = 1990, 2020

VEG_CATEGORIES = frozenset({"crop", "pasture"})


# --- Loaders ---
def load_sset(path: Path = F_SSET) -> pd.DataFrame:
    """One row per Piura BD SSET declaration: canonical string ``CodigoSSET``, cleaned
    ``area_m2`` (non-positive -> NaN), derived ``year`` (implausible -> NaN), raw ``crop_raw``.
    """
    df = pd.read_excel(
        path,
        sheet_name="DATOS",
        usecols=["DEPARTAMENTO", "CULTIVO", "FECHA EMPADRONAMIENTO", "AREA", "Codigo SSET",
                 "NOMBRES"],
    )
    df = df[df.DEPARTAMENTO == "PIURA"].copy()
    df = df.rename(columns={"Codigo SSET": "CodigoSSET", "FECHA EMPADRONAMIENTO": "fecha",
                            "CULTIVO": "crop_raw", "AREA": "area_m2", "NOMBRES": "owner"})

    # CodigoSSET is int64 here, string on the bridge/polygon side -> canonicalise.
    df["CodigoSSET"] = df["CodigoSSET"].astype("Int64").astype(str).str.strip()

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["year"] = df["fecha"].dt.year
    df.loc[~df["year"].between(YEAR_MIN, YEAR_MAX), "year"] = np.nan
    df["year"] = df["year"].astype("Int64")

    # area m^2 is dirty: null non-positive values
    df["area_m2"] = pd.to_numeric(df["area_m2"], errors="coerce")
    df.loc[df["area_m2"] <= 0, "area_m2"] = np.nan

    df = df[df["crop_raw"].notna()].copy()
    print(f"  SSET Piura crop declarations: {len(df):,} "
          f"over {df.CodigoSSET.nunique():,} distinct CodigoSSET")
    return df[["CodigoSSET", "crop_raw", "year", "fecha", "area_m2", "owner"]]


def load_bridge(path: Path = F_BRIDGE) -> pd.DataFrame:
    """Load the ``CodigoSSET ↔ COD_PREDIO`` bridge (unique, non-null pairs)."""
    b, _ = pyreadstat.read_dta(str(path), usecols=["COD_PREDIO", "CodigoSSET"],
                               encoding="latin1")
    for col in ("COD_PREDIO", "CodigoSSET"):
        b[col] = b[col].astype(str).str.strip().replace("nan", np.nan)
    b = b.dropna().drop_duplicates()
    print(f"  bridge pairs: {len(b):,} "
          f"({b.CodigoSSET.nunique():,} CodigoSSET → {b.COD_PREDIO.nunique():,} COD_PREDIO)")
    return b


def load_polygons(path: Path = F_POLY) -> gpd.GeoDataFrame:
    """Parcel polygons keyed by ``COD_PREDIO``, reprojected to EPSG:4326 for GEE. Drops
    null-code polygons and dissolves duplicate codes to one geometry."""
    poly = pyogrio.read_dataframe(str(path), columns=["COD_PREDIO", "AREA_S_HA"])
    poly = poly[poly.COD_PREDIO.notna()].copy()
    poly["COD_PREDIO"] = poly.COD_PREDIO.astype(str).str.strip()
    if poly.COD_PREDIO.duplicated().any():
        poly = poly.dissolve(by="COD_PREDIO", aggfunc="first").reset_index()
    poly = poly.rename(columns={"AREA_S_HA": "area_ha"}).to_crs(4326)
    print(f"  polygons: {len(poly):,} unique COD_PREDIO · CRS {poly.crs}")
    return poly


# --- Crop-label normalisation applied to the SSET table ---
def explode_crops(sset: pd.DataFrame) -> pd.DataFrame:
    """Explode each declaration's raw label into one row per normalised crop. Rows yielding
    no crop token (pure noise) are dropped — never a recognised crop type."""
    label_map = {lab: normalize_label(lab) for lab in sset.crop_raw.unique()}
    sset = sset.copy()
    sset["_pairs"] = sset.crop_raw.map(label_map)
    sset = sset[sset["_pairs"].map(len) > 0].explode("_pairs", ignore_index=True)
    if sset.empty:
        # real for the smallest departments (Callao: 404 parcels); without this the empty
        # `_pairs` infers float dtype and `.str` raises mid-run.
        sset["crop"] = pd.Series(dtype="object")
        sset["category"] = pd.Series(dtype="object")
        return sset.drop(columns="_pairs")
    sset["crop"] = sset["_pairs"].str[0]
    sset["category"] = sset["_pairs"].str[1]
    return sset.drop(columns="_pairs")


def normalization_audit(sset_raw: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct raw ``CULTIVO`` label → its normalised crops, for inspection."""
    counts = sset_raw.crop_raw.value_counts()
    rows = []
    for label, n in counts.items():
        pairs = normalize_label(label)
        rows.append({
            "crop_raw": label,
            "n_rows": int(n),
            "crops": [c for c, _ in pairs],
            "categories": [cat for _, cat in pairs],
            "n_crops": len(pairs),
        })
    return pd.DataFrame(rows)


# --- Join chain + aggregation ---
def _mode_year(years: pd.Series) -> int | float:
    """Representative (most frequent, ties → earliest) non-null year for a polygon."""
    y = years.dropna()
    if y.empty:
        return pd.NA
    counts = y.value_counts()
    return int(counts[counts == counts.max()].index.min())


def build(save: bool = True) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full pipeline and (optionally) write the three outputs."""
    print("Loading sources …")
    sset_raw = load_sset()
    bridge = load_bridge()
    poly = load_polygons()
    poly_ids = set(poly.COD_PREDIO)

    print("Normalising crop labels …")
    records = explode_crops(sset_raw)

    print("Joining SSET → bridge → polygons (real keys only) …")
    linked = records.merge(bridge, on="CodigoSSET", how="inner")
    linked = linked[linked.COD_PREDIO.isin(poly_ids)].copy()
    linked = linked.sort_values(["COD_PREDIO", "year", "crop"]).reset_index(drop=True)

    # ---- Long / record grain: one row per (polygon, CodigoSSET, crop, category, year) ----
    records_out = (
        linked[["COD_PREDIO", "CodigoSSET", "crop", "category", "year", "fecha",
                "area_m2", "crop_raw", "owner"]]
        .drop_duplicates(["COD_PREDIO", "CodigoSSET", "crop", "year"])
        .reset_index(drop=True)
    )

    # ---- Polygon grain: crops as a list, aggregated per COD_PREDIO ----
    # de-dup (crop, category) per polygon in alphabetical order so the two list columns align
    crop_cat = (
        linked[["COD_PREDIO", "crop", "category"]]
        .drop_duplicates()
        .sort_values(["COD_PREDIO", "crop"])
        .groupby("COD_PREDIO")
        .agg(crops=("crop", list), crop_categories=("category", list))
    )
    other = linked.groupby("COD_PREDIO").agg(
        codigo_sset=("CodigoSSET", lambda s: sorted(set(s))),
        years=("year", lambda s: sorted({int(y) for y in s.dropna()})),
        year=("year", _mode_year),
        n_records=("crop_raw", "size"),
    )
    agg = crop_cat.join(other).reset_index()
    agg["n_labels"] = agg.crops.map(len)
    agg["is_vegetated"] = agg.crop_categories.map(
        lambda cats: any(c in VEG_CATEGORIES for c in cats)
    )

    polygons_out = (
        poly.merge(agg, on="COD_PREDIO", how="inner")
        [["COD_PREDIO", "crops", "crop_categories", "n_labels", "is_vegetated",
          "codigo_sset", "year", "years", "n_records", "area_ha", "geometry"]]
    )

    _report(polygons_out, records_out)

    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        audit = normalization_audit(sset_raw)
        p_poly = OUT / "training_crop_polygon.parquet"
        p_rec = OUT / "training_crop_records.parquet"
        p_map = OUT / "crop_normalization_map.csv"
        polygons_out.to_parquet(p_poly, index=False)
        records_out.to_parquet(p_rec, index=False)
        audit.to_csv(p_map, index=False)
        for p in (p_poly, p_rec, p_map):
            print(f"  wrote {p.relative_to(ROOT)}  ({p.stat().st_size / 1e6:.1f} MB)")

    return polygons_out, records_out, sset_raw


def _report(polygons: gpd.GeoDataFrame, records: pd.DataFrame) -> None:
    """Print a compact summary of what was built (and the key caveats)."""
    print("\n=== result ===")
    print(f"  polygons with ≥1 crop label : {len(polygons):,}")
    print(f"  crop records (exploded)     : {len(records):,}")
    print(f"  polygons with >1 crop label : {(polygons.n_labels > 1).sum():,} "
          f"({(polygons.n_labels > 1).mean():.1%})")
    print(f"  polygons with vegetation    : {polygons.is_vegetated.sum():,} "
          f"({polygons.is_vegetated.mean():.1%})  (rest are pure fallow/land-prep)")
    ny = polygons.years.map(len)
    print(f"  polygons with >1 crop-year  : {(ny > 1).sum():,} ({(ny > 1).mean():.2%}) "
          f"→ effectively one label year per parcel")

    cats = records.category.value_counts()
    print("\n  crop records by category:")
    for cat, n in cats.items():
        print(f"    {cat:12s} {n:8,d}")

    top = records[records.category == "crop"].crop.value_counts().head(12)
    print("\n  top crop labels (category=crop):")
    for crop, n in top.items():
        print(f"    {crop:22s} {n:7,d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-save", action="store_true",
                        help="run the pipeline but do not write output files")
    args = parser.parse_args()
    build(save=not args.no_save)


if __name__ == "__main__":
    main()

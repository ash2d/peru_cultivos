"""Build the ``(polygon, crop, time)`` training table for the Piura crop classifier.

This is the scripted, reproducible version of ``notebooks/03_pett_crop_polygon.ipynb``. It
follows the same reliable, real-key join chain (no fragile name matching):

    BD SSET  ──CodigoSSET──►  grafica_tabular_Piura.dta  ──COD_PREDIO──►  qgis polygons
    (crop, registration date)      (bridge, best coverage)                   (geometry)

and adds the crop-label cleaning from :mod:`crop_classifier.crop_normalization` so the free
-text ``CULTIVO`` field becomes a tidy, normalised list of crops per parcel, ready to attach
Landsat features to.

Outputs (in ``data/processed/``):

* ``training_crop_polygon.parquet`` — **the deliverable**: one row per polygon
  (``COD_PREDIO``) with its geometry, a **list of normalised crop names**, the aligned
  category of each, the contributing ``CodigoSSET`` keys, a representative year and the full
  set of years, parcel area (ha) and record counts.
* ``training_crop_records.parquet`` — the exploded long form: one row per
  ``(COD_PREDIO, CodigoSSET, crop, category, year)`` with the original raw label kept, for
  auditing and for any per-record modelling.
* ``crop_normalization_map.csv`` — every distinct raw ``CULTIVO`` label → its normalised
  crop list + category + row count, so every cleaning decision is inspectable.

Decisions made here (all intentionally conservative — see the module docstring of
``crop_normalization`` for the label logic):

* **Bridge = ``grafica_tabular_Piura.dta``** (covers ~69% of SSET crop keys; the ``catastro``
  bridge is a strict subset and ``qgis/PIURA.dta`` carries no ``CodigoSSET``).
* **Nothing is dropped by crop type.** Fallow / land-prep / pasture / unspecified survive as
  tokens, only *flagged* by ``category`` so the modeller can filter later.
* **Multiple crops per polygon are kept as a list** (both intercrop labels like
  ``CAFE Y PLATANO`` and different declarations across a parcel's records are unioned).
* **``year`` is the titling/registration year** derived from ``FECHA EMPADRONAMIENTO`` — it is
  unreliable as a growing season (batch/stub dates) and in practice collapses onto the
  ~1998–99 titling wave; treat it as an approximate label year, not a scene selector.
* **Areas are cleaned**: SSET ``AREA`` (m², dirty) has non-positive values nulled; the
  parcel's ``AREA_S_HA`` (hectares, from the polygon layer) is carried as the reliable area.

Run with::

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

# --------------------------------------------------------------------------------------
# Paths (repo-root relative, so the script runs from anywhere)
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

F_SSET = RAW / "BD SSET(MOQUEGUA-PASCO-PIURA).xlsx"
F_BRIDGE = RAW / "grafica_tabular_Piura.dta"
F_POLY = RAW / "qgis_stefany" / "CATASTRO_CENAGRO_PIURA_WGS84_Z17S_FINAL.shp"

# Registration years outside this window are treated as data errors and nulled. The real
# titling waves are ~1996–2019; anything else is a typo/stub in FECHA EMPADRONAMIENTO.
YEAR_MIN, YEAR_MAX = 1990, 2020

# Categories that count as actual vegetation (a usable classifier label).
VEG_CATEGORIES = frozenset({"crop", "pasture"})


# --------------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------------
def load_sset(path: Path = F_SSET) -> pd.DataFrame:
    """Load the Piura BD SSET crop declarations with a crop label.

    Returns one row per declaration, with a canonicalised string ``CodigoSSET``, a cleaned
    ``area_m2`` (non-positive -> NaN), a derived ``year`` (implausible -> NaN) and the raw
    ``crop_raw`` label kept for the audit trail.
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

    # CodigoSSET is int64 here but a string on the bridge/polygon side -> canonicalise.
    df["CodigoSSET"] = df["CodigoSSET"].astype("Int64").astype(str).str.strip()

    # Year from an unreliable registration date; null impossible years.
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["year"] = df["fecha"].dt.year
    df.loc[~df["year"].between(YEAR_MIN, YEAR_MAX), "year"] = np.nan
    df["year"] = df["year"].astype("Int64")

    # Area in m^2 is dirty: null non-positive values (0 / negatives are not real parcels).
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
    """Load parcel polygons keyed by ``COD_PREDIO`` (EPSG:4326).

    Drops null-code polygons, collapses duplicate codes to one geometry, and reprojects to
    WGS84 lon/lat for downstream Earth Engine / Landsat work.
    """
    poly = pyogrio.read_dataframe(str(path), columns=["COD_PREDIO", "AREA_S_HA"])
    poly = poly[poly.COD_PREDIO.notna()].copy()
    poly["COD_PREDIO"] = poly.COD_PREDIO.astype(str).str.strip()
    if poly.COD_PREDIO.duplicated().any():
        poly = poly.dissolve(by="COD_PREDIO", aggfunc="first").reset_index()
    poly = poly.rename(columns={"AREA_S_HA": "area_ha"}).to_crs(4326)
    print(f"  polygons: {len(poly):,} unique COD_PREDIO · CRS {poly.crs}")
    return poly


# --------------------------------------------------------------------------------------
# Crop-label normalisation applied to the SSET table
# --------------------------------------------------------------------------------------
def explode_crops(sset: pd.DataFrame) -> pd.DataFrame:
    """Explode each declaration's raw label into one row per normalised crop.

    A distinct-label cache keeps this fast (labels repeat heavily). Rows whose label yields
    no crop token at all (pure noise) are dropped — that removes *unparseable* cells, never a
    recognised crop type.
    """
    label_map = {lab: normalize_label(lab) for lab in sset.crop_raw.unique()}
    sset = sset.copy()
    sset["_pairs"] = sset.crop_raw.map(label_map)
    sset = sset[sset["_pairs"].map(len) > 0].explode("_pairs", ignore_index=True)
    if sset.empty:
        # No parseable crop anywhere — real for the smallest departments (Callao has 404
        # parcels). Without this the empty `_pairs` column infers float dtype and `.str`
        # raises, so an ordinary "nothing here" turns into a crash mid-run.
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


# --------------------------------------------------------------------------------------
# Join chain + aggregation
# --------------------------------------------------------------------------------------
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

    # ---- Polygon grain: crops as a list, everything aggregated per COD_PREDIO ----
    # De-duplicate (crop, category) within a polygon, keeping a stable alphabetical order so
    # the two list columns stay aligned.
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

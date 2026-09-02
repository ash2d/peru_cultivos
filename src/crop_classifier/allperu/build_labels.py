"""All-Peru ``(polygon, crop, year)`` build — the Chain-A join run per department.

Same join logic and label normalisation as ``crop_classifier.build_training_data`` (the
Piura-only reference); this loops over :func:`allperu.sources.departments` and carries a
``dept`` column. Per-department caches (``cache/sset_*.parquet``, ``cache/bridge_*.parquet``)
make re-runs cheap — reading a 200 MB xlsx costs minutes.

Outputs (``data/processed/all_peru_full/``) mirror the Piura build: ``training_crop_polygon``
/ ``training_crop_records`` (+ ``dept``), ``crop_normalization_map.csv``, and
``build_report.csv`` (per-department attrition at every join stage).

    uv run python -m crop_classifier.allperu.build_labels
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import pyreadstat
import shapely

from crop_classifier.allperu.sources import (
    Dept,
    _norm,
    departments,
    shapefile_view,
)
from crop_classifier.build_training_data import (
    VEG_CATEGORIES,
    YEAR_MAX,
    YEAR_MIN,
    _mode_year,
    explode_crops,
    normalization_audit,
)

ROOT = Path(__file__).resolve().parents[3]
# The FULL build has its own workspace; ``all_peru`` holds the Piura-scale sample everything
# downstream trains on — kept apart so the sample can be redrawn without re-reading 500 MB
# of spreadsheets.
OUT = ROOT / "data" / "processed" / "all_peru_full"
CACHE = OUT / "cache"
SHP_VIEW = CACHE / "shp"

SSET_COLS = ["DEPARTAMENTO", "CULTIVO", "FECHA EMPADRONAMIENTO", "AREA", "Codigo SSET",
             "NOMBRES"]


def canon_key(s: pd.Series) -> pd.Series:
    """Canonical ``CodigoSSET``: strip whitespace *and leading zeros*.

    The biggest gotcha in the all-Peru drop: most bridge files zero-pad the key to 9 chars
    (``030406693``) while BD SSET stores it unpadded (``30406693``), so a naive string join
    returns zero matches and reads as "no linkable data". Ancash: 0 -> 369,089 keys. Piura
    is unaffected (both sides already 9 digits).
    """
    s = s.astype(str).str.strip()
    stripped = s.str.lstrip("0")
    # an all-zeros key collapses to "" under lstrip; keep it as "0"
    stripped = stripped.where(stripped.ne("") | s.eq(""), "0")
    return stripped.replace({"": np.nan, "nan": np.nan, "<NA>": np.nan, "None": np.nan})


# --- Loaders (cached) ---
def load_sset_workbook(path: Path) -> pd.DataFrame:
    """Read one BD SSET workbook (all its departments), cached to parquet.

    Reads every ``DATOS*`` sheet — the Arequipa/Ayacucho/Cajamarca workbook is split across
    ``DATOS1``/``DATOS2`` (Excel's row limit), and assuming one ``DATOS`` loses a department.
    """
    cache = CACHE / f"sset_{path.stem.replace(' ', '_')}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    t = time.time()
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True)
    sheets = [s for s in wb.sheetnames if s.upper().startswith("DATOS")]
    wb.close()
    if not sheets:
        raise ValueError(f"{path.name}: no DATOS* sheet (found {wb.sheetnames})")
    parts = [pd.read_excel(path, sheet_name=s, usecols=SSET_COLS) for s in sheets]
    df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    # `Codigo SSET` is int64 in some workbooks, str in others -> canonicalise to str.
    # Int64 first so 1.0e9 floats do not become "1000000000.0".
    key = df["Codigo SSET"]
    if not pd.api.types.is_object_dtype(key):
        key = key.astype("Int64")
    df["Codigo SSET"] = canon_key(key)
    df["DEPARTAMENTO"] = df["DEPARTAMENTO"].astype(str).str.strip().str.upper()
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    print(f"    read {path.name} sheets={sheets} ({len(df):,} rows) "
          f"in {time.time() - t:.0f}s -> cached")
    return df


def build_dept_sset_caches(depts: list[Dept]) -> None:
    """One pass over every workbook -> one crop-declaration cache per department.

    A department's rows are not confined to the workbook named after it (Lima spills across
    three; Ayacucho, La Libertad, Lambayeque also), so every workbook is scanned.
    """
    wanted = {d.sset_key for d in depts}
    if all((CACHE / f"sset_dept_{d.name}.parquet").exists() for d in depts):
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    parts: dict[str, list[pd.DataFrame]] = {k: [] for k in wanted}
    for wb_path in sorted({d.sset for d in depts}):
        wb = load_sset_workbook(wb_path)
        key = wb["DEPARTAMENTO"].map(_norm)
        for k, grp in wb[key.isin(wanted)].groupby(key[key.isin(wanted)]):
            parts[k].append(grp)
        del wb
    by_key = {d.sset_key: d.name for d in depts}
    for k, frames in parts.items():
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=SSET_COLS)
        df.to_parquet(CACHE / f"sset_dept_{by_key[k]}.parquet", index=False)
        if len(frames) > 1:
            print(f"    {by_key[k]}: {len(df):,} declarations from "
                  f"{len(frames)} workbooks")


def sset_for(dept: Dept) -> pd.DataFrame:
    """This department's crop declarations, cleaned exactly as the Piura loader does."""
    cache = CACHE / f"sset_dept_{dept.name}.parquet"
    if not cache.exists():
        build_dept_sset_caches([dept])
    df = pd.read_parquet(cache).copy()
    df = df.rename(columns={"Codigo SSET": "CodigoSSET",
                            "FECHA EMPADRONAMIENTO": "fecha", "CULTIVO": "crop_raw",
                            "AREA": "area_m2", "NOMBRES": "owner"})
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["year"] = df["fecha"].dt.year
    df.loc[~df["year"].between(YEAR_MIN, YEAR_MAX), "year"] = np.nan
    df["year"] = df["year"].astype("Int64")
    df["area_m2"] = pd.to_numeric(df["area_m2"], errors="coerce")
    df.loc[df["area_m2"] <= 0, "area_m2"] = np.nan
    df = df[df["crop_raw"].notna()].copy()
    return df[["CodigoSSET", "crop_raw", "year", "fecha", "area_m2", "owner"]]


def bridge_for(dept: Dept) -> pd.DataFrame:
    """``CodigoSSET <-> COD_PREDIO`` pairs for one department, cached."""
    cache = CACHE / f"bridge_{dept.name}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    b, _ = pyreadstat.read_dta(str(dept.bridge),
                               usecols=["COD_PREDIO", "CodigoSSET"], encoding="latin1")
    b["COD_PREDIO"] = b["COD_PREDIO"].astype(str).str.strip().replace("nan", np.nan)
    b["CodigoSSET"] = canon_key(b["CodigoSSET"])
    b = b.dropna().drop_duplicates()
    CACHE.mkdir(parents=True, exist_ok=True)
    b.to_parquet(cache, index=False)
    return b


def polygons_for(dept: Dept) -> gpd.GeoDataFrame:
    """Parcel polygons keyed by ``COD_PREDIO``, EPSG:4326.

    Via :func:`shapefile_view` — most departments ship the attribute table under the wrong
    basename, and without it the read silently returns no columns.
    """
    shp = shapefile_view(dept, SHP_VIEW)
    fields = set(pyogrio.read_info(str(shp))["fields"])
    if "COD_PREDIO" not in fields:
        raise KeyError(f"{dept.name}: shapefile has no COD_PREDIO (fields={sorted(fields)})")
    area_col = next((c for c in ("AREA_S_HA", "AREA_HA") if c in fields), None)
    cols = ["COD_PREDIO"] + ([area_col] if area_col else [])
    poly = pyogrio.read_dataframe(str(shp), columns=cols)
    poly = poly[poly.COD_PREDIO.notna()].copy()
    poly["COD_PREDIO"] = poly.COD_PREDIO.astype(str).str.strip()
    poly = poly[poly.COD_PREDIO != ""]
    if poly.COD_PREDIO.duplicated().any():
        poly = poly.dissolve(by="COD_PREDIO", aggfunc="first").reset_index()
    if area_col:
        poly = poly.rename(columns={area_col: "area_ha"})
    else:
        poly["area_ha"] = np.nan
    poly = poly.to_crs(4326)
    poly["geometry"] = clean_geometry(poly.geometry)
    return poly[~poly.geometry.is_empty & poly.geometry.notna()]


def clean_geometry(geom: gpd.GeoSeries) -> gpd.GeoSeries:
    """Force 2D and repair self-intersections.

    ⚠️ Z coordinates are the trap: La Libertad's cadastre has 3D polygons (5,411 sampled
    parcels), and Earth Engine's GeoJSON validator rejects them with a bare
    ``EEException`` — shapely and geopandas accept them, so nothing complains until a
    multi-hour extraction dies partway through. ``make_valid`` also repairs 19
    self-intersecting rings; non-areal degenerates are dropped by the caller.
    """
    out = shapely.force_2d(geom.values)
    invalid = ~shapely.is_valid(out)
    if invalid.any():
        out[invalid] = shapely.make_valid(out[invalid])
    return gpd.GeoSeries(out, index=geom.index, crs=geom.crs)


# --- Per-department build ---
def build_dept(dept: Dept) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict]:
    """Run the whole chain for one department. Returns (polygons, records, report row)."""
    t0 = time.time()
    sset = sset_for(dept)
    bridge = bridge_for(dept)
    poly = polygons_for(dept)
    poly_ids = set(poly.COD_PREDIO)

    records = explode_crops(sset)
    linked = records.merge(bridge, on="CodigoSSET", how="inner")
    linked = linked[linked.COD_PREDIO.isin(poly_ids)].copy()
    rep = {
        "dept": dept.name,
        "sset_declarations": len(sset),
        "sset_keys": sset.CodigoSSET.nunique(),
        "bridge_pairs": len(bridge),
        "polygons_total": len(poly),
        "linked_records": len(linked),
        "linked_polygons": linked.COD_PREDIO.nunique(),
        "key_coverage": round(
            linked.CodigoSSET.nunique() / max(sset.CodigoSSET.nunique(), 1), 4),
        "seconds": round(time.time() - t0, 1),
    }
    if linked.empty:
        print(f"  {dept.name:14s} NO LINKED RECORDS")
        return gpd.GeoDataFrame(), pd.DataFrame(), rep

    linked = linked.sort_values(["COD_PREDIO", "year", "crop"]).reset_index(drop=True)
    records_out = (
        linked[["COD_PREDIO", "CodigoSSET", "crop", "category", "year", "fecha",
                "area_m2", "crop_raw", "owner"]]
        .drop_duplicates(["COD_PREDIO", "CodigoSSET", "crop", "year"])
        .reset_index(drop=True)
    )
    records_out["dept"] = dept.name

    crop_cat = (linked[["COD_PREDIO", "crop", "category"]].drop_duplicates()
                .sort_values(["COD_PREDIO", "crop"]).groupby("COD_PREDIO")
                .agg(crops=("crop", list), crop_categories=("category", list)))
    other = linked.groupby("COD_PREDIO").agg(
        codigo_sset=("CodigoSSET", lambda s: sorted(set(s))),
        years=("year", lambda s: sorted({int(y) for y in s.dropna()})),
        year=("year", _mode_year),
        n_records=("crop_raw", "size"),
    )
    agg = crop_cat.join(other).reset_index()
    agg["n_labels"] = agg.crops.map(len)
    agg["is_vegetated"] = agg.crop_categories.map(
        lambda cats: any(c in VEG_CATEGORIES for c in cats))

    polygons_out = poly.merge(agg, on="COD_PREDIO", how="inner")
    polygons_out["dept"] = dept.name
    polygons_out = polygons_out[
        ["COD_PREDIO", "dept", "crops", "crop_categories", "n_labels", "is_vegetated",
         "codigo_sset", "year", "years", "n_records", "area_ha", "geometry"]]

    print(f"  {dept.name:14s} {rep['linked_records']:>8,} records  "
          f"{rep['linked_polygons']:>8,} polygons  "
          f"(key cov {rep['key_coverage']:.1%}, {rep['seconds']:.0f}s)")
    return polygons_out, records_out, rep


def build(save: bool = True, only: list[str] | None = None
          ) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    """Build every linkable department and concatenate."""
    depts = [d for d in departments() if not only or d.name in only]
    print(f"Building {len(depts)} departments …")
    build_dept_sset_caches(depts)
    polys, recs, reps = [], [], []
    for d in depts:
        p, r, rep = build_dept(d)
        reps.append(rep)
        if len(p):
            polys.append(p)
            recs.append(r)
    polygons = gpd.GeoDataFrame(pd.concat(polys, ignore_index=True), crs=4326)
    records = pd.concat(recs, ignore_index=True)
    report = pd.DataFrame(reps)

    # COD_PREDIO embeds the department, so a cross-department collision is a data error
    dup = polygons.COD_PREDIO.duplicated().sum()
    if dup:
        raise ValueError(f"{dup:,} COD_PREDIO collide across departments — the key is not "
                         f"nationally unique, downstream joins would silently merge parcels")

    print(f"\n=== all-Peru result ===\n  polygons {len(polygons):,}   "
          f"records {len(records):,}   departments {polygons.dept.nunique()}")
    print(f"  polygons with >1 crop label : {(polygons.n_labels > 1).mean():.1%}")
    print(f"  polygons with vegetation    : {polygons.is_vegetated.mean():.1%}")
    ny = polygons.years.map(len)
    print(f"  polygons with >1 crop-year  : {(ny > 1).sum():,} ({(ny > 1).mean():.2%})")
    print("\n  year distribution:")
    yr = polygons.year.value_counts().sort_index()
    for y, n in yr.items():
        print(f"    {y}: {n:,}")

    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        polygons.to_parquet(OUT / "training_crop_polygon.parquet", index=False)
        records.to_parquet(OUT / "training_crop_records.parquet", index=False)
        report.to_csv(OUT / "build_report.csv", index=False)
        normalization_audit(records.rename(columns={"crop_raw": "crop_raw"})).to_csv(
            OUT / "crop_normalization_map.csv", index=False)
        print(f"\n  wrote 4 files to {OUT.relative_to(ROOT)}")
    return polygons, records, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--only", nargs="*", help="department folder names, e.g. PIURA TUMBES")
    a = ap.parse_args()
    build(save=not a.no_save, only=a.only)


if __name__ == "__main__":
    main()

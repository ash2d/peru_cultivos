"""Per-parcel climate covariates: mean temperature and total rainfall.

Two products, deliberately kept separate because they differ in what they can safely be
used for:

``parcel_climate_normals.parquet`` — **WorldClim 2.1**, 30 arc-second (~1 km at the
equator), the 1970–2000 climatological normal. Twelve monthly mean-temperature rasters and
twelve monthly total-rainfall rasters, sampled at each parcel and reduced to an annual mean
temperature (°C) and an annual rainfall total (mm/year), with the monthly profile and a
handful of derived seasonality descriptors kept alongside.

``parcel_rainfall_annual.parquet`` — **CHIRPS 2.0**, 0.05° (~5.5 km), one *actual* rainfall
total per parcel **per calendar year**. This is the year-resolved companion, and it is the
one that can be used across years.

⚠️ **The normals are time-invariant, and this project has measured what that costs.** A
feature that takes the same value in 1998 and in 2023 cannot express change, so a model
given one will read stability into a panel whether or not the land was stable
(``docs/RESULTS.md`` §4.4, §5; ``LESSONS.md``). Use ``parcel_climate_normals`` for a
**single-year** classifier — where "this parcel sits in a 1,600 mm/yr valley" is a genuine
covariate for what can grow there — and use ``parcel_rainfall_annual`` for anything applied
across years. If both go in, they must be evaluated on LODO as well as CV, because a 1 km
climate surface is a smooth function of location and therefore a proxy for
``centroid_lat``, whose CV/LODO story is the one CLAUDE.md opens with.

**Why a centroid sample rather than a zonal mean over the polygon.** The largest parcel in
the national table is 50 ha; a single WorldClim 30 arc-second cell is ~86 ha at Peru's
latitudes and a CHIRPS cell ~3,000 ha. **No parcel is larger than one climate pixel**, so a
polygon mean and a centroid sample would return the same number for essentially every
parcel, at a few hundred times the cost. ``build_normals`` verifies this rather than
assuming it, and reports the largest parcel as a fraction of a pixel.

Sources are downloaded once into ``data/raw/`` and are not redistributed:

* WorldClim 2.1 — https://geodata.ucdavis.edu/climate/worldclim/2_1/base/
  (``wc2.1_30s_tavg.zip``, ``wc2.1_30s_prec.zip``), Fick & Hijmans 2017.
* CHIRPS 2.0 annual — https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_annual/tifs/,
  Funk et al. 2015. Read **remotely** with a windowed ``/vsicurl`` request over Peru's
  bounding box; the files are uncompressed and row-striped, so ~320 of 2,000 rows are
  transferred per year and nothing global is stored.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from crop_classifier.paths import ROOT

RAW = ROOT / "data" / "raw" / "worldclim"
OUT_DIR = ROOT / "data" / "processed" / "climate"
PARCELS = ROOT / "data" / "processed" / "all_peru_full" / "modeling_parcels.parquet"

# Peru plus a margin, in EPSG:4326. Used only to window the remote CHIRPS reads.
PERU_BBOX = (-82.0, -19.0, -68.0, -3.0)
CHIRPS_URL = ("/vsicurl/https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
              "global_annual/tifs/chirps-v2.0.{year}.tif")
CHIRPS_NODATA = -9000.0
MONTHS = [f"{m:02d}" for m in range(1, 13)]


# ---------------------------------------------------------------------------------
# WorldClim normals
# ---------------------------------------------------------------------------------
def _unzip(var: str) -> Path:
    """``wc2.1_30s_<var>.zip`` -> a directory of twelve monthly GeoTIFFs."""
    z = RAW / f"wc2.1_30s_{var}.zip"
    if not z.exists():
        raise SystemExit(
            f"missing {z}. Download it once:\n"
            f"  curl -L -o {z} https://geodata.ucdavis.edu/climate/worldclim/2_1/"
            f"base/wc2.1_30s_{var}.zip")
    out = RAW / f"wc2.1_30s_{var}"
    out.mkdir(exist_ok=True)
    if len(list(out.glob("*.tif"))) < 12:
        with zipfile.ZipFile(z) as f:
            f.extractall(out)
    return out


def _sample_points(tif: Path | str, lon: np.ndarray, lat: np.ndarray,
                   nodata_below: float | None = None,
                   search: int = 3) -> np.ndarray:
    """Value of ``tif`` at each (lon, lat), with a nearest-valid-cell fallback.

    A parcel whose centroid lands on a masked cell — the coastal fringe, mostly, where a
    1 km land mask cuts inside a parcel that really is on land — would otherwise come back
    NaN and quietly drop the parcel from any model that uses the column. Rather than accept
    that, the sample falls back to the nearest valid cell inside a ``search``-cell window
    and the count of parcels that needed it is reported.
    """
    import rasterio

    with rasterio.open(str(tif)) as src:
        arr = src.read(1).astype("float64")
        nod = src.nodata
        if nod is not None:
            arr[arr == nod] = np.nan
        if nodata_below is not None:
            arr[arr <= nodata_below] = np.nan
        rows, cols = rasterio.transform.rowcol(src.transform, lon, lat)
        rows = np.clip(np.asarray(rows), 0, src.height - 1)
        cols = np.clip(np.asarray(cols), 0, src.width - 1)
        vals = arr[rows, cols]

        bad = np.isnan(vals)
        if bad.any():
            # expanding square search: cheapest fix that is still deterministic
            for r0, c0, i in zip(rows[bad], cols[bad], np.where(bad)[0]):
                for k in range(1, search + 1):
                    win = arr[max(r0 - k, 0):r0 + k + 1, max(c0 - k, 0):c0 + k + 1]
                    if np.isfinite(win).any():
                        vals[i] = np.nanmean(win)
                        break
    return vals


def build_normals(parcels_path: Path | None = None,
                  out_path: Path | None = None) -> pd.DataFrame:
    """One row per parcel: annual mean temperature, annual rainfall, and the profile."""
    import rasterio

    parcels_path = Path(parcels_path or PARCELS)
    out_path = Path(out_path or OUT_DIR / "parcel_climate_normals.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = pd.read_parquet(parcels_path,
                        columns=["COD_PREDIO", "dept", "area_ha",
                                 "centroid_lon", "centroid_lat"])
    lon = p["centroid_lon"].to_numpy(float)
    lat = p["centroid_lat"].to_numpy(float)
    print(f"{len(p):,} parcels from {parcels_path}")

    tdir, pdir = _unzip("tavg"), _unzip("prec")

    # --- the sanity check the docstring promises: is a parcel smaller than a pixel? ---
    with rasterio.open(str(sorted(tdir.glob('*.tif'))[0])) as src:
        deg = abs(src.transform.a)
    # metres per degree of longitude at Peru's mid-latitude (~ -10 deg)
    px_m = deg * 111_320 * np.cos(np.deg2rad(10.0))
    px_ha = (px_m * deg * 111_320) / 10_000
    print(f"WorldClim cell {deg * 3600:.0f} arc-sec ~ {px_m:.0f} m ~ {px_ha:.0f} ha; "
          f"largest parcel {p.area_ha.max():.1f} ha = "
          f"{p.area_ha.max() / px_ha:.2f} of one cell "
          f"({(p.area_ha > px_ha).sum()} parcels exceed a cell)")

    tmon, pmon = {}, {}
    for m in MONTHS:
        tmon[m] = _sample_points(tdir / f"wc2.1_30s_tavg_{m}.tif", lon, lat)
        pmon[m] = _sample_points(pdir / f"wc2.1_30s_prec_{m}.tif", lon, lat)
        print(f"  month {m}: tavg {np.nanmean(tmon[m]):6.2f} C   "
              f"prec {np.nanmean(pmon[m]):7.1f} mm")

    T = np.column_stack([tmon[m] for m in MONTHS])
    P = np.column_stack([pmon[m] for m in MONTHS])

    out = pd.DataFrame({"COD_PREDIO": p["COD_PREDIO"].values, "dept": p["dept"].values})
    out["tmean_c"] = T.mean(1)                       # annual mean temperature, degC
    out["precip_mm_yr"] = P.sum(1)                   # annual rainfall total, mm/year
    out["tmean_c_warmest_month"] = T.max(1)
    out["tmean_c_coolest_month"] = T.min(1)
    out["t_range_c"] = T.max(1) - T.min(1)           # annual temperature range, degC
    out["precip_mm_wettest_month"] = P.max(1)
    out["precip_mm_driest_month"] = P.min(1)
    # WorldClim BIO15: SD of monthly rainfall as a percentage of the monthly mean. High =
    # one short wet season; low = rain spread through the year.
    with np.errstate(invalid="ignore", divide="ignore"):
        out["precip_seasonality_cv"] = 100 * P.std(1, ddof=0) / np.where(
            P.mean(1) > 0, P.mean(1), np.nan)
        # De Martonne aridity index, P / (T + 10). Below ~10 is arid, above ~40 humid.
        out["aridity_index_dm"] = out["precip_mm_yr"] / (out["tmean_c"] + 10.0)
    out["n_dry_months"] = (P < 50).sum(1)            # months under 50 mm
    for i, m in enumerate(MONTHS):
        out[f"tmean_c_m{m}"] = T[:, i]
    for i, m in enumerate(MONTHS):
        out[f"precip_mm_m{m}"] = P[:, i]

    n_nan = int(out["tmean_c"].isna().sum() + out["precip_mm_yr"].isna().sum())
    print(f"\nunresolved after the nearest-valid-cell fallback: {n_nan}")
    for c in out.columns:
        if out[c].dtype == "float64":
            out[c] = out[c].astype("float32")
    out.to_parquet(out_path, index=False)
    print(f"wrote {out_path} ({len(out):,} x {out.shape[1]})")
    _describe(out)
    return out


def _describe(out: pd.DataFrame) -> None:
    print("\nannual mean temperature (degC) and annual rainfall (mm/year), by department:")
    g = (out.groupby("dept")
         .agg(n=("COD_PREDIO", "size"),
              tmean_c=("tmean_c", "mean"),
              precip_mm_yr=("precip_mm_yr", "mean"),
              precip_p10=("precip_mm_yr", lambda s: s.quantile(0.10)),
              precip_p90=("precip_mm_yr", lambda s: s.quantile(0.90)),
              n_dry_months=("n_dry_months", "mean"))
         .round(1).sort_values("precip_mm_yr"))
    print(g.to_string())


# ---------------------------------------------------------------------------------
# CHIRPS year-resolved rainfall
# ---------------------------------------------------------------------------------
def build_rainfall_annual(years: range | list[int] | None = None,
                          parcels_path: Path | None = None,
                          out_path: Path | None = None) -> pd.DataFrame:
    """One row per parcel, one column per calendar year: rainfall total in mm.

    Wide rather than long on purpose: 727k parcels x 30 years is 22 M rows long, and every
    consumer in this project joins climate onto a parcel table by ``COD_PREDIO``, not onto
    a parcel-year. Melt it if a panel needs it.
    """
    import rasterio
    from rasterio.windows import from_bounds

    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

    years = list(years or range(1996, 2025))
    parcels_path = Path(parcels_path or PARCELS)
    out_path = Path(out_path or OUT_DIR / "parcel_rainfall_annual.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = pd.read_parquet(parcels_path,
                        columns=["COD_PREDIO", "dept", "centroid_lon", "centroid_lat"])
    lon = p["centroid_lon"].to_numpy(float)
    lat = p["centroid_lat"].to_numpy(float)
    out = pd.DataFrame({"COD_PREDIO": p["COD_PREDIO"].values, "dept": p["dept"].values})

    for y in years:
        url = CHIRPS_URL.format(year=y)
        with rasterio.open(url) as src:
            win = from_bounds(*PERU_BBOX, src.transform)
            arr = src.read(1, window=win).astype("float64")
            tr = src.window_transform(win)
        arr[arr <= CHIRPS_NODATA] = np.nan
        rows, cols = rasterio.transform.rowcol(tr, lon, lat)
        rows = np.clip(np.asarray(rows), 0, arr.shape[0] - 1)
        cols = np.clip(np.asarray(cols), 0, arr.shape[1] - 1)
        v = arr[rows, cols]
        bad = np.isnan(v)
        if bad.any():
            for r0, c0, i in zip(rows[bad], cols[bad], np.where(bad)[0]):
                for k in range(1, 4):
                    w = arr[max(r0 - k, 0):r0 + k + 1, max(c0 - k, 0):c0 + k + 1]
                    if np.isfinite(w).any():
                        v[i] = np.nanmean(w)
                        break
        out[f"precip_mm_{y}"] = v.astype("float32")
        print(f"  {y}: mean {np.nanmean(v):7.1f} mm   "
              f"unresolved {int(np.isnan(v).sum())}")

    cols_y = [f"precip_mm_{y}" for y in years]
    out["precip_mm_mean"] = out[cols_y].mean(1).astype("float32")
    out["precip_mm_cv"] = (out[cols_y].std(1) / out[cols_y].mean(1)).astype("float32")
    out.to_parquet(out_path, index=False)
    print(f"\nwrote {out_path} ({len(out):,} x {out.shape[1]})")
    print("\nCHIRPS annual rainfall (mm), national parcel mean by year:")
    print(out[cols_y].mean().round(0).to_string())
    return out

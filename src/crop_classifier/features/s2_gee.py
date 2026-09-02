"""Sentinel-2 extraction for the endpoint-labelling campaign (docs/s2_labelling/plan.md).

Mirrors ``features/landsat_gee.py`` — chunked, cached, resumable — and reuses its
``_retry`` / ``_call_with_deadline`` verbatim, since a silent GEE hang is not an exception.

Differences from the Landsat store:

1. Per-date parcel medians, not raw pixels (LTAE/PSE-LTAE are dropped, so a server-side
   median is ~50x cheaper to export).
2. The parcel is eroded 10 m before reduction to kill edge/mixed pixels; if erosion empties
   the geometry the original is used and ``eroded=False`` recorded.
3. One 24-month window per parcel, centred on its Esri ``imagery_date``. The Aug 1 - Jul 31
   agricultural year containing that date is a sub-interval, so one extraction serves both
   the model features (§5.4) and the annotator's NDVI trace.

Run::

    CC_PROC=data/processed/all_peru CC_FEAT=data/processed/all_peru/features_s2 \\
      uv run python -m crop_classifier.cli allperu s2-extract
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# Shared on purpose: one place for the wall-clock deadline, so a fix reaches every extraction.
from crop_classifier.features.landsat_gee import (  # noqa: F401  (re-exported)
    ChunkTimeout,
    _call_with_deadline,
    _chunk_id,
    _is_split_error,
    _retry,
    init_ee,
)
from crop_classifier.paths import feat

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CSP_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"

# Cloud Score+ threshold. Per-pixel, covers 2015-06 on. The ONLY cloud mask beyond SCL=1:
# over-masking thins the observation series, which RESULTS.md §9.2 measures as harmful.
CS_THRESHOLD = 0.60

# 10 m bands + the two 20 m SWIRs, renamed onto `features.indices.BANDS` so
# `indices.add_indices` works unchanged and the S2 store carries the same 11 channels.
S2_BANDS = {"B2": "B", "B3": "G", "B4": "R", "B8": "NIR", "B11": "SWIR1", "B12": "SWIR2"}

# Per-pixel NDVI, computed server-side before reduction, so its quantiles are meaningful:
# a quantile of a ratio is not the ratio of the quantiles, so the labelling ribbon (a
# p25-p75 spread of greenness across the parcel's pixels) cannot be recovered from band
# medians. Deliberately not named `NDVI`: `indices.add_indices` writes that column from the
# band medians, and the two differ by ~0.005-0.02 (Jensen's inequality). The model features
# keep the band-median NDVI; only the trace uses these.
S2_NDVI_BAND = "NDVIpx"
NDVI_PX_COLS = ("NDVI_px_p25", "NDVI_px_p50", "NDVI_px_p75")
RIBBON_PERCENTILES = (25, 75)

# An unremoved baseline-04.00 offset would be +1000 DN = +0.10 reflectance in every band.
# The harmonisation check is judged against this, not against zero.
UNHARMONISED_STEP = 0.10

# Channels the harmonisation check measures; the verdict is taken from the raw bands only
# (see `harmonisation_check`). NDVI is carried for context.
HARMONISATION_CHANNELS = ("SWIR1", "R", "NIR", "NDVI")
BANDS_FOR_VERDICT = ("SWIR1", "R", "NIR")

ERODE_M = 10.0
MIN_PIXELS = 5          # drop a date with fewer clear eroded pixels
TRACE_MONTHS = 12       # +/- months around imagery_date
REDUCE_SCALE = 10       # metres


def f_pixels() -> Path:
    return feat() / "s2_perdate.parquet"


# --- Windows (§5.4) ---
def ag_year(date) -> tuple[dt.date, dt.date]:
    """The agricultural year Aug 1 - Jul 31 containing ``date`` (one full sierra season)."""
    d = pd.Timestamp(date).date()
    start_year = d.year if d.month >= 8 else d.year - 1
    return dt.date(start_year, 8, 1), dt.date(start_year + 1, 7, 31)


def trace_window(date, months: int = TRACE_MONTHS) -> tuple[dt.date, dt.date]:
    """24 months centred on ``date`` — the annotator's NDVI trace, and the extraction span.

    The agricultural year containing ``date`` is always inside this interval, so one
    extraction serves both.
    """
    d = pd.Timestamp(date)
    return ((d - pd.DateOffset(months=months)).date(),
            (d + pd.DateOffset(months=months)).date())


def window_key(date, months: int = TRACE_MONTHS) -> str:
    """Month-resolution key so parcels sharing a window share a GEE request."""
    lo, hi = trace_window(date, months)
    return f"{lo:%Y%m}_{hi:%Y%m}"


# --- Geometry ---
def erode_parcels(parcels: gpd.GeoDataFrame, erode_m: float = ERODE_M,
                  metric_crs: int = 32718) -> gpd.GeoDataFrame:
    """Shrink each parcel by ``erode_m`` to drop edge/mixed pixels; flag the fallbacks.

    Done offline in a metric CRS so the result is inspectable and the fallback is explicit:
    a parcel the erosion empties keeps its original outline and gets ``eroded=False``.
    """
    out = parcels.copy()
    m = out.geometry.to_crs(metric_crs)
    shrunk = m.buffer(-erode_m)
    ok = ~(shrunk.is_empty | shrunk.isna()) & shrunk.is_valid & (shrunk.area > 0)
    geom = shrunk.where(ok, m).to_crs(4326)
    out["eroded"] = ok.values
    out["geometry"] = geom.values
    return out


# --- Collection ---
def masked_collection(lo: dt.date, hi: dt.date, region, cs_threshold=CS_THRESHOLD):
    """Harmonised, Cloud-Score+-masked S2 collection over ``region`` in ``[lo, hi)``."""
    ee = init_ee()

    def _prep(img):
        # cs_cdf is cumulative clear-sky probability; >=0.60 keeps confidently-clear pixels.
        clear = img.select("cs_cdf").gte(cs_threshold)
        # SCL class 1 = saturated/defective; the only extra mask (see module docstring).
        not_defective = img.select("SCL").neq(1)
        out = (img.select(list(S2_BANDS)).rename(list(S2_BANDS.values()))
               .updateMask(clear.And(not_defective)))
        # per-pixel NDVI, for real quantiles downstream (see S2_NDVI_BAND)
        out = out.addBands(out.normalizedDifference(["NIR", "R"]).rename(S2_NDVI_BAND))
        return out.copyProperties(img, ["system:time_start", "SPACECRAFT_NAME"])

    return (ee.ImageCollection(S2_COLLECTION)
            .filterBounds(region)
            .filterDate(str(lo), str(hi))
            .linkCollection(ee.ImageCollection(CSP_COLLECTION), ["cs_cdf"])
            .map(_prep))


def _to_fc(chunk: gpd.GeoDataFrame):
    ee = init_ee()
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry(geom.__geo_interface__), {"cid": str(cid)})
        for cid, geom in zip(chunk["COD_PREDIO"], chunk.geometry)])


def _fetch_all(fc) -> list[dict]:
    """Paginated fetch — plain ``getInfo`` caps at 5000 elements."""
    ee = init_ee()
    rows, token = [], None
    while True:
        params = {"expression": fc, "pageSize": 5000}
        if token:
            params["pageToken"] = token
        page = _retry(lambda: ee.data.computeFeatures(params))
        rows.extend(f["properties"] for f in page.get("features", []))
        token = page.get("nextPageToken")
        if not token:
            return rows


def perdate_chunk(chunk: gpd.GeoDataFrame, lo: dt.date, hi: dt.date) -> pd.DataFrame:
    """One row per parcel x acquisition date: band medians, NDVI quartiles, pixel count.

    ``median + count + percentile`` are combined with ``sharedInputs=True``, so all come
    from one pass over the pixels. The band percentiles are discarded (only NDVI's are
    used); to keep them, extend the ``keep`` list below.
    """
    ee = init_ee()
    region = ee.Geometry.Rectangle(list(chunk.total_bounds))
    coll = masked_collection(lo, hi, region)
    fc = _to_fc(chunk)
    bands = list(S2_BANDS.values())

    def _reduce(img):
        d = ee.Date(img.get("system:time_start"))
        reducer = (ee.Reducer.median()
                   .combine(ee.Reducer.count(), sharedInputs=True)
                   .combine(ee.Reducer.percentile(list(RIBBON_PERCENTILES)),
                            sharedInputs=True))
        red = img.reduceRegions(collection=fc, reducer=reducer,
                                scale=REDUCE_SCALE, tileScale=4)
        return red.map(lambda f: f.set({
            "date": d.format("YYYY-MM-dd"),
            "doy": d.getRelative("day", "year").add(1)}))

    rows = _fetch_all(coll.map(_reduce).flatten())
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns={"cid": "COD_PREDIO"})
    med = {f"{b}_median": b for b in bands if f"{b}_median" in df.columns}
    df = df.rename(columns=med)
    df = df.rename(columns={f"{S2_NDVI_BAND}_median": "NDVI_px_p50",
                            f"{S2_NDVI_BAND}_p25": "NDVI_px_p25",
                            f"{S2_NDVI_BAND}_p75": "NDVI_px_p75"})
    cnt_cols = [c for c in df.columns if c.endswith("_count")]
    if cnt_cols:
        # every band shares the mask, so the counts agree; take the 10 m one
        df["n_px"] = df[cnt_cols].max(axis=1)
        df = df.drop(columns=cnt_cols)
    keep = ["COD_PREDIO", "date", "doy", "n_px", *bands, *NDVI_PX_COLS]
    df = df[[c for c in keep if c in df.columns]]
    return df.dropna(subset=[b for b in bands if b in df.columns], how="all")


# --- Runner ---
# Largest bbox (deg²) a chunk may span. The chunk bbox becomes the `filterBounds`
# rectangle, and every intersecting S2 granule is reduced per image — so cost scales with
# extent, not parcel count. Grouping by window alone drew parcels from all 14 departments
# and gave near-continental chunks that stalled at 0 % CPU. An S2 granule is ~110 km, so
# ~1 deg² keeps a chunk to a few granule columns. Same lesson as `landsat_gee._chunk_todo`,
# learned twice.
MAX_CHUNK_BBOX_DEG2 = 1.0


def _bbox_area(g: gpd.GeoDataFrame) -> float:
    b = g.total_bounds
    return float((b[2] - b[0]) * (b[3] - b[1]))


def _chunks(parcels: gpd.GeoDataFrame, chunk_size: int,
            max_bbox: float = MAX_CHUNK_BBOX_DEG2) -> list:
    """``[(window_key, lo, hi, chunk), …]`` — one window per chunk, geographically tight.

    Within a window, parcels are visited in Hilbert order and packed greedily; a chunk is
    cut short as soon as adding the next parcel would push its bounding box past
    ``max_bbox``.
    """
    todo = []
    for key, grp in parcels.groupby("_window", sort=True):
        lo, hi = trace_window(grp["imagery_date"].iloc[0])
        try:
            grp = grp.iloc[np.argsort(grp.hilbert_distance().values)]
        except Exception:  # noqa: BLE001 — older geopandas
            grp = grp.sort_values(["centroid_lon", "centroid_lat"])
        start = 0
        for i in range(1, len(grp) + 1):
            piece = grp.iloc[start:i]
            over = _bbox_area(piece) > max_bbox and len(piece) > 1
            if over:
                todo.append((key, lo, hi, grp.iloc[start:i - 1]))
                start = i - 1
            elif i - start >= chunk_size:
                todo.append((key, lo, hi, piece))
                start = i
        if start < len(grp):
            todo.append((key, lo, hi, grp.iloc[start:]))
    return todo


def _run_chunk(chunk, lo, hi, min_size: int = 4) -> pd.DataFrame:
    """Retry transient failures; halve the chunk on a "computation too big" error."""
    try:
        return _retry(lambda: perdate_chunk(chunk, lo, hi))
    except Exception as e:  # noqa: BLE001
        if _is_split_error(e) and len(chunk) > min_size:
            mid = len(chunk) // 2
            print(f"    chunk too big ({str(e)[:70]}) — splitting "
                  f"{len(chunk)} -> {mid}+{len(chunk) - mid}", flush=True)
            return pd.concat([_run_chunk(chunk.iloc[:mid], lo, hi, min_size),
                              _run_chunk(chunk.iloc[mid:], lo, hi, min_size)],
                             ignore_index=True)
        raise


def extract(parcels: gpd.GeoDataFrame, chunk_size: int = 25,
            feat_dir: Path | None = None, min_pixels: int = MIN_PIXELS,
            max_chunks: int | None = None, workers: int = 4) -> pd.DataFrame:
    """Extract per-date parcel medians for every parcel's 24-month window.

    ``parcels`` needs ``COD_PREDIO``, ``geometry`` and ``imagery_date``. Resumable: each
    chunk is a content-addressed parquet under ``<feat>/s2_chunks/``.
    """
    init_ee()
    feat_dir = feat_dir or feat()
    chunk_dir = feat_dir / "s2_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    p = erode_parcels(parcels)
    p["_window"] = [window_key(d) for d in p["imagery_date"]]
    if "centroid_lon" not in p.columns:
        rp = p.geometry.representative_point()
        p["centroid_lon"], p["centroid_lat"] = rp.x, rp.y

    todo = _chunks(p, chunk_size)
    bb = [_bbox_area(c) for *_, c in todo]
    print(f"S2 extraction: {len(p):,} parcels, {p['_window'].nunique()} windows "
          f"-> {len(todo)} chunks (median bbox {np.median(bb):.2f} deg², "
          f"max {max(bb):.2f})")

    pending = [(n, key, lo, hi, chunk) for n, (key, lo, hi, chunk) in enumerate(todo)
               if not (chunk_dir / f"s2_{key}_{_chunk_id(chunk)}.parquet").exists()]
    if max_chunks is not None:
        pending = pending[:max_chunks]
    print(f"  {len(todo) - len(pending)} cached, {len(pending)} to fetch")

    def _one(job):
        n, key, lo, hi, chunk = job
        f = chunk_dir / f"s2_{key}_{_chunk_id(chunk)}.parquet"
        df = _run_chunk(chunk, lo, hi)
        df.to_parquet(f, index=False)
        return f"  [{n + 1}/{len(todo)}] {key}: {len(df):,} parcel-dates"

    if pending:
        # Threads, not processes: every call is IO-bound on GEE. Kept modest — a concurrency
        # breach triggers GEE "Restricted Mode", which arrives as a silent 900 s hang.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for msg in ex.map(_one, pending):
                print(msg, flush=True)

    files = sorted(chunk_dir.glob("s2_*.parquet"))
    if not files:
        return pd.DataFrame()
    out = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    out = _consolidate(out, p, min_pixels)
    out.to_parquet(f_pixels(), index=False)
    print(f"\n{f_pixels().name}: {len(out):,} parcel-dates, "
          f"{out.COD_PREDIO.nunique():,} parcels "
          f"(median {out.groupby('COD_PREDIO').size().median():.0f} clear dates/parcel)")
    return out


def _consolidate(df: pd.DataFrame, parcels: pd.DataFrame,
                 min_pixels: int) -> pd.DataFrame:
    """Merge same-date granule splits, apply the pixel floor, attach the erosion flag.

    A parcel in a tile overlap is reduced once per granule on the same date, giving two
    rows correct over their own part of the parcel. They are combined as a
    pixel-count-weighted mean of the two medians (keeping the larger one would drop real
    observations at tile-edge parcels). The NDVI quartiles are averaged the same way; this
    affects only the ribbon width on the handful of split dates.
    """
    from crop_classifier.features.indices import BANDS

    bands = [b for b in BANDS if b in df.columns]
    vals = bands + [c for c in NDVI_PX_COLS if c in df.columns]
    df = df.dropna(subset=["n_px"])
    df["n_px"] = df["n_px"].astype(float)
    df = df[df["n_px"] > 0]

    dup = df.duplicated(["COD_PREDIO", "date"], keep=False)
    if dup.any():
        def _combine(g):
            w = g["n_px"].values
            out = {b: float(np.average(g[b].values, weights=w)) for b in vals}
            out["n_px"] = float(w.sum())
            out["doy"] = int(g["doy"].iloc[0])
            return pd.Series(out)
        merged = (df[dup].groupby(["COD_PREDIO", "date"], as_index=False)
                  .apply(_combine, include_groups=False).reset_index(drop=True))
        df = pd.concat([df[~dup], merged], ignore_index=True)

    n_before = len(df)
    df = df[df["n_px"] >= min_pixels]
    print(f"  pixel floor n_px >= {min_pixels}: kept {len(df):,} of {n_before:,} "
          f"parcel-dates")

    flags = parcels[["COD_PREDIO", "eroded"]].drop_duplicates("COD_PREDIO")
    df = df.merge(flags, on="COD_PREDIO", how="left")
    return df.sort_values(["COD_PREDIO", "date"]).reset_index(drop=True)


# --- §5.1 harmonisation verification — a 20-min check that guards the whole store ---
def harmonisation_check(px: pd.DataFrame | None = None,
                        out_dir: Path | str = "docs/figures") -> pd.DataFrame:
    """Confirm there is no radiometric step at the 2022-01-25 baseline-04.00 boundary.

    ``_HARMONIZED`` is supposed to remove the +1000 DN offset; RESULTS.md §9.5 is what
    happens when a correction is assumed to work rather than checked.

    Phenology is controlled by construction: the contrast is within parcel and within
    month-of-year, and the identical procedure is run at a placebo cut one year earlier
    (no baseline change) as the noise floor. A naive six-months-before/after mean reports a
    +0.055 NDVI "step" that is entirely seasonal.

    Returns the summary; writes ``s2_harmonisation_check.csv``, the monthly series and a
    figure.
    """
    from crop_classifier.features.indices import add_indices, scale_sr_s2

    if px is None:
        px = pd.read_parquet(f_pixels())
    d = add_indices(scale_sr_s2(px))
    d["date"] = pd.to_datetime(d["date"])
    d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
    d["moy"] = d["date"].dt.month

    cut = pd.Timestamp("2022-01-25")
    placebo = pd.Timestamp("2021-01-25")
    rows = []
    for name, c in (("baseline_04.00", cut), ("placebo_-1y", placebo)):
        for col in HARMONISATION_CHANNELS:
            rows.append({"cut": name, "cut_date": c.date().isoformat(), "channel": col,
                         **_paired_step(d, c, col)})
    summary = pd.DataFrame(rows)

    # Two corrections, neither optional:
    # (1) The estimator has a ~+0.018 residual even with no step (a month's observations
    #     fall on different days across years), so judge `step - placebo`, not `step`.
    # (2) Judge against the failure, not zero: an unremoved +1000 DN offset is +0.10 per band.
    plac = summary[summary.cut == "placebo_-1y"].set_index("channel")["step"]
    summary["placebo_step"] = summary["channel"].map(plac)
    summary["step_corrected"] = summary["step"] - summary["placebo_step"]
    summary["expected_if_unharmonised"] = UNHARMONISED_STEP
    summary["frac_of_failure"] = (summary["step_corrected"].abs()
                                  / summary["expected_if_unharmonised"])
    # Verdict is read off the raw BANDS, never an index: the baseline change is additive in
    # band reflectance, while NDVI also absorbs real year-to-year greenness change (it reads
    # -0.031 here vs bands ~+0.01 — that gap is Peruvian weather, not Sentinel-2).
    summary["diagnostic"] = summary["channel"].isin(BANDS_FOR_VERDICT)
    summary["verdict"] = np.where(
        ~summary["diagnostic"],
        "context only — an index absorbs real interannual change",
        np.where(summary["frac_of_failure"] < 0.25,
                 "harmonised — placebo-corrected step is <1/4 of an unremoved offset",
                 "⚠️ CHECK: step is a material fraction of an unremoved +1000 DN offset"))
    real = summary[summary.cut == "baseline_04.00"].set_index("channel")

    print("\n=== S2 harmonisation check ===")
    print("per-parcel step across each cut date, seasonal cycle fitted out:")
    print(summary.round(4).to_string(index=False))
    for ch in summary["channel"].unique():
        print(f"  {ch}: measured {real.loc[ch, 'step']:+.4f}, "
              f"estimator residual (placebo) {plac[ch]:+.4f} "
              f"-> corrected {real.loc[ch, 'step_corrected']:+.4f}; "
              f"an unremoved offset would be ~{UNHARMONISED_STEP:+.2f}")

    monthly = (d.groupby("month")[["NDVI", "SWIR1"]].median()
               .join(d.groupby("month").size().rename("n")))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(out / "s2_harmonisation_monthly.csv")
    summary.to_csv(out / "s2_harmonisation_check.csv", index=False)
    _harmonisation_figure(monthly, summary, out / "s2_harmonisation_check.png")
    return summary


def _paired_step(d: pd.DataFrame, cut: pd.Timestamp, col: str,
                 months: int = 12, min_obs: int = 8) -> dict:
    """Per-parcel step across ``cut``, with the seasonal cycle fitted out, not binned out.

    Per parcel, over a +/- ``months`` window::

        value ~ a + b*cos(2*pi*t) + c*sin(2*pi*t) + step * 1[date >= cut]

    ``step`` is the reported coefficient (median across parcels); the order-1 harmonic
    matches ``assemble._summaries``. A within-month-of-year pairing looks as principled but
    samples different parts of a steep seasonal curve across years (~+0.018 on a pure sine
    with no step).
    """
    win = d[(d["date"] >= cut - pd.DateOffset(months=months))
            & (d["date"] < cut + pd.DateOffset(months=months))].copy()
    if win.empty:
        return {"n_pairs": 0, "n_parcels": 0, "step": float("nan"),
                "step_sd": float("nan")}
    t = (win["date"] - cut).dt.days.values / 365.25
    win["_c"], win["_s"] = np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)
    win["_after"] = (win["date"] >= cut).astype(float)

    steps = []
    for _, g in win.groupby("COD_PREDIO", sort=False):
        y = g[col].values.astype(float)
        ok = np.isfinite(y)
        # both sides present, and enough points to identify 4 coefficients
        if ok.sum() < min_obs or g.loc[ok, "_after"].nunique() < 2:
            continue
        X = np.c_[np.ones(ok.sum()), g["_c"].values[ok], g["_s"].values[ok],
                  g["_after"].values[ok]]
        try:
            beta = np.linalg.lstsq(X, y[ok], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        steps.append(float(beta[3]))
    if not steps:
        return {"n_pairs": 0, "n_parcels": 0, "step": float("nan"),
                "step_sd": float("nan")}
    steps = np.asarray(steps)
    return {"n_pairs": int(len(win)), "n_parcels": int(len(steps)),
            "step": float(np.median(steps)), "step_sd": float(steps.std())}


def _harmonisation_figure(monthly: pd.DataFrame, summary: pd.DataFrame,
                          path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [c for c in ("SWIR1", "NDVI") if c in set(summary["channel"])]
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 3.6), squeeze=False)
    for ax, col in zip(axes[0], cols):
        ax.plot(monthly.index, monthly[col], "-o", ms=4, lw=1.8, color="#2a6f97")
        ax.axvline(pd.Timestamp("2022-01-25"), color="#c1121f", ls="--", lw=1.4)
        ax.axvline(pd.Timestamp("2021-01-25"), color="#8b98a5", ls=":", lw=1.2)
        s = summary[(summary.channel == col)
                    & (summary.cut == "baseline_04.00")].iloc[0]
        ax.set_title(f"{col} — step {s.step:+.4f}, placebo {s.placebo_step:+.4f}, "
                     f"corrected {s.step_corrected:+.4f}", fontsize=9)
        ax.tick_params(labelsize=8, rotation=30)
        ax.grid(alpha=0.25, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Sentinel-2 S2_SR_HARMONIZED: is there a radiometric step at the "
                 "2022-01-25 baseline-04.00 boundary?\n"
                 "monthly medians (the visible swing is the growing season). The step is a "
                 "per-parcel regression coefficient with the\nseasonal cycle fitted out, "
                 "against a placebo cut one year earlier (dotted) as the noise floor.",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def observation_report(px: pd.DataFrame, parcels: pd.DataFrame) -> pd.DataFrame:
    """Clear observations per parcel-year by department (§5.6).

    The selva is cloud-limited and that number bounds what a model can do there, so it is
    measured rather than assumed.
    """
    p = px.copy()
    p["date"] = pd.to_datetime(p["date"])
    dept = parcels.set_index("COD_PREDIO")["dept"]
    p["dept"] = p["COD_PREDIO"].map(dept)
    per = (p.groupby(["dept", "COD_PREDIO"]).size().rename("n_obs_24mo") / 2.0)
    rep = (per.groupby("dept").agg(["size", "median", "mean", "min", "max"])
           .rename(columns={"size": "n_parcels", "median": "median_obs_per_year",
                            "mean": "mean_obs_per_year", "min": "min_obs",
                            "max": "max_obs"}))
    return rep.round(1)

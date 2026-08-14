"""Assemble the raw pixel store into the two model representations (plan.md §6, A3).

Reads ``data/processed/features/pixels_<year>.parquet`` (stage-2 raw store) and writes:

* ``features_lightgbm.parquet`` — one row per parcel: **whole-year summary statistics**
  per channel (median/mean/std/min/max/p25/p75/amplitude), a linear **slope**, and
  order-1 **harmonic-fit coefficients** (mean + cos + sin — peak timing without a time
  grid), plus static features. NaN where under-constrained; LightGBM handles NaN natively.
  **No binning, no interpolation.**
* ``tensor_perdate.npz``  — LTAE input: ``X [N, Tmax, C]`` per-date parcel medians,
  ``doy [N, Tmax]``, ``mask [N, Tmax]``, ``cod_predio``, ``channels``.
* ``tensor_pixelset.npz`` — PSE-LTAE input: ``X [N, Tmax, P, C]`` per-date per-pixel
  values, ``pixmask [N, Tmax, P]`` (+ same doy/mask/meta). Pixels are identified by their
  30 m grid location; the P most-observed pixels per parcel are kept.
* ``feature_meta.parquet`` — per-parcel ``n_dates, n_valid_pixels, max_gap_obs``.

Run with::

    uv run python -m crop_classifier.cli features assemble
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crop_classifier.features.indices import BANDS, CHANNELS, add_indices, scale_sr
from crop_classifier.paths import feat

T_MAX = 64   # cap on dates kept per parcel (evenly subsampled beyond this)
P_MAX = 8    # pixels kept per parcel for the pixel-set tensor
MIN_HARMONIC_OBS = 4  # below this the harmonic/slope fits are NaN

FN_LGBM = "features_lightgbm.parquet"
FN_PERDATE = "tensor_perdate.npz"
FN_PIXELSET = "tensor_pixelset.npz"
FN_META = "feature_meta.parquet"


def load_pixels(feat_dir: Path | None = None,
                years: list[int] | None = None) -> pd.DataFrame:
    """Concatenate the raw per-year pixel stores, optionally restricted to ``years``."""
    feat_dir = feat_dir or feat()
    files = sorted(feat_dir.glob("pixels_*.parquet"))
    if years is not None:
        want = {str(y) for y in years}
        files = [f for f in files if f.stem.split("_")[1] in want]
    if not files:
        raise FileNotFoundError(
            f"no pixels_<year>.parquet under {feat_dir}"
            + (f" for years {years}" if years else "") + " — run extraction")
    px = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if years is not None and "year" in px:
        px = px[px["year"].isin(years)]
    px = scale_sr(px)
    # pixel identity = 30 m grid cell (stable across dates)
    px["px_id"] = (px["lon"].round(4).astype(str) + "_" + px["lat"].round(4).astype(str))
    return px


def assert_one_year_per_parcel(px: pd.DataFrame) -> None:
    """Guard the multi-year landmine (plan §7.0.1).

    Every downstream grouping here is keyed on ``(COD_PREDIO, doy)``. That is only sound
    while a parcel's pixels come from a single year — otherwise observations from
    different years silently merge into one "date", producing plausible-looking but
    completely wrong features. The training store holds one year per parcel; the panel
    must therefore assemble **one year at a time** (``years=[Y]``).
    """
    if "year" not in px.columns:
        return
    n = px.groupby("COD_PREDIO")["year"].nunique()
    bad = n[n > 1]
    if len(bad):
        eg = sorted(px.loc[px.COD_PREDIO == bad.index[0], "year"].unique())
        raise ValueError(
            f"{len(bad):,} parcels have pixels in more than one year "
            f"(e.g. {bad.index[0]}: {eg}). "
            "Assemble one year at a time: assemble(years=[Y], out_dir=…).")


def per_date_medians(px: pd.DataFrame) -> pd.DataFrame:
    """Median across a parcel's clear pixels per acquisition date + the 11 channels.

    Callers must have ensured one year per parcel (``assert_one_year_per_parcel``);
    ``doy`` alone identifies an acquisition only within a single year.
    """
    g = (px.groupby(["COD_PREDIO", "doy"], sort=True)
           .agg({**{b: "median" for b in BANDS}, "px_id": "nunique", "mission": "first"})
           .rename(columns={"px_id": "n_px"}).reset_index())
    return add_indices(g)


# ------------------------------------------------------------------------------------
# LightGBM whole-year summary features (no binning — A3)
# ------------------------------------------------------------------------------------
def _summaries(dates: np.ndarray, vals: np.ndarray) -> dict[str, float]:
    """Summary + slope + order-1 harmonic coefficients for one channel series."""
    v = vals[~np.isnan(vals)]
    t = dates[~np.isnan(vals)] / 365.25
    out = {}
    if len(v) == 0:
        keys = ["median", "mean", "std", "min", "max", "p25", "p75", "amp",
                "slope", "h_mean", "h_cos", "h_sin"]
        return {k: np.nan for k in keys}
    out["median"], out["mean"], out["std"] = np.median(v), v.mean(), v.std()
    out["min"], out["max"] = v.min(), v.max()
    out["p25"], out["p75"] = np.percentile(v, 25), np.percentile(v, 75)
    out["amp"] = out["max"] - out["min"]
    if len(v) >= MIN_HARMONIC_OBS:
        A = np.c_[np.ones_like(t), t]
        out["slope"] = np.linalg.lstsq(A, v, rcond=None)[0][1]
        H = np.c_[np.ones_like(t), np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)]
        h = np.linalg.lstsq(H, v, rcond=None)[0]
        out["h_mean"], out["h_cos"], out["h_sin"] = h
    else:
        out["slope"] = out["h_mean"] = out["h_cos"] = out["h_sin"] = np.nan
    return out


def build_lightgbm_features(pd_med: pd.DataFrame, parcels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, grp in pd_med.groupby("COD_PREDIO", sort=False):
        doy = grp["doy"].values.astype(float)
        row: dict[str, object] = {"COD_PREDIO": cid, "n_dates": len(grp),
                                  "n_valid_pixels": int(grp["n_px"].max()),
                                  "frac_l7": float((grp["mission"] == 7).mean())}
        for ch in CHANNELS:
            for k, v in _summaries(doy, grp[ch].values.astype(float)).items():
                row[f"{ch}_{k}"] = v
        rows.append(row)
    feats = pd.DataFrame(rows)
    statics = parcels[["COD_PREDIO", "area_ha", "n_pixels_est", "centroid_lat",
                       "n_valid_obs", "max_gap"]].copy()
    statics["n_valid_obs"] = statics["n_valid_obs"].astype("float32")
    statics["max_gap"] = statics["max_gap"].astype("float32")
    return feats.merge(statics, on="COD_PREDIO", how="left")


# ------------------------------------------------------------------------------------
# Per-date + pixel-set tensors (LTAE / PSE-LTAE)
# ------------------------------------------------------------------------------------
def _subsample_dates(doys: np.ndarray, t_max: int) -> np.ndarray:
    """Indices of at most ``t_max`` dates, evenly spread across the observed sequence."""
    if len(doys) <= t_max:
        return np.arange(len(doys))
    return np.unique(np.linspace(0, len(doys) - 1, t_max).round().astype(int))


def build_tensors(px: pd.DataFrame, pd_med: pd.DataFrame,
                  t_max: int = T_MAX, p_max: int = P_MAX) -> dict[str, np.ndarray]:
    cods = pd_med["COD_PREDIO"].unique()
    n, c = len(cods), len(CHANNELS)
    X = np.zeros((n, t_max, c), dtype=np.float32)
    DOY = np.zeros((n, t_max), dtype=np.int16)
    MASK = np.zeros((n, t_max), dtype=bool)
    XP = np.zeros((n, t_max, p_max, c), dtype=np.float32)
    PIXMASK = np.zeros((n, t_max, p_max), dtype=bool)

    px = add_indices(px)
    px_g = dict(tuple(px.groupby("COD_PREDIO", sort=False)))
    # pre-grouped parcel-date medians: a per-parcel boolean scan here is O(N^2) and takes
    # hours at full scale (33k parcels x ~1M parcel-dates)
    med_g = dict(tuple(pd_med.groupby("COD_PREDIO", sort=False)))
    for i, cid in enumerate(cods):
        med = med_g[cid].sort_values("doy")
        keep = _subsample_dates(med["doy"].values, t_max)
        med = med.iloc[keep]
        tt = len(med)
        X[i, :tt] = med[CHANNELS].values.astype(np.float32)
        DOY[i, :tt] = med["doy"].values
        MASK[i, :tt] = True

        pxg = px_g[cid]
        top_px = pxg["px_id"].value_counts().index[:p_max].tolist()
        px_pos = {p: j for j, p in enumerate(top_px)}
        sub = pxg[pxg["px_id"].isin(px_pos)]
        doy_pos = {int(d): t for t, d in enumerate(med["doy"].values)}
        for (d, pid), rows in sub.groupby(["doy", "px_id"]):
            t = doy_pos.get(int(d))
            if t is None:
                continue
            j = px_pos[pid]
            XP[i, t, j] = rows[CHANNELS].iloc[0].values.astype(np.float32)
            PIXMASK[i, t, j] = True

    return {"cod_predio": cods.astype(str), "channels": np.array(CHANNELS),
            "X": np.nan_to_num(X), "doy": DOY, "mask": MASK,
            "XP": np.nan_to_num(XP), "pixmask": PIXMASK}


# ------------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------------
def assemble(t_max: int = T_MAX, p_max: int = P_MAX,
             years: list[int] | None = None,
             feat_dir: Path | None = None,
             out_dir: Path | None = None,
             parcels: pd.DataFrame | None = None,
             harmonize_oli: bool = False,
             harmonize_coefficients: dict | None = None) -> None:
    """Raw pixel store -> the three model representations.

    ``years`` restricts the input to those crop years and ``out_dir`` redirects the
    outputs — together these give the multi-year panel one isolated bundle per year
    (plan §7.0.1/§7.2). ``harmonize_oli`` applies the Roy et al. (2016) OLI->ETM+
    coefficients to L8/L9 observations so a panel spanning the 2013 sensor boundary does
    not inject a spurious trend (D7).
    """
    feat_dir = feat_dir or feat()
    out_dir = out_dir or feat_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if parcels is None:
        import geopandas as gpd

        from crop_classifier.paths import proc
        parcels = gpd.read_parquet(proc() / "modeling_parcels.parquet")
    print("loading raw pixel store …")
    px = load_pixels(feat_dir, years)
    print(f"  {len(px):,} pixel-obs, {px.COD_PREDIO.nunique():,} parcels")
    assert_one_year_per_parcel(px)
    if harmonize_oli:
        from crop_classifier.perennial.harmonization import oli_to_etm
        px = oli_to_etm(px, coefficients=harmonize_coefficients)

    pd_med = per_date_medians(px)
    print(f"  {len(pd_med):,} parcel-dates")

    print("building LightGBM summary features …")
    feats = build_lightgbm_features(pd_med, parcels)
    feats.to_parquet(out_dir / FN_LGBM, index=False)
    print(f"  {FN_LGBM}: {feats.shape[0]:,} rows x {feats.shape[1]} cols")

    print("building per-date + pixel-set tensors …")
    t = build_tensors(px, pd_med, t_max, p_max)
    np.savez_compressed(out_dir / FN_PERDATE, cod_predio=t["cod_predio"],
                        channels=t["channels"], X=t["X"], doy=t["doy"], mask=t["mask"])
    np.savez_compressed(out_dir / FN_PIXELSET, cod_predio=t["cod_predio"],
                        channels=t["channels"], X=t["XP"], doy=t["doy"], mask=t["mask"],
                        pixmask=t["pixmask"])
    print(f"  {FN_PERDATE}: X {t['X'].shape};  {FN_PIXELSET}: X {t['XP'].shape}")

    meta = (pd_med.groupby("COD_PREDIO")
            .agg(n_dates=("doy", "size"), n_valid_pixels=("n_px", "max"))
            .reset_index())
    meta.to_parquet(out_dir / FN_META, index=False)
    print(f"  {FN_META}: {len(meta):,} parcels")


if __name__ == "__main__":
    assemble()

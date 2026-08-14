"""S2 per-date medians -> the LightGBM summary feature block (s2_labelling_plan.md §5.5).

Only the summary block is emitted. LTAE/PSE-LTAE are dropped from this strand (they lose
on LODO 0.442 vs 0.477, LOYO 0.4784 and W2 -0.1025), so nothing needs the per-date or
pixel-set tensors and there is no reason to build them.

Two settled negative results are enforced here rather than left to a config: **no
``centroid_lat``/``centroid_lon``** (spatial memorisation — CV +0.047, LODO -0.060) and
**no acquisition metadata** (``frac_l7`` manufactured a trend; the ablation cost +0.0013,
i.e. nothing). Neither is written, so neither can be silently re-admitted downstream.

The feature window is the **agricultural year Aug 1 - Jul 31 containing the parcel's Esri
``imagery_date``**, cut out of the extracted 24-month trace.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from crop_classifier.features.assemble import _summaries
from crop_classifier.features.indices import CHANNELS, add_indices, scale_sr_s2
from crop_classifier.features.s2_gee import ag_year, f_pixels
from crop_classifier.paths import feat

FN_LGBM = "s2_features_lightgbm.parquet"
FN_META = "s2_feature_meta.parquet"


def restrict_to_ag_year(px: pd.DataFrame, parcels: pd.DataFrame) -> pd.DataFrame:
    """Keep only observations inside each parcel's own agricultural year."""
    d = px.copy()
    d["date"] = pd.to_datetime(d["date"])
    win = {}
    for cid, imd in zip(parcels["COD_PREDIO"].astype(str), parcels["imagery_date"]):
        lo, hi = ag_year(imd)
        win[cid] = (pd.Timestamp(lo), pd.Timestamp(hi))
    lo = d["COD_PREDIO"].astype(str).map(lambda c: win.get(c, (None, None))[0])
    hi = d["COD_PREDIO"].astype(str).map(lambda c: win.get(c, (None, None))[1])
    keep = lo.notna() & (d["date"] >= lo) & (d["date"] <= hi)
    print(f"  agricultural-year window: {int(keep.sum()):,} of {len(d):,} parcel-dates")
    return d[keep]


def build_features(px: pd.DataFrame, parcels: pd.DataFrame) -> pd.DataFrame:
    """One row per parcel: 12 summaries x 11 channels + observation/geometry statics."""
    d = add_indices(scale_sr_s2(px))
    # `_summaries` fits its slope and harmonic on a time axis in years; a window that
    # straddles New Year needs a continuous axis, so use days since the window start
    # rather than day-of-year, which would wrap.
    d = d.sort_values(["COD_PREDIO", "date"])
    rows = []
    for cid, g in d.groupby("COD_PREDIO", sort=False):
        t = (g["date"] - g["date"].min()).dt.days.values.astype(float)
        row: dict[str, object] = {
            "COD_PREDIO": str(cid), "n_dates": len(g),
            "n_px_median": float(g["n_px"].median()),
            "n_px_min": float(g["n_px"].min()),
            "eroded": bool(g["eroded"].iloc[0]) if "eroded" in g else True,
        }
        for ch in CHANNELS:
            for k, v in _summaries(t, g[ch].values.astype(float)).items():
                row[f"{ch}_{k}"] = v
        rows.append(row)
    feats = pd.DataFrame(rows)
    statics = parcels[["COD_PREDIO", "area_ha"]].copy()
    statics["COD_PREDIO"] = statics["COD_PREDIO"].astype(str)
    return feats.merge(statics, on="COD_PREDIO", how="left")


def assemble(px: pd.DataFrame | None = None, parcels: pd.DataFrame | None = None,
             out_dir: Path | None = None) -> pd.DataFrame:
    out_dir = Path(out_dir) if out_dir else feat()
    out_dir.mkdir(parents=True, exist_ok=True)
    if px is None:
        px = pd.read_parquet(f_pixels())
    if parcels is None:
        import geopandas as gpd

        from crop_classifier.allperu.label_sample import F_SAMPLE
        from crop_classifier.allperu.label_sample import out_dir as sdir
        parcels = gpd.read_parquet(sdir() / F_SAMPLE)

    print(f"loading {len(px):,} parcel-dates for {px.COD_PREDIO.nunique():,} parcels")
    win = restrict_to_ag_year(px, parcels)
    feats = build_features(win, parcels)
    feats.to_parquet(out_dir / FN_LGBM, index=False)

    banned = [c for c in feats.columns
              if c.startswith("centroid_") or c in ("frac_l7", "mission", "doy")]
    if banned:
        raise AssertionError(f"settled-negative features leaked into the store: {banned}")

    meta = (win.groupby("COD_PREDIO")
            .agg(n_dates=("date", "size"), n_px_median=("n_px", "median"),
                 first_date=("date", "min"), last_date=("date", "max"))
            .reset_index())
    meta.to_parquet(out_dir / FN_META, index=False)
    print(f"{FN_LGBM}: {feats.shape[0]:,} rows x {feats.shape[1]} cols "
          f"(median {meta.n_dates.median():.0f} dates/parcel)")
    return feats


def coverage_report(px: pd.DataFrame, parcels: pd.DataFrame) -> pd.DataFrame:
    """Per-department clear observations in the agricultural year (§5.6, gate input)."""
    win = restrict_to_ag_year(px, parcels)
    n = win.groupby("COD_PREDIO").size().rename("n_dates")
    p = parcels.set_index(parcels["COD_PREDIO"].astype(str))
    j = pd.DataFrame({"n_dates": n}).join(p["dept"], how="left")
    rep = (j.groupby("dept")["n_dates"]
           .agg(["size", "median", "mean", "min", "max"])
           .rename(columns={"size": "n_parcels"}))
    rep["frac_lt_10_obs"] = j.groupby("dept")["n_dates"].apply(lambda s: (s < 10).mean())
    return rep.round(2)


def missing_parcels(px: pd.DataFrame, parcels: pd.DataFrame) -> pd.Series:
    """Parcels with no usable S2 date at all — the attrition to report, not to hide."""
    have = set(px["COD_PREDIO"].astype(str))
    miss = parcels[~parcels["COD_PREDIO"].astype(str).isin(have)]
    return miss.groupby("dept").size() if len(miss) else pd.Series(dtype=int)


if __name__ == "__main__":
    assemble()



"""Refit the OLI->ETM+ correction on *this* data — the follow-up to the step-3a failure.

``perennial/harmonization.py`` carries the Roy et al. (2016) coefficients. Step 3a ran them
for the first time and found them **worse than no correction**: the PETT-``PERENNIAL``
control pool shifts −0.0423 with raw OLI and **−0.1073** with the published correction
(RESULTS.md §9.3.2). Those were fitted on Collection-1 data over North America; this pipeline
reads Collection-2 Level-2 surface reflectance over Peru.

The fix is to stop borrowing them. L7 and L8 fly 8 days apart, but adjacent WRS paths
overlap, so a parcel in a sidelap is imaged by **both sensors on the same day** — same
ground, illumination and atmosphere, different instrument. Over 2015/2019/2022 the panel has
**~35,000 such same-day parcel-date pairs**: Roy's observational design on our own data.

What this can and cannot fix, stated before running (T-D6):

* it CAN remove a per-band linear offset/gain difference between the instruments;
* it CANNOT remove a difference in *when* each satellite looks, in cloud-mask behaviour, or
  in spatial sampling. If the step-3a control shift survives a locally-fitted correction,
  the problem is not radiometry and step 3 stays closed — which is itself the answer.

Run with::

    CC_PROC=data/processed/all_peru CC_FEAT=data/processed/all_peru/features \\
      uv run python -m crop_classifier.cli allperu oli-refit --years 2015,2019,2022
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from crop_classifier.features import assemble as asm
from crop_classifier.features.indices import BANDS
from crop_classifier.paths import proc
from crop_classifier.perennial.harmonization import ROY2016_OLI_TO_ETM
from crop_classifier.perennial.panel import panel_feat

DEFAULT_YEARS = (2015, 2019, 2022)
MAX_DAYS = 1          # same-day (or next-day) acquisitions only
MIN_PIXELS = 3        # per parcel-date, before its median is trusted

FN_COEF = "oli_refit_coefficients.json"


def build_pairs(years: tuple[int, ...] = DEFAULT_YEARS, max_days: int = MAX_DAYS,
                min_pixels: int = MIN_PIXELS) -> pd.DataFrame:
    """Same-parcel, same-day (±``max_days``) L7 / OLI parcel-date band medians.

    Medians over the parcel's clear pixels, not individual pixels: the two sensors do not
    share a pixel grid, so a pixel-level join would be comparing different ground.
    """
    frames = []
    for y in years:
        l7p = panel_feat() / f"pixels_{y}.parquet"
        olp = panel_feat() / "oli" / f"pixels_{y}.parquet"
        if not (l7p.exists() and olp.exists()):
            continue
        a = asm.scale_sr(pd.read_parquet(l7p))
        b = asm.scale_sr(pd.read_parquet(olp))
        ga = a.groupby(["COD_PREDIO", "doy"]).agg(
            {**{c: "median" for c in BANDS}, "lon": "size"}).rename(
            columns={"lon": "n_px"}).reset_index()
        gb = b.groupby(["COD_PREDIO", "doy"]).agg(
            {**{c: "median" for c in BANDS}, "lon": "size"}).rename(
            columns={"lon": "n_px"}).reset_index()
        ga = ga[ga["n_px"] >= min_pixels]
        gb = gb[gb["n_px"] >= min_pixels]
        m = ga.merge(gb, on="COD_PREDIO", suffixes=("_l7", "_oli"))
        m = m[(m["doy_l7"] - m["doy_oli"]).abs() <= max_days].copy()
        m["year"] = y
        frames.append(m)
    if not frames:
        raise FileNotFoundError("no paired L7/OLI pixel stores found")
    out = pd.concat(frames, ignore_index=True)
    print(f"{len(out):,} same-day parcel-date pairs over {list(years)} "
          f"({out.COD_PREDIO.nunique():,} parcels)")
    return out


def fit_coefficients(pairs: pd.DataFrame, bands: tuple[str, ...] = tuple(BANDS),
                     clip: float = 0.005) -> dict:
    """Per band, OLS of the ETM+ value on the OLI value: ``etm = slope · oli + intercept``.

    Same form as Roy et al., so a drop-in replacement for ``ROY2016_OLI_TO_ETM``. Theil-Sen
    is computed alongside as a robustness check — surface reflectance has a heavy tail from
    cloud-mask misses, and if the two fits disagree materially OLS should not be trusted.
    """
    from scipy import stats

    out: dict = {}
    for band in bands:
        x = pairs[f"{band}_oli"].to_numpy(float)
        y = pairs[f"{band}_l7"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y) & (x > clip) & (y > clip)
        if ok.sum() < 500:
            continue
        x, y = x[ok], y[ok]
        res = stats.linregress(x, y)
        ts = stats.theilslopes(y, x, 0.95)
        roy_s, roy_i = ROY2016_OLI_TO_ETM[band]
        out[band] = {
            "slope": float(res.slope), "intercept": float(res.intercept),
            "r2": float(res.rvalue ** 2), "n": int(ok.sum()),
            "theilsen_slope": float(ts[0]), "theilsen_intercept": float(ts[1]),
            "roy_slope": roy_s, "roy_intercept": roy_i,
            # what each correction does to a typical mid-range reflectance
            "mean_oli": float(x.mean()), "mean_l7": float(y.mean()),
            "raw_bias_l7_minus_oli": float(y.mean() - x.mean()),
            "residual_bias_fitted": float((y - (res.slope * x + res.intercept)).mean()),
            "residual_bias_roy": float((y - (roy_s * x + roy_i)).mean()),
        }
    return out


def as_coefficient_map(fit: dict, kind: str = "offset") -> dict[str, tuple[float, float]]:
    """``fit`` -> the ``{band: (slope, intercept)}`` form ``harmonization.oli_to_etm`` takes.

    ``kind="offset"`` is the **default and the only one this data supports**: gain fixed at
    1.0, a per-band additive bias. The paired sample cannot identify a *slope* — same-day band
    medians correlate at only 0.12-0.60 and the SD of their difference (0.064-0.102)
    **exceeds the SD of either sensor's own values** (0.065-0.088), so OLS is diluted to
    nonsense (blue slope 0.12 vs a physical ~0.85) and Theil-Sen is unreliable too. The
    *mean* difference is precise (27,573 pairs, SE ~0.0006), so an offset is what the data
    identifies.

    ``"ols"`` and ``"theilsen"`` are kept so the diluted fits can be reproduced and rejected.
    """
    if kind == "offset":
        return {b: (1.0, float(v["raw_bias_l7_minus_oli"])) for b, v in fit.items()}
    key = ("theilsen_slope", "theilsen_intercept") if kind == "theilsen" else \
          ("slope", "intercept")
    return {b: (float(v[key[0]]), float(v[key[1]])) for b, v in fit.items()}


def report(fit: dict) -> pd.DataFrame:
    """Fitted vs published coefficients, and the bias each one leaves behind."""
    rows = []
    for b, v in fit.items():
        rows.append({
            "band": b, "n": v["n"], "r2": round(v["r2"], 4),
            "fitted_slope": round(v["slope"], 4),
            "fitted_intercept": round(v["intercept"], 5),
            "theilsen_slope": round(v["theilsen_slope"], 4),
            "roy_slope": v["roy_slope"], "roy_intercept": v["roy_intercept"],
            "bias_none": round(v["raw_bias_l7_minus_oli"], 5),
            "bias_roy": round(v["residual_bias_roy"], 5),
            "bias_fitted": round(v["residual_bias_fitted"], 5),
        })
    return pd.DataFrame(rows)


def run(years: tuple[int, ...] = DEFAULT_YEARS, max_days: int = MAX_DAYS,
        save: bool = True) -> dict:
    pairs = build_pairs(years, max_days=max_days)
    fit = fit_coefficients(pairs)
    rep = report(fit)
    print(rep.to_string(index=False))
    print("\nbias columns are mean(ETM+ − corrected OLI) in reflectance units; "
          "0 is perfect.\n  none   = no correction\n  roy    = the published coefficients"
          "\n  fitted = refitted on these same-day pairs")
    if save:
        with open(proc() / FN_COEF, "w") as f:
            json.dump({"years": list(years), "max_days": max_days,
                       "n_pairs": int(len(pairs)), "bands": fit}, f, indent=2)
        pairs.to_parquet(proc() / "oli_refit_pairs.parquet", index=False)
        print(f"\nwrote {proc() / FN_COEF} and oli_refit_pairs.parquet")
    return fit


def load_coefficients(path: Path | None = None,
                      kind: str = "offset") -> dict[str, tuple[float, float]]:
    p = Path(path or (proc() / FN_COEF))
    with open(p) as f:
        return as_coefficient_map(json.load(f)["bands"], kind=kind)

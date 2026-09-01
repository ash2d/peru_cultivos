"""OLI -> ETM+ radiometric harmonisation (plan decision D7).

The mission-mixing audit (``docs/DATA.md`` §7.1) found L5/L7 mixing
radiometrically harmless. **That finding does not extend to L8/L9.** OLI has different
band centres and spectral response functions from TM/ETM+, so training on 1998 TM data and
predicting 2020 OLI data without correction injects a sensor step directly into the
headline trend — exactly where a spurious "shift to perennials" would appear.

Coefficients are the OLS regressions of Roy et al. (2016), *Remote Sensing of Environment*
185:57-70, "Characterization of Landsat-7 to Landsat-8 reflective wavelength and
normalized difference vegetation index continuity", Table 2 — the **OLI -> ETM+**
direction, applied to surface reflectance. L9 carries the same OLI-2 design as L8 and is
harmonised with the same coefficients (the standard practice; the L8/L9 cross-calibration
difference is far smaller than the OLI/ETM+ one).

Applied at *assembly* time, never at extraction: the raw pixel store stays raw, so the
correction can be revised without re-running GEE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ⚠️ MEASURED ON THIS DATA AND REJECTED (docs/RESULTS.md §6.4).
# Applying these coefficients makes the PETT-PERENNIAL control-pool shift 2.5x WORSE
# (-0.0423 -> -0.1073) and lowers class agreement. On same-day L7/OLI parcel pairs over Peru,
# raw OLI already reads *greener* than ETM+ (mean NDVI +0.055) and these coefficients push it
# further away (+0.069) instead of back. A locally-refitted per-band offset does much better
# (archive/oli_refit.py) and STILL fails, because the cross-sensor difference is cover-type
# dependent (0.025 NDVI between perennial and pasture parcels) and no global linear map can
# remove a difference that lies between classes. Do not enable OLI on the strength of this
# module.
# band -> (slope, intercept) for  etm = slope * oli + intercept   (surface reflectance)
ROY2016_OLI_TO_ETM: dict[str, tuple[float, float]] = {
    "B":     (0.8474, 0.0003),
    "G":     (0.8483, 0.0088),
    "R":     (0.9047, 0.0061),
    "NIR":   (0.8462, 0.0412),
    "SWIR1": (0.8937, 0.0254),
    "SWIR2": (0.9071, 0.0172),
}

OLI_MISSIONS = (8, 9)


def oli_to_etm(px: pd.DataFrame, missions: tuple[int, ...] = OLI_MISSIONS,
               coefficients: dict[str, tuple[float, float]] | None = None
               ) -> pd.DataFrame:
    """Return ``px`` with OLI rows' reflectance mapped onto the ETM+ scale.

    Expects **scaled surface reflectance** (i.e. after ``indices.scale_sr``) and a
    ``mission`` column. Rows from TM/ETM+ are returned untouched, so calling this on a
    pre-2013 store is a no-op.
    """
    coefficients = coefficients or ROY2016_OLI_TO_ETM
    if "mission" not in px.columns:
        return px
    out = px.copy()
    is_oli = out["mission"].isin(missions).to_numpy()
    if not is_oli.any():
        return out
    for band, (slope, intercept) in coefficients.items():
        if band in out.columns:
            # copy=True: to_numpy can hand back a read-only view of the block, and the
            # in-place write below then raises. It happened not to on the wide float64
            # frames this is normally called with, which is exactly how a latent bug survives.
            v = out[band].to_numpy(dtype=float, copy=True)
            v[is_oli] = slope * v[is_oli] + intercept
            out[band] = v
    return out


def mission_step_report(feat: pd.DataFrame, year_col: str = "year",
                        cols: tuple[str, ...] = ("NDVI_median", "NDVI_amp", "BSI_max")
                        ) -> pd.DataFrame:
    """Per-year medians of the drift-sensitive features (§7.4.1 diagnostic).

    Step changes at 1999 (L7 arrives), 2012 (L7 SLC-off only) and 2013 (L8) are sensor
    artefacts, not land use. Harmonisation should shrink the 2013 step materially; if it
    does not, do not believe the trend.
    """
    have = [c for c in cols if c in feat.columns]
    g = feat.groupby(year_col)[have].median()
    g["n_parcels"] = feat.groupby(year_col).size()
    g["step_vs_prev"] = np.nan
    if have:
        g["step_vs_prev"] = g[have[0]].diff()
    return g.reset_index()

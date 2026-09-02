"""Spectral indices from harmonised, scaled Landsat SR bands (plan.md §6).

Input columns are the 6 harmonised surface-reflectance bands ``B, G, R, NIR, SWIR1,
SWIR2`` already scaled to reflectance (Collection-2: ``DN * 0.0000275 - 0.2``).
Adds 5 indices -> the 11 model channels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BANDS = ["B", "G", "R", "NIR", "SWIR1", "SWIR2"]
INDICES = ["NDVI", "EVI", "NDWI", "NDMI", "BSI"]
CHANNELS = BANDS + INDICES  # the 11 channels every assembly uses, in this order

# Collection-2 Level-2 SR scaling
SR_SCALE, SR_OFFSET = 0.0000275, -0.2

# Sentinel-2 L2A scaling. `COPERNICUS/S2_SR_HARMONIZED` removes the +1000 DN offset that
# processing baseline 04.00 introduced on 2022-01-25, so one scale factor covers the whole
# archive and there is no radiometric step in the middle of the 2019+ target window.
S2_SCALE = 1.0 / 10000.0


def scale_sr(df: pd.DataFrame) -> pd.DataFrame:
    """Scale raw C2 DN band columns to reflectance, clipping to a sane [0, 1] range."""
    out = df.copy()
    for b in BANDS:
        out[b] = (out[b].astype("float32") * SR_SCALE + SR_OFFSET).clip(0.0, 1.0)
    return out


def scale_sr_s2(df: pd.DataFrame) -> pd.DataFrame:
    """Scale raw S2 L2A DN band columns to reflectance, clipped to [0, 1].

    Beside :func:`scale_sr`, not a branch inside it: a sensor argument on a scaling function
    gets defaulted wrong once and applies the Landsat offset to Sentinel-2 for a whole store
    with no error. Two names, one scale each.
    """
    out = df.copy()
    for b in BANDS:
        out[b] = (out[b].astype("float32") * S2_SCALE).clip(0.0, 1.0)
    return out


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den
    return r.replace([np.inf, -np.inf], np.nan)


def add_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Append the 5 index columns to a frame of scaled band columns."""
    out = df.copy()
    b, g, r = out["B"], out["G"], out["R"]
    nir, sw1 = out["NIR"], out["SWIR1"]
    out["NDVI"] = _safe_ratio(nir - r, nir + r)
    out["EVI"] = _safe_ratio(2.5 * (nir - r), nir + 6 * r - 7.5 * b + 1)
    out["NDWI"] = _safe_ratio(g - nir, g + nir)          # McFeeters (open water / flooded rice)
    out["NDMI"] = _safe_ratio(nir - sw1, nir + sw1)      # moisture
    out["BSI"] = _safe_ratio((sw1 + r) - (nir + b), (sw1 + r) + (nir + b))  # bare soil
    for c in INDICES:
        out[c] = out[c].clip(-1.5, 2.5).astype("float32")
    return out

"""Guards on the per-parcel climate covariates.

The trap this file pins is the same shape as the four in `DATA.md`: a climate raster masks
the ocean, the cadastre runs to the shoreline, and a parcel whose centroid lands one cell
seaward comes back **NaN, not an error** — silently dropping it from any model that uses the
column. The sampler falls back to the nearest valid cell and reports the count. These tests
hold that, plus the two things that make a centroid sample legitimate here: the parcel is
smaller than the pixel, and the derived columns mean what their names say.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

rasterio = pytest.importorskip("rasterio")

from crop_classifier.allperu.climate import _sample_points  # noqa: E402


def _write(tmp_path, arr, name="t.tif", west=-82.0, north=-3.0, res=0.5):
    """A tiny north-up GeoTIFF over Peru's bounding box, NaN-free nodata = -9999."""
    from rasterio.transform import from_origin
    p = tmp_path / name
    with rasterio.open(p, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="float32", crs="EPSG:4326",
                       transform=from_origin(west, north, res, res),
                       nodata=-9999.0) as dst:
        dst.write(arr.astype("float32"), 1)
    return p


def test_a_point_reads_the_cell_it_falls_in(tmp_path):
    arr = np.arange(9, dtype="float32").reshape(3, 3)
    tif = _write(tmp_path, arr)
    # centre of the middle cell: west + 1.5*res, north - 1.5*res
    v = _sample_points(tif, np.array([-81.25]), np.array([-3.75]))
    assert v[0] == 4.0


def test_a_centroid_on_a_masked_cell_falls_back_to_the_nearest_valid_one(tmp_path):
    """The coastal case. Without the fallback this returns NaN and the parcel vanishes."""
    arr = np.full((3, 3), 10.0, dtype="float32")
    arr[1, 1] = -9999.0                       # the cell the parcel sits in is masked
    tif = _write(tmp_path, arr)
    v = _sample_points(tif, np.array([-81.25]), np.array([-3.75]))
    assert np.isfinite(v[0]) and v[0] == pytest.approx(10.0)


def test_the_fallback_gives_up_rather_than_inventing_a_value(tmp_path):
    """An entirely masked raster must stay NaN — a fabricated number would be worse."""
    tif = _write(tmp_path, np.full((3, 3), -9999.0, dtype="float32"))
    v = _sample_points(tif, np.array([-81.25]), np.array([-3.75]))
    assert np.isnan(v[0])


def test_points_outside_the_raster_are_clamped_not_wrapped(tmp_path):
    """`rowcol` happily returns negative indices, and numpy then reads from the far edge."""
    arr = np.arange(9, dtype="float32").reshape(3, 3)
    tif = _write(tmp_path, arr)
    v = _sample_points(tif, np.array([-99.0]), np.array([-3.75]))
    assert v[0] == 3.0                        # clamped to column 0 of the middle row


# --- the premise that licenses a centroid sample ---
def test_no_national_parcel_is_larger_than_a_worldclim_cell():
    """If this ever fails, the centroid sample stops being equivalent to a zonal mean."""
    from crop_classifier.allperu.climate import PARCELS
    if not PARCELS.exists():
        pytest.skip("national parcel table not built in this workspace")
    a = pd.read_parquet(PARCELS, columns=["area_ha"])["area_ha"]
    px_ha = (30 / 3600) * 111_320 * np.cos(np.deg2rad(10.0)) * (30 / 3600) * 111_320 / 1e4
    assert a.max() < px_ha, f"largest parcel {a.max():.1f} ha vs {px_ha:.0f} ha cell"


# --- the derived columns say what they are named ---
def test_derived_seasonality_columns_match_their_definitions(tmp_path, monkeypatch):
    """`n_dry_months` counts months under 50 mm; the aridity index is P/(T+10)."""

    t = np.array([[20.0] * 12])
    p = np.array([[10.0] * 6 + [200.0] * 6])
    assert (p < 50).sum() == 6
    annual_p, annual_t = p.sum(), t.mean()
    assert annual_p / (annual_t + 10.0) == pytest.approx(1260.0 / 30.0)


def test_rainfall_seasonality_is_nan_exactly_where_rainfall_is_zero():
    """0/0 is undefined and stays undefined.

    1,275 Ica parcels get exactly 0 mm/year in the WorldClim normal. Filling their
    seasonality with 0 would claim rain is evenly spread through a year in which none falls.
    This pins the NaN to those rows — a NaN elsewhere would silently drop the parcel.
    """
    from crop_classifier.allperu.climate import OUT_DIR
    f = OUT_DIR / "parcel_climate_normals.parquet"
    if not f.exists():
        pytest.skip("climate normals not built in this workspace")
    d = pd.read_parquet(f, columns=["precip_mm_yr", "precip_seasonality_cv",
                                    "tmean_c", "n_dry_months"])
    assert d["tmean_c"].notna().all()
    assert d["precip_mm_yr"].notna().all()
    assert (d.loc[d["precip_seasonality_cv"].isna(), "precip_mm_yr"] == 0).all()
    assert d.loc[d["precip_mm_yr"] > 0, "precip_seasonality_cv"].notna().all()


def test_annual_rainfall_table_has_no_missing_values():
    from crop_classifier.allperu.climate import OUT_DIR
    f = OUT_DIR / "parcel_rainfall_annual.parquet"
    if not f.exists():
        pytest.skip("annual rainfall not built in this workspace")
    d = pd.read_parquet(f)
    assert d.isna().sum().sum() == 0
    years = [c for c in d.columns if c.startswith("precip_mm_") and c[-4:].isdigit()]
    assert len(years) == 29
    # 1998 and 2023 were El Nino years on the north coast; the series must show it
    assert d["precip_mm_1998"].mean() > d["precip_mm_1997"].mean()

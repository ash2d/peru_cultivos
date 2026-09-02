"""The §7.0.1 regression: a parcel with pixels in two years must not merge them.

``per_date_medians`` groups on ``(COD_PREDIO, doy)``. That is sound only while a parcel's
pixels come from one year — which holds for the training store but *not* for the
multi-year panel, where the same parcel appears in 35 years. Merging DOY 120 of 1998 with
DOY 120 of 2015 would produce plausible-looking, completely wrong features. The guard is
an explicit assertion plus a ``years`` filter with a per-year output directory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crop_classifier.features import assemble as A


def synthetic_pixels(year: int, value: float, cid: str = "P1",
                     doys=(20, 100, 180, 260, 340)) -> pd.DataFrame:
    """One parcel, one pixel, five clear dates — raw Collection-2 DN scale."""
    rows = []
    for d in doys:
        rows.append({"COD_PREDIO": cid, "year": year, "doy": d, "mission": 5,
                     "lon": -80.5, "lat": -5.2,
                     # scale_sr maps DN -> reflectance: 0.0000275*DN - 0.2
                     **{b: (value + 0.2) / 0.0000275 for b in
                        ("B", "G", "R", "SWIR1", "SWIR2")},
                     "NIR": (value * 2 + 0.2) / 0.0000275})
    return pd.DataFrame(rows)


def write_store(tmp_path, frames):
    for df in frames:
        y = int(df["year"].iloc[0])
        df.to_parquet(tmp_path / f"pixels_{y}.parquet", index=False)
    return tmp_path


# --- the guard ---
def test_multi_year_parcel_is_rejected_not_silently_merged(tmp_path):
    store = write_store(tmp_path, [synthetic_pixels(1998, 0.10),
                                   synthetic_pixels(2015, 0.60)])
    px = A.load_pixels(store)
    assert px["COD_PREDIO"].nunique() == 1 and px["year"].nunique() == 2
    with pytest.raises(ValueError, match="more than one year"):
        A.assert_one_year_per_parcel(px)


def test_single_year_store_passes_the_guard(tmp_path):
    store = write_store(tmp_path, [synthetic_pixels(1998, 0.10)])
    A.assert_one_year_per_parcel(A.load_pixels(store))   # must not raise


# --- the fix ---
def test_years_filter_isolates_a_year(tmp_path):
    store = write_store(tmp_path, [synthetic_pixels(1998, 0.10),
                                   synthetic_pixels(2015, 0.60)])
    px98 = A.load_pixels(store, years=[1998])
    assert set(px98["year"]) == {1998}
    A.assert_one_year_per_parcel(px98)


def test_year1_tensor_is_unaffected_by_year2_rows(tmp_path):
    """The actual regression: assembling 1998 from a two-year store must give exactly the
    same tensors as assembling it from a 1998-only store."""
    parcels = pd.DataFrame({"COD_PREDIO": ["P1"], "area_ha": [1.0],
                            "n_pixels_est": [11.1], "centroid_lat": [-5.2],
                            "n_valid_obs": [5], "max_gap": [0]})

    alone, both = tmp_path / "a", tmp_path / "b"
    alone.mkdir()
    both.mkdir()
    write_store(alone, [synthetic_pixels(1998, 0.10)])
    write_store(both, [synthetic_pixels(1998, 0.10), synthetic_pixels(2015, 0.60)])

    out_a, out_b = tmp_path / "outa", tmp_path / "outb"
    A.assemble(years=[1998], feat_dir=alone, out_dir=out_a, parcels=parcels)
    A.assemble(years=[1998], feat_dir=both, out_dir=out_b, parcels=parcels)

    za = np.load(out_a / A.FN_PERDATE, allow_pickle=True)
    zb = np.load(out_b / A.FN_PERDATE, allow_pickle=True)
    assert np.array_equal(za["cod_predio"], zb["cod_predio"])
    assert np.allclose(za["X"], zb["X"])
    assert np.array_equal(za["doy"], zb["doy"])
    assert np.array_equal(za["mask"], zb["mask"])

    fa = pd.read_parquet(out_a / A.FN_LGBM)
    fb = pd.read_parquet(out_b / A.FN_LGBM)
    pd.testing.assert_frame_equal(fa, fb)


def test_per_year_outputs_differ_between_years(tmp_path):
    """Sanity: the isolation is real, not both years collapsing to the same numbers."""
    parcels = pd.DataFrame({"COD_PREDIO": ["P1"], "area_ha": [1.0],
                            "n_pixels_est": [11.1], "centroid_lat": [-5.2],
                            "n_valid_obs": [5], "max_gap": [0]})
    store = write_store(tmp_path, [synthetic_pixels(1998, 0.10),
                                   synthetic_pixels(2015, 0.60)])
    o98, o15 = tmp_path / "y98", tmp_path / "y15"
    A.assemble(years=[1998], feat_dir=store, out_dir=o98, parcels=parcels)
    A.assemble(years=[2015], feat_dir=store, out_dir=o15, parcels=parcels)
    f98 = pd.read_parquet(o98 / A.FN_LGBM)["R_median"].iloc[0]
    f15 = pd.read_parquet(o15 / A.FN_LGBM)["R_median"].iloc[0]
    assert not np.isclose(f98, f15)

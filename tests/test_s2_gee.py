"""Unit tests for the Sentinel-2 extraction (docs/s2_labelling/plan.md).

Four things here fail silently and would corrupt the whole feature store:

* **band renaming** — if it does not land exactly on ``indices.BANDS`` then
  ``add_indices`` computes NDVI from the wrong bands and nothing errors;
* **the S2 scale factor** — applying the Landsat offset, or applying ``/10000`` twice,
  produces plausible-looking reflectance that is simply wrong;
* **the Aug-Jul window boundary** — off by one on either side of 1 January and half the
  parcels get the wrong agricultural year;
* **``_call_with_deadline``** — a GEE hang is not an error, so nothing else catches it.
"""

from __future__ import annotations

import datetime as dt
import time

import numpy as np
import pandas as pd
import pytest

from crop_classifier.features.indices import BANDS, S2_SCALE, add_indices, scale_sr_s2
from crop_classifier.features.landsat_gee import ChunkTimeout, _call_with_deadline
from crop_classifier.features.s2_gee import (
    MIN_PIXELS,
    NDVI_PX_COLS,
    S2_BANDS,
    S2_NDVI_BAND,
    _consolidate,
    ag_year,
    harmonisation_check,
    trace_window,
    window_key,
)


class TestBands:
    def test_renaming_lands_exactly_on_the_shared_channel_set(self):
        """`indices.add_indices` must work unchanged on the S2 store."""
        assert list(S2_BANDS.values()) == BANDS

    def test_source_bands_are_the_10m_set_plus_the_two_swirs(self):
        assert list(S2_BANDS) == ["B2", "B3", "B4", "B8", "B11", "B12"]

    def test_indices_compute_on_renamed_s2_columns(self):
        df = pd.DataFrame({b: [3000.0] for b in BANDS})
        out = add_indices(scale_sr_s2(df))
        for c in ("NDVI", "EVI", "NDWI", "NDMI", "BSI"):
            assert c in out.columns and out[c].notna().all()

    def test_the_per_pixel_ndvi_band_is_not_called_NDVI(self):
        """`add_indices` writes a column called `NDVI` from the band *medians*.

        The per-pixel quantiles are a different quantity — a quantile of a ratio is not the
        ratio of the quantiles — so sharing the name would let one silently overwrite the
        other depending on call order, and the plotted median could then sit outside its
        own p25-p75 band.
        """
        assert S2_NDVI_BAND != "NDVI"
        assert "NDVI" not in NDVI_PX_COLS
        assert all(c.startswith("NDVI_px_") for c in NDVI_PX_COLS)

    def test_the_quartile_columns_do_not_collide_with_any_model_channel(self):
        from crop_classifier.features.indices import CHANNELS
        assert not set(NDVI_PX_COLS) & set(CHANNELS)


class TestScale:
    def test_scale_is_applied_once_and_only_once(self):
        df = pd.DataFrame({b: [2500.0] for b in BANDS})
        out = scale_sr_s2(df)
        assert out["NIR"].iloc[0] == pytest.approx(0.25)
        # applying it again must NOT silently look plausible in the same range
        twice = scale_sr_s2(out)
        assert twice["NIR"].iloc[0] == pytest.approx(0.25 * S2_SCALE)

    def test_no_landsat_offset_is_applied(self):
        """Landsat's -0.2 offset on S2 DNs would send low reflectance negative."""
        out = scale_sr_s2(pd.DataFrame({b: [500.0] for b in BANDS}))
        assert out["B"].iloc[0] == pytest.approx(0.05)

    def test_values_are_clipped_to_reflectance_range(self):
        out = scale_sr_s2(pd.DataFrame({b: [-100.0, 14000.0] for b in BANDS}))
        assert out["R"].min() >= 0.0 and out["R"].max() <= 1.0


class TestWindows:
    @pytest.mark.parametrize("date,start,end", [
        # August 1 itself opens a new agricultural year
        ("2021-08-01", dt.date(2021, 8, 1), dt.date(2022, 7, 31)),
        # July 31 closes the previous one
        ("2022-07-31", dt.date(2021, 8, 1), dt.date(2022, 7, 31)),
        # either side of 1 January must stay in the SAME agricultural year
        ("2021-12-31", dt.date(2021, 8, 1), dt.date(2022, 7, 31)),
        ("2022-01-01", dt.date(2021, 8, 1), dt.date(2022, 7, 31)),
        ("2022-08-01", dt.date(2022, 8, 1), dt.date(2023, 7, 31)),
    ])
    def test_ag_year_boundary(self, date, start, end):
        assert ag_year(date) == (start, end)

    def test_trace_is_24_months_centred(self):
        lo, hi = trace_window("2022-06-15")
        assert (lo, hi) == (dt.date(2021, 6, 15), dt.date(2023, 6, 15))

    def test_ag_year_is_always_inside_the_trace_window(self):
        """One extraction has to serve both, so containment is a hard requirement."""
        for d in pd.date_range("2019-01-01", "2025-12-01", freq="17D"):
            lo, hi = trace_window(d)
            a0, a1 = ag_year(d)
            assert lo <= a0 and a1 <= hi

    def test_window_key_groups_parcels_sharing_a_window(self):
        assert window_key("2022-06-15") == window_key("2022-06-02")
        assert window_key("2022-06-15") != window_key("2022-07-15")


class TestConsolidate:
    def _frame(self, rows):
        return pd.DataFrame(rows)

    def test_pixel_floor_drops_thin_dates(self):
        px = self._frame([
            {"COD_PREDIO": "a", "date": "2022-01-01", "doy": 1, "n_px": 40.0,
             **{b: 3000.0 for b in BANDS}},
            {"COD_PREDIO": "a", "date": "2022-01-11", "doy": 11, "n_px": 2.0,
             **{b: 3000.0 for b in BANDS}},
        ])
        parcels = pd.DataFrame({"COD_PREDIO": ["a"], "eroded": [True]})
        out = _consolidate(px, parcels, MIN_PIXELS)
        assert len(out) == 1 and out["n_px"].iloc[0] == 40.0

    def test_same_date_granule_split_is_pixel_weighted_not_dropped(self):
        """A parcel straddling a tile edge is reduced twice; both halves are real data."""
        px = self._frame([
            {"COD_PREDIO": "a", "date": "2022-01-01", "doy": 1, "n_px": 30.0,
             **{b: 1000.0 for b in BANDS}},
            {"COD_PREDIO": "a", "date": "2022-01-01", "doy": 1, "n_px": 10.0,
             **{b: 5000.0 for b in BANDS}},
        ])
        parcels = pd.DataFrame({"COD_PREDIO": ["a"], "eroded": [True]})
        out = _consolidate(px, parcels, MIN_PIXELS)
        assert len(out) == 1
        assert out["n_px"].iloc[0] == 40.0
        assert out["NIR"].iloc[0] == pytest.approx(2000.0)   # (30*1000 + 10*5000)/40

    def test_ndvi_quartiles_are_merged_across_a_granule_split_too(self):
        """Added with the ribbon: a column the merge forgets is silently dropped by the
        `_combine` rebuild, and the parcel loses its band on exactly the split dates."""
        px = self._frame([
            {"COD_PREDIO": "a", "date": "2022-01-01", "doy": 1, "n_px": 30.0,
             **{b: 1000.0 for b in BANDS},
             "NDVI_px_p25": 0.1, "NDVI_px_p50": 0.2, "NDVI_px_p75": 0.3},
            {"COD_PREDIO": "a", "date": "2022-01-01", "doy": 1, "n_px": 10.0,
             **{b: 5000.0 for b in BANDS},
             "NDVI_px_p25": 0.5, "NDVI_px_p50": 0.6, "NDVI_px_p75": 0.7},
        ])
        parcels = pd.DataFrame({"COD_PREDIO": ["a"], "eroded": [True]})
        out = _consolidate(px, parcels, MIN_PIXELS)
        assert len(out) == 1
        for c in NDVI_PX_COLS:
            assert out[c].notna().all(), f"{c} lost in the granule merge"
        assert out["NDVI_px_p50"].iloc[0] == pytest.approx(0.3)  # (30*.2 + 10*.6)/40
        # the merged row must still be a usable band, not a crossed one
        assert (out["NDVI_px_p25"] <= out["NDVI_px_p50"]).all()
        assert (out["NDVI_px_p50"] <= out["NDVI_px_p75"]).all()

    def test_a_store_without_the_quartiles_still_consolidates(self):
        """The pre-2026-08-13 store has no quantile columns; it must not raise."""
        px = self._frame([
            {"COD_PREDIO": "a", "date": "2022-01-01", "doy": 1, "n_px": 30.0,
             **{b: 1000.0 for b in BANDS}},
            {"COD_PREDIO": "a", "date": "2022-01-01", "doy": 1, "n_px": 10.0,
             **{b: 5000.0 for b in BANDS}},
        ])
        parcels = pd.DataFrame({"COD_PREDIO": ["a"], "eroded": [True]})
        assert len(_consolidate(px, parcels, MIN_PIXELS)) == 1

    def test_erosion_flag_is_carried_through(self):
        px = self._frame([{"COD_PREDIO": "a", "date": "2022-01-01", "doy": 1,
                           "n_px": 40.0, **{b: 3000.0 for b in BANDS}}])
        parcels = pd.DataFrame({"COD_PREDIO": ["a"], "eroded": [False]})
        out = _consolidate(px, parcels, MIN_PIXELS)
        assert out["eroded"].iloc[0] is np.False_ or out["eroded"].iloc[0] is False


class TestHarmonisationCheck:
    """The check's own two failure modes: seeing the growing season, and reading an index.

    An unremoved baseline-04.00 offset is an additive change in *band* reflectance, so the
    verdict is taken from bands. The seasonal cycle is fitted out per parcel rather than
    binned out, and a placebo cut one year earlier gives the noise floor.
    """

    def _synthetic(self, offset_dn=0.0, cut="2022-01-25"):
        """Three years of a strongly seasonal parcel, with an optional +DN band offset."""
        dates = pd.date_range("2019-06-01", "2022-12-31", freq="10D")
        cutd = pd.Timestamp(cut)
        rows = []
        for p in range(20):
            for d in dates:
                season = 1 + 0.5 * np.sin(2 * np.pi * (d.dayofyear - 60) / 365.25)
                off = offset_dn if d >= cutd else 0.0
                rows.append({"COD_PREDIO": f"p{p}", "date": d.strftime("%Y-%m-%d"),
                             "n_px": 40.0,
                             "B": 800.0 * season + off, "G": 1200.0 * season + off,
                             "R": 1000.0 * season + off, "NIR": 3000.0 * season + off,
                             "SWIR1": 2400.0 * season + off,
                             "SWIR2": 1800.0 * season + off})
        return pd.DataFrame(rows)

    def test_a_pure_growing_season_reports_no_step(self, tmp_path):
        """The version this replaced reported +0.055 on data with no step at all."""
        out = harmonisation_check(self._synthetic(offset_dn=0.0), out_dir=tmp_path)
        band = out[(out.cut == "baseline_04.00") & (out.channel == "SWIR1")].iloc[0]
        assert abs(band.step_corrected) < 0.005
        assert "harmonised" in band.verdict

    def test_a_real_offset_is_detected(self, tmp_path):
        """An unremoved +1000 DN offset must NOT be waved through."""
        out = harmonisation_check(self._synthetic(offset_dn=1000.0), out_dir=tmp_path)
        band = out[(out.cut == "baseline_04.00") & (out.channel == "SWIR1")].iloc[0]
        assert band.step_corrected == pytest.approx(0.10, abs=0.02)
        assert "CHECK" in band.verdict

    def test_the_verdict_never_comes_from_an_index(self, tmp_path):
        """NDVI absorbs real interannual change, so it cannot decide this question."""
        out = harmonisation_check(self._synthetic(), out_dir=tmp_path)
        ndvi = out[out.channel == "NDVI"]
        assert (~ndvi["diagnostic"]).all()
        assert ndvi["verdict"].str.contains("context only").all()

    def test_the_placebo_cut_is_always_reported(self, tmp_path):
        out = harmonisation_check(self._synthetic(), out_dir=tmp_path)
        assert set(out["cut"]) == {"baseline_04.00", "placebo_-1y"}


class TestDeadline:
    def test_raises_on_a_hang(self):
        """The measured failure this exists for: alive, silent, no exception, 13.4 h."""
        with pytest.raises(ChunkTimeout):
            _call_with_deadline(lambda: time.sleep(5), 0.2)

    def test_returns_normally_when_fast(self):
        assert _call_with_deadline(lambda: 42, 5.0) == 42

    def test_reraises_the_callees_own_error(self):
        with pytest.raises(ValueError, match="boom"):
            _call_with_deadline(lambda: (_ for _ in ()).throw(ValueError("boom")), 5.0)


class TestTransientClassification:
    """GEE throttles must be waited out, not treated as failures.

    "Too many concurrent aggregations" killed a running extraction at 144 of 215 chunks.
    It is a refusal to serve *right now*, not an error in the request.
    """

    @pytest.mark.parametrize("msg", [
        "EEException: Too many concurrent aggregations.",
        "Exceeded Earth Engine concurrency limit. Restricted Mode.",
        "429 Too Many Requests",
        "Connection reset by peer",
    ])
    def test_throttles_and_network_errors_are_retried(self, msg):
        from crop_classifier.features.landsat_gee import _TRANSIENT_MSGS, _is_split_error
        low = msg.lower()
        assert any(s in low for s in _TRANSIENT_MSGS), msg
        assert not _is_split_error(Exception(msg))

    def test_a_computation_too_big_error_is_not_retried_but_split(self):
        from crop_classifier.features.landsat_gee import _is_split_error
        assert _is_split_error(Exception("User memory limit exceeded"))

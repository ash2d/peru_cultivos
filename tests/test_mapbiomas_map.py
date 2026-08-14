"""MapBiomas code -> class mapping (plan §13). No GEE needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crop_classifier.perennial import mapbiomas as M


# ---- the mapping -----------------------------------------------------------------------
def test_known_codes_map_as_documented():
    assert M.code_to_class(40) == "ANNUAL"            # rice — verified empirically
    assert M.code_to_class(62) == "ANNUAL"            # cotton
    assert M.code_to_class(36) == "PERENNIAL"
    assert M.code_to_class(46) == "PERENNIAL"         # coffee
    assert M.code_to_class(15) == "PASTURE_FALLOW"
    assert M.code_to_class(21) == "MOSAIC"            # deliberately its own category
    assert M.code_to_class(3) is None                 # natural formation -> excluded
    assert M.code_to_class(24) is None                # urban -> excluded


def test_unknown_code_raises_rather_than_passing_through():
    """A silent pass-through would corrupt the benchmark, so this must be loud."""
    with pytest.raises(KeyError, match="unknown MapBiomas code"):
        M.code_to_class(999)
    with pytest.raises(KeyError):
        M.code_name(999)


def test_mosaic_is_not_silently_assigned_to_a_class():
    """Class 21 is ~63 % of our parcels; folding it into a class by default would decide
    the benchmark by fiat."""
    assert M.LEGEND[21][1] == "MOSAIC"
    assert "MOSAIC" not in {"PERENNIAL", "ANNUAL", "PASTURE_FALLOW"}


# ---- per-parcel histogram reduction -------------------------------------------------------
def test_histogram_to_row_takes_the_mode_and_reports_purity():
    row = M.histogram_to_row({"40": 6, "21": 2}, "P1", 1999)
    assert row["mb_code"] == 40
    assert row["mb_class"] == "ANNUAL"
    assert row["mb_purity"] == pytest.approx(0.75)
    assert row["mb_n_pixels"] == 8


def test_histogram_to_row_reports_class_fractions():
    row = M.histogram_to_row({"40": 5, "36": 3, "3": 2}, "P1", 1999)
    assert row["mb_frac_ANNUAL"] == pytest.approx(0.5)
    assert row["mb_frac_PERENNIAL"] == pytest.approx(0.3)
    assert row["mb_frac_OTHER"] == pytest.approx(0.2)     # forest -> not one of ours
    assert sum(row[f"mb_frac_{c}"] for c in
               ("PERENNIAL", "ANNUAL", "PASTURE_FALLOW", "MOSAIC", "OTHER")) \
        == pytest.approx(1.0)


def test_empty_histogram_is_missing_not_zero():
    row = M.histogram_to_row({}, "P1", 1999)
    assert pd.isna(row["mb_code"]) and row["mb_class"] is None
    assert row["mb_n_pixels"] == 0
    assert np.isnan(row["mb_purity"])


def test_unknown_code_in_a_histogram_raises():
    with pytest.raises(KeyError):
        M.histogram_to_row({"999": 5}, "P1", 1999)


# ---- the comparison ------------------------------------------------------------------------
@pytest.fixture
def panel_and_parcels():
    panel = pd.DataFrame([
        {"COD_PREDIO": "A", "year": 1999, "mb_code": 40, "mb_class": "ANNUAL",
         "mb_purity": 1.0},
        {"COD_PREDIO": "B", "year": 1999, "mb_code": 21, "mb_class": "MOSAIC",
         "mb_purity": 0.9},
        {"COD_PREDIO": "C", "year": 1999, "mb_code": 15, "mb_class": "PASTURE_FALLOW",
         "mb_purity": 0.4},
    ])
    parcels = pd.DataFrame({"COD_PREDIO": ["A", "B", "C"], "year": [1999] * 3,
                            "label": ["ANNUAL", "PERENNIAL", "PASTURE_FALLOW"]})
    return panel, parcels


def test_mosaic_excluded_by_default(panel_and_parcels):
    panel, parcels = panel_and_parcels
    r = M.agreement_vs_pett(panel, parcels)
    assert r["n"] == 2                     # the MOSAIC parcel is not scored
    assert r["accuracy"] == pytest.approx(1.0)


def test_mosaic_can_be_counted_as_a_class(panel_and_parcels):
    panel, parcels = panel_and_parcels
    r = M.agreement_vs_pett(panel, parcels, mosaic_as="PERENNIAL")
    assert r["n"] == 3
    assert r["accuracy"] == pytest.approx(1.0)
    r2 = M.agreement_vs_pett(panel, parcels, mosaic_as="ANNUAL")
    assert r2["accuracy"] == pytest.approx(2 / 3)


def test_purity_filter_drops_mixed_parcels(panel_and_parcels):
    panel, parcels = panel_and_parcels
    r = M.agreement_vs_pett(panel, parcels, min_purity=0.5)
    assert r["n"] == 1                     # only the pure rice parcel survives


def test_code_histogram_labels_every_code(panel_and_parcels):
    panel, _ = panel_and_parcels
    h = M.code_histogram(panel)
    assert set(h["name"]) == {"Rice", "Mosaic of uses", "Pasture"}
    assert h["share"].sum() == pytest.approx(1.0)

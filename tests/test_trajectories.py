"""Trajectory smoothing and change detection.

⛔ CLOSED ROUTE. Per-parcel annual trajectories were abandoned — the panel gate failed on
every arm (docs/RESULTS.md §3, §4.4). This module is built, unit-tested and **stays unrun**.
These tests are kept so the code does not rot if the estimand is ever revived.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crop_classifier.perennial import trajectories as T

pytestmark = pytest.mark.closed_route

GAP = T.GAP
ANN, PAS, PER = 0, 1, 2
CLASSES = ["ANNUAL", "PASTURE_FALLOW", "PERENNIAL"]


# --- series construction ---
def test_abstained_years_become_gaps_not_guesses():
    preds = pd.DataFrame({
        "COD_PREDIO": ["A", "A", "A"],
        "year": [2000, 2001, 2002],
        "pred_label": ["ANNUAL", None, "ANNUAL"],
        "abstained": [False, True, False]})
    series, observed, classes, years = T.to_series(preds, classes=CLASSES)
    assert series[0].tolist() == [ANN, GAP, ANN]
    assert observed[0].tolist() == [True, False, True]


def test_missing_parcel_years_become_gaps():
    preds = pd.DataFrame({"COD_PREDIO": ["A"], "year": [2000],
                          "pred_label": ["ANNUAL"], "abstained": [False]})
    series, _, _, _ = T.to_series(preds, years=[2000, 2001], classes=CLASSES)
    assert series[0].tolist() == [ANN, GAP]


# --- mode filter ---
def test_mode_filter_removes_a_single_year_spike():
    s = np.array([[PER, PER, ANN, PER, PER]])
    assert T.mode_filter(s)[0].tolist() == [PER] * 5


def test_mode_filter_keeps_a_real_sustained_change():
    s = np.array([[ANN, ANN, ANN, PER, PER, PER]])
    assert T.mode_filter(s)[0].tolist() == [ANN, ANN, ANN, PER, PER, PER]


def test_mode_filter_leaves_gaps_as_gaps():
    s = np.array([[PER, GAP, PER]])
    out = T.mode_filter(s)
    assert out[0, 1] == GAP
    assert out[0, 0] == PER and out[0, 2] == PER


def test_mode_filter_does_not_invent_a_change_on_a_tie():
    # window {ANN, PER} is a 1-1 tie -> the parcel keeps its own class
    s = np.array([[ANN, PER]])
    out = T.mode_filter(s)
    assert out[0].tolist() == [ANN, PER]


# --- minimum-duration rule ---
def test_two_year_excursion_does_not_fire():
    # PERENNIAL x4, ANNUAL x2, PERENNIAL x4 — the 2-year dip is not a transition
    s = np.array([[PER] * 4 + [ANN] * 2 + [PER] * 4])
    years = list(range(2000, 2010))
    tr = T.detect_transitions(s, years, CLASSES, ["A"], min_duration=3)
    assert tr.empty


def test_three_year_change_does_fire():
    s = np.array([[ANN] * 4 + [PER] * 4])
    years = list(range(2000, 2008))
    tr = T.detect_transitions(s, years, CLASSES, ["A"], min_duration=3)
    assert len(tr) == 1
    r = tr.iloc[0]
    assert (r["from_class"], r["to_class"]) == ("ANNUAL", "PERENNIAL")
    assert r["year_of_change"] == 2004
    assert r["n_years_before"] == 4 and r["n_years_after"] == 4


def test_min_duration_is_counted_in_observed_years_not_calendar_years():
    """Gaps must not break a run: ANNUAL, gap, ANNUAL still counts as two observed years."""
    s = np.array([[ANN, GAP, ANN, ANN, PER, PER, GAP, PER]])
    years = list(range(2000, 2008))
    tr = T.detect_transitions(s, years, CLASSES, ["A"], min_duration=3)
    assert len(tr) == 1
    assert tr.iloc[0]["year_of_change"] == 2004


def test_short_series_yields_no_transitions():
    s = np.array([[ANN, PER]])
    assert T.detect_transitions(s, [2000, 2001], CLASSES, ["A"]).empty


def test_flickering_series_yields_no_sustained_transition():
    s = np.array([[ANN, PER] * 5])
    years = list(range(2000, 2010))
    assert T.detect_transitions(s, years, CLASSES, ["A"], min_duration=3).empty


# --- flicker rate ---
def test_flicker_rate_flags_the_noisy_parcel_only():
    s = np.array([[ANN, PER] * 5,            # 9 changes in 10 years -> flickering
                  [PER] * 10,                # stable
                  [ANN] * 5 + [PER] * 5])    # one real change
    assert T.flicker_rate(s).tolist() == [True, False, False]


def test_flicker_ignores_gaps():
    s = np.array([[PER, GAP, PER, GAP, PER]])
    assert T.flicker_rate(s).tolist() == [False]


# --- aggregation ---
def test_transition_matrix_counts_directions():
    tr = pd.DataFrame({"from_class": ["ANNUAL", "ANNUAL", "PERENNIAL"],
                       "to_class": ["PERENNIAL", "PERENNIAL", "ANNUAL"],
                       "year_of_change": [2001, 2005, 2003]})
    m = T.transition_matrix(tr, CLASSES)
    assert m.loc["ANNUAL", "PERENNIAL"] == 2
    assert m.loc["PERENNIAL", "ANNUAL"] == 1
    assert T.transition_matrix(tr, CLASSES, period=(2000, 2002)).loc[
        "ANNUAL", "PERENNIAL"] == 1


def test_class_share_ignores_unobserved_parcels():
    s = np.array([[PER, GAP], [ANN, ANN]])
    df = T.class_share_by_year(s, [2000, 2001], CLASSES)
    y2001 = df[df.year == 2001].set_index("class")["share"]
    assert y2001["ANNUAL"] == 1.0            # the gap parcel is excluded, not counted
    assert y2001["PERENNIAL"] == 0.0
    assert df[df.year == 2001]["n_observed"].iloc[0] == 1


def test_class_share_uses_area_and_sampling_weights():
    s = np.array([[PER], [ANN]])
    df = T.class_share_by_year(s, [2000], CLASSES, area_ha=np.array([1.0, 3.0]),
                               weights=np.array([2.0, 2.0]))
    by = df.set_index("class")
    assert by.loc["PERENNIAL", "area"] == 2.0     # 1 ha x weight 2
    assert by.loc["ANNUAL", "area"] == 6.0
    assert by.loc["PERENNIAL", "share"] == 0.25

"""Tests for the temporal-OOD work (docs/all_peru/temporal_ood_plan.md steps 2 and 3).

These pin **decisions**, not implementations:

* the year-leak ranking is computed **within region**, because cohort is confounded with
  place — an unconditional ranking measures geography;
* LODYO re-scores existing LODO predictions per cohort, so place *and* year are out of
  distribution at once. This is the evaluation that catches what LOYO cannot;
* the 3a acceptance thresholds are registered constants and the verdict is the conjunction of
  all three legs, so a later edit cannot quietly relax one;
* quantile alignment writes to a **suffixed** bundle directory and never touches the audited
  panel bundle (T-D4).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from crop_classifier.allperu import density as D
from crop_classifier.allperu import oli_overlap as O
from crop_classifier.allperu import yearleak as YL


# ------------------------------------------------------------------------------------
# 2a — the year-leak audit
# ------------------------------------------------------------------------------------
def test_eta_squared_is_one_when_a_feature_is_the_group_and_zero_when_it_is_noise():
    g = np.repeat([0, 1, 2], 50)
    perfect = g.astype(float)
    rng = np.random.default_rng(0)
    noise = rng.normal(size=len(g))
    e = YL.eta_squared(np.c_[perfect, noise], g)
    assert e[0] == pytest.approx(1.0)
    assert e[1] < 0.10


def test_year_leak_ranking_is_computed_within_region_not_marginally():
    """A feature that is purely a *place* effect must not be ranked as a year leak. Titling
    swept region by region, so cohort and region are confounded and the marginal ANOVA cannot
    tell them apart."""
    # region and cohort are perfectly aligned; the feature is a pure region effect
    region = np.repeat([0, 1, 2, 3], 40)
    cohort = region.copy()
    place_only = region.astype(float) + np.random.default_rng(1).normal(0, 0.01, len(region))
    x = place_only[:, None]

    marginal = YL.eta_squared(x, cohort)[0]
    within = YL.eta_squared(YL._demean_by(x, region), cohort)[0]
    assert marginal > 0.95          # looks like a devastating year leak
    assert within < 0.05            # and is nothing of the kind


def test_top_k_returns_the_within_region_ranking_by_default():
    audit = pd.DataFrame({"feature": ["a", "b", "c"],
                          "eta2_year": [0.9, 0.1, 0.5],
                          "eta2_year_within_region": [0.01, 0.30, 0.20]})
    assert YL.top_k(audit, 2) == ["b", "c"]
    assert YL.top_k(audit, 2, by="eta2_year") == ["a", "c"]


# ------------------------------------------------------------------------------------
# 2b — LODYO, the joint out-of-distribution evaluation
# ------------------------------------------------------------------------------------
def test_lodyo_scores_lodo_predictions_per_cohort(tmp_path, monkeypatch):
    """The decision: hold out the *department* (LODO already did) and then vary the *year*.
    LOYO holds place approximately fixed and so cannot see spatial memorisation; this can."""
    import geopandas as gpd
    from shapely.geometry import Point

    from crop_classifier.allperu import loyo as LO

    monkeypatch.setenv("CC_PROC", str(tmp_path))
    n = 800
    rng = np.random.default_rng(0)
    cods = [f"p{i}" for i in range(n)]
    years = np.where(np.arange(n) < n // 2, 2000, 2005)
    y_true = rng.integers(0, 3, n)
    # cohort 2000 is scored perfectly, cohort 2005 at chance -> the split must be visible
    y_pred = np.where(years == 2000, y_true, (y_true + 1) % 3)
    pd.DataFrame({"COD_PREDIO": cods, "y_true": y_true, "y_pred": y_pred,
                  "dept": "X"}).to_parquet(tmp_path / "lodo_predictions_t.parquet")
    gpd.GeoDataFrame({"COD_PREDIO": cods, "year": years.astype(float)},
                     geometry=[Point(0, 0)] * n,
                     crs=4326).to_parquet(tmp_path / "modeling_parcels.parquet")

    res = LO.lodo_by_cohort(tag="t", min_parcels=100)
    assert sorted(res["cohort"]) == [2000, 2005]
    got = dict(zip(res.cohort, res.macro_f1))
    assert got[2000] == pytest.approx(1.0)
    assert got[2005] < 0.1
    assert (tmp_path / "lodyo_summary_t.json").exists()
    summ = json.load(open(tmp_path / "lodyo_summary_t.json"))
    assert summ["n_cohorts"] == 2


# ------------------------------------------------------------------------------------
# 3 — OLI overlap
# ------------------------------------------------------------------------------------
def test_oli_acceptance_thresholds_are_registered_constants():
    """Registered before the test was run (plan §3a). Pinned so a later edit that relaxes one
    is a visible change to the criterion, not a quiet change to the answer."""
    assert O.MAX_MEDIAN_ABS_DP == pytest.approx(0.05)
    assert O.MIN_CLASS_AGREEMENT == pytest.approx(0.95)
    assert O.MAX_CONTROL_SHIFT == pytest.approx(0.02)


def test_oli_verdict_is_the_conjunction_of_all_three_legs():
    """Passing the per-parcel noise leg is not passing 3a. The control-pool shift is the leg
    the estimand rests on, and it must be able to fail the whole test on its own."""
    assert O.verdict(0.01, 0.99, 0.001)["pass"]
    # each leg alone is sufficient to fail
    assert not O.verdict(0.20, 0.99, 0.001)["pass"]
    assert not O.verdict(0.01, 0.60, 0.001)["pass"]
    assert not O.verdict(0.01, 0.99, 0.10)["pass"]
    # the control shift is signed-symmetric: a shift the *other* way fails identically
    assert not O.verdict(0.01, 0.99, -0.10)["pass"]


def test_oli_measured_result_fails_on_the_control_leg_only():
    """The measured 2015/2019/2022 numbers (RESULTS.md §9.3). Recorded as a test so a later
    refactor that changes the verdict wiring is caught: harmonised OLI passes the per-parcel
    noise leg and fails the control leg, and the raw arm fails it less."""
    harmonised = O.verdict(0.0311, 0.6164, -0.1073)
    raw = O.verdict(0.0293, 0.6309, -0.0423)
    assert harmonised["pass_median_dp"] and raw["pass_median_dp"]
    assert not harmonised["pass_control"] and not raw["pass_control"]
    assert not harmonised["pass"] and not raw["pass"]


def test_oli_missions_are_l8_l9_only_so_the_overlap_is_a_clean_contrast():
    """The whole design is: same parcel-year, two *disjoint* sensor sets. If OLI extraction
    also pulled L7 the comparison would be against itself."""
    assert O.OLI_MISSIONS == {"L8", "L9"}
    assert "L7" not in O.OLI_MISSIONS and "L5" not in O.OLI_MISSIONS


def test_oli_bundles_never_collide_with_the_audited_panel_bundle():
    """T-D4: nothing overwrites an existing table. The harmonised and raw arms must also be
    distinguishable from each other."""
    assert O.bundle_dir(2019).name == "2019_oli"
    assert O.bundle_dir(2019).name != "2019"


# ------------------------------------------------------------------------------------
# 2c — quantile alignment
# ------------------------------------------------------------------------------------
def test_quantile_align_maps_onto_the_reference_distribution():
    rng = np.random.default_rng(0)
    ref = rng.normal(10, 2, 5000)
    vals = rng.normal(0, 1, 5000)
    out = D.quantile_align(vals, ref)
    assert abs(np.mean(out) - 10) < 0.2
    assert abs(np.std(out) - 2) < 0.2
    # and it is monotone — rank order is preserved, which is what makes it a *mapping*
    assert (np.argsort(out) == np.argsort(vals)).all()


def test_quantile_align_is_a_noop_on_a_degenerate_feature():
    v = np.full(100, 3.0)
    out = D.quantile_align(v, np.arange(100.0))
    assert np.allclose(out, v)


def test_aligned_bundles_are_written_to_a_suffixed_directory(tmp_path):
    from crop_classifier.features import assemble as asm

    base = tmp_path / "2019"
    base.mkdir()
    vals = np.linspace(0.1, 0.4, 50)
    pd.DataFrame({"COD_PREDIO": [f"p{i}" for i in range(50)],
                  "NDVI_median": vals}).to_parquet(base / asm.FN_LGBM)
    ref = pd.DataFrame({"NDVI_median": np.linspace(0.5, 0.8, 50)})
    D.write_aligned_panel_bundles(["NDVI_median"], ref, [2019], panel_feat_dir=tmp_path)

    original = pd.read_parquet(base / asm.FN_LGBM)
    aligned = pd.read_parquet(tmp_path / "2019_qmap" / asm.FN_LGBM)
    assert np.allclose(original["NDVI_median"], vals)               # untouched
    assert aligned["NDVI_median"].min() >= 0.5                      # mapped onto ref


# ------------------------------------------------------------------------------------
# 3c — refitting the OLI correction (RESULTS.md §9.5)
# ------------------------------------------------------------------------------------
def test_refit_default_is_offset_only_because_a_slope_is_not_identified():
    """The decision: on this data only a per-band *offset* is estimable. Same-day parcel
    medians disagree more than parcels differ from each other (SD of the difference exceeds
    the SD of either sensor's values), so any slope fitted from them is diluted — OLS gives
    0.116 in blue against a physical ~0.85. Defaulting to a slope would silently ship that."""
    from crop_classifier.allperu import oli_refit as RF

    fit = {"NIR": {"slope": 0.52, "intercept": 0.11, "theilsen_slope": 0.83,
                   "theilsen_intercept": 0.03, "raw_bias_l7_minus_oli": -0.0189}}
    assert RF.as_coefficient_map(fit) == {"NIR": (1.0, -0.0189)}          # default
    assert RF.as_coefficient_map(fit, kind="ols")["NIR"][0] == 0.52       # reproducible
    assert RF.as_coefficient_map(fit, kind="theilsen")["NIR"][0] == 0.83


def test_custom_coefficients_override_the_published_ones_end_to_end():
    """`assemble(harmonize_coefficients=...)` must actually reach `oli_to_etm`, otherwise a
    refit would silently apply Roy's numbers instead."""
    from crop_classifier.perennial.harmonization import ROY2016_OLI_TO_ETM, oli_to_etm

    px = pd.DataFrame({"mission": [8, 5], "NIR": [0.30, 0.30], "R": [0.10, 0.10]})
    roy = oli_to_etm(px)
    mine = oli_to_etm(px, coefficients={"NIR": (1.0, -0.02), "R": (1.0, 0.01)})
    assert roy.loc[0, "NIR"] == pytest.approx(
        ROY2016_OLI_TO_ETM["NIR"][0] * 0.30 + ROY2016_OLI_TO_ETM["NIR"][1])
    assert mine.loc[0, "NIR"] == pytest.approx(0.28)
    assert mine.loc[0, "R"] == pytest.approx(0.11)
    # non-OLI rows are never touched, whichever coefficients are used
    assert mine.loc[1, "NIR"] == pytest.approx(0.30)


def test_a_global_linear_map_cannot_remove_a_between_class_difference():
    """The §9.5.4 result, as an arithmetic fact rather than a measurement: any map of the form
    `a*x + b` applied to both classes shifts them together. It can null the average error and
    leave the class *spread* untouched — which is exactly what every OLI arm did."""
    perennial_err, pasture_err = 0.0704, 0.0452      # measured raw NDVI bias, RESULTS §9.5.4
    spread = perennial_err - pasture_err
    for a, b in [(1.0, -0.05), (0.9, 0.0), (1.2, 0.03)]:
        assert (a * perennial_err + b) - (a * pasture_err + b) == pytest.approx(a * spread)
    # only a=0 removes it, and that discards the signal along with the bias
    assert spread == pytest.approx(0.0252, abs=1e-4)

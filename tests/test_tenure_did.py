"""Tests for the two-period tenure DiD (docs/RESULTS.md §7).

Every test here pins a **decision**, not an implementation detail. The decisions are the ones
that, if silently changed by a later refactor, would leave every downstream number looking
plausible and wrong:

* **R1** the at-risk restriction — what makes both arms the same kind of land;
* **R4** the parcel-specific pre-period — what makes "before" actually mean before;
* **N-D9** the control is ``NO INSCRITO`` at both observations, not "everyone else";
* the placebo's **equivalence** logic and its three outcomes;
* the **horizon-normalised** band, so placebo contrasts of different lengths are held to the
  same stringency;
* the pilot's published coefficients (N-D7 keeps them on the record);
* **v3 (the reopening)** — the pre-trend correction: the formula, its error propagation, the
  amplification factor derived from window midpoints rather than hard-coded, and the
  registered decision rule on both sides of its threshold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crop_classifier.allperu import tenure_did as TD

PROC = Path("data/processed/all_peru")
PANEL = PROC / "panel_predictions_nolat_aug_yleak10.parquet"


# ------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------
def make_panel(parcels: dict[str, dict], years=range(1999, 2024)) -> pd.DataFrame:
    """``{parcel: {...attrs, 'p': prob or callable(year)}}`` -> a merged-input frame."""
    rows = []
    for cod, a in parcels.items():
        for y in years:
            p = a["p"](y) if callable(a["p"]) else a["p"]
            rows.append({
                "COD_PREDIO": cod, "year": y, "prob_PERENNIAL": p,
                "abstained": False, "pett_label": a.get("pett_label", "ANNUAL"),
                "tenure": a.get("tenure", "NO INSCRITO"),
                "became_registered": a.get("became_registered", False),
                "reg_year": a.get("reg_year", 2004),
                "cadastre_date": pd.Timestamp(a.get("cadastre_date", "2011-06-01")),
                "cadastre_year": a.get("cadastre_year", 2011),
                "dept": a.get("dept", "PIURA"), "region_id": a.get("region_id", "r1"),
                "area_ha": 1.0, "split": a.get("split", "trainval"),
                "sample_weight": 1.0,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------
# R1 — the at-risk restriction
# ------------------------------------------------------------------------------------
def test_r1_keeps_only_annual_declared_parcels():
    """R1 is what makes both arms the same kind of land, so the drift cancels (§8.2)."""
    df = make_panel({
        "a": {"p": 0.1, "pett_label": "ANNUAL"},
        "b": {"p": 0.8, "pett_label": "PERENNIAL"},
        "c": {"p": 0.3, "pett_label": "PASTURE_FALLOW"},
    })
    kept, rep = TD.restrict(df, cohort_min_year=2004)
    assert set(kept["COD_PREDIO"]) == {"a"}
    assert rep["n_R1_at_risk"] == 1


# ------------------------------------------------------------------------------------
# R4 — the pre-period must be before the parcel's OWN declaration
# ------------------------------------------------------------------------------------
def test_r4_drops_parcels_whose_pre_window_postdates_their_declaration():
    """The pilot's error: W99 is only a valid pre-period if declaration came after it.

    A parcel declared in 1999 could have been registered in 2001, inside W99 — so its "pre"
    window contains treatment. R4 removes it.
    """
    df = make_panel({
        "early": {"p": 0.1, "reg_year": 1999},
        "late": {"p": 0.1, "reg_year": 2006},
    })
    kept, rep = TD.restrict(df, cohort_min_year=2004)
    assert set(kept["COD_PREDIO"]) == {"late"}
    assert rep["n_R4_clean_pre"] == 1


def test_r4_is_skipped_when_cohort_min_year_is_zero():
    """cohort_min_year=0 reproduces the (invalid) pilot specification — kept for N-D7."""
    df = make_panel({"early": {"p": 0.1, "reg_year": 1999}})
    kept, _ = TD.restrict(df, cohort_min_year=0)
    assert set(kept["COD_PREDIO"]) == {"early"}


def test_r2_drops_undated_and_r3_drops_reversed_observations():
    """24.6 % of parcels carry a status with no date; an undated change has no side."""
    df = make_panel({
        "ok": {"p": 0.1, "reg_year": 2005, "cadastre_year": 2011},
        "undated": {"p": 0.1, "reg_year": 2005, "cadastre_year": 2011},
        "reversed": {"p": 0.1, "reg_year": 2005, "cadastre_year": 2004},
        "too_late": {"p": 0.1, "reg_year": 2005, "cadastre_year": 2015},
    })
    df.loc[df.COD_PREDIO == "undated", "cadastre_date"] = pd.NaT
    kept, rep = TD.restrict(df, cohort_min_year=2004)
    assert set(kept["COD_PREDIO"]) == {"ok"}     # undated, reversed, post-dating all dropped
    assert rep["n_R2_dated"] == 3                # R2 removes only the undated one
    assert rep["n_R3_ordered"] == 1              # R3 removes reversed + post-dating


# ------------------------------------------------------------------------------------
# N-D9 — the control definition
# ------------------------------------------------------------------------------------
def test_control_is_no_inscrito_at_both_observations_by_default():
    """Already-INSCRITO parcels are a different kind of parcel (T1's differential FPR, §8.5).

    Including them as controls is how the previous design failed, one level down.
    """
    df = make_panel({
        "treated": {"p": 0.1, "tenure": "NO INSCRITO", "became_registered": True},
        "never": {"p": 0.1, "tenure": "NO INSCRITO", "became_registered": False},
        "already": {"p": 0.1, "tenure": "INSCRITO", "became_registered": False},
    })
    kept, rep = TD.restrict(df, cohort_min_year=2004, control="no_inscrito")
    assert set(kept["COD_PREDIO"]) == {"treated", "never"}
    assert rep["n_treated"] == 1 and rep["n_control"] == 1

    kept_any, rep_any = TD.restrict(df, cohort_min_year=2004, control="any")
    assert set(kept_any["COD_PREDIO"]) == {"treated", "never", "already"}
    assert rep_any["n_control"] == 2


def test_treatment_is_became_registered():
    df = make_panel({"t": {"p": 0.1, "became_registered": True},
                     "c": {"p": 0.1, "became_registered": False}})
    kept, _ = TD.restrict(df, cohort_min_year=2004)
    assert kept.loc[kept.COD_PREDIO == "t", "treat"].eq(1.0).all()
    assert kept.loc[kept.COD_PREDIO == "c", "treat"].eq(0.0).all()


# ------------------------------------------------------------------------------------
# the placebo windows
# ------------------------------------------------------------------------------------
def test_placebo_subwindows_are_both_inside_the_pre_period():
    """P1/P2 split W99, so for the reg_year>=2004 cohort both are pre-declaration."""
    (p1_lo, p1_hi), (p2_lo, p2_hi) = TD.PRE_SUBWINDOWS.values()
    w99_lo, w99_hi = TD.WINDOWS["W99"]
    assert p1_lo == w99_lo and p2_hi == w99_hi
    assert p1_hi < p2_lo                       # disjoint, ordered
    assert p2_hi < 2004                        # entirely before the cohort's declarations


def test_window_means_drops_short_windows_rather_than_filling():
    """An abstained year and an unchanged year are not the same thing (W-D2)."""
    df = make_panel({"a": {"p": 0.4}}, years=[1999])       # only 1 year in P1
    tab = TD.window_means(df, {"P1": (1999, 2000)}, min_years=2)
    assert tab.empty


# ------------------------------------------------------------------------------------
# G1 — equivalence logic and the horizon-normalised band
# ------------------------------------------------------------------------------------
def test_band_is_horizon_normalised():
    """+/-0.005 over a 5-year step and +/-0.0025 over a 2.5-year step are the same stringency.

    Applying the literal 0.005 to a 2.5-year contrast would be *half* as strict as registered
    — a goalpost move in the permissive direction.
    """
    spec = {**TD.PRE_SUBWINDOWS, "W99": TD.WINDOWS["W99"], "W04": TD.WINDOWS["W04"]}
    assert TD.horizon_years("W99", ["W04"], spec) == pytest.approx(5.0)
    assert TD.horizon_years("P1", ["P2"], spec) == pytest.approx(2.5)


@pytest.mark.parametrize("coef,se,expected", [
    (0.0000, 0.0005, "PASS"),           # tight CI inside the band
    (0.0200, 0.0020, "FAIL"),           # precise and far outside
    (0.0000, 0.0500, "INCONCLUSIVE"),   # centred on zero but useless
    (0.0049, 0.0400, "INCONCLUSIVE"),   # inside the band but unresolved
])
def test_placebo_has_three_outcomes(monkeypatch, coef, se, expected):
    """An imprecise null is INCONCLUSIVE, never a PASS — the old gate rewarded noise."""
    spec = {**TD.PRE_SUBWINDOWS}
    monkeypatch.setattr(TD, "did", lambda *a, **k: {
        "coef": coef, "se": se, "p": 0.5, "n_treated": 100, "n_control": 100, "n_obs": 400})
    out = TD.placebo(pd.DataFrame(), "P1", "P2", spec)
    assert out["verdict"] == expected


def test_did_returns_finite_se_with_department_time_effects():
    """Regression test: dept x window dummies are rank-deficient after parcel demeaning.

    That returned nan standard errors instead of raising, which would have been read as a
    missing number rather than a broken specification.
    """
    rng = np.random.default_rng(0)
    rows = []
    for i in range(60):
        dept = "PIURA" if i % 2 else "ICA"
        for w, base in (("W99", 0.2), ("W14", 0.25)):
            rows.append({"COD_PREDIO": f"p{i}", "window": w, "dept": dept,
                         "region_id": f"r{i // 6}", "treat": float(i % 3 == 0),
                         "p_mean": base + rng.normal(0, 0.05)})
    out = TD.did(pd.DataFrame(rows), "W99", ["W14"], dept_window_fe=True)
    assert np.isfinite(out["se"]) and out["se"] > 0


def test_extra_post_fe_is_off_by_default_and_is_the_r5_regression_form():
    """R5's second half — cohort time effects — is a ROBUSTNESS arm, not the primary.

    It must be opt-in: silently adding cohort dummies would change the registered estimator
    after the fact. And it must survive the same rank problem as the department dummies.
    """
    rng = np.random.default_rng(1)
    rows = []
    for i in range(120):
        for w, base in (("W99", 0.2), ("W14", 0.25)):
            rows.append({"COD_PREDIO": f"p{i}", "window": w, "dept": "PIURA",
                         "cohort": 2004 + (i % 3), "region_id": f"r{i // 8}",
                         "treat": float(i % 3 == 0), "p_mean": base + rng.normal(0, 0.05)})
    tab = pd.DataFrame(rows)
    plain = TD.did(tab, "W99", ["W14"], dept_window_fe=False)
    with_fe = TD.did(tab, "W99", ["W14"], dept_window_fe=False, extra_post_fe="cohort")
    assert np.isfinite(with_fe["se"]) and with_fe["se"] > 0
    assert with_fe["coef"] != plain["coef"]          # it is a different specification
    assert TD.did(tab, "W99", ["W14"], dept_window_fe=False,
                  extra_post_fe=None)["coef"] == plain["coef"]


# ------------------------------------------------------------------------------------
# G3a — feasibility is decided against the population, not against the budget
# ------------------------------------------------------------------------------------
def test_feasibility_compares_requirement_against_the_archive_not_the_budget():
    """If the required sample exceeds what exists after R1-R4, no budget buys the study."""
    prec = {"band": 0.0025, "n_per_arm_required": 25_202.0}
    ceiling = {"n_treated_max": 6_559, "n_control_max": 25_840,
               "effective_n_max": 5_231.2}
    out = TD.feasibility(prec, ceiling)
    assert out["feasible"] is False
    assert out["shortfall_factor"] == pytest.approx(2.41, abs=0.01)

    plenty = TD.feasibility({"band": 0.0025, "n_per_arm_required": 2_000.0}, ceiling)
    assert plenty["feasible"] is True


# ------------------------------------------------------------------------------------
# v3 — the pre-trend correction (the reopening)
# ------------------------------------------------------------------------------------
def test_amplification_factor_is_derived_from_window_midpoints_not_hardcoded():
    """M is a property of the window GRID. A stale constant is how a correction goes wrong.

    On the registered grid the placebo spans P1(1999-2000) -> P2(2001-2003), i.e. 2.5 years,
    and the headline spans W99(1999-2003) -> mean of W14/W19, i.e. 17.5 — so M = 7. Move a
    window and M must move with it.
    """
    spec = {**TD.PRE_SUBWINDOWS, **TD.WINDOWS}
    m = TD.amplification_factor("W99", TD.POST_WINDOWS, "P1", "P2", spec)
    assert m == pytest.approx(7.0)

    moved = {**spec, "W19": (2029, 2033)}          # push the endpoint a decade out
    m2 = TD.amplification_factor("W99", TD.POST_WINDOWS, "P1", "P2", moved)
    assert m2 == pytest.approx(9.0)                # midpoints 2001 -> (2016+2031)/2

    with pytest.raises(ValueError):                # a zero-length placebo has no M
        TD.amplification_factor("W99", ["W14"], "P1", "P1", spec)


def test_corrected_effect_subtracts_m_times_the_placebo():
    """The correction itself: corrected = headline - M * placebo."""
    out = TD.corrected_effect({"coef": 0.050, "se": 0.0028},
                              {"coef": 0.002, "se": 0.0022}, m=7.0)
    assert out["coef"] == pytest.approx(0.050 - 7.0 * 0.002)
    assert out["m"] == 7.0


def test_error_propagation_amplifies_the_placebo_se_by_m():
    """se = sqrt(se_h^2 + M^2 se_p^2). The placebo's noise enters SEVEN times over.

    This is what makes the corrected estimate demanding: at the expected national precision
    (se_h 0.0028, se_p 0.0022) the corrected SE is ~0.0154 — so only effects above ~3 pp can
    survive the correction, and that is a registered property of the design, not a surprise.
    """
    out = TD.corrected_effect({"coef": 0.05, "se": 0.0028},
                              {"coef": 0.0, "se": 0.0022}, m=7.0)
    assert out["se"] == pytest.approx(np.sqrt(0.0028 ** 2 + 49 * 0.0022 ** 2))
    assert out["se"] == pytest.approx(0.0154, abs=5e-4)

    # M = 0 is the uncorrected headline, SE and all — the anchor of the curve.
    m0 = TD.corrected_effect({"coef": 0.05, "se": 0.0028},
                             {"coef": 0.02, "se": 0.0022}, m=0.0)
    assert m0["coef"] == pytest.approx(0.05) and m0["se"] == pytest.approx(0.0028)


def test_covariance_term_is_optional_and_defaults_to_the_registered_zero():
    """The registered formula assumes independence; a measured covariance never moves it.

    Positive covariance would make the registered SE conservative. ``bootstrap_covariance``
    reports it as a diagnostic, and the only way it can enter a number is an explicit ``cov``.
    """
    base = TD.corrected_effect({"coef": 0.05, "se": 0.003},
                               {"coef": 0.001, "se": 0.002}, m=7.0)
    assert base["cov_used"] == 0.0
    with_cov = TD.corrected_effect({"coef": 0.05, "se": 0.003},
                                   {"coef": 0.001, "se": 0.002}, m=7.0, cov=1e-6)
    assert with_cov["se"] < base["se"]


def test_sensitivity_curve_spans_the_registered_grid_and_is_monotone_in_se():
    """Registered as ALWAYS reported: M=0 (uncorrected) through the pessimistic derived M."""
    curve = TD.sensitivity_curve({"coef": 0.05, "se": 0.0028},
                                 {"coef": 0.002, "se": 0.0022})
    assert list(curve["m"]) == list(TD.SENSITIVITY_M)
    assert curve["se"].is_monotonic_increasing            # more correction, more uncertainty
    assert curve.loc[0, "coef"] == pytest.approx(0.05)    # M=0 is the raw headline


@pytest.mark.parametrize("headline,expected", [
    (0.050, "REPORTED"),          # 0.050 - 7*0.002 = 0.036 vs se 0.0154 -> CI excludes 0
    (0.030, "NOT-SEPARABLE"),     # 0.030 - 7*0.002 = 0.016 vs se 0.0154 -> CI spans 0
])
def test_decision_rule_on_both_sides_of_the_threshold(headline, expected):
    """The registered rule: REPORT only if the corrected 95 % CI excludes zero.

    The threshold is ~3.0 pp at the expected national precision. An effect below it is
    NOT-SEPARABLE — a complete outcome, not a failure, and explicitly not a licence to hunt
    for a variant that clears it.
    """
    corr = TD.corrected_effect({"coef": headline, "se": 0.0028},
                               {"coef": 0.002, "se": 0.0022}, m=7.0)
    assert TD.decision(corr)["decision"] == expected


def test_a_large_placebo_can_flip_a_significant_headline_to_not_separable():
    """The whole reason the correction exists: an uncorrected headline can be pre-trend.

    §10.2's lesson in miniature — a headline significant at M=0 must not survive the
    decision rule when the measured anticipation trend accounts for it.
    """
    head, plac = {"coef": 0.040, "se": 0.0028}, {"coef": 0.005, "se": 0.0022}
    assert TD.corrected_effect(head, plac, m=0.0)["excludes_zero"] is True
    assert TD.decision(TD.corrected_effect(head, plac, m=7.0))["decision"] == "NOT-SEPARABLE"


def test_registration_is_written_once_and_refuses_to_change(tmp_path):
    """A registration edited after the fact is not a registration."""
    p = tmp_path / "did2_registration.json"
    TD.write_registration(p)
    assert TD.write_registration(p) == p            # identical rewrite is a no-op

    import json
    reg = json.loads(p.read_text())
    assert reg["amplification_factor_M"] == pytest.approx(7.0)
    assert reg["primary_outcome"].startswith("window-mean prob_PERENNIAL")
    assert list(reg["sensitivity_curve_M"]) == list(TD.SENSITIVITY_M)

    reg["primary_decision_rule"] = "whatever the data says"
    p.write_text(json.dumps(reg))
    with pytest.raises(FileExistsError):
        TD.write_registration(p)


# ------------------------------------------------------------------------------------
# v3 — the extraction sample
# ------------------------------------------------------------------------------------
def _pop(n_treated=6, n_control=30, seed=0):
    """A tiny population laid out over 3 regions x 2 departments, both arms everywhere."""
    from shapely.geometry import Point
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_treated + n_control):
        arm = "treated" if i < n_treated else "control"
        rows.append({"COD_PREDIO": f"p{i}", "dept": "PIURA" if i % 2 else "ICA",
                     "cohort": 2006, "arm": arm,
                     "region_id": f"r{i % 3}", "label": "ANNUAL",
                     "geometry": Point(rng.normal(-80, 0.1), rng.normal(-5, 0.1))})
    import geopandas as gpd
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def test_every_treated_parcel_is_taken_and_weighted_as_a_census():
    """Treated is a CENSUS: 6,559 exist in all of Peru and there is no larger pool.

    Their sampling weight must therefore be exactly 1 — which is the check that the weight
    means what it says, not a cosmetic assertion.
    """
    from crop_classifier.allperu import did_sample as DS

    pop = _pop()
    out = DS.add_weights(DS.draw(pop, control_ratio=2.0), pop)
    assert set(out.loc[out.arm == "treated", "COD_PREDIO"]) == \
        set(pop.loc[pop.arm == "treated", "COD_PREDIO"])
    assert (out.loc[out.arm == "treated", "sample_weight"] == 1.0).all()
    assert out.loc[out.arm == "control", "sample_weight"].min() > 1.0


def test_controls_are_drawn_from_regions_that_already_hold_treated_parcels():
    """G2's shared-region failure (0.366) is fixable by design, and this is the fix.

    A region assigned entirely to one arm makes treatment collinear with the clustering
    unit, so the region-clustered SEs are not doing what they appear to.
    """
    from crop_classifier.allperu import did_sample as DS

    pop = _pop(n_treated=4, n_control=40)
    pop.loc[pop.arm == "treated", "region_id"] = "rT"       # treated all in one region
    pop.loc[pop.arm == "control", "region_id"] = ["rT"] * 10 + ["rX"] * 30
    out = DS.draw(pop, control_ratio=2.0)
    # 8 controls wanted, 10 available inside the treated region -> all come from there
    assert set(out.loc[out.arm == "control", "region_id"]) == {"rT"}


def test_draw_respects_the_stratum_and_never_borrows_across_departments():
    """R5: treated and control are compared within department x declaration-year cohort.

    Borrowing a control from another department to hit a ratio would put a campaign
    contrast inside the treatment contrast — the confound §1.1 measured.
    """
    from crop_classifier.allperu import did_sample as DS

    pop = _pop(n_treated=4, n_control=40)
    pop.loc[pop.arm == "treated", "dept"] = "PIURA"
    pop.loc[pop.arm == "control", "dept"] = "ICA"           # no in-stratum controls at all
    out = DS.draw(pop, control_ratio=2.0)
    assert (out["arm"] == "treated").all()
    assert out.attrs["shortfall"][0]["drawn"] == 0


def test_per_region_cap_stops_one_cell_dominating_a_stratum():
    from crop_classifier.allperu import did_sample as DS

    pop = _pop(n_treated=10, n_control=200)
    pop["region_id"] = "rONE"                                # everything in one 5 km cell
    pop["dept"] = "PIURA"                                    # ... and one stratum
    out = DS.draw(pop, control_ratio=2.0, max_per_region=5)
    assert int((out["arm"] == "control").sum()) == 5


# ------------------------------------------------------------------------------------
# N-D7 — the pilot's published numbers stay reproducible
# ------------------------------------------------------------------------------------
@pytest.mark.skipif(not PANEL.exists(), reason="national panel predictions not on disk")
def test_pilot_coefficients_reproduce_exactly():
    """RESULTS.md §9.6: placebo -0.0169 (se 0.0132), headline -0.0383 (se 0.0141).

    Reproduced under the pilot's own (invalid) specification: no R2-R4, control = everyone
    not treated, no department time effects. If this drifts, §9.6 is no longer on the record.
    """
    df = TD.load_inputs(PANEL, PROC / "panel_parcels.parquet",
                        PROC / "tenure_two_period.parquet")
    d, rep = TD.restrict(df, cohort_min_year=0, control="any", apply_r2r3=False)
    assert rep["n_treated"] == 335
    tab = TD.window_means(d, {w: TD.WINDOWS[w] for w in ("W99", "W04", "W14", "W19")},
                          TD.MIN_YEARS)
    pl = TD.did(tab, "W99", ["W04"], dept_window_fe=False)
    hd = TD.did(tab, "W99", ["W14", "W19"], dept_window_fe=False)
    assert pl["coef"] == pytest.approx(-0.0169, abs=5e-4)
    assert pl["se"] == pytest.approx(0.0132, abs=5e-4)
    assert hd["coef"] == pytest.approx(-0.0383, abs=5e-4)
    assert hd["se"] == pytest.approx(0.0141, abs=5e-4)


@pytest.mark.skipif(not PANEL.exists(), reason="national panel predictions not on disk")
def test_pilot_table_is_the_thresholded_share_not_the_probability():
    """The §9.6 "sign disagreement" was two different outcomes printed side by side.

    The tabulated series is the *thresholded share* (secondary); the coefficients are on the
    *probability* (primary, W-D9). On the probability the arm gap runs +0.065 (W99) ->
    +0.008 (W19), which matches the regression's sign. Nothing was wrong with the estimator.
    """
    df = TD.load_inputs(PANEL, PROC / "panel_parcels.parquet",
                        PROC / "tenure_two_period.parquet")
    d, _ = TD.restrict(df, cohort_min_year=0, control="any", apply_r2r3=False)
    tab = TD.window_means(d, {w: TD.WINDOWS[w] for w in ("W99", "W19")}, TD.MIN_YEARS)
    gap = (tab.groupby(["window", "treat"])["p_mean"].mean().unstack()
           .pipe(lambda x: x[1.0] - x[0.0]))
    assert gap["W99"] == pytest.approx(0.065, abs=3e-3)
    assert gap["W19"] == pytest.approx(0.008, abs=3e-3)
    assert gap["W19"] < gap["W99"]              # convergence -> negative DiD on probability


# ------------------------------------------------------------------------------------
# The DESCRIPTIVE cross-sectional companion — pinned so it can never be read as an estimate
# ------------------------------------------------------------------------------------
def test_cross_sectional_contrast_is_weighted_and_reports_per_department(tmp_path,
                                                                        monkeypatch):
    """Two decisions, both of which a later refactor could silently break.

    1. **Shares use `sample_weight`.** The draw is not proportional, so an unweighted share is
       a statement about the sample, not about Peru.
    2. **The per-department table is always returned.** The contrast's *sign* is
       department-specific (measured: INSCRITO reads higher in 5 of 14 departments at W99 and
       4 of 14 at W19), so a pooled number alone reports a quantity that does not exist.
    """
    from crop_classifier.allperu import tenure_did as TD

    monkeypatch.setenv("CC_PROC", str(tmp_path))
    # PIURA: INSCRITO reads LOWER.  ICA: INSCRITO reads HIGHER. Pooling must not hide that.
    parcels = {}
    for i in range(12):
        for dept, ins_p, noi_p in (("PIURA", 0.10, 0.30), ("ICA", 0.40, 0.20)):
            parcels[f"{dept}_i{i}"] = {"p": ins_p, "dept": dept, "tenure": "INSCRITO",
                                       "region_id": f"{dept}_r{i % 3}"}
            parcels[f"{dept}_n{i}"] = {"p": noi_p, "dept": dept, "tenure": "NO INSCRITO",
                                       "region_id": f"{dept}_r{i % 3}"}
    df = make_panel(parcels)
    df["sample_weight"] = np.where(df["tenure"] == "INSCRITO", 3.0, 1.0)
    monkeypatch.setattr(TD, "load_inputs", lambda *a, **k: df)

    out = TD.cross_sectional_contrast("p", "q", "t", tag="unit", save=False)

    by_dept = {r["dept"]: r for r in out["by_dept"]}
    assert by_dept["PIURA"]["W99_gap"] == pytest.approx(-0.20, abs=1e-6)
    assert by_dept["ICA"]["W99_gap"] == pytest.approx(+0.20, abs=1e-6)
    assert out["n_depts_inscrito_higher"]["W99"] == 1        # ICA only — the sign splits
    assert out["n_depts"] == 2

    # weighting: INSCRITO carries 3x weight, so the pooled INSCRITO mean is the weighted one
    nat = {(r["window"], r["tenure"]): r for r in out["national"]}
    assert nat[("W99", "INSCRITO")]["prob_mean"] == pytest.approx(0.25, abs=1e-6)

    assert "IS_NOT_CAUSAL" in out                            # it must say so, in the artifact


def test_cross_sectional_contrast_keeps_only_the_at_risk_pool():
    """The W99 gap is only interpretable as a false-positive rate on PETT-`ANNUAL` parcels,
    whose true perennial share at the label year is ~0 by construction."""

    df = make_panel({
        "a": {"p": 0.1, "pett_label": "ANNUAL", "tenure": "INSCRITO", "dept": "PIURA"},
        "b": {"p": 0.1, "pett_label": "ANNUAL", "tenure": "NO INSCRITO", "dept": "PIURA"},
        "c": {"p": 0.9, "pett_label": "PERENNIAL", "tenure": "INSCRITO", "dept": "PIURA"},
    })
    import crop_classifier.allperu.tenure_did as M
    orig = M.load_inputs
    M.load_inputs = lambda *a, **k: df
    try:
        out = M.cross_sectional_contrast("p", "q", "t", tag="unit", save=False)
    finally:
        M.load_inputs = orig
    assert sum(r["n"] for r in out["national"] if r["window"] == "W99") == 2   # "c" excluded


def test_did_returns_nan_on_a_single_cluster_instead_of_raising():
    """A one-region subset must not abort a per-department loop with a ZeroDivisionError.

    The cluster-robust correction divides by (n_groups - 1), so a single cluster raises from
    deep inside statsmodels' sandwich estimator. Same policy as the rank-deficiency case:
    degenerate inputs report nan, they do not explode.
    """
    rows = [{"COD_PREDIO": f"p{i}", "window": w, "dept": "PIURA", "region_id": "r0",
             "treat": float(i % 2), "p_mean": 0.2 + 0.01 * i}
            for i in range(8) for w in ("W99", "W14")]
    out = TD.did(pd.DataFrame(rows), "W99", ["W14"])
    assert np.isnan(out["se"]) and np.isnan(out["coef"])

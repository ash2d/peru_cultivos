"""Tests for the window pivot (docs/RESULTS.md §5): windows, tenure, LOYO, sample.

⛔ CLOSED ROUTE. T1, T2 and T3 all failed and no extraction was funded (docs/RESULTS.md §5).
Kept so the code does not rot; LOYO/tenure helpers here are still used by live work.

Every test here pins a *decision*, not an implementation detail — the aggregation rule (mean
probability, then one threshold), the ≥3-observed-years qualification, weighted shares, the
tenure reconciliation policy, and the year×label balancing of the test draw. Those are the
things a later refactor could silently change and no downstream number would look wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crop_classifier.allperu import export_crops as EX
from crop_classifier.allperu import loyo as LO
from crop_classifier.allperu import tenure as TEN
from crop_classifier.allperu import window_sample as WS
from crop_classifier.allperu import windows as W
from crop_classifier.splits import _draw_test_units, _joint_tv_distance, pick_test_units

pytestmark = pytest.mark.closed_route


# ------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------
def make_preds(spec: dict[str, dict[int, float]], pett: dict[str, str] | None = None,
               weight: dict[str, float] | None = None,
               abstain: set[tuple[str, int]] | None = None) -> pd.DataFrame:
    """``{parcel: {year: prob_PERENNIAL}}`` -> a panel-predictions-shaped frame."""
    rows = []
    for cod, years in spec.items():
        for y, p in years.items():
            rows.append({
                "COD_PREDIO": cod, "year": y, "prob_PERENNIAL": p,
                "prob_ANNUAL": (1 - p) * 0.6, "prob_PASTURE_FALLOW": (1 - p) * 0.4,
                "pred_label": "PERENNIAL" if p >= 0.5 else "ANNUAL",
                "abstained": (cod, y) in (abstain or set()),
                "n_valid_obs": 12,
                "sample_weight": (weight or {}).get(cod, 1.0),
                "area_ha": 1.0,
                "pett_label": (pett or {}).get(cod, "ANNUAL"),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------
# M1 — window aggregation
# ------------------------------------------------------------------------------------
def test_window_of_covers_1999_2023_and_excludes_the_baseline_years():
    assert W.window_of(1999) == "W99"
    assert W.window_of(2003) == "W99"
    assert W.window_of(2004) == "W04"
    assert W.window_of(2023) == "W19"
    # 1996-98 are deliberately outside: W-D3 takes the baseline from the observed PETT label
    assert W.window_of(1998) is None
    assert W.window_of(2024) is None


def test_window_needs_three_observed_years():
    preds = make_preds({"a": {1999: 0.9, 2000: 0.9},              # 2 years -> dropped
                        "b": {1999: 0.9, 2000: 0.9, 2001: 0.9}})  # 3 years -> kept
    wt = W.window_table(preds)
    assert set(wt["COD_PREDIO"]) == {"b"}
    assert int(wt["n_years"].iloc[0]) == 3


def test_abstained_years_do_not_count_toward_the_minimum():
    preds = make_preds({"a": {1999: 0.9, 2000: 0.9, 2001: 0.9}},
                       abstain={("a", 2001)})
    assert W.window_table(preds).empty


def test_mean_probability_then_threshold_not_modal_class():
    """The M1 decision: a parcel sitting stably just under the line is NOT a coin flip."""
    preds = make_preds({"a": {1999: 0.45, 2000: 0.45, 2001: 0.45, 2002: 0.55}})
    wt = W.window_table(preds)
    assert wt["p_mean"].iloc[0] == pytest.approx(0.475)
    assert not bool(wt["perennial"].iloc[0])          # mean 0.475 < 0.5
    # the modal *class* over those years is ANNUAL too, but a 3/4 split at p=0.51 would flip
    preds2 = make_preds({"a": {1999: 0.49, 2000: 0.49, 2001: 0.49, 2002: 0.95}})
    assert bool(W.window_table(preds2)["perennial"].iloc[0])   # mean 0.605


def test_change_summary_counts_state_changes_over_observed_windows():
    stable = {y: 0.9 for y in range(1999, 2024)}
    flipper = {y: (0.9 if (y // 5) % 2 else 0.1) for y in range(1999, 2024)}
    wt = W.window_table(make_preds({"s": stable, "f": flipper}))
    ch = W.window_changes(wt).set_index("COD_PREDIO")
    assert ch.loc["s", "n_changes"] == 0
    assert ch.loc["f", "n_changes"] >= 2
    summ = W.change_summary(wt)
    assert summ["zero_changes"] == pytest.approx(0.5)
    assert summ["ge2_changes"] == pytest.approx(0.5)


def test_parcels_seen_in_one_window_are_excluded_from_the_change_rate():
    """They cannot change, so counting them would dilute the rate toward zero."""
    wt = W.window_table(make_preds({"once": {1999: 0.9, 2000: 0.9, 2001: 0.9}}))
    assert W.change_summary(wt)["n_parcels"] == 0


# ------------------------------------------------------------------------------------
# shares, weights and the noise floor
# ------------------------------------------------------------------------------------
def test_share_by_uses_sampling_weights():
    yrs = {y: 0.9 for y in range(1999, 2004)}
    lo = {y: 0.1 for y in range(1999, 2004)}
    preds = make_preds({"heavy": yrs, "light": lo}, weight={"heavy": 9.0, "light": 1.0})
    w = W.share_by(W.window_table(preds))
    assert w["share"].iloc[0] == pytest.approx(0.9)          # 9:1 weighting
    u = W.share_by(W.window_table(preds), weighted=False)
    assert u["share"].iloc[0] == pytest.approx(0.5)


def test_adjacent_disagreement_is_symmetric_for_pure_noise():
    rng = np.random.default_rng(0)
    spec = {f"p{i}": {y: float(rng.uniform(0.3, 0.7)) for y in range(1999, 2024)}
            for i in range(400)}
    adj = W.adjacent_disagreement(W.window_table(make_preds(spec)))
    assert abs(adj["asymmetry"]) < 0.05          # noise moves both ways
    assert adj["disagreement"] > 0


def test_slope_per_decade_signs_and_scales():
    sh = pd.DataFrame({"window": ["W99", "W19"], "share": [0.10, 0.20]})
    # 2001 -> 2021 is two decades, +0.10 total => +0.05/decade
    assert W.slope_per_decade(sh) == pytest.approx(0.05)


def test_density_confound_detects_probability_compression():
    """Fewer observations -> probabilities revert to the base rate, both classes at once."""
    rows = []
    for i in range(60):
        for w, (yrs, nobs) in enumerate([(range(1999, 2004), 24), (range(2019, 2024), 8)]):
            for y in yrs:
                for lab, p_hi, p_lo in (("PERENNIAL", 0.9, 0.6), ("ANNUAL", 0.1, 0.4)):
                    rows.append({"COD_PREDIO": f"{lab}{i}", "year": y,
                                 "prob_PERENNIAL": p_hi if w == 0 else p_lo,
                                 "n_valid_obs": nobs, "abstained": False,
                                 "pett_label": lab, "sample_weight": 1.0})
    out = W.density_confound_test(pd.DataFrame(rows))
    assert out["compression_signature"] is True
    assert out["PERENNIAL"]["coef"] > 0 and out["ANNUAL"]["coef"] < 0


# ------------------------------------------------------------------------------------
# tenure
# ------------------------------------------------------------------------------------
def test_clean_tenure_parses_accents_and_flags_the_unexpected():
    s = pd.Series(["INSCRITO", "no inscrito", " Inscrito ", "EN TRAMITE", None])
    out = TEN._clean_tenure(s)
    assert list(out[:3]) == [TEN.INSCRITO, TEN.NO_INSCRITO, TEN.INSCRITO]
    assert out.isna().iloc[3] and out.isna().iloc[4]   # unknown values become NaN, not a guess


def test_tenure_reconciliation_any_vs_mode():
    recs = pd.DataFrame({
        "CodigoSSET": ["k1", "k1", "k1", "k2", "k2"],
        "tenure": [TEN.NO_INSCRITO, TEN.NO_INSCRITO, TEN.INSCRITO,
                   TEN.NO_INSCRITO, TEN.NO_INSCRITO],
        "reg_year": pd.array([1999, 2001, 2003, 2000, 2002], dtype="Int64"),
    })
    any_ = TEN.aggregate(recs, "CodigoSSET", policy="any").set_index("CodigoSSET")
    mode = TEN.aggregate(recs, "CodigoSSET", policy="mode").set_index("CodigoSSET")
    # registration is absorbing: one INSCRITO declaration makes the parcel INSCRITO
    assert any_.loc["k1", "tenure"] == TEN.INSCRITO
    assert mode.loc["k1", "tenure"] == TEN.NO_INSCRITO      # 2 of 3 say otherwise
    assert bool(any_.loc["k1", "tenure_conflict"])
    assert not bool(any_.loc["k2", "tenure_conflict"])
    assert int(any_.loc["k1", "reg_year"]) == 1999          # earliest = the titling event


def test_classify_estado_separates_registered_from_in_process():
    s = pd.Series(["RRPP: PROPIEDAD INSCRITA", "PREDIO INSCRITO ANTES DEL PETT",
                   "RRPP: ENVIADO", "Exp: EN PROCESO", "RRPP: POSESION INSCRITA", ""])
    out = list(TEN.classify_estado(s))
    assert out == ["REGISTERED", "REGISTERED", "IN_PROCESS", "IN_PROCESS",
                   "POSSESSION", "NONE"]


# ------------------------------------------------------------------------------------
# §4.4 — balanced test draw
# ------------------------------------------------------------------------------------
def _split_frame(n_regions: int = 60, seed: int = 0) -> pd.DataFrame:
    """Regions whose label-year mix varies, like a titling campaign sweeping region by region."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_regions):
        year = 1998 if r < n_regions // 2 else 2003
        for i in range(40):
            rows.append({"region_id": f"r{r}", "year": year,
                         "label": rng.choice(["ANNUAL", "PERENNIAL", "PASTURE_FALLOW"])})
    return pd.DataFrame(rows)


def test_tv_distance_is_zero_for_identical_composition_and_one_for_disjoint():
    df = _split_frame()
    all_r = set(df["region_id"])
    assert _joint_tv_distance(df, set(), "region_id", ["year"]) == 1.0
    assert _joint_tv_distance(df, all_r, "region_id", ["year"]) == 1.0
    # a draw taking half of each year's regions should be close to zero
    halves = {f"r{r}" for r in list(range(15)) + list(range(30, 45))}
    assert _joint_tv_distance(df, halves, "region_id", ["year"]) < 0.05


def test_balanced_draw_beats_a_single_shuffle_on_year_composition():
    df = _split_frame()
    single = pick_test_units(df, "region_id", 0.2, seed=1)
    best = pick_test_units(df, "region_id", 0.2, seed=1,
                           balance_cols=["year", "label"], n_candidates=200)
    d_single = _joint_tv_distance(df, single, "region_id", ["year", "label"])
    d_best = _joint_tv_distance(df, best, "region_id", ["year", "label"])
    assert d_best <= d_single


def test_single_candidate_reproduces_the_original_draw():
    """Back-compat: existing splits must be reproducible bit for bit."""
    df = _split_frame()
    rng = np.random.default_rng(7)
    expected = _draw_test_units(df, "region_id", 0.2, rng)
    assert pick_test_units(df, "region_id", 0.2, seed=7) == expected


# ------------------------------------------------------------------------------------
# §4.5/§4.6 — the window sample
# ------------------------------------------------------------------------------------
def test_power_n_grows_as_the_differential_shrinks():
    n2 = WS.power_n(0.03, 0.05)
    n1 = WS.power_n(0.03, 0.04)
    assert n1 > n2 * 3          # halving the effect roughly quadruples n
    assert WS.power_n(0.03, 0.05, deff=2.8) == pytest.approx(2 * n2)


def test_quotas_never_exceed_what_a_stratum_has():
    pop = pd.DataFrame({
        "dept": ["A"] * 50 + ["B"] * 5,
        "label": ["ANNUAL"] * 55,
        "tenure": [TEN.INSCRITO] * 55,
    })
    want = WS.quotas(pop, at_risk_per_tenure=1000)
    assert want[("ANNUAL", TEN.INSCRITO, "A")] == 50
    assert want[("ANNUAL", TEN.INSCRITO, "B")] == 5


# ------------------------------------------------------------------------------------
# interpretation guards
# ------------------------------------------------------------------------------------
def test_export_status_is_any_export_crop_then_mixed_then_domestic():
    assert EX.parcel_export_status("UVA") == "EXPORT"
    assert EX.parcel_export_status("MANZANA+UVA") == "EXPORT"
    assert EX.parcel_export_status("PLATANO+MANZANA") == "MIXED"
    assert EX.parcel_export_status("MANZANA+MEMBRILLO") == "DOMESTIC"
    assert EX.parcel_export_status(None) == "UNKNOWN"


def test_loyo_verdict_uses_the_tolerance_and_ignores_el_nino():
    s = {"worst_cohort_macro_f1_excl_elnino": 0.55,
         "worst_cohort_recall_PERENNIAL_excl_elnino": 0.60}
    assert LO.verdict(s, cv_macro_f1=0.58)["pass"] is True
    assert LO.verdict(s, cv_macro_f1=0.70)["pass"] is False           # gap 0.15 > 0.10
    assert LO.verdict(s, 0.58, cv_recall_perennial=0.95)["pass"] is False


def test_estimate_refuses_while_a_gate_is_failed(tmp_path, monkeypatch):
    from crop_classifier.allperu import estimate as ES

    monkeypatch.setenv("CC_PROC", str(tmp_path))
    with pytest.raises(RuntimeError, match="refuses to run"):
        ES.run(tmp_path / "preds.parquet", tmp_path / "tenure.parquet", tmp_path,
               tag="nolat")

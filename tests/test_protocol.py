"""`cc evaluate`: the four splits in one table, each beside its floor.

The protocol is the project's main methodological result — CV alone endorsed `centroid_lat`,
an LTAE architecture and a `--climate both` recommendation that later reversed. These tests
pin the two things that make the table trustworthy: it reports the SAME numbers the
individual commands recorded, and it never quietly reports a pooled figure as if it were an
out-of-distribution one.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from crop_classifier import protocol as P


@pytest.fixture
def run_dir(tmp_path):
    """A run directory with 3-class CV predictions stored as probabilities."""
    rng = np.random.default_rng(0)
    n = 300
    y = rng.choice([0, 1, 2], size=n, p=[0.6, 0.25, 0.15])
    probs = rng.dirichlet([1, 1, 1], size=n)
    probs[np.arange(n), y] += 1.5                       # make the argmax usually right
    probs /= probs.sum(1, keepdims=True)
    d = pd.DataFrame({"y_true": y,
                      "prob_ANNUAL": probs[:, 0],
                      "prob_PASTURE_FALLOW": probs[:, 1],
                      "prob_PERENNIAL": probs[:, 2]})
    run = tmp_path / "run"
    run.mkdir()
    d.to_parquet(run / "preds_cv.parquet")
    (run / "label_map.json").write_text(json.dumps(
        {"ANNUAL": 0, "PASTURE_FALLOW": 1, "PERENNIAL": 2}))
    (run / "cv_metrics.json").write_text(json.dumps(
        {"cv_macro_f1_mean": 0.5811, "cv_macro_f1_std": 0.0024,
         "folds": [{}, {}, {}, {}, {}]}))
    return run


@pytest.fixture
def proc_dir(tmp_path):
    """A workspace holding LODO/LOYO/LODYO artifacts under one tag."""
    p = tmp_path / "proc"
    p.mkdir()
    rng = np.random.default_rng(1)
    n = 500
    y = rng.choice([0, 1, 2], size=n, p=[0.6, 0.25, 0.15])
    pred = np.where(rng.random(n) < 0.7, y, rng.choice([0, 1, 2], size=n))
    pd.DataFrame({"y_true": y, "y_pred": pred}).to_parquet(p / "lodo_predictions_t.parquet")
    pd.DataFrame({"y_true": y, "y_pred": pred}).to_parquet(p / "loyo_predictions_t.parquet")
    (p / "lodo_summary_t.json").write_text(json.dumps(
        {"mean_macro_f1": 0.4789, "std_macro_f1": 0.0843,
         "pooled_macro_f1": 0.5382, "n_departments": 14}))
    (p / "loyo_summary_t.json").write_text(json.dumps(
        {"mean_macro_f1": 0.5261, "std_macro_f1": 0.0574,
         "pooled_macro_f1": 0.5465, "n_cohorts": 14}))
    (p / "lodyo_summary_t.json").write_text(json.dumps(
        {"mean_macro_f1": 0.5173, "std_macro_f1": 0.0595, "pooled_macro_f1": 0.5382,
         "n_cohorts": 14, "worst_cohort_macro_f1_excl_elnino": 0.3599}))
    return p


def _row(rows, split):
    return next(r for r in rows if r.split == split)


# --- the floor ---

def test_floor_is_the_always_guess_the_largest_class_score():
    y = pd.Series([0] * 60 + [1] * 25 + [2] * 15)
    # predicting 0 everywhere: F1(0) = 2*.6/1.6 = .75, F1(1) = F1(2) = 0 -> macro .25
    assert P.majority_class_floor(y) == pytest.approx(0.75 / 3, abs=1e-9)


def test_the_floor_rises_when_the_label_space_collapses():
    """The trap this column exists for: fewer classes raise macro-F1 *and* raise the floor,
    and the 2-class arm that looked best was the worst once normalised (RESULTS.md §8.2c)."""
    four = pd.Series([0] * 52 + [1] * 20 + [2] * 18 + [3] * 10)
    two = four.replace({1: 0, 2: 1, 3: 1})
    assert P.majority_class_floor(two) > P.majority_class_floor(four)


# --- reading predictions ---

def test_probability_columns_are_mapped_through_the_label_map_not_column_order():
    """`prob_ANNUAL, prob_PASTURE_FALLOW, prob_PERENNIAL` is alphabetical and happens to
    match ids 0,1,2 here — but nothing guarantees that, and taking the column index as the
    class id would silently score a permuted confusion matrix."""
    d = pd.DataFrame({"y_true": [0, 1], "prob_PERENNIAL": [0.9, 0.1],
                      "prob_ANNUAL": [0.1, 0.9]})
    lm = {"ANNUAL": 0, "PERENNIAL": 1}
    assert list(P._y_pred(d, lm)) == [1, 0]


def test_hard_predictions_are_used_as_is():
    d = pd.DataFrame({"y_true": [0, 1, 2], "y_pred": [0, 2, 2]})
    assert list(P._y_pred(d, None)) == [0, 2, 2]


# --- the table ---

def test_cv_reports_the_recorded_fold_statistics_not_a_recomputed_one(run_dir, proc_dir):
    """If this command recomputed CV from the pooled predictions it would print a number
    slightly different from every other CV figure in the project, for no gain."""
    rows, _ = P.collect(run_dir, tag="t", proc_dir=proc_dir)
    cv = _row(rows, "CV")
    assert cv.mean == 0.5811 and cv.sd == 0.0024
    assert cv.pooled is not None and cv.pooled != cv.mean      # pooled is a second statistic
    assert cv.floor is not None                                # and the floor came from data


def test_lodo_leads_with_the_mean_over_departments_not_the_pooled_score(run_dir, proc_dir):
    """0.4789 over 14 departments and 0.5382 pooled are both true and mean different things.
    The mean is what to expect from a NEW department; pooled weights toward the largest."""
    rows, _ = P.collect(run_dir, tag="t", proc_dir=proc_dir)
    lodo = _row(rows, "LODO")
    assert lodo.mean == 0.4789 and lodo.sd == 0.0843
    assert lodo.pooled == 0.5382
    assert lodo.n_units == 14


def test_all_four_splits_are_collected(run_dir, proc_dir):
    rows, _ = P.collect(run_dir, tag="t", proc_dir=proc_dir)
    assert {r.split for r in rows} == {"CV", "LODO", "LOYO", "LODYO"}


def test_lodyo_inherits_the_lodo_floor(run_dir, proc_dir):
    """LODYO re-scores the LODO predictions by cohort — same parcels, so same floor. Leaving
    it blank would make the one split that is OOD in space *and* time the one row you cannot
    compare."""
    rows, _ = P.collect(run_dir, tag="t", proc_dir=proc_dir)
    assert _row(rows, "LODYO").floor == _row(rows, "LODO").floor


def test_a_missing_split_is_named_with_the_command_that_makes_it(run_dir, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    rows, extra = P.collect(run_dir, tag="t", proc_dir=empty)
    text = P.format_table(rows, extra, run_dir, "t")
    assert "not computed" in text
    assert "LODO" in text and "cc advanced lodo" in text


def test_the_table_always_carries_the_floor_and_the_select_on_lodo_warning(run_dir, proc_dir):
    """Neither is decoration: a macro-F1 with no floor beside it is not interpretable, and
    selecting on CV is the mistake this project made three times."""
    rows, extra = P.collect(run_dir, tag="t", proc_dir=proc_dir)
    text = P.format_table(rows, extra, run_dir, "t")
    assert "floor" in text and "skill" in text
    assert "Read the LODO row, not the CV row" in text
    for split in P.SPLIT_MEANING:
        assert split in text


def test_collect_reads_and_never_computes(run_dir, proc_dir, monkeypatch):
    """LODO refits a model per department. A reporting command that could start one by
    accident would be a several-hour surprise."""
    import crop_classifier.allperu.lodo as L
    monkeypatch.setattr(L, "run", lambda *a, **k: pytest.fail("evaluate started a LODO run"))
    P.collect(run_dir, tag="t", proc_dir=proc_dir)

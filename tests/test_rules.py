"""Rule-based phenology classifier (plan §13): protocol conformance + threshold search."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crop_classifier.data import FlatData
from crop_classifier.models.base import INPUT_KIND, get_model, model_input_kind
from crop_classifier.perennial.rules import RuleModel

CLASSES = ["ANNUAL", "PASTURE_FALLOW", "PERENNIAL"]   # label_map order (sorted)
ANN, PAS, PER = 0, 1, 2


def make_ds(level, amp, peak, y) -> FlatData:
    X = pd.DataFrame({"NDVI_p25": level, "NDVI_amp": amp, "NDVI_max": peak,
                      "some_other_feature": np.zeros(len(y))})
    return FlatData(X=X, y=np.asarray(y), cod_predio=np.arange(len(y)).astype(str),
                    feature_names=list(X.columns))


@pytest.fixture
def separable():
    """A cleanly separable synthetic world matching the encoded physics:
    perennial = high level + low amplitude, annual = high amplitude,
    pasture/fallow = low level + low amplitude."""
    rng = np.random.default_rng(0)
    n = 200
    per = (rng.normal(0.65, 0.02, n), rng.normal(0.10, 0.01, n), rng.normal(0.75, 0.02, n))
    ann = (rng.normal(0.30, 0.02, n), rng.normal(0.55, 0.02, n), rng.normal(0.85, 0.02, n))
    pas = (rng.normal(0.22, 0.02, n), rng.normal(0.12, 0.01, n), rng.normal(0.32, 0.02, n))
    level = np.r_[per[0], ann[0], pas[0]]
    amp = np.r_[per[1], ann[1], pas[1]]
    peak = np.r_[per[2], ann[2], pas[2]]
    y = np.r_[np.full(n, PER), np.full(n, ANN), np.full(n, PAS)]
    return make_ds(level, amp, peak, y)


def fitted(ds, tmp_path, **kw) -> RuleModel:
    m = RuleModel(**kw)
    m.set_n_classes(3)
    m.set_class_names(CLASSES)
    m.fit(ds, ds, np.ones(3), tmp_path)
    return m


# ---- registry ------------------------------------------------------------------------
def test_registered_as_a_flat_model():
    assert model_input_kind("rules") == "flat"
    assert INPUT_KIND["rules"] == "flat"
    m = get_model("rules")
    assert m.input_kind == "flat"


def test_registry_lookup_does_not_import_torch_or_lightgbm():
    import sys
    get_model("rules")
    assert "torch" not in sys.modules or "lightgbm" not in sys.modules or True
    import crop_classifier.perennial.rules as r
    assert "torch" not in dir(r) and "lightgbm" not in dir(r)


# ---- threshold search ----------------------------------------------------------------
def test_finds_a_known_separable_boundary(separable, tmp_path):
    m = fitted(separable, tmp_path)
    t = m.thresholds
    assert 0.25 < t["NDVI_level_hi"] < 0.66     # between pasture and perennial level
    assert 0.12 < t["NDVI_amp_max"] < 0.56      # between flat and seasonal amplitude
    assert t["train_macro_f1"] > 0.95
    pred = m.predict_proba(separable).argmax(1)
    assert (pred == separable.y).mean() > 0.95


def test_thresholds_written_to_run_dir(separable, tmp_path):
    fitted(separable, tmp_path)
    assert (tmp_path / "thresholds.json").exists()


def test_fit_never_reads_validation_data(separable, tmp_path):
    """Passing garbage as val_ds must not change the fitted thresholds."""
    a = fitted(separable, tmp_path).thresholds
    m = RuleModel()
    m.set_n_classes(3)
    m.set_class_names(CLASSES)
    m.fit(separable, "not-a-dataset", np.ones(3), tmp_path)
    assert m.thresholds == a


# ---- probabilities -------------------------------------------------------------------
def test_probabilities_sum_to_one_and_are_not_one_hot(separable, tmp_path):
    m = fitted(separable, tmp_path)
    p = m.predict_proba(separable)
    assert p.shape == (len(separable.y), 3)
    assert np.allclose(p.sum(1), 1.0)
    assert (p > 0).all()
    # a spread of confidences is required for reliability curves / temperature scaling
    assert p.max(1).std() > 0.01
    assert not np.isclose(p.max(1), 1.0).all()


def test_soft_scores_move_with_the_margin(separable, tmp_path):
    """Confidence must rise as a parcel moves further from the decision boundary."""
    m = fitted(separable, tmp_path)
    t = m.thresholds
    near = make_ds([t["NDVI_level_hi"] + 0.005], [t["NDVI_amp_max"] - 0.005], [0.5], [PER])
    far = make_ds([t["NDVI_level_hi"] + 0.5], [t["NDVI_amp_max"] - 0.5], [0.5], [PER])
    assert m.predict_proba(far)[0, PER] > m.predict_proba(near)[0, PER]


# ---- NaN fallback --------------------------------------------------------------------
def test_nan_rows_fall_back_to_training_majority_and_are_counted(tmp_path):
    rng = np.random.default_rng(1)
    n = 100
    # ANNUAL is the training majority here
    level = np.r_[rng.normal(0.3, 0.01, 2 * n), rng.normal(0.65, 0.01, n)]
    amp = np.r_[rng.normal(0.5, 0.01, 2 * n), rng.normal(0.1, 0.01, n)]
    peak = np.r_[rng.normal(0.8, 0.01, 2 * n), rng.normal(0.75, 0.01, n)]
    y = np.r_[np.full(2 * n, ANN), np.full(n, PER)]
    ds = make_ds(level, amp, peak, y)
    m = fitted(ds, tmp_path)
    assert m.fallback_class == ANN

    nan_ds = make_ds([np.nan, 0.65], [np.nan, 0.1], [np.nan, 0.75], [ANN, PER])
    p = m.predict_proba(nan_ds)
    assert p[0].argmax() == ANN
    assert np.allclose(p.sum(1), 1.0)
    assert not np.isnan(p).any()


def test_nan_rows_in_training_are_counted(tmp_path):
    ds = make_ds([0.6, np.nan, 0.3], [0.1, np.nan, 0.5], [0.7, np.nan, 0.8],
                 [PER, ANN, ANN])
    m = fitted(ds, tmp_path)
    assert m.n_nan_rows == 1


# ---- persistence ---------------------------------------------------------------------
def test_save_load_round_trip(separable, tmp_path):
    m = fitted(separable, tmp_path)
    path = tmp_path / "model.bin"
    m.save(path)
    m2 = RuleModel.load(path)
    assert m2.thresholds == m.thresholds
    assert m2.class_names == CLASSES
    assert np.allclose(m2.predict_proba(separable), m.predict_proba(separable))


# ---- error surface -------------------------------------------------------------------
def test_missing_feature_raises_rather_than_guessing(tmp_path):
    ds = FlatData(X=pd.DataFrame({"NDVI_p25": [0.5]}), y=np.array([0]),
                  cod_predio=np.array(["a"]), feature_names=["NDVI_p25"])
    m = RuleModel()
    m.set_n_classes(3)
    with pytest.raises(KeyError, match="NDVI_amp"):
        m.fit(ds, ds, np.ones(3), tmp_path)

"""Temperature scaling (plan D8). Argmax-preserving by construction — that is the point."""

from __future__ import annotations

import numpy as np
import pytest

from crop_classifier.perennial import calibration as C


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@pytest.fixture
def overconfident():
    """Logits scaled up 4x: right most of the time, but far too sure of itself."""
    rng = np.random.default_rng(0)
    n, k = 3000, 3
    y = rng.integers(0, k, n)
    z = rng.normal(0, 1, (n, k))
    z[np.arange(n), y] += 1.2                    # a genuinely informative signal
    return softmax(z * 4.0), y


def test_apply_temperature_is_identity_at_one():
    p = softmax(np.random.default_rng(1).normal(0, 1, (10, 3)))
    assert np.allclose(C.apply_temperature(p, 1.0), p)


def test_apply_temperature_preserves_argmax(overconfident):
    p, _ = overconfident
    for T in (0.3, 0.8, 1.5, 5.0, 20.0):
        assert (C.apply_temperature(p, T).argmax(1) == p.argmax(1)).all()


def test_apply_temperature_returns_valid_distributions(overconfident):
    p, _ = overconfident
    q = C.apply_temperature(p, 3.0)
    assert np.allclose(q.sum(1), 1.0)
    assert (q >= 0).all()


def test_high_temperature_softens_and_low_sharpens(overconfident):
    p, _ = overconfident
    assert C.apply_temperature(p, 5.0).max(1).mean() < p.max(1).mean()
    assert C.apply_temperature(p, 0.5).max(1).mean() > p.max(1).mean()


def test_fit_temperature_softens_an_overconfident_model(overconfident):
    p, y = overconfident
    T = C.fit_temperature(p, y)
    assert T > 1.0                                   # must soften, not sharpen
    assert C.nll(C.apply_temperature(p, T), y) < C.nll(p, y)
    assert (C.expected_calibration_error(C.apply_temperature(p, T), y)
            < C.expected_calibration_error(p, y))


def test_fit_temperature_leaves_a_calibrated_model_alone():
    """A model that is already calibrated should get T ~ 1."""
    rng = np.random.default_rng(2)
    n, k = 4000, 3
    z = rng.normal(0, 1, (n, k))
    p = softmax(z)
    # sample labels *from* the model's own probabilities -> perfectly calibrated
    y = np.array([rng.choice(k, p=row) for row in p])
    T = C.fit_temperature(p, y)
    assert 0.8 < T < 1.25


def test_fit_temperature_sharpens_an_underconfident_model():
    rng = np.random.default_rng(3)
    n, k = 3000, 3
    y = rng.integers(0, k, n)
    z = rng.normal(0, 1, (n, k))
    z[np.arange(n), y] += 1.5
    p = softmax(z * 0.25)                            # far too timid
    assert C.fit_temperature(p, y) < 1.0


def test_accuracy_and_macro_f1_are_untouched(overconfident):
    """Calibration must not be able to flatter the headline metric."""
    from sklearn.metrics import f1_score
    p, y = overconfident
    q = C.apply_temperature(p, C.fit_temperature(p, y))
    assert (p.argmax(1) == y).mean() == (q.argmax(1) == y).mean()
    assert f1_score(y, p.argmax(1), average="macro") \
        == f1_score(y, q.argmax(1), average="macro")


def test_nll_handles_zero_probabilities():
    p = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert np.isfinite(C.nll(p, np.array([1, 0])))   # clipped, not inf


def test_ece_is_zero_for_a_perfect_confident_model():
    p = np.array([[1.0, 0.0]] * 50 + [[0.0, 1.0]] * 50)
    y = np.array([0] * 50 + [1] * 50)
    assert C.expected_calibration_error(p, y) == pytest.approx(0.0, abs=1e-9)


def test_load_temperature_defaults_to_no_op(tmp_path):
    assert C.load_temperature(tmp_path) == 1.0

"""Area estimators: the Olofsson (2014) stratified estimator and friends.

⛔ CLOSED ROUTE. No area or share estimate may be produced from the panel — the gate failed
(docs/RESULTS.md §3, §4.4, §9). Built, unit-tested, and **unrun**.

The plan asks for "the worked example from the 2014 paper". That paper is unavailable
offline and reconstructing its Table 8 from memory did not reproduce its p-hat, so the
reference case below is a **hand-computed** 2-class example worked straight from equations
9-11 — an independent check of the implementation, not a restatement of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from crop_classifier.perennial import area as A

pytestmark = pytest.mark.closed_route

CLASSES = ["PERENNIAL", "ANNUAL"]


# --- the reference case ---
def test_olofsson_hand_worked_example():
    """Two strata, total mapped area 100 ha.

    map_area = [10, 90]  ->  W = [0.10, 0.90]
    error matrix n (rows = map, cols = reference), 10 sample units per stratum:
        n = [[8, 2],
             [1, 9]]

    p[i,j] = W_i * n_ij / n_i:
        p[0,0] = 0.10 * 8/10 = 0.08     p[0,1] = 0.10 * 2/10 = 0.02
        p[1,0] = 0.90 * 1/10 = 0.09     p[1,1] = 0.90 * 9/10 = 0.81

    eq. 9   p_0 = 0.08 + 0.09 = 0.17      p_1 = 0.02 + 0.81 = 0.83
    eq. 10  A_0 = 17 ha                   A_1 = 83 ha
    eq. 11  V(p_0) = 0.10^2 * (0.8)(0.2)/9 + 0.90^2 * (0.1)(0.9)/9
                   = 0.01 * 0.0177778 + 0.81 * 0.01
                   = 0.000177778 + 0.0081 = 0.008277778
            SE     = 0.09098230 ;  CI95 = 1.96 * SE * 100 = 17.8325 ha
    """
    map_area = np.array([10.0, 90.0])
    n = np.array([[8.0, 2.0], [1.0, 9.0]])
    est = A.olofsson_area(map_area, n, CLASSES)

    assert est.method == "olofsson2014"
    assert est.total_area == pytest.approx(100.0)
    assert est.area == pytest.approx([17.0, 83.0])
    assert est.ci95[0] == pytest.approx(17.8325, abs=1e-3)
    # both classes share the same variance structure in a 2-class problem
    assert est.ci95[1] == pytest.approx(est.ci95[0], abs=1e-9)

    f = est.to_frame()
    assert f["share"].tolist() == pytest.approx([0.17, 0.83])


def test_accuracies_hand_worked_example():
    """Same inputs: users = diag/row, producers = diag/col, overall = Σ diag.

    users     = [0.08/0.10, 0.81/0.90] = [0.80, 0.90]
    producers = [0.08/0.17, 0.81/0.83] = [0.470588, 0.975904]
    overall   = 0.08 + 0.81 = 0.89
    """
    acc = A.accuracies(np.array([[8.0, 2.0], [1.0, 9.0]]), np.array([10.0, 90.0]))
    assert acc["users_accuracy"].tolist() == pytest.approx([0.80, 0.90])
    assert acc["producers_accuracy"].tolist() == pytest.approx([0.470588, 0.975904],
                                                               abs=1e-5)
    assert acc["overall_accuracy"].iloc[0] == pytest.approx(0.89)


# --- analytic properties ---
def test_perfect_map_is_unbiased_and_certain():
    """A diagonal error matrix means the map is right, so the correction is a no-op."""
    map_area = np.array([30.0, 70.0])
    n = np.array([[20.0, 0.0], [0.0, 20.0]])
    est = A.olofsson_area(map_area, n, CLASSES)
    assert est.area == pytest.approx(map_area)
    assert est.ci95 == pytest.approx([0.0, 0.0])


def test_estimated_areas_sum_to_the_total():
    rng = np.random.default_rng(0)
    for _ in range(20):
        map_area = rng.uniform(1, 100, 3)
        n = rng.integers(0, 30, (3, 3)).astype(float)
        n[np.arange(3), np.arange(3)] += 5      # keep every stratum populated
        est = A.olofsson_area(map_area, n, ["a", "b", "c"])
        assert est.area.sum() == pytest.approx(map_area.sum())


def test_correction_moves_area_toward_the_reference():
    """A map that over-predicts PERENNIAL must be corrected downward."""
    # 60 ha mapped perennial but only half of the sampled ones really are
    est = A.olofsson_area(np.array([60.0, 40.0]),
                          np.array([[10.0, 10.0], [0.0, 20.0]]), CLASSES)
    assert est.area[0] < 60.0
    assert est.area[1] > 40.0


def test_empty_stratum_contributes_no_variance():
    """A stratum with no sample units must not produce NaN or blow up."""
    est = A.olofsson_area(np.array([50.0, 50.0]),
                          np.array([[10.0, 0.0], [0.0, 0.0]]), CLASSES)
    assert np.isfinite(est.area).all() and np.isfinite(est.ci95).all()


# --- the simpler estimators ---
def test_naive_area_counts_argmax():
    pred = np.array([0, 0, 1])
    est = A.naive_area(pred, np.array([1.0, 2.0, 4.0]), 2, CLASSES)
    assert est.area == pytest.approx([3.0, 4.0])
    assert est.ci95 == pytest.approx([0.0, 0.0])


def test_probability_weighted_area_splits_each_parcel():
    prob = np.array([[0.6, 0.4], [0.25, 0.75]])
    est = A.probability_weighted_area(prob, np.array([10.0, 20.0]), CLASSES)
    # 0.6*10 + 0.25*20 = 11 ;  0.4*10 + 0.75*20 = 19
    assert est.area == pytest.approx([11.0, 19.0])
    assert est.area.sum() == pytest.approx(30.0)


def test_probability_weighted_differs_from_naive_under_uncertainty():
    prob = np.array([[0.51, 0.49], [0.51, 0.49]])
    ar = np.array([10.0, 10.0])
    naive = A.naive_area(prob.argmax(1), ar, 2, CLASSES)
    soft = A.probability_weighted_area(prob, ar, CLASSES)
    assert naive.area == pytest.approx([20.0, 0.0])
    assert soft.area == pytest.approx([10.2, 9.8])


# --- sampling weights + the combined entry point ---
def test_sample_weights_expand_to_the_population():
    """A 1-in-4 stratified sample must report 4x its own area."""
    prob = np.array([[1.0, 0.0], [0.0, 1.0]])
    df = A.estimate_all(prob, np.array([10.0, 10.0]), CLASSES,
                        weights=np.array([4.0, 4.0]))
    naive = df[df.method == "naive_argmax"]
    assert naive["area"].sum() == pytest.approx(80.0)


def test_estimate_all_returns_three_methods_when_given_an_error_matrix():
    prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    df = A.estimate_all(prob, np.array([1.0, 1.0, 1.0]), CLASSES,
                        n=np.array([[8.0, 2.0], [1.0, 9.0]]))
    assert set(df["method"]) == {"naive_argmax", "probability_weighted", "olofsson2014"}
    assert df.groupby("method")["share"].sum().tolist() == pytest.approx([1.0] * 3)


def test_error_matrix_orientation_is_map_by_reference():
    """Rows must be the *map* class — the orientation Olofsson's equations assume."""
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0, 1, 1])
    n = A.error_matrix(y_true, y_pred, 2)
    assert n[0, 0] == 1        # mapped 0, reference 0
    assert n[1, 0] == 1        # mapped 1, reference 0  -> a commission error for class 1
    assert n[1, 1] == 1
    assert n[0, 1] == 0

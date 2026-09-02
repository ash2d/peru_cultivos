"""Area estimation: naive, probability-weighted, and bias-corrected (plan D8, §9.4).

The headline number is a **per-year area share**. A naive count of argmax predictions is
biased by classifier error — with macro-F1 ≈ 0.7 the bias is comparable to the trend being
measured — and it flickers year to year. Three estimators are produced and all three are
shown:

(a) **naive** — Σ area of parcels whose argmax is the class. Biased.
(b) **probability-weighted** — Σ P(class) × area. Softens flicker, still biased, but a
    useful middle line.
(c) **Olofsson et al. (2014)** stratified estimator with 95 % CIs, error matrix from the
    locked-test confusion. The defensible one for a paper: corrects the bias *and* reports
    the uncertainty a bare trend line hides.

Reference: Olofsson, Foody, Herold, Stehman, Woodcock & Wulder (2014), "Good practices for
estimating area and assessing accuracy of land change", *Remote Sensing of Environment*
148:42-57 — equations 9-11 (stratified estimator), 3-5 (accuracies).

The map classes are the strata. Sampling weights (the panel is a stratified *sample* of
parcels, §7.2) multiply into the stratum areas so every estimate is a population estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AreaEstimate:
    """Per-class area with its uncertainty. ``ci95`` is the half-width."""

    classes: list[str]
    area: np.ndarray            # estimated area per class (same unit as the input)
    ci95: np.ndarray            # 95 % confidence half-width, 0 for uncorrected estimators
    method: str
    total_area: float

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "class": self.classes, "area": self.area, "ci95": self.ci95,
            "share": self.area / self.total_area,
            "share_ci95": self.ci95 / self.total_area,
            "method": self.method})


def naive_area(pred: np.ndarray, area: np.ndarray, n_classes: int,
               classes: list[str]) -> AreaEstimate:
    """(a) Σ area over parcels whose argmax is the class."""
    a = np.array([area[pred == c].sum() for c in range(n_classes)], dtype=float)
    return AreaEstimate(classes, a, np.zeros(n_classes), "naive_argmax", float(a.sum()))


def probability_weighted_area(prob: np.ndarray, area: np.ndarray,
                              classes: list[str]) -> AreaEstimate:
    """(b) Σ P(class) × area — every parcel contributes to every class in proportion."""
    a = (prob * area[:, None]).sum(axis=0)
    return AreaEstimate(classes, a, np.zeros(len(classes)), "probability_weighted",
                        float(a.sum()))


def error_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Counts ``n[i, j]`` = reference class *j* among sample units mapped as class *i*.

    Rows are **map** classes (the strata), columns **reference** classes — the orientation
    Olofsson's equations assume.
    """
    n = np.zeros((n_classes, n_classes), dtype=float)
    for i, j in zip(y_pred, y_true):
        n[int(i), int(j)] += 1
    return n


def olofsson_area(map_area: np.ndarray, n: np.ndarray,
                  classes: list[str]) -> AreaEstimate:
    """(c) Stratified bias-corrected area with 95 % CIs (Olofsson 2014 eqs. 9-11).

    ``map_area[i]`` is the mapped area of stratum *i* (from the full map, expanded by
    sampling weights); ``n[i, j]`` is the reference-sample error matrix. Strata with no
    sample units contribute their mapped area at zero estimated variance, not dropped.
    """
    n_cls = len(classes)
    total = float(map_area.sum())
    w = map_area / total                        # stratum weight W_i
    n_i = n.sum(axis=1)                         # sample size per map stratum

    # p[i, j]: estimated area proportion of (map i, reference j)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(n_i[:, None] > 0, w[:, None] * n / np.maximum(n_i[:, None], 1), 0.0)

    p_j = p.sum(axis=0)                         # eq. 9: estimated proportion of class j
    area = p_j * total                          # eq. 10

    # eq. 11: S(p_j)^2 = Σ_i W_i^2 * (n_ij/n_i)(1 - n_ij/n_i) / (n_i - 1)
    var = np.zeros(n_cls)
    for j in range(n_cls):
        s = 0.0
        for i in range(n_cls):
            if n_i[i] > 1:
                phat = n[i, j] / n_i[i]
                s += w[i] ** 2 * phat * (1 - phat) / (n_i[i] - 1)
        var[j] = s
    ci95 = 1.96 * np.sqrt(var) * total
    return AreaEstimate(classes, area, ci95, "olofsson2014", total)


def accuracies(n: np.ndarray, map_area: np.ndarray) -> pd.DataFrame:
    """Area-weighted user's / producer's / overall accuracy (eqs. 3-5)."""
    total = float(map_area.sum())
    w = map_area / total
    n_i = n.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(n_i[:, None] > 0, w[:, None] * n / np.maximum(n_i[:, None], 1), 0.0)
    p_i, p_j = p.sum(axis=1), p.sum(axis=0)
    diag = np.diag(p)
    with np.errstate(invalid="ignore", divide="ignore"):
        users = np.where(p_i > 0, diag / p_i, np.nan)
        producers = np.where(p_j > 0, diag / p_j, np.nan)
    return pd.DataFrame({"users_accuracy": users, "producers_accuracy": producers,
                         "overall_accuracy": diag.sum()})


def estimate_all(prob: np.ndarray, area: np.ndarray, classes: list[str],
                 n: np.ndarray | None = None,
                 weights: np.ndarray | None = None) -> pd.DataFrame:
    """All three estimators for one year, stacked. ``weights`` expands a sample panel to
    the population (§7.2); ``n`` is the locked-test error matrix."""
    w = np.ones(len(area)) if weights is None else np.asarray(weights, dtype=float)
    eff_area = np.asarray(area, dtype=float) * w
    pred = prob.argmax(1)
    out = [naive_area(pred, eff_area, len(classes), classes).to_frame(),
           probability_weighted_area(prob, eff_area, classes).to_frame()]
    if n is not None:
        map_area = np.array([eff_area[pred == c].sum() for c in range(len(classes))])
        out.append(olofsson_area(map_area, n, classes).to_frame())
    return pd.concat(out, ignore_index=True)

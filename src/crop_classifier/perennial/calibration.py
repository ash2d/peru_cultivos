"""Temperature scaling (plan decision D8, Phase 4 step 2).

The headline deliverable is a **probability-weighted, bias-corrected area share per year**,
and the abstain gate uses a confidence threshold. Both are meaningless on miscalibrated
probabilities — and the 12-class work explicitly found miscalibration under weighted
cross-entropy (``docs/PIPELINE.md`` §1: val loss rises while val macro-F1 rises).

Temperature scaling (Guo et al. 2017) is the minimal fix: one scalar ``T`` applied to the
logits, ``softmax(log p / T)``, fitted by minimising NLL on **held-out** validation
probabilities. It cannot change any argmax, so accuracy and macro-F1 are untouched — only
the confidences move.

Fitted on fold-0 validation predictions (data the final model never trained on), stored as
``temperature.json`` in the run dir, applied at inference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12


def apply_temperature(prob: np.ndarray, T: float) -> np.ndarray:
    """Re-soften/sharpen a probability matrix. ``T`` > 1 softens, < 1 sharpens."""
    if T == 1.0:
        return prob
    logp = np.log(np.clip(prob, EPS, None)) / T
    logp -= logp.max(axis=1, keepdims=True)      # stabilise before exponentiating
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def nll(prob: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(prob[np.arange(len(y)), y], EPS, None)).mean())


def expected_calibration_error(prob: np.ndarray, y: np.ndarray,
                               n_bins: int = 10) -> float:
    """Top-class ECE: mean |confidence - accuracy| weighted by bin population."""
    conf, pred = prob.max(1), prob.argmax(1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def fit_temperature(prob: np.ndarray, y: np.ndarray,
                    lo: float = 0.05, hi: float = 20.0,
                    tol: float = 1e-4) -> float:
    """Minimise NLL over ``T`` by golden-section search.

    NLL is unimodal in ``T``, so a derivative-free 1-D search is exact enough and avoids
    pulling in an optimiser (this module stays importable next to lightgbm *and* torch).
    """
    phi = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = nll(apply_temperature(prob, c), y), nll(apply_temperature(prob, d), y)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll(apply_temperature(prob, c), y)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll(apply_temperature(prob, d), y)
    return float((a + b) / 2)


def _prob_matrix(df: pd.DataFrame, classes: list[str]) -> np.ndarray:
    return np.nan_to_num(df[[f"prob_{c}" for c in classes]].to_numpy(dtype=float))


def calibrate_run(run_dir: Path, preds_file: str = "fold0/preds_val.parquet",
                  save: bool = True) -> dict:
    """Fit ``T`` on a run's held-out validation predictions and record the improvement."""
    run_dir = Path(run_dir)
    with open(run_dir / "label_map.json") as f:
        label_map = json.load(f)
    classes = [c for c, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    df = pd.read_parquet(run_dir / preds_file)
    prob, y = _prob_matrix(df, classes), df["y_true"].to_numpy(dtype=int)
    T = fit_temperature(prob, y)
    cal = apply_temperature(prob, T)
    out = {
        "temperature": T,
        "fitted_on": preds_file,
        "n": int(len(y)),
        "nll_before": nll(prob, y), "nll_after": nll(cal, y),
        "ece_before": expected_calibration_error(prob, y),
        "ece_after": expected_calibration_error(cal, y),
        # temperature scaling is argmax-preserving; this is the assertion, not a hope
        "argmax_unchanged": bool((prob.argmax(1) == cal.argmax(1)).all()),
    }
    print(f"temperature {T:.3f}  NLL {out['nll_before']:.4f} -> {out['nll_after']:.4f}  "
          f"ECE {out['ece_before']:.4f} -> {out['ece_after']:.4f}")
    if not out["argmax_unchanged"]:
        raise AssertionError("temperature scaling changed an argmax — numerically broken")
    if save:
        with open(run_dir / "temperature.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {run_dir / 'temperature.json'}")
    return out


def load_temperature(run_dir: Path) -> float:
    """The run's fitted temperature, or 1.0 (a no-op) if it was never calibrated."""
    p = Path(run_dir) / "temperature.json"
    if not p.exists():
        return 1.0
    with open(p) as f:
        return float(json.load(f)["temperature"])

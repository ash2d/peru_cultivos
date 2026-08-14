"""Metrics + reports (plan.md §11). Macro-F1 is the primary number; accuracy is context.

``compute_metrics`` is used by the trainer per fold; ``full_report`` writes
``metrics.json / confusion.png / per_class.csv / stratified.csv / reliability.png``
into a run dir from a saved predictions parquet.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = y_prob.argmax(1)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "accuracy": float((y_true == y_pred).mean()),
        "n": int(len(y_true)),
    }


def per_class_table(y_true, y_pred, classes: list[str]) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(classes)), zero_division=0)
    return pd.DataFrame({"class": classes, "precision": p, "recall": r, "f1": f,
                         "support": s})


def confusion_png(y_true, y_pred, classes: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(0.55 * len(classes) + 2,) * 2)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cmn[i, j] >= 0.01:
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set(xlabel="predicted", ylabel="true", title="row-normalised confusion")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def reliability_png(y_true, y_prob, path: Path, n_bins: int = 10) -> float:
    """Top-class reliability curve; returns the Brier score of the top-class prob."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conf = y_prob.max(1)
    correct = (y_prob.argmax(1) == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    mids, accs, fracs = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if m.any():
            mids.append((lo + hi) / 2)
            accs.append(correct[m].mean())
            fracs.append(m.mean())
    brier = float(brier_score_loss(correct, conf))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.plot(mids, accs, "o-")
    ax.set(xlabel="confidence", ylabel="accuracy", title=f"reliability (Brier {brier:.3f})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return brier


def stratified_table(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Macro-F1 by area / coverage / year strata (§11) from a predictions frame."""
    rows = []
    strata = {
        "area": pd.cut(pred_df["area_ha"], [0, 0.25, 0.5, 1, 2, 5, 50]),
        "n_valid_obs": pd.cut(pred_df["n_valid_obs"].astype(float),
                              [3, 5, 8, 12, 20, 1000]),
        "max_gap": pd.cut(pred_df["max_gap"].astype(float), [-1, 2, 4, 6, 12]),
        "year": pred_df["year"],
    }
    for name, s in strata.items():
        for g, sub in pred_df.groupby(s, observed=True):
            if len(sub) < 10:
                continue
            rows.append({"stratum": name, "bin": str(g), "n": len(sub),
                         "macro_f1": f1_score(sub["y_true"], sub["y_pred"],
                                              average="macro"),
                         "accuracy": float((sub["y_true"] == sub["y_pred"]).mean())})
    return pd.DataFrame(rows)


def full_report(run_dir: Path, preds_file: str = "preds_test.parquet") -> dict:
    """Write the §11 report bundle for a saved predictions parquet in ``run_dir``."""
    run_dir = Path(run_dir)
    df = pd.read_parquet(run_dir / preds_file)
    with open(run_dir / "label_map.json") as f:
        label_map = json.load(f)
    classes = [c for c, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    prob_cols = [f"prob_{c}" for c in classes]
    y_true = df["y_true"].values
    y_prob = np.nan_to_num(df[prob_cols].values)  # a diverged model must not crash reports
    y_pred = y_prob.argmax(1)
    df["y_pred"] = y_pred

    m = compute_metrics(y_true, y_prob)
    m["majority_baseline_acc"] = float(np.bincount(y_true).max() / len(y_true))
    m["coverage"] = float(df["quality_ok"].mean()) if "quality_ok" in df else 1.0
    per_class_table(y_true, y_pred, classes).to_csv(run_dir / "per_class.csv", index=False)
    confusion_png(y_true, y_pred, classes, run_dir / "confusion.png")
    m["brier_top"] = reliability_png(y_true, y_prob, run_dir / "reliability.png")
    if {"area_ha", "n_valid_obs"} <= set(df.columns):
        stratified_table(df).to_csv(run_dir / "stratified.csv", index=False)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    print(json.dumps(m, indent=2))
    return m

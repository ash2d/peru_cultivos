"""Pooled spatial-CV comparison across models (plan §6 step 1).

Produces the same bundle as ``runs/model_comparison_12c/``: a metrics table plus three
figures (per-fold macro-F1, pooled metrics, per-class F1). Every number comes from
``preds_cv.parquet`` — pooled *validation* folds, never the locked test set.

Run with::

    CC_RUNS=runs/perennial uv run python -m crop_classifier.perennial.compare \\
        runs/perennial/rules_3c runs/perennial/lightgbm_3c … --out runs/perennial/comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from crop_classifier.evaluate import compute_metrics, per_class_table


def _load(run_dir: Path, preds_file: str = "preds_cv.parquet"):
    run_dir = Path(run_dir)
    with open(run_dir / "label_map.json") as f:
        label_map = json.load(f)
    classes = [c for c, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    df = pd.read_parquet(run_dir / preds_file)
    prob = np.nan_to_num(df[[f"prob_{c}" for c in classes]].to_numpy(dtype=float))
    return df, prob, classes


def summarise(run_dir: Path, preds_file: str = "preds_cv.parquet") -> dict:
    """Pooled metrics + the per-fold spread for one run."""
    run_dir = Path(run_dir)
    df, prob, classes = _load(run_dir, preds_file)
    y = df["y_true"].to_numpy(dtype=int)
    m = compute_metrics(y, prob)
    m["model"] = run_dir.name
    # majority-class baselines: the bar every model must clear (criterion S1)
    counts = np.bincount(y, minlength=len(classes))
    maj = int(counts.argmax())
    m["majority_baseline_acc"] = float(counts.max() / len(y))
    m["majority_baseline_macro_f1"] = float(
        f1_score(y, np.full_like(y, maj), average="macro", labels=range(len(classes)),
                 zero_division=0))
    cv_path = run_dir / "cv_metrics.json"
    if cv_path.exists():
        with open(cv_path) as f:
            cv = json.load(f)
        m["cv_macro_f1_mean"] = cv["cv_macro_f1_mean"]
        m["cv_macro_f1_std"] = cv["cv_macro_f1_std"]
        m["fold_macro_f1"] = [f["macro_f1"] for f in cv["folds"]]
    return m


def per_class_frame(run_dir: Path, preds_file: str = "preds_cv.parquet") -> pd.DataFrame:
    df, prob, classes = _load(run_dir, preds_file)
    t = per_class_table(df["y_true"].to_numpy(dtype=int), prob.argmax(1), classes)
    t["model"] = Path(run_dir).name
    return t


def compare(run_dirs: list[Path], out_dir: Path,
            preds_file: str = "preds_cv.parquet") -> pd.DataFrame:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [summarise(r, preds_file) for r in run_dirs]
    tbl = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    per_class = pd.concat([per_class_frame(r, preds_file) for r in run_dirs],
                          ignore_index=True)

    cols = ["model", "macro_f1", "balanced_accuracy", "accuracy", "cohen_kappa",
            "cv_macro_f1_mean", "cv_macro_f1_std", "majority_baseline_macro_f1",
            "majority_baseline_acc", "n"]
    flat = tbl[[c for c in cols if c in tbl.columns]]
    flat.to_csv(out_dir / "model_comparison.csv", index=False)
    per_class.to_csv(out_dir / "per_class_f1.csv", index=False)
    print(flat.to_string(index=False))

    # ---- per-fold macro-F1 (the honest spatial-CV error bar) ----
    if "fold_macro_f1" in tbl:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for _, r in tbl.iterrows():
            folds = r.get("fold_macro_f1")
            if isinstance(folds, list):
                ax.plot(range(len(folds)), folds, "o-", label=r["model"])
        ax.set(xlabel="spatial CV fold", ylabel="macro-F1",
               title="per-fold macro-F1 (regional difficulty)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "cv_macro_f1_by_fold.png", dpi=130)
        plt.close(fig)

    # ---- pooled metrics ----
    metrics = ["macro_f1", "balanced_accuracy", "accuracy", "cohen_kappa"]
    fig, ax = plt.subplots(figsize=(1.6 * len(tbl) + 4, 4))
    w = 0.8 / len(metrics)
    x = np.arange(len(tbl))
    for i, mt in enumerate(metrics):
        ax.bar(x + i * w, tbl[mt], w, label=mt)
    ax.axhline(tbl["majority_baseline_macro_f1"].iloc[0], color="k", ls="--", lw=1,
               label="majority macro-F1")
    ax.axhline(tbl["majority_baseline_acc"].iloc[0], color="grey", ls=":", lw=1,
               label="majority accuracy")
    ax.set_xticks(x + 0.4, tbl["model"], rotation=20, ha="right", fontsize=8)
    ax.set(ylabel="score", title="pooled spatial-CV metrics")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "pooled_cv_metrics.png", dpi=130)
    plt.close(fig)

    # ---- per-class F1 ----
    piv = per_class.pivot(index="class", columns="model", values="f1")
    fig, ax = plt.subplots(figsize=(1.2 * len(piv) + 4, 4))
    piv.plot.bar(ax=ax, width=0.8)
    ax.set(ylabel="F1", title="per-class F1 (pooled spatial CV)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "per_class_f1.png", dpi=130)
    plt.close(fig)

    print(f"\nwrote {out_dir}/model_comparison.csv + 3 figures")
    return flat


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare runs on pooled spatial CV")
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--preds", default="preds_cv.parquet")
    a = ap.parse_args()
    compare(a.runs, a.out, a.preds)


if __name__ == "__main__":
    main()

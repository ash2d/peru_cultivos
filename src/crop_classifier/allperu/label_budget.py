"""How many labelled parcels does a land-state classifier need? — measured, not assumed.

Sizes a photo-interpretation campaign (`docs/s2_labelling/plan.md`) by running a
learning curve on the **existing** national Landsat store and PETT labels. That is a proxy for
a Sentinel-2 campaign, and a deliberately conservative one: S2 gives ~9x the pixels per parcel,
so the achievable ceiling should be higher. What transfers is the *shape* — where the curve
flattens — which is the question a budget needs answered.

Two draw protocols, because the difference between them turns out to be worth ~8x the budget:

* ``region`` — whole 5 km regions, as every other sample in this project is drawn. This is
  what an unstratified campaign yields.
* ``stratified`` — equal parcels per class, scattered. This is what a campaign stratified on
  *predicted* class yields, which is what the endpoint plan specifies. Validation is always the
  untouched spatially-separated fold, so scattered training parcels cannot inflate the score.

Run: ``uv run python -m crop_classifier.allperu.label_budget``
(needs ``CC_PROC``/``CC_FEAT`` pointing at the national workspace).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from crop_classifier.data import fold_split, load_parcels, make_flat
from crop_classifier.paths import proc

SIZES = (500, 1000, 2000, 4000, 8000, 16000)
FOLDS = (0, 1, 2)
SEEDS = (0, 1, 2)
# Location features are withheld throughout: LODO showed centroid_lat is spatial
# memorisation, and a label-budget curve computed with it would be measuring the wrong thing.
NOLAT = ("centroid_lat", "centroid_lon")


def _drop_lat(X: pd.DataFrame) -> pd.DataFrame:
    return X.drop(columns=[c for c in NOLAT if c in X.columns])


def _draw_regions(parcels: pd.DataFrame, idx: pd.Index, cod: np.ndarray, n_target: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Whole regions until n_target parcels are reached.

    Returns positions into the *feature matrix* (``cod``), not into ``idx`` — make_flat
    inner-joins the store, so the two differ by the parcels the store lacks.
    """
    sub = parcels.loc[idx]
    regions = np.asarray(sub["region_id"].unique(), dtype=object)
    rng.shuffle(regions)
    sizes = sub.groupby("region_id").size()
    keep, total = [], 0
    for r in regions:
        keep.append(r)
        total += int(sizes[r])
        if total >= n_target:
            break
    want = set(sub.loc[sub["region_id"].isin(keep), "COD_PREDIO"])
    return np.where(pd.Index(cod).isin(want))[0]


def _draw_stratified(y: np.ndarray, n_target: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Equal parcels per class, scattered."""
    per_cls = n_target // len(np.unique(y))
    return np.concatenate([
        rng.choice(np.where(y == c)[0], min(per_cls, int((y == c).sum())), replace=False)
        for c in np.unique(y)])


def curve(protocol: str = "stratified", sizes=SIZES, folds=FOLDS,
          seeds=SEEDS) -> pd.DataFrame:
    import lightgbm as lgb

    parcels = load_parcels(require_quality=True)
    with open(proc() / "label_map.json") as f:
        per_id = json.load(f)["PERENNIAL"]
    rows = []
    for fold in folds:
        tr_idx, va_idx = fold_split(parcels, fold)
        va = make_flat(parcels, va_idx)
        Xva, yva_raw = _drop_lat(va.X), va.y
        tr = make_flat(parcels, tr_idx)
        Xtr, ytr_raw = _drop_lat(tr.X), tr.y
        for size in sizes:
            for seed in seeds:
                rng = np.random.default_rng(97 * fold + seed)
                sel = (_draw_regions(parcels, tr_idx, tr.cod_predio, size, rng)
                       if protocol == "region" else _draw_stratified(ytr_raw, size, rng))
                for binary in (False, True):
                    ytr = (ytr_raw == per_id).astype(int) if binary else ytr_raw
                    yva = (yva_raw == per_id).astype(int) if binary else yva_raw
                    if len(np.unique(ytr[sel])) < 2:
                        continue
                    clf = lgb.LGBMClassifier(
                        objective="binary" if binary else "multiclass",
                        num_class=None if binary else 3, n_estimators=400,
                        learning_rate=0.05, num_leaves=31, class_weight="balanced",
                        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                        verbosity=-1, n_jobs=8, random_state=seed)
                    clf.fit(Xtr.iloc[sel], ytr[sel])
                    pred = clf.predict(Xva)
                    rows.append({
                        "protocol": protocol, "target": "binary" if binary else "3class",
                        "fold": fold, "seed": seed, "size_target": size,
                        "n_labels": len(sel),
                        "macro_f1": f1_score(yva, pred, average="macro"),
                        "perennial_f1": (f1_score(yva, pred) if binary
                                         else f1_score(yva, pred, average=None)[per_id])})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    # Group on the REQUESTED size: a whole-region draw overshoots by a variable amount,
    # so grouping on the realised count would split every point into singletons.
    g = (df.groupby(["protocol", "target", "size_target"])
         .agg(n_labels=("n_labels", "mean"), macro_f1=("macro_f1", "mean"),
              macro_sd=("macro_f1", "std"),
              perennial_f1=("perennial_f1", "mean"), n_runs=("macro_f1", "size"))
         .reset_index())
    ceil = (g.sort_values("size_target").groupby(["protocol", "target"])
            .agg(c_macro=("macro_f1", "last"), c_per=("perennial_f1", "last")))
    g = g.merge(ceil, on=["protocol", "target"])
    g["pct_of_ceiling"] = 100 * g["macro_f1"] / g["c_macro"]
    g["pct_perennial"] = 100 * g["perennial_f1"] / g["c_per"]
    return g.drop(columns=["c_macro", "c_per"])


def figure(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from crop_classifier.perennial.figures import C_GRID, C_INK, C_SERIES
    C2 = "#e8961f"

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, target, title in ((axes[0], "3class", "3 classes — perennial / annual / other"),
                              (axes[1], "binary", "binary — perennial vs everything else")):
        for proto, c, name in (("stratified", C_SERIES, "stratified by class (recommended)"),
                               ("region", C2, "whole 5 km blocks (unstratified)")):
            s = summary[(summary.target == target) & (summary.protocol == proto)]
            s = s.sort_values("size_target")
            if s.empty:
                continue
            ax.plot(s["size_target"], s["macro_f1"], "-o", ms=6, lw=2.3, color=c, label=name)
            ax.fill_between(s["size_target"], s["macro_f1"] - s["macro_sd"].fillna(0),
                            s["macro_f1"] + s["macro_sd"].fillna(0), color=c, alpha=0.15, lw=0)
        ax.set_xscale("log")
        ax.set_xlabel("labelled parcels", fontsize=9, color=C_INK)
        ax.set_ylabel("macro-F1 on held-out regions", fontsize=9, color=C_INK)
        ax.set_title(title, fontsize=9.5, color=C_INK)
        ax.axvline(2000, color=C_INK, ls=":", lw=1.1, alpha=0.7)
        ax.annotate("2,000", (2000, ax.get_ylim()[0]), xytext=(4, 4),
                    textcoords="offset points", fontsize=8, color=C_INK)
        ax.grid(alpha=0.25, color=C_GRID, lw=0.6)
        ax.tick_params(labelsize=8.5, colors=C_INK)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("How many labelled parcels are needed? Stratifying by class is worth about "
                 "8x the labelling budget\n"
                 "Landsat features + PETT labels as a conservative proxy for a Sentinel-2 "
                 "campaign. Shading = SD over 3 folds x 3 draws.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def build(out_dir: Path | str = "docs/figures") -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.concat([curve("region"), curve("stratified")], ignore_index=True)
    summary = summarise(df)
    summary.to_csv(out / "label_budget_curve.csv", index=False)
    figure(summary, out / "label_budget.png")
    return summary


if __name__ == "__main__":
    print(build().round(3).to_string(index=False))

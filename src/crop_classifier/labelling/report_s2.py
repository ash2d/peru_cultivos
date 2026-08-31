"""Readable results for the S2 endpoint-labelling campaign.

Everything this module prints is meant to be pasted into a document and understood there:
each table carries its own units, its own n, and — where a gate is involved — the criterion
next to the reading, so nothing has to be looked up elsewhere.

⚠️ **Nothing here touches the locked test split.** Every number is either cross-validation
inside ``trainval`` or leave-one-department-out over ``trainval`` + pilot. The 201-parcel
locked test (161 of them labelled) is not read by any function in this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from crop_classifier.paths import ROOT

RUNS = ROOT / "runs" / "s2_labels"
LABELS_DIR = ROOT / "data" / "processed" / "all_peru" / "labels_s2"

MODEL_LABEL = {"lightgbm": "LightGBM", "ltae": "LTAE", "rules": "rules"}
TARGET_DESC = {
    "t5": "all 5 labelled classes",
    "t4": "NON_AGRICULTURE pooled into OTHER",
    "t3": "3 classes; WOODY_NON_CROP + NON_AGRICULTURE dropped",
    "t3w": "3 classes; WOODY_NON_CROP counted as PERENNIAL",
}


def _arms() -> list[tuple[str, bool, str, Path]]:
    out = []
    for f in sorted(RUNS.glob("*/*/cv_metrics.json")):
        ws = f.parent.parent.name
        out.append((ws.replace("ws_", "").replace("_pilot", ""),
                    ws.endswith("_pilot"), f.parent.name, f.parent))
    return out


def _classes(ws: Path) -> list[str]:
    with open(ws / "label_map.json") as f:
        m = json.load(f)
    return [c for c, _ in sorted(m.items(), key=lambda kv: kv[1])]


def oof(run_dir: Path) -> pd.DataFrame:
    """Pooled out-of-fold predictions: every trainval parcel predicted exactly once."""
    frames = []
    for f in sorted(run_dir.glob("fold*/preds_val.parquet")):
        d = pd.read_parquet(f)
        d["fold"] = int(f.parent.name.replace("fold", ""))
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    probs = [c for c in d.columns if c.startswith("prob_")]
    d["y_pred"] = np.array([c[5:] for c in probs])[d[probs].to_numpy().argmax(1)]
    return d


# ---------------------------------------------------------------------------------
def headline() -> pd.DataFrame:
    """The full arm table: CV and LODO macro-F1 side by side, with fold/department spread."""
    rows = []
    for tgt, pilot, model, run in _arms():
        m = json.loads((run / "cv_metrics.json").read_text())
        f1 = [x["macro_f1"] for x in m["folds"]]
        ws = LABELS_DIR / f"ws_{tgt}{'_pilot' if pilot else ''}"
        lf = ws / f"lodo_{model}.csv"
        lodo = pd.read_csv(lf) if lf.exists() else None
        rows.append({
            "target": tgt, "train_pool": "trainval+pilot" if pilot else "trainval",
            "model": MODEL_LABEL.get(model, model),
            "cv_macro_f1": round(float(np.mean(f1)), 3),
            "cv_sd": round(float(np.std(f1, ddof=1)), 3),
            "cv_worst_fold": round(min(f1), 3), "cv_best_fold": round(max(f1), 3),
            "lodo_macro_f1": round(lodo.macro_f1.mean(), 3) if lodo is not None else np.nan,
            "lodo_sd": round(lodo.macro_f1.std(), 3) if lodo is not None else np.nan,
            "lodo_worst_dept": (lodo.loc[lodo.macro_f1.idxmin(), "dept"]
                                if lodo is not None else ""),
            "n_train_per_fold": int(np.mean([x["n_train"] for x in m["folds"]])),
            "n_val_per_fold": int(np.mean([x["n_val"] for x in m["folds"]])),
        })
    t = pd.DataFrame(rows).sort_values(["target", "train_pool", "model"])
    t.to_csv(LABELS_DIR / "model_comparison.csv", index=False)
    return t


def paired_folds(target: str, pilot: bool, a: str, b: str) -> dict:
    """Paired per-fold difference ``a - b`` in macro-F1, with a two-sided t-test.

    Paired because both models see identical folds; a t-test on 5 paired differences is
    weak evidence and is reported as such rather than as a verdict.
    """
    from scipy import stats
    suf = "_pilot" if pilot else ""
    def f1s(m):
        p = RUNS / f"ws_{target}{suf}" / m / "cv_metrics.json"
        return np.array([x["macro_f1"] for x in json.loads(p.read_text())["folds"]])
    da, db = f1s(a), f1s(b)
    d = da - db
    t, p = stats.ttest_rel(da, db)
    return {"target": target, "train_pool": "trainval+pilot" if pilot else "trainval",
            "comparison": f"{MODEL_LABEL.get(a, a)} - {MODEL_LABEL.get(b, b)}",
            "mean_diff_macro_f1": round(float(d.mean()), 3),
            "folds_won": int((d > 0).sum()), "n_folds": len(d),
            "p_value": round(float(p), 3)}


def per_class(target: str, pilot: bool, model: str) -> pd.DataFrame:
    """Precision / recall / F1 / support per class, pooled over all 5 out-of-fold sets."""
    from sklearn.metrics import precision_recall_fscore_support
    suf = "_pilot" if pilot else ""
    run = RUNS / f"ws_{target}{suf}" / model
    ws = LABELS_DIR / f"ws_{target}{suf}"
    classes = _classes(ws)
    d = oof(run)
    yt = d["y_true"].values
    yp = np.array([classes.index(c) for c in d["y_pred"]])
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=range(len(classes)),
                                                 zero_division=0)
    return pd.DataFrame({"class": classes, "precision": p.round(3), "recall": r.round(3),
                         "f1": f.round(3), "n_parcels": s})


def confusion(target: str, pilot: bool, model: str) -> pd.DataFrame:
    """Counts of parcels: rows = what the human called it, columns = what the model said."""
    suf = "_pilot" if pilot else ""
    ws = LABELS_DIR / f"ws_{target}{suf}"
    classes = _classes(ws)
    d = oof(RUNS / f"ws_{target}{suf}" / model)
    d["human"] = [classes[i] for i in d["y_true"]]
    cm = pd.crosstab(d["human"], d["y_pred"]).reindex(index=classes, columns=classes,
                                                      fill_value=0)
    cm.index.name = "human label"
    cm.columns.name = "model prediction"
    return cm


def lodo_table(target: str, models: list[str] | None = None) -> pd.DataFrame:
    """Per-department macro-F1 for each model, one row per department."""
    models = models or ["lightgbm", "rules", "ltae"]
    ws = LABELS_DIR / f"ws_{target}_pilot"
    out = None
    for m in models:
        f = ws / f"lodo_{m}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)[["dept", "n", "macro_f1"]].rename(
            columns={"macro_f1": MODEL_LABEL.get(m, m), "n": "n_parcels_held_out"})
        out = d if out is None else out.merge(
            d.drop(columns="n_parcels_held_out"), on="dept")
    return out.sort_values("dept") if out is not None else pd.DataFrame()


def paired_depts(target: str, a: str, b: str) -> dict:
    """Paired per-**department** difference ``a - b`` in held-out macro-F1.

    The out-of-distribution counterpart to ``paired_folds``. Both models are scored on the
    same held-out departments, so the pairing is exact; with 10-14 departments this is the
    largest paired sample the campaign has, and it is still small.
    """
    from scipy import stats
    ws = LABELS_DIR / f"ws_{target}_pilot"
    da = pd.read_csv(ws / f"lodo_{a}.csv").set_index("dept")["macro_f1"]
    db = pd.read_csv(ws / f"lodo_{b}.csv").set_index("dept")["macro_f1"]
    idx = da.index.intersection(db.index)
    d = (da[idx] - db[idx]).values
    t, p = stats.ttest_rel(da[idx], db[idx])
    return {"target": target,
            "comparison": f"{MODEL_LABEL.get(a, a)} - {MODEL_LABEL.get(b, b)}",
            "mean_diff_macro_f1": round(float(d.mean()), 3),
            "depts_won": int((d > 0).sum()), "n_depts": len(d),
            "p_value": round(float(p), 3)}


def cv_vs_lodo(target: str) -> pd.DataFrame:
    """The gap that decides everything on this data: same model, seen vs unseen department."""
    h = headline()
    h = h[(h.target == target) & (h.train_pool == "trainval+pilot")]
    out = h[["model", "cv_macro_f1", "lodo_macro_f1"]].copy()
    out["drop_cv_to_lodo"] = (out.cv_macro_f1 - out.lodo_macro_f1).round(3)
    return out.sort_values("lodo_macro_f1", ascending=False)


# ---------------------------------------------------------------------------------
def majority_baseline(target: str, pilot: bool = True) -> dict:
    """What "always guess the largest class" scores on this target.

    ⚠️ **This is why macro-F1 must never be compared across targets with different class
    counts.** Averaging F1 over 2 classes when one is never predicted still collects a full
    score on the other and divides by 2, so the floor rises as classes are removed: 0.124
    (5-class) → 0.171 (4-class) → 0.228/0.253 (3-class) → **0.41–0.47 (2-class)**. A model
    that "improves" from 0.67 to 0.72 by collapsing classes has usually lost skill.
    """
    import geopandas as gpd
    from sklearn.metrics import f1_score

    ws = LABELS_DIR / f"ws_{target}{'_pilot' if pilot else ''}"
    d = gpd.read_parquet(ws / "modeling_parcels.parquet")
    d = d[d["split"] == "trainval"]
    y = d["label_id"].values
    maj = np.bincount(y).argmax()
    return {"target": target, "n_trainval": len(d),
            "n_classes": int(d["label_id"].nunique()),
            "majority_class_share": round(float(np.bincount(y).max() / len(y)), 3),
            "majority_macro_f1": round(
                float(f1_score(y, np.full_like(y, maj), average="macro")), 3)}


def perennial_f1(target: str, pilot: bool, model: str) -> dict:
    """F1 for the `PERENNIAL` class alone, pooled out-of-fold.

    The one number that **is** comparable across targets: it is the same class in each,
    scored the same way, and it is unaffected by how many other classes exist. Use it, not
    macro-F1, to answer "did collapsing the label space help?".
    """
    from sklearn.metrics import precision_recall_fscore_support
    suf = "_pilot" if pilot else ""
    ws = LABELS_DIR / f"ws_{target}{suf}"
    classes = _classes(ws)
    if "PERENNIAL" not in classes:
        return {}
    d = oof(RUNS / f"ws_{target}{suf}" / model)
    i = classes.index("PERENNIAL")
    yt = (d["y_true"].values == i)
    yp = (np.array([classes.index(c) for c in d["y_pred"]]) == i)
    p, r, f, _ = precision_recall_fscore_support(yt, yp, average="binary",
                                                 zero_division=0)
    return {"target": target, "train_pool": "trainval+pilot" if pilot else "trainval",
            "model": MODEL_LABEL.get(model, model),
            "perennial_precision": round(float(p), 3),
            "perennial_recall": round(float(r), 3),
            "perennial_f1": round(float(f), 3),
            "n_perennial": int(yt.sum()), "n_parcels": int(len(yt)),
            "prevalence": round(float(yt.mean()), 3)}


def skill_table(models: tuple[str, ...] = ("lightgbm", "ltae"),
                pilot: bool = True) -> pd.DataFrame:
    """Every target's macro-F1 **against its own majority-class floor**.

    A raw macro-F1 cannot be read across targets, because the floor moves with the number
    of classes (0.124 at 5 classes, 0.467 at 2). ``skill`` normalises it —
    ``(score - floor) / (1 - floor)`` — so 0 is "no better than always guessing the largest
    class" and 1 is perfect, on every target alike. Reported for CV *and* LODO, because the
    two disagree about which model wins.
    """
    h = headline()
    h = h[h.train_pool == ("trainval+pilot" if pilot else "trainval")]
    base = {t: majority_baseline(t, pilot)["majority_macro_f1"]
            for t in h["target"].unique()}
    inv = {v: k for k, v in MODEL_LABEL.items()}
    rows = []
    for _, r in h.iterrows():
        m = inv.get(r["model"], r["model"])
        if m not in models:
            continue
        b = base[r["target"]]
        per = perennial_f1(r["target"], pilot, m)
        rows.append({
            "target": r["target"], "n_classes": majority_baseline(
                r["target"], pilot)["n_classes"],
            "model": r["model"], "majority_floor": b,
            "cv_macro_f1": r["cv_macro_f1"],
            "cv_skill": round((r["cv_macro_f1"] - b) / (1 - b), 3),
            "lodo_macro_f1": r["lodo_macro_f1"],
            "lodo_skill": round((r["lodo_macro_f1"] - b) / (1 - b), 3)
            if pd.notna(r["lodo_macro_f1"]) else np.nan,
            "perennial_f1": per.get("perennial_f1", np.nan),
        })
    order = {t: i for i, t in enumerate(["t5", "t4", "t3", "t3w", "t2", "t2w"])}
    out = pd.DataFrame(rows)
    return out.sort_values(["target", "model"], key=lambda s: s.map(order).fillna(-1)
                           if s.name == "target" else s)

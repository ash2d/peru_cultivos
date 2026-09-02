"""Climate covariates as extra model inputs on the S2 endpoint labels.

Arms ``none`` / ``temp`` (``tmean_c``) / ``rain`` (``precip_mm_yr``) / ``both``, fitted per
model class against the *same* parcels, frozen folds and seeds. The two columns are
WorldClim 2.1 1970–2000 normals sampled at each parcel centroid (``allperu.climate``).

⚠️ Two warnings that decide how the result must be read:
1. A 1 km climate surface is a smooth function of location, so these columns proxy
   ``centroid_lat`` — the project's canonical CV-up / LODO-down feature (``RESULTS.md``
   §4.2). Every arm is reported CV *and* LODO; ``location_proxy_audit`` measures how much
   department identity the columns carry.
2. The normals are time-invariant — fine here only: a single-epoch endpoint classifier has
   no trend for a constant to manufacture. Do not carry these columns into a panel.

Each variant is its own ``CC_PROC`` workspace, symlinking the base workspace's
``modeling_parcels.parquet`` and ``label_map.json`` so split / folds / dead-zones are
identical across arms by construction. The flat path reads the extra columns from the
feature parquet; the LTAE path has no static input, so the variant also writes
``statics.npz`` for ``data.SeqDataset`` / ``models.ltae``.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from crop_classifier.labelling import train_prep as P
from crop_classifier.paths import ROOT

CLIMATE_PARQUET = ROOT / "data" / "processed" / "climate" / "parcel_climate_normals.parquet"
CLIMATE_SETS: dict[str, list[str]] = {
    "none": [],
    "temp": ["tmean_c"],
    "rain": ["precip_mm_yr"],
    "both": ["tmean_c", "precip_mm_yr"],
    # ⭐ the control, not a proposal: two raw coordinates in place of the climate columns —
    # same count, same time-invariance, same smoothness, no agro-climatic content. If
    # `latlon` buys as much as `both`, climate is contributing location (RESULTS.md §4.2).
    "latlon": ["centroid_lat", "centroid_lon"],
}
ARMS = list(CLIMATE_SETS)
CLIMATE_ONLY = ["tmean_c", "precip_mm_yr"]
LOCATION_ONLY = ["centroid_lat", "centroid_lon"]


def ws_dir(target: str, climate: str, include_pilot: bool = True) -> Path:
    """Workspace for one (target, climate) arm. ``none`` is the existing base workspace."""
    base = P.ws_dir(target, include_pilot)
    return base if climate == "none" else base.parent / f"{base.name}__clim_{climate}"


def _covariate_table(parcels: gpd.GeoDataFrame) -> pd.DataFrame:
    """Climate normals and centroid coordinates for these parcels, one row each.

    Coordinates come from the parcel's own geometry, not a stored column, so the control
    arm covers exactly the same parcels as the climate arms.
    """
    if not CLIMATE_PARQUET.exists():
        raise SystemExit(
            f"missing {CLIMATE_PARQUET} — build it once with\n"
            f"  uv run python -m crop_classifier.cli allperu climate normals")
    cl = pd.read_parquet(CLIMATE_PARQUET, columns=["COD_PREDIO", *CLIMATE_ONLY])
    cl["COD_PREDIO"] = cl["COD_PREDIO"].astype(str)
    pt = parcels.geometry.representative_point().to_crs(4326)
    want = pd.DataFrame({"COD_PREDIO": parcels["COD_PREDIO"].astype(str).values,
                         "centroid_lat": pt.y.values, "centroid_lon": pt.x.values})
    out = want.drop_duplicates("COD_PREDIO").merge(cl, on="COD_PREDIO", how="left")
    miss = int(out[CLIMATE_ONLY].isna().any(axis=1).sum())
    if miss:
        # a climate NaN drops the parcel from LightGBM's split logic differently per arm —
        # exactly the comparison being made
        raise SystemExit(f"{miss} of {len(out)} labelled parcels have no climate value; "
                         f"rebuild the normals over a parcel table that covers them")
    return out


def build_variant(target: str = "t4", climate: str = "both",
                  include_pilot: bool = True, verbose: bool = True) -> Path:
    """Base workspace + climate columns -> a ``CC_PROC`` workspace for one arm."""
    if climate not in CLIMATE_SETS:
        raise SystemExit(f"unknown climate arm {climate!r}; expected {ARMS}")
    base = P.ws_dir(target, include_pilot)
    if not (base / "modeling_parcels.parquet").exists():
        # a climate arm is the base label set plus two columns — build the base here
        print(f"{base.name} not built yet — building it first")
        P.build_workspace(target, include_pilot=include_pilot)
    if climate == "none":
        return base

    out = ws_dir(target, climate, include_pilot)
    (out / "features").mkdir(parents=True, exist_ok=True)
    for fn in ("modeling_parcels.parquet", "label_map.json"):
        link = out / fn
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(base / fn)

    cols = CLIMATE_SETS[climate]
    parcels = gpd.read_parquet(base / "modeling_parcels.parquet")
    clim = _covariate_table(parcels)

    feats = pd.read_parquet(P.FEAT_S2 / "s2_features_lightgbm.parquet")
    feats["COD_PREDIO"] = feats["COD_PREDIO"].astype(str)
    feats = feats.merge(clim[["COD_PREDIO", *cols]], on="COD_PREDIO", how="left")
    feats.to_parquet(out / "features" / "features_lightgbm.parquet", index=False)

    # the sequence tensor is unchanged; the statics ride alongside it
    link = out / "features" / "tensor_perdate.npz"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(P.FEAT_S2 / "tensor_perdate.npz")

    st = clim.set_index("COD_PREDIO").loc[:, cols]
    np.savez(out / "features" / "statics.npz",
             cod_predio=np.array(st.index, dtype=object),
             X=st.to_numpy(dtype="float32"),
             names=np.array(cols, dtype=object))

    if verbose:
        j = parcels[["COD_PREDIO", "label", "dept"]].merge(clim, on="COD_PREDIO")
        print(f"[{target}{'+pilot' if include_pilot else ''} | clim={climate}] "
              f"{len(parcels)} parcels, +{len(cols)} column(s) {cols} -> {out}")
        print(j.groupby("label")[cols].agg(["mean", "std"]).round(2).to_string())
    return out


def build_all(target: str = "t4", include_pilot: bool = True) -> None:
    for c in ARMS:
        build_variant(target, c, include_pilot)
        print()


# --- Is climate just latitude again? ---
def location_proxy_audit(target: str = "t4", include_pilot: bool = True) -> dict:
    """How much of *where the parcel is* do the two climate columns carry?

    ``centroid_lat`` gains +0.047 CV / loses 0.060 LODO (§4.2) by letting the model name
    the department. The honest question is whether a model can recover the department from
    these two numbers alone — a 5-fold accuracy well above the department prior means the
    same failure mode is available.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    base = P.ws_dir(target, include_pilot)
    parcels = gpd.read_parquet(base / "modeling_parcels.parquet")
    parcels = parcels[parcels["split"] != "test"].copy()
    clim = _covariate_table(parcels)
    df = parcels[["COD_PREDIO", "dept", "label"]].merge(clim, on="COD_PREDIO")

    cols = CLIMATE_ONLY
    corr = df[[*cols, "centroid_lat", "centroid_lon"]].corr(method="spearman")
    prior = df["dept"].value_counts(normalize=True).max()
    acc = float(cross_val_score(RandomForestClassifier(300, random_state=0),
                                df[cols], df["dept"], cv=5, scoring="accuracy").mean())
    acc_lat = float(cross_val_score(RandomForestClassifier(300, random_state=0),
                                    df[["centroid_lat"]], df["dept"], cv=5,
                                    scoring="accuracy").mean())
    out = {"n": len(df), "n_dept": int(df["dept"].nunique()),
           "dept_prior": float(prior),
           "dept_acc_from_climate": acc, "dept_acc_from_centroid_lat": acc_lat,
           "spearman_tmean_lat": float(corr.loc["tmean_c", "centroid_lat"]),
           "spearman_precip_lat": float(corr.loc["precip_mm_yr", "centroid_lat"]),
           "spearman_tmean_precip": float(corr.loc["tmean_c", "precip_mm_yr"])}
    print("\nIs climate a location proxy?")
    print(f"  department recovered from (tmean_c, precip_mm_yr) alone: "
          f"{acc:.3f} 5-fold accuracy over {out['n_dept']} departments "
          f"(prior {prior:.3f}); from centroid_lat alone: {acc_lat:.3f}")
    print(f"  Spearman  tmean~lat {out['spearman_tmean_lat']:+.3f}   "
          f"precip~lat {out['spearman_precip_lat']:+.3f}   "
          f"tmean~precip {out['spearman_tmean_precip']:+.3f}")
    print("\n  class means:")
    print(df.groupby("label")[cols].agg(["mean", "std"]).round(1).to_string())
    print("\n  department means:")
    print(df.groupby("dept")[cols].mean().round(1).sort_values("precip_mm_yr").to_string())
    (base / "climate_location_audit.json").write_text(json.dumps(out, indent=2))
    return out


# --- The comparison table ---
RUNS = ROOT / "runs" / "s2_labels"
MODELS = ["lightgbm", "ltae", "rules"]


def _majority_floor(counts: pd.Series) -> float:
    """macro-F1 of "always guess the largest class".

    Class *c* gets recall 1, precision ``p = n_c / N``, F1 ``2p/(1+p)``; others score 0.
    Over the class count, that is the floor — and it moves with the class count, which is
    why §8.2c prints it beside every macro-F1.
    """
    p = float(counts.max() / counts.sum())
    return (2 * p / (1 + p)) / len(counts)


def _cv_pool(run: Path) -> pd.DataFrame | None:
    """Out-of-fold predictions pooled over the CV folds of one run."""
    fs = sorted(run.glob("fold*/preds_val.parquet"))
    if not fs:
        return None
    return pd.concat([pd.read_parquet(f).assign(fold=int(f.parent.name[4:]))
                      for f in fs], ignore_index=True)


def _class_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    from sklearn.metrics import f1_score
    f = f1_score(y_true, y_pred, average=None, labels=range(len(classes)),
                 zero_division=0)
    return dict(zip(classes, (float(x) for x in f)))


def collect(target: str = "t4", include_pilot: bool = True) -> pd.DataFrame:
    """One row per (model class, climate arm): CV, LODO, floor-normalised skill."""
    base = P.ws_dir(target, include_pilot)
    with open(base / "label_map.json") as f:
        label_map = json.load(f)
    classes = [c for c, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    parcels = gpd.read_parquet(base / "modeling_parcels.parquet")
    tv = parcels[parcels["split"] != "test"]
    floor = _majority_floor(tv["label"].value_counts())

    rows = []
    for model in MODELS:
        for clim in ARMS:
            ws = ws_dir(target, clim, include_pilot)
            run = RUNS / ws.name / model
            cvf = run / "cv_metrics.json"
            if not cvf.exists():
                continue
            m = json.loads(cvf.read_text())
            folds = pd.DataFrame(m["folds"])
            pool = _cv_pool(run)
            perclass = {}
            if pool is not None:
                probs = pool[[f"prob_{c}" for c in classes]].to_numpy()
                perclass = _class_f1(pool["y_true"].to_numpy(), probs.argmax(1), classes)

            tag = "" if clim == "none" else f"_clim_{clim}"
            lf = ws / f"lodo_{model}{tag}.csv"
            lodo = pd.read_csv(lf) if lf.exists() else None
            lp = ws / f"lodo_{model}{tag}_preds.parquet"
            lod_perclass = {}
            if lp.exists():
                d = pd.read_parquet(lp)
                lod_perclass = _class_f1(d["y_true"].to_numpy(), d["y_pred"].to_numpy(),
                                         classes)

            row = {
                "model": model, "arm": clim,
                "cv_macro_f1": float(folds.macro_f1.mean()),
                "cv_sd": float(folds.macro_f1.std()),
                "cv_accuracy": float(folds.accuracy.mean()),
                "lodo_macro_f1": float(lodo.macro_f1.mean()) if lodo is not None else np.nan,
                "lodo_sd": float(lodo.macro_f1.std()) if lodo is not None else np.nan,
                "lodo_accuracy": float(lodo.accuracy.mean()) if lodo is not None else np.nan,
                "n_dept": int(len(lodo)) if lodo is not None else 0,
                "cv_perennial_f1": perclass.get("PERENNIAL", np.nan),
                "lodo_perennial_f1": lod_perclass.get("PERENNIAL", np.nan),
            }
            row["cv_skill"] = (row["cv_macro_f1"] - floor) / (1 - floor)
            row["lodo_skill"] = (row["lodo_macro_f1"] - floor) / (1 - floor)
            row["cv_minus_lodo"] = row["cv_macro_f1"] - row["lodo_macro_f1"]
            rows.append(row)

    t = pd.DataFrame(rows)
    t["majority_floor"] = floor
    t["target"] = target
    t["n_trainval"] = len(tv)
    # deltas against this model class's own no-climate arm
    for col in ("cv_macro_f1", "lodo_macro_f1", "cv_accuracy", "lodo_accuracy"):
        base_val = t.set_index(["model", "arm"])[col].unstack()["none"]
        t[f"d_{col}"] = t[col] - t["model"].map(base_val)
    order = {m: i for i, m in enumerate(MODELS)}
    t = t.sort_values(["model", "arm"],
                      key=lambda s: s.map(order) if s.name == "model"
                      else s.map({c: i for i, c in enumerate(ARMS)}))
    return t.reset_index(drop=True)


def fold_paired(target: str = "t4", include_pilot: bool = True) -> pd.DataFrame:
    """Per-fold and per-department deltas against each model's own no-climate arm.

    A mean over 5 folds or 14 departments is 5 or 14 numbers, and at ~140 val parcels per
    fold a 0.02 macro-F1 difference sits inside one fold's spread. This says whether an
    arm's gain is consistent or one fold.
    """
    rows = []
    for model in MODELS:
        base_run = RUNS / ws_dir(target, "none", include_pilot).name / model
        if not (base_run / "cv_metrics.json").exists():
            continue
        b_cv = pd.DataFrame(json.loads((base_run / "cv_metrics.json").read_text())["folds"])
        b_ws = ws_dir(target, "none", include_pilot)
        b_lodo = (pd.read_csv(b_ws / f"lodo_{model}.csv")
                  if (b_ws / f"lodo_{model}.csv").exists() else None)
        for clim in ARMS[1:]:
            ws = ws_dir(target, clim, include_pilot)
            run = RUNS / ws.name / model
            if not (run / "cv_metrics.json").exists():
                continue
            cv = pd.DataFrame(json.loads((run / "cv_metrics.json").read_text())["folds"])
            d = cv.macro_f1.values - b_cv.macro_f1.values
            lf = ws / f"lodo_{model}_clim_{clim}.csv"
            dl = None
            if b_lodo is not None and lf.exists():
                j = b_lodo[["dept", "macro_f1"]].merge(
                    pd.read_csv(lf)[["dept", "macro_f1"]], on="dept",
                    suffixes=("_none", "_clim"))
                dl = (j.macro_f1_clim - j.macro_f1_none).values
            rows.append({
                "model": model, "arm": clim,
                "cv_folds_improved": f"{int((d > 0).sum())}/{len(d)}",
                "cv_delta_min": float(d.min()), "cv_delta_max": float(d.max()),
                "lodo_depts_improved": f"{int((dl > 0).sum())}/{len(dl)}" if dl is not None else "",
                "lodo_delta_mean": float(dl.mean()) if dl is not None else np.nan,
                "lodo_delta_min": float(dl.min()) if dl is not None else np.nan,
                "lodo_delta_max": float(dl.max()) if dl is not None else np.nan,
                "lodo_p_wilcoxon": _wilcoxon_p(dl),
            })
    return pd.DataFrame(rows)


def _wilcoxon_p(d: np.ndarray | None) -> float:
    """Two-sided paired Wilcoxon over the per-department deltas.

    Paired: same 14 departments both sides, their difficulty varies far more than the arms
    (LODO SD ~0.12 vs a ~0.03 effect), so an unpaired test finds nothing. ⚠️ Still 14
    numbers and the arms share training data — read it as "is the sign consistent".
    """
    if d is None or len(d) < 6:
        return float("nan")
    from scipy.stats import wilcoxon
    try:
        return float(wilcoxon(d).pvalue)
    except ValueError:      # all-zero differences
        return float("nan")


def climate_importance(target: str = "t4", include_pilot: bool = True) -> pd.DataFrame:
    """Where the climate columns land in LightGBM's gain ranking, per arm.

    ⚠️ Gain is not contribution — measured twice (LESSONS.md: dropping 29.8 % of gain cost
    0.007 macro-F1). Read as "did the booster look at the column"; the main table's
    ``d_*_macro_f1`` columns are what it was worth.
    """
    from crop_classifier.models.trees import LightGBMModel
    rows = []
    for clim in ARMS[1:]:
        run = RUNS / ws_dir(target, clim, include_pilot).name / "lightgbm"
        if not (run / "model.bin").exists():
            continue
        imp = LightGBMModel.load(run / "model.bin").feature_importance()
        imp["rank"] = range(1, len(imp) + 1)
        tot = imp.gain.sum()
        for c in CLIMATE_SETS[clim]:
            m = imp[imp.feature == c]
            if not len(m):
                continue
            r = m.iloc[0]
            rows.append({"arm": clim, "feature": c, "gain_rank": int(r["rank"]),
                         "of_n_features": len(imp),
                         "gain_share": float(r["gain"] / tot)})
    return pd.DataFrame(rows)


def per_class(target: str = "t4", include_pilot: bool = True) -> pd.DataFrame:
    """Per-class out-of-fold F1 for every arm — which class the covariate actually moves.

    macro-F1 hides which class changed; the question turns only on ``PERENNIAL``, and §8.2c
    established the binding constraint is the ``PERENNIAL`` / ``WOODY_NON_CROP`` boundary.
    """
    base = P.ws_dir(target, include_pilot)
    with open(base / "label_map.json") as f:
        label_map = json.load(f)
    classes = [c for c, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    rows = []
    for model in MODELS:
        for clim in ARMS:
            run = RUNS / ws_dir(target, clim, include_pilot).name / model
            pool = _cv_pool(run)
            if pool is None:
                continue
            probs = pool[[f"prob_{c}" for c in classes]].to_numpy()
            rows.append({"model": model, "arm": clim,
                         **_class_f1(pool["y_true"].to_numpy(), probs.argmax(1), classes)})
    return pd.DataFrame(rows)


def lodo_per_class(target: str = "t4", include_pilot: bool = True) -> pd.DataFrame:
    """Per-class F1 on the pooled LODO predictions — the out-of-department counterpart of
    ``per_class`` (which pools only CV folds).

    Arms differ in *which* class they move: on ``t3w`` ``temp`` is worth +0.058 on
    ``PERENNIAL``, +0.011 on ``ANNUAL``. The verdict turns on ``PERENNIAL``, so show it
    separately on the axis the verdict is read on.
    """
    base = P.ws_dir(target, include_pilot)
    with open(base / "label_map.json") as f:
        label_map = json.load(f)
    classes = [c for c, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    rows = []
    for model in MODELS:
        for clim in ARMS:
            ws = ws_dir(target, clim, include_pilot)
            tag = "" if clim == "none" else f"_clim_{clim}"
            f = ws / f"lodo_{model}{tag}_preds.parquet"
            if not f.exists():
                continue
            d = pd.read_parquet(f)
            rows.append({"model": model, "arm": clim,
                         **_class_f1(d["y_true"].to_numpy(), d["y_pred"].to_numpy(),
                                     classes)})
    t = pd.DataFrame(rows)
    for c in classes:
        if c in t:
            b = t.set_index(["model", "arm"])[c].unstack()["none"]
            t[f"d_{c}"] = t[c] - t["model"].map(b)
    return t


def control_paired(target: str = "t4", include_pilot: bool = True) -> pd.DataFrame:
    """Each climate arm minus the ``latlon`` control, paired over the same departments.

    ⚠️ This, not ``d_lodo_macro_f1``, is what the §8.8 verdict rests on: "does climate help
    *more than a same-shaped surrogate carrying only location*". Both arms score on the
    same 14 held-out departments, so the difference is paired — test it as one.
    """
    rows = []
    for model in MODELS:
        ctl = _lodo_series(target, model, "latlon", include_pilot)
        if ctl is None:
            continue
        for clim in ARMS[1:-1]:          # temp, rain, both — latlon is the comparator
            a = _lodo_series(target, model, clim, include_pilot)
            if a is None:
                continue
            d = (a - ctl).dropna()
            rows.append({"target": target, "model": model, "arm": clim,
                         "lodo_arm": float(a.mean()), "lodo_latlon": float(ctl.mean()),
                         "arm_minus_control": float(d.mean()),
                         "depts_beating_control": f"{int((d > 0).sum())}/{len(d)}",
                         "paired_p": _wilcoxon_p(d.to_numpy())})
    return pd.DataFrame(rows)


def _lodo_series(target: str, model: str, arm: str,
                 include_pilot: bool = True) -> pd.Series | None:
    """Per-department LODO macro-F1 for one arm, indexed by department."""
    ws = ws_dir(target, arm, include_pilot)
    tag = "" if arm == "none" else f"_clim_{arm}"
    f = ws / f"lodo_{model}{tag}.csv"
    if not f.exists():
        return None
    return pd.read_csv(f)[["dept", "macro_f1"]].set_index("dept")["macro_f1"]


def woody_boundary(target: str = "t4", include_pilot: bool = True) -> pd.DataFrame:
    """Does an arm's gain sit on the ``PERENNIAL`` / ``WOODY_NON_CROP`` boundary?

    ⚠️ The boundary is rainfall-aligned in the labels: above 700 mm/yr only 1 of 87 usable
    declared-perennial parcels was called ``PERENNIAL``, 65.5 % ``WOODY_NON_CROP`` (28.8 % /
    21.2 % below); the crops there are coffee and cacao. So a rainfall column could buy
    macro-F1 by learning the annotator's cut (§8.8 caveat).

    Restricted to parcels whose true label is one of the two, how often does the model pick
    the right one? ``t3w`` deletes the boundary, so this only means anything where both
    classes exist. §8.8b: ``temp`` improves it (+0.054), ``rain`` does not (−0.018).
    """
    base = P.ws_dir(target, include_pilot)
    with open(base / "label_map.json") as f:
        label_map = json.load(f)
    classes = [c for c, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    if not {"PERENNIAL", "WOODY_NON_CROP"} <= set(classes):
        return pd.DataFrame()
    iP, iW = classes.index("PERENNIAL"), classes.index("WOODY_NON_CROP")

    rows = []
    for model in MODELS:
        for clim in ARMS:
            ws = ws_dir(target, clim, include_pilot)
            tag = "" if clim == "none" else f"_clim_{clim}"
            f = ws / f"lodo_{model}{tag}_preds.parquet"
            if not f.exists():
                continue
            d = pd.read_parquet(f)
            y, pr = d["y_true"].to_numpy(), d["y_pred"].to_numpy()
            m = np.isin(y, [iP, iW])
            f1 = _class_f1(y, pr, classes)
            others = [f1[c] for c in classes if c not in ("PERENNIAL", "WOODY_NON_CROP")]
            rows.append({
                "model": model, "arm": clim, "n_P_or_W": int(m.sum()),
                "pw_subproblem_acc": float((y[m] == pr[m]).mean()),
                "p_called_woody": float((pr[y == iP] == iW).mean()),
                "woody_called_p": float((pr[y == iW] == iP).mean()),
                "perennial_f1": f1["PERENNIAL"], "woody_f1": f1["WOODY_NON_CROP"],
                "other_classes_f1": float(np.mean(others)) if others else np.nan})
    t = pd.DataFrame(rows)
    for c in ("pw_subproblem_acc", "perennial_f1", "woody_f1", "other_classes_f1"):
        b = t.set_index(["model", "arm"])[c].unstack()["none"]
        t[f"d_{c}"] = t[c] - t["model"].map(b)
    return t


def bracket(targets: tuple[str, ...] = ("t4", "t3w"),
            include_pilot: bool = True) -> pd.DataFrame:
    """The same arms at both ends of the ``t4`` / ``t3w`` codebook bracket, side by side.

    ⚠️ ``t4`` and ``t3w`` are not two candidate models — they differ only in which side
    ``WOODY_NON_CROP`` sits on, worth +0.190 on ``PERENNIAL`` F1 (§8.2c), larger than any
    modelling effect. A feature verdict is worth only its agreement across the two: §8.8b
    caught ``--climate both`` going 12/14 p=0.004 -> 7/14 p=0.345 one codebook decision away.

    ⚠️ The majority-class floor moves with the target (0.171 / 0.228), so read ``*_skill``
    and ``*_perennial_f1``, not raw macro-F1, across targets.
    """
    got = {t: collect(t, include_pilot) for t in targets}
    got = {t: v for t, v in got.items() if len(v)}
    if len(got) < 2:
        return pd.DataFrame()
    cols = ["cv_macro_f1", "lodo_macro_f1", "lodo_sd", "n_dept", "cv_skill", "lodo_skill",
            "cv_perennial_f1", "lodo_perennial_f1", "d_cv_macro_f1", "d_lodo_macro_f1",
            "majority_floor"]
    out = None
    for t, v in got.items():
        v = v[["model", "arm", *cols]].add_suffix(f"_{t}").rename(
            columns={f"model_{t}": "model", f"arm_{t}": "arm"})
        out = v if out is None else out.merge(v, on=["model", "arm"], how="outer")
    order = {m: i for i, m in enumerate(MODELS)}
    arms = {c: i for i, c in enumerate(ARMS)}
    return out.sort_values(["model", "arm"],
                           key=lambda s: s.map(order) if s.name == "model" else s.map(arms)
                           ).reset_index(drop=True)


def report(target: str = "t4", include_pilot: bool = True) -> pd.DataFrame:
    t = collect(target, include_pilot)
    if not len(t):
        raise SystemExit("no fitted arms found under runs/s2_labels — fit them first")
    floor = float(t["majority_floor"].iloc[0])
    print(f"\n=== S2 endpoint labels, target {target}"
          f"{'+pilot' if include_pilot else ''}: climate covariate arms ===")
    print(f"{int(t['n_trainval'].iloc[0])} trainval parcels, "
          f"majority-class macro-F1 floor {floor:.3f}\n")
    show = ["model", "arm", "cv_macro_f1", "cv_sd", "cv_accuracy",
            "lodo_macro_f1", "lodo_sd", "lodo_accuracy", "n_dept",
            "cv_minus_lodo", "cv_skill", "lodo_skill",
            "cv_perennial_f1", "lodo_perennial_f1",
            "d_cv_macro_f1", "d_lodo_macro_f1"]
    print(t[show].round(3).to_string(index=False))

    if "latlon" in set(t["arm"]):
        print("\n⚠️ `latlon` is the CONTROL arm, not a proposal: centroid lat/lon in place "
              "of the two climate\n   columns. Read `both` against it, not against `none` "
              "alone — a climate gain that `latlon`\n   matches is geography, which is "
              "RESULTS.md §4.2.")

    # every table printed here is also written — §8.8 quotes numbers from all four
    written = [P.labels_dir() / f"climate_arms_{target}.csv"]
    t.round(4).to_csv(written[0], index=False)

    fp = fold_paired(target, include_pilot)
    if len(fp):
        print("\nper-fold / per-department consistency of each delta:")
        print(fp.round(3).to_string(index=False))
        written.append(P.labels_dir() / f"climate_arms_{target}_consistency.csv")
        fp.round(4).to_csv(written[-1], index=False)

    pc = per_class(target, include_pilot)
    if len(pc):
        print("\nout-of-fold per-class F1 (CV pool):")
        print(pc.round(3).to_string(index=False))
        written.append(P.labels_dir() / f"climate_arms_{target}_per_class.csv")
        pc.round(4).to_csv(written[-1], index=False)

    lpc = lodo_per_class(target, include_pilot)
    if len(lpc):
        print("\nper-class F1 on the pooled LODO predictions "
              "(the axis the verdict is read on):")
        print(lpc.round(3).to_string(index=False))
        written.append(P.labels_dir() / f"climate_arms_{target}_lodo_per_class.csv")
        lpc.round(4).to_csv(written[-1], index=False)

    cp = control_paired(target, include_pilot)
    if len(cp):
        print("\n⭐ each climate arm MINUS the `latlon` control, paired over the same "
              "departments\n   (this, not the delta against `none`, is what the verdict "
              "rests on):")
        print(cp.round(3).to_string(index=False))
        written.append(P.labels_dir() / f"climate_arms_{target}_vs_control.csv")
        cp.round(4).to_csv(written[-1], index=False)

    wb = woody_boundary(target, include_pilot)
    if len(wb):
        print("\nthe PERENNIAL vs WOODY_NON_CROP sub-problem, LODO — is the gain the "
              "annotator's\nrainfall-aligned boundary? (see the docstring; `t3w` deletes "
              "this boundary):")
        print(wb.round(3).to_string(index=False))
        written.append(P.labels_dir() / f"climate_arms_{target}_woody_boundary.csv")
        wb.round(4).to_csv(written[-1], index=False)

    try:
        ci = climate_importance(target, include_pilot)
        if len(ci):
            print("\nLightGBM gain rank of the climate columns "
                  "(gain is not contribution — see the docstring):")
            print(ci.round(4).to_string(index=False))
            written.append(P.labels_dir() / f"climate_arms_{target}_gain.csv")
            ci.round(4).to_csv(written[-1], index=False)
    except Exception as e:
        print(f"(climate importance skipped: {e})")

    br = bracket(include_pilot=include_pilot)
    if len(br):
        print("\n⭐ the t4 / t3w codebook bracket, side by side. The majority-class floor "
              "MOVES between\n   targets (0.171 / 0.228), so read `*_skill` and "
              "`*_perennial_f1`, not the raw macro-F1s:")
        show = ["model", "arm"] + [c for c in br.columns
                                   if c.startswith(("lodo_skill", "lodo_perennial_f1",
                                                    "d_lodo_macro_f1"))]
        print(br[show].round(3).to_string(index=False))
        written.append(P.labels_dir() / "climate_arms_bracket_t4_t3w.csv")
        br.round(4).to_csv(written[-1], index=False)

    print("\nwrote:")
    for f in written:
        print(f"  {f}")
    return t

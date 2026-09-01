"""Turn returned S2 labels into a trainable workspace, then drive the model arms.

``ingest`` writes ``labelled_parcels.parquet``, which stores the annotator's label
**verbatim** over six values and deliberately takes no modelling decision. This module
takes those decisions explicitly, one workspace per target, so that every arm is a file on
disk that can be re-read rather than a flag threaded through a training call.

Each workspace is a ``CC_PROC`` directory holding ``modeling_parcels.parquet`` +
``label_map.json`` in exactly the schema ``data.load_parcels`` expects, so ``train.py``,
``evaluate.py`` and the model registry run **unmodified** against it. The feature store is
pointed at ``features_s2/`` via ``CC_FEAT``; the S2 summary block is symlinked to the name
``data.FN_LGBM`` expects rather than copied.

⚠️ **Never import this module in the same process as a torch model.** It reaches LightGBM
through ``train``/``baseline``; macOS co-loading of the two libomp copies segfaults. The
CLI runs one arm per process for exactly that reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from crop_classifier.config_loader import load_yaml_config
from crop_classifier.paths import ROOT, labels_dir

# ---------------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------------
# The label spaces live in `config/labels/*.yaml`, one file each, with the reasoning for
# the choice written next to it. They used to be a literal here, which made adding one —
# the thing most likely to be needed when new labels arrive — a Python edit inside a module
# that must never be imported alongside torch.
#
# `TARGETS` and `RULES_INCOMPATIBLE` are resolved on every access rather than bound at
# import, so a label set added to that directory is usable immediately, with no reload and
# no code change. See `crop_classifier/label_sets.py` and the README beside the configs.


def __getattr__(name: str):                     # PEP 562
    if name == "TARGETS":
        from crop_classifier.label_sets import targets
        return targets()
    if name == "RULES_INCOMPATIBLE":
        from crop_classifier.label_sets import rules_incompatible
        return rules_incompatible()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# `OTHER` is "farmable land not currently a crop" — the codebook narrowed it to exactly
# that — which is what `PASTURE_FALLOW` means in the Landsat label space.
TO_LANDSAT = {"PERENNIAL": "PERENNIAL", "ANNUAL": "ANNUAL", "OTHER": "PASTURE_FALLOW"}

FEAT_S2 = ROOT / "data" / "processed" / "all_peru" / "features_s2"
SPLIT_CFG = ROOT / "src" / "crop_classifier" / "config" / "split_s2labels.yaml"
N_FOLDS = 5


def ws_dir(target: str, include_pilot: bool = False) -> Path:
    return labels_dir() / f"ws_{target}{'_pilot' if include_pilot else ''}"


# ---------------------------------------------------------------------------------
# Workspace construction
# ---------------------------------------------------------------------------------
def _xy(df: gpd.GeoDataFrame, crs: int) -> np.ndarray:
    p = df.geometry.representative_point().to_crs(crs)
    return np.c_[p.x.values, p.y.values]


def _reassign_folds(df: pd.DataFrame, seed: int) -> pd.Series:
    """Fold ids for a trainval set that now includes the pilot.

    The frozen assignment is **preserved** for every parcel that already had one; a pilot
    parcel in a region that is already spoken for inherits that region's fold, and only the
    genuinely new regions are drawn. Reshuffling the whole thing would have made the two
    arms differ in fold membership as well as in sample size, and then nothing could be
    attributed to the extra data.
    """
    fold = df["fold"].astype(int).copy()
    by_region = (df.loc[fold >= 0].groupby("region_id")["fold"].first())
    inherit = df["region_id"].map(by_region)
    m = (fold < 0) & inherit.notna()
    fold[m] = inherit[m].astype(int)

    new = df.index[fold < 0]
    if len(new):
        # round-robin over the new regions, smallest fold first, so the added parcels land
        # where they most even out the fold sizes
        rng = np.random.default_rng(seed)
        regions = list(df.loc[new, "region_id"].unique())
        rng.shuffle(regions)
        sizes = fold[fold >= 0].value_counts().reindex(range(N_FOLDS)).fillna(0).to_dict()
        for r in regions:
            k = min(sizes, key=lambda kk: sizes[kk])
            rows = new[(df.loc[new, "region_id"] == r).values]
            fold[rows] = k
            sizes[k] += len(rows)
    return fold


def _buffer_columns(df: gpd.GeoDataFrame, cfg: dict) -> pd.DataFrame:
    """Recompute the spatial dead-zones over *this* parcel set."""
    from scipy.spatial import cKDTree

    xy = _xy(df, cfg.get("metric_crs", 32718))
    buf = float(cfg["buffer_m"])
    out = pd.DataFrame(index=df.index)

    def excl(held: np.ndarray, cand: np.ndarray) -> np.ndarray:
        v = np.zeros(len(df), dtype=bool)
        if held.any() and cand.any():
            d, _ = cKDTree(xy[held]).query(xy[cand], k=1)
            v[np.where(cand)[0][d <= buf]] = True
        return v

    is_test = (df["split"] == "test").values
    out["buffer_excl_test"] = excl(is_test, ~is_test)
    for k in range(N_FOLDS):
        held = (df["fold"] == k).values
        out[f"buffer_excl_fold{k}"] = excl(held, ~is_test & ~held)
    return out


def build_workspace(target: str, include_pilot: bool = False,
                    verbose: bool = True) -> Path:
    """``labelled_parcels.parquet`` -> a ``CC_PROC`` workspace for one label target."""
    from crop_classifier.label_sets import load as load_label_set
    label_set = load_label_set(target)          # raises, naming the config dir, if unknown
    cfg = load_yaml_config(SPLIT_CFG)

    lab = gpd.read_parquet(labels_dir() / "labelled_parcels.parquet")
    lab = lab[lab["usable"]].copy()
    sample = gpd.read_parquet(labels_dir() / "label_sample.parquet")
    extra = ["COD_PREDIO", "year", "n_valid_obs", "max_gap", "n_pixels_est",
             "buffer_excl_test", *[f"buffer_excl_fold{k}" for k in range(N_FOLDS)]]
    df = lab.merge(sample[extra], on="COD_PREDIO", how="left")

    # ---- label mapping ----
    df["label"] = label_set.apply(df["label"])
    n_before = len(df)
    df = df[df["label"].notna()].copy()

    classes = sorted(df["label"].unique())
    label_map = {c: i for i, c in enumerate(classes)}
    df["label_id"] = df["label"].map(label_map).astype(int)
    # `quality_ok` gates `load_parcels`. It is a *Landsat* extraction-quality flag and is
    # NA for every parcel in this campaign, which would silently drop the entire sample;
    # the S2 equivalents (`no_s2_observations`, `sub_pixel_parcel`) were already applied by
    # `ingest`, so what survives to here is by construction quality-passed.
    df["quality_ok"] = True

    if include_pilot:
        df.loc[df["split"] == "pilot", "split"] = "trainval"
        df["fold"] = _reassign_folds(df, int(cfg["seed"]))
        buf = _buffer_columns(df, cfg)
        df[buf.columns] = buf

    out = ws_dir(target, include_pilot)
    out.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(df, geometry="geometry", crs=lab.crs).to_parquet(
        out / "modeling_parcels.parquet", index=False)
    with open(out / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # the feature store under the file names data.py expects
    fdir = out / "features"
    fdir.mkdir(exist_ok=True)
    for src, dst in [("s2_features_lightgbm.parquet", "features_lightgbm.parquet"),
                     ("tensor_perdate.npz", "tensor_perdate.npz")]:
        link, tgt = fdir / dst, FEAT_S2 / src
        if link.is_symlink() or link.exists():
            link.unlink()
        if tgt.exists():
            link.symlink_to(tgt)

    if verbose:
        print(f"[{target}{'+pilot' if include_pilot else ''}] {len(df)} parcels "
              f"({n_before - len(df)} dropped by the mapping), classes {classes}")
        print(pd.crosstab(df["label"], df["split"]).to_string())
        tv = df[df["split"] == "trainval"]
        print("per-fold trainval:")
        print(pd.crosstab(tv["fold"], tv["label"]).to_string())
        sterile = [int((tv[f"buffer_excl_fold{k}"] & (tv["fold"] != k)).sum())
                   for k in range(N_FOLDS)]
        print(f"3 km dead-zone removes {sterile} of "
              f"{len(tv)} trainval rows from each fold's train side")
    return out


# ---------------------------------------------------------------------------------
# The Landsat baseline, transferred
# ---------------------------------------------------------------------------------
def landsat_baseline(run: Path, target: str = "t3",
                     include_pilot: bool = False) -> pd.DataFrame:
    """Score the saved Landsat primary on the labelled parcels' **S2** features.

    ⚠️ This is a **lower bound on the Landsat model, not a measurement of it.** The booster
    was fitted on Landsat 5/7 surface reflectance and is being fed Sentinel-2 surface
    reflectance under the same column names. RESULTS.md §9.3.2 established on this project's
    own data that the sensor difference is cover-type dependent and that no global linear
    map removes it, so the transfer costs an unknown, non-uniform amount of skill. It
    answers "what does the model the project already has say about these parcels", which is
    worth knowing, and it cannot adjudicate G4.

    The alternative — the Landsat panel predictions computed on genuine Landsat features —
    covers 22 of the labelled parcels and is reported beside this as a direction check.
    """
    from crop_classifier.models.trees import LightGBMModel

    ws = ws_dir(target, include_pilot)
    parcels = gpd.read_parquet(ws / "modeling_parcels.parquet")
    feats = pd.read_parquet(FEAT_S2 / "s2_features_lightgbm.parquet")
    sample = gpd.read_parquet(labels_dir() / "label_sample.parquet")
    feats = feats.merge(sample[["COD_PREDIO", "n_pixels_est"]], on="COD_PREDIO",
                        how="left")

    model = LightGBMModel.load(run / "model.bin")
    with open(run / "label_map.json") as f:
        ls_classes = [c for c, _ in sorted(json.load(f).items(), key=lambda kv: kv[1])]

    missing = [c for c in model.feature_names if c not in feats.columns]
    if missing:
        raise KeyError(f"S2 store is missing {len(missing)} baseline features: {missing}")
    sub = parcels[["COD_PREDIO", "label", "split", "dept"]].merge(
        feats, on="COD_PREDIO", how="inner")
    prob = model.booster.predict(sub[model.feature_names])

    out = sub[["COD_PREDIO", "label", "split", "dept"]].copy()
    for i, c in enumerate(ls_classes):
        out[f"prob_{c}"] = prob[:, i]
    out["pred_landsat"] = [ls_classes[i] for i in prob.argmax(1)]
    out["y_landsat"] = out["label"].map(TO_LANDSAT)
    return out


def report_baseline(out: pd.DataFrame, ws: Path) -> pd.DataFrame:
    """Print + persist the transferred Landsat model's read on the labelled parcels."""
    from crop_classifier.evaluate import per_class_table

    scored = out[out["y_landsat"].notna()]
    classes = sorted(set(scored["y_landsat"]) | set(scored["pred_landsat"]))
    ci = {c: i for i, c in enumerate(classes)}
    yt = scored["y_landsat"].map(ci).values
    yp = scored["pred_landsat"].map(ci).values
    from sklearn.metrics import f1_score
    macro = f1_score(yt, yp, average="macro")
    print(f"\nLandsat primary transferred to S2 features, {len(scored)} parcels "
          f"with a Landsat-space counterpart:")
    print(f"  macro-F1 {macro:.3f}   accuracy {(yt == yp).mean():.3f}")
    print(pd.crosstab(scored["y_landsat"], scored["pred_landsat"],
                      rownames=["photo-interpreted"], colnames=["Landsat model"]
                      ).to_string())
    print(per_class_table(yt, yp, classes).round(3).to_string(index=False))
    # what the model does with the classes it has no word for
    orphan = out[out["y_landsat"].isna()]
    if len(orphan):
        print(f"\nthe {len(orphan)} parcels with no Landsat-space counterpart "
              f"({', '.join(sorted(orphan['label'].unique()))}) are called:")
        print(pd.crosstab(orphan["label"], orphan["pred_landsat"]).to_string())
    out.to_parquet(ws / "landsat_baseline_preds.parquet", index=False)
    with open(ws / "landsat_baseline.json", "w") as f:
        json.dump({"n": int(len(scored)), "macro_f1": float(macro),
                   "accuracy": float((yt == yp).mean()),
                   "caveat": "cross-sensor transfer: Landsat-fitted booster on S2 "
                             "reflectance. A LOWER BOUND on the Landsat model, not a "
                             "measurement of it. Cannot adjudicate G4."}, f, indent=2)
    return out


def panel_check() -> pd.DataFrame | None:
    """The 22 labelled parcels that also carry genuine Landsat panel predictions.

    Too few to be a result. It exists to say whether the cross-sensor transfer above points
    the same way as the same model on its own sensor, which is the only thing 22 parcels can
    settle.
    """
    proc = ROOT / "data" / "processed" / "all_peru"
    f = proc / "panel_predictions_nolat_aug_yleak10.parquet"
    if not f.exists():
        return None
    lab = gpd.read_parquet(labels_dir() / "labelled_parcels.parquet")
    lab = lab[lab["usable"]].copy()
    lab["COD_PREDIO"] = lab["COD_PREDIO"].astype(str)
    pan = pd.read_parquet(f)
    pan["COD_PREDIO"] = pan["COD_PREDIO"].astype(str)
    # `pred_label` is null wherever the panel abstained; those parcel-years carry no
    # prediction and must not vote
    pan = pan[(pan["year"] >= 2019) & pan["pred_label"].notna()]
    pan["pred_label"] = pan["pred_label"].astype(str)
    # one row per parcel: the modal prediction over 2019-23, so a single noisy year does
    # not decide it
    modal = (pan.groupby("COD_PREDIO")["pred_label"]
             .agg(lambda v: v.mode().iat[0]).rename("pred_panel"))
    j = lab[["COD_PREDIO", "label"]].join(modal, on="COD_PREDIO", how="inner")
    j = j[j["pred_panel"].notna()]
    j["y_landsat"] = j["label"].map(TO_LANDSAT)
    return j


def report(target: str | None = None, include_pilot: bool | None = None
           ) -> pd.DataFrame:
    """Every fitted arm, its CV spread and its LODO mean, in one table.

    Reports the **fold spread**, not just the mean. At 35-71 parcels per fold a difference
    of 0.05 macro-F1 sits inside one fold's variation, and a table of bare means invites
    exactly the selection error CLAUDE.md opens with.
    """
    rdir = ROOT / "runs" / "s2_labels"
    rows = []
    for f in sorted(rdir.glob("*/*/cv_metrics.json")):
        ws = f.parent.parent.name
        tgt = ws.replace("ws_", "").replace("_pilot", "")
        pil = ws.endswith("_pilot")
        if target is not None and tgt != target:
            continue
        if include_pilot is not None and pil != include_pilot:
            continue
        m = json.loads(f.read_text())
        f1 = [x["macro_f1"] for x in m["folds"]]
        lodo_f = labels_dir() / ws / f"lodo_{f.parent.name}.csv"
        lodo = pd.read_csv(lodo_f)["macro_f1"] if lodo_f.exists() else None
        rows.append({
            "target": tgt, "pilot": pil, "model": f.parent.name,
            "cv_macro_f1": round(m["cv_macro_f1_mean"], 3),
            "cv_sd": round(m["cv_macro_f1_std"], 3),
            "cv_min_fold": round(min(f1), 3), "cv_max_fold": round(max(f1), 3),
            "lodo_macro_f1": round(lodo.mean(), 3) if lodo is not None else np.nan,
            "lodo_sd": round(lodo.std(), 3) if lodo is not None else np.nan,
            "n_train": int(np.mean([x["n_train"] for x in m["folds"]])),
            "n_val": int(np.mean([x["n_val"] for x in m["folds"]])),
        })
    t = pd.DataFrame(rows).sort_values(["target", "pilot", "model"])
    if len(t):
        print(t.to_string(index=False))
        t.to_csv(labels_dir() / "model_comparison.csv", index=False)
        print(f"\nwrote {labels_dir() / 'model_comparison.csv'}")
    return t


# ---------------------------------------------------------------------------------
# Leave-one-department-out
# ---------------------------------------------------------------------------------
def dept_transfer(target: str = "t4", model_name: str = "lightgbm",
                  model_kw: dict | None = None, min_labels: int = 35,
                  include_pilot: bool = True, ws: Path | None = None,
                  tag: str = "") -> pd.DataFrame:
    """Hold out a whole **department**, train on the rest, score the held-out one.

    This is the only out-of-distribution axis this campaign has. Cross-validation here
    holds out 5 km regions *inside departments the model has already seen*, and every
    parcel is photo-interpreted at a single endpoint epoch, so there is no year axis to
    hold out either (LOYO/LODYO do not apply to this store). CLAUDE.md's headline result
    — ``centroid_lat`` gaining +0.047 on CV and losing 0.060 on LODO — is a result about
    exactly this gap, so a CV number here is reported *with* its LODO number or not at all.

    The **locked test split is excluded outright** from both sides: this is a
    trainval(+pilot) procedure and it must not spend the held-out set. The pilot is folded
    in by default because it is a separate pool from the test set, so using it costs
    nothing that is being kept.

    Writes ``lodo_<model>.csv`` beside the workspace, one row per department.
    """
    import os

    # `ws` overrides the workspace so a feature-set variant (labelling.climate_arms) can
    # reuse this procedure unchanged; `tag` names its output file.
    ws = Path(ws) if ws is not None else ws_dir(target, include_pilot)
    os.environ["CC_PROC"] = str(ws)
    os.environ["CC_FEAT"] = str(ws / "features")

    from crop_classifier.data import class_weights, load_parcels, make_dataset
    from crop_classifier.evaluate import compute_metrics
    from crop_classifier.models import get_model

    parcels = load_parcels(require_quality=True)
    parcels = parcels[parcels["split"] != "test"].copy()
    with open(ws / "label_map.json") as f:
        label_map = json.load(f)
    classes = [c for c, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    counts = parcels.groupby("dept").size()
    depts = sorted(counts[counts >= min_labels].index)
    print(f"LODO {model_name} on {target}{'+pilot' if include_pilot else ''}: "
          f"{len(depts)} of {len(counts)} departments have >= {min_labels} labels "
          f"({len(parcels)} parcels, locked test excluded)")

    rows, preds = [], []
    for dpt in depts:
        held = parcels["dept"] == dpt
        tr_idx, te_idx = parcels.index[~held], parcels.index[held]
        model = get_model(model_name, **(model_kw or {}))
        model.set_n_classes(len(classes))
        if hasattr(model, "set_class_names"):
            model.set_class_names(classes)
        train_ds = make_dataset(model.input_kind, parcels, tr_idx)
        norm = getattr(train_ds, "normalizer", None)
        test_ds = make_dataset(model.input_kind, parcels, te_idx, normalizer=norm)
        # early stopping needs a validation set that is not the held-out department;
        # a random 15 % of the training departments serves, so no held-out row is seen
        rng = np.random.default_rng(0)
        va = rng.random(len(tr_idx)) < 0.15
        val_ds = make_dataset(model.input_kind, parcels, tr_idx[va], normalizer=norm)
        fit_dir = ws / f"lodo_{model_name}{tag}_{dpt}"
        fit_dir.mkdir(parents=True, exist_ok=True)
        model.fit(train_ds, val_ds, class_weights(train_ds.y, len(classes)), fit_dir)
        prob = model.predict_proba(test_ds)
        m = compute_metrics(test_ds.y, prob)
        m["dept"], m["n_train"] = dpt, len(train_ds.y)
        rows.append(m)
        p = pd.DataFrame({"COD_PREDIO": test_ds.cod_predio, "dept": dpt,
                          "y_true": test_ds.y, "y_pred": prob.argmax(1)})
        preds.append(p)
        print(f"  {dpt:<14} macro_f1={m['macro_f1']:.3f}  acc={m['accuracy']:.3f}  "
              f"(n={m['n']}, train {m['n_train']})")

    out = pd.DataFrame(rows)[["dept", "n", "n_train", "macro_f1", "accuracy",
                              "balanced_accuracy", "weighted_f1", "cohen_kappa"]]
    stem = f"lodo_{model_name}{tag}"
    out.to_csv(ws / f"{stem}.csv", index=False)
    pd.concat(preds, ignore_index=True).to_parquet(
        ws / f"{stem}_preds.parquet", index=False)
    print(f"\nLODO mean macro-F1 {out.macro_f1.mean():.3f} "
          f"± {out.macro_f1.std():.3f} over {len(out)} departments")
    print(f"wrote {ws / f'{stem}.csv'}")
    return out

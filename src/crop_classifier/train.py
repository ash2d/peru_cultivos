"""Config-driven training: spatial CV -> final refit -> (optional) locked-test eval.

Plan.md §9/§13. For the chosen model: run k-fold spatially-blocked CV (buffer dead-zones
applied), report macro-F1 mean ± std, refit on all trainval minus the test buffer, save
the artifact + resolved config into ``runs/<model>_<timestamp>/``. ``--eval-test``
evaluates the final model on the locked test set — do this ONCE, at the end.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from crop_classifier import models  # noqa: F401  (registers the three models)
from crop_classifier.data import (
    class_weights,
    final_split,
    fold_split,
    load_parcels,
    make_dataset,
    parse_year_spec,
    resolve_drop_features,
    restrict_years,
)
from crop_classifier.evaluate import compute_metrics
from crop_classifier.models.base import get_model
from crop_classifier.paths import proc, runs


def _set_class_names(model, classes: list[str]) -> None:
    """Optional hook: models that reason about class *semantics* (the rule model) need the
    names, not just the count. Statistical models ignore it."""
    if hasattr(model, "set_class_names"):
        model.set_class_names(classes)


def _save_preds(ds, parcels, y_prob, classes, path: Path) -> None:
    """Predictions keyed on the dataset's own COD_PREDIO order (store may drop parcels)."""
    out = pd.DataFrame({"COD_PREDIO": ds.cod_predio, "y_true": ds.y})
    for i, c in enumerate(classes):
        out[f"prob_{c}"] = y_prob[:, i]
    attrs = parcels[["COD_PREDIO", "label", "year", "area_ha",
                     "n_valid_obs", "max_gap", "quality_ok"]]
    out = out.merge(attrs, on="COD_PREDIO", how="left")
    out.to_parquet(path, index=False)


def train(model_name: str = "lightgbm", n_folds: int | None = None,
          eval_test: bool = False, run_name: str | None = None,
          model_kw: dict | None = None,
          drop_features: str | list[str] | None = None,
          train_years: str | list[int] | None = None,
          augment_feat: Path | str | None = None) -> Path:
    """``drop_features``: columns (or ``DROP_SETS`` aliases) to withhold from the flat
    model — see ``data.META_FEATURES`` and RESULTS.md §4.6. Recorded in
    ``cv_metrics.json``; the fitted model also stores its own surviving feature list, which
    is what inference aligns to.

    ``train_years`` (``"1999-2023"``, or a comma list) restricts the **training** cohort to
    parcels with those PETT label years, in both the CV folds and the final refit.
    Validation and locked-test membership are deliberately left untouched, so CV stays
    directly comparable to a run without the flag. Motivation: ~80 % of labels are 1998–99
    and 1997–98 was the catastrophic Piura El Nino, whose imagery the model cannot read
    (RESULTS.md §8.1/§8.2).

    ``augment_feat`` is a degraded copy of the feature store
    (``allperu.density.build_degraded_features``) appended to the **train** side only, so
    each parcel appears at both its own and the endpoint observation density under one
    label. Val/test are untouched for the same reason as ``train_years``: CV has to stay
    comparable, and a degraded row in a validation fold would be a row the model trained
    on. See docs/all_peru/temporal_ood_plan.md §1b."""
    drop_features = resolve_drop_features(drop_features)
    train_years = parse_year_spec(train_years)
    parcels = load_parcels(require_quality=True)
    with open(proc() / "label_map.json") as f:
        label_map = json.load(f)
    classes = [c for c, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    n_classes = len(classes)
    folds = sorted(parcels.loc[parcels.fold >= 0, "fold"].unique())
    if n_folds is not None:
        folds = folds[:n_folds]

    run_dir = runs() / (run_name or f"{model_name}_{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    model_kw = model_kw or {}
    print(f"model={model_name}  parcels={len(parcels):,}  classes={n_classes}  "
          f"folds={list(folds)}  -> {run_dir}")
    if drop_features:
        print(f"withholding {len(drop_features)} features: {drop_features}")
    if train_years:
        print(f"restricting TRAIN to label years {train_years[0]}-{train_years[-1]} "
              f"({len(train_years)} years); val/test membership unchanged")

    # ---- spatial CV ----
    cv = []
    for k in folds:
        fold_dir = run_dir / f"fold{k}"
        fold_dir.mkdir(exist_ok=True)
        tr_idx, va_idx = fold_split(parcels, k)
        tr_idx = restrict_years(parcels, tr_idx, train_years)
        model = get_model(model_name, **model_kw)
        model.set_n_classes(n_classes)
        _set_class_names(model, classes)
        train_ds = make_dataset(model.input_kind, parcels, tr_idx,
                                drop_features=drop_features, augment_feat=augment_feat)
        norm = getattr(train_ds, "normalizer", None)
        val_ds = make_dataset(model.input_kind, parcels, va_idx, normalizer=norm,
                              drop_features=drop_features)
        model.fit(train_ds, val_ds, class_weights(train_ds.y, n_classes), fold_dir)
        prob = model.predict_proba(val_ds)
        m = compute_metrics(val_ds.y, prob)
        m["fold"] = int(k)
        m["n_train"], m["n_val"] = len(train_ds.y), len(val_ds.y)
        cv.append(m)
        _save_preds(val_ds, parcels, prob, classes, fold_dir / "preds_val.parquet")
        print(f"  fold {k}: macro_f1={m['macro_f1']:.3f}  "
              f"(train {m['n_train']:,} / val {m['n_val']:,})")

    cv_df = pd.DataFrame(cv)
    summary = {"model": model_name, "model_kw": model_kw,
               "drop_features": drop_features,
               "train_years": train_years,
               "augment_feat": str(augment_feat) if augment_feat else None,
               "cv_macro_f1_mean": float(cv_df.macro_f1.mean()),
               "cv_macro_f1_std": float(cv_df.macro_f1.std()),
               "folds": cv}
    print(f"CV macro-F1: {summary['cv_macro_f1_mean']:.3f} "
          f"± {summary['cv_macro_f1_std']:.3f}")

    # ---- final refit on all trainval minus test buffer ----
    tr_idx, te_idx = final_split(parcels)
    tr_idx = restrict_years(parcels, tr_idx, train_years)
    model = get_model(model_name, **model_kw)
    model.set_n_classes(n_classes)
    _set_class_names(model, classes)
    train_ds = make_dataset(model.input_kind, parcels, tr_idx,
                            drop_features=drop_features, augment_feat=augment_feat)
    norm = getattr(train_ds, "normalizer", None)
    # early stopping still needs a val set: reuse fold-0 val (already excluded by buffer)
    _, va_idx = fold_split(parcels, folds[0])
    val_ds = make_dataset(model.input_kind, parcels, va_idx, normalizer=norm,
                          drop_features=drop_features)
    model.fit(train_ds, val_ds, class_weights(train_ds.y, n_classes), run_dir)
    model.save(run_dir / "model.bin")
    with open(run_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    with open(run_dir / "cv_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- optional single locked-test evaluation ----
    if eval_test:
        test_ds = make_dataset(model.input_kind, parcels, te_idx, normalizer=norm,
                               drop_features=drop_features)
        prob = model.predict_proba(test_ds)
        m = compute_metrics(test_ds.y, prob)
        print(f"LOCKED TEST macro-F1: {m['macro_f1']:.3f}  (n={m['n']:,})")
        _save_preds(test_ds, parcels, prob, classes, run_dir / "preds_test.parquet")
        with open(run_dir / "test_metrics.json", "w") as f:
            json.dump(m, f, indent=2)

    print(f"run saved to {run_dir}")
    return run_dir


def sweep(model_name: str, n_trials: int = 30, n_folds: int = 3) -> None:
    """Optuna sweep, objective = mean spatial-CV macro-F1 (plan §9). Never touches test."""
    import optuna

    spaces = {
        "lightgbm": lambda t: {"num_leaves": t.suggest_int("num_leaves", 15, 255),
                               "learning_rate": t.suggest_float("lr", 0.01, 0.2, log=True),
                               "min_child_samples": t.suggest_int("mcs", 5, 100)},
        "ltae": lambda t: {"d_model": t.suggest_categorical("d_model", [64, 128, 256]),
                           "dropout": t.suggest_float("dropout", 0.0, 0.5),
                           "lr": t.suggest_float("lr", 1e-4, 5e-3, log=True)},
        "psetae": lambda t: {"d_model": t.suggest_categorical("d_model", [64, 128, 256]),
                             "d_pix": t.suggest_categorical("d_pix", [32, 64]),
                             "dropout": t.suggest_float("dropout", 0.0, 0.5),
                             "lr": t.suggest_float("lr", 1e-4, 5e-3, log=True)},
    }
    parcels = load_parcels()
    with open(proc() / "label_map.json") as f:
        label_map = json.load(f)
    classes = [c for c, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    n_classes = len(classes)
    folds = sorted(parcels.loc[parcels.fold >= 0, "fold"].unique())[:n_folds]

    def objective(trial):
        kw = spaces[model_name](trial)
        scores = []
        for k in folds:
            tr_idx, va_idx = fold_split(parcels, k)
            model = get_model(model_name, **kw)
            model.set_n_classes(n_classes)
            _set_class_names(model, classes)
            tmp = runs() / "_sweep_tmp"
            tmp.mkdir(parents=True, exist_ok=True)
            train_ds = make_dataset(model.input_kind, parcels, tr_idx)
            val_ds = make_dataset(model.input_kind, parcels, va_idx,
                                  normalizer=getattr(train_ds, "normalizer", None))
            model.fit(train_ds, val_ds, class_weights(train_ds.y, n_classes), tmp)
            scores.append(compute_metrics(val_ds.y, model.predict_proba(val_ds))["macro_f1"])
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    out = runs() / f"sweep_{model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(out / "sweep.csv", index=False)
    with open(out / "best.json", "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f,
                  indent=2)
    print(f"best macro-F1 {study.best_value:.3f} with {study.best_params}")

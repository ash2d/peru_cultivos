"""Leave-one-department-out — the spatial-generalisation test Piura could not run.

The Piura pipeline's spatial CV holds out 5 km *regions*: neighbouring cells, same
department, same agro-climate, same titling campaign. It measures "can the model read a
field it has not seen", not "can the model read a **place** it has not seen". With 15
departments spanning the coastal desert, the sierra and the high jungle, the second question
is finally askable, and it is the one that matters for the stated objective — a classifier
that is *spatially generalisable*.

For each department: train on every **other** department, evaluate on that one. Training
parcels within ``buffer_m`` of a held-out parcel are dropped, so parcels straddling a shared
border cannot leak (the same 1.5 km dead-zone the normal split uses).

Read the output as a **lower bound**, and read the spread as the real result. A department
that scores far below the pooled CV number is telling you the model has learned something
local — a regional cropping calendar, a soil background — rather than the phenology of
perennials in general.

Run with::

    CC_PROC=data/processed/all_peru CC_FEAT=data/processed/all_peru/features \\
      uv run python -m crop_classifier.allperu.lodo --drop-features meta
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from crop_classifier import models  # noqa: F401  (registers the models)
from crop_classifier.data import (
    class_weights,
    load_parcels,
    make_dataset,
    parse_year_spec,
    resolve_drop_features,
    restrict_years,
)
from crop_classifier.evaluate import compute_metrics
from crop_classifier.models.base import get_model
from crop_classifier.paths import proc, runs
from crop_classifier.splits import buffer_exclusions

METRIC_CRS = 32718          # one grid for the whole country; see allperu/sample.py


def run(model_name: str = "lightgbm", drop_features: str | None = "meta",
        train_years: str | None = None, buffer_m: float = 1500.0,
        min_parcels: int = 200, save: bool = True, tag: str = "",
        augment_feat: str | None = None) -> pd.DataFrame:
    """Fit ``n_departments`` models, each blind to one department.

    ``tag`` suffixes every artifact and nests the per-department fits under
    ``runs/lodo/<tag>/``, so a second architecture can be run over the same departments
    without overwriting the first — comparing two models' transfer is the whole point of
    running a second one (RESULTS.md §6.2). Untagged keeps the original layout.
    """
    drop = resolve_drop_features(drop_features)
    years = parse_year_spec(train_years)
    parcels = load_parcels(require_quality=True)
    if "dept" not in parcels.columns:
        raise KeyError("modeling_parcels has no `dept` column — this is the all-Peru "
                       "workspace only (set CC_PROC=data/processed/all_peru)")
    with open(proc() / "label_map.json") as f:
        label_map = json.load(f)
    classes = [c for c, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    n_classes = len(classes)

    m = parcels.geometry.representative_point().to_crs(METRIC_CRS)
    xy = np.c_[m.x.values, m.y.values]

    rows, preds = [], []
    depts = [d for d, n in parcels.dept.value_counts().items() if n >= min_parcels]
    print(f"leave-one-department-out over {len(depts)} departments "
          f"(>= {min_parcels} parcels), model={model_name}, drop={drop or 'none'}")

    rng = np.random.default_rng(42)
    for d in sorted(depts):
        held = (parcels.dept == d).values
        excl = buffer_exclusions(xy, held, ~held, buffer_m)
        pool = parcels.index[~held & ~excl]
        te_idx = parcels.index[held]
        pool = restrict_years(parcels, pool, years)

        # Early stopping needs a validation set, and it must NOT be the held-out
        # department — that would tune the stopping point on the very data being scored.
        # Carve ~10 % of the *training* departments' regions instead, grouped by region so
        # the inner split is spatial too.
        regions = parcels.loc[pool, "region_id"].unique()
        val_regions = set(rng.choice(regions, max(1, int(0.10 * len(regions))),
                                     replace=False))
        in_val = parcels.loc[pool, "region_id"].isin(val_regions).values
        tr_idx, va_idx = pool[~in_val], pool[in_val]

        model = get_model(model_name)
        model.set_n_classes(n_classes)
        if hasattr(model, "set_class_names"):
            model.set_class_names(classes)
        out_dir = (runs() / "lodo" / tag / d) if tag else (runs() / "lodo" / d)
        out_dir.mkdir(parents=True, exist_ok=True)
        train_ds = make_dataset(model.input_kind, parcels, tr_idx, drop_features=drop,
                                augment_feat=augment_feat)
        norm = getattr(train_ds, "normalizer", None)
        val_ds = make_dataset(model.input_kind, parcels, va_idx, normalizer=norm,
                              drop_features=drop)
        test_ds = make_dataset(model.input_kind, parcels, te_idx, normalizer=norm,
                               drop_features=drop)
        if len(getattr(test_ds, "y", [])) == 0:
            print(f"  {d:14s} no features in store — skipped")
            continue
        model.fit(train_ds, val_ds, class_weights(train_ds.y, n_classes), out_dir)
        prob = model.predict_proba(test_ds)
        met = compute_metrics(test_ds.y, prob)
        met.update(dept=d, n_train=len(train_ds.y), n_test=len(test_ds.y),
                   n_buffered=int(excl.sum()))
        for i, c in enumerate(classes):
            mask = test_ds.y == i
            met[f"recall_{c}"] = (float((prob.argmax(1)[mask] == i).mean())
                                  if mask.any() else float("nan"))
            met[f"support_{c}"] = int(mask.sum())
        rows.append(met)
        p = pd.DataFrame({"COD_PREDIO": test_ds.cod_predio, "y_true": test_ds.y,
                          "y_pred": prob.argmax(1), "dept": d})
        preds.append(p)
        print(f"  {d:14s} macro_f1={met['macro_f1']:.3f}  acc={met['accuracy']:.3f}  "
              f"(train {met['n_train']:,} / test {met['n_test']:,}, "
              f"{met['n_buffered']:,} buffered)")

    res = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    print(f"\nLODO macro-F1: mean {res.macro_f1.mean():.4f} ± {res.macro_f1.std():.4f}  "
          f"(min {res.macro_f1.min():.3f} {res.loc[res.macro_f1.idxmin(), 'dept']}, "
          f"max {res.macro_f1.max():.3f} {res.loc[res.macro_f1.idxmax(), 'dept']})")
    # parcel-weighted, i.e. what a single national model scores on unseen places
    allp = pd.concat(preds, ignore_index=True)
    from sklearn.metrics import f1_score
    pooled = f1_score(allp.y_true, allp.y_pred, average="macro",
                      labels=range(n_classes), zero_division=0)
    print(f"pooled over all held-out parcels: macro-F1 {pooled:.4f}  (n={len(allp):,})")

    if save:
        suf = f"_{tag}" if tag else ""
        res.to_csv(proc() / f"lodo_metrics{suf}.csv", index=False)
        allp.to_parquet(proc() / f"lodo_predictions{suf}.parquet", index=False)
        with open(proc() / f"lodo_summary{suf}.json", "w") as f:
            json.dump({"model": model_name, "drop_features": drop,
                       "train_years": years, "buffer_m": buffer_m,
                       "mean_macro_f1": float(res.macro_f1.mean()),
                       "std_macro_f1": float(res.macro_f1.std()),
                       "pooled_macro_f1": float(pooled),
                       "n_departments": len(res)}, f, indent=2)
        print(f"wrote lodo_metrics{suf}.csv / lodo_predictions{suf}.parquet to {proc()}")
        # LODYO comes free: the per-department fits already exist, and re-scoring their
        # predictions per label-year cohort is the only evaluation here that is out of
        # distribution in space AND time. It is computed automatically because the one time
        # it was optional it was also the one thing that overturned the selection
        # (RESULTS.md 9.2.1) -- an evaluation nobody remembers to run is an evaluation that
        # does not exist.
        try:
            from crop_classifier.allperu.loyo import lodo_by_cohort
            lodo_by_cohort(tag=tag or "", save=True)
        except Exception as e:                                   # pragma: no cover
            print(f"  (LODYO skipped: {e})")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="lightgbm")
    ap.add_argument("--drop-features", default="meta")
    ap.add_argument("--train-years", default=None)
    ap.add_argument("--buffer-m", type=float, default=1500.0)
    ap.add_argument("--min-parcels", type=int, default=200)
    ap.add_argument("--tag", default="",
                    help="suffix artifacts + nest run dirs, so a second model's LODO "
                         "does not overwrite the first")
    ap.add_argument("--augment-feat", default=None)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    run(model_name=a.model, drop_features=a.drop_features, train_years=a.train_years,
        buffer_m=a.buffer_m, min_parcels=a.min_parcels, save=not a.no_save, tag=a.tag,
        augment_feat=a.augment_feat)


if __name__ == "__main__":
    main()

"""Leave-one-**year**-out cohort transfer — the temporal analogue of LODO (window_plan §4.3).

The gap in the existing design: everything is held out in *space* and scored at the *label
year*. The question that decides whether a 2019-23 prediction is licensed — *does this model
work in a year it never trained on?* — is untestable by construction in Piura, where ~80 % of
labels are 1998-99.

The national data makes it askable for the first time: label years spread over 1997-2006. For
each label-year cohort ``y``: train on cohorts ≠ ``y``, test on cohort ``y`` at its own label
year.

**Region is the confound and it is handled the same way the El Niño test handles it**
(``perennial/diagnostics.py::elnino_confound_test``): titling swept region by region, so a
cohort is also a *place*. Scoring a cohort on regions the training cohorts never covered
measures space, not time. The default therefore restricts to regions containing **≥2
cohorts**, so region is held approximately fixed while year varies, and reports the
unrestricted arm alongside as the loose upper bound on the cohort effect.

Expect **1998 to fail** — it is the El Niño cohort (``../perennial/RESULTS.md`` §8.1) and
W-D3 means the deliverable no longer depends on it. The pass criterion is therefore evaluated
on cohorts other than 1998.

Run with::

    CC_PROC=data/processed/all_peru CC_FEAT=data/processed/all_peru/features \\
      uv run python -m crop_classifier.allperu.loyo --drop-features meta,location
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
    resolve_drop_features,
)
from crop_classifier.evaluate import compute_metrics
from crop_classifier.models.base import get_model
from crop_classifier.paths import proc, runs
from crop_classifier.splits import buffer_exclusions

METRIC_CRS = 32718          # one grid for the whole country; see allperu/sample.py

# Cohort excluded from the pass criterion but always reported: the 1997-98 El Nino.
EXPECTED_FAIL_YEARS = (1998,)
TOLERANCE = 0.10            # worst cohort must be within this of pooled CV (T2)

# The registered criterion is a MINIMUM over cohorts, so it is decided by whichever cohort is
# smallest — nationally that is 1996 with 464 parcels and ~60 PERENNIAL, where a macro-F1
# moves +-0.03 on resampling alone. Comparing two model arms on that number compares noise.
# `worst_cohort_macro_f1_min_support` reports the same statistic over cohorts big enough for
# the difference to mean something. It is reported ALONGSIDE the registered number, never
# instead of it (T-D6: do not move a goalpost, add a second one and say why).
MIN_SUPPORT_FOR_WORST = 1000


def run(model_name: str = "lightgbm", drop_features: str | None = "meta,location",
        buffer_m: float = 1500.0, min_parcels: int = 300, min_cohorts_per_region: int = 2,
        shared_regions_only: bool = True, save: bool = True, tag: str = "",
        augment_feat: str | None = None) -> pd.DataFrame:
    """Fit one model per label-year cohort, each blind to that cohort.

    ``shared_regions_only`` is the design point: it keeps only parcels in regions that hold
    at least ``min_cohorts_per_region`` distinct cohorts, so the held-out year is compared
    against training data from the *same places*. Turn it off to see how much of the cohort
    gap is really a place gap.
    """
    drop = resolve_drop_features(drop_features)
    parcels = load_parcels(require_quality=True)
    if "year" not in parcels.columns:
        raise KeyError("modeling_parcels has no `year` column")
    with open(proc() / "label_map.json") as f:
        label_map = json.load(f)
    classes = [c for c, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    n_classes = len(classes)

    parcels = parcels[parcels["year"].notna()].copy()
    parcels["year"] = parcels["year"].astype(int)
    if shared_regions_only:
        n_coh = parcels.groupby("region_id")["year"].nunique()
        keep = set(n_coh[n_coh >= min_cohorts_per_region].index)
        before = len(parcels)
        parcels = parcels[parcels["region_id"].isin(keep)].copy()
        print(f"shared-region restriction: {len(parcels):,} of {before:,} parcels in "
              f"{len(keep):,} regions holding >= {min_cohorts_per_region} cohorts")

    m = parcels.geometry.representative_point().to_crs(METRIC_CRS)
    xy = np.c_[m.x.values, m.y.values]

    cohorts = [int(y) for y, n in parcels["year"].value_counts().items()
               if n >= min_parcels]
    print(f"leave-one-year-out over {len(cohorts)} cohorts "
          f"(>= {min_parcels} parcels), model={model_name}, drop={drop or 'none'}")

    rng = np.random.default_rng(42)
    rows, preds = [], []
    for y in sorted(cohorts):
        held = (parcels["year"] == y).values
        # A cohort is also a place: buffer the training side against held-out parcels so a
        # neighbouring parcel titled in a different year cannot leak across the boundary.
        excl = buffer_exclusions(xy, held, ~held, buffer_m)
        pool = parcels.index[~held & ~excl]
        te_idx = parcels.index[held]

        regions = parcels.loc[pool, "region_id"].unique()
        val_regions = set(rng.choice(regions, max(1, int(0.10 * len(regions))),
                                     replace=False))
        in_val = parcels.loc[pool, "region_id"].isin(val_regions).values
        tr_idx, va_idx = pool[~in_val], pool[in_val]

        model = get_model(model_name)
        model.set_n_classes(n_classes)
        if hasattr(model, "set_class_names"):
            model.set_class_names(classes)
        out_dir = (runs() / "loyo" / tag / str(y)) if tag else (runs() / "loyo" / str(y))
        out_dir.mkdir(parents=True, exist_ok=True)
        train_ds = make_dataset(model.input_kind, parcels, tr_idx, drop_features=drop,
                                augment_feat=augment_feat)
        norm = getattr(train_ds, "normalizer", None)
        val_ds = make_dataset(model.input_kind, parcels, va_idx, normalizer=norm,
                              drop_features=drop)
        test_ds = make_dataset(model.input_kind, parcels, te_idx, normalizer=norm,
                               drop_features=drop)
        if len(getattr(test_ds, "y", [])) == 0:
            print(f"  {y} no features in store — skipped")
            continue
        model.fit(train_ds, val_ds, class_weights(train_ds.y, n_classes), out_dir)
        prob = model.predict_proba(test_ds)
        met = compute_metrics(test_ds.y, prob)
        met.update(cohort=y, n_train=len(train_ds.y), n_test=len(test_ds.y),
                   n_buffered=int(excl.sum()))
        for i, c in enumerate(classes):
            mask = test_ds.y == i
            met[f"recall_{c}"] = (float((prob.argmax(1)[mask] == i).mean())
                                  if mask.any() else float("nan"))
            met[f"support_{c}"] = int(mask.sum())
        rows.append(met)
        preds.append(pd.DataFrame({"COD_PREDIO": test_ds.cod_predio, "y_true": test_ds.y,
                                   "y_pred": prob.argmax(1), "cohort": y}))
        print(f"  {y}  macro_f1={met['macro_f1']:.3f}  acc={met['accuracy']:.3f}  "
              f"recall_PERENNIAL={met.get('recall_PERENNIAL', float('nan')):.3f}  "
              f"(train {met['n_train']:,} / test {met['n_test']:,})")

    res = pd.DataFrame(rows).sort_values("cohort")
    allp = pd.concat(preds, ignore_index=True)
    from sklearn.metrics import f1_score
    pooled = float(f1_score(allp.y_true, allp.y_pred, average="macro",
                            labels=range(n_classes), zero_division=0))

    ok = res[~res["cohort"].isin(EXPECTED_FAIL_YEARS)]
    worst = float(ok["macro_f1"].min()) if len(ok) else float("nan")
    worst_rec = float(ok["recall_PERENNIAL"].min()) if "recall_PERENNIAL" in ok else np.nan
    big = ok[ok["n_test"] >= MIN_SUPPORT_FOR_WORST]
    worst_big = float(big["macro_f1"].min()) if len(big) else float("nan")
    worst_rec_big = (float(big["recall_PERENNIAL"].min())
                     if len(big) and "recall_PERENNIAL" in big else float("nan"))
    print(f"\nLOYO macro-F1: mean {res.macro_f1.mean():.4f} ± {res.macro_f1.std():.4f}; "
          f"pooled {pooled:.4f}")
    print(f"worst cohort excluding {EXPECTED_FAIL_YEARS}: {worst:.4f} "
          f"({int(ok.loc[ok.macro_f1.idxmin(), 'cohort']) if len(ok) else '-'})")

    if save:
        suf = f"_{tag}" if tag else ""
        res.to_csv(proc() / f"loyo_metrics{suf}.csv", index=False)
        allp.to_parquet(proc() / f"loyo_predictions{suf}.parquet", index=False)
        summary = {"model": model_name, "drop_features": drop, "buffer_m": buffer_m,
                   "augment_feat": str(augment_feat) if augment_feat else None,
                   "shared_regions_only": shared_regions_only,
                   "min_cohorts_per_region": min_cohorts_per_region,
                   "mean_macro_f1": float(res.macro_f1.mean()),
                   "std_macro_f1": float(res.macro_f1.std()),
                   "pooled_macro_f1": pooled,
                   "worst_cohort_macro_f1_excl_elnino": worst,
                   "worst_cohort_recall_PERENNIAL_excl_elnino": worst_rec,
                   "min_support_for_worst": MIN_SUPPORT_FOR_WORST,
                   "worst_cohort_macro_f1_min_support": worst_big,
                   "worst_cohort_recall_PERENNIAL_min_support": worst_rec_big,
                   "n_cohorts": int(len(res)),
                   "tolerance": TOLERANCE}
        with open(proc() / f"loyo_summary{suf}.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote loyo_metrics{suf}.csv / loyo_summary{suf}.json to {proc()}")
    return res


def lodo_by_cohort(tag: str = "nolat", min_parcels: int = 300,
                   save: bool = True) -> pd.DataFrame:
    """Score an existing **LODO** run's predictions per label-year cohort — "LODYO".

    The evaluation that settles what LOYO alone cannot. LOYO holds region approximately
    fixed (``shared_regions_only``) so that year varies and place does not — which means it
    inherits spatial CV's blind spot exactly: a time-invariant lookup table such as
    ``centroid_lat`` is as available in LOYO as in CV, and helps by the same amount.

    Re-scoring the LODO predictions by cohort holds out the **department** *and* varies the
    year, so a feature that only memorises place cannot help. It costs nothing — the
    per-department fits already exist — and it is the number to read when CV, LODO and LOYO
    disagree (docs/RESULTS.md §6.2).

    ⚠️ It is not a substitute for LOYO: each cohort here is scored by 14 different models,
    one per department, so it measures the *joint* out-of-distribution setting, not the
    temporal one alone.
    """
    import geopandas as gpd
    from sklearn.metrics import f1_score

    suf = f"_{tag}" if tag else ""
    preds = pd.read_parquet(proc() / f"lodo_predictions{suf}.parquet")
    par = gpd.read_parquet(proc() / "modeling_parcels.parquet")[["COD_PREDIO", "year"]]
    d = preds.merge(par, on="COD_PREDIO", how="left")
    d = d[d["year"].notna()].copy()
    d["year"] = d["year"].astype(int)
    labels = sorted(pd.unique(np.r_[d.y_true.to_numpy(), d.y_pred.to_numpy()]))
    rows = []
    for y, sub in d.groupby("year"):
        if len(sub) < min_parcels:
            continue
        rows.append({"cohort": int(y), "n": len(sub),
                     "macro_f1": float(f1_score(sub.y_true, sub.y_pred, average="macro",
                                                labels=labels, zero_division=0)),
                     "accuracy": float((sub.y_true == sub.y_pred).mean())})
    res = pd.DataFrame(rows).sort_values("cohort").reset_index(drop=True)
    ok = res[~res["cohort"].isin(EXPECTED_FAIL_YEARS)]
    big = ok[ok["n"] >= MIN_SUPPORT_FOR_WORST]
    summary = {
        "tag": tag, "n_cohorts": int(len(res)),
        "mean_macro_f1": float(res.macro_f1.mean()),
        "std_macro_f1": float(res.macro_f1.std()),
        "pooled_macro_f1": float(f1_score(d.y_true, d.y_pred, average="macro",
                                          labels=labels, zero_division=0)),
        "worst_cohort_macro_f1_excl_elnino": float(ok.macro_f1.min()) if len(ok) else np.nan,
        "worst_cohort_macro_f1_min_support": (float(big.macro_f1.min()) if len(big)
                                              else float("nan")),
        "min_support_for_worst": MIN_SUPPORT_FOR_WORST,
    }
    print(f"LODYO ({tag or 'untagged'}): mean {summary['mean_macro_f1']:.4f} "
          f"± {summary['std_macro_f1']:.4f}; pooled {summary['pooled_macro_f1']:.4f}; "
          f"worst {summary['worst_cohort_macro_f1_excl_elnino']:.4f} "
          f"(>= {MIN_SUPPORT_FOR_WORST}: "
          f"{summary['worst_cohort_macro_f1_min_support']:.4f})")
    if save:
        res.to_csv(proc() / f"lodyo_metrics{suf}.csv", index=False)
        with open(proc() / f"lodyo_summary{suf}.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote lodyo_metrics{suf}.csv / lodyo_summary{suf}.json to {proc()}")
    return res


def verdict(summary: dict, cv_macro_f1: float,
            cv_recall_perennial: float | None = None) -> dict:
    """T2 pass/fail: worst non-El-Niño cohort within ``TOLERANCE`` of pooled CV."""
    d_f1 = cv_macro_f1 - summary["worst_cohort_macro_f1_excl_elnino"]
    out = {"cv_macro_f1": cv_macro_f1, "gap_macro_f1": float(d_f1),
           "pass_macro_f1": bool(d_f1 < TOLERANCE)}
    if cv_recall_perennial is not None:
        d_r = cv_recall_perennial - summary["worst_cohort_recall_PERENNIAL_excl_elnino"]
        out.update(cv_recall_PERENNIAL=cv_recall_perennial, gap_recall=float(d_r),
                   pass_recall=bool(d_r < TOLERANCE))
    out["pass"] = bool(out["pass_macro_f1"] and out.get("pass_recall", True))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="lightgbm")
    ap.add_argument("--drop-features", default="meta,location")
    ap.add_argument("--buffer-m", type=float, default=1500.0)
    ap.add_argument("--min-parcels", type=int, default=300)
    ap.add_argument("--min-cohorts-per-region", type=int, default=2)
    ap.add_argument("--all-regions", action="store_true",
                    help="do NOT restrict to regions holding >= 2 cohorts (upper bound)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--augment-feat", default=None)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    run(model_name=a.model, drop_features=a.drop_features, buffer_m=a.buffer_m,
        min_parcels=a.min_parcels, min_cohorts_per_region=a.min_cohorts_per_region,
        shared_regions_only=not a.all_regions, save=not a.no_save, tag=a.tag,
        augment_feat=a.augment_feat)


if __name__ == "__main__":
    main()

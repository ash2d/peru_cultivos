"""Year-leak audit — the temporal-OOD work (docs/RESULTS.md §6).

The generalised form of the ``frac_l7`` finding (``../perennial/RESULTS.md`` §4.6): *any*
feature that identifies **which year you are in** is a feature that will mislead you in a
year you have never seen. ``frac_l7`` was the obvious case — it ramps 0 → 1 as L7 replaces
L5 and was the 2nd-highest-gain feature — but it was found by inspection, not by a test. This
module is the test.

Two rankings, deliberately, because they fail in different ways:

* **``eta2_year``** — the share of a feature's variance explained by the label-year cohort,
  i.e. a one-way ANOVA R². Direct, cheap, and computed for every feature independently, so a
  feature is not hidden by a correlated neighbour.
* **LightGBM gain** on a model whose *target is the cohort*. Captures interactions the ANOVA
  cannot, but gain is diluted across correlated features and — measured twice in this project
  — **gain ≠ contribution**.

⚠️ **Cohort is confounded with place.** Titling swept region by region, so "features that
identify 1998" and "features that identify Piura" are partly the same features. Both
rankings are therefore also reported **within region** (region means removed first), which is
the only version that isolates time. The unconditional version is kept alongside as the
loose upper bound, exactly as ``loyo.py`` keeps its unrestricted arm.

Nothing here decides anything. It **generates candidates**; LOYO decides (T-D1/T-D2).

Run with::

    CC_PROC=data/processed/all_peru CC_FEAT=data/processed/all_peru/features \\
      uv run python -m crop_classifier.cli allperu year-leak --drop-features meta,location
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from crop_classifier.data import load_parcels, make_flat, resolve_drop_features
from crop_classifier.paths import proc


def _demean_by(x: np.ndarray, gid: np.ndarray) -> np.ndarray:
    """Remove group means from every column of ``x`` (absorbing a fixed effect)."""
    codes, _ = pd.factorize(gid)
    n_g = codes.max() + 1
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        col = x[:, j]
        ok = np.isfinite(col)
        m = np.zeros(n_g)
        c = np.bincount(codes[ok], minlength=n_g).astype(float)
        m[c > 0] = (np.bincount(codes[ok], weights=col[ok], minlength=n_g)[c > 0]
                    / c[c > 0])
        out[:, j] = col - m[codes]
    return out


def eta_squared(x: np.ndarray, gid: np.ndarray) -> np.ndarray:
    """One-way ANOVA R² of each column of ``x`` on the grouping ``gid``.

    "How much of this feature is just *which cohort* the parcel belongs to." Bounded [0, 1];
    NaNs are dropped per column.
    """
    codes, _ = pd.factorize(gid)
    n_g = codes.max() + 1
    out = np.full(x.shape[1], np.nan)
    for j in range(x.shape[1]):
        col = x[:, j]
        ok = np.isfinite(col)
        if ok.sum() < 100:
            continue
        v = col[ok]
        c = codes[ok]
        cnt = np.bincount(c, minlength=n_g).astype(float)
        gm = np.zeros(n_g)
        gm[cnt > 0] = np.bincount(c, weights=v, minlength=n_g)[cnt > 0] / cnt[cnt > 0]
        ss_tot = float(((v - v.mean()) ** 2).sum())
        ss_bet = float((cnt * (gm - v.mean()) ** 2).sum())
        out[j] = ss_bet / ss_tot if ss_tot > 0 else np.nan
    return out


def run(drop_features: str | None = "meta,location", min_parcels: int = 300,
        shared_regions_only: bool = True, min_cohorts_per_region: int = 2,
        fit_gain: bool = True, save: bool = True, tag: str = "") -> pd.DataFrame:
    """Rank features by how strongly they identify the label-year cohort.

    ``shared_regions_only`` mirrors ``loyo.run``: keep only parcels in regions holding at
    least ``min_cohorts_per_region`` cohorts, so region is approximately fixed while year
    varies. Without it the ranking is contaminated by pure geography.
    """
    drop = resolve_drop_features(drop_features)
    parcels = load_parcels(require_quality=True)
    parcels = parcels[parcels["year"].notna()].copy()
    parcels["year"] = parcels["year"].astype(int)
    if shared_regions_only:
        n_coh = parcels.groupby("region_id")["year"].nunique()
        keep = set(n_coh[n_coh >= min_cohorts_per_region].index)
        parcels = parcels[parcels["region_id"].isin(keep)].copy()
    cohorts = {y for y, n in parcels["year"].value_counts().items() if n >= min_parcels}
    parcels = parcels[parcels["year"].isin(cohorts)].copy()

    ds = make_flat(parcels, parcels.index, drop_features=drop)
    idx = pd.Index(ds.cod_predio)
    meta = (parcels.set_index("COD_PREDIO").loc[idx, ["year", "region_id"]]
            .reset_index())
    X = ds.X.to_numpy(dtype=float)
    names = list(ds.feature_names)
    yr = meta["year"].to_numpy()
    reg = meta["region_id"].to_numpy()

    print(f"year-leak audit: {len(X):,} parcels, {len(cohorts)} cohorts, "
          f"{len(names)} features, drop={drop or 'none'}")
    eta_raw = eta_squared(X, yr)
    eta_within = eta_squared(_demean_by(X, reg), yr)
    out = pd.DataFrame({"feature": names, "eta2_year": eta_raw,
                        "eta2_year_within_region": eta_within})

    if fit_gain:
        import lightgbm as lgb

        classes = sorted(cohorts)
        ymap = {y: i for i, y in enumerate(classes)}
        y = np.array([ymap[v] for v in yr])
        Xdf = pd.DataFrame(X, columns=names)
        params = {"objective": "multiclass", "num_class": len(classes),
                  "learning_rate": 0.1, "num_leaves": 63, "verbosity": -1,
                  "seed": 42, "num_threads": 0}
        booster = lgb.train(params, lgb.Dataset(Xdf, label=y, free_raw_data=False),
                            num_boost_round=200)
        gain = booster.feature_importance("gain")
        out["cohort_gain"] = gain
        out["cohort_gain_share"] = gain / max(gain.sum(), 1e-9)

        # Held-out predictability is the number that means something. An in-sample fit on
        # 134 features reaches 1.000 and says only that the model memorised 47 k rows.
        # The split is by REGION, because cohort is confounded with place: a random-row
        # split would let the model recognise the held-out parcel's neighbours.
        rng = np.random.default_rng(42)
        regs = pd.unique(reg)
        held = set(rng.choice(regs, max(1, int(0.2 * len(regs))), replace=False))
        te = np.array([r in held for r in reg])
        b2 = lgb.train(params, lgb.Dataset(Xdf[~te], label=y[~te], free_raw_data=False),
                       num_boost_round=200)
        acc = float((b2.predict(Xdf[te]).argmax(1) == y[te]).mean())
        base = float(pd.Series(y[te]).value_counts(normalize=True).max())
        print(f"cohort is predictable from the features at held-out accuracy {acc:.3f} "
              f"(majority baseline {base:.3f}) over {len(classes)} cohorts")
        out.attrs["cohort_accuracy_heldout"] = acc
        out.attrs["cohort_majority_baseline"] = base

    out = out.sort_values("eta2_year_within_region", ascending=False).reset_index(drop=True)
    if save:
        suf = f"_{tag}" if tag else ""
        p = proc() / f"year_leak_audit{suf}.csv"
        out.to_csv(p, index=False)
        summ = {"n_parcels": int(len(X)), "n_cohorts": len(cohorts),
                "drop_features": drop,
                "shared_regions_only": shared_regions_only,
                "cohort_accuracy_heldout": out.attrs.get("cohort_accuracy_heldout"),
                "cohort_majority_baseline": out.attrs.get("cohort_majority_baseline"),
                "top10_within_region": out.head(10)["feature"].tolist()}
        with open(proc() / f"year_leak_summary{suf}.json", "w") as f:
            json.dump(summ, f, indent=2)
        print(f"wrote {p}")
    return out


def top_k(audit: pd.DataFrame, k: int,
          by: str = "eta2_year_within_region") -> list[str]:
    """The ``k`` most year-identifying features — a LOYO *candidate* list, not a decision."""
    return audit.sort_values(by, ascending=False).head(k)["feature"].tolist()

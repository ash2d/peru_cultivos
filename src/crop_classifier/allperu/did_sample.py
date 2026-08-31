"""The extraction sample for the pre-trend-corrected tenure DiD (``docs/RESULTS.md`` §7).

The v2 study stopped at N3 because its gate demanded *proof* that the pre-trend was
negligible, and the equivalence test that would need 25,202 parcels per arm while all of Peru
holds 6,559 (RESULTS.md §10.4). The v3 design measures the pre-trend and **subtracts** it,
which is what the archive can actually support — so the sample is sized to take **every
qualifying treated parcel**, not to reach a precision target that does not exist.

Design, in the order it binds:

1. **Population** = R1-R4 survivors (``tenure_did.restrict``'s restrictions, applied here to
   the *full* national parcel table rather than to a panel that was drawn for another
   purpose). Treated = ``became_registered``; control = ``NO INSCRITO`` at both observations
   (N-D9).
2. **Strata** = ``department x declaration-year cohort x arm`` (**R5**) — the registration
   rate is non-monotone in the gap between the two tenure observations, so the gap marks a
   departmental campaign wave rather than a duration.
3. **Treated is a census.** All 6,559 of them; there is no larger pool to draw from and
   leaving any behind buys nothing.
4. **Controls at 2:1 within stratum, preferring regions that already hold treated parcels.**
   The existing panel failed G2's shared-region check at 0.366 — treatment was largely
   collinear with the clustering unit, so the region-clustered SEs were not doing what they
   appeared to. Drawing controls region-first is the fix that is available by design.
5. **Whole 5 km regions**, never scattered parcels: crops grow in single-crop blocks, so a
   scattered draw destroys spatial coherence and inflates apparent difficulty.
6. **Sampling weights** travel with every row. The draw is deliberately not proportional —
   *any share reported without them is a statement about the sample, not about Peru*.

⚠️ **The 2:1 target is not reachable in the strata that need it, and topping up elsewhere
would be waste, not rescue.** LA_LIBERTAD 2006 holds 2,849 treated parcels against 2,193
eligible controls in the same department-cohort. Controls drawn from a stratum with no
treated parcels are absorbed by the department x POST fixed effects and contribute **nothing**
to the treatment coefficient, so the shortfall is taken as-is and reported.

Run with::

    CC_PROC=data/processed/all_peru_did CC_FEAT=data/processed/all_peru_did/features \\
    CC_RUNS=runs/all_peru uv run python -m crop_classifier.cli allperu did-sample \\
        --source data/processed/all_peru_full \\
        --tenure data/processed/all_peru/tenure_two_period.parquet \\
        --model-sample data/processed/all_peru/modeling_parcels.parquet
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from crop_classifier.allperu.sample import region_grid
from crop_classifier.allperu.tenure_did import POST_WINDOWS, WINDOWS
from crop_classifier.paths import proc

CONTROL_RATIO = 2.0        # controls per treated parcel, within stratum
MAX_PER_REGION = 220       # no single 5 km cell may dominate a stratum
COHORT_MIN_YEAR = 2004     # R4 for the registered W99 pre-window


def build_population(source: Path | str, tenure: Path | str,
                     cohort_min_year: int = COHORT_MIN_YEAR,
                     post_windows: list[str] | None = None,
                     model_sample: Path | str | None = None) -> gpd.GeoDataFrame:
    """R1-R4 over the full national table, with arms, cohorts and 5 km regions attached.

    ``model_sample`` is the *model's own* parcel table; joining its ``split`` is what makes
    G2's training-set-membership diagnostic computable. A parcel absent from it was never
    seen by the model at all, which is a third state and is kept as ``NaN`` rather than
    folded into "not trainval".
    """
    post = post_windows or POST_WINDOWS
    first_post_year = min(WINDOWS[w][0] for w in post)

    src = gpd.read_parquet(Path(source) / "modeling_parcels.parquet")
    ten = pd.read_parquet(tenure).drop_duplicates("COD_PREDIO")
    d = src[src["label"] == "ANNUAL"].merge(                                    # R1
        ten.drop(columns=[c for c in ("dept",) if c in ten.columns]),
        on="COD_PREDIO", how="inner")
    steps = {"R1_at_risk": int(len(d))}

    d = d[d["cadastre_date"].notna() & d["reg_year"].notna()]                   # R2
    steps["R2_dated"] = int(len(d))
    cy = pd.to_datetime(d["cadastre_date"]).dt.year
    d = d[(cy >= d["reg_year"]) & (cy < first_post_year)].copy()                # R3
    steps["R3_ordered"] = int(len(d))
    d = d[d["reg_year"] >= cohort_min_year].copy()                             # R4
    steps["R4_clean_pre"] = int(len(d))

    d["cadastre_year"] = pd.to_datetime(d["cadastre_date"]).dt.year
    d["arm"] = np.where(d["became_registered"], "treated",
                        np.where(d["tenure"] == "NO INSCRITO", "control", "drop"))
    d = d[d["arm"] != "drop"].copy()                                            # N-D9
    steps["after_control_definition"] = int(len(d))
    d["cohort"] = d["reg_year"].astype(int)                                     # R5
    d["region_id"] = region_grid(d)

    if model_sample is not None and Path(model_sample).exists():
        ms = gpd.read_parquet(model_sample)
        cols = [c for c in ("COD_PREDIO", "split") if c in ms.columns]
        d = d.merge(ms[cols].drop_duplicates("COD_PREDIO"), on="COD_PREDIO", how="left")

    d.attrs["steps"] = steps
    print("population after R1-R4 + N-D9:")
    for k, v in steps.items():
        print(f"  {k:26s} {v:>9,}")
    print(d.groupby("arm").size().to_string())
    return gpd.GeoDataFrame(d, geometry="geometry", crs=src.crs)


def draw(pop: gpd.GeoDataFrame, control_ratio: float = CONTROL_RATIO,
         max_per_region: int = MAX_PER_REGION, seed: int = 42) -> gpd.GeoDataFrame:
    """Every treated parcel, plus controls at ``control_ratio`` within each stratum.

    Controls are taken **region-first**, and regions that already hold treated parcels of the
    same stratum are offered first. That is the whole fix for G2's shared-region failure: a
    region assigned entirely to one arm makes treatment collinear with the clustering unit
    and turns every local shock into a confound.
    """
    rng = np.random.default_rng(seed)
    treated = pop[pop["arm"] == "treated"]
    controls = pop[pop["arm"] == "control"]
    picked = [treated]
    shortfall: list[dict] = []

    for (dept, cohort), t_grp in treated.groupby(["dept", "cohort"], observed=True):
        quota = int(round(len(t_grp) * control_ratio))
        pool = controls[(controls["dept"] == dept) & (controls["cohort"] == cohort)]
        if pool.empty:
            shortfall.append({"dept": dept, "cohort": int(cohort), "n_treated": len(t_grp),
                              "quota": quota, "drawn": 0})
            continue
        # regions holding treated parcels of this stratum come first, in random order
        t_regions = set(t_grp["region_id"])
        order = pd.Series(pool["region_id"].unique())
        order = order.sample(frac=1.0, random_state=seed)
        order = pd.concat([order[order.isin(t_regions)], order[~order.isin(t_regions)]])

        got, taken = 0, []
        for region in order:
            if got >= quota:
                break
            grp = pool[pool["region_id"] == region]
            take = min(len(grp), max_per_region, quota - got)
            taken.append(grp.sample(take, random_state=int(rng.integers(1 << 31))))
            got += take
        if taken:
            picked.append(pd.concat(taken))
        if got < quota:
            shortfall.append({"dept": dept, "cohort": int(cohort), "n_treated": len(t_grp),
                              "quota": quota, "drawn": got})

    out = gpd.GeoDataFrame(pd.concat(picked, ignore_index=True), crs=pop.crs)
    out = out.drop_duplicates("COD_PREDIO")
    out.attrs["shortfall"] = shortfall
    return out


def add_weights(sample: gpd.GeoDataFrame, pop: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """``sample_weight`` = stratum population / stratum sample, stratum = dept x cohort x arm.

    Treated parcels are a census, so their weight is 1 by construction — which is exactly the
    check that the weight is doing what it claims.
    """
    keys = ["dept", "cohort", "arm"]
    pop_n = pop.groupby(keys, observed=True).size()
    smp_n = sample.groupby(keys, observed=True).size()
    w = (pop_n / smp_n).rename("sample_weight")
    out = sample.merge(w, left_on=keys, right_index=True, how="left")
    out["sample_weight"] = (out["sample_weight"].replace([np.inf, -np.inf], np.nan)
                            .fillna(1.0).astype(float))
    return out


def _deff(w: np.ndarray) -> float:
    """Kish design effect ``1 + CV(w)^2`` — what the weights cost in effective n."""
    return float(1.0 + (w.std() / w.mean()) ** 2) if len(w) else float("nan")


def sample(source: Path | str, tenure: Path | str,
           model_sample: Path | str | None = None,
           cohort_min_year: int = COHORT_MIN_YEAR,
           control_ratio: float = CONTROL_RATIO, max_per_region: int = MAX_PER_REGION,
           seed: int = 42, save: bool = True) -> gpd.GeoDataFrame:
    """Draw the DiD extraction sample into ``CC_PROC`` and report what it achieved."""
    pop = build_population(source, tenure, cohort_min_year=cohort_min_year,
                           model_sample=model_sample)
    out = draw(pop, control_ratio=control_ratio, max_per_region=max_per_region, seed=seed)
    short = out.attrs.get("shortfall", [])
    out = add_weights(out, pop)

    n_t = int((out["arm"] == "treated").sum())
    n_c = int((out["arm"] == "control").sum())
    by_region = out.groupby("region_id")["arm"].nunique()
    shared = float(out["region_id"].isin(by_region[by_region > 1].index).mean())

    print(f"\nsampled {len(out):,} parcels: {n_t:,} treated + {n_c:,} control "
          f"({n_c / max(n_t, 1):.2f}:1)")
    print(f"  expanded population {out['sample_weight'].sum():,.0f}; "
          f"design effect {_deff(out['sample_weight'].to_numpy(float)):.2f}")
    print(f"  parcels in a region holding BOTH arms: {shared:.3f} "
          f"(the existing panel: 0.366)")
    print("\nby department:")
    print(pd.crosstab(out["dept"], out["arm"]).to_string())
    if short:
        s = pd.DataFrame(short)
        print(f"\nstrata short of the {control_ratio:g}:1 target "
              f"({int((s['quota'] - s['drawn']).sum()):,} controls unavailable):")
        print(s.sort_values("quota", ascending=False).head(12).to_string(index=False))

    if save:
        d = proc()
        for name in ("modeling_parcels.parquet", "panel_parcels.parquet"):
            if (d / name).exists():
                raise FileExistsError(f"{d / name} exists — never overwrite a drawn sample")
        out.to_parquet(d / "modeling_parcels.parquet", index=False)
        # the extraction reads panel_parcels.parquet; it is the same table here because the
        # DiD panel *is* the sample — there is no second down-sampling stage.
        out.to_parquet(d / "panel_parcels.parquet", index=False)
        src_map = Path(source) / "label_map.json"
        if src_map.exists():
            shutil.copy(src_map, d / "label_map.json")
        meta = {"source": str(source), "tenure": str(tenure),
                "model_sample": str(model_sample) if model_sample else None,
                "cohort_min_year": cohort_min_year, "control_ratio": control_ratio,
                "max_per_region": max_per_region, "seed": seed,
                "population_steps": pop.attrs.get("steps", {}),
                "n_population": int(len(pop)), "n_sampled": int(len(out)),
                "n_treated": n_t, "n_control": n_c,
                "achieved_ratio": n_c / max(n_t, 1),
                "shared_region_share": shared,
                "deff": _deff(out["sample_weight"].to_numpy(float)),
                "shortfall_strata": short,
                "by_dept_arm": {f"{k[0]}|{k[1]}": int(v) for k, v in
                                out.groupby(["dept", "arm"]).size().items()},
                "by_cohort_arm": {f"{k[0]}|{k[1]}": int(v) for k, v in
                                  out.groupby(["cohort", "arm"]).size().items()}}
        with open(d / "did_sample_meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=float)
        print(f"\nwrote {d / 'modeling_parcels.parquet'}, panel_parcels.parquet "
              f"and did_sample_meta.json")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--tenure", type=Path, required=True)
    ap.add_argument("--model-sample", type=Path, default=None)
    ap.add_argument("--cohort-min-year", type=int, default=COHORT_MIN_YEAR)
    ap.add_argument("--control-ratio", type=float, default=CONTROL_RATIO)
    ap.add_argument("--max-per-region", type=int, default=MAX_PER_REGION)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    sample(a.source, a.tenure, model_sample=a.model_sample,
           cohort_min_year=a.cohort_min_year, control_ratio=a.control_ratio,
           max_per_region=a.max_per_region, seed=a.seed, save=not a.no_save)


if __name__ == "__main__":
    main()

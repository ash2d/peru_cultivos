"""The window-deliverable extraction sample: stratified on ``dept × tenure × label`` (§4.5/§4.6).

Earlier samples never stratified on tenure — balance "came out fine by luck"
(``../perennial/RESULTS.md`` §7.5). The deliverable now rests on the ``INSCRITO × ANNUAL``
cell, so it is stratified rather than lucky.

Design, in the order it binds:

1. **Population** = parcels with a 3-class PETT label, non-null ``ESTADO en RRPP``, passing
   the area gate (§2).
2. **Strata** = ``department × tenure × PETT label``.
3. **Oversample the at-risk cells** (``ANNUAL`` × each tenure) to the §4.6 target; keep
   ``PERENNIAL`` proportional — it is the drift control (M2) and needs ~1,500+ parcels to
   read at all.
4. **Whole 5 km regions**, never scattered parcels — as in ``sample.py``: crops grow in
   single-crop blocks, so a scattered draw destroys the buffered spatial split.
5. **Sampling weights** (stratum population / stratum sample) travel with every row. *Any
   share reported without them is wrong* — this design deliberately distorts the tenure and
   class mix.

Cost: ~13-15 k parcels × 10 years (1999-2003 + 2019-2023) ≈ 130-150 k parcel-years,
comparable to the existing 25-year panel's 114 k — ~3× the parcels at the same cost, buying
10 years instead of 25. That trade is the point.

⚠️ **Drawing this sample is cheap; extracting it is not.** As of 2026-08-10 the T1/T2/T3
gates in RESULTS.md §5 have all **failed** and the stop rule says no GEE budget until that is
understood. This module makes the draw reproducible *before* anyone spends it, not
permission to spend.

Run with::

    CC_PROC=data/processed/all_peru_window uv run python -m crop_classifier.archive.window_sample \\
        --source data/processed/all_peru_full \\
        --tenure data/processed/all_peru/tenure_by_predio.parquet
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
from crop_classifier.paths import proc

# §4.6 targets. `n_at_risk_per_tenure` is set by the power calculation below; the control
# quotas are what make the M2 drift check and the PASTURE_FALLOW comparison readable.
TARGET_AT_RISK_PER_TENURE = 5_000
TARGET_PERENNIAL = 1_500
TARGET_PASTURE = 1_500


def power_n(p1: float, p2: float, alpha: float = 0.05, power: float = 0.80,
            deff: float = 1.4) -> float:
    """Parcels per tenure group to detect ``p2 - p1`` in a two-proportion test.

    ``deff`` is the design effect from the deliberately unequal-probability weights, so
    effective n is below the parcel count. RESULTS.md §5 quoted ~1,500 before and ~2,100
    after ``deff``; the ≥5,000 target is for department fixed effects and heterogeneity, not
    the headline test.
    """
    from scipy.stats import norm

    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    delta = abs(p2 - p1)
    if delta == 0:
        return float("inf")
    n = (z_a + z_b) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / delta ** 2
    return float(n * deff)


def power_table(rates: tuple[float, ...] = (0.03, 0.04, 0.05),
                base: float = 0.03, deff: float = 1.4) -> pd.DataFrame:
    """n per group for a range of alternative conversion rates against ``base``."""
    rows = [{"p_control": base, "p_treated": r, "differential_pp": (r - base) * 100,
             "n_per_group": round(power_n(base, r, deff=deff))}
            for r in rates if r != base]
    return pd.DataFrame(rows)


def build_population(source: Path, tenure: pd.DataFrame) -> gpd.GeoDataFrame:
    """The §2 population: labelled + tenure-resolved parcels, with region ids attached."""
    src = gpd.read_parquet(Path(source) / "modeling_parcels.parquet")
    if "dept" not in src.columns:
        raise KeyError("source table has no `dept` column — was it built by "
                       "allperu.build_labels?")
    n0 = len(src)
    src = src.merge(tenure[["COD_PREDIO", "tenure", "reg_year"]], on="COD_PREDIO",
                    how="left")
    n_tenure = int(src["tenure"].notna().sum())
    src = src[src["tenure"].notna()].copy()
    src["_region"] = region_grid(src)
    print(f"population: {len(src):,} of {n0:,} labelled parcels carry tenure "
          f"({n_tenure / max(n0, 1):.1%})")
    print(pd.crosstab(src["label"], src["tenure"]).to_string())
    return src


def quotas(pop: gpd.GeoDataFrame, at_risk_per_tenure: int = TARGET_AT_RISK_PER_TENURE,
           n_perennial: int = TARGET_PERENNIAL,
           n_pasture: int = TARGET_PASTURE) -> pd.Series:
    """Target parcels per ``(label, tenure)`` cell, split across departments by sqrt share.

    At-risk cells get a flat per-tenure target (the whole point); the two control classes get
    a fixed total on the same sqrt rule, keeping small departments represented without
    Cajamarca and Ancash swamping the controls.
    """
    want = {}
    totals = {"ANNUAL": None, "PERENNIAL": n_perennial, "PASTURE_FALLOW": n_pasture}
    for label, total in totals.items():
        sub = pop[pop["label"] == label]
        for ten, grp in sub.groupby("tenure"):
            target = at_risk_per_tenure if label == "ANNUAL" else total // 2
            sizes = grp.groupby("dept").size()
            w = np.sqrt(sizes.astype(float))
            alloc = np.minimum((w / w.sum() * target).round().astype(int), sizes)
            for dept, n in alloc.items():
                want[(label, ten, dept)] = int(n)
    return pd.Series(want).rename("quota")


def draw(pop: gpd.GeoDataFrame, want: pd.Series, max_per_region: int = 220,
         seed: int = 42) -> gpd.GeoDataFrame:
    """Fill each stratum's quota by taking whole regions, capped, until it is met."""
    rng = np.random.default_rng(seed)
    picked = []
    for (label, ten, dept), quota in want.items():
        if quota <= 0:
            continue
        sub = pop[(pop["label"] == label) & (pop["tenure"] == ten)
                  & (pop["dept"] == dept)]
        if sub.empty:
            continue
        regions = sub.groupby("_region").size().sample(frac=1.0, random_state=seed)
        got, taken = 0, []
        for region, _ in regions.items():
            if got >= quota:
                break
            grp = sub[sub["_region"] == region]
            take = min(len(grp), max_per_region, quota - got)
            taken.append(grp.sample(take, random_state=int(rng.integers(1 << 31))))
            got += take
        if taken:
            picked.append(pd.concat(taken))
    out = gpd.GeoDataFrame(pd.concat(picked, ignore_index=True), crs=pop.crs)
    return out.drop_duplicates("COD_PREDIO")


def sample(source: Path, tenure_path: Path, at_risk_per_tenure: int = TARGET_AT_RISK_PER_TENURE,
           n_perennial: int = TARGET_PERENNIAL, n_pasture: int = TARGET_PASTURE,
           max_per_region: int = 220, seed: int = 42,
           save: bool = True) -> gpd.GeoDataFrame:
    """Draw the window sample and write it (with weights) into ``CC_PROC``."""
    tenure = pd.read_parquet(tenure_path)
    pop = build_population(source, tenure)
    want = quotas(pop, at_risk_per_tenure, n_perennial, n_pasture)
    out = draw(pop, want, max_per_region=max_per_region, seed=seed)

    # weight = stratum population / stratum sample, stratum = (dept, label, tenure).
    keys = ["dept", "label", "tenure"]
    pop_n = pop.groupby(keys, observed=True).size()
    smp_n = out.groupby(keys, observed=True).size()
    w = (pop_n / smp_n).rename("population_weight")
    out = out.merge(w, left_on=keys, right_index=True, how="left")
    out["population_weight"] = (out["population_weight"]
                                .replace([np.inf, -np.inf], np.nan).fillna(1.0))
    out = out.drop(columns="_region")

    print(f"\nsampled {len(out):,} parcels from {len(pop):,} "
          f"({out['population_weight'].sum():,.0f} expanded)")
    print(pd.crosstab(out["label"], out["tenure"]).to_string())
    print("\nby department:")
    print(pd.crosstab(out["dept"], [out["label"], out["tenure"]]).to_string())
    print(f"\ndesign effect from weights: "
          f"{_deff(out['population_weight'].to_numpy(float)):.2f}")

    if save:
        d = proc()
        out.to_parquet(d / "modeling_parcels.parquet", index=False)
        src_map = Path(source) / "label_map.json"
        if src_map.exists():
            shutil.copy(src_map, d / "label_map.json")
        with open(d / "window_sample_meta.json", "w") as f:
            json.dump({"source": str(source), "tenure": str(tenure_path),
                       "at_risk_per_tenure": at_risk_per_tenure,
                       "n_perennial": n_perennial, "n_pasture": n_pasture,
                       "max_per_region": max_per_region, "seed": seed,
                       "sampled_n": int(len(out)), "population_n": int(len(pop)),
                       "deff": _deff(out["population_weight"].to_numpy(float)),
                       "cells": {f"{k[0]}|{k[1]}": int(v)
                                 for k, v in out.groupby(["label", "tenure"]).size()
                                 .items()}}, f, indent=2)
        print(f"\nwrote {d / 'modeling_parcels.parquet'} + window_sample_meta.json")
    return out


def _deff(w: np.ndarray) -> float:
    """Kish design effect ``1 + CV(w)^2`` — how much the weights cost in effective n."""
    return float(1.0 + (w.std() / w.mean()) ** 2) if len(w) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--tenure", type=Path, required=True)
    ap.add_argument("--at-risk-per-tenure", type=int, default=TARGET_AT_RISK_PER_TENURE)
    ap.add_argument("--n-perennial", type=int, default=TARGET_PERENNIAL)
    ap.add_argument("--n-pasture", type=int, default=TARGET_PASTURE)
    ap.add_argument("--max-per-region", type=int, default=220)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    print(power_table().to_string(index=False), "\n")
    sample(a.source, a.tenure, at_risk_per_tenure=a.at_risk_per_tenure,
           n_perennial=a.n_perennial, n_pasture=a.n_pasture,
           max_per_region=a.max_per_region, seed=a.seed, save=not a.no_save)


if __name__ == "__main__":
    main()

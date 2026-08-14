"""Down-sample the all-Peru parcel table back to Piura scale.

The new data is ~10x Piura, and the brief is to keep the total modelling cost unchanged:
"aim to have in total as much data as was previously used ... as for the whole of the new
data". Every downstream cost — GEE extraction hours, feature-store size, training time —
scales with parcel count, so the sample is taken here, once, and everything after it runs at
the Piura budget.

**What the sample is optimising for.** The deliverable is a classifier that generalises
*spatially* (to departments it never trained on) and *temporally* (across a multi-year
panel). So the sample deliberately trades parcels-per-place for number-of-places:

1. **Whole 5 km regions are sampled, never scattered parcels.** Crops in Peru grow in
   single-crop blocks (~86 % of adjacent parcels share a crop), so a scattered sample would
   leave each region too sparse for the spatially-blocked split's buffer dead-zone to mean
   anything, and would inflate apparent difficulty by removing every parcel's neighbours.
2. **Departments are allocated by sqrt-proportional share, not proportional.** Cajamarca and
   Ancash hold over half the linked parcels between them; proportional allocation would make
   an "all-Peru" model that is mostly two departments, which is exactly the spatial
   generalisation this is meant to test. Square-root allocation is the standard compromise
   between proportional (efficient for a national total) and equal (efficient for
   between-department contrasts). Small departments also get a floor.
3. **Parcels per region are capped.** A cap converts "more parcels" into "more places" at
   fixed cost — the binding constraint for spatial generalisation.

Class balance is deliberately **not** forced: the perennial/annual/pasture prior is a real
property of Peruvian agriculture and re-weighting it here would corrupt any area share.
Sampling weights (stratum population / stratum sample) are written alongside so a population
quantity can still be recovered.

Run with::

    CC_PROC=data/processed/all_peru \\
      uv run python -m crop_classifier.allperu.sample --source data/processed/all_peru_full
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from crop_classifier.paths import proc

# Piura's 3-class modelling table, which this is sized against.
PIURA_N = 56_419

# One metric CRS for the whole country so region ids come from a single continuous grid.
# UTM 18S: Peru reaches ~6.4 deg either side of the 75W central meridian, a scale error
# under ~0.7 % — under 40 m on a 5 km region edge, which cannot move a parcel more than one
# cell and does not matter for a sampling stratum.
METRIC_CRS = 32718


def region_grid(gdf: gpd.GeoDataFrame, region_km: float = 5.0) -> pd.Series:
    """5 km grid cell id per parcel — the same construction ``splits.assign`` uses."""
    m = gdf.geometry.representative_point().to_crs(METRIC_CRS)
    size = region_km * 1000.0
    return pd.Series([f"r{int(x // size)}_{int(y // size)}"
                      for x, y in zip(m.x.values, m.y.values)], index=gdf.index)


def allocate(sizes: pd.Series, total: int, floor: int = 800) -> pd.Series:
    """Square-root-proportional allocation of ``total`` across departments, with a floor.

    ``floor`` is a minimum quota per department, capped by what it actually has — without it
    Callao (a few hundred parcels) and Moquegua would round to nothing and the model would
    have no exposure to them at all.
    """
    order = sizes.index                       # return in the caller's order, not sorted
    sizes = sizes.sort_values(ascending=False)
    quota = pd.Series(0, index=sizes.index, dtype=int)
    # start everyone at min(floor, available), then share the remainder by sqrt weight
    quota[:] = np.minimum(floor, sizes.values)
    remaining = total - int(quota.sum())
    if remaining > 0:
        headroom = (sizes - quota).clip(lower=0)
        w = np.sqrt(sizes.astype(float))
        w = w * (headroom > 0)
        for _ in range(20):                       # iterate: capped depts spill to the rest
            if w.sum() <= 0 or remaining <= 0:
                break
            add = np.minimum((w / w.sum() * remaining).round().astype(int), headroom)
            quota += add
            headroom = (sizes - quota).clip(lower=0)
            remaining = total - int(quota.sum())
            w = np.sqrt(sizes.astype(float)) * (headroom > 0)
    return quota.clip(upper=sizes).reindex(order)


def sample(source: Path, target_n: int = PIURA_N, region_km: float = 5.0,
           max_per_region: int = 220, floor: int = 800, seed: int = 42,
           save: bool = True) -> gpd.GeoDataFrame:
    """Sample whole regions per department until each department's quota is met."""
    rng = np.random.default_rng(seed)
    src = gpd.read_parquet(Path(source) / "modeling_parcels.parquet")
    if "dept" not in src.columns:
        raise KeyError("source table has no `dept` column — was it built by "
                       "allperu.build_labels?")
    src = src.copy()
    src["_region"] = region_grid(src, region_km)

    sizes = src.groupby("dept").size()
    quota = allocate(sizes, target_n, floor=floor)
    print(f"source {len(src):,} parcels in {src.dept.nunique()} departments; "
          f"target {target_n:,}")

    picked = []
    rows = []
    for dept, want in quota.items():
        sub = src[src.dept == dept]
        regions = sub.groupby("_region").size().sample(frac=1.0, random_state=seed)
        taken, got = [], 0
        for region, n in regions.items():
            if got >= want:
                break
            grp = sub[sub._region == region]
            take = min(len(grp), max_per_region, want - got)
            taken.append(grp.sample(take, random_state=int(rng.integers(1 << 31))))
            got += take
        got_df = pd.concat(taken) if taken else sub.iloc[:0]
        picked.append(got_df)
        rows.append({"dept": dept, "available": int(sizes[dept]), "quota": int(want),
                     "sampled": len(got_df), "regions_available": sub._region.nunique(),
                     "regions_sampled": got_df["_region"].nunique() if len(got_df) else 0})
        print(f"  {dept:14s} {len(got_df):>7,} of {int(sizes[dept]):>8,} "
              f"({got_df['_region'].nunique() if len(got_df) else 0:>4} of "
              f"{sub._region.nunique():>5} regions)")

    out = gpd.GeoDataFrame(pd.concat(picked, ignore_index=True), crs=src.crs)

    # Weight = stratum population / stratum sample, stratum = (dept, label). An area share
    # computed on this sample must be expanded; never report it raw.
    #
    # Deliberately NOT called `sample_weight`: `perennial/panel.py` computes a column of that
    # name for its own, different stratification (panel parcel -> modelling sample). The two
    # are successive stages of one design and must **multiply**, not collide — panel parcel
    # -> modelling sample -> national population. `build_panel` composes them.
    pop = src.groupby(["dept", "label"], observed=True).size()
    smp = out.groupby(["dept", "label"], observed=True).size()
    w = (pop / smp).rename("population_weight")
    out = out.merge(w, left_on=["dept", "label"], right_index=True, how="left")
    out["population_weight"] = (out["population_weight"]
                                .replace([np.inf, -np.inf], np.nan).fillna(1.0))
    out = out.drop(columns="_region")

    report = pd.DataFrame(rows)
    print(f"\nsampled {len(out):,} parcels "
          f"({len(out) / len(src):.1%} of the linked all-Peru set)")
    print("\nclass mix — source vs sample:")
    cmp = pd.DataFrame({"source": src.label.value_counts(normalize=True),
                        "sample": out.label.value_counts(normalize=True)}).round(4)
    print(cmp.to_string())
    print("\nlabel-year mix — source vs sample (top 12):")
    ycmp = pd.DataFrame({"source": src.year.value_counts(normalize=True),
                         "sample": out.year.value_counts(normalize=True)}
                        ).sort_index().round(4)
    print(ycmp.head(12).to_string())

    if save:
        d = proc()
        out.to_parquet(d / "modeling_parcels.parquet", index=False)
        report.to_csv(d / "sample_report.csv", index=False)
        # label_map.json belongs to the label build, which ran in the SOURCE workspace.
        # train.py/infer.py read it from the current one, so carry it across or every
        # downstream command fails with FileNotFoundError.
        src_map = Path(source) / "label_map.json"
        if src_map.exists():
            shutil.copy(src_map, d / "label_map.json")
        with open(d / "sample_meta.json", "w") as f:
            json.dump({"source": str(source), "target_n": target_n,
                       "sampled_n": len(out), "region_km": region_km,
                       "max_per_region": max_per_region, "floor": floor, "seed": seed,
                       "source_n": len(src),
                       "class_mix": out.label.value_counts(normalize=True).round(4).to_dict()
                       }, f, indent=2)
        print(f"\nwrote {d / 'modeling_parcels.parquet'} + sample_report.csv/sample_meta.json")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, required=True,
                    help="workspace holding the FULL all-Peru modeling_parcels.parquet")
    ap.add_argument("--target-n", type=int, default=PIURA_N)
    ap.add_argument("--max-per-region", type=int, default=220)
    ap.add_argument("--floor", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    sample(a.source, target_n=a.target_n, max_per_region=a.max_per_region,
           floor=a.floor, seed=a.seed, save=not a.no_save)


if __name__ == "__main__":
    main()

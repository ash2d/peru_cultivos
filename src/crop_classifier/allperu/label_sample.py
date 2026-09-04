"""Draw the S2 endpoint-labelling sample (docs/s2_labelling/plan.md).

One round, 1,000 parcels, no second pass — so everything here must be defensible before the
draw, not corrected after. Four things, all persisted:

1. :func:`build_universe` — the eligible universe over the 14 linkable departments (§1):
   geometry + ``COD_PREDIO`` + ``area_ha >= 0.15``. The Esri imagery filter is applied
   later against the probe (§4) — it costs a network round trip per parcel.
2. :func:`sensitivity_sweep` — the class mix a 2:1:1 declared-class allocation realises,
   swept over unmeasured declared-``ANNUAL``->perennial conversion rates (§2.1).
3. :func:`draw` — the stratified draw: department (sqrt-proportional, floor), declared
   class (2:1:1), a 2-per-5 km-region cap and a 40 %-per-declared-crop cap per cell.
4. weights — ``weight = N_h / n_h`` on the eligible population, written at draw time (§2.2).

⚠️ The declared PETT class is a ~1998 declaration. It balances the draw only, is never shown
to the labeller, and is not the label.

Run::

    CC_PROC=data/processed/all_peru uv run python -m crop_classifier.cli \\
        allperu label-universe --source data/processed/all_peru_full
    CC_PROC=data/processed/all_peru uv run python -m crop_classifier.cli \\
        allperu label-draw
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from crop_classifier.allperu.build_labels import clean_geometry
from crop_classifier.paths import labels_dir

# One metric CRS for the whole country, as in allperu.sample.METRIC_CRS.
METRIC_CRS = 32718
REGION_KM = 5.0

# §1 eligibility: below ~0.15 ha a 10 m inward buffer leaves too few S2 pixels to summarise.
MIN_AREA_HA = 0.15

# §2.1 allocation
TOTAL = 1_000
DEPT_FLOOR = 55
CLASS_RATIO = {"PERENNIAL": 2, "ANNUAL": 1, "PASTURE_FALLOW": 1}
MAX_PER_REGION = 2
MAX_CROP_SHARE = 0.40

# §11 step 3 pilot, drawn disjointly from the main 1,000.
PILOT_N = 120
# §2.1 double-labelled overlap, drawn from within the 1,000 for Cohen's kappa (§9).
OVERLAP_N = 100

F_UNIVERSE = "label_universe.parquet"
F_SAMPLE = "label_sample.parquet"
F_META = "label_sample_meta.json"
F_ALLOC = "label_allocation.csv"
F_SWEEP = "label_class_mix_sweep.csv"


def _d() -> Path:
    return labels_dir()


def out_dir() -> Path:
    d = _d()
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- 1. Eligible universe ---
def region_grid(gdf: gpd.GeoDataFrame, region_km: float = REGION_KM) -> pd.Series:
    """5 km grid cell id per parcel — the same construction ``splits.assign`` uses."""
    m = gdf.geometry.representative_point().to_crs(METRIC_CRS)
    size = region_km * 1000.0
    return pd.Series([f"r{int(x // size)}_{int(y // size)}"
                      for x, y in zip(m.x.values, m.y.values)], index=gdf.index)


def build_universe(source: Path, min_area_ha: float = MIN_AREA_HA,
                   save: bool = True) -> gpd.GeoDataFrame:
    """The eligible universe over the 14 linkable departments (§1, filters 1-3).

    ``source`` holds the full national ``modeling_parcels.parquet``
    (``data/processed/all_peru_full``), not the 56 k modelling sample: the campaign is drawn
    from the population. The department list is derived from the table, not hard-coded.
    """
    src = gpd.read_parquet(Path(source) / "modeling_parcels.parquet")
    n0 = len(src)
    rows = [{"filter": "start", "n": n0}]

    src = src[src["COD_PREDIO"].notna() & (src["COD_PREDIO"].astype(str) != "")]
    rows.append({"filter": "cod_predio non-null", "n": len(src)})

    src = src.copy()
    src["geometry"] = clean_geometry(src.geometry)
    ok = (src.geometry.notna() & src.geometry.is_valid & ~src.geometry.is_empty
          & src.geometry.geom_type.isin(["Polygon", "MultiPolygon"]))
    src = src[ok]
    rows.append({"filter": "valid 2D areal geometry", "n": len(src)})

    src = src[src["area_ha"] >= min_area_ha]
    rows.append({"filter": f"area_ha >= {min_area_ha}", "n": len(src)})

    src = src.reset_index(drop=True)
    src["region_id"] = region_grid(src)
    src["centroid_lon"] = src.geometry.representative_point().x
    src["centroid_lat"] = src.geometry.representative_point().y

    filt = pd.DataFrame(rows)
    print(filt.to_string(index=False))
    print(f"\neligible universe: {len(src):,} parcels "
          f"({len(src) / n0:.1%} of the linked national table), "
          f"{src.dept.nunique()} departments, {src.region_id.nunique():,} regions")
    print(src.groupby(["dept", "label"]).size().unstack(fill_value=0).to_string())

    if save:
        d = out_dir()
        src.to_parquet(d / F_UNIVERSE, index=False)
        filt.to_csv(d / "label_universe_filters.csv", index=False)
        print(f"\nwrote {d / F_UNIVERSE}")
    return src


# --- 2. Allocation ---
def allocate_departments(sizes: pd.Series, total: int = TOTAL,
                         floor: int = DEPT_FLOOR) -> pd.Series:
    """Sqrt-proportional allocation across departments with a per-department floor.

    Same shape as ``allperu.sample.allocate`` (floor first, remainder by sqrt weight), but
    re-implemented here: this one allocates a 1,000-parcel labelling budget and must hit the
    total exactly — a 50-parcel overshoot is 5 % of the campaign.
    """
    order = sizes.index
    sizes = sizes.sort_values(ascending=False)
    quota = pd.Series(np.minimum(floor, sizes.values), index=sizes.index, dtype=int)
    remaining = total - int(quota.sum())
    if remaining > 0:
        headroom = (sizes - quota).clip(lower=0)
        w = np.sqrt(sizes.astype(float)) * (headroom > 0)
        share = (w / w.sum() * remaining) if w.sum() > 0 else w * 0
        add = np.minimum(np.floor(share).astype(int), headroom)
        quota += add
        # rounding remainder, largest-fraction-first, so the total is exact
        frac = (share - np.floor(share)).sort_values(ascending=False)
        for dept in frac.index:
            if int(quota.sum()) >= total:
                break
            if quota[dept] < sizes[dept]:
                quota[dept] += 1
    elif remaining < 0:
        # more departments x floor than budget: trim the smallest departments first
        for dept in sizes.sort_values().index:
            while int(quota.sum()) > total and quota[dept] > 1:
                quota[dept] -= 1
    return quota.clip(upper=sizes).reindex(order)


def allocate_classes(n: int, ratio: dict[str, int] | None = None) -> dict[str, int]:
    """Split a department quota across declared classes at 2:1:1 (§2.1), summing to ``n``.

    The over-weighted class rounds up, then the remainder is shared largest-remainder. Plain
    largest-remainder over all three would give 55 -> 27/14/14, sharing the odd parcel away
    from the class the tilt exists for. So the tilt takes the rounding: 55 -> 28/14/13,
    110 -> 55/28/27 (the plan's worked examples).
    """
    ratio = ratio or CLASS_RATIO
    tot = sum(ratio.values())
    exact = {k: n * v / tot for k, v in ratio.items()}
    order = sorted(ratio, key=lambda k: (-ratio[k], list(ratio).index(k)))
    lead = order[0]
    out = {lead: min(n, int(np.floor(exact[lead] + 0.5)))}
    left = n - out[lead]
    rest = order[1:]
    for k in rest:
        out[k] = int(np.floor(exact[k]))
    left -= sum(out[k] for k in rest)
    for k in sorted(rest, key=lambda k: (-(exact[k] % 1), list(ratio).index(k))):
        if left <= 0:
            break
        out[k] += 1
        left -= 1
    if left > 0:                      # tiny n: everything floored to zero
        out[lead] += left
    return out


def allocation_table(sizes_dept_class: pd.DataFrame, total: int = TOTAL,
                     floor: int = DEPT_FLOOR) -> pd.DataFrame:
    """``(dept, declared_class) -> target``, with the cell population alongside.

    A cell short of its target spills back to the other classes of the same department, so
    the department total is preserved (the department is the stratum LODO needs; the class
    ratio is only a tilt).
    """
    dept_quota = allocate_departments(sizes_dept_class.sum(axis=1), total, floor)
    rows = []
    for dept, want in dept_quota.items():
        pop = sizes_dept_class.loc[dept]
        target = allocate_classes(int(want))
        # cap each class at its population, redistribute the shortfall
        for _ in range(3):
            short = sum(max(0, target[c] - int(pop.get(c, 0))) for c in target)
            if short == 0:
                break
            target = {c: min(target[c], int(pop.get(c, 0))) for c in target}
            head = {c: int(pop.get(c, 0)) - target[c] for c in target}
            live = [c for c in target if head[c] > 0]
            if not live:
                break
            for i in range(short):
                c = live[i % len(live)]
                if head[c] > 0:
                    target[c] += 1
                    head[c] -= 1
        for c, t in target.items():
            rows.append({"dept": dept, "declared_class": c,
                         "n_population": int(pop.get(c, 0)), "n_target": int(t)})
    return pd.DataFrame(rows)


# --- 2.1 sensitivity of the realised class mix to unmeasured conversion (persisted) ---
def sensitivity_sweep(rates=(0.05, 0.15, 0.30),
                      ratio: dict[str, int] | None = None) -> pd.DataFrame:
    """Realised observed-class mix under a range of declared->observed transition rates.

    The allocation stratifies on a 1996-2006 declaration while the labels are 2019+, so the
    realised mix depends on unmeasured transition rates. The question is whether 2:1:1
    survives the plausible range, so this sweeps the genuinely unknown rate — declared
    ``ANNUAL`` -> observed perennial — and fixes the two stable ones:

    * declared ``PERENNIAL`` stays perennial at 0.85 (conversion runs one way), remainder
      to OTHER;
    * declared ``PASTURE_FALLOW`` converts to perennial at a third of the annual rate, and
      to annual at 0.09 (fallow ground getting sown is the commonest 20-year outcome, and
      dropping that flow is itself a less realistic assumption).

    Persisted as ``label_class_mix_sweep.csv`` so the allocation choice is on the record
    with its sensitivity.
    """
    ratio = ratio or CLASS_RATIO
    tot = sum(ratio.values())
    share = {k: v / tot for k, v in ratio.items()}
    per_stays, past_to_annual = 0.85, 0.09
    rows = []
    for r in rates:
        past_conv = r / 3.0
        obs = {
            "PERENNIAL": (share["PERENNIAL"] * per_stays
                          + share["ANNUAL"] * r
                          + share["PASTURE_FALLOW"] * past_conv),
            "ANNUAL": (share["ANNUAL"] * (1 - r)
                       + share["PASTURE_FALLOW"] * past_to_annual),
            "OTHER": (share["PERENNIAL"] * (1 - per_stays)
                      + share["PASTURE_FALLOW"] * (1 - past_conv - past_to_annual)),
        }
        rows.append({"annual_to_perennial": r, **{k: round(v, 4) for k, v in obs.items()}})
    return pd.DataFrame(rows)


# --- 2b. Candidate pool for the Esri probe (§4 — order matters) ---
def candidate_pool(universe: gpd.GeoDataFrame, total: int = TOTAL + PILOT_N,
                   factor: float = 2.5, floor: int = DEPT_FLOOR,
                   seed: int = 20260812, save: bool = True) -> gpd.GeoDataFrame:
    """Per (dept x class) cell, a random ``factor`` x over-draw to probe for Esri dates.

    Draw the pool, probe it, then draw the sample from the eligible sub-population. Filtering
    a finished draw would distort inclusion probabilities; drawing from the eligible
    sub-population does not, and the per-stratum eligible fraction is what lets §2.2's
    weights refer to the eligible population.

    No region or crop cap here: those are properties of the final draw, and applying them to
    the pool would shrink the set it can choose from.
    """
    rng = np.random.default_rng(seed)
    sizes = universe.groupby(["dept", "label"]).size().unstack(fill_value=0)
    for c in CLASS_RATIO:
        if c not in sizes.columns:
            sizes[c] = 0
    alloc = allocation_table(sizes[list(CLASS_RATIO)], total, floor=floor)
    out = []
    for _, r in alloc.iterrows():
        cell = universe[(universe.dept == r.dept) & (universe.label == r.declared_class)]
        want = min(len(cell), int(np.ceil(r.n_target * factor)))
        if want:
            out.append(cell.sample(want, random_state=int(rng.integers(1 << 31))))
    pool = gpd.GeoDataFrame(pd.concat(out), crs=universe.crs)
    print(f"candidate pool: {len(pool):,} parcels ({factor}x a {total}-parcel allocation) "
          f"across {pool.dept.nunique()} departments")
    if save:
        d = out_dir()
        pool.to_parquet(d / "label_candidate_pool.parquet", index=False)
        alloc.to_csv(d / "label_allocation_pool.csv", index=False)
        print(f"wrote {d / 'label_candidate_pool.parquet'}")
    return pool


def supplement_pool(universe: gpd.GeoDataFrame, already: set[str],
                    depts: list[str], total: int = TOTAL + PILOT_N,
                    factor: float = 6.0, floor: int = DEPT_FLOOR,
                    seed: int = 20260813, save: bool = True) -> gpd.GeoDataFrame:
    """Enlarge the candidate pool for departments the first pool could not fill.

    Two things ran the small departments out at 2.5x, neither a population limit: Pasco's
    Esri imagery is 40 % eligible, and Huancavelica's parcels sit in 25 of 44 regions so the
    2-per-region cap bites first.

    A second simple random sample from the same strata, not a targeted top-up: drawing
    preferentially from unused regions would break the weights. Two SRS draws without
    replacement from one stratum pool exactly.
    """
    rng = np.random.default_rng(seed)
    sizes = universe.groupby(["dept", "label"]).size().unstack(fill_value=0)
    for c in CLASS_RATIO:
        if c not in sizes.columns:
            sizes[c] = 0
    alloc = allocation_table(sizes[list(CLASS_RATIO)], total, floor=floor)
    rest = universe[~universe["COD_PREDIO"].astype(str).isin(already)]
    out = []
    for _, r in alloc[alloc.dept.isin(depts)].iterrows():
        cell = rest[(rest.dept == r.dept) & (rest.label == r.declared_class)]
        have = int(((universe.dept == r.dept) & (universe.label == r.declared_class)
                    & universe["COD_PREDIO"].astype(str).isin(already)).sum())
        want = max(0, int(np.ceil(r.n_target * factor)) - have)
        want = min(want, len(cell))
        if want:
            out.append(cell.sample(want, random_state=int(rng.integers(1 << 31))))
    pool = gpd.GeoDataFrame(pd.concat(out), crs=universe.crs) if out else universe.iloc[:0]
    print(f"supplementary pool: {len(pool):,} further parcels in "
          f"{pool.dept.nunique() if len(pool) else 0} departments (to {factor}x target)")
    if save and len(pool):
        pool.to_parquet(out_dir() / "label_candidate_pool_supp.parquet", index=False)
    return pool


def eligible_population(universe: pd.DataFrame, probed: pd.DataFrame,
                        elig: pd.Series) -> pd.DataFrame:
    """Estimate ``N_h`` on the eligible population from the probed candidate pool.

    Only the pool was probed, so a stratum's eligible population is estimated as
    ``N_h(universe) x eligible_rate_h(pool)`` — unbiased because the pool is an SRS within
    the stratum. Returned with the ingredients so the number can be audited.
    """
    p = probed.copy()
    p["_elig"] = elig.values
    rate = (p.groupby(["dept", "declared_class"])["_elig"]
            .agg(["mean", "size"]).rename(columns={"mean": "elig_rate",
                                                   "size": "n_probed"}))
    pop = (universe.groupby(["dept", "label"]).size().rename("N_universe")
           .rename_axis(["dept", "declared_class"]))
    out = pd.concat([pop, rate], axis=1)
    out["elig_rate"] = out["elig_rate"].fillna(0.0)
    out["N_eligible"] = (out["N_universe"] * out["elig_rate"]).round().astype("Int64")
    return out.reset_index()


# --- 3. The draw ---
def _primary_crop(crop_set: pd.Series) -> pd.Series:
    """First declared crop token — the unit the 40 % crop cap is applied to."""
    return (crop_set.fillna("").astype(str).str.split("|").str[0]
            .str.split(",").str[0].str.strip().replace("", "UNKNOWN"))


def _draw_cell(cell: pd.DataFrame, want: int, rng: np.random.Generator,
               used_regions: dict[str, int], max_per_region: int,
               max_crop_share: float) -> pd.DataFrame:
    """Take ``want`` parcels from one (dept x class) cell under both caps.

    Region cap is global across the draw (``used_regions`` is shared), so two classes cannot
    both spend one region's budget. The crop cap is per-cell and is relaxed, not enforced
    blindly: if the population cannot fill the quota under it, it is dropped for the
    remainder and the relaxation is reported.
    """
    cell = cell.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
    # `ceil`, not `floor`: with three crops and want=7 a floored cap of 2 is infeasible by
    # integer arithmetic alone, so the relaxation would fire on non-concentrated cells.
    crop_cap = max(1, int(np.ceil(want * max_crop_share)))
    taken: list[int] = []
    crop_count: dict[str, int] = {}

    for relax_crop in (False, True):
        for i, row in cell.iterrows():
            if len(taken) >= want:
                break
            if i in taken:
                continue
            reg = row["region_id"]
            if used_regions.get(reg, 0) >= max_per_region:
                continue
            crop = row["_crop"]
            if not relax_crop and crop_count.get(crop, 0) >= crop_cap:
                continue
            taken.append(i)
            used_regions[reg] = used_regions.get(reg, 0) + 1
            crop_count[crop] = crop_count.get(crop, 0) + 1
        if len(taken) >= want:
            break
    return cell.loc[taken]


def draw(universe: gpd.GeoDataFrame | None = None, total: int = TOTAL,
         floor: int = DEPT_FLOOR, pilot_n: int = PILOT_N, overlap_n: int = OVERLAP_N,
         max_per_region: int = MAX_PER_REGION, max_crop_share: float = MAX_CROP_SHARE,
         pop_eligible: pd.DataFrame | None = None,
         alloc_sizes: pd.DataFrame | None = None,
         seed: int = 20260812, save: bool = True) -> gpd.GeoDataFrame:
    """Draw the pilot + the main sample, with weights, from the eligible universe.

    Order is the plan's §4: the universe passed in must already be filtered to parcels whose
    Esri imagery is <=1.2 m and dated >=2019. Filtering after a draw would distort inclusion
    probabilities; drawing from the eligible sub-population does not.

    The pilot (§11 step 3) is drawn first and disjointly, so the codebook can be frozen
    against parcels outside the 1,000.
    """
    d = out_dir()
    if universe is None:
        universe = gpd.read_parquet(d / F_UNIVERSE)
    u = universe.copy()
    u["_crop"] = _primary_crop(u.get("crop_set", pd.Series("", index=u.index)))
    rng = np.random.default_rng(seed)

    # The allocation is a property of the population. ``universe`` here is normally the
    # eligible candidate pool (~2.5x a per-department quota), which would allocate itself
    # almost uniformly, so the design comes from ``alloc_sizes`` and the pool only decides
    # what is available.
    if alloc_sizes is None:
        alloc_sizes = u.groupby(["dept", "label"]).size().unstack(fill_value=0)
    sizes = alloc_sizes.copy()
    for c in CLASS_RATIO:
        if c not in sizes.columns:
            sizes[c] = 0
    sizes = sizes[list(CLASS_RATIO)]

    # ---- pilot first, so the main draw cannot overlap it ----
    pilot = pd.DataFrame()
    if pilot_n:
        alloc_p = allocation_table(sizes, pilot_n, floor=max(1, pilot_n // 14))
        pilot = _draw_from_allocation(u, alloc_p, rng, max_per_region, max_crop_share)
        pilot["batch"] = "pilot"
        u = u.drop(index=pilot.index)

    # ---- main draw ----
    alloc = allocation_table(sizes, total, floor=floor)
    main = _draw_from_allocation(u, alloc, rng, max_per_region, max_crop_share,
                                 top_up=True)
    main["batch"] = "main"

    out = pd.concat([pilot, main]) if len(pilot) else main
    out = gpd.GeoDataFrame(out.reset_index(drop=True), geometry="geometry",
                           crs=universe.crs)
    out["stratum"] = out["dept"] + "|" + out["label"]
    out = out.rename(columns={"label": "declared_class"})

    # ---- weights: eligible stratum population / stratum sample (§2.2) ----
    # `pop_eligible` is the probe-based estimate of the eligible population (§4). Without it
    # the weights would expand to the candidate pool, not a population — so fall back to the
    # passed universe and say so.
    if pop_eligible is not None:
        pop = (pop_eligible.set_index(["dept", "declared_class"])["N_eligible"]
               .astype("float64").rename("N_h"))
    else:
        print("⚠️ no eligible-population estimate given — weights expand to the frame "
              "passed in, not to the eligible national population")
        pop = universe.groupby(["dept", "label"]).size().rename("N_h") \
                      .rename_axis(["dept", "declared_class"])
    smp = out[out.batch == "main"].groupby(["dept", "declared_class"]).size().rename("n_h")
    wt = pd.concat([pop, smp], axis=1)
    wt["weight"] = (wt["N_h"] / wt["n_h"]).replace([np.inf, -np.inf], np.nan)
    out = out.merge(wt[["N_h", "n_h", "weight"]], left_on=["dept", "declared_class"],
                    right_index=True, how="left")
    out.loc[out.batch == "pilot", "weight"] = np.nan   # the pilot is not a sample

    # ---- kappa overlap: 100 of the main draw, spread across departments ----
    out["overlap"] = False  # kappa overlap: 100 of the main draw, spread across departments
    if overlap_n:
        m = out.index[out.batch == "main"]
        per_dept = out.loc[m].groupby("dept").apply(
            lambda g: g.sample(min(len(g), max(1, round(overlap_n / out.loc[m, "dept"]
                                                        .nunique()))),
                               random_state=int(rng.integers(1 << 31))).index.tolist(),
            include_groups=False)
        picked = [i for lst in per_dept for i in lst][:overlap_n]
        if len(picked) < overlap_n:
            rest = [i for i in m if i not in set(picked)]
            picked += list(pd.Index(rest).to_series()
                           .sample(overlap_n - len(picked),
                                   random_state=int(rng.integers(1 << 31))))
        out.loc[picked, "overlap"] = True

    out["item_id"] = [f"{b[0].upper()}{i:05d}" for i, b in enumerate(out["batch"])]
    out = out.drop(columns=["_crop"], errors="ignore")

    _report(out, alloc, universe, max_per_region, max_crop_share, pop_eligible)

    if save:
        # ⛔ `label_sample.parquet` is the key every returned label is joined through. A
        # second campaign must go to its own directory, never over this one.
        if (d / F_SAMPLE).exists():
            raise SystemExit(
                f"{d / F_SAMPLE} already exists — a draw would replace the sample that "
                f"the existing labels are keyed to.\n"
                f"For a NEW round, point CC_LABELS at a new directory first:\n"
                f"  CC_LABELS={d.parent / (d.name + '_round2')} uv run cc ... campaign draw\n"
                f"docs/howto/04_label_and_train.md")
        out.to_parquet(d / F_SAMPLE, index=False)
        alloc.to_csv(d / F_ALLOC, index=False)
        sweep = sensitivity_sweep()
        sweep.to_csv(d / F_SWEEP, index=False)
        meta = {
            "seed": seed, "total_target": total, "n_main": int((out.batch == "main").sum()),
            "n_pilot": int((out.batch == "pilot").sum()),
            "n_overlap": int(out.overlap.sum()),
            "dept_floor": floor, "class_ratio": CLASS_RATIO,
            "max_per_region": max_per_region, "max_crop_share": max_crop_share,
            "min_area_ha": MIN_AREA_HA, "region_km": REGION_KM,
            "n_universe_eligible": int(len(universe)),
            "n_regions_used": int(out.region_id.nunique()),
            "class_mix_sweep": sweep.to_dict("records"),
            "declared_mix_main": (out[out.batch == "main"].declared_class
                                  .value_counts(normalize=True).round(4).to_dict()),
        }
        with open(d / F_META, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\nwrote {d / F_SAMPLE}, {F_ALLOC}, {F_SWEEP}, {F_META}")
    return out


# --- 4. Frozen train/test split (§3) ---
def assign_split(sample: gpd.GeoDataFrame | None = None,
                 config_path: Path | None = None, save: bool = True) -> gpd.GeoDataFrame:
    """Freeze the split on the drawn sample, before a single parcel is labelled.

    Deliberately not ``splits.assign``: that reads and writes
    ``proc()/modeling_parcels.parquet`` (here the completed national table) and would
    overwrite it. The primitives are reused (autocorrelation audit, balanced test draw,
    buffered dead-zones) so the policy matches every other split; only the file differs.

    The unit is the 5 km ``region_id``, and the test set is drawn stratified on
    ``dept x declared_class`` so every department appears on both sides.
    """
    from crop_classifier import splits as SP

    d = out_dir()
    if sample is None:
        sample = gpd.read_parquet(d / F_SAMPLE)
    cfg = SP.load_config(config_path or SP.CONFIG_DIR / "split_s2labels.yaml")

    df = sample.copy()
    # `pick_test_units` and the audit key on `label`/`label_id`. Before labelling the only
    # class available is the declared one — right for balancing a draw, wrong to show a
    # labeller — so it is aliased privately here and dropped again below.
    df["label"] = df["declared_class"]
    codes = {c: i for i, c in enumerate(sorted(df["label"].unique()))}
    df["label_id"] = df["label"].map(codes).astype(int)

    m = df.geometry.representative_point().to_crs(cfg.get("metric_crs", 32718))
    xy = np.c_[m.x.values, m.y.values]
    size = cfg["block_km"] * 1000.0
    df["block_id"] = [f"{int(x // size)}_{int(y // size)}" for x, y in xy]
    rsize = cfg["region_km"] * 1000.0
    df["region_id"] = [f"r{int(x // rsize)}_{int(y // rsize)}" for x, y in xy]

    audit = SP.autocorrelation_audit(xy, df["label_id"].values, cfg)
    print(f"autocorrelation: baseline {audit['baseline_agreement']:.3f}, "
          f"decorrelation range {audit['decorrelation_range_m']} m")

    test_regions = SP.pick_test_units(df, "region_id", cfg["test_frac"], cfg["seed"],
                                      balance_cols=cfg.get("balance_test_on"),
                                      n_candidates=int(cfg.get("n_test_candidates", 1)))
    df["split"] = np.where(df["region_id"].isin(test_regions), "test", "trainval")
    # the pilot is not part of the evaluated sample
    df.loc[df.batch == "pilot", "split"] = "pilot"

    tv = df[df["split"] == "trainval"]
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=cfg["n_folds"], shuffle=True,
                                random_state=cfg["seed"])
    df["fold"] = -1
    for k, (_, val_idx) in enumerate(sgkf.split(tv, tv["label_id"],
                                                groups=tv["region_id"])):
        df.loc[tv.index[val_idx], "fold"] = k

    is_test = (df["split"] == "test").values
    is_tv = (df["split"] == "trainval").values
    df["buffer_excl_test"] = SP.buffer_exclusions(xy, is_test, is_tv, cfg["buffer_m"])
    for k in range(cfg["n_folds"]):
        held = (df["fold"] == k).values
        df[f"buffer_excl_fold{k}"] = SP.buffer_exclusions(xy, held, is_tv & ~held,
                                                          cfg["buffer_m"])

    n_bt = int(df["buffer_excl_test"].sum())
    frac = n_bt / max(1, int(is_tv.sum()))
    print(f"\ntest: {int(is_test.sum()):,} parcels in {len(test_regions):,} regions "
          f"({is_test.mean():.1%} of the sample)")
    print(f"buffer {cfg['buffer_m']} m sterilises {n_bt:,} of {int(is_tv.sum()):,} "
          f"trainval parcels ({frac:.1%})"
          + ("  ⚠️ >10 % — consider dropping to 1500 m" if frac > 0.10 else ""))
    print("fold sizes:", df[df.fold >= 0].fold.value_counts().sort_index().tolist())
    print("\ntest/trainval composition by department:")
    print(pd.crosstab(df.loc[df.split != "pilot", "dept"],
                      df.loc[df.split != "pilot", "split"]).to_string())
    print("\nby declared class:")
    print(pd.crosstab(df.loc[df.split != "pilot", "declared_class"],
                      df.loc[df.split != "pilot", "split"]).to_string())

    df = df.drop(columns=["label", "label_id"])
    if save:
        df.to_parquet(d / F_SAMPLE, index=False)
        meta = {"config": cfg, "audit": audit,
                "n_test_regions": len(test_regions),
                "n_regions": int(df.region_id.nunique()),
                "buffer_excluded_test": n_bt,
                "buffer_excluded_frac": round(frac, 4),
                "split_counts": df["split"].value_counts().to_dict(),
                "frozen": "before labelling — do not redraw"}
        with open(d / "splits_meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"\nfroze the split into {d / F_SAMPLE} and splits_meta.json")
    return df


def _draw_from_allocation(u: pd.DataFrame, alloc: pd.DataFrame,
                          rng: np.random.Generator, max_per_region: int,
                          max_crop_share: float,
                          top_up: bool = False) -> pd.DataFrame:
    """Draw every cell; optionally top a department back up to its designed total.

    A cell can come up short because the eligible pool ran out (Piura's perennials are 32 %
    eligible) or the region cap bit. ``top_up`` re-spends that shortfall on the same
    department's other classes (the department is the stratum LODO needs). Departments are
    never topped up from each other — that would undo the sqrt-proportional allocation.
    """
    used_regions: dict[str, int] = {}
    picked, got = [], {}
    for _, r in alloc.iterrows():
        cell = u[(u.dept == r.dept) & (u.label == r.declared_class)]
        if cell.empty or r.n_target <= 0:
            got[(r.dept, r.declared_class)] = pd.DataFrame()
            continue
        take = _draw_cell(cell, int(r.n_target), rng, used_regions,
                          max_per_region, max_crop_share)
        got[(r.dept, r.declared_class)] = take
        picked.append(take)

    if top_up:
        chosen = set().union(*[set(t.index) for t in got.values() if len(t)]) \
            if picked else set()
        for dept, grp in alloc.groupby("dept"):
            short = int(sum(max(0, int(r.n_target) - len(got.get((r.dept,
                                                                  r.declared_class),
                                                                 [])))
                            for _, r in grp.iterrows()))
            if short <= 0:
                continue
            rest = u[(u.dept == dept) & ~u.index.isin(chosen)]
            if rest.empty:
                continue
            extra = _draw_cell(rest, short, rng, used_regions, max_per_region,
                               max_crop_share)
            if len(extra):
                picked.append(extra)
                chosen |= set(extra.index)
                print(f"  top-up {dept}: +{len(extra)} of {short} short")
    return pd.concat(picked) if picked else u.iloc[:0]


def _report(out: pd.DataFrame, alloc: pd.DataFrame, universe: pd.DataFrame,
            max_per_region: int, max_crop_share: float,
            pop_eligible: pd.DataFrame | None = None) -> None:
    main = out[out.batch == "main"]
    print(f"\ndrawn: {len(main):,} main + {int((out.batch == 'pilot').sum())} pilot "
          f"+ {int(out.overlap.sum())} double-labelled "
          f"= {len(main) + int((out.batch == 'pilot').sum()) + int(out.overlap.sum()):,} "
          f"parcel-labellings")
    print("\nper department (main draw):")
    tab = (main.groupby(["dept", "declared_class"]).size().unstack(fill_value=0))
    tab["total"] = tab.sum(axis=1)
    tab["regions"] = main.groupby("dept")["region_id"].nunique()
    print(tab.to_string())
    short = alloc.merge(main.groupby(["dept", "declared_class"]).size()
                        .rename("n_drawn"), left_on=["dept", "declared_class"],
                        right_index=True, how="left").fillna({"n_drawn": 0})
    miss = short[short.n_drawn < short.n_target]
    if len(miss):
        print(f"\n⚠️ {len(miss)} cells short of target (population or region cap):")
        print(miss.to_string(index=False))
    reg = main.groupby("region_id").size()
    print(f"\nregion cap {max_per_region}: max realised {reg.max()}, "
          f"{len(reg):,} distinct regions, {reg.mean():.2f} parcels/region")
    if "crop_set" in main.columns:
        cshare = (main.assign(_c=_primary_crop(main.crop_set))
                  .groupby(["dept", "declared_class"])["_c"]
                  .apply(lambda s: s.value_counts(normalize=True).iloc[0]))
        over = cshare[cshare > max_crop_share + 1e-9]
        print(f"crop cap {max_crop_share:.0%}: {len(over)} of {len(cshare)} cells exceed "
              f"it (population-limited), max share {cshare.max():.0%}")
    print("\nweight check — expanded sample vs eligible population per class:")
    exp = main.groupby("declared_class")["weight"].sum()
    if pop_eligible is not None:
        pop = pop_eligible.groupby("declared_class")["N_eligible"].sum()
    else:
        pop = universe.groupby("label").size()
    print(pd.DataFrame({"expanded": exp.round(0), "population": pop}).to_string())

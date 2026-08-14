"""The multi-year panel: annual inference over every parcel, every year (plan §7).

This is the phase that turns a classifier into the deliverable. It is also where the bugs
live, because the existing extraction assumes **one year per parcel**. The three landmines
(plan §7.0) are handled as follows:

1. ``assemble`` grouped on ``(COD_PREDIO, doy)`` with no year -> fixed upstream with a
   ``years`` filter, a per-year ``out_dir`` and ``assert_one_year_per_parcel``
   (``tests/test_assemble_years.py``);
2. ``run_coverage`` dedupes on ``COD_PREDIO`` and writes one file -> called **once per
   year** with an explicit ``out=``;
3. ``run_pixels`` hard-coded the shared feature dir -> now takes ``feat_dir``; the panel
   writes to ``FEAT/"panel"`` so the audited training store stays immutable.

Cost: full inference over ~56 k parcels x 35 years is ~110 h of GEE. The panel is
therefore a **stratified sample** (D6) whose size is chosen from a measured rate
(``timing_probe``), with sampling weights so area shares expand to the population.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from crop_classifier.features import assemble as asm
from crop_classifier.features import landsat_gee as lg
from crop_classifier.paths import feat, proc


def panel_feat() -> Path:
    """Panel pixel store, a ``panel/`` subdirectory of the current feature store.

    Call-time, not a constant, so ``CC_FEAT`` selects it (paths.py). Keeping the panel in
    its own subdirectory is what stops a 28-year experimental extraction from mutating the
    audited single-year training store.
    """
    p = feat() / "panel"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---- mission policy: TM/ETM+ only, no OLI (user decision, 2026-08-04) ----------------
# The classifier's training data is 52.8 % L5 (TM) + 47.2 % L7 (ETM+) and just 530 OLI
# observations out of 5.65 M — it has effectively never seen L8/L9. Restricting the panel
# to L5+L7 means **every year is inferred on radiometry the model was trained on**, which
# removes the single largest transfer risk (a TM->OLI step landing in 2013, exactly where
# an export-crop expansion would also appear) at the cost of the 2024+ years.
# The Roy et al. harmonisation stays implemented but unused; re-enable by clearing this.
PANEL_MISSIONS: set[str] | None = {"L5", "L7"}

# ---- year range: 1996-2023, measured ------------------------------------------------
# Start 1996: the Landsat archive over Piura is essentially empty before it — gate pass
# 3.7 % (1990), 0.0 % (1992, *zero* clear acquisitions), 4.7 % (1995) vs 100 % (1996).
# End 2023: L7 acquisitions stop — measured gate pass 0.0 % in 2024 and 2025 (mean 0.07
# and 0.00 clear observations per parcel). Without OLI there is no 2024+.
# Thin years to flag rather than interpolate (measured, L5+L7): 1997 48.3 % (the 1997-98
# El Nino), 2009 47.0 %, 2011 37.3 % (Landsat 5's degraded final years — adding L5 back
# rescues neither, so they are thin under any TM/ETM+ policy), 2012 80.3 % (L7 SLC-off
# alone). See docs/DATA.md §7.1.
PANEL_START_YEAR = 1996
PANEL_END_YEAR = 2023
DEFAULT_YEARS = list(range(PANEL_START_YEAR, PANEL_END_YEAR + 1))


def panel_dirs(year: int, bundle_suffix: str = "") -> tuple[Path, Path]:
    """``(pixel store, per-year assembled bundle)`` for one panel year.

    ``bundle_suffix`` selects an *alternative* assembly of the same year — e.g. the
    quantile-aligned bundles of ``allperu.density.write_aligned_panel_bundles``, which live
    in ``<panel>/<year>_qmap/``. Empty (the default) is the audited bundle, so no existing
    caller moves.
    """
    pf = panel_feat()
    return pf, pf / f"{year}{bundle_suffix}"


# ------------------------------------------------------------------------------------
# 1. panel definition
# ------------------------------------------------------------------------------------
def build_panel(n: int = 12000, seed: int = 42, save: bool = True,
                n_test_forced: int | None = 3000) -> gpd.GeoDataFrame:
    """Stratified sample by (label x region), plus locked-test parcels forced in.

    Test parcels are forced in because the temporal-transfer check (§7.3) scores
    predictions at label-year ± k against held-out labels. **Deviation from the plan:** it
    assumed forcing in *every* test parcel, but this test set is 9,894 parcels — 81 % of a
    12,000-parcel panel — which would leave the trend sample spatially confined to the 29
    test regions. ``n_test_forced`` caps the forced subset (3,000 is ample for a per-k
    accuracy curve) so the rest of the budget buys spatial coverage for the area-share
    trend, which is the actual deliverable. Pass ``None`` for the plan's literal behaviour.

    Sampling weights are written alongside: an area share computed on the sample must be
    expanded to the population, never reported as-is.
    """
    rng = np.random.default_rng(seed)
    parcels = gpd.read_parquet(proc() / "modeling_parcels.parquet")
    eligible = parcels[parcels["quality_ok"] == True].copy()  # noqa: E712

    is_test = eligible["split"] == "test"
    forced = is_test
    if n_test_forced is not None and int(is_test.sum()) > n_test_forced:
        # stratify the forced subset by label so rare classes keep their §7.3 support
        keep = (eligible[is_test].groupby("label", group_keys=False, observed=True)
                .apply(lambda g: g.sample(
                    max(1, int(round(n_test_forced * len(g) / int(is_test.sum())))),
                    random_state=seed)))
        forced = eligible.index.isin(keep.index)
        forced = pd.Series(forced, index=eligible.index)
    pool = eligible[~forced]
    strata = pool.groupby(["label", "region_id"], observed=True)

    # proportional allocation, at least one parcel per non-empty stratum
    n_extra = max(n - int(forced.sum()), 0)
    sizes = strata.size()
    alloc = np.maximum(1, np.round(sizes / sizes.sum() * n_extra)).astype(int)
    alloc = np.minimum(alloc, sizes)

    picked = []
    for (key, grp), k in zip(strata, alloc):
        take = min(int(k), len(grp))
        picked.append(grp.sample(take, random_state=int(rng.integers(1 << 31))))
    sample = pd.concat(picked) if picked else pool.iloc[:0]

    panel = pd.concat([eligible[forced], sample]).drop_duplicates("COD_PREDIO")
    # weight = stratum population / stratum sample; test parcels are a census (weight 1)
    pop = eligible.groupby(["label", "region_id"], observed=True).size()
    smp = panel.groupby(["label", "region_id"], observed=True).size()
    w = (pop / smp).rename("sample_weight")
    panel = panel.merge(w, left_on=["label", "region_id"], right_index=True, how="left")
    panel["sample_weight"] = panel["sample_weight"].fillna(1.0)
    # If the workspace was itself a sample of a larger population (the all-Peru build, whose
    # `population_weight` expands the modelling sample to all 14 departments), compose the
    # two stages so a panel weight expands the whole way to the population.
    if "population_weight" in panel.columns:
        panel["sample_weight"] = panel["sample_weight"] * panel["population_weight"]
        print(f"composed with population_weight -> weights expand to "
              f"{panel['sample_weight'].sum():,.0f} parcels")

    print(f"panel: {len(panel):,} parcels "
          f"({int(forced.sum()):,} forced locked-test + {len(sample):,} sampled) "
          f"from {len(eligible):,} eligible")
    print(panel["label"].value_counts().to_string())
    print(f"sample weights: median {panel['sample_weight'].median():.2f}, "
          f"max {panel['sample_weight'].max():.2f}")
    if save:
        out = proc() / "panel_parcels.parquet"
        panel.to_parquet(out, index=False)
        print(f"wrote {out}")
    return panel


# ------------------------------------------------------------------------------------
# 2. cost calibration (§7.1) — measure before committing 20+ hours
# ------------------------------------------------------------------------------------
def timing_probe(n_parcels: int = 500, years: tuple[int, ...] = (1995, 2005, 2015, 2023),
                 seed: int = 42, save: bool = True,
                 out: Path | None = None) -> pd.DataFrame:
    """Measure s/parcel-year and MB/parcel-year per era.

    Landsat availability grows over time (more missions after 2013 => more observations
    per parcel-year => slower and bigger), so the 1998-derived 0.19 s/parcel-year rate
    must **not** be extrapolated to 2023.

    ``out`` names the destination CSV; it defaults to ``docs/figures/panel_budget.csv``,
    the Piura panel's measured budget. **A second panel must pass its own path** — that file
    is the record of what the first extraction cost and overwriting it destroys the only
    evidence of a rate that was never supposed to be extrapolated.
    """
    lg.MISSION_FILTER = PANEL_MISSIONS
    panel = gpd.read_parquet(proc() / "panel_parcels.parquet")
    sub = panel.sample(min(n_parcels, len(panel)), random_state=seed)
    rows = []
    for y in years:
        probe_dir = panel_feat() / "_probe" / str(y)
        probe_dir.mkdir(parents=True, exist_ok=True)
        p = sub.copy()
        p["year"] = y
        t0 = time.time()
        cov = lg.run_coverage(parcels=p, out=probe_dir / f"coverage_{y}.parquet",
                              years=[y])
        t_cov = time.time() - t0
        surv = p[p["COD_PREDIO"].isin(
            cov.loc[cov["n_valid_obs"] >= 4, "COD_PREDIO"])]
        t0 = time.time()
        lg.run_pixels(parcels=surv, years=[y], feat_dir=probe_dir,
                      only_quality_ok=False)
        t_px = time.time() - t0
        mb = sum(f.stat().st_size for f in probe_dir.rglob("*.parquet")) / 1e6
        rows.append({"year": y, "n_parcels": len(p), "gate_pass": len(surv) / len(p),
                     "s_coverage": t_cov, "s_pixels": t_px,
                     "s_per_parcel_year": (t_cov + t_px) / len(p),
                     "mb_per_parcel_year": mb / len(p)})
        print(f"  {y}: {(t_cov + t_px) / len(p):.3f} s/parcel-year, "
              f"{mb / len(p) * 1000:.1f} kB/parcel-year, "
              f"gate {len(surv) / len(p):.1%}", flush=True)
    df = pd.DataFrame(rows)
    if save:
        dest = Path(out) if out is not None else Path("docs/figures/panel_budget.csv")
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=False)
        print(f"wrote {dest}")
    return df


# ------------------------------------------------------------------------------------
# 3. extraction (§7.2)
# ------------------------------------------------------------------------------------
def extract_panel(years: list[int] | None = None, chunk_size: int = 400,
                  pixel_chunk_size: int = 40, max_chunks: int | None = None) -> None:
    """Per year: coverage -> per-year gate -> pixels. Fully resumable.

    The gate is applied **per year** into ``panel_coverage_{Y}.parquet``; it never touches
    the label-year ``modeling_parcels.parquet``.
    """
    years = years or DEFAULT_YEARS
    lg.MISSION_FILTER = PANEL_MISSIONS       # TM/ETM+ only — see PANEL_MISSIONS above
    panel = gpd.read_parquet(proc() / "panel_parcels.parquet")
    pf = panel_feat()
    min_obs = 4

    for y in years:
        p = panel.copy()
        p["year"] = y
        cov_path = pf / f"coverage_{y}.parquet"
        print(f"\n=== {y}: coverage ===", flush=True)
        cov = lg.run_coverage(parcels=p, out=cov_path, years=[y],
                              chunk_size=chunk_size, max_chunks=max_chunks)
        cov["quality_ok"] = cov["n_valid_obs"] >= min_obs
        cov.to_parquet(proc() / f"panel_coverage_{y}.parquet", index=False)
        surv = p[p["COD_PREDIO"].isin(cov.loc[cov["quality_ok"], "COD_PREDIO"])]
        print(f"{y}: gate {len(surv):,}/{len(p):,} ({len(surv) / len(p):.1%}) pass",
              flush=True)
        print(f"=== {y}: pixels ===", flush=True)
        lg.run_pixels(parcels=surv, years=[y], feat_dir=pf,
                      only_quality_ok=False, chunk_size=pixel_chunk_size,
                      max_chunks=max_chunks)


def rebuild_year_stores(years: list[int] | None = None) -> pd.DataFrame:
    """Rebuild ``pixels_<year>.parquet`` from every chunk on disk, and count what it holds.

    ``run_pixels`` combines a year by globbing **every** chunk present at the moment it runs,
    so with concurrent workers on disjoint year ranges the worker that finishes first writes
    a per-year store for years still being extracted. That is how `pixels_2023.parquet` once
    sat on disk at 62 % of its parcels looking perfectly well-formed (RESULTS.md §7.1). Run
    this once after **all** workers have exited.
    """
    pf = panel_feat()
    chunk_dir = pf / "pixels_chunks"
    rows = []
    for y in (years or DEFAULT_YEARS):
        files = sorted(chunk_dir.glob(f"px_{y}_*.parquet"))
        if not files:
            rows.append({"year": y, "n_chunks": 0, "n_pixel_obs": 0, "n_parcels": 0})
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df = df.drop_duplicates(subset=[c for c in ("COD_PREDIO", "doy", "lon", "lat",
                                                    "mission") if c in df.columns])
        df.to_parquet(pf / f"pixels_{y}.parquet", index=False)
        rows.append({"year": y, "n_chunks": len(files), "n_pixel_obs": int(len(df)),
                     "n_parcels": int(df["COD_PREDIO"].nunique())})
        print(f"pixels_{y}.parquet: {len(df):,} pixel-obs, "
              f"{df['COD_PREDIO'].nunique():,} parcels", flush=True)
    return pd.DataFrame(rows)


def verify_years(years: list[int] | None = None, tol: float = 0.99) -> pd.DataFrame:
    """Count the parcels in each year's pixel store against that year's gate survivors.

    **This is the only acceptable completeness check.** "The process ended", "the file
    exists" and "no traceback" have each been wrong here at least once, and the failure mode
    is a truncated year that looks well-formed. Returns a frame with a ``complete`` column;
    the caller should refuse to assemble while any year is incomplete.

    ⚠️ **There is a structural floor of a few tenths of a percent, and ``tol`` is set from
    it rather than from taste.** Measured on the DiD panel (2026-08-12): every year loses the
    *same* ~85 of 14,625 parcels — Jaccard 0.95 between 1999 and 2020, 86 parcels in the union
    over six years — and they are sub-pixel (median 0.11 ha, 1.23 estimated pixels, against
    1.00 ha / 11.06 for the sample). A parcel smaller than a Landsat pixel can pass the
    *coverage* gate, which counts scene observations, and still return no pixel rows.

    So an absolute ratio alone cannot separate "truncated" from "sub-pixel". The second
    statistic is what does: a structural loss is **the same size in every year**, while a
    truncated year is a year-specific outlier (the real incident was 62 % in one year and
    ~99 % in the rest). ``deficit_ratio_to_median`` flags that, and is reported always.
    """
    pf = panel_feat()
    rows = []
    for y in (years or DEFAULT_YEARS):
        cov_path = proc() / f"panel_coverage_{y}.parquet"
        px_path = pf / f"pixels_{y}.parquet"
        n_expected = 0
        if cov_path.exists():
            cov = pd.read_parquet(cov_path)
            n_expected = int(cov["quality_ok"].fillna(False).astype(bool).sum())
        n_got = 0
        if px_path.exists():
            n_got = int(pd.read_parquet(px_path, columns=["COD_PREDIO"])
                        ["COD_PREDIO"].nunique())
        frac = n_got / n_expected if n_expected else float("nan")
        rows.append({"year": y, "gate_survivors": n_expected, "parcels_in_store": n_got,
                     "frac": frac, "deficit": n_expected - n_got})
    df = pd.DataFrame(rows)
    med = float(df.loc[df["deficit"] > 0, "deficit"].median()) if (df["deficit"] > 0).any() \
        else 0.0
    df["deficit_ratio_to_median"] = df["deficit"] / med if med > 0 else 0.0
    # A year is complete if it clears the absolute floor AND its deficit is not an outlier
    # against the other years' — the second clause is what catches a truncated year even if
    # the structural floor were ever large enough to hide one.
    df["complete"] = ((df["gate_survivors"] > 0) & (df["frac"] >= tol)
                      & (df["deficit_ratio_to_median"] <= 3.0))
    print(df.to_string(index=False))
    bad = df[~df["complete"]]
    if len(bad):
        print(f"\n⛔ {len(bad)} year(s) INCOMPLETE: {list(bad['year'])}")
    else:
        print(f"\n✅ all {len(df)} years complete "
              f"({df['parcels_in_store'].sum():,} parcel-years in the pixel stores)")
    return df


def assemble_panel(years: list[int] | None = None,
                   harmonize_oli: bool = False) -> None:
    """One isolated feature bundle per year (the §7.0.1 fix in action).

    ``harmonize_oli`` defaults **off**: under ``PANEL_MISSIONS = {"L5", "L7"}`` there is no
    OLI data to harmonise, and applying an unvalidated correction to nothing is worse than
    not applying it. Turn it on only if OLI is re-admitted.
    """
    years = years or DEFAULT_YEARS
    panel = gpd.read_parquet(proc() / "panel_parcels.parquet")
    for y in years:
        if not (panel_feat() / f"pixels_{y}.parquet").exists():
            print(f"{y}: no pixel store — skipping")
            continue
        _, out_dir = panel_dirs(y)
        print(f"\n=== assembling {y} ===", flush=True)
        statics = panel.copy()
        cov_path = proc() / f"panel_coverage_{y}.parquet"
        if cov_path.exists():
            cov = pd.read_parquet(cov_path)[["COD_PREDIO", "n_valid_obs", "max_gap"]]
            statics = statics.drop(columns=["n_valid_obs", "max_gap"],
                                   errors="ignore").merge(cov, on="COD_PREDIO",
                                                          how="left")
        asm.assemble(years=[y], feat_dir=panel_feat(), out_dir=out_dir, parcels=statics,
                     harmonize_oli=harmonize_oli)


# ------------------------------------------------------------------------------------
# 4. inference (§7.2)
# ------------------------------------------------------------------------------------
def infer_panel(run_dir: Path, years: list[int] | None = None,
                tau: float = 0.0, save: bool = True,
                out: Path | None = None, bundle_suffix: str = "") -> pd.DataFrame:
    """Score every panel year against its own feature bundle and stack the results.

    ``out`` overrides the default ``panel_predictions.parquet`` so a second model can be
    inferred over the same panel without clobbering the first — the two are only
    comparable if both survive.
    """
    from crop_classifier.infer import infer

    # Resolve the destination BEFORE the expensive loop. This used to be evaluated only
    # after every year had been scored, so a bad --out threw away the whole run.
    out_path = Path(out) if out is not None else proc() / "panel_predictions.parquet"
    if save and not out_path.parent.is_dir():
        raise NotADirectoryError(f"output directory does not exist: {out_path.parent}")

    years = years or DEFAULT_YEARS
    panel = gpd.read_parquet(proc() / "panel_parcels.parquet")
    weights = panel.set_index("COD_PREDIO")["sample_weight"]
    out = []
    for y in years:
        _, feat_dir = panel_dirs(y, bundle_suffix)
        if not (feat_dir / "features_lightgbm.parquet").exists():
            print(f"{y}: not assembled — skipping")
            continue
        p = panel.copy()
        p["year"] = y
        cov_path = proc() / f"panel_coverage_{y}.parquet"
        if cov_path.exists():
            cov = pd.read_parquet(cov_path)[["COD_PREDIO", "n_valid_obs", "max_gap",
                                             "quality_ok"]]
            p = p.drop(columns=["n_valid_obs", "max_gap", "quality_ok"],
                       errors="ignore").merge(cov, on="COD_PREDIO", how="left")
            p["quality_ok"] = p["quality_ok"].astype("boolean")
        res = infer(run_dir, parcels=p, tau=tau, feat_dir=feat_dir)
        res["year"] = y
        out.append(res)
    panel_preds = pd.concat(out, ignore_index=True)
    panel_preds["sample_weight"] = panel_preds["COD_PREDIO"].map(weights).fillna(1.0)
    panel_preds = panel_preds.merge(panel[["COD_PREDIO", "area_ha", "label"]]
                                    .rename(columns={"label": "pett_label"}),
                                    on="COD_PREDIO", how="left")
    if save:
        panel_preds.to_parquet(out_path, index=False)
        print(f"\nwrote {out_path}: {len(panel_preds):,} parcel-years")
    return panel_preds


# ------------------------------------------------------------------------------------
# 5. temporal transfer (§7.3) — the key internal check
# ------------------------------------------------------------------------------------
def temporal_transfer(panel_preds: pd.DataFrame, parcels: gpd.GeoDataFrame | None = None,
                      test_only: bool = True, k_max: int = 5) -> pd.DataFrame:
    """Accuracy vs temporal distance ``k`` from each parcel's label year.

    A parcel labelled PERENNIAL in 1998 was almost certainly perennial in 1997 and 2000 —
    orchards do not appear or vanish annually — so accuracy against the y0 label as a
    function of k measures temporal transfer. **Caveat that must be stated:** real land-use
    change also contributes to the decay, so this is a *lower bound* on model stability.
    A cliff at a specific year (especially 2012, or the L8 boundary at 2013) is a sensor
    artefact, not land-use change.
    """
    from sklearn.metrics import f1_score

    if parcels is None:
        parcels = gpd.read_parquet(proc() / "modeling_parcels.parquet")
    ref = parcels[["COD_PREDIO", "label", "year", "split"]].rename(
        columns={"year": "label_year"})
    df = panel_preds.merge(ref, on="COD_PREDIO", how="inner")
    if test_only:
        df = df[df["split"] == "test"]
    df = df[~df["abstained"].astype(bool) & df["pred_label"].notna()]
    df["k"] = df["year"] - df["label_year"]
    df = df[df["k"].abs() <= k_max]

    rows = []
    for k, sub in df.groupby("k"):
        rows.append({"k": int(k), "n": len(sub),
                     "accuracy": float((sub["pred_label"] == sub["label"]).mean()),
                     "macro_f1": float(f1_score(sub["label"], sub["pred_label"],
                                                average="macro", zero_division=0))})
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-year panel (plan §7)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("-n", type=int, default=12000)
    sub.add_parser("probe")
    for name in ("extract", "assemble", "infer"):
        s = sub.add_parser(name)
        s.add_argument("--years", default="1996-2023")
        if name == "infer":
            s.add_argument("--run", type=Path, required=True)
    a = ap.parse_args()
    yrs = None
    if getattr(a, "years", None):
        lo, hi = (int(x) for x in a.years.split("-"))
        yrs = list(range(lo, hi + 1))
    if a.cmd == "build":
        build_panel(n=a.n)
    elif a.cmd == "probe":
        print(json.dumps(timing_probe().to_dict("records"), indent=2))
    elif a.cmd == "extract":
        extract_panel(years=yrs)
    elif a.cmd == "assemble":
        assemble_panel(years=yrs)
    elif a.cmd == "infer":
        infer_panel(a.run, years=yrs)


if __name__ == "__main__":
    main()

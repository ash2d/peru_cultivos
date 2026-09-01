"""Runner for the S2 endpoint-labelling campaign (docs/s2_labelling/plan.md).

One function per step, in the plan's order — cheapest step that can kill the plan first.
Every step reads what the previous one wrote from ``<CC_PROC>/labels_s2/`` and writes its
own artefact there, so any step can be re-run on its own and the campaign is resumable.

    universe -> pool -> probe -> draw -> split -> chips -> extract
             -> harmonisation -> html -> (humans label) -> assemble -> ingest
             -> transitions
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from crop_classifier.allperu import esri_dates as E
from crop_classifier.allperu import label_sample as LS
from crop_classifier.paths import feat

# Departments whose 2.5x candidate pool could not fill their quota — Pasco's imagery is
# 40 % eligible and the sierra departments run out of *regions*, not parcels, under the
# 2-per-region cap. Recorded here rather than passed by hand so the campaign reproduces.
SHORT_DEPTS = ["HUANCAVELICA", "MOQUEGUA", "PASCO", "PIURA", "TACNA", "TUMBES",
               "LAMBAYEQUE"]

F_PROBE = "esri_dates_national.csv"
F_PROBE_SUPP = "esri_dates_national_supp.csv"
F_CANDIDATES = "label_candidates_eligible.parquet"
F_POPELIG = "label_eligible_population.csv"
F_GATES = "gates.json"


def d() -> Path:
    return LS.out_dir()


def chip_dir() -> Path:
    p = d() / "chips"
    p.mkdir(parents=True, exist_ok=True)
    return p


def html_dir(lang: str = "en") -> Path:
    """``html/`` for English, ``html_<lang>/`` otherwise.

    Separate directories rather than a suffix on each filename: the shard name is what the
    labeller is asked for by name and what comes back in the CSV, and it must stay
    ``shard01_A`` in every language so a returned file is unambiguous.
    """
    p = d() / ("html" if lang == "en" else f"html_{lang}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _record_gate(name: str, value, criterion: str, passed: bool, **extra) -> None:
    """Append a gate outcome to ``gates.json`` — measured once, on the record forever."""
    f = d() / F_GATES
    cur = json.loads(f.read_text()) if f.exists() else {}
    cur[name] = {"value": value, "criterion": criterion, "pass": bool(passed), **extra}
    f.write_text(json.dumps(cur, indent=2, default=str))
    print(f"\n{'PASS' if passed else 'FAIL'}  {name}: {value} (criterion {criterion})")


# ------------------------------------------------------------------------------------
def step_universe(source: Path) -> None:
    """Build the draw frame. Falls back to the committed national parcel table.

    The campaign of record drew from the full population (``all_peru_full``, 726,808
    parcels, not committed). A clone has the 56,419-parcel national sample instead, which is
    a perfectly good frame for drawing *more* parcels to label — it is simply a smaller one,
    and it is already restricted to the linkable 14 departments.
    """
    src = Path(source)
    if not (src / "modeling_parcels.parquet").exists():
        alt = Path("data/processed/all_peru")
        if not (alt / "modeling_parcels.parquet").exists():
            raise SystemExit(f"no modeling_parcels.parquet under {src} or {alt}")
        print(f"  {src}/modeling_parcels.parquet is not present (437 MB, not committed) — "
              f"drawing from {alt}/modeling_parcels.parquet, 56,419 parcels")
        src = alt
    LS.build_universe(src)


def step_pool(supp_depts: list[str] | None = None) -> None:
    u = gpd.read_parquet(d() / LS.F_UNIVERSE)
    f = d() / F_PROBE
    if not f.exists():
        LS.candidate_pool(u)
        return
    already = set(pd.read_csv(f)["COD_PREDIO"].astype(str))
    LS.supplement_pool(u, already, supp_depts or SHORT_DEPTS)


def step_probe(workers: int = 8) -> None:
    """Probe whichever pool file is newest and has unprobed parcels; then gate G0."""
    for pool_f, cache in ((d() / "label_candidate_pool.parquet", d() / F_PROBE),
                          (d() / "label_candidate_pool_supp.parquet",
                           d() / F_PROBE_SUPP)):
        if pool_f.exists():
            pool = gpd.read_parquet(pool_f)
            E.probe_frame(pool, workers=workers, cache=cache)
    _consolidate_probe()


def _consolidate_probe() -> gpd.GeoDataFrame:
    """Merge probe results onto the pools, apply §1 filter 4, evaluate **G0**."""
    pools = [gpd.read_parquet(d() / f) for f in
             ("label_candidate_pool.parquet", "label_candidate_pool_supp.parquet")
             if (d() / f).exists()]
    pool = gpd.GeoDataFrame(pd.concat(pools, ignore_index=True), crs=pools[0].crs)
    pool["COD_PREDIO"] = pool["COD_PREDIO"].astype(str)
    pool = pool.drop_duplicates("COD_PREDIO")

    probes = [pd.read_csv(d() / f) for f in (F_PROBE, F_PROBE_SUPP)
              if (d() / f).exists()]
    pr = pd.concat(probes, ignore_index=True)
    pr["COD_PREDIO"] = pr["COD_PREDIO"].astype(str)
    pr = pr.drop_duplicates("COD_PREDIO")
    pr = pr.merge(pool[["COD_PREDIO", "label"]].rename(
        columns={"label": "declared_class"}), on="COD_PREDIO", how="inner")
    pr["_elig"] = E.eligible(pr)

    rate = float(pr["_elig"].mean())
    by_dept = pr.groupby("dept")["_elig"].mean().round(3)
    _record_gate("G0_esri_eligible_share", round(rate, 4), ">= 0.80", rate >= 0.80,
                 by_department=by_dept.to_dict(), n_probed=int(len(pr)))
    print("\neligible share by department:")
    print(by_dept.to_string())

    u = gpd.read_parquet(d() / LS.F_UNIVERSE)
    u["COD_PREDIO"] = u["COD_PREDIO"].astype(str)
    popE = LS.eligible_population(u, pr, pr["_elig"])
    popE.to_csv(d() / F_POPELIG, index=False)

    cand = pool[pool.COD_PREDIO.isin(set(pr.loc[pr._elig, "COD_PREDIO"]))].merge(
        pr[["COD_PREDIO", "res", "year", "date"]].rename(
            columns={"res": "imagery_res", "date": "imagery_date",
                     "year": "imagery_year"}), on="COD_PREDIO", how="left")
    cand = gpd.GeoDataFrame(cand, geometry="geometry", crs=pool.crs)
    cand.to_parquet(d() / F_CANDIDATES, index=False)
    print(f"\n{len(cand):,} eligible candidates written to {F_CANDIDATES}")
    return cand


def step_draw() -> None:
    cand = gpd.read_parquet(d() / F_CANDIDATES)
    popE = pd.read_csv(d() / F_POPELIG)
    sizes = popE.pivot(index="dept", columns="declared_class",
                       values="N_eligible").fillna(0)
    s = LS.draw(cand, alloc_sizes=sizes, pop_eligible=popE)
    main = int((s.batch == "main").sum())
    _record_gate("draw_size", main, f"== {LS.TOTAL}", main >= LS.TOTAL * 0.95,
                 note="shortfalls are eligible-pool or region-cap limits, reported "
                      "per cell in the draw log")


def step_split() -> None:
    LS.assign_split()


def cadastre_path() -> Path:
    """The polygon table the chips draw neighbouring parcel boundaries from.

    The full national table (726,808 parcels) is 437 MB and is not committed, so a clone
    falls back to the 56,419-parcel national sample. That changes only how many *context*
    outlines a chip shows — never which parcel is being labelled, which comes from
    ``label_sample.parquet``.
    """
    for c in (Path("data/processed/all_peru_full/modeling_parcels.parquet"),
              Path("data/processed/all_peru/modeling_parcels.parquet")):
        if c.exists():
            return c
    raise SystemExit("no parcel polygon table found under data/processed/ "
                     "(docs/DATA_ACCESS.md)")


def step_chips(workers: int = 3, overwrite: bool = False) -> None:
    from crop_classifier.labelling import chips

    s = gpd.read_parquet(d() / LS.F_SAMPLE)
    cad = gpd.read_parquet(cadastre_path(), columns=["COD_PREDIO", "geometry"])
    stats = chips.render_all(s, cad, chip_dir(), cache_dir=d() / "tiles",
                             workers=workers, overwrite=overwrite)
    with open(d() / "chip_report.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)


def step_extract(chunk_size: int = 25, max_chunks: int | None = None,
                 workers: int = 4) -> None:
    from crop_classifier.features import s2_gee as S

    s = gpd.read_parquet(d() / LS.F_SAMPLE)
    px = S.extract(s, chunk_size=chunk_size, max_chunks=max_chunks,
                   workers=workers)
    if len(px):
        rep = S.observation_report(px, s)
        rep.to_csv(d() / "s2_observation_report.csv")
        print("\nclear S2 observations per parcel-year, by department:")
        print(rep.to_string())


def step_harmonisation() -> None:
    from crop_classifier.features import s2_gee as S
    S.harmonisation_check()


def step_assemble() -> None:
    """Per-date medians -> the LightGBM summary block, plus the coverage report.

    Runs before `ingest` because the ingest needs `s2_feature_meta.parquet` to apply the
    sub-pixel-parcel exclusion. A parcel with too few usable S2 pixels is mixed-pixel label
    noise, and that caps a model outright however good the human label is.
    """
    from crop_classifier.features import s2_assemble as A
    from crop_classifier.features.s2_gee import f_pixels

    s = gpd.read_parquet(d() / LS.F_SAMPLE)
    px = pd.read_parquet(f_pixels())
    A.assemble(px, s)
    cov = A.coverage_report(px, s)
    cov.to_csv(d() / "s2_coverage_report.csv")
    print("\nclear S2 dates in the agricultural year, by department:")
    print(cov.to_string())
    miss = A.missing_parcels(px, s)
    if len(miss):
        print(f"\n⚠️ {int(miss.sum())} parcels have no usable S2 date at all:")
        print(miss.to_string())


def item_key_path() -> Path:
    """The item_id -> COD_PREDIO key, from whichever language set was built.

    The key is language-independent (same shards, same item ids, same order), so ingest must
    not require the English build to exist just because it is the default.
    """
    for cand in [html_dir("en") / "item_key.csv",
                 *sorted(d().glob("html_*/item_key.csv"))]:
        if cand.exists():
            return cand
    raise SystemExit("no item_key.csv found — run `s2-labels html` first")


def step_html(lang: str = "en") -> None:
    from crop_classifier.features.s2_gee import f_pixels
    from crop_classifier.labelling import build_html as BH

    s = gpd.read_parquet(d() / LS.F_SAMPLE)
    px = pd.read_parquet(f_pixels())
    BH.build(s, chip_dir(), px, html_dir(lang), lang=lang)


def step_ingest(csv_dir: Path) -> None:
    from crop_classifier.features.s2_assemble import FN_META
    from crop_classifier.labelling.ingest import ingest

    s = gpd.read_parquet(d() / LS.F_SAMPLE)
    key = pd.read_csv(item_key_path())
    meta_f = feat() / FN_META
    meta = pd.read_parquet(meta_f) if meta_f.exists() else None
    ingest(Path(csv_dir), s, key, s2_meta=meta, out_dir=d())


def step_combine() -> None:
    """Merge THIS round with the campaign of record, into a third round you can train on.

    A round is deliberately kept apart from the campaign of record while it is being built
    (`--round <name>` moves both its labels and its feature store), because a draw replaces
    a sample and an assemble replaces a feature table. But the reason to label more parcels
    is to train on *all* of them, so this writes the union — labels, sample and the LightGBM
    feature block — into ``labels_s2_<name>_all`` / ``features_s2_<name>_all``, leaving both
    inputs untouched.

    Train on it with ``--round <name>_all``.

    ⚠️ Two things it does not do. It does not rebuild ``tensor_perdate.npz``, so the merged
    round trains LightGBM (the model carried forward) and not LTAE. And the merged test
    split contains the round-of-record's **already-spent** locked test — a score on it
    confirms nothing that has not been confirmed once already, so select on CV and LODO.
    """
    from crop_classifier.features.s2_assemble import FN_LGBM, FN_META
    from crop_classifier.paths import feat, proc

    rd, rf = d(), feat()                                   # this round
    od, of = proc() / "labels_s2", proc() / "features_s2"   # the campaign of record
    if rd == od:
        raise SystemExit("`combine` needs a round: pass --round <name>, the same one you "
                         "drew and labelled (docs/howto/06_label_more_parcels.md)")
    name = rd.name.replace("labels_s2_", "")
    md, mf = proc() / f"labels_s2_{name}_all", proc() / f"features_s2_{name}_all"
    md.mkdir(parents=True, exist_ok=True)
    mf.mkdir(parents=True, exist_ok=True)

    def _read(f: Path):
        """Geo where there is geometry, plain pandas where there is not."""
        try:
            return gpd.read_parquet(f)
        except ValueError:
            return pd.read_parquet(f)

    def _cat(a: Path, b: Path, key: str = "COD_PREDIO"):
        fa, fb = _read(a), _read(b)
        miss = set(fa.columns) ^ set(fb.columns)
        if miss:
            raise SystemExit(
                f"{a.name}: the two rounds do not have the same columns "
                f"({sorted(miss)[:6]}). They were built by different code versions; "
                f"re-run the round's `assemble` before combining.")
        out = pd.concat([fa, fb[~fb[key].astype(str).isin(set(fa[key].astype(str)))]],
                        ignore_index=True)
        return gpd.GeoDataFrame(out, crs=fa.crs) if isinstance(fa, gpd.GeoDataFrame) else out

    for fn in ("labelled_parcels.parquet", LS.F_SAMPLE):
        m = _cat(od / fn, rd / fn)
        m.to_parquet(md / fn, index=False)
        print(f"{fn}: {len(m):,} parcels "
              f"({len(m) - len(_read(od / fn)):+,} from round '{name}')")
    for fn in (FN_LGBM, FN_META):
        _cat(of / fn, rf / fn).to_parquet(mf / fn, index=False)

    print(f"\nwrote {md} and {mf}\n"
          f"train on it with:  cc -w national_s2 labelling train prep "
          f"--round {name}_all --target t3w --climate temp")


def step_transitions() -> None:
    from crop_classifier.labelling.ingest import transition_matrix

    lab = gpd.read_parquet(d() / "labelled_parcels.parquet")
    t = transition_matrix(lab)
    t.to_csv(d() / "declared_to_observed_transitions.csv")
    print("\nweighted declared (1996-2006) -> observed (2019+) transition matrix:")
    print(t.to_string())


STEPS = {
    "universe": lambda **k: step_universe(k["source"]),
    "pool": lambda **k: step_pool(k.get("supp_depts")),
    "probe": lambda **k: step_probe(max(4, k.get("workers", 8))),
    "draw": lambda **k: step_draw(),
    "split": lambda **k: step_split(),
    "chips": lambda **k: step_chips(k.get("workers", 3), k.get("overwrite", False)),
    "extract": lambda **k: step_extract(k.get("chunk_size", 25), k.get("max_chunks"),
                                       k.get("workers", 4)),
    "harmonisation": lambda **k: step_harmonisation(),
    "assemble": lambda **k: step_assemble(),
    "html": lambda **k: step_html(k.get("lang", "en")),
    "ingest": lambda **k: step_ingest(k["csv_dir"]),
    "combine": lambda **k: step_combine(),
    "transitions": lambda **k: step_transitions(),
}


def run_step(step: str, **kwargs) -> None:
    if step not in STEPS:
        raise SystemExit(f"unknown step {step!r}; expected one of {sorted(STEPS)}")
    if step == "ingest" and not kwargs.get("csv_dir"):
        raise SystemExit("ingest needs --csv-dir")
    STEPS[step](**kwargs)

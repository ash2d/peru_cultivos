"""One row per parcel, one column per observation of it — the export table.

Everything this project measures about a parcel, side by side::

    COD_PREDIO | tenure | pett_year | pett_class | cen_class | s2_class | s2_pred_class

Four instruments, three points in time, and no analysis: the declared crop (~1997-2006), the
2012 census, a human reading of 2019+ imagery, and the Sentinel-2 classifier's reading of the
same. `cenagro_shift.py` answers a question with these; this module just hands them over.

**The universe is the census-linked panel by default** — 95,941 parcels with both a PETT
declaration and a CENAGRO 2012 observation, the largest set a clone of this repo can build
(`national_panel.parquet` is committed). ``--universe all`` widens it to every PETT parcel,
726,808, which needs the uncommitted national table. Every later observation is a LEFT join,
so a column is null where that instrument never looked at that parcel — which is most of
them. 865 parcels carry a human label and only 157 of those are in the linked panel; the
campaign drew from the PETT population, not from the name-linked census subset, so that
overlap is incidental and far too small to estimate a 2012->2025 change from
(`RESULTS.md` §8.6).

⚠️ Three things the columns, not the process, are here to keep straight:

1. **`s2_pred_source` says whether a prediction is honest.** The classifier was fitted on
   these very parcels, so scoring the campaign with `cc predict` grades the model on its own
   training rows. The default here reads `fold*/preds_val.parquet` (`out_of_fold`) and
   `preds_test.parquet` (`locked_test`) instead — held out by construction. `applied` is the
   only source that can cover a parcel the model never saw (`howto/02_parcel_table.md`).
2. **`weight` is not optional for any share.** The campaign over-sampled `PERENNIAL` ~3x, so
   an unweighted mean of `s2_class` over this table is about three times the population's.
   It is null for the 120 pilot parcels, which were drawn under a different design.
3. **`WOODY_NON_CROP` is never folded silently.** Under ``--classes 4`` (the default) it is
   its own class and the classifier column comes from a `t4` run that can emit it. Under
   ``--classes 3`` the declared label space has no word for it, so it is reported twice —
   `s2_class` leaves it unmapped, `s2_class_woody_perennial` reads it as perennial canopy.
   The choice moves the imagery endpoint by 26 pp against a census effect of about 10 pp.

Run it with::

    uv run cc -w national analysis parcel-table
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from crop_classifier.allperu.cenagro_shift import S2_TO_DECLARED
from crop_classifier.paths import ROOT

PANEL = ROOT / "data" / "processed" / "cenagro" / "national_panel.parquet"
PETT_FULL = ROOT / "data" / "processed" / "all_peru_full" / "modeling_parcels.parquet"
TENURE = ROOT / "data" / "processed" / "all_peru" / "tenure_by_predio.parquet"
LABELS = ROOT / "data" / "processed" / "all_peru" / "labels_s2" / "labelled_parcels.parquet"
RUNS = {4: ROOT / "runs" / "s2_labels" / "ws_t4_pilot__clim_temp" / "lightgbm",
        3: ROOT / "runs" / "s2_labels" / "ws_t3w_pilot__clim_temp" / "lightgbm"}
DEFAULT_RUN = RUNS[3]
OUT = ROOT / "data" / "processed" / "cenagro" / "parcel_table.parquet"

# The four-class vocabulary (`--classes 4`, the default): the only one all four instruments
# can be written in without a class being invented or silently folded. `PASTURE_FALLOW` and
# `OTHER` are the same state under two instruments' names — farmable ground not currently
# cropped — and `NON_AGRICULTURE` joins them because the declared side has no word for it.
CLASSES4 = ("PERENNIAL", "ANNUAL", "WOODY_NON_CROP", "OTHER")
TO_4 = {"PERENNIAL": "PERENNIAL", "ANNUAL": "ANNUAL", "WOODY_NON_CROP": "WOODY_NON_CROP",
        "PASTURE_FALLOW": "OTHER", "OTHER": "OTHER", "NON_AGRICULTURE": "OTHER"}

# The model of record predicts `t3w`: {ANNUAL, OTHER, PERENNIAL}. `OTHER` is what
# `PASTURE_FALLOW` is called in the imagery label space (`labelling.train_prep.TO_LANDSAT`),
# so renaming it here is what puts all four observations in one vocabulary.
PRED_TO_DECLARED = {"PERENNIAL": "PERENNIAL", "ANNUAL": "ANNUAL", "OTHER": "PASTURE_FALLOW"}

COLUMNS4 = [
    "COD_PREDIO", "dept", "area_ha",
    "tenure", "frac_inscrito", "reg_year",
    "pett_year", "pett_class",
    "cen_class", "cen_sown_ha", "cen_any_export", "n_producers", "link_confidence",
    "s2_label", "s2_class", "imagery_date", "weight",
    "s2_pred_label", "s2_pred_class", "s2_pred_proba", "s2_pred_source",
    "n_observations",
]

COLUMNS = [
    "COD_PREDIO", "dept", "area_ha",
    # tenure at the declaration (`ESTADO en RRPP`), plus the year it was registered
    "tenure", "frac_inscrito", "reg_year",
    # 1. the PETT declaration
    "pett_year", "pett_class",
    # 2. the 2012 census
    "cen_class", "cen_sown_ha", "cen_any_export", "n_producers", "link_confidence",
    # 3. a human reading of 2019+ imagery
    "s2_label", "s2_class", "s2_class_woody_perennial", "imagery_date", "weight",
    # 4. the Sentinel-2 classifier on the same imagery
    "s2_pred_label", "s2_pred_class", "s2_pred_proba", "s2_pred_source",
    "n_observations",
]


def _key(df: pd.DataFrame) -> pd.DataFrame:
    """`COD_PREDIO` as text on every side — the join returns zero rows otherwise."""
    out = df.copy()
    out["COD_PREDIO"] = out["COD_PREDIO"].astype(str)
    return out


def _argmax_label(df: pd.DataFrame) -> pd.DataFrame:
    """`prob_*` columns -> (`s2_pred_label`, `s2_pred_proba`)."""
    prob = [c for c in df.columns if c.startswith("prob_")]
    if not prob:
        raise KeyError(f"no prob_* columns in {sorted(df.columns)}")
    classes = [c[len("prob_"):] for c in prob]
    out = df[["COD_PREDIO"]].copy()
    out["s2_pred_label"] = [classes[i] for i in df[prob].to_numpy().argmax(axis=1)]
    out["s2_pred_proba"] = df[prob].max(axis=1).to_numpy()
    return out


def held_out_predictions(run_dir: Path = DEFAULT_RUN) -> pd.DataFrame:
    """The run's predictions on parcels it did **not** train on, tagged by which held them.

    `fold*/preds_val.parquet` is the cross-validated read (the fold that held each parcel
    out) and `preds_test.parquet` the locked test's. Not `perennial.pool_cv`, which writes
    into the run directory and keeps only the folds — this is a read-only assembly and the
    locked-test rows belong in the export too.

    ⚠️ A locked-test parcel appears in **both**: it carries a fold id, and `data.fold_split`
    puts every row of that fold in the fold's validation set while the train side takes
    `split == "trainval"` only. Both readings are held out; the final refit's is the one
    kept. Two rows for a *trainval* parcel would be a split bug and still raises.
    """
    run_dir = Path(run_dir)
    parts = []
    for f in sorted(run_dir.glob("fold*/preds_val.parquet")):
        parts.append(_argmax_label(pd.read_parquet(f)).assign(s2_pred_source="out_of_fold"))
    folds = _key(pd.concat(parts, ignore_index=True)) if parts else None
    if folds is not None:
        dup = int(folds["COD_PREDIO"].duplicated().sum())
        if dup:
            raise ValueError(f"{dup:,} parcels appear in more than one validation fold of "
                             f"{run_dir} — overlapping folds, a split bug")
    test = run_dir / "preds_test.parquet"
    tst = (_key(_argmax_label(pd.read_parquet(test))).assign(s2_pred_source="locked_test")
           if test.exists() else None)
    if folds is None and tst is None:
        raise FileNotFoundError(
            f"no fold*/preds_val.parquet or preds_test.parquet under {run_dir}. Fit the "
            f"model of record with `uv run cc reproduce s2-model`, or pass --preds with the "
            f"output of `cc predict` (docs/howto/03_score_parcels.md §C).")
    if folds is None:
        return tst
    if tst is None:
        return folds
    return pd.concat([tst, folds[~folds["COD_PREDIO"].isin(tst["COD_PREDIO"])]],
                     ignore_index=True)


def applied_predictions(path: Path) -> pd.DataFrame:
    """A `cc predict` output, for parcels scored after the fact (`howto/02_parcel_table.md`).

    ⚠️ Tagged `applied` and never mixed with the held-out sources: pointed at the campaign's
    own parcels this is the model reading its training data, which scores far above the 0.774
    it earned on the locked test.
    """
    df = _key(pd.read_parquet(path))
    if "pred_label" in df.columns:
        out = df[["COD_PREDIO"]].copy()
        out["s2_pred_label"] = df["pred_label"]
        out["s2_pred_proba"] = df.get("pred_proba")
        if "abstained" in df.columns:                 # an abstention is not a prediction
            out.loc[df["abstained"].to_numpy(dtype=bool), ["s2_pred_label",
                                                           "s2_pred_proba"]] = None
    else:
        out = _argmax_label(df)
    out["s2_pred_source"] = "applied"
    return out[out["s2_pred_label"].notna()]


def universe(which: str = "linked") -> pd.DataFrame:
    """The parcels the table is about.

    ``linked`` is the census panel — the 95,941 parcels with both a PETT declaration and a
    CENAGRO 2012 record, and the largest set a clone of this repo can build. ``all`` is every
    parcel in the PETT registry (726,808 over 14 departments), which needs the uncommitted
    national table: the census columns are then null on the ~87 % that never linked.
    """
    if which == "all":
        if not PETT_FULL.exists():
            raise SystemExit(
                f"--universe all needs {PETT_FULL.relative_to(ROOT)}, which is not "
                f"committed (437 MB). Build it with `cc data link` from the licensed "
                f"archive (docs/DATA_ACCESS.md), or use --universe linked.")
        pett = pd.read_parquet(PETT_FULL,
                               columns=["COD_PREDIO", "dept", "label", "year", "area_ha"])
        pett = _key(pett).rename(columns={"label": "pett_class", "year": "pett_year"})
        cen = _key(pd.read_parquet(PANEL)).drop(
            columns=["dept", "pett_class", "pett_year", "area_ha", "tenure",
                     "frac_inscrito"], errors="ignore")
        return pett.merge(cen, on="COD_PREDIO", how="left")
    if which != "linked":
        raise SystemExit(f"--universe must be 'linked' or 'all', not {which!r}")
    if not PANEL.exists():
        raise SystemExit(
            f"missing {PANEL.relative_to(ROOT)} — build it with\n"
            f"  uv run cc -w national analysis perennial-shift")
    return _key(pd.read_parquet(PANEL))


def build(run: Path | None = None, preds: Path | None = None, out: Path | None = OUT,
          classes: int = 4, which: str = "linked", verbose: bool = True) -> pd.DataFrame:
    """The export table. `preds` (a `cc predict` output) replaces the run's held-out read.

    ``classes=4`` writes every class column in {PERENNIAL, ANNUAL, WOODY_NON_CROP, OTHER}
    and reads the classifier column off a `t4` run; ``classes=3`` keeps the declared label
    space, where woody canopy has no class and is reported twice instead.
    """
    if classes not in RUNS:
        raise SystemExit(f"--classes must be 4 or 3, not {classes}")
    run = RUNS[classes] if run is None else run
    df = universe(which)

    # one source for tenure, so the two universes are built the same way
    df = df.drop(columns=["tenure", "frac_inscrito"], errors="ignore")
    ten = _key(pd.read_parquet(TENURE,
                               columns=["COD_PREDIO", "tenure", "frac_inscrito", "reg_year"]))
    df = df.merge(ten, on="COD_PREDIO", how="left")

    if LABELS.exists():
        lab = _key(pd.read_parquet(LABELS, columns=["COD_PREDIO", "label", "imagery_date",
                                                    "weight", "usable"]))
        lab = lab[lab["usable"]].drop(columns="usable").rename(columns={"label": "s2_label"})
        if classes == 4:
            lab["s2_class"] = lab["s2_label"].map(TO_4)
        else:
            lab["s2_class"] = lab["s2_label"].map(S2_TO_DECLARED)
            lab["s2_class_woody_perennial"] = lab["s2_class"].where(
                lab["s2_label"] != "WOODY_NON_CROP", "PERENNIAL")
        df = df.merge(lab, on="COD_PREDIO", how="left")

    p = None
    if preds is not None:
        p = applied_predictions(preds)
    elif run is not None and Path(run).exists():
        p = held_out_predictions(Path(run))
    elif verbose:
        print(f"⚠️ no model run at {run} — the classifier column is empty. Fit it with\n"
              f"   uv run cc -w national_s2 labelling train prep --target t{classes}"
              f"{'' if classes == 4 else 'w'} --pilot --climate temp\n"
              f"   uv run cc -w national_s2 labelling train fit  --target t{classes}"
              f"{'' if classes == 4 else 'w'} --pilot --climate temp --model lightgbm")
    if p is not None:
        p["s2_pred_class"] = p["s2_pred_label"].map(
            TO_4 if classes == 4 else PRED_TO_DECLARED)
        df = df.merge(p, on="COD_PREDIO", how="left")

    cols = COLUMNS4 if classes == 4 else COLUMNS
    if classes == 4:                                     # the declared side has no woody class
        for c in ("pett_class", "cen_class"):
            df[c] = df[c].map(TO_4) if c in df.columns else pd.NA
    for c in cols:                                       # a missing input leaves its column
        if c not in df.columns:                          # present and null, never absent
            df[c] = pd.NA
    seen = ["pett_class", "cen_class", "s2_class", "s2_pred_class"]
    df["n_observations"] = df[seen].notna().sum(axis=1)
    df = df[cols].sort_values(["dept", "COD_PREDIO"]).reset_index(drop=True)

    if verbose:
        _summarise(df)
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        (df.to_csv(out, index=False) if out.suffix == ".csv"
         else df.to_parquet(out, index=False))
        print(f"wrote {out} — {len(df):,} rows x {len(df.columns)} columns")
    return df


def _summarise(df: pd.DataFrame) -> None:
    """What is actually in the table, so a null column is visible before it is averaged."""
    print(f"=== parcel table: {len(df):,} parcels over "
          f"{df['dept'].nunique()} departments ===")
    print(f"  PETT declaration  {df.pett_year.min():.0f}-{df.pett_year.max():.0f}: "
          f"{df.pett_class.notna().sum():>7,}")
    print(f"  CENAGRO 2012                 : {df.cen_class.notna().sum():>7,}")
    mapped = int(df.s2_class.notna().sum())
    note = ("" if mapped == int(df.s2_label.notna().sum()) else
            f" ({mapped} mapped; the rest are WOODY_NON_CROP / NON_AGRICULTURE, read "
            f"`s2_class_woody_perennial` beside them)")
    print(f"  human label, 2019+ imagery   : {df.s2_label.notna().sum():>7,}{note}")
    print(f"  classifier, 2019+ imagery    : {df.s2_pred_class.notna().sum():>7,} "
          f"{df.s2_pred_source.value_counts().to_dict()}")
    per_parcel = df["n_observations"].value_counts().sort_index().to_dict()
    print(f"  observations per parcel      : {per_parcel}")
    if int(df.s2_class.notna().sum()):
        print(f"⚠️ any share off the imagery columns must use `weight` — the campaign "
              f"over-sampled\n   PERENNIAL ~3x, and these "
              f"{int(df.s2_class.notna().sum())} parcels are not a sample of the rest "
              f"(RESULTS.md §8.6).")

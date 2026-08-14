"""Is ``PERENNIAL`` the same thing as *export*? (window_plan.md §6.4)

The 3-class label space proxies the research question — "has land shifted from domestic
annual crops to export perennial crops?" — with a land-*state* classifier, because that is
what a 30 m Landsat year can separate. The proxy is imperfect in both directions:

* **perennial ≠ export.** Coffee, plátano and naranja are the largest tokens in the
  ``PERENNIAL`` lexicon and are substantially domestic; coca, manzana, membrillo, níspero,
  durazno and tuna are essentially all domestic.
* **export ≠ perennial.** Asparagus, paprika and, later, blueberry are export *annuals* and
  sit in the ``ANNUAL`` class by construction.

This module makes the size of that gap measurable instead of assumed: it maps the perennial
lexicon onto an explicit export basket and reports what fraction of ``PERENNIAL``-labelled
parcels actually carry an export-oriented crop, nationally and by department. It bounds the
interpretation; it does not change any label.

The basket is a **judgement call about Peruvian agro-export**, written down so it can be
argued with. ``MIXED`` is used where a crop is genuinely both (organic banana from Piura is
exported, most plátano is not).

Run with::

    CC_PROC=data/processed/all_peru uv run python -m crop_classifier.allperu.export_crops
"""

from __future__ import annotations

import argparse
import json

import geopandas as gpd
import numpy as np
import pandas as pd

from crop_classifier.paths import ROOT, proc

# Peru's perennial agro-export basket (SIEA/ADEX top exported crops of the 2010s-20s).
EXPORT = {
    "UVA", "VID", "PARRA",                       # table grape — the largest by value
    "PALTA",                                     # avocado (Hass)
    "MANGO", "MANGO KENT", "MANGO CRIOLLO",
    "CACAO", "CACAOTAL", "CACA",
    "CAFE", "CAFETAL", "CAFE TYPICA", "CAFE CATIMOR", "CAFE CATURRA", "CAFE BOURBON",
    "CAFE PACHE",                                # coffee — export-dominant
    "OLIVO", "OLIVAR", "ACEITUNA",               # olive (Tacna/Moquegua)
    "TARA",                                      # tannin/gum export
    "OREGANO",                                   # Tacna/Moquegua export herb
    "PALMA ACEITERA", "PALMA",                   # oil palm
    "MANDARINA", "TORONJA", "CITRICOS",          # export citrus
    "CAMU CAMU", "CASTAÑA", "AGUAJE",            # Amazon export niches
}

# Both domestic and exported in material volume — reported separately, never silently
# counted as export.
MIXED = {
    "PLATANO", "BANANO", "BANANERO", "PLATANO SEDA", "PLATANO DE SEDA",
    "PLATANO DOMINICO", "PLTANO", "PLATAO", "PLATNO", "PALTANO", "PLANTANO",
    "PLATABNO", "PLATANAO",                      # organic banana IS a Piura export
    "LIMON", "LIMON SUTIL", "LIMA",              # some processed-juice export
    "GRANADILLA", "MARACUYA", "PAPAYA",          # small export volumes
    "FRUTALES", "HUERTO", "HUERTA", "CITRICO",   # unspecified orchard — unresolvable
}

# Everything else in the perennial lexicon is treated as domestic. Listed explicitly for the
# largest tokens so the classification is auditable rather than a fall-through.
DOMESTIC_MAJOR = {
    "MANZANA", "MANZANO", "MANZANOS", "COCA", "NISPERO", "MELOCOTON", "DURAZNO",
    "DURAZNERO", "MEMBRILLO", "NARANJA", "NARANJAO", "NARANAJA", "NARANAJ", "NARANAJO",
    "TUNA", "TUNAL", "TUNALES", "ACHIOTE", "PERA", "PERAL", "PERA ITALIA", "CAPULI",
    "SAUCO", "HIGO", "HIGUERA", "GRANADA", "GUAYABA", "CHIRIMOYA", "LUCUMA", "LUCUMO",
    "PACAE", "PACAY", "GUABO", "GUANABANA", "HUANABANA", "ZAPOTE", "MAMEY", "CIRUELA",
    "CIRUELO", "TAMARINDO", "COCO", "POMARROSA", "PIJUAYO", "MANDARINO",
}


def classify(token: str) -> str:
    """One crop token -> ``EXPORT`` / ``MIXED`` / ``DOMESTIC``."""
    t = str(token).strip().upper()
    if t in EXPORT:
        return "EXPORT"
    if t in MIXED:
        return "MIXED"
    return "DOMESTIC"


def parcel_export_status(crop_set: str) -> str:
    """A parcel's status from its ``crop_set`` (``A+B+C``).

    A parcel counts as ``EXPORT`` if **any** of its crops is an export crop — the export
    reading of the parcel is about whether export production is present, and the label
    itself was already resolved by group priority, not by area share.
    """
    if crop_set is None or (isinstance(crop_set, float) and np.isnan(crop_set)):
        return "UNKNOWN"
    parts = [p for p in str(crop_set).split("+") if p]
    if not parts:
        return "UNKNOWN"
    kinds = {classify(p) for p in parts}
    for k in ("EXPORT", "MIXED"):
        if k in kinds:
            return k
    return "DOMESTIC"


def report(parcels: gpd.GeoDataFrame | pd.DataFrame | None = None,
           weight_col: str = "population_weight", save: bool = True) -> pd.DataFrame:
    """Export composition of the ``PERENNIAL`` class, weighted, overall and by department."""
    if parcels is None:
        parcels = gpd.read_parquet(proc() / "modeling_parcels.parquet")
    p = parcels[parcels["label"] == "PERENNIAL"].copy()
    p["export_status"] = p["crop_set"].map(parcel_export_status)
    w = (p[weight_col].to_numpy(float) if weight_col in p.columns
         else np.ones(len(p), dtype=float))
    p["_w"] = w

    overall = (p.groupby("export_status")["_w"].sum() / w.sum()).rename("share")
    counts = p["export_status"].value_counts().rename("n_parcels")
    out = pd.concat([counts, overall], axis=1).reset_index(names="export_status")

    by_dept = None
    if "dept" in p.columns:
        by_dept = (p.pivot_table(index="dept", columns="export_status", values="_w",
                                 aggfunc="sum", fill_value=0.0))
        by_dept = by_dept.div(by_dept.sum(axis=1), axis=0).round(3)

    top = (p.groupby(["export_status", "crop_set"]).size()
           .sort_values(ascending=False).groupby(level=0).head(6))

    print("PERENNIAL parcels by export status (weighted share):")
    print(out.to_string(index=False))
    if by_dept is not None:
        print("\nby department:")
        print(by_dept.to_string())
    print("\nlargest crop sets per status:")
    print(top.to_string())

    if save:
        out.to_csv(proc() / "export_status_summary.csv", index=False)
        if by_dept is not None:
            by_dept.to_csv(proc() / "export_status_by_dept.csv")
        with open(proc() / "export_status.json", "w") as f:
            json.dump({"weighted_share": overall.to_dict(),
                       "n_parcels": counts.to_dict(),
                       "note": "EXPORT = any crop in the export basket; MIXED = crops that "
                               "are materially both (banana, lime, unspecified orchard); "
                               "DOMESTIC = the rest. Judgement call, see module docstring."},
                      f, indent=2)
        print(f"\nwrote export_status_summary.csv / export_status.json to {proc()}")
    return out


def woody_noncrop_bound(save: bool = True) -> dict:
    """§6.3: how much of the cadastre is *declared* woody non-crop, i.e. a known FP source.

    ``woody_noncrop_policy: exclude`` removes algarrobo, eucalyptus and forest plantations
    from **training**, not from the world: at inference they read as ``PERENNIAL``. This
    bounds the size of that pool from the declarations themselves. It is a **lower** bound —
    it counts only parcels that *declared* a woody non-crop, not natural woody vegetation
    that invaded an abandoned parcel over the following 20 years, which nothing in this
    project observes.
    """
    # The exclusion audit is written by the label build, which for the national strand ran in
    # the FULL workspace; the sampled workspace only holds what survived it.
    f = proc() / "label_exclusions.csv"
    if not f.exists():
        alt = ROOT / "data" / "processed" / "all_peru_full" / "label_exclusions.csv"
        if not alt.exists():
            raise FileNotFoundError(f"{f} — run `perennial labels` in this workspace first")
        f = alt
    ex = pd.read_csv(f).set_index("reason")["n_parcels"]
    # Denominator MUST come from the same workspace as the exclusion counts — the sampled
    # workspace holds 56 k parcels while the exclusions were counted over all 727 k, and
    # mixing them overstates the bound by an order of magnitude.
    labels = pd.read_parquet(f.parent / "modeling_parcels.parquet", columns=["label"])
    n_woody = int(ex.get("woody non-crop (not an export crop)", 0))
    n_eligible = len(labels)
    out = {
        "source": str(f.parent),
        "n_declared_woody_noncrop": n_woody,
        "n_eligible_parcels": n_eligible,
        "share_of_eligible_plus_woody": float(n_woody / (n_eligible + n_woody)),
        "perennial_share_of_eligible": float((labels["label"] == "PERENNIAL").mean()),
        "note": "LOWER bound: counts declared woody non-crop only. If every such parcel "
                "reads PERENNIAL at inference it adds this much to the predicted perennial "
                "pool — compare against the ~2 pp differential the design is powered for.",
    }
    print(json.dumps(out, indent=2))
    if save:
        with open(proc() / "woody_noncrop_bound.json", "w") as fh:
            json.dump(out, fh, indent=2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    report(save=not a.no_save)
    print()
    woody_noncrop_bound(save=not a.no_save)


if __name__ == "__main__":
    main()

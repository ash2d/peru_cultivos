"""CENAGRO 2012 as the "before" observation, instead of the PETT declaration.

Every before/after in this project so far has used the **PETT** declared crop as its
baseline: one observation per parcel, made ~1996–2006 when the parcel was titled. The 2012
agricultural census is a **second, independently collected** observation of what is growing
on (some of) the same parcels, 6–16 years later, and it lets two questions be asked that the
PETT baseline cannot:

1. **PETT → CENAGRO**, a paired change between two *declared* observations on the same
   parcel, with no satellite and no classifier anywhere in it;
2. **CENAGRO → photo-interpreted 2019+**, which replaces a ~1998 baseline with a 2012 one
   and so measures a 7-year rather than a 21-year window.

⚠️ **Read `docs/DATA.md` §2 Chain B before using any of this.** The census carries **no
parcel key** — no `COD_PREDIO`, no `CodigoSSET`. The only link is the **farmer's name**, and
the link is therefore *farmer-level, not parcel-level*: a producer with three parcels is
matched to a person, and which of their polygons a census row refers to is uncertain. Even
on the recovered `Base_Cenagro_PETT_Piura` crosswalk, an independent name match reproduces
the same `COD_PREDIO` only ~43 % of the time. `merged_parcels.parquet` carries
`link_confidence` (high/medium/low) and every figure here is reported broken out by it,
because that is the only honest way to show how much of the answer is the link rather than
the land.

⚠️ **CENAGRO exists for Piura only** (`IV_CENAGRO_Piura.dta`). Nothing here is national, and
Piura is the department the whole project's earlier strands were built on — so it is *not* a
neutral sample of Peru.

**Both sides are classified with the same lexicon machinery** (`perennial/labels3.py`),
because otherwise part of any measured "change" would be a change of definition. The census
config (`config/perennial_cenagro.yaml`) is `perennial_allperu.yaml` **plus tokens only** —
no existing assignment is altered.

⚠️ **That extension was necessary, and finding out why is the point.** The census uses
fuller crop names than the titling registry ("LIMON ACIDO" not "LIMON"). Audited over 59,855
census token-instances, 2,448 (**4.09 %**) fell through to `crop_fallback: ANNUAL` — and
**1,969 of them were the single token `VERGEL FRUTICOLA`, literally "fruit orchard"**. Left
alone, the largest unmapped token in the census would have been silently counted as an
annual crop, biasing the 2012 perennial share *down* and the measured shift *up*.
`token_audit()` below reproduces the check; run it before trusting any number here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crop_classifier.paths import ROOT

CONFIG = ROOT / "src" / "crop_classifier" / "config" / "perennial_cenagro.yaml"
MERGED = ROOT / "data" / "processed" / "merged_parcels.parquet"
PETT_NATIONAL = ROOT / "data" / "processed" / "all_peru_full" / "modeling_parcels.parquet"
LABELS_S2 = ROOT / "data" / "processed" / "all_peru" / "labels_s2" / "labelled_parcels.parquet"
OUT_DIR = ROOT / "data" / "processed" / "cenagro"

# The census lists a parcel's crops separated by " | ".
CROP_SEP = "|"
# The photo-interpreted classes, collapsed onto the declared label space so the two can be
# crossed. `WOODY_NON_CROP` deliberately has **no** mapping: it is the codebook's hardest
# call and folding it either way is the decision, not a preprocessing step — both readings
# are reported side by side instead.
S2_TO_DECLARED = {"PERENNIAL": "PERENNIAL", "ANNUAL": "ANNUAL",
                  "OTHER": "PASTURE_FALLOW", "NON_AGRICULTURE": None,
                  "WOODY_NON_CROP": None}


# ---------------------------------------------------------------------------------
def classify_crop_list(series: pd.Series, config_path: Path | None = None) -> pd.Series:
    """`"LIMON ACIDO | MELON"` -> `PERENNIAL`, via the project's own 3-class lexicon.

    Uses `crop_normalization.normalize_label` for the token split and
    `labels3.assign_group` for the group, so a census crop name is resolved by exactly the
    rules a PETT crop name is. `None` where nothing in the cell resolves.
    """
    from crop_classifier.crop_normalization import normalize_label
    from crop_classifier.perennial.labels3 import (
        _stage_regex,
        assign_group,
        build_resolver,
        load_config,
    )

    cfg = load_config(config_path or CONFIG)
    resolver = build_resolver(cfg)
    stage_re = _stage_regex(cfg)

    cache: dict[str, str | None] = {}

    def one(cell: object) -> str | None:
        if not isinstance(cell, str) or not cell.strip():
            return None
        if cell in cache:
            return cache[cell]
        crops, cats = [], []
        for piece in cell.split(CROP_SEP):
            for crop, cat in normalize_label(piece):
                crops.append(crop)
                cats.append(cat)
        g = assign_group(crops, cats, cfg, resolver, stage_re)[0] if crops else None
        cache[cell] = g
        return g

    return series.map(one)


def token_audit(config_path: Path | None = None) -> pd.DataFrame:
    """Every distinct census crop token, what it resolved to, and how.

    The row that matters is ``source == "crop_fallback"``: a token the lexicon has no entry
    for, assigned `ANNUAL` because annuals dominate the unlisted tail. A large fallback share
    does not raise — it just quietly moves the answer — so this is printed as a share and
    compared against the config's own ``max_unassigned_frac`` budget.
    """
    from collections import Counter

    from crop_classifier.crop_normalization import normalize_label
    from crop_classifier.perennial.labels3 import (
        _stage_regex,
        build_resolver,
        load_config,
        resolve_token,
    )

    cfg = load_config(config_path or CONFIG)
    resolver = build_resolver(cfg)
    stage_re = _stage_regex(cfg)
    m = pd.read_parquet(MERGED, columns=["cen_crops_all"])
    tok: Counter = Counter()
    for cell in m["cen_crops_all"].dropna():
        for piece in str(cell).split(CROP_SEP):
            for crop, cat in normalize_label(piece):
                tok[(crop,) + resolve_token(crop, cat, cfg, resolver, stage_re)] += 1
    d = pd.DataFrame([{"token": k[0], "group": k[1], "source": k[2], "n_instances": v}
                      for k, v in tok.items()]).sort_values("n_instances", ascending=False)
    total = d["n_instances"].sum()
    fb = d[d["source"] == "crop_fallback"]["n_instances"].sum()
    budget = cfg.get("max_unassigned_frac", 0.02)
    print(f"census lexicon: {len(d)} distinct tokens over {total:,} instances; "
          f"unmapped -> {cfg['crop_fallback']}: {fb:,} ({fb / total:.2%}, "
          f"budget {budget:.0%}) {'OK' if fb / total <= budget else 'OVER BUDGET'}")
    return d


def _share_table(df: pd.DataFrame, before: str, after: str,
                 classes: tuple[str, ...] = ("PERENNIAL", "ANNUAL", "PASTURE_FALLOW")
                 ) -> pd.DataFrame:
    """Composition before, composition after, and the change — in percentage points.

    Percentage *points* rather than a ratio because a share that moves 4 % → 8 % is a
    +4 pp change and a doubling, and only one of those two numbers is additive across
    classes. Both are given.
    """
    n = len(df)
    rows = []
    for c in classes:
        b = float((df[before] == c).mean())
        a = float((df[after] == c).mean())
        # binomial SE of a paired difference is not the difference of the marginal SEs;
        # McNemar's discordant pairs are what carry the information
        b_only = int(((df[before] == c) & (df[after] != c)).sum())
        a_only = int(((df[before] != c) & (df[after] == c)).sum())
        disc = b_only + a_only
        se = np.sqrt(disc) / n if disc else 0.0
        rows.append({
            "class": c,
            f"{before}_pct": round(100 * b, 1),
            f"{after}_pct": round(100 * a, 1),
            "change_pp": round(100 * (a - b), 1),
            "ci95_pp": round(100 * 1.96 * se, 1),
            "ratio": round(a / b, 2) if b else np.nan,
            "n_left_class": b_only, "n_joined_class": a_only,
        })
    out = pd.DataFrame(rows)
    out.attrs["n"] = n
    return out


# ---------------------------------------------------------------------------------
def build(config_path: Path | None = None, save: bool = True) -> dict:
    """The whole comparison. Returns the tables; prints them with their units."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}

    merged = pd.read_parquet(
        MERGED, columns=["COD_PREDIO", "CodigoSSET", "link_confidence",
                         "cen_crops_all", "cen_crop_primary", "cen_area_ha_sum",
                         "sset_crops_all", "parc_area_ha_cen"])
    merged["COD_PREDIO"] = merged["COD_PREDIO"].astype(str)
    print(f"CENAGRO<->PETT crosswalk: {len(merged):,} rows, "
          f"{merged.COD_PREDIO.nunique():,} distinct polygons (Piura only)")

    # ---- classify the census side with the project's own lexicon ----
    merged["cen_class"] = classify_crop_list(merged["cen_crops_all"], config_path)
    res = merged["cen_class"].notna().mean()
    print(f"census crop lists resolved to a 3-class label: {res:.1%} "
          f"({int(merged['cen_class'].isna().sum()):,} unresolved or blank)")

    # one row per polygon: a producer's parcels can repeat a COD_PREDIO, and the
    # crosswalk is farmer-level, so collapse on the highest-priority class present
    prio = {"PERENNIAL": 0, "ANNUAL": 1, "PASTURE_FALLOW": 2}
    merged["_p"] = merged["cen_class"].map(prio)
    cen = (merged.sort_values("_p")
           .groupby("COD_PREDIO", as_index=False)
           .agg(cen_class=("cen_class", "first"),
                link_confidence=("link_confidence", "first"),
                cen_area_ha=("cen_area_ha_sum", "sum")))
    cen = cen[cen["cen_class"].notna()]

    # ---- the PETT side: the project's canonical declared label ----
    pett = pd.read_parquet(PETT_NATIONAL,
                           columns=["COD_PREDIO", "dept", "label", "year", "area_ha"])
    pett["COD_PREDIO"] = pett["COD_PREDIO"].astype(str)
    pett = pett.rename(columns={"label": "pett_class", "year": "pett_year"})

    paired = cen.merge(pett, on="COD_PREDIO", how="inner")
    out["paired"] = paired
    print(f"\nparcels with BOTH a PETT declaration and a CENAGRO 2012 observation: "
          f"{len(paired):,}")
    print(f"  PETT declaration years: {paired.pett_year.min():.0f}"
          f"-{paired.pett_year.max():.0f} (median {paired.pett_year.median():.0f})")

    # ---- 1. PETT -> CENAGRO, all matched parcels ----
    t1 = _share_table(paired, "pett_class", "cen_class")
    out["pett_to_cenagro"] = t1
    print(f"\n=== 1. PETT declaration -> CENAGRO 2012, same parcel, n={len(paired):,} ===")
    print("shares of parcels, %; change in percentage points with a paired (McNemar) CI")
    print(t1.to_string(index=False))

    # ---- 1b. ⚠️ the comparison that is actually like-for-like ----
    # The two instruments do not share a class space. PETT recorded a land *state* and has
    # an explicit "EN DESCANSO" token; CENAGRO question 024 asks which crop is grown, so a
    # parcel lying fallow contributes no crop row at all and simply leaves the frame. That
    # is why PASTURE_FALLOW appears to collapse 17.9 % -> 1.1 %: it is not land change, it
    # is the census having no way to say it. The only defensible comparison is therefore
    # **conditional on a crop being recorded on both sides**.
    crop_only = paired[paired.pett_class.isin(["PERENNIAL", "ANNUAL"])
                       & paired.cen_class.isin(["PERENNIAL", "ANNUAL"])].copy()
    t1b = _share_table(crop_only, "pett_class", "cen_class",
                       classes=("PERENNIAL", "ANNUAL"))
    out["pett_to_cenagro_cropped_only"] = t1b
    print(f"\n=== 1b. ⭐ the like-for-like comparison: parcels with a CROP recorded on "
          f"both sides, n={len(crop_only):,} ===")
    print("PASTURE_FALLOW is dropped from BOTH sides. The census cannot record fallow — a "
          "parcel\nwith no crop contributes no row — so the apparent collapse above is an "
          "instrument\ndifference, not land change.")
    print(t1b.to_string(index=False))

    # area-weighted, on the CADASTRAL area. The census self-reported parcel area is
    # uncorrelated with the cadastral polygon area (Pearson ~0.01, DATA.md Chain B) and
    # must not be used as a weight.
    w = crop_only["area_ha"]
    ar = []
    for c in ("PERENNIAL", "ANNUAL"):
        b_ = float(w[crop_only.pett_class == c].sum() / w.sum())
        a_ = float(w[crop_only.cen_class == c].sum() / w.sum())
        ar.append({"class": c, "pett_pct_of_area": round(100 * b_, 1),
                   "cenagro_pct_of_area": round(100 * a_, 1),
                   "change_pp": round(100 * (a_ - b_), 1)})
    t1c = pd.DataFrame(ar)
    out["pett_to_cenagro_area_weighted"] = t1c
    print(f"\nthe same weighted by CADASTRAL area ({w.sum():,.0f} ha total):")
    print(t1c.to_string(index=False))

    # gross flows, which a net share hides entirely
    print("\ngross parcel flows (counts), PETT declaration -> CENAGRO 2012:")
    print(pd.crosstab(crop_only.pett_class, crop_only.cen_class,
                      rownames=["PETT ~1999"], colnames=["CENAGRO 2012"]).to_string())

    # ---- 2. the same, split by how good the census link is ----
    rows = []
    for lc in ["high", "medium", "low"]:
        sub = paired[paired.link_confidence == lc]
        if len(sub) < 50:
            continue
        t = _share_table(sub, "pett_class", "cen_class")
        t.insert(0, "link_confidence", lc)
        t.insert(1, "n", len(sub))
        rows.append(t)
    t2 = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out["by_link_confidence"] = t2
    print("\n=== 2. the same, by census link quality ===")
    print("⚠️ if the answer moves with link quality, the answer is partly the link")
    print(t2.to_string(index=False))

    # ---- 3. CENAGRO 2012 -> photo-interpreted 2019+ ----
    if LABELS_S2.exists():
        import geopandas as gpd
        lab = gpd.read_parquet(LABELS_S2)
        lab = lab[lab["usable"]].copy()
        lab["COD_PREDIO"] = lab["COD_PREDIO"].astype(str)
        lab = lab[["COD_PREDIO", "label", "dept", "declared_class"]].rename(
            columns={"label": "s2_label"})
        j = cen.merge(lab, on="COD_PREDIO", how="inner")
        out["cenagro_to_s2"] = j
        print("\n=== 3. CENAGRO 2012 -> photo-interpreted 2019+ ===")
        print(f"parcels with BOTH a CENAGRO observation and a 2019+ human label: "
              f"**{len(j)}**")
        if len(j):
            print(pd.crosstab(j["cen_class"], j["s2_label"]).to_string())
    if save:
        for k, v in out.items():
            if isinstance(v, pd.DataFrame) and len(v):
                v.to_csv(OUT_DIR / f"{k}.csv", index=False)
        print(f"\nwrote {len(out)} tables to {OUT_DIR}")
    return out

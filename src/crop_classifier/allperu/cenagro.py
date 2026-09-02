"""CENAGRO 2012 as the "before" observation, instead of the PETT declaration.

The 2012 census is a second, independently collected observation of what grows on (some of)
the same parcels, 6–16 years after the PETT declaration. It lets two questions be asked:
PETT → CENAGRO (a paired change between two *declared* observations, no satellite anywhere)
and CENAGRO → photo-interpreted 2019+ (a 7-year rather than 21-year window).

⚠️ Read `DATA.md` §2 Chain B first. The census carries no parcel key — the only link is the
farmer's name, so it is farmer-level, not parcel-level. Even on the recovered
`Base_Cenagro_PETT_Piura` crosswalk, an independent name match reproduces `COD_PREDIO` only
~43 % of the time. Every figure here is broken out by `link_confidence` (high/medium/low).

⚠️ Piura only (reads `merged_parcels.parquet`), so not a neutral sample of Peru.

⭐ `allperu/cenagro_shift.py` repeats this over all 14 linkable departments on a nationally
built name link — 63,766 like-for-like parcels vs 8,669 here (`RESULTS.md` §8.6). This
module is the first, independently built version; its Piura answer (+12.5 pp) vs the
national build's Piura arm (+11.6 pp) is the only external check either has. **For anything
national, use `cenagro_shift.py`.**

Both sides use the same lexicon machinery (`perennial/labels3.py`), or part of any "change"
would be a change of definition. The census config is `perennial_allperu.yaml` plus tokens
only. ⚠️ The census uses fuller crop names ("LIMON ACIDO" not "LIMON"): of 59,855 token
instances, 2,448 (4.09 %) fell through to `crop_fallback: ANNUAL`, 1,969 of them the single
token `VERGEL FRUTICOLA` ("fruit orchard") — left alone it biases the shift *up*.
`token_audit()` reproduces the check.
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

CROP_SEP = "|"
# Photo-interpreted classes collapsed onto the declared label space. `WOODY_NON_CROP` is
# left unmapped on purpose — the codebook's hardest call; both readings are reported.
S2_TO_DECLARED = {"PERENNIAL": "PERENNIAL", "ANNUAL": "ANNUAL",
                  "OTHER": "PASTURE_FALLOW", "NON_AGRICULTURE": None,
                  "WOODY_NON_CROP": None}


def classify_crop_list(series: pd.Series, config_path: Path | None = None) -> pd.Series:
    """`"LIMON ACIDO | MELON"` -> `PERENNIAL`, via the project's own 3-class lexicon.

    Same `normalize_label` + `assign_group` path a PETT crop name takes. `None` where
    nothing in the cell resolves.
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

    ``source == "crop_fallback"`` is a token with no lexicon entry, assigned `ANNUAL`; a
    large fallback share does not raise, it just moves the answer, so it is printed as a
    share against the config's ``max_unassigned_frac`` budget.
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
    """Composition before, after, and the change in percentage points (plus the ratio —
    4 % → 8 % is +4 pp *and* a doubling, and only the pp is additive across classes)."""
    n = len(df)
    rows = []
    for c in classes:
        b = float((df[before] == c).mean())
        a = float((df[after] == c).mean())
        # paired difference: McNemar's discordant pairs carry the information, not the marginals
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

    # one row per polygon: the crosswalk is farmer-level, so collapse on the highest-priority class
    prio = {"PERENNIAL": 0, "ANNUAL": 1, "PASTURE_FALLOW": 2}
    merged["_p"] = merged["cen_class"].map(prio)
    cen = (merged.sort_values("_p")
           .groupby("COD_PREDIO", as_index=False)
           .agg(cen_class=("cen_class", "first"),
                link_confidence=("link_confidence", "first"),
                cen_area_ha=("cen_area_ha_sum", "sum")))
    cen = cen[cen["cen_class"].notna()]

    # ---- the PETT side: the project's canonical declared label ----
    if not PETT_NATIONAL.exists():
        raise SystemExit(
            f"{PETT_NATIONAL.relative_to(ROOT)} is 437 MB and is not committed, and this "
            f"Piura-only comparison rebuilds from it. The national comparison that "
            f"supersedes it does reproduce from the clone: `cc -w national analysis "
            f"perennial-shift` (docs/howto/08_perennial_change_by_tenure.md).")
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

    # ---- 1b. ⚠️ the like-for-like comparison ----
    # The instruments don't share a class space: CENAGRO Q024 asks which crop is grown, so a
    # fallow parcel contributes no row and leaves the frame — PASTURE_FALLOW "collapses"
    # 17.9 -> 1.1 % as an instrument artefact. Only defensible comparison: conditional on a
    # crop recorded on both sides.
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

    # weighted by CADASTRAL area — the census self-reported area is uncorrelated with it
    # (Pearson ~0.01, DATA.md Chain B) and must not be a weight.
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

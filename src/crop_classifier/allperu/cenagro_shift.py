"""PETT declaration → CENAGRO 2012 → photo-interpreted 2019+, nationally, split by tenure.

Three observations of the same parcels from three unrelated instruments:

| # | observation | when | how |
|---|---|---|---|
| 1 | PETT declared crop | ~1996–2006 | the farmer told a titling clerk |
| 2 | CENAGRO question 024 | 2012 | the farmer told a census enumerator |
| 3 | photo-interpretation | 2019+ | a human read Sentinel-2 / Esri imagery |

1→2 uses no satellite and no classifier — two declared observations of the same land, the
cleanest change measurement this project has. Each contrast is reported as perennial share
of parcels and of cadastral area, split by tenure at the PETT declaration (`ESTADO en RRPP`).

⚠️ Descriptive, not causal — title is not randomly assigned, so a tenure gap in the change
is a fact about the two groups. The causal estimate is the v3 DiD (`RESULTS.md` §7).

⚠️ Three silent biases, each handled explicitly:
1. The census cannot record fallow (Q024 asks which crop is grown), so the like-for-like
   comparison is conditional on a crop recorded on both sides; both versions are printed.
2. The S2 sample over-samples PERENNIAL ~3×, so every S2 share uses the design weight
   `weight` (= N_h/n_h), unweighted printed beside it.
3. Area is weighted by cadastral polygon area, never the census `P037_SS` (Pearson ~0.01
   with it, `DATA.md` Chain B).

⚠️ The census link is farmer-level, not parcel-level (`cenagro_link.py`); 1→2 is broken out
by `link_confidence` — if the answer moves with link quality, part of it is the link.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crop_classifier.paths import ROOT

CENAGRO_DIR = ROOT / "data" / "raw" / "Cenagro_IV"
LINK = ROOT / "data" / "processed" / "cenagro" / "cenagro_pett_link.parquet"
PETT = ROOT / "data" / "processed" / "all_peru_full" / "modeling_parcels.parquet"
# `PETT` (437 MB) is not committed; two small derived files stand in so this reproduces from
# a clone: the built panel, and the dept x declared-class counts post-stratification needs.
PANEL = ROOT / "data" / "processed" / "cenagro" / "national_panel.parquet"
PETT_POP = ROOT / "data" / "processed" / "cenagro" / "pett_population_by_dept_class.csv"
TENURE = ROOT / "data" / "processed" / "all_peru" / "tenure_by_predio.parquet"
LABELS_S2 = ROOT / "data" / "processed" / "all_peru" / "labels_s2" / "labelled_parcels.parquet"
CONFIG = ROOT / "src" / "crop_classifier" / "config" / "perennial_cenagro.yaml"
OUT_DIR = ROOT / "data" / "processed" / "cenagro"

CROP_SEP = " | "
CLASSES = ("PERENNIAL", "ANNUAL", "PASTURE_FALLOW")

# Photo-interpreted classes folded onto the declared label space. `WOODY_NON_CROP` is left
# unmapped on purpose — the codebook's hardest call; both readings are reported.
S2_TO_DECLARED = {"PERENNIAL": "PERENNIAL", "ANNUAL": "ANNUAL",
                  "OTHER": "PASTURE_FALLOW", "NON_AGRICULTURE": None,
                  "WOODY_NON_CROP": None}


def _lexicon():
    from crop_classifier.perennial.labels3 import (
        _stage_regex,
        build_resolver,
        load_config,
    )
    cfg = load_config(CONFIG)
    return cfg, build_resolver(cfg), _stage_regex(cfg)


def classify_crop_list(series: pd.Series) -> pd.Series:
    """`"LIMON ACIDO | MELON"` -> `PERENNIAL`, via the project's own 3-class lexicon.

    Census and PETT sides use the same rules, so a measured change is land, not definition.
    """
    from crop_classifier.crop_normalization import normalize_label
    from crop_classifier.perennial.labels3 import assign_group

    cfg, resolver, stage_re = _lexicon()
    cache: dict[str, str | None] = {}

    def one(cell: object) -> str | None:
        if not isinstance(cell, str) or not cell.strip():
            return None
        if cell in cache:
            return cache[cell]
        crops, cats = [], []
        for piece in cell.split("|"):
            for crop, cat in normalize_label(piece):
                crops.append(crop)
                cats.append(cat)
        g = assign_group(crops, cats, cfg, resolver, stage_re)[0] if crops else None
        cache[cell] = g
        return g

    return series.map(one)


def crop_names() -> dict[int, str]:
    from crop_classifier.allperu.cenagro_extract import crop_code_table
    t = crop_code_table()
    return dict(zip(t["code"].astype(int), t["TITULO"].astype(str).str.strip()))


def census_producer_crops(depts: list[str]) -> pd.DataFrame:
    """One row per census producer: every crop it declared in 2012, and their sown area."""
    code2crop = crop_names()
    out = []
    for d in depts:
        c = pd.read_parquet(CENAGRO_DIR / f"{d}.parquet",
                            columns=["NPRIN", "P024_03", "P025", "P029_02"])
        code = pd.to_numeric(c["P024_03"], errors="coerce").astype("Int64")
        c = c[code.notna()].copy()
        c["crop"] = code[code.notna()].astype(int).map(code2crop)
        g = c.groupby("NPRIN").agg(
            cen_crops_all=("crop", lambda s: CROP_SEP.join(sorted({x for x in s if x}))),
            cen_sown_ha=("P025", "sum"),
            cen_n_crop_rows=("crop", "size"),
            cen_any_export=("P029_02", lambda s: float((s == 1).any())))
        g["dept"] = d
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def token_audit(depts: list[str]) -> pd.DataFrame:
    """Every distinct census crop token nationally, what it resolved to, and how.

    ⚠️ Run before trusting any number below. ``source == "crop_fallback"`` is a token with
    no lexicon entry, assigned `ANNUAL`; a large fallback share does not raise, it just
    moves the answer. Piura: 4.09 % fell through, 80 % of it `VERGEL FRUTICOLA` ("fruit
    orchard") — fixing it moved the headline +2.4 pp -> +12.5 pp. National vocab is larger.
    """
    from collections import Counter

    from crop_classifier.crop_normalization import normalize_label
    from crop_classifier.perennial.labels3 import resolve_token

    cfg, resolver, stage_re = _lexicon()
    code2crop = crop_names()
    tok: Counter = Counter()
    for d in depts:
        c = pd.read_parquet(CENAGRO_DIR / f"{d}.parquet", columns=["P024_03"])
        code = pd.to_numeric(c["P024_03"], errors="coerce").astype("Int64").dropna()
        for name, n in code.astype(int).map(code2crop).value_counts().items():
            for crop, cat in normalize_label(str(name)):
                tok[(crop,) + resolve_token(crop, cat, cfg, resolver, stage_re)] += n
    d_ = pd.DataFrame([{"token": k[0], "group": k[1], "source": k[2], "n_instances": v}
                       for k, v in tok.items()]).sort_values("n_instances", ascending=False)
    total = d_["n_instances"].sum()
    fb = d_.loc[d_["source"] == "crop_fallback", "n_instances"].sum()
    budget = cfg.get("max_unassigned_frac", 0.02)
    print(f"national census lexicon: {len(d_):,} distinct tokens over {total:,} "
          f"crop-row instances\n  unmapped -> {cfg['crop_fallback']}: {fb:,} "
          f"({fb / total:.2%}, budget {budget:.0%}) "
          f"{'OK' if fb / total <= budget else '⚠️ OVER BUDGET'}")
    top = d_[d_["source"] == "crop_fallback"].head(10)
    if len(top):
        print("  ⚠️ the unmapped tail, sorted by frequency — read the top ten:")
        for _, r in top.iterrows():
            print(f"     {r.token[:44]:<44} {r.n_instances:>9,} "
                  f"({r.n_instances / max(fb, 1):.1%} of the tail)")
    return d_


def _share_row(before: pd.Series, after: pd.Series, cls: str,
               w: pd.Series | None = None) -> dict:
    """One class's before/after share and the paired change, weighted or not.

    CI is McNemar's — a paired difference's information is in the discordant pairs, so a
    per-margin binomial SE would overstate it.
    """
    if w is None:
        w = pd.Series(1.0, index=before.index)
    tot = float(w.sum())
    b = float(w[before == cls].sum()) / tot
    a = float(w[after == cls].sum()) / tot
    b_only = (before == cls) & (after != cls)
    a_only = (before != cls) & (after == cls)
    # effective discordant count under weighting (Kish): (Σw)² / Σw²
    wd = w[b_only | a_only]
    n_eff = (float(w.sum()) ** 2 / float((w ** 2).sum())) if (w ** 2).sum() else 0.0
    p_disc = float(wd.sum()) / tot if tot else 0.0
    se = np.sqrt(p_disc / n_eff) if n_eff else 0.0
    return {"class": cls,
            "before_pct": round(100 * b, 1), "after_pct": round(100 * a, 1),
            "change_pp": round(100 * (a - b), 1),
            "ci95_pp": round(100 * 1.96 * se, 1),
            "ratio": round(a / b, 2) if b else np.nan,
            "n_left": int(b_only.sum()), "n_joined": int(a_only.sum())}


def share_table(df: pd.DataFrame, before: str, after: str,
                classes: tuple[str, ...] = CLASSES,
                weight: str | None = None) -> pd.DataFrame:
    w = df[weight] if weight else None
    out = pd.DataFrame([_share_row(df[before], df[after], c, w) for c in classes])
    out.attrs["n"] = len(df)
    return out


def perennial_change(df: pd.DataFrame, before: str, after: str,
                     weight: str | None = None) -> dict:
    """Just the PERENNIAL row, as a dict — the unit of every tenure comparison below."""
    w = df[weight] if weight else None
    r = _share_row(df[before], df[after], "PERENNIAL", w)
    r["n"] = len(df)
    return r


def by_tenure(df: pd.DataFrame, before: str, after: str, weight: str | None = None,
              label: str = "") -> pd.DataFrame:
    """PERENNIAL change for INSCRITO vs NO INSCRITO, and the difference between them.

    The difference row is the number the question asks for; its CI adds the variances (the
    two groups are disjoint sets of parcels).
    """
    rows = []
    for t in ["INSCRITO", "NO INSCRITO"]:
        sub = df[df["tenure"] == t]
        if not len(sub):
            continue
        r = perennial_change(sub, before, after, weight)
        r["tenure"] = t
        rows.append(r)
    if len(rows) == 2:
        i, n = rows[0], rows[1]
        se = np.sqrt((i["ci95_pp"] / 1.96) ** 2 + (n["ci95_pp"] / 1.96) ** 2)
        rows.append({"tenure": "difference (INSCRITO − NO INSCRITO)",
                     "before_pct": round(i["before_pct"] - n["before_pct"], 1),
                     "after_pct": round(i["after_pct"] - n["after_pct"], 1),
                     "change_pp": round(i["change_pp"] - n["change_pp"], 1),
                     "ci95_pp": round(1.96 * se, 1),
                     "n": i["n"] + n["n"]})
    out = pd.DataFrame(rows)
    out.insert(0, "comparison", label)
    cols = ["comparison", "tenure", "n", "before_pct", "after_pct", "change_pp", "ci95_pp"]
    return out[[c for c in cols if c in out.columns]]


def area_shares(df: pd.DataFrame, before: str, after: str, area: str = "area_ha",
                weight: str | None = None) -> pd.DataFrame:
    """The same comparison weighted by cadastral polygon area, in hectares."""
    w = df[area] * (df[weight] if weight else 1.0)
    tot = float(w.sum())
    rows = []
    for c in CLASSES:
        b = float(w[df[before] == c].sum()) / tot
        a = float(w[df[after] == c].sum()) / tot
        rows.append({"class": c, "before_pct_of_area": round(100 * b, 1),
                     "after_pct_of_area": round(100 * a, 1),
                     "change_pp": round(100 * (a - b), 1)})
    out = pd.DataFrame(rows)
    out.attrs["total_ha"] = tot
    return out


def build_panel(depts: list[str] | None = None) -> pd.DataFrame:
    """One row per parcel: PETT declared class, CENAGRO 2012 class, tenure, area, S2 label.

    The census side is collapsed onto the parcel by class priority (PERENNIAL first): the
    link is farmer-level, several producers can hit one parcel, and any perennial recorded
    makes it perennial under `group_priority`.
    """
    if not PETT.exists():
        if not PANEL.exists():
            raise SystemExit(
                f"needs either {PETT.relative_to(ROOT)} (the full national parcel table, "
                f"not committed) or the committed {PANEL.relative_to(ROOT)}. "
                f"docs/DATA_ACCESS.md")
        print(f"  reading the committed panel {PANEL.relative_to(ROOT)} — the full national "
              f"parcel table is not present, so the panel is not rebuilt from the link")
        df = pd.read_parquet(PANEL)
        return df[df["dept"].isin(depts)] if depts else df

    link = pd.read_parquet(LINK)
    if depts:
        link = link[link["dept"].isin(depts)]
    depts = sorted(link["dept"].unique())

    cen = census_producer_crops(depts)
    cen["cen_class"] = classify_crop_list(cen["cen_crops_all"])

    j = link.merge(cen.drop(columns="dept"), on="NPRIN", how="inner")
    j = j[j["cen_class"].notna()]
    prio = {"PERENNIAL": 0, "ANNUAL": 1, "PASTURE_FALLOW": 2}
    conf = {"high": 0, "medium": 1, "low": 2}
    j["_p"] = j["cen_class"].map(prio)
    j["_c"] = j["link_confidence"].map(conf)
    parcel = (j.sort_values(["_c", "_p"])
              .groupby("COD_PREDIO", as_index=False)
              .agg(cen_class=("cen_class", "first"),
                   link_confidence=("link_confidence", "first"),
                   cen_sown_ha=("cen_sown_ha", "sum"),
                   cen_any_export=("cen_any_export", "max"),
                   n_producers=("NPRIN", "nunique")))

    pett = pd.read_parquet(PETT, columns=["COD_PREDIO", "dept", "label", "year", "area_ha"])
    pett["COD_PREDIO"] = pett["COD_PREDIO"].astype(str)
    pett = pett.rename(columns={"label": "pett_class", "year": "pett_year"})
    parcel["COD_PREDIO"] = parcel["COD_PREDIO"].astype(str)
    df = parcel.merge(pett, on="COD_PREDIO", how="inner")

    ten = pd.read_parquet(TENURE, columns=["COD_PREDIO", "tenure", "frac_inscrito"])
    ten["COD_PREDIO"] = ten["COD_PREDIO"].astype(str)
    df = df.merge(ten, on="COD_PREDIO", how="left")

    # ⚠️ "before" must precede "after" — a few PETT declarations post-date the 2012 census.
    late = int((df["pett_year"] >= 2012).sum())
    if late:
        print(f"  dropped {late} parcels whose PETT declaration is 2012 or later — the "
              f"'before' must precede the census")
        df = df[df["pett_year"] < 2012]
    return df


def s2_panel() -> pd.DataFrame:
    """The photo-interpreted parcels, with their design weight and tenure.

    ⚠️ `weight` is not optional — the stratified sample over-samples PERENNIAL, so this
    table's unweighted perennial share is ~3× the population's.
    """
    s2 = pd.read_parquet(LABELS_S2, columns=["COD_PREDIO", "label", "dept", "declared_class",
                                             "stratum", "weight", "usable", "area_ha"])
    s2 = s2[s2["usable"]].copy()
    s2["COD_PREDIO"] = s2["COD_PREDIO"].astype(str)
    s2["s2_class"] = s2["label"].map(S2_TO_DECLARED)
    # the alternative reading: WOODY_NON_CROP counted as perennial canopy
    s2["s2_class_woody_perennial"] = s2["s2_class"].where(
        s2["label"] != "WOODY_NON_CROP", "PERENNIAL")
    ten = pd.read_parquet(TENURE, columns=["COD_PREDIO", "tenure"])
    ten["COD_PREDIO"] = ten["COD_PREDIO"].astype(str)
    return s2.merge(ten, on="COD_PREDIO", how="left")


def poststratify(df: pd.DataFrame) -> pd.DataFrame:
    """Add `ps_weight`, reweighting the name-linked panel to the national PETT population.

    ⚠️ The name link is not a random sample — matching needs a name on both sides and a
    district agreement, which selects larger, valley-floor, better-documented parcels
    (linked panel 16.6 % PERENNIAL vs the population's 9.9 %), inflating every level here.
    Post-stratifying on department × declared class (the selection variables) restores each
    cell's national count; it cannot fix selection *within* a cell, so the reweighted
    headline is a check on the unweighted one — if they agree, composition was not it.
    """
    # read-only in both branches: `export_pett_population()` is the one place that writes
    # the committed counts, and it is called on purpose.
    if PETT.exists():
        pop = pd.read_parquet(PETT, columns=["COD_PREDIO", "dept", "label"])
        pop = (pop.groupby(["dept", "label"]).size().rename("N_pop").reset_index())
    else:
        pop = pd.read_csv(PETT_POP)          # the same counts, committed
    pop = pop.rename(columns={"label": "pett_class"})
    got = (df.groupby(["dept", "pett_class"]).size().rename("n_link").reset_index())
    w = pop.merge(got, on=["dept", "pett_class"], how="right")
    w["ps_weight"] = w["N_pop"] / w["n_link"]
    return df.merge(w[["dept", "pett_class", "ps_weight"]], on=["dept", "pett_class"],
                    how="left")


def export_pett_population() -> Path:
    """Write the department x declared-class counts that ``poststratify`` reweights to.

    Needs the uncommitted full national parcel table — run once by someone who has it,
    everyone else reads the CSV.
    """
    pop = pd.read_parquet(PETT, columns=["COD_PREDIO", "dept", "label"])
    pop = pop.groupby(["dept", "label"]).size().rename("N_pop").reset_index()
    PETT_POP.parent.mkdir(parents=True, exist_ok=True)
    pop.to_csv(PETT_POP, index=False)
    print(f"wrote {PETT_POP} — {len(pop)} cells, {pop.N_pop.sum():,} parcels")
    return PETT_POP


def build(depts: list[str] | None = None, save: bool = True) -> dict[str, pd.DataFrame]:
    """The whole comparison. Prints every table with its units; returns them."""
    out: dict[str, pd.DataFrame] = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_panel(depts)
    dl = sorted(df["dept"].unique())
    print(f"=== 0. the panel ===\nparcels with BOTH a PETT declaration and a CENAGRO 2012 "
          f"observation: **{len(df):,}** across {len(dl)} departments")
    print(f"  PETT declaration years {df.pett_year.min():.0f}-{df.pett_year.max():.0f} "
          f"(median {df.pett_year.median():.0f}); cadastral area "
          f"{df.area_ha.sum():,.0f} ha")
    print(f"  link confidence: {df.link_confidence.value_counts().to_dict()}")
    print(f"  tenure at declaration: {df.tenure.value_counts(dropna=False).to_dict()}")

    print("\n--- the census lexicon audit (run this before trusting anything below) ---")
    out["token_audit"] = token_audit(dl)

    # ---- 1. PETT -> CENAGRO, all matched parcels ------------------------------------
    t1 = share_table(df, "pett_class", "cen_class")
    out["pett_to_cenagro"] = t1
    print(f"\n=== 1. PETT declaration -> CENAGRO 2012, same parcel, n={len(df):,} ===")
    print("shares of parcels, %; change in percentage points, paired (McNemar) CI")
    print(t1.to_string(index=False))

    # ---- 1b. the comparison that is actually like-for-like ---------------------------
    crop = df[df.pett_class.isin(["PERENNIAL", "ANNUAL"])
              & df.cen_class.isin(["PERENNIAL", "ANNUAL"])].copy()
    out["panel_crop_only"] = crop
    t1b = share_table(crop, "pett_class", "cen_class", classes=("PERENNIAL", "ANNUAL"))
    out["pett_to_cenagro_crop_only"] = t1b
    print(f"\n=== 1b. ⭐ like-for-like: a CROP recorded on BOTH sides, n={len(crop):,} ===")
    print("PASTURE_FALLOW dropped from BOTH sides. The census cannot record fallow — a parcel"
          "\nwith no crop contributes no row — so the collapse above is an instrument "
          "difference,\nnot land change.")
    print(t1b.to_string(index=False))

    t1c = area_shares(crop, "pett_class", "cen_class")
    out["pett_to_cenagro_area"] = t1c
    print(f"\nthe same weighted by CADASTRAL area ({t1c.attrs['total_ha']:,.0f} ha):")
    print(t1c.to_string(index=False))

    print("\ngross parcel flows (counts):")
    print(pd.crosstab(crop.pett_class, crop.cen_class,
                      rownames=["PETT ~1999"], colnames=["CENAGRO 2012"]).to_string())

    # ---- 2. ⭐ the tenure split -------------------------------------------------------
    print("\n=== 2. ⭐ PERENNIAL change by tenure at the PETT declaration ===")
    print("⚠️ descriptive. Title is not randomly assigned; a gap here is a fact about the two"
          "\ngroups, not an effect of titling (RESULTS.md §7 is the causal estimate).")
    t2 = pd.concat([
        by_tenure(df, "pett_class", "cen_class", label="1→2 all parcels"),
        by_tenure(crop, "pett_class", "cen_class", label="1→2 crop-on-both-sides"),
    ], ignore_index=True)
    out["tenure_split_parcels"] = t2
    print("\nshare of PARCELS that are PERENNIAL, %:")
    print(t2.to_string(index=False))

    rows = []
    for t in ["INSCRITO", "NO INSCRITO"]:
        sub = crop[crop.tenure == t]
        a = area_shares(sub, "pett_class", "cen_class")
        r = a[a["class"] == "PERENNIAL"].iloc[0].to_dict()
        r.update({"tenure": t, "n": len(sub), "total_ha": round(a.attrs["total_ha"])})
        rows.append(r)
    t2b = pd.DataFrame(rows)[["tenure", "n", "total_ha", "before_pct_of_area",
                              "after_pct_of_area", "change_pp"]]
    out["tenure_split_area"] = t2b
    print("\nshare of cadastral AREA that is PERENNIAL, % (crop-on-both-sides):")
    print(t2b.to_string(index=False))

    # ---- 3. does the answer move with link quality? ----------------------------------
    rows = []
    for lc in ["high", "medium", "low"]:
        sub = crop[crop.link_confidence == lc]
        if len(sub) < 200:
            continue
        r = perennial_change(sub, "pett_class", "cen_class")
        r["link_confidence"] = lc
        rows.append(r)
    t3 = pd.DataFrame(rows)[["link_confidence", "n", "before_pct", "after_pct",
                             "change_pp", "ci95_pp"]]
    out["by_link_confidence"] = t3
    print("\n=== 3. ⚠️ the same by census link quality ===")
    print("if the answer moves with link quality, part of the answer is the link")
    print(t3.to_string(index=False))

    # ---- 4. per department -----------------------------------------------------------
    rows = []
    for d in dl:
        sub = crop[crop.dept == d]
        if len(sub) < 100:
            continue
        r = perennial_change(sub, "pett_class", "cen_class")
        r["dept"] = d
        ti = by_tenure(sub, "pett_class", "cen_class")
        diff = ti.loc[ti.tenure.str.startswith("difference"), "change_pp"]
        r["tenure_diff_pp"] = float(diff.iloc[0]) if len(diff) else np.nan
        rows.append(r)
    t4 = pd.DataFrame(rows)[["dept", "n", "before_pct", "after_pct", "change_pp",
                             "ci95_pp", "tenure_diff_pp"]]
    out["by_dept"] = t4
    print("\n=== 4. per department (crop-on-both-sides) ===")
    print(t4.to_string(index=False))

    # ---- 4b. ⭐ does the answer survive reweighting to the national population? -------
    ps = poststratify(crop)
    print("\n=== 4b. ⭐ reweighted to the national PETT population "
          "(department × declared class) ===")
    print("⚠️ the linked panel is 16.6 % PERENNIAL against the population's 9.9 % — the name\n"
          "link over-selects perennial parcels. If the reweighted change matches the raw one,\n"
          "composition was not driving the answer.")
    raw = perennial_change(crop, "pett_class", "cen_class")
    pw = perennial_change(ps, "pett_class", "cen_class", weight="ps_weight")
    t4b = pd.DataFrame([{**raw, "estimator": "raw (linked panel)"},
                        {**pw, "estimator": "post-stratified to national"}])
    t4b = t4b[["estimator", "n", "before_pct", "after_pct", "change_pp", "ci95_pp"]]
    out["poststratified"] = t4b
    print(t4b.to_string(index=False))
    t4c = by_tenure(ps, "pett_class", "cen_class", weight="ps_weight",
                    label="1→2 post-stratified")
    out["poststratified_tenure"] = t4c
    print("\nand its tenure split:")
    print(t4c.to_string(index=False))
    a_ps = area_shares(ps, "pett_class", "cen_class", weight="ps_weight")
    out["poststratified_area"] = a_ps
    print(f"\nand by cadastral area ({a_ps.attrs['total_ha']:,.0f} weighted ha):")
    print(a_ps.to_string(index=False))

    # ---- 5. the photo-interpreted endpoint -------------------------------------------
    out.update(_s2_section(df))

    if save:
        for k, v in out.items():
            if isinstance(v, pd.DataFrame) and len(v):
                v.to_csv(OUT_DIR / f"national_{k}.csv", index=False)
        df.to_parquet(OUT_DIR / "national_panel.parquet", index=False)
        print(f"\nwrote {len(out)} tables + national_panel.parquet to {OUT_DIR}")
    return out


def _s2_section(cen_panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """PETT → photo-interpreted 2019+, and CENAGRO 2012 → photo-interpreted 2019+.

    ⚠️ Every share is design-weighted; the unweighted number is printed beside it so the
    scaling is visible.
    """
    out: dict[str, pd.DataFrame] = {}
    s2 = s2_panel()
    print("\n=== 5. PETT declaration -> photo-interpreted 2019+ ===")
    print(f"usable human labels: {len(s2):,} parcels; tenure known for "
          f"{int(s2.tenure.notna().sum()):,}")
    print(f"raw label mix: {s2.label.value_counts().to_dict()}")

    # the weight is doing real work — show how much
    mix = pd.DataFrame({
        "unweighted_pct": (100 * s2.label.value_counts(normalize=True)).round(1),
        "design_weighted_pct": (100 * s2.groupby("label")["weight"].sum()
                                / s2["weight"].sum()).round(1)}).fillna(0.0)
    out["s2_label_mix"] = mix.reset_index(names="label")
    print("\n⚠️ what the design weight does to the label mix (the sample over-samples "
          "PERENNIAL):")
    print(mix.to_string())

    for name, col in [("WOODY_NON_CROP excluded", "s2_class"),
                      ("WOODY_NON_CROP read as PERENNIAL",
                       "s2_class_woody_perennial")]:
        d = s2[s2[col].notna() & s2["declared_class"].notna()].copy()
        crop = d[d.declared_class.isin(["PERENNIAL", "ANNUAL"])
                 & d[col].isin(["PERENNIAL", "ANNUAL"])].copy()
        uw = perennial_change(crop, "declared_class", col)
        wt = perennial_change(crop, "declared_class", col, weight="weight")
        print(f"\n--- {name} — crop-on-both-sides, n={len(crop):,} ---")
        print(f"  unweighted     : {uw['before_pct']:>5.1f} % -> {uw['after_pct']:>5.1f} % "
              f"({uw['change_pp']:+.1f} pp)")
        print(f"  design-weighted: {wt['before_pct']:>5.1f} % -> {wt['after_pct']:>5.1f} % "
              f"({wt['change_pp']:+.1f} pp ± {wt['ci95_pp']:.1f})  ⭐ the estimate")
        t = by_tenure(crop, "declared_class", col, weight="weight",
                      label=f"1→3 {name}")
        out[f"s2_tenure_{'excl' if col == 's2_class' else 'woody'}"] = t
        print("  by tenure (design-weighted, PERENNIAL share of parcels, %):")
        print(t.to_string(index=False))

    # ---- the three-way overlap -------------------------------------------------------
    both = cen_panel[["COD_PREDIO", "cen_class", "tenure"]].merge(
        s2[["COD_PREDIO", "s2_class", "s2_class_woody_perennial", "weight",
            "declared_class"]], on="COD_PREDIO", how="inner")
    out["three_way"] = both
    print(f"\n=== 6. parcels with ALL THREE observations "
          f"(PETT + CENAGRO 2012 + a 2019+ human label): **{len(both)}** ===")
    if len(both):
        print(pd.crosstab(both.cen_class, both.s2_class.fillna("(unmapped)")).to_string())
    if len(both) < 100:
        print("⚠️ too few to estimate a 2012→2019 change from. This is not a fixable sample"
              "\nsize: the labelling campaign drew from the PETT population, not from the"
              "\nname-linked census subset, so the overlap is incidental.")
    return out


# The figure. Palette is slots 1–2 of the project's validated categorical theme
# (blue #2a78d6 / orange #eb6834): CVD ΔE 24.7, normal-vision ΔE 33.6, both ≥ 3:1 on light.
FIG_DIR = ROOT / "docs" / "figures"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
SURFACE = "#fcfcfb"
TENURE_COLOR = {"INSCRITO": "#2a78d6", "NO INSCRITO": "#eb6834"}
# x positions: the median year of each instrument, on a real time axis
T_PETT, T_CEN, T_S2 = 2001, 2012, 2025


def _level(df: pd.DataFrame, col: str, weight: str | None = None) -> dict:
    """Weighted PERENNIAL share with a Wilson 95 % interval, in percent.

    (1) n is Kish's effective size (Σw)²/Σw², not the row count — a weighted share carries
    less information than its n, and on the imagery arm n_eff is a fraction of n. (2) Wilson
    not Wald: a share cannot be negative, and Wald runs below zero at this precision.
    """
    w = df[weight] if weight else pd.Series(1.0, index=df.index)
    tot = float(w.sum())
    p = float(w[df[col] == "PERENNIAL"].sum()) / tot
    n = tot ** 2 / float((w ** 2).sum())          # Kish effective n
    z = 1.959963985
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z / denom * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return {"pct": 100 * p,
            "lo": 100 * max(centre - half, 0.0),
            "hi": 100 * min(centre + half, 1.0),
            "n_eff": n}


def figure_values() -> dict[str, dict]:
    """The six levels per tenure group, recomputed from the panels. Never hard-coded here."""
    df = build_panel()
    crop = df[df.pett_class.isin(["PERENNIAL", "ANNUAL"])
              & df.cen_class.isin(["PERENNIAL", "ANNUAL"])].copy()
    ps = poststratify(crop)
    s2 = s2_panel()
    s2e = s2[s2.declared_class.isin(["PERENNIAL", "ANNUAL"])
             & s2.s2_class.isin(["PERENNIAL", "ANNUAL"])]
    s2w = s2[s2.declared_class.isin(["PERENNIAL", "ANNUAL"])
             & s2.s2_class_woody_perennial.isin(["PERENNIAL", "ANNUAL"])]

    out: dict[str, dict] = {}
    for t in ("INSCRITO", "NO INSCRITO"):
        a, b, c = ps[ps.tenure == t], s2e[s2e.tenure == t], s2w[s2w.tenure == t]
        out[t] = {
            "cen": [_level(a, "pett_class", "ps_weight"),
                    _level(a, "cen_class", "ps_weight")],
            "n_cen": len(a),
            "s2": [_level(b, "declared_class", "weight"),
                   _level(b, "s2_class", "weight")],
            "n_s2": len(b),
            "woody": [_level(c, "declared_class", "weight"),
                      _level(c, "s2_class_woody_perennial", "weight")],
            "n_woody": len(c),
        }
    return out


def figure(path: Path | None = None) -> Path:
    """Draw `docs/figures/perennial_over_time_by_tenure.png`.

    Drawing code lives only in `docs/figures/perennial_over_time_by_tenure.py` (a
    self-contained matplotlib script). This recomputes the numbers, checks them against the
    ones baked into that script, and calls its `draw()` — so the two cannot drift apart.
    """
    import importlib.util

    script = FIG_DIR / "perennial_over_time_by_tenure.py"
    spec = importlib.util.spec_from_file_location("_perennial_fig", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    values = figure_values()
    drift = []
    for t, s in values.items():
        for key in ("cen", "s2", "woody"):
            for i, lv in enumerate(s[key]):
                baked = mod.VALUES[t][key][i]
                if abs(lv["pct"] - baked[0]) > 0.05:
                    drift.append(f"{t}/{key}[{i}]: data {lv['pct']:.2f} vs "
                                 f"script {baked[0]:.2f}")
    if drift:
        print("⚠️  the standalone script's baked-in values have drifted from the data:")
        for d in drift:
            print(f"     {d}")
        print(f"     update VALUES in {script} — printing the fresh block below")
        print(_values_literal(values))

    out = mod.draw(values, path or FIG_DIR / "perennial_over_time_by_tenure.png")
    print(f"wrote {out}")
    for t, s in values.items():
        print(f"  {t:<12} census {s['cen'][0]['pct']:.1f} -> {s['cen'][1]['pct']:.1f} % | "
              f"imagery {s['s2'][0]['pct']:.1f} -> {s['s2'][1]['pct']:.1f} % "
              f"[{s['s2'][1]['lo']:.1f}, {s['s2'][1]['hi']:.1f}] | "
              f"woody {s['woody'][0]['pct']:.1f} -> {s['woody'][1]['pct']:.1f} % "
              f"[{s['woody'][1]['lo']:.1f}, {s['woody'][1]['hi']:.1f}]")
    return out


def _values_literal(values: dict[str, dict]) -> str:
    """The `VALUES` block, formatted for pasting into the standalone script."""
    lines = ["VALUES: dict[str, dict] = {"]
    for t, s in values.items():
        lines.append(f'    "{t}": {{')
        for key, n in (("cen", "n_cen"), ("s2", "n_s2"), ("woody", "n_woody")):
            pts = ", ".join(f'({lv["pct"]:.2f}, {lv["lo"]:.2f}, {lv["hi"]:.2f})'
                            for lv in s[key])
            lines.append(f'        "{key}": [{pts}], "{n}": {s[n]},')
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines)

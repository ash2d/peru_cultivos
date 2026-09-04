"""Descriptive statistics of the PETT registry and the 2012 census, nationally and by
department — crop mix, tenure, and the change between the two instruments.

One table, one row per department plus a national row::

    uv run cc -w national analysis summary
    uv run cc -w national analysis summary --by-dept --out summary.csv

Three universes are in it, and they are not the same parcels — the count beside each block
says which:

| block            | universe                                            | n         |
|------------------|-----------------------------------------------------|-----------|
| `pett_*`         | every parcel with a declared crop and a polygon      | 726,808   |
| `tenure_*`       | every bridged parcel, with both dated observations   | 1,780,580 |
| `cen_*`, change  | parcels linked to a CENAGRO 2012 record by name      | 63,766    |

⚠️ The change columns are the **crop-on-both-sides** comparison (`PASTURE_FALLOW` dropped
from both sides). The census asks which crop is grown, so a fallow parcel leaves the frame
entirely; comparing all parcels reads that instrument difference as land change
(`cenagro_shift.py`).

⚠️ The linked panel is 16.6 % perennial against the population's 9.9 %, so the national
change row is post-stratified on department x declared class. The per-department rows are
not — within a department there is nothing left to reweight onto.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from crop_classifier.allperu.cenagro_shift import perennial_change, poststratify
from crop_classifier.paths import ROOT

PETT_POP = ROOT / "data" / "processed" / "cenagro" / "pett_population_by_dept_class.csv"
TENURE = ROOT / "data" / "processed" / "all_peru" / "tenure_two_period.parquet"
PANEL = ROOT / "data" / "processed" / "cenagro" / "national_panel.parquet"

COLUMNS = [
    "dept",
    # the PETT declaration, ~1997-2006
    "n_pett", "pett_perennial_pct", "pett_annual_pct", "pett_pasture_pct",
    # tenure: at the declaration, at the 2011-12 cadastre, and the move between them
    "n_tenure", "inscrito_pct", "registered_2011_pct", "newly_registered_pct",
    # the census-linked panel: 2012 level, and the change from the declaration
    "n_linked", "perennial_before_pct", "perennial_after_pct", "perennial_change_pp",
    "change_ci95_pp", "tenure_gap_pp",
]


def _pett() -> pd.DataFrame:
    """Crop mix of the full PETT population, from the committed dept x class counts."""
    if not PETT_POP.exists():
        raise SystemExit(f"missing {PETT_POP.relative_to(ROOT)}")
    pop = pd.read_csv(PETT_POP)
    wide = pop.pivot_table(index="dept", columns="label", values="N_pop",
                           aggfunc="sum", fill_value=0)
    out = pd.DataFrame({"n_pett": wide.sum(axis=1)})
    for cls, col in [("PERENNIAL", "pett_perennial_pct"), ("ANNUAL", "pett_annual_pct"),
                     ("PASTURE_FALLOW", "pett_pasture_pct")]:
        out[col] = (wide.get(cls, 0) / wide.sum(axis=1) * 100).round(1)
    return out


def _tenure() -> pd.DataFrame:
    """Registered at the declaration, registered at the cadastre, and the move between.

    `became_registered` is the parcel-level change in `inscrito`: unregistered when the
    farmer declared, registered in the ~2011 cadastre. It is the treatment of the DiD
    (`RESULTS.md` §7), and 8.6 % of parcels nationally.
    """
    if not TENURE.exists():
        raise SystemExit(f"missing {TENURE.relative_to(ROOT)} — build it with "
                         f"`uv run cc -w national data tenure`")
    t = pd.read_parquet(TENURE, columns=["dept", "tenure", "cadastre_status",
                                         "became_registered"])
    g = t.groupby("dept")
    return pd.DataFrame({
        "n_tenure": g.size(),
        "inscrito_pct": g["tenure"].apply(lambda s: s.eq("INSCRITO").mean() * 100).round(1),
        "registered_2011_pct": g["cadastre_status"].apply(
            lambda s: s.eq("REGISTERED").mean() * 100).round(1),
        "newly_registered_pct": g["became_registered"].mean().mul(100).round(1),
    })


def crop_panel() -> pd.DataFrame:
    """The linked panel, restricted to a crop recorded on both sides — the like-for-like set."""
    if not PANEL.exists():
        raise SystemExit(f"missing {PANEL.relative_to(ROOT)} — build it with "
                         f"`uv run cc -w national analysis perennial-shift`")
    p = pd.read_parquet(PANEL)
    return p[p.pett_class.isin(["PERENNIAL", "ANNUAL"])
             & p.cen_class.isin(["PERENNIAL", "ANNUAL"])].copy()


def _change(sub: pd.DataFrame, weight: str | None = None) -> dict:
    """PERENNIAL before/after/change on one group, plus the titled − untitled gap in it."""
    r = perennial_change(sub, "pett_class", "cen_class", weight)
    out = {"n_linked": r["n"], "perennial_before_pct": r["before_pct"],
           "perennial_after_pct": r["after_pct"], "perennial_change_pp": r["change_pp"],
           "change_ci95_pp": r["ci95_pp"], "tenure_gap_pp": None}
    arms = [perennial_change(sub[sub.tenure == t], "pett_class", "cen_class", weight)
            for t in ("INSCRITO", "NO INSCRITO") if (sub.tenure == t).any()]
    if len(arms) == 2:
        out["tenure_gap_pp"] = round(arms[0]["change_pp"] - arms[1]["change_pp"], 1)
    return out


def build(dept: str | None = None, by_dept: bool = False,
          out: Path | None = None) -> pd.DataFrame:
    pett, ten, crop = _pett(), _tenure(), crop_panel()
    depts = sorted(set(pett.index) | set(ten.index))
    if dept:
        depts = [d for d in depts if d.upper() == dept.upper()]
        if not depts:
            raise SystemExit(f"unknown department {dept!r}; have {sorted(set(pett.index))}")

    rows = []
    if not dept:
        # the national row: the change is post-stratified because the name link over-selects
        # perennial parcels, and the reweighting is what makes it a statement about Peru
        rows.append({"dept": "PERU (all 14)", "n_pett": int(pett.n_pett.sum()),
                     **{c: round(float((pett[c] * pett.n_pett).sum() / pett.n_pett.sum()), 1)
                        for c in ("pett_perennial_pct", "pett_annual_pct",
                                  "pett_pasture_pct")},
                     "n_tenure": int(ten.n_tenure.sum()),
                     **{c: round(float((ten[c] * ten.n_tenure).sum() / ten.n_tenure.sum()), 1)
                        for c in ("inscrito_pct", "registered_2011_pct",
                                  "newly_registered_pct")},
                     **_change(poststratify(crop), weight="ps_weight")})
    if by_dept or dept:
        for d in depts:
            sub = crop[crop.dept == d]
            rows.append({"dept": d, **pett.reindex([d]).iloc[0].to_dict(),
                         **ten.reindex([d]).iloc[0].to_dict(),
                         **(_change(sub) if len(sub) >= 100 else {})})

    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    df["n_pett"] = df["n_pett"].astype("Int64")
    df["n_tenure"] = df["n_tenure"].astype("Int64")
    df["n_linked"] = df["n_linked"].astype("Int64")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"wrote {out}")
    return df


def report(dept: str | None = None, by_dept: bool = False,
           out: Path | None = None) -> pd.DataFrame:
    df = build(dept, by_dept, out)
    print(df.to_string(index=False, na_rep="-"))
    print("\nn_pett     parcels with a declared crop and a polygon (PETT, ~1997-2006)")
    print("n_tenure   bridged parcels with both dated tenure observations "
          "(declaration, ~2011 cadastre)")
    print("n_linked   parcels linked to CENAGRO 2012 by farmer name, crop recorded on "
          "both sides")
    print("change     perennial share of parcels, declaration -> 2012, in percentage points."
          "\n           The national row is post-stratified to the PETT population; the "
          "department rows are not.")
    print("tenure_gap INSCRITO minus NO INSCRITO in that change. ⚠️ descriptive — title is "
          "not\n           randomly assigned (RESULTS.md §7 is the causal estimate).")
    return df

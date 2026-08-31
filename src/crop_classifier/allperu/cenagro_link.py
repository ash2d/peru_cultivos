"""National CENAGRO 2012 ⇄ PETT link, by farmer name — Chain B for 14 departments.

`allperu/cenagro.py` compares the PETT declaration with the 2012 census on **Piura only**,
because the crosswalk it reads (`merged_parcels.parquet`) was built by notebook 02 for Piura.
The 25-department census extract (`DATA.md` §1.5) removes that limit for the **14 departments
the cadastral bridge covers** (§3).

⚠️ **The census carries no parcel key.** No `COD_PREDIO`, no `CodigoSSET`, no DNI. The only
link is the farmer's **name**, so the link is *farmer-level, not parcel-level*: a producer with
three parcels is matched to a person, and which of their polygons a census row describes is
uncertain. Every number built on this must be reported by `link_confidence`, because that is
the only honest way to show how much of the answer is the link rather than the land.

The national link is **structurally simpler** than notebook 02's. That notebook matched the
census against three different Piura files (SSET, catastro, polygons) and reconciled them.
Nationally, `Grafica_Tabular/<Dept>.dta` already carries `COD_PREDIO`, `CodigoSSET`, `NOMBRES`
**and** a 6-digit district `id_dist` in one table — the same district code the census builds as
`UBIGEO` from `P001+P002+P003`. So one target, with a district check for free.

Three match routes, strictest first, exactly as notebook 02 defined them:

* ``full``   apellido paterno + materno + 1º + 2º nombre, in order
* ``tokset`` the same tokens, order-insensitive (the two surnames are often swapped)
* ``core``   paterno | materno | 1º nombre only (drops given-name variation)

⚠️ **A name is not a key and common names are not uniformly common.** Matching is done
**within department**, and a name key that pulls more than `MAX_CANDIDATES` distinct parcels is
**dropped, not resolved** — at that multiplicity the match carries no information, and keeping
it would quietly load the sample with the departments that have the most repeated surnames.
`build()` reports how many producers that removes.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
import pyreadstat

from crop_classifier.paths import ROOT

CENAGRO_DIR = ROOT / "data" / "raw" / "Cenagro_IV"
BRIDGE_DIR = ROOT / "data" / "raw" / "Grafica_Tabular"
OUT_DIR = ROOT / "data" / "processed" / "cenagro"

# A name key matching more than this many distinct parcels in one department is not an
# identifying match. Dropped rather than resolved; the count is reported.
MAX_CANDIDATES = 25

# census file stem -> Grafica_Tabular file stem, for the 14 linkable departments (DATA.md §3)
LINKABLE: dict[str, str] = {
    "Ancash": "Ancash", "Arequipa": "Arequipa", "Ayacucho": "Ayacucho",
    "Cajamarca": "Cajamarca", "Huancavelica": "Huancavelica", "Ica": "Ica",
    "La_Libertad": "La_Libertad", "Lambayeque": "Lambayeque", "Lima": "Lima",
    "Moquegua": "Moquegua", "Pasco": "Pasco", "Piura": "Piura", "Tacna": "Tacna",
    "Tumbes": "Tumbes",
}

_PARTICLES = {"DE", "DEL", "LA", "LAS", "LOS", "SAN", "SANTA", "MC", "MAC"}
ROUTES = [("full", "name_full"), ("tokset", "name_tokset"), ("core", "name_core")]
_ROUTE_RANK = {"full": 0, "tokset": 1, "core": 2}


# --------------------------------------------------------------------------------------
def norm_txt(s: object) -> str | None:
    """Uppercase, strip accents, drop punctuation, collapse spaces."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z ]", " ", s).upper()
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _glue_particles(toks: list[str]) -> list[str]:
    out, i = [], 0
    while i < len(toks):
        if toks[i] in _PARTICLES and i + 1 < len(toks):
            out.append(toks[i] + " " + toks[i + 1])
            i += 2
        else:
            out.append(toks[i])
            i += 1
    return out


def parse_full_name(raw: object) -> tuple[str | None, str | None, list[str]]:
    """One concatenated name -> (apellido paterno, materno, [given names]).

    ⚠️ **Assumption**: with no field boundaries, the first two tokens are the two surnames and
    the rest are given names (Peruvian convention). A comma, where present, is the true
    surname/given split and overrides the heuristic. Leading particles glue to the next token.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None, None, []
    has_comma = "," in str(raw)
    left, _, right = str(raw).partition(",")
    if has_comma:
        ap = _glue_particles((norm_txt(left) or "").split())
        giv = (norm_txt(right) or "").split()
    else:
        toks = _glue_particles((norm_txt(raw) or "").split())
        ap, giv = toks[:2], toks[2:]
    return (ap[0] if ap else None), (ap[1] if len(ap) > 1 else None), giv


def name_keys(ap1: pd.Series, ap2: pd.Series, givens: pd.Series) -> pd.DataFrame:
    """Already-split components -> the three match keys, loosest last."""
    out = pd.DataFrame(index=ap1.index)
    out["name1"] = ap1.map(norm_txt)
    out["name2"] = ap2.map(norm_txt)
    g = givens.map(lambda x: [norm_txt(t) for t in (x or []) if norm_txt(t)])
    out["name3"] = g.map(lambda x: x[0] if len(x) > 0 else None)
    out["name4"] = g.map(lambda x: x[1] if len(x) > 1 else None)
    out["name_full"] = (out[["name1", "name2", "name3", "name4"]].fillna("")
                        .agg(" ".join, axis=1).str.strip()
                        .str.replace(r"\s+", " ", regex=True))
    out["name_tokset"] = out["name_full"].map(
        lambda s: " ".join(sorted(s.split())) if s else None)
    out["name_core"] = (out["name1"].fillna("") + "|" + out["name2"].fillna("")
                        + "|" + out["name3"].fillna(""))
    # a key with no surname at all identifies nobody
    blank = out["name_full"].eq("") | out["name1"].isna()
    out.loc[blank, ["name_full", "name_tokset", "name_core"]] = None
    return out


# --------------------------------------------------------------------------------------
def census_names(dept: str) -> pd.DataFrame:
    """One row per census producer, with the three match keys and the district UBIGEO."""
    d = pd.read_parquet(CENAGRO_DIR / f"{dept}.parquet",
                        columns=["NPRIN", "UBIGEO", "P009_01", "P009_02", "P009_03",
                                 "P009_04"])
    d = d.drop_duplicates("NPRIN").reset_index(drop=True)
    keys = name_keys(d["P009_01"], d["P009_02"],
                     d["P009_03"].map(lambda s: (norm_txt(s) or "").split()))
    out = pd.concat([d[["NPRIN", "UBIGEO"]], keys], axis=1)
    out["razon_social"] = d["P009_04"].map(norm_txt)
    return out


def bridge_names(dept: str) -> pd.DataFrame:
    """One row per cadastral parcel, with the three match keys and its district UBIGEO."""
    d, _ = pyreadstat.read_dta(
        str(BRIDGE_DIR / f"{dept}.dta"),
        usecols=["COD_PREDIO", "CodigoSSET", "NOMBRES", "id_dist", "area_ha"],
        encoding="latin1")
    d["COD_PREDIO"] = d["COD_PREDIO"].astype(str).str.strip()
    d = d[d["COD_PREDIO"].ne("") & d["COD_PREDIO"].ne("nan")]
    d = d.drop_duplicates("COD_PREDIO").reset_index(drop=True)
    parts = d["NOMBRES"].map(parse_full_name)
    keys = name_keys(parts.map(lambda t: t[0]), parts.map(lambda t: t[1]),
                     parts.map(lambda t: t[2]))
    out = pd.concat([d[["COD_PREDIO", "CodigoSSET", "area_ha"]], keys], axis=1)
    out["UBIGEO"] = d["id_dist"].astype(str).str.strip()
    return out


def link_dept(dept: str) -> tuple[pd.DataFrame, dict]:
    """Census producers -> one best `COD_PREDIO` each, with a confidence grade.

    Returns (link, report). `link` has one row per matched producer.
    """
    cen = census_names(dept)
    bri = bridge_names(dept)
    named_cen = int(cen["name_full"].notna().sum())
    named_bri = int(bri["name_full"].notna().sum())

    frames, dropped = [], 0
    for route, key in ROUTES:
        left = cen.loc[cen[key].notna(), ["NPRIN", "UBIGEO", key]]
        right = bri.loc[bri[key].notna(), ["COD_PREDIO", "UBIGEO", key]]
        # ⚠️ a name that pulls > MAX_CANDIDATES parcels is not identifying: drop the key
        n_right = right.groupby(key)["COD_PREDIO"].transform("nunique")
        dropped += int((n_right > MAX_CANDIDATES).sum())
        right = right[n_right <= MAX_CANDIDATES]
        m = left.merge(right, on=key, suffixes=("", "_t"))
        m["route"] = route
        m["ubigeo_ok"] = m["UBIGEO"] == m["UBIGEO_t"]
        frames.append(m[["NPRIN", "COD_PREDIO", "route", "ubigeo_ok"]])

    cand = pd.concat(frames, ignore_index=True)
    cand["r"] = cand["route"].map(_ROUTE_RANK)
    cand = cand.sort_values("r").drop_duplicates(["NPRIN", "COD_PREDIO"]).drop(columns="r")

    # prefer: same district > unique candidate > strictest route
    cand["n_cod"] = cand.groupby("NPRIN")["COD_PREDIO"].transform("nunique")
    cand["score"] = (cand["ubigeo_ok"].astype(int) * 4
                     + (cand["n_cod"] == 1).astype(int) * 2
                     + cand["route"].map({"full": 2, "tokset": 1, "core": 0}))
    best = (cand.sort_values("score", ascending=False)
            .drop_duplicates("NPRIN")[["NPRIN", "COD_PREDIO", "route", "ubigeo_ok", "n_cod"]])
    best["link_confidence"] = np.where(
        best["ubigeo_ok"] & (best["n_cod"] == 1), "high",
        np.where(best["ubigeo_ok"] | (best["n_cod"] == 1), "medium", "low"))
    best["dept"] = dept

    rep = {"dept": dept,
           "census_producers": len(cen), "census_named": named_cen,
           "bridge_parcels": len(bri), "bridge_named": named_bri,
           "matched": len(best),
           "matched_pct_of_named": round(100 * len(best) / max(named_cen, 1), 1),
           "high": int((best.link_confidence == "high").sum()),
           "medium": int((best.link_confidence == "medium").sum()),
           "low": int((best.link_confidence == "low").sum()),
           "keys_dropped_too_common": dropped}
    return best, rep


def build(depts: list[str] | None = None, save: bool = True
          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The national link. Writes `cenagro_pett_link.parquet` + its report."""
    links, reps = [], []
    for d in depts or list(LINKABLE):
        lk, rep = link_dept(d)
        links.append(lk)
        reps.append(rep)
        print(f"  {d:<13} {rep['census_named']:>8,} named producers -> "
              f"{rep['matched']:>7,} matched ({rep['matched_pct_of_named']:>5.1f} %)  "
              f"high {rep['high']:>6,} / med {rep['medium']:>6,} / low {rep['low']:>6,}",
              flush=True)
    link = pd.concat(links, ignore_index=True)
    report = pd.DataFrame(reps)
    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        link.to_parquet(OUT_DIR / "cenagro_pett_link.parquet", index=False)
        report.to_csv(OUT_DIR / "cenagro_pett_link_report.csv", index=False)
    print(f"\nlinked {len(link):,} census producers to a COD_PREDIO across "
          f"{link.dept.nunique()} departments")
    print(link.link_confidence.value_counts().to_dict())
    return link, report

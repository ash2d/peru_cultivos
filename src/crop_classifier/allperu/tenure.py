"""Land-tenure (``ESTADO en RRPP``) per parcel — the treatment variable of the window pivot.

RESULTS.md §5 makes the tenure contrast the deliverable, so the column has to come out of
the raw workbooks onto ``COD_PREDIO`` — no processed table carries it.

Three quiet traps:
1. The sheet is not always ``DATOS`` — ``BD SSET(AREQUIPA-AYACUCHO-CAJAMARCA).xlsx`` splits
   1.6 M rows across ``DATOS1``/``DATOS2`` (Excel's row ceiling). Every ``DATOS*`` is read.
2. A department's rows are not confined to "its" workbook (``DATA.md`` §4.3) — scan all,
   route by ``DEPARTAMENTO``.
3. Keys are zero-padded on the bridge side, not in BD SSET — go through ``canon_key``.

Tenure is per *declaration*, so it is aggregated twice: per ``CodigoSSET`` (declarations of
one parcel) and per ``COD_PREDIO`` (several SSET keys bridge to one polygon). Registration
is an absorbing state, so the default reconciliation is ``any`` (INSCRITO if any declaration
says so); ``mode`` is a sensitivity arm, and ``tenure_conflict`` always reports the
disagreement rate.

    uv run python -m crop_classifier.allperu.tenure          # build caches + audit
"""

from __future__ import annotations

import argparse
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from crop_classifier.allperu.build_labels import CACHE, canon_key
from crop_classifier.allperu.sources import Dept, _norm, departments

TENURE_COLS = ["DEPARTAMENTO", "Codigo SSET", "ESTADO en RRPP", "FECHA EMPADRONAMIENTO"]

INSCRITO = "INSCRITO"
NO_INSCRITO = "NO INSCRITO"


def _clean_tenure(s: pd.Series) -> pd.Series:
    """Free text -> ``INSCRITO`` / ``NO INSCRITO`` / NaN.

    Documented as binary and non-null, but still free text: anything else returns NaN and is
    counted by :func:`audit`, so a third value shows up as missingness rather than being
    folded into a group.
    """
    # `.astype(str)` leaves NA as float NaN under Arrow dtypes, which breaks
    # `unicodedata.normalize`; go through `string` + fillna so a null reads as "unparsed".
    t = (s.astype("string").fillna("")
         .map(lambda x: unicodedata.normalize("NFKD", str(x))
              .encode("ascii", "ignore").decode())
         .str.upper().str.replace(r"[^A-Z ]", " ", regex=True)
         .str.split().str.join(" "))
    out = pd.Series(np.nan, index=s.index, dtype=object)
    out[t.eq(INSCRITO)] = INSCRITO
    out[t.isin({NO_INSCRITO, "NOINSCRITO"})] = NO_INSCRITO
    return out


def workbook_tenure(path: Path, cache_dir: Path | None = None) -> pd.DataFrame:
    """One workbook's ``(dept, CodigoSSET, tenure, reg_year)`` rows, cached to parquet.

    Reading a 200 MB xlsx costs minutes, so cache it — separately from
    ``build_labels.load_sset_workbook``'s cache, which was written without the tenure column.
    """
    cd = cache_dir or CACHE
    cache = cd / f"tenure_{path.stem.replace(' ', '_')}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    import openpyxl

    t0 = time.time()
    wb = openpyxl.load_workbook(path, read_only=True)
    sheets = [s for s in wb.sheetnames if s.upper().startswith("DATOS")]
    wb.close()
    if not sheets:
        raise ValueError(f"{path.name}: no DATOS* sheet")
    parts = [pd.read_excel(path, sheet_name=s, usecols=TENURE_COLS) for s in sheets]
    df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]

    key = df["Codigo SSET"]
    if not pd.api.types.is_object_dtype(key):
        key = key.astype("Int64")
    out = pd.DataFrame({
        "dept_sset": df["DEPARTAMENTO"].map(_norm),
        "CodigoSSET": canon_key(key),
        "tenure": _clean_tenure(df["ESTADO en RRPP"]),
        "tenure_raw": df["ESTADO en RRPP"].astype(str),
        "reg_year": pd.to_datetime(df["FECHA EMPADRONAMIENTO"],
                                   errors="coerce").dt.year.astype("Int64"),
    })
    cd.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    print(f"  read {path.name} sheets={sheets} ({len(out):,} rows) "
          f"in {time.time() - t0:.0f}s -> cached", flush=True)
    return out


def tenure_records(depts: list[Dept] | None = None,
                   cache_dir: Path | None = None) -> pd.DataFrame:
    """Every declaration's tenure, over every workbook, restricted to linkable departments."""
    depts = depts or departments()
    wanted = {d.sset_key: d.name for d in depts}
    parts = []
    for wb_path in sorted({d.sset for d in depts}):
        df = workbook_tenure(wb_path, cache_dir=cache_dir)
        df = df[df["dept_sset"].isin(wanted)]
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["dept"] = out["dept_sset"].map(wanted)
    return out.drop(columns="dept_sset")


def _resolve(frac: pd.Series, policy: str) -> pd.Series:
    """``frac_inscrito`` -> a tenure value, vectorised.

    ``any`` (default): INSCRITO if any declaration says so — registration is absorbing, so a
    later NO INSCRITO is stale, not a de-registration. ``mode``: majority, ties to INSCRITO.
    """
    cut = 0.0 if policy == "any" else 0.5
    op = frac.gt(cut) if policy == "any" else frac.ge(cut)
    return pd.Series(np.where(op, INSCRITO, NO_INSCRITO), index=frac.index, dtype=object)


def aggregate(records: pd.DataFrame, by: str, policy: str = "any") -> pd.DataFrame:
    """Collapse declaration-level tenure onto ``by`` (``CodigoSSET`` or ``COD_PREDIO``).

    Returns ``tenure`` (per ``policy``), ``tenure_conflict`` (group held both values),
    ``frac_inscrito``, ``n_tenure_records`` and the first ``reg_year``.

    Fully vectorised — ~4 M declarations over ~3 M keys; a ``groupby.apply`` lambda over
    that many groups runs for hours.
    """
    r = records[records["tenure"].notna()].copy()
    r["_insc"] = (r["tenure"] == INSCRITO).astype(float)
    grp = r.groupby(by, observed=True, sort=False)
    frac = grp["_insc"].mean()
    out = pd.DataFrame({
        "tenure": _resolve(frac, policy),
        "frac_inscrito": frac,
        "tenure_conflict": frac.between(0.0, 1.0, inclusive="neither"),
        "n_tenure_records": grp.size(),
    })
    if "reg_year" in records.columns:
        # earliest declaration year = the titling event
        out["reg_year"] = grp["reg_year"].min().astype("Int64")
    return out.reset_index()


def bridge_pairs(depts: list[Dept] | None = None) -> pd.DataFrame:
    """``CodigoSSET -> COD_PREDIO`` for every linkable department (uses the build's cache)."""
    from crop_classifier.allperu.build_labels import bridge_for

    parts = []
    for d in depts or departments():
        b = bridge_for(d)
        b = b.copy()
        b["dept"] = d.name
        parts.append(b)
    return pd.concat(parts, ignore_index=True)


def tenure_by_predio(depts: list[Dept] | None = None, policy: str = "any",
                     cache_dir: Path | None = None) -> pd.DataFrame:
    """The table everything downstream joins: one row per ``COD_PREDIO``.

    Several ``CodigoSSET`` keys bridge to one ``COD_PREDIO``, so tenure is reconciled twice
    under the same ``policy`` — within a key, then across a polygon's keys.
    """
    depts = depts or departments()
    recs = tenure_records(depts, cache_dir=cache_dir)
    by_key = aggregate(recs, "CodigoSSET", policy=policy)
    pairs = bridge_pairs(depts)
    joined = pairs.merge(by_key, on="CodigoSSET", how="inner")
    joined["_insc"] = (joined["tenure"] == INSCRITO).astype(float)
    grp = joined.groupby("COD_PREDIO", observed=True, sort=False)
    frac = grp["_insc"].mean()
    out = pd.DataFrame({
        "tenure": _resolve(frac, policy),
        "frac_inscrito": frac,
        "tenure_conflict": frac.between(0.0, 1.0, inclusive="neither"),
        "n_tenure_records": grp["n_tenure_records"].sum(),
        "n_sset_keys": grp.size(),
        "reg_year": grp["reg_year"].min().astype("Int64"),
        "dept": grp["dept"].first(),
    }).reset_index()
    return out


# --- A SECOND, LATER tenure observation — the answer to RESULTS.md §7 ---
# The bridge .dta also carries the cadastre's titling-pipeline status (`estado`) and its cut
# date (`fech_tran`, ≈2011-2012). With BD SSET's `ESTADO en RRPP` (status at declaration,
# ~1997-2006) that is two dated observations of registration status per parcel — what §6.2
# assumed did not exist. Caveats: `fech_tran` dates the snapshot not the inscription event
# (timing is bounded, not observed); `estado` is a pipeline state, not the same binary
# ("RRPP: ENVIADO" is neither); one extra period -> two-period DiD, not an event study.
_REGISTERED_PREFIXES = ("RRPP: PROPIEDAD INSCRITA", "PREDIO INSCRITO ANTES DEL PETT",
                        "RRPP: RECTIFICACION INSCRITA")
_POSSESSION = ("RRPP: POSESION INSCRITA",)
_IN_PROCESS_PREFIXES = ("RRPP:", "EXP:")


def classify_estado(s: pd.Series) -> pd.Series:
    """Cadastre ``estado`` -> ``REGISTERED`` / ``POSSESSION`` / ``IN_PROCESS`` / ``NONE``."""
    t = (s.astype(str)
         .map(lambda x: unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode())
         .str.upper().str.strip())
    out = pd.Series("NONE", index=s.index, dtype=object)
    out[t.str.startswith(_IN_PROCESS_PREFIXES)] = "IN_PROCESS"
    out[t.str.startswith(_POSSESSION)] = "POSSESSION"
    out[t.str.startswith(_REGISTERED_PREFIXES)] = "REGISTERED"
    return out


def cadastre_status(dept: Dept, cache_dir: Path | None = None) -> pd.DataFrame:
    """One department's later (≈2011) tenure observation, per ``COD_PREDIO``."""
    cd = cache_dir or CACHE
    cache = cd / f"cadastre_status_{dept.name}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    import pyreadstat

    df, _ = pyreadstat.read_dta(
        str(dept.bridge),
        usecols=["COD_PREDIO", "CodigoSSET", "estado", "condicion", "fech_tran"],
        encoding="latin1")
    df["COD_PREDIO"] = df["COD_PREDIO"].astype(str).str.strip()
    df["CodigoSSET"] = canon_key(df["CodigoSSET"])
    df["cadastre_status"] = classify_estado(df["estado"])
    day = pd.to_numeric(df["fech_tran"], errors="coerce")
    df["cadastre_date"] = pd.Timestamp("1960-01-01") + pd.to_timedelta(day, unit="D")
    df["dept"] = dept.name
    out = (df[["COD_PREDIO", "dept", "cadastre_status", "condicion", "cadastre_date"]]
           .dropna(subset=["COD_PREDIO"])
           .sort_values("cadastre_status")          # REGISTERED sorts first among ties
           .drop_duplicates("COD_PREDIO"))
    cd.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


def tenure_two_period(depts: list[Dept] | None = None,
                      policy: str = "any") -> pd.DataFrame:
    """Join the ~1998-2006 SSET tenure to the ~2011 cadastre status, per parcel.

    Adds ``became_registered`` (NO INSCRITO at declaration, REGISTERED in the cadastre) —
    the two-period DiD treatment; its base rate decides whether §6.2's route is available.
    """
    depts = depts or departments()
    t0 = tenure_by_predio(depts, policy=policy)
    t1 = pd.concat([cadastre_status(d) for d in depts], ignore_index=True)
    out = t0.merge(t1.drop(columns="dept"), on="COD_PREDIO", how="inner")
    out["became_registered"] = ((out["tenure"] == NO_INSCRITO)
                                & (out["cadastre_status"] == "REGISTERED"))
    return out


def attach(parcels: pd.DataFrame, tenure: pd.DataFrame | None = None,
           policy: str = "any") -> pd.DataFrame:
    """Left-join tenure onto any table keyed by ``COD_PREDIO``, keeping unmatched rows.

    Unmatched keep ``tenure = NaN``, not dropped — §2's population is parcels with non-null
    tenure, and the loss must be visible.
    """
    t = tenure if tenure is not None else tenure_by_predio(policy=policy)
    cols = [c for c in ("COD_PREDIO", "tenure", "frac_inscrito", "tenure_conflict",
                        "n_tenure_records", "n_sset_keys", "reg_year") if c in t.columns]
    return parcels.merge(t[cols], on="COD_PREDIO", how="left")


def audit(policy: str = "any", save: bool = True) -> pd.DataFrame:
    """Per-department tenure availability + conflict rate, at record and parcel level."""
    depts = departments()
    recs = tenure_records(depts)
    pred = tenure_by_predio(depts, policy=policy)
    rows = []
    for d in sorted({x.name for x in depts}):
        r = recs[recs.dept == d]
        p = pred[pred.dept == d]
        rows.append({
            "dept": d,
            "records": len(r),
            "records_null_tenure": int(r["tenure"].isna().sum()),
            "record_inscrito": round(float((r["tenure"] == INSCRITO).mean()), 4)
            if len(r) else np.nan,
            "predios": len(p),
            "predio_inscrito": round(float((p["tenure"] == INSCRITO).mean()), 4)
            if len(p) else np.nan,
            "predio_conflict": round(float(p["tenure_conflict"].mean()), 4)
            if len(p) else np.nan,
        })
    out = pd.DataFrame(rows).sort_values("predio_inscrito", ascending=False)
    print(out.to_string(index=False))
    bad = recs.loc[recs["tenure"].isna(), "tenure_raw"].value_counts().head(10)
    if len(bad):
        print("\nunparsed ESTADO en RRPP values (top 10):")
        print(bad.to_string())
    if save:
        from crop_classifier.paths import proc
        f = proc() / "tenure_audit.csv"
        out.to_csv(f, index=False)
        pred.to_parquet(proc() / "tenure_by_predio.parquet", index=False)
        print(f"\nwrote {f} and tenure_by_predio.parquet ({len(pred):,} parcels)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", default="any", choices=["any", "mode"])
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    audit(policy=a.policy, save=not a.no_save)


if __name__ == "__main__":
    main()

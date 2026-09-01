"""Slim the 25 CENAGRO 2012 department files from OneDrive into per-department Parquet.

The source is `Departamentos_IV_CENAGRO (sin posesionario)/<Dept>.dta` — 25 Stata files,
~18 GB, **409 columns each**, one row per *producer x parcel x crop-order*. Almost none of
those columns are used by this project, and reading a 2.5 GB `.dta` to get eight of them is
what makes every census script slow.

This module reads each file **once**, with an explicit `usecols`, and writes
`data/raw/Cenagro_IV/<Dept>.parquet`. The output is still **long** — one row per
parcel x crop-order — because every downstream consumer aggregates differently
(`allperu/cenagro.py` collapses to a class per polygon; notebook 02 sums area per producer).

⚠️ **The farmer's name lives in UNLABELLED columns.** `P009_01/02/03` carry apellido paterno /
materno / nombres and have an **empty** variable label in the `.dta` — they must be identified
by value, not by label. They are the *only* link from the census to the rest of the project
(the census carries no `COD_PREDIO` and no `CodigoSSET`), so they are non-negotiable in the
keep-list. `P009_04` is the razón social for juridical producers and `P009_05` the RUC.

⚠️ **"sin posesionario" is a folder name, not a filter.** The posesionario tenure records
(`P037_04_01`) are present in every department — see `docs/DATA.md` §1.5 for the measured
rates. Nothing has been stripped, so the 25 files remain comparable with each other.

Run:  uv run python -m crop_classifier.cli cenagro-extract --all
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyreadstat

from crop_classifier.paths import ROOT


def src_dir() -> Path:
    """The 25 CENAGRO 2012 department ``.dta`` files (~17.5 GB).

    Configured as ``cenagro_source_dir`` in ``workspaces.yaml`` — it was an absolute path to
    one machine's OneDrive mount, which nobody else could have. A function, not a constant,
    so importing this module does not require the share to be configured or mounted: only
    the two callers below need it, and only when actually extracting.
    """
    from crop_classifier.workspace import cenagro_source_dir
    return cenagro_source_dir()
OUT_DIR = ROOT / "data" / "raw" / "Cenagro_IV"
CROP_TABLE = ROOT / "data" / "raw" / "IV CENAGRO - Tabla_Cultivos_Totales.xlsx"

# rows read per pass; Cajamarca and Puno are >2 GB and must not be held whole
CHUNK = 1_200_000

# --------------------------------------------------------------------------------------
# The keep-list. Grouped by the level of the census hierarchy each column belongs to; a
# producer-level value is repeated on every one of that producer's rows.
KEEP: dict[str, list[str]] = {
    # --- producer / holding identity ---------------------------------------------------
    "keys": ["TIPO_REC", "P001", "P002", "P003", "P007X", "P008", "NPRIN", "RESULTADO"],
    # --- the name: the ONLY link to PETT. Unlabelled in the .dta; verified by value. -----
    "name": ["P009_01", "P009_02", "P009_03", "P009_04", "P009_05"],
    # --- location -----------------------------------------------------------------------
    "geo": ["LONG_DECI", "LAT_DECI", "WALTITUD", "WREGION", "WPISO"],
    # --- producer attributes already consumed downstream --------------------------------
    "producer": ["P016", "WP111", "WP112", "P019", "P019_01", "P020_01", "P021",
                 "P022", "P022_01"],
    # --- INEI's own land-use totals for the holding (ha) --------------------------------
    "wsup": ["WSUP03", "WSUP04", "WSUP07", "WSUP10", "WSUP11", "WSUP12", "WSUP13",
             "WSUP18"],
    # --- parcel identity and tenure regime ----------------------------------------------
    "parcel": ["NPARC", "P023_01", "P023_04", "P023_07"],
    "tenure": ["P037_01_01", "P037_01_02", "P037_01_03",
               "P037_02_01", "P037_02_02",
               "P037_03_01", "P037_03_02",
               "P037_04_01", "P037_04_02",
               "P037_05_01", "P037_05_02",
               "P037_SS",
               "P038_01", "P038_02", "P038_03", "P038_04", "P039_01", "P040"],
    # ⚠️ undocumented derived registration variables shipped with the source. They are
    # *nearly* but not exactly reproducible from P037_01_03 (96-98 %); DATA.md §1.5.
    # Kept for cross-checking only — derive tenure security from P037_01_03.
    "registered": ["registrado", "registrado_1", "registrado_2", "p_registrado",
                   "GP", "GP_1", "X_t"],
    # --- crop -----------------------------------------------------------------------------
    "crop": ["P024_01", "P024_03", "P025", "P026", "P027", "P028",
             "P029_01", "P029_02", "P029_03", "P036"],
}
USECOLS: list[str] = [c for v in KEEP.values() for c in v]

# columns the source stores as text; everything else is read as float64. Declared rather
# than inferred because a chunked write needs one schema for the whole file and pandas
# will happily give chunk 1 an object column and chunk 2 a float one.
STR_COLS = {"TIPO_REC", "P001", "P002", "P003", "P007X", "P008", "NPRIN",
            "P009_01", "P009_02", "P009_03", "P009_04", "P023_07", "P024_03",
            "P037_01_03"}
# derived, added here
DERIVED_STR = ["dept", "UBIGEO", "P009_05"]

SCHEMA = pa.schema(
    [(c, pa.string() if c in STR_COLS else pa.float64()) for c in USECOLS if c != "P009_05"]
    + [(c, pa.string()) for c in DERIVED_STR]
)

# 14 departments reach a parcel polygon through the cadastral bridge (DATA.md §3). The
# bridge ships 15 files; Callao then yields nothing (395 polygons, no surviving overlap).
# The other 10 census files are extracted anyway — they are still valid census data for a
# national descriptive, they just cannot be joined to a polygon or to a PETT crop.
LINKABLE = {"Ancash", "Arequipa", "Ayacucho", "Cajamarca", "Huancavelica", "Ica",
            "La_Libertad", "Lambayeque", "Lima", "Moquegua", "Pasco", "Piura", "Tacna",
            "Tumbes"}


# --------------------------------------------------------------------------------------
def _tidy(df: pd.DataFrame, dept: str) -> pd.DataFrame:
    """One chunk -> the declared schema, plus `dept` and `UBIGEO`."""
    for c in USECOLS:
        if c not in df.columns:
            df[c] = pd.NA
    for c in STR_COLS:
        s = df[c]
        if not pd.api.types.is_string_dtype(s):
            # a code column that arrived numeric: render it as an integer string
            s = s.map(lambda v: "" if pd.isna(v) else str(int(v)) if float(v).is_integer()
                      else str(v))
        df[c] = s.astype("string").str.strip().replace({"": pd.NA, ".": pd.NA})
    # RUC: 11 digits, stored as a float. Keep it as text — it is an identifier.
    df["P009_05"] = df["P009_05"].map(
        lambda v: pd.NA if pd.isna(v) else f"{int(v):011d}").astype("string")
    for c in USECOLS:
        if c not in STR_COLS and c != "P009_05":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["dept"] = dept
    df["UBIGEO"] = (df["P001"].fillna("").str.zfill(2)
                    + df["P002"].fillna("").str.zfill(2)
                    + df["P003"].fillna("").str.zfill(2)).astype("string")
    return df[[f.name for f in SCHEMA]]


def extract_one(dept: str, out_dir: Path = OUT_DIR, overwrite: bool = False) -> dict:
    """Read `<dept>.dta` in chunks, write `<dept>.parquet`, return the audit row."""
    src = src_dir() / f"{dept}.dta"
    if not src.exists():
        raise FileNotFoundError(src)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{dept}.parquet"
    if dst.exists() and not overwrite:
        print(f"  {dept}: exists, skipping (use --overwrite)")
        return {"dept": dept, "status": "skipped"}

    t0 = time.time()
    _, meta = pyreadstat.read_dta(str(src), metadataonly=True, encoding="latin1")
    n_in, ncol_in = meta.number_rows, len(meta.column_names)
    missing = [c for c in USECOLS if c not in meta.column_names]

    rows_out = 0
    writer = pq.ParquetWriter(dst, SCHEMA, compression="zstd")
    try:
        reader = pyreadstat.read_file_in_chunks(
            pyreadstat.read_dta, str(src), chunksize=CHUNK,
            usecols=[c for c in USECOLS if c in meta.column_names], encoding="latin1")
        for chunk, _ in reader:
            tidy = _tidy(chunk, dept)
            writer.write_table(pa.Table.from_pandas(tidy, schema=SCHEMA,
                                                    preserve_index=False))
            rows_out += len(tidy)
    finally:
        writer.close()

    return {"dept": dept, "status": "ok",
            "rows_in": n_in, "rows_out": rows_out,
            "cols_in": ncol_in, "cols_out": len(SCHEMA),
            "mb_in": round(src.stat().st_size / 1e6, 1),
            "mb_out": round(dst.stat().st_size / 1e6, 1),
            "missing_cols": ",".join(missing),
            "secs": round(time.time() - t0, 1),
            "linkable": dept in LINKABLE}


def departments() -> list[str]:
    return sorted(p.stem for p in src_dir().glob("*.dta"))


def extract(depts: list[str] | None = None, overwrite: bool = False) -> pd.DataFrame:
    rows = []
    for d in depts or departments():
        print(f"[{d}] ...", flush=True)
        r = extract_one(d, overwrite=overwrite)
        rows.append(r)
        if r["status"] == "ok":
            print(f"  {r['rows_in']:,} rows / {r['cols_in']} cols "
                  f"({r['mb_in']:,.0f} MB)  ->  {r['rows_out']:,} rows / "
                  f"{r['cols_out']} cols ({r['mb_out']:,.1f} MB)  "
                  f"[{r['mb_in'] / max(r['mb_out'], 1e-9):.0f}x]  {r['secs']}s", flush=True)
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "_extract_audit.csv", index=False)
    return df


# --------------------------------------------------------------------------------------
def crop_code_table() -> pd.DataFrame:
    """The official question-024 crop code list, sheet "Permanente".

    Despite the sheet name it is the **whole** crop vocabulary (3,351 codes, transitory
    included) with an `Exportable` flag. Codes are 1-4 characters and are **not**
    zero-padded consistently on either side, so both sides are canonicalised to an int.
    """
    d = pd.read_excel(CROP_TABLE, sheet_name="Permanente", skiprows=3,
                      usecols=[1, 2, 3], names=["CODIGO", "TITULO", "Exportable"])
    d = d.dropna(subset=["CODIGO"]).copy()
    d["code"] = pd.to_numeric(d["CODIGO"], errors="coerce").astype("Int64")
    return d.dropna(subset=["code"])


def verify(out_dir: Path = OUT_DIR) -> dict[str, pd.DataFrame]:
    """Everything that has to be checked before this extraction is used. Prints and returns.

    Five checks, each of which has a way of failing silently if it is not made:

    1. **rows in vs rows out** — a chunked write that loses a chunk still produces a valid
       Parquet file.
    2. **`P009_01` non-blank rate** — the farmer's surname is the only link to PETT. It is
       *unlabelled* in the `.dta`, so a wrong column would still read as a string column.
    3. **identical column set across the 25 files** — anything else breaks a national concat.
    4. **`P024_03` resolves against the crop table** — an unresolved code silently becomes a
       missing crop rather than an error (`CLAUDE.md`, the unmapped-token trap).
    5. **the posesionario rate** — the source folder is named "sin posesionario". If that
       were a row filter it would remove a *tenure category* from some departments and not
       others, and every cross-department tenure comparison would be invalid.
    """
    files = sorted(out_dir.glob("*.parquet"))
    audit = pd.read_csv(out_dir / "_extract_audit.csv") if (
        out_dir / "_extract_audit.csv").exists() else pd.DataFrame()
    tbl = crop_code_table()
    known = set(tbl["code"].dropna().astype(int))

    rows, colsets = [], {}
    for f in files:
        d = pd.read_parquet(f, columns=["NPRIN", "P009_01", "P009_04", "P024_03",
                                        "P037_01_01", "P037_04_01", "P037_02_01",
                                        "P037_03_01", "P037_05_01", "P025"])
        colsets[f.stem] = tuple(pq.ParquetFile(f).schema_arrow.names)
        prod = d.drop_duplicates("NPRIN")
        code = pd.to_numeric(d["P024_03"], errors="coerce").astype("Int64")
        has_code = code.notna()
        unres = has_code & ~code.isin(known)
        rows.append({
            "dept": f.stem,
            "rows": len(d),
            "producers": prod["NPRIN"].nunique(),
            "P009_01_nonblank_pct": round(100 * prod["P009_01"].notna().mean(), 1),
            "P009_04_only_pct": round(
                100 * (prod["P009_01"].isna() & prod["P009_04"].notna()).mean(), 1),
            "crop_rows_pct": round(100 * has_code.mean(), 1),
            "crop_code_unresolved_pct": round(100 * unres.sum() / max(has_code.sum(), 1), 2),
            "propietario_pct": round(100 * (prod["P037_01_01"] == 1).mean(), 1),
            "posesionario_pct": round(100 * (prod["P037_04_01"] == 1).mean(), 1),
            "comunero_pct": round(100 * (prod["P037_02_01"] == 1).mean(), 1),
            "arrendatario_pct": round(100 * (prod["P037_03_01"] == 1).mean(), 1),
            "linkable": f.stem in LINKABLE,
        })
    v = pd.DataFrame(rows)

    ref = colsets[files[0].stem]
    v["cols_match"] = [colsets[s] == ref for s in v["dept"]]

    if len(audit):
        v = v.merge(audit[["dept", "rows_in", "rows_out", "mb_in", "mb_out"]],
                    on="dept", how="left")
        v["rows_ok"] = v["rows_in"] == v["rows"]

    print(f"\n=== CENAGRO 2012 extraction audit — {len(files)} departments ===")
    print(v.to_string(index=False))
    bad = list(v.loc[~v["cols_match"], "dept"])
    print(f"\ncolumn set identical across all {len(files)} files: "
          f"{'YES' if not bad else 'NO — ' + str(bad)}")
    if "rows_ok" in v:
        print(f"rows out == rows in for every department: "
              f"{'YES' if v['rows_ok'].all() else 'NO'}")
        print(f"total: {v['rows_in'].sum():,} rows, "
              f"{v['mb_in'].sum() / 1000:.1f} GB -> {v['mb_out'].sum() / 1000:.2f} GB "
              f"({v['mb_in'].sum() / v['mb_out'].sum():.0f}x smaller)")
    lo = v.nsmallest(3, "P009_01_nonblank_pct")[["dept", "P009_01_nonblank_pct"]]
    print(f"\nlowest P009_01 (apellido paterno) non-blank rates:\n{lo.to_string(index=False)}")
    print(f"\nposesionario (P037_04_01) share of producers: "
          f"{v['posesionario_pct'].min():.1f}%-{v['posesionario_pct'].max():.1f}%, "
          f"present in {(v['posesionario_pct'] > 0).sum()}/{len(v)} departments")
    return {"per_dept": v, "crop_table": tbl}

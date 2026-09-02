"""T6 — external validation against MIDAGRI/SIEA district statistics (RESULTS.md §5).

Everything else here is internal: the classifier is checked against the same PETT
declarations it trained on. That cannot detect a *systematic* error shared by training and
inference — chiefly the §6.1 risk that **the export boom happened largely outside this
cadastre**, on newly irrigated desert developed by agro-export firms not 0.5 ha PETT
smallholdings. If so, a true 2 → 4 % conversion rate on titled smallholdings is the *answer*,
not a detection failure, and no classifier improvement recovers the rest.

MIDAGRI's SIEA publishes district-level harvested hectares by crop back to the 1990s
(*Series históricas de producción agrícola*, https://siea.midagri.gob.pe/portal/ ; also on
https://www.datosabiertos.gob.pe/). Not redistributed with this repo and not fetched
automatically: pass a downloaded CSV.

The comparison is deliberately weak-form, because the two quantities are not the same thing:

* SIEA measures **harvested hectares of all farmland** in a district; this project measures
  the **share of titled PETT parcels** predicted perennial. A level match is not expected.
* What *is* comparable is the **direction and the cross-district ranking** of perennial
  growth. If districts where SIEA says perennial area tripled are not the districts where the
  model says the perennial share rose, the model is not tracking the phenomenon at all.

So the reported statistics are Spearman rank correlation of district-level change, plus the
share-of-agreement in sign — not a regression coefficient anyone should quote as a level.

Expected CSV columns (rename with ``--map``): ``district`` (or ``ubigeo``), ``year``,
``crop``, ``area_ha``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

from crop_classifier.allperu.build_labels import CACHE
from crop_classifier.allperu.export_crops import classify
from crop_classifier.allperu.sources import Dept, departments
from crop_classifier.archive import windows as W
from crop_classifier.paths import proc


def parcel_districts(depts: list[Dept] | None = None) -> pd.DataFrame:
    """``COD_PREDIO -> (dept, provincia, distrito, ubigeo)`` from the bridge tables, cached.

    The modelling tables carry no administrative unit below department; the bridge .dta does
    (``DISTRITO``, ``PROVINCIA``, ``id_dist``), and it is the only place a district lives.
    """
    parts = []
    for d in depts or departments():
        cache = CACHE / f"district_{d.name}.parquet"
        if cache.exists():
            parts.append(pd.read_parquet(cache))
            continue
        df, _ = pyreadstat.read_dta(
            str(d.bridge), usecols=["COD_PREDIO", "DEPARTAMENTO", "PROVINCIA", "DISTRITO",
                                    "id_dist"], encoding="latin1")
        df["COD_PREDIO"] = df["COD_PREDIO"].astype(str).str.strip()
        df = df.rename(columns={"id_dist": "ubigeo"})
        df["dept"] = d.name
        df = df.drop_duplicates("COD_PREDIO")
        CACHE.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def load_siea(path: Path, colmap: dict[str, str] | None = None) -> pd.DataFrame:
    """Read a SIEA district-crop-year area table and tag each crop EXPORT/MIXED/DOMESTIC."""
    df = pd.read_csv(path)
    if colmap:
        df = df.rename(columns=colmap)
    missing = {"year", "crop", "area_ha"} - set(df.columns)
    if missing:
        raise KeyError(f"{path}: missing columns {sorted(missing)} — use --map to rename")
    if "ubigeo" not in df.columns and "district" not in df.columns:
        raise KeyError(f"{path}: needs `ubigeo` or `district`")
    df["crop"] = df["crop"].astype(str).str.upper().str.strip()
    df["export_status"] = df["crop"].map(classify)
    return df


def district_perennial_change(preds_path: Path, tenure_path: Path | None = None,
                              pre: str = "W99", post: str = "W19") -> pd.DataFrame:
    """Model-side district series: weighted perennial share per window, and its change."""
    preds = pd.read_parquet(preds_path)
    wt = W.window_table(preds)
    dis = parcel_districts()
    wt = wt.merge(dis[["COD_PREDIO", "ubigeo", "DISTRITO", "dept"]], on="COD_PREDIO",
                  how="left")
    wt = wt[wt["ubigeo"].notna()]
    sh = W.share_by(wt, ["ubigeo"])
    piv = sh.pivot(index="ubigeo", columns="window", values="share")
    n = sh.pivot(index="ubigeo", columns="window", values="n")
    out = pd.DataFrame({"share_pre": piv.get(pre), "share_post": piv.get(post),
                        "n_pre": n.get(pre), "n_post": n.get(post)})
    out["model_change"] = out["share_post"] - out["share_pre"]
    return out.reset_index()


def compare(preds_path: Path, siea_path: Path, colmap: dict[str, str] | None = None,
            pre_years: tuple[int, int] = (1999, 2003),
            post_years: tuple[int, int] = (2019, 2023),
            min_parcels: int = 30, save: bool = True) -> dict:
    """Rank-compare model district perennial growth against SIEA perennial-area growth."""
    from scipy.stats import spearmanr

    model = district_perennial_change(preds_path)
    model = model[(model["n_pre"] >= min_parcels) & (model["n_post"] >= min_parcels)]

    siea = load_siea(Path(siea_path), colmap)
    key = "ubigeo" if "ubigeo" in siea.columns else "district"
    per = siea[siea["export_status"].isin({"EXPORT", "MIXED"})]
    pre = (per[per["year"].between(*pre_years)].groupby(key)["area_ha"].mean()
           .rename("siea_pre"))
    post = (per[per["year"].between(*post_years)].groupby(key)["area_ha"].mean()
            .rename("siea_post"))
    tot_pre = (siea[siea["year"].between(*pre_years)].groupby(key)["area_ha"].mean()
               .rename("siea_total_pre"))
    tot_post = (siea[siea["year"].between(*post_years)].groupby(key)["area_ha"].mean()
                .rename("siea_total_post"))
    s = pd.concat([pre, post, tot_pre, tot_post], axis=1)
    s["siea_share_pre"] = s["siea_pre"] / s["siea_total_pre"]
    s["siea_share_post"] = s["siea_post"] / s["siea_total_post"]
    s["siea_change"] = s["siea_share_post"] - s["siea_share_pre"]

    j = model.merge(s.reset_index().rename(columns={key: "ubigeo"}), on="ubigeo",
                    how="inner").dropna(subset=["model_change", "siea_change"])
    if j.empty:
        raise ValueError("no districts matched — check the ubigeo/district key formats")
    rho, p = spearmanr(j["model_change"], j["siea_change"])
    sign_agree = float((np.sign(j["model_change"]) == np.sign(j["siea_change"])).mean())
    out = {
        "n_districts": int(len(j)),
        "spearman_rho": float(rho), "spearman_p": float(p),
        "sign_agreement": sign_agree,
        "model_mean_change": float(j["model_change"].mean()),
        "siea_mean_change": float(j["siea_change"].mean()),
        "note": "Levels are NOT comparable — SIEA counts harvested hectares of all farmland, "
                "this counts the share of titled PETT parcels. Only the direction and the "
                "cross-district ranking are.",
    }
    print(json.dumps(out, indent=2))
    if save:
        j.to_csv(proc() / "external_district_comparison.csv", index=False)
        with open(proc() / "external_validation.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote external_validation.json + external_district_comparison.csv to {proc()}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--siea", type=Path, required=True,
                    help="downloaded MIDAGRI/SIEA district-crop-year CSV")
    ap.add_argument("--map", default=None,
                    help="JSON dict renaming SIEA columns, e.g. "
                         "'{\"anio\":\"year\",\"cultivo\":\"crop\",\"sup_cosechada\":\"area_ha\"}'")
    ap.add_argument("--min-parcels", type=int, default=30)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    compare(a.preds, a.siea, json.loads(a.map) if a.map else None,
            min_parcels=a.min_parcels, save=not a.no_save)


if __name__ == "__main__":
    main()

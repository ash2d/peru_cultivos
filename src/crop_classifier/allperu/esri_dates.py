"""Gate zero for the endpoint labelling campaign: what imagery date would a labeller see?

Tile counts from a bbox query are not the decision-relevant number — one huge old tile and
one small new one count the same. This samples real parcel centroids from the at-risk pool
and asks the Esri World Imagery metadata service what the acquisition date is AT THAT POINT,
which is what the labeller actually gets.

Queries the high-resolution metadata layers only (30cm / 60cm / 1.2m); anything coarser is
not usable for calling tree crowns.
"""
from __future__ import annotations

import datetime as dt
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

BASE = ("https://services.arcgisonline.com/arcgis/rest/services/"
        "World_Imagery/MapServer/{layer}/query")
LAYERS = {9: "30cm", 10: "60cm", 11: "1.2m"}
DEPTS = ["LA_LIBERTAD", "CAJAMARCA", "PIURA", "LAMBAYEQUE"]
N_PER_DEPT = 60


def probe(args):
    cod, lon, lat, dept = args
    best = None
    for layer, res in LAYERS.items():
        try:
            r = requests.get(BASE.format(layer=layer), timeout=45, params={
                "geometry": f"{lon},{lat}", "geometryType": "esriGeometryPoint",
                "inSR": 4326, "spatialRel": "esriSpatialRelIntersects",
                "outFields": "SRC_DATE2,SRC_RES,SRC_DESC", "returnGeometry": "false",
                "f": "json"})
            feats = r.json().get("features", [])
        except Exception:
            feats = []
        for f in feats:
            a = f["attributes"]
            v = a.get("SRC_DATE2")
            if not v:
                continue
            d = dt.datetime.utcfromtimestamp(v / 1000)
            # Resolution is already fixed by the layer loop, which stops at the first
            # layer that answers — so the only choice left is between several dated
            # footprints overlapping this point at that resolution. Take the **most
            # recent**: it is the one Esri is most likely to be rendering, and this is a
            # 2019+ endpoint campaign.
            #
            # This previously read `(res, d) < (best["res"], best["date"])`, which with
            # `res` constant keeps the *oldest* — the opposite of what the comment beside
            # it claimed. Measured before changing it: of 200 sampled campaign parcels,
            # **0 have more than one dated feature**, so the branch never fired and no
            # drawn parcel's `imagery_date` changes. Fixed so the next dataset does not
            # inherit a silent disagreement between code and comment.
            if best is None or d > best["date"]:
                best = {"res": res, "date": d, "desc": a.get("SRC_DESC"),
                        "src_res": a.get("SRC_RES")}
        if best is not None:
            break            # 30cm beats 60cm beats 1.2m; stop at the first hit
    return {"COD_PREDIO": cod, "dept": dept, "lon": lon, "lat": lat,
            "res": best["res"] if best else None,
            "year": best["date"].year if best else None,
            "date": best["date"].date().isoformat() if best else None,
            "desc": best["desc"] if best else None}


def probe_frame(parcels: gpd.GeoDataFrame, workers: int = 8,
                cache: Path | None = None, chunk: int = 500) -> pd.DataFrame:
    """Probe an **arbitrary** parcel frame — one row per ``COD_PREDIO``.

    The original :func:`main` hard-codes the at-risk pool and 4 departments; the S2
    labelling campaign needs the same probe over a national candidate pool of thousands
    (s2_labelling_plan.md §4), so the loop lives here and ``main`` keeps its behaviour.

    ``cache`` is a CSV appended chunk by chunk. ~2,500 probes are ~2,500 HTTPS round trips
    and a dropped connection halfway through should not cost the whole run, so anything
    already in the cache is skipped on a re-run. Persist the **full** result including
    ineligible parcels — the eligible fraction per department is gate G0.
    """
    need = parcels.copy()
    done = pd.DataFrame()
    if cache is not None and Path(cache).exists():
        done = pd.read_csv(cache)
        need = need[~need["COD_PREDIO"].astype(str).isin(done["COD_PREDIO"].astype(str))]
        print(f"cache: {len(done):,} already probed, {len(need):,} to go")
    if len(need):
        c = need.geometry.representative_point()
        jobs = list(zip(need["COD_PREDIO"].astype(str), c.x, c.y, need["dept"]))
        for i in range(0, len(jobs), chunk):
            part = jobs[i:i + chunk]
            with ThreadPoolExecutor(max_workers=workers) as ex:
                rows = list(ex.map(probe, part))
            df = pd.DataFrame(rows)
            done = pd.concat([done, df], ignore_index=True)
            if cache is not None:
                done.to_csv(cache, index=False)
            print(f"  probed {min(i + chunk, len(jobs)):,}/{len(jobs):,}", flush=True)
    return done.drop_duplicates("COD_PREDIO").reset_index(drop=True)


def eligible(df: pd.DataFrame, max_res_m: float = 1.2,
             min_year: int = 2019) -> pd.Series:
    """§1 filter 4: imagery at the centroid is <= ``max_res_m`` and dated >= ``min_year``."""
    res_m = df["res"].map({"30cm": 0.3, "60cm": 0.6, "1.2m": 1.2})
    return res_m.notna() & (res_m <= max_res_m) & df["year"].fillna(0).ge(min_year)


def main(out_path: str) -> None:
    p = gpd.read_parquet("data/processed/all_peru/modeling_parcels.parquet")
    p = p[p["label"] == "ANNUAL"]                     # the at-risk pool
    jobs = []
    for d in DEPTS:
        sub = p[p["dept"] == d]
        if sub.empty:
            continue
        sub = sub.sample(min(N_PER_DEPT, len(sub)), random_state=0)
        c = sub.geometry.centroid
        jobs += list(zip(sub["COD_PREDIO"], c.x, c.y, sub["dept"]))
    print(f"probing {len(jobs)} parcel centroids …", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(probe, jobs))
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    print("\n=== imagery year at parcel centroid, by department ===")
    print(pd.crosstab(df["dept"], df["year"], dropna=False).to_string())
    print("\n=== best resolution available ===")
    print(pd.crosstab(df["dept"], df["res"].fillna("NONE")).to_string())
    print("\n=== share of parcels whose imagery falls in a usable endpoint window ===")
    for lo, hi, name in [(2019, 2023, "2019-2023 (S2 endpoint)"),
                         (2015, 2018, "2015-2018 (early S2)"),
                         (2024, 2030, "2024+ (after S2 window)"),
                         (1990, 2014, "pre-2015 (unusable)")]:
        m = df["year"].between(lo, hi)
        print(f"  {name:26s} {m.mean():6.1%}   " +
              "  ".join(f"{d}:{df.loc[df.dept == d, 'year'].between(lo, hi).mean():.0%}"
                        for d in DEPTS))
    miss = df["year"].isna().mean()
    print(f"  {'no high-res metadata':26s} {miss:6.1%}")


if __name__ == "__main__":
    main(sys.argv[1])

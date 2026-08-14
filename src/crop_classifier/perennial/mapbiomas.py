"""MapBiomas Peru benchmark: extraction + class mapping (plan §8).

MapBiomas Peru Collection 3 (1985-2024, 30 m, annual) is the only external product
covering the whole period, so it is the external benchmark — and, if it predicts the PETT
labels better than our model does, the honest recommendation is to use it instead (§8.4,
criterion S3).

**Independence caveat, to be restated wherever these numbers appear:** MapBiomas Peru is
itself Landsat-derived at 30 m, so it shares sensors, cloud regimes and mixed-pixel
problems with our model. It is a benchmark, not ground truth, and agreement between the
two is not evidence that either is correct.

Legend
------
Verified two ways rather than assumed (the plan's table was explicitly marked "do not
trust blind"):

1. the published MapBiomas **Collection 3 legend-code table** for the shared
   RAISG/Andean scheme (Bolivia Col-3 legend PDF, identical code structure), and
2. **empirically**, by cross-tabulating the codes inside parcels whose PETT crop we know:
   code 40 covers 54 % of ARROZ parcels' pixels but ~1 % of orchards' -> 40 is Rice;
   code 21 covers 84 % of mango/lime orchards; code 4 is the Piura dry forest.

**The finding that matters.** The codes actually present over the Piura parcel bbox,
1990-2024, are::

    3, 4, 5, 9, 11, 12, 13, 21, 23, 24, 25, 27, 29, 30, 32, 33, 40, 66, 68, 72

The only agricultural ones are **21 (mosaic of uses), 40 (rice) and 72 (other crops)**.
Codes 36/46/47/48 (perennial crop, coffee, citrus, other perennial) and 15 (pasture)
**never appear** — MapBiomas Peru Collection 3 assigns no perennial-crop and no pasture
class anywhere in Piura. Inside our parcels it is ~63 % class 21, with orchards (84 %),
coffee (45 %) and fallow (58 %) all landing in that same class.

So MapBiomas cannot separate perennial from annual in this landscape at all. That is a
*result*, not a bug: it settles criterion S3 in the custom model's favour, but for a
reason that must be reported plainly rather than as a win on a level playing field. The
codes are kept in ``LEGEND`` anyway so the mapping stays correct if a later collection
starts using them.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from crop_classifier.features.landsat_gee import _chunk_id, _run_chunk, _to_fc, init_ee
from crop_classifier.paths import proc

ASSET = ("projects/mapbiomas-public/assets/peru/collection3/"
         "mapbiomas_peru_collection3_integration_v1")
BAND = "classification_{year}"

# code -> (name, our class). ``None`` = excluded from the comparison; "MOSAIC" is its own
# category because class 21 has no clean mapping and is *large* here (§8.2).
LEGEND: dict[int, tuple[str, str | None]] = {
    3:  ("Forest", None),                        # incl. shade-coffee agroforestry
    4:  ("Open forest (dry forest)", None),      # Piura bosque seco
    5:  ("Mangrove", None),                      # Vice / San Pedro estuaries
    6:  ("Flooded forest", None),
    9:  ("Forest plantation", None),             # woody non-crop (D1) -> excluded
    11: ("Flooded grassland/shrubland", None),
    12: ("Grassland/shrubland", None),
    13: ("Other non-forest natural formation", None),
    15: ("Pasture", "PASTURE_FALLOW"),
    18: ("Agriculture", "ANNUAL"),
    21: ("Mosaic of uses", "MOSAIC"),
    23: ("Beach, dune, sand", None),
    24: ("Urban infrastructure", None),
    25: ("Other non-vegetated anthropic area", None),
    26: ("Water body", None),
    27: ("Not observed", None),
    29: ("Rocky outcrop", None),
    30: ("Mining", None),
    31: ("Aquaculture", None),
    32: ("Hypersaline tidal flat", None),
    33: ("River, lake", None),
    34: ("Glacier", None),
    36: ("Perennial crop", "PERENNIAL"),
    39: ("Soybean", "ANNUAL"),
    40: ("Rice", "ANNUAL"),
    41: ("Other temporary crop", "ANNUAL"),
    46: ("Coffee", "PERENNIAL"),
    47: ("Citrus", "PERENNIAL"),
    48: ("Other perennial crop", "PERENNIAL"),
    61: ("Salt flat", None),
    62: ("Cotton", "ANNUAL"),
    66: ("Scrubland (matorral)", None),
    68: ("Other non-vegetated natural area", None),
    72: ("Other crops", "ANNUAL"),
    81: ("Andean grassland/shrubland", None),
    82: ("Flooded Andean grassland/shrubland", None),
}


def code_name(code: int) -> str:
    if code not in LEGEND:
        raise KeyError(f"unknown MapBiomas code {code} — verify against the Collection-3 "
                       f"legend before adding it; silently passing it through would "
                       f"corrupt the comparison")
    return LEGEND[code][0]


def code_to_class(code: int) -> str | None:
    """Our 3-class label for a MapBiomas code. Unknown codes **raise** (§13)."""
    if code not in LEGEND:
        raise KeyError(f"unknown MapBiomas code {code} — verify against the Collection-3 "
                       f"legend before adding it")
    return LEGEND[code][1]


def histogram_to_row(hist: dict[str, float], cid: str, year: int) -> dict:
    """Mode class + purity + our-class fractions for one parcel-year.

    Our parcels are ~0.5 ha ≈ 5 Landsat pixels, so a single mixed pixel flips the mode:
    ``mb_purity`` (the mode's pixel fraction) is the right filter for a fair comparison.
    """
    counts = {int(float(k)): v for k, v in (hist or {}).items() if v}
    total = sum(counts.values())
    if not total:
        return {"COD_PREDIO": cid, "year": year, "mb_code": pd.NA, "mb_class": None,
                "mb_purity": np.nan, "mb_n_pixels": 0}
    code = max(counts, key=counts.get)
    row = {"COD_PREDIO": cid, "year": year, "mb_code": code,
           "mb_class": code_to_class(code), "mb_purity": counts[code] / total,
           "mb_n_pixels": int(total)}
    # fraction vector over our classes — a mode is not enough at 5 pixels per parcel
    frac: Counter = Counter()
    for c, v in counts.items():
        frac[code_to_class(c) or "OTHER"] += v / total
    for name in ("PERENNIAL", "ANNUAL", "PASTURE_FALLOW", "MOSAIC", "OTHER"):
        row[f"mb_frac_{name}"] = float(frac.get(name, 0.0))
    return row


# ------------------------------------------------------------------------------------
# extraction (mirrors landsat_gee's chunked, resumable, content-addressed pattern)
# ------------------------------------------------------------------------------------
def mapbiomas_chunk(chunk: gpd.GeoDataFrame, years: list[int]) -> pd.DataFrame:
    """One ``getInfo`` for a chunk over **all** years at once — every year is a band of
    one image, so the whole benchmark costs a fraction of the Landsat extraction."""
    ee = init_ee()
    bands = [BAND.format(year=y) for y in years]
    img = ee.Image(ASSET).select(bands)
    red = img.reduceRegions(collection=_to_fc(chunk),
                            reducer=ee.Reducer.frequencyHistogram(),
                            scale=30, tileScale=4).getInfo()
    rows = []
    for f in red["features"]:
        p = f["properties"]
        cid = p["cid"]
        for y, b in zip(years, bands):
            rows.append(histogram_to_row(p.get(b), cid, y))
    return pd.DataFrame(rows)


def run(parcels: gpd.GeoDataFrame | None = None, years: list[int] | None = None,
        chunk_size: int = 200, out: Path | None = None,
        max_chunks: int | None = None) -> pd.DataFrame:
    """Extract the MapBiomas panel for every parcel, resumable by chunk."""
    init_ee()
    years = list(years or range(1996, 2025))
    if parcels is None:
        parcels = gpd.read_parquet(proc() / "modeling_parcels.parquet")
    out = out or proc() / "mapbiomas_panel.parquet"
    chunk_dir = out.parent / "mapbiomas_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    grp = parcels.sort_values(["centroid_lon", "centroid_lat"])
    todo = [grp.iloc[i:i + chunk_size] for i in range(0, len(grp), chunk_size)]
    print(f"mapbiomas: {len(parcels):,} parcels x {len(years)} years "
          f"-> {len(todo)} chunks")

    done = 0
    for n, chunk in enumerate(todo):
        f = chunk_dir / f"mb_{_chunk_id(chunk)}.parquet"
        if f.exists():
            continue
        # _run_chunk passes (chunk, year); years are baked in here instead
        df = _run_chunk(lambda c, _y: mapbiomas_chunk(c, years), chunk, years[0])
        df.to_parquet(f, index=False)
        print(f"  [{n + 1}/{len(todo)}] {len(df):,} parcel-years", flush=True)
        done += 1
        if max_chunks is not None and done >= max_chunks:
            print(f"  max_chunks={max_chunks} reached — stopping (resumable)")
            break

    files = sorted(chunk_dir.glob("mb_*.parquet"))
    panel = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    panel = panel.drop_duplicates(["COD_PREDIO", "year"])
    panel.to_parquet(out, index=False)
    print(f"wrote {out}: {len(panel):,} parcel-years, "
          f"{panel.COD_PREDIO.nunique():,} parcels")
    return panel


# ------------------------------------------------------------------------------------
# the comparison (§8.4)
# ------------------------------------------------------------------------------------
def code_histogram(panel: pd.DataFrame) -> pd.DataFrame:
    """What MapBiomas actually assigns inside our parcels — build the map from this, not
    from an assumed legend."""
    h = (panel.groupby("mb_code").size().sort_values(ascending=False)
         .rename("n_parcel_years").reset_index())
    h["name"] = h["mb_code"].map(lambda c: code_name(int(c)))
    h["our_class"] = h["mb_code"].map(lambda c: code_to_class(int(c)))
    h["share"] = h["n_parcel_years"] / h["n_parcel_years"].sum()
    return h


def agreement_vs_pett(panel: pd.DataFrame, parcels: pd.DataFrame,
                      mosaic_as: str | None = None,
                      min_purity: float = 0.0) -> dict:
    """Score MapBiomas against the PETT labels at each parcel's label year (criterion S3).

    ``mosaic_as`` decides what class 21 counts as; report both with it excluded (None) and
    with it counted as whichever class maximises agreement, and **say which you did**.
    """
    from sklearn.metrics import f1_score

    ref = parcels[["COD_PREDIO", "label", "year"]]
    df = panel.merge(ref, on=["COD_PREDIO", "year"], how="inner")
    df = df[df["mb_purity"] >= min_purity]
    mb = df["mb_class"].copy()
    if mosaic_as is not None:
        mb = mb.replace({"MOSAIC": mosaic_as})
    keep = mb.isin(["PERENNIAL", "ANNUAL", "PASTURE_FALLOW"])
    y_true, y_pred = df.loc[keep, "label"], mb[keep]
    classes = ["ANNUAL", "PASTURE_FALLOW", "PERENNIAL"]
    if not len(y_true):
        return {"n": 0, "macro_f1": np.nan, "accuracy": np.nan,
                "mosaic_as": mosaic_as, "min_purity": min_purity}
    return {"n": int(len(y_true)),
            "coverage": float(keep.mean()),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                       labels=classes, zero_division=0)),
            "accuracy": float((y_true.values == y_pred.values).mean()),
            "mosaic_as": mosaic_as, "min_purity": min_purity}


def main() -> None:
    ap = argparse.ArgumentParser(description="MapBiomas Peru extraction (plan §8)")
    ap.add_argument("--years", default="1996-2024")
    ap.add_argument("--chunk-size", type=int, default=200)
    ap.add_argument("--max-chunks", type=int, default=None)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.years.split("-"))
    run(years=list(range(lo, hi + 1)), chunk_size=a.chunk_size,
        max_chunks=a.max_chunks)


if __name__ == "__main__":
    main()

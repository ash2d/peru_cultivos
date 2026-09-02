"""Turn a file of parcel polygons into the table the pipeline scores.

For someone who has polygons (shapefile, GeoPackage, GeoJSON, parquet) and wants a trained
model's reading of them. It writes ``modeling_parcels.parquet``, the one input every later
command reads, with placeholder labels — a list of parcels to predict, not training data.

Landsat model (calendar year, features from ``cc satellite extract``)::

    uv run python tools/prepare_parcels.py my_farms.shp --year 2020 --out data/mine

Sentinel-2 model (24-month window around a date, features from ``cc labelling campaign
extract``)::

    uv run python tools/prepare_parcels.py my_farms.shp --for s2 \\
        --imagery-date 2025-03-01 --out data/mine2025

The Sentinel-2 form also writes ``labels_s2/label_sample.parquet``, which is what
``cc labelling campaign extract`` and ``assemble`` read, and links the assembled feature table
to the name ``cc predict`` expects. docs/howto/04_predict_new_parcels.md walks the whole thing.

Every polygon needs an id and a date. The id is whatever column names the parcel, copied into
``COD_PREDIO``. Areas outside 0.09-50 ha are kept but reported, because the models were fitted
inside that range and a prediction outside it is an extrapolation.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import geopandas as gpd
import pandas as pd

CLIMATE_PARQUET = Path("data/processed/climate/parcel_climate_normals.parquet")
CLIMATE_ARMS = {"temp": ["tmean_c"], "rain": ["precip_mm_yr"],
                "both": ["tmean_c", "precip_mm_yr"]}

PIXEL_HA = 0.09          # one Landsat pixel, 30 x 30 m
AREA_MIN_HA, AREA_MAX_HA = 0.09, 50.0
STORES = ("landsat", "s2")


def _ag_year_end(date) -> int:
    """The agricultural year (Aug 1 - Jul 31) containing ``date``, named by its end year.

    The same convention as ``features/s2_gee.ag_year``: an image from March 2025 belongs to
    the season that started in August 2024, and this project calls that season 2025.
    """
    d = pd.Timestamp(date)
    return d.year + 1 if d.month >= 8 else d.year


def prepare(src: Path, year: int | None = None, year_col: str | None = None,
            id_col: str | None = None, out_dir: Path | None = None, save: bool = True,
            store: str = "landsat", imagery_date: str | None = None,
            sample: int | None = None, seed: int = 0) -> gpd.GeoDataFrame:
    """Read a polygon file, return the parcel table, and write it into ``out_dir``."""
    if store not in STORES:
        raise SystemExit(f"--for must be one of {', '.join(STORES)}")
    src = Path(src)
    gdf = gpd.read_parquet(src) if src.suffix == ".parquet" else gpd.read_file(src)
    if gdf.crs is None:
        raise SystemExit(f"{src} has no coordinate system; set one before running this.")
    if sample is not None and sample < len(gdf):
        gdf = gdf.sample(sample, random_state=seed).reset_index(drop=True)

    if id_col:
        if id_col not in gdf.columns:
            raise SystemExit(f"--id-col {id_col!r} is not a column in {src}")
        ids = gdf[id_col].astype(str)
    elif "COD_PREDIO" in gdf.columns:
        ids = gdf["COD_PREDIO"].astype(str)
    else:
        # no id in the file: number the rows, so the output can still be joined back
        ids = pd.Series([f"P{i:06d}" for i in range(len(gdf))], index=gdf.index)
    if ids.duplicated().any():
        raise SystemExit(f"parcel ids are not unique ({int(ids.duplicated().sum())} repeats)")

    if store == "s2":
        if not imagery_date:
            raise SystemExit("--for s2 needs --imagery-date, e.g. --imagery-date 2025-03-01")
        try:
            imd = pd.Timestamp(imagery_date).date()
        except ValueError:
            raise SystemExit(f"--imagery-date {imagery_date!r} is not a date") from None
        if imd < dt.date(2019, 1, 1):
            raise SystemExit("--imagery-date before 2019 is outside the Sentinel-2 window "
                             "this project's S2 models were fitted on")
        years = pd.Series(_ag_year_end(imd), index=gdf.index)
    elif year_col:
        if year_col not in gdf.columns:
            raise SystemExit(f"--year-col {year_col!r} is not a column in {src}")
        years = gdf[year_col].astype(int)
    elif year is not None:
        years = pd.Series(int(year), index=gdf.index)
    else:
        raise SystemExit("give --year (one year for every parcel) or --year-col")

    # geometry: 2D, EPSG:4326 for Earth Engine; area from an equal-area projection
    geom = gdf.geometry.force_2d()
    area_ha = gpd.GeoSeries(geom, crs=gdf.crs).to_crs(6933).area / 10_000
    out = gpd.GeoDataFrame({
        "COD_PREDIO": ids.values,
        "label": "UNKNOWN",              # placeholder: these parcels are not labelled
        "label_id": 0,
        "label_reason": "prediction_only",
        "crop_set": "",
        "year": years.values,
        "area_ha": area_ha.values,
        "n_pixels_est": (area_ha / PIXEL_HA).values,
    }, geometry=geom.values, crs=gdf.crs).to_crs(4326)
    cen = out.geometry.representative_point()
    out["centroid_lon"], out["centroid_lat"] = cen.x, cen.y
    # carried through when the source has it: the S2 extraction's observation report groups
    # by department, and a missing column stops it after the pixels are already paid for
    out["dept"] = gdf["dept"].astype(str).values if "dept" in gdf.columns else ""
    out["n_valid_obs"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out["max_gap"] = pd.array([pd.NA] * len(out), dtype="Int64")
    if store == "s2":
        # `quality_ok` is the *Landsat* coverage flag; the S2 store has its own gates at
        # assemble time. `load_parcels` filters on == True, so leaving it NA here would score
        # zero parcels with no error — same reason labelling/train_prep sets it True.
        out["quality_ok"] = pd.array([True] * len(out), dtype="boolean")
        out["imagery_date"] = pd.Timestamp(imagery_date).date()
    else:
        # filled by `cc satellite extract --stage coverage`
        out["quality_ok"] = pd.array([pd.NA] * len(out), dtype="boolean")
    cols = ["COD_PREDIO", "label", "label_id", "label_reason", "crop_set", "year",
            "area_ha", "n_pixels_est", "centroid_lon", "centroid_lat", "dept",
            "n_valid_obs", "max_gap", "quality_ok"]
    out = out[cols + (["imagery_date"] if store == "s2" else []) + ["geometry"]]

    small = int((out["area_ha"] < AREA_MIN_HA).sum())
    big = int((out["area_ha"] > AREA_MAX_HA).sum())
    print(f"{len(out):,} parcels, {store} store, year {out.year.min()}-{out.year.max()}")
    if small or big:
        print(f"  {small:,} under {AREA_MIN_HA} ha and {big:,} over {AREA_MAX_HA} ha — "
              f"outside the range the models were fitted on; their predictions are "
              f"extrapolation")
    if save:
        d = Path(out_dir or "data/mine")
        (d / "features").mkdir(parents=True, exist_ok=True)
        out.to_parquet(d / "modeling_parcels.parquet", index=False)
        print(f"wrote {d / 'modeling_parcels.parquet'}")
        if store == "s2":
            (d / "labels_s2").mkdir(parents=True, exist_ok=True)
            out.to_parquet(d / "labels_s2" / "label_sample.parquet", index=False)
            print(f"wrote {d / 'labels_s2' / 'label_sample.parquet'} "
                  f"(what `cc labelling campaign extract` reads)")
            # `cc predict` loads data.FN_LGBM; the S2 assemble writes its own name. The link
            # is made now and resolves once `campaign assemble` has run.
            link = d / "features" / "features_lightgbm.parquet"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to("s2_features_lightgbm.parquet")
    return out


def attach_climate(out_dir: Path, arm: str = "temp") -> Path:
    """Add the climate columns a `--climate` model was trained with to a feature table.

    A model fitted with ``--climate temp`` scores on ``tmean_c``, which an imagery-only
    feature table lacks (``cc predict`` stops with "feature store is missing 1 model
    features"). Values come from the committed per-parcel normals, so this works only for
    parcels in the project's national table — for others, build the normals over them
    (`cc allperu climate normals`) or use a model trained without climate.
    """
    if arm not in CLIMATE_ARMS:
        raise SystemExit(f"--attach-climate must be one of {', '.join(CLIMATE_ARMS)}")
    if not CLIMATE_PARQUET.exists():
        raise SystemExit(f"missing {CLIMATE_PARQUET} — build it with "
                         f"`uv run cc -w national allperu climate normals`")
    d = Path(out_dir)
    fdir = d / "features"
    src = next((f for f in (fdir / "s2_features_lightgbm.parquet",
                            fdir / "features_lightgbm.parquet") if f.is_file()), None)
    if src is None:
        raise SystemExit(f"no feature table in {fdir} — run the extraction and assemble "
                         f"steps first")
    cols = CLIMATE_ARMS[arm]
    feats = pd.read_parquet(src)
    feats["COD_PREDIO"] = feats["COD_PREDIO"].astype(str)
    cl = pd.read_parquet(CLIMATE_PARQUET, columns=["COD_PREDIO", *cols])
    cl["COD_PREDIO"] = cl["COD_PREDIO"].astype(str)
    feats = feats.drop(columns=[c for c in cols if c in feats.columns])
    feats = feats.merge(cl, on="COD_PREDIO", how="left")

    miss = int(feats[cols].isna().any(axis=1).sum())
    if miss == len(feats):
        raise SystemExit(
            f"none of these {len(feats):,} parcels is in {CLIMATE_PARQUET.name}, so every "
            f"climate value would be empty. Use a model trained without --climate, or "
            f"build the normals over these parcels first.")
    if miss:
        print(f"WARNING: {miss:,} of {len(feats):,} parcels have no climate value and will "
              f"be scored with it missing")

    out = fdir / "features_lightgbm.parquet"
    if out.is_symlink():
        out.unlink()                     # replace the link with a real table
    feats.to_parquet(out, index=False)
    print(f"added {cols} to {out} ({len(feats):,} parcels)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("polygons", type=Path, nargs="?",
                    help="shapefile, GeoPackage, GeoJSON or parquet")
    ap.add_argument("--attach-climate", default=None, choices=sorted(CLIMATE_ARMS),
                    help="second pass, after the features exist: add the climate columns a "
                         "`--climate` model needs to <out>/features/")
    ap.add_argument("--for", dest="store", default="landsat", choices=STORES,
                    help="which feature store the model you will use was trained on")
    ap.add_argument("--year", type=int, default=None, help="landsat: season to predict")
    ap.add_argument("--year-col", default=None, help="landsat: a year column per parcel")
    ap.add_argument("--imagery-date", default=None,
                    help="s2: the date to read, e.g. 2025-03-01. The season around it "
                         "(Aug-Jul) is what gets summarised")
    ap.add_argument("--id-col", default=None, help="column naming each parcel")
    ap.add_argument("--sample", type=int, default=None, help="take N parcels at random")
    ap.add_argument("--seed", type=int, default=0, help="seed for --sample")
    ap.add_argument("--out", type=Path, default=Path("data/mine"),
                    help="workspace directory to write into (default data/mine)")
    a = ap.parse_args()
    if a.attach_climate:
        attach_climate(a.out, a.attach_climate)
        return
    if a.polygons is None:
        raise SystemExit("give a polygon file, or --attach-climate to add climate columns")
    prepare(a.polygons, year=a.year, year_col=a.year_col, id_col=a.id_col, out_dir=a.out,
            store=a.store, imagery_date=a.imagery_date, sample=a.sample, seed=a.seed)


if __name__ == "__main__":
    main()

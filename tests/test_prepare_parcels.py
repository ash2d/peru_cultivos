"""`tools/prepare_parcels.py` writes a parcel table the rest of the pipeline can read.

The table's *shape* is the contract: `cc satellite extract`, `cc satellite assemble` and
`cc predict` all read `modeling_parcels.parquet` and fail late and confusingly if a column
is missing — `label_id` in particular, which inference needs even though the parcels are
unlabelled. This pins the columns and the two conversions that are easy to get wrong.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["COD_PREDIO", "label", "label_id", "label_reason", "crop_set", "year",
            "area_ha", "n_pixels_est", "centroid_lon", "centroid_lat", "dept",
            "n_valid_obs", "max_gap", "quality_ok", "geometry"]


def _module():
    spec = importlib.util.spec_from_file_location(
        "prepare_parcels", ROOT / "tools" / "prepare_parcels.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prepare_parcels"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def polygons(tmp_path) -> Path:
    # ~1 ha squares near Piura, in the shapefiles' own UTM zone
    boxes = [Polygon([(x, 0), (x + 100, 0), (x + 100, 100), (x, 100)])
             for x in (500_000, 500_200)]
    f = tmp_path / "p.gpkg"
    gpd.GeoDataFrame({"id": ["a", "b"]}, geometry=boxes, crs=32717).to_file(f, driver="GPKG")
    return f


def test_it_writes_every_column_the_pipeline_reads(polygons, tmp_path):
    out = _module().prepare(polygons, year=2020, id_col="id", out_dir=tmp_path / "ws")
    assert list(out.columns) == REQUIRED
    written = gpd.read_parquet(tmp_path / "ws" / "modeling_parcels.parquet")
    assert list(written.columns) == REQUIRED
    assert written["label_id"].tolist() == [0, 0]          # placeholder, not a label
    assert written["quality_ok"].isna().all()              # filled by the coverage pass


def test_geometry_is_reprojected_and_area_is_hectares(polygons, tmp_path):
    out = _module().prepare(polygons, year=2020, id_col="id", out_dir=tmp_path / "ws")
    assert out.crs.to_epsg() == 4326                       # Earth Engine takes 4326 only
    assert out["area_ha"].between(0.9, 1.1).all()          # 100 m x 100 m = 1 ha
    assert out["COD_PREDIO"].tolist() == ["a", "b"]


def test_it_refuses_input_it_cannot_score(polygons, tmp_path):
    m = _module()
    with pytest.raises(SystemExit):                        # no year given
        m.prepare(polygons, id_col="id", out_dir=tmp_path / "ws", save=False)
    with pytest.raises(SystemExit):                        # id column that is not there
        m.prepare(polygons, year=2020, id_col="nope", out_dir=tmp_path / "ws", save=False)


def test_s2_mode_writes_what_the_campaign_extraction_reads(polygons, tmp_path):
    m = _module()
    ws = tmp_path / "ws"
    out = m.prepare(polygons, store="s2", imagery_date="2025-03-01", id_col="id", out_dir=ws)
    # the agricultural year Aug 2024 - Jul 2025 is named by its end year
    assert out["year"].tolist() == [2025, 2025]
    assert out["imagery_date"].iloc[0].isoformat() == "2025-03-01"
    # `load_parcels` filters on quality_ok == True, and the Landsat coverage gate never runs
    # on this store: NA here would score zero parcels and report an empty result
    assert (out["quality_ok"] == True).all()  # noqa: E712
    assert (ws / "labels_s2" / "label_sample.parquet").is_file()
    link = ws / "features" / "features_lightgbm.parquet"
    assert link.is_symlink() and str(link.readlink()) == "s2_features_lightgbm.parquet"


def test_s2_mode_needs_a_date_inside_the_sentinel_2_era(polygons, tmp_path):
    m = _module()
    with pytest.raises(SystemExit):
        m.prepare(polygons, store="s2", id_col="id", out_dir=tmp_path, save=False)
    with pytest.raises(SystemExit):
        m.prepare(polygons, store="s2", imagery_date="2005-03-01", id_col="id",
                  out_dir=tmp_path, save=False)


def test_attach_climate_refuses_a_parcel_set_it_cannot_cover(tmp_path):
    m = _module()
    if not m.CLIMATE_PARQUET.exists():
        pytest.skip("per-parcel climate normals not present in this clone")
    fdir = tmp_path / "features"
    fdir.mkdir(parents=True)
    import pandas as pd
    pd.DataFrame({"COD_PREDIO": ["not-a-parcel-1", "not-a-parcel-2"],
                  "NDVI_median": [0.4, 0.5]}).to_parquet(
        fdir / "s2_features_lightgbm.parquet", index=False)
    with pytest.raises(SystemExit):          # every value would be empty
        m.attach_climate(tmp_path, "temp")


def test_attach_climate_replaces_the_symlink_with_a_real_table(tmp_path):
    m = _module()
    if not m.CLIMATE_PARQUET.exists():
        pytest.skip("per-parcel climate normals not present in this clone")
    import pandas as pd
    known = pd.read_parquet(m.CLIMATE_PARQUET, columns=["COD_PREDIO"]).head(3)
    fdir = tmp_path / "features"
    fdir.mkdir(parents=True)
    pd.DataFrame({"COD_PREDIO": known["COD_PREDIO"].astype(str),
                  "NDVI_median": [0.4, 0.5, 0.6]}).to_parquet(
        fdir / "s2_features_lightgbm.parquet", index=False)
    (fdir / "features_lightgbm.parquet").symlink_to("s2_features_lightgbm.parquet")
    out = m.attach_climate(tmp_path, "temp")
    assert not out.is_symlink()
    got = pd.read_parquet(out)
    assert "tmean_c" in got.columns and got["tmean_c"].notna().all()

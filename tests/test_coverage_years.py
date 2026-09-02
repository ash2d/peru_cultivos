"""Regression: per-year coverage must not contaminate across years (plan §7.0.2).

``run_coverage`` writes content-addressed chunks into ``out.parent/"coverage_chunks"`` and
then combines them. It used to glob **every** ``cov_*.parquet`` there and dedupe on
``COD_PREDIO`` alone — so two panel years whose ``out=`` files share a parent directory
would silently return the *first* year's numbers for every parcel.

The plan's fix ("call it once per year with an explicit ``out=``") is **not sufficient**:
the chunk directory is derived from ``out.parent``. Caught for real in the panel timing
probe — 1995/1996/2005 all reported an identical 4.7 % gate pass, median 2 obs; the real
numbers are 4.7 %, 100 %, 100 %.

These tests exercise the combine logic with pre-seeded chunk files, so no GEE is needed.
"""

from __future__ import annotations

import hashlib

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from crop_classifier.features import landsat_gee as lg


@pytest.fixture
def parcels():
    df = gpd.GeoDataFrame(
        {"COD_PREDIO": ["A", "B", "C"], "year": [1998, 1998, 1998],
         "centroid_lon": [-80.5, -80.4, -80.3], "centroid_lat": [-5.2, -5.1, -5.0]},
        geometry=[Point(-80.5, -5.2), Point(-80.4, -5.1), Point(-80.3, -5.0)],
        crs=4326)
    return df


def seed_chunk(chunk_dir, year, parcels, n_valid_obs):
    """Write the chunk file run_coverage would have written for this parcel set/year."""
    chunk_dir.mkdir(parents=True, exist_ok=True)
    key = ",".join(sorted(parcels["COD_PREDIO"].astype(str)))
    cid = hashlib.md5(key.encode()).hexdigest()[:10]
    pd.DataFrame({"COD_PREDIO": parcels["COD_PREDIO"], "year": year,
                  "n_valid_obs": n_valid_obs, "max_gap": 0}).to_parquet(
        chunk_dir / f"cov_{year}_{cid}.parquet", index=False)


def combine(tmp_path, parcels, year, monkeypatch):
    """Run run_coverage's combine path only (every chunk already exists -> no GEE)."""
    monkeypatch.setattr(lg, "init_ee", lambda *a, **k: None)
    p = parcels.copy()
    p["year"] = year
    return lg.run_coverage(parcels=p, out=tmp_path / f"coverage_{year}.parquet",
                           years=[year], chunk_size=10)


def test_years_sharing_an_output_directory_do_not_contaminate(tmp_path, parcels,
                                                              monkeypatch):
    """The actual regression. Both years write into the same parent dir on purpose."""
    chunk_dir = tmp_path / "coverage_chunks"
    seed_chunk(chunk_dir, 1995, parcels, n_valid_obs=2)     # thin archive year
    seed_chunk(chunk_dir, 2005, parcels, n_valid_obs=9)     # dense year

    cov95 = combine(tmp_path, parcels, 1995, monkeypatch)
    cov05 = combine(tmp_path, parcels, 2005, monkeypatch)

    assert set(cov95["year"]) == {1995}
    assert set(cov05["year"]) == {2005}
    assert cov95["n_valid_obs"].tolist() == [2, 2, 2]
    assert cov05["n_valid_obs"].tolist() == [9, 9, 9]
    # the bug's signature: identical numbers for two very different years
    assert cov95["n_valid_obs"].median() != cov05["n_valid_obs"].median()


def test_combine_keeps_one_row_per_parcel_year(tmp_path, parcels, monkeypatch):
    chunk_dir = tmp_path / "coverage_chunks"
    seed_chunk(chunk_dir, 1998, parcels, n_valid_obs=6)
    cov = combine(tmp_path, parcels, 1998, monkeypatch)
    assert len(cov) == len(parcels)
    assert not cov.duplicated(["COD_PREDIO", "year"]).any()


def test_duplicate_chunks_for_one_year_are_still_deduped(tmp_path, parcels, monkeypatch):
    """Two chunk files covering the same parcels+year (e.g. after a chunk-size change)."""
    chunk_dir = tmp_path / "coverage_chunks"
    seed_chunk(chunk_dir, 1998, parcels, n_valid_obs=6)
    pd.DataFrame({"COD_PREDIO": parcels["COD_PREDIO"], "year": 1998,
                  "n_valid_obs": 6, "max_gap": 0}).to_parquet(
        chunk_dir / "cov_1998_otherhash.parquet", index=False)
    cov = combine(tmp_path, parcels, 1998, monkeypatch)
    assert len(cov) == len(parcels)


def test_without_a_years_filter_all_years_are_returned(tmp_path, parcels, monkeypatch):
    """The training-store path: one row per parcel-year, every year kept."""
    chunk_dir = tmp_path / "coverage_chunks"
    seed_chunk(chunk_dir, 1998, parcels, n_valid_obs=6)
    seed_chunk(chunk_dir, 1999, parcels, n_valid_obs=8)
    monkeypatch.setattr(lg, "init_ee", lambda *a, **k: None)
    p = parcels.copy()
    cov = lg.run_coverage(parcels=p, out=tmp_path / "coverage.parquet", chunk_size=10)
    assert set(cov["year"]) == {1998, 1999}
    assert len(cov) == 2 * len(parcels)


def test_missions_include_l9_from_2021():
    """D7: without L9, 2022-24 would run on L8 + SLC-off L7 alone."""
    assert "L9" not in lg.missions_for_year(2020)
    assert "L9" in lg.missions_for_year(2021)
    assert lg.missions_for_year(2023)["L9"][0] == "LANDSAT/LC09/C02/T1_L2"
    assert set(lg.missions_for_year(1998)) == {"L5"}
    assert set(lg.missions_for_year(2005)) == {"L5", "L7"}


# --- Completeness is measured in PARCELS, never in files (RESULTS.md §7.1) ---
def test_verify_years_catches_a_truncated_year_that_looks_well_formed(tmp_path,
                                                                     monkeypatch):
    """A per-year pixel store is rewritten from *every* chunk on disk when ``run_pixels``
    combines, so a worker finishing early truncates a year another is still extracting.
    `pixels_2023.parquet` once sat at 62 % of its parcels, well-formed and wrong. "The
    process ended" and "the file exists" have both been wrong here; only the parcel count
    has not.
    """
    from crop_classifier.perennial import panel as P

    monkeypatch.setenv("CC_PROC", str(tmp_path / "proc"))
    monkeypatch.setenv("CC_FEAT", str(tmp_path / "feat"))
    (tmp_path / "proc").mkdir(parents=True, exist_ok=True)

    for year, n_gate, n_store in ((2014, 100, 100), (2015, 100, 100), (2016, 100, 100),
                                  (2019, 100, 62)):
        pd.DataFrame({"COD_PREDIO": [f"p{i}" for i in range(n_gate)],
                      "quality_ok": [True] * n_gate}).to_parquet(
            tmp_path / "proc" / f"panel_coverage_{year}.parquet", index=False)
        pd.DataFrame({"COD_PREDIO": [f"p{i}" for i in range(n_store)],
                      "doy": range(n_store)}).to_parquet(
            P.panel_feat() / f"pixels_{year}.parquet", index=False)

    v = P.verify_years([2014, 2015, 2016, 2019]).set_index("year")
    assert bool(v.loc[2014, "complete"]) is True
    assert bool(v.loc[2019, "complete"]) is False        # 0.62 — the real incident
    assert v.loc[2019, "parcels_in_store"] == 62


def test_verify_years_tolerates_a_uniform_sub_pixel_floor(tmp_path, monkeypatch):
    """A parcel smaller than a Landsat pixel passes the coverage gate and yields no rows.

    Measured on the DiD panel: the *same* ~85 of 14,625 parcels are absent every year (median
    0.11 ha, 1.23 estimated pixels) — a structural floor, not truncation, and the tell is
    that it is identical every year. `verify` must pass it while still failing a year-specific
    deficit; a guessed tolerance above the floor once false-alarmed a complete extraction.
    """
    from crop_classifier.perennial import panel as P

    monkeypatch.setenv("CC_PROC", str(tmp_path / "proc"))
    monkeypatch.setenv("CC_FEAT", str(tmp_path / "feat"))
    (tmp_path / "proc").mkdir(parents=True, exist_ok=True)

    for year in (1999, 2000, 2001, 2014):
        pd.DataFrame({"COD_PREDIO": [f"p{i}" for i in range(1000)],
                      "quality_ok": [True] * 1000}).to_parquet(
            tmp_path / "proc" / f"panel_coverage_{year}.parquet", index=False)
        # the same 6 sub-pixel parcels are missing every year (0.6 %, as measured)
        pd.DataFrame({"COD_PREDIO": [f"p{i}" for i in range(994)],
                      "doy": range(994)}).to_parquet(
            P.panel_feat() / f"pixels_{year}.parquet", index=False)

    v = P.verify_years([1999, 2000, 2001, 2014])
    assert bool(v["complete"].all())
    assert v["frac"].min() == pytest.approx(0.994, abs=1e-3)


def test_rebuild_year_stores_uses_every_chunk_and_dedupes(tmp_path, monkeypatch):
    """The repair for the above: rebuild from the chunk set once all workers have exited."""
    from crop_classifier.perennial import panel as P

    monkeypatch.setenv("CC_FEAT", str(tmp_path / "feat"))
    chunk_dir = P.panel_feat() / "pixels_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    a = pd.DataFrame({"COD_PREDIO": ["p1", "p2"], "doy": [10, 20], "lon": [0.0, 0.0],
                      "lat": [0.0, 0.0], "mission": [5, 5]})
    a.to_parquet(chunk_dir / "px_2014_aaaa.parquet", index=False)
    a.to_parquet(chunk_dir / "px_2014_bbbb.parquet", index=False)   # a duplicate chunk
    pd.DataFrame({"COD_PREDIO": ["p3"], "doy": [30], "lon": [0.0], "lat": [0.0],
                  "mission": [7]}).to_parquet(chunk_dir / "px_2014_cccc.parquet",
                                              index=False)

    out = P.rebuild_year_stores([2014]).set_index("year")
    assert out.loc[2014, "n_chunks"] == 3
    assert out.loc[2014, "n_parcels"] == 3          # p1, p2, p3
    assert out.loc[2014, "n_pixel_obs"] == 3        # the duplicated rows collapse

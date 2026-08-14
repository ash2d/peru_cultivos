"""Unit tests for the all-Peru extension (docs/all_peru/).

Three things here are load-bearing and fail *silently* if they regress, which is why each
has a test rather than a comment:

* ``canon_key`` — zero-padded bridge keys join to nothing, and "nothing" looks like a
  department with no linkable data rather than a bug;
* ``allocate`` — a bad allocation makes an "all-Peru" model that is really two departments;
* ``sample`` — sampling scattered parcels instead of whole regions would quietly destroy the
  block structure the spatial split depends on.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import Point

from crop_classifier.allperu.build_labels import canon_key, clean_geometry
from crop_classifier.allperu.sample import allocate, region_grid, sample
from crop_classifier.allperu.sources import _norm
from crop_classifier.data import FN_LGBM
from crop_classifier.features.landsat_gee import _bbox_area, _chunk_todo


class TestCanonKey:
    def test_strips_leading_zeros(self):
        """The Ancash case: bridge pads to 9, BD SSET does not."""
        out = canon_key(pd.Series(["030406693", "30406693", " 030406693 "]))
        assert out.tolist() == ["30406693"] * 3

    def test_piura_form_is_untouched(self):
        """Piura keys carry no leading zeros, so the fix must be a no-op there."""
        keys = ["270014306", "123456789"]
        assert canon_key(pd.Series(keys)).tolist() == keys

    def test_all_zero_key_survives_as_zero(self):
        assert canon_key(pd.Series(["000"])).tolist() == ["0"]

    def test_nulls_become_nan(self):
        out = canon_key(pd.Series(["nan", "", "None", "<NA>"]))
        assert out.isna().all()

    def test_joins_only_after_canonicalisation(self):
        sset = pd.DataFrame({"CodigoSSET": canon_key(pd.Series(["30406693", "30406694"]))})
        bridge = pd.DataFrame({"CodigoSSET": pd.Series(["030406693", "030406694"]),
                               "COD_PREDIO": ["a", "b"]})
        assert len(sset.merge(bridge, on="CodigoSSET")) == 0          # raw: no match
        bridge["CodigoSSET"] = canon_key(bridge["CodigoSSET"])
        assert len(sset.merge(bridge, on="CodigoSSET")) == 2          # canon: both match


class TestAllocate:
    def test_sqrt_allocation_favours_small_departments(self):
        """A 100x bigger department must not get a 100x bigger quota."""
        sizes = pd.Series({"BIG": 500_000, "SMALL": 5_000})
        q = allocate(sizes, total=20_000, floor=0)
        assert q.sum() == pytest.approx(20_000, abs=2)
        # proportional would give BIG 99 % — sqrt should be far kinder to SMALL
        assert q["BIG"] / q["SMALL"] < 20        # proportional ratio is 100

    def test_floor_protects_tiny_departments(self):
        sizes = pd.Series({"BIG": 500_000, "TINY": 300})
        q = allocate(sizes, total=10_000, floor=800)
        assert q["TINY"] == 300                  # floor capped by what exists
        assert q["BIG"] <= 500_000

    def test_never_exceeds_availability(self):
        sizes = pd.Series({"A": 100, "B": 200, "C": 50})
        q = allocate(sizes, total=10_000, floor=0)
        assert (q <= sizes).all()
        assert q.sum() == sizes.sum()            # cannot invent parcels

    def test_hits_target_when_supply_is_ample(self):
        sizes = pd.Series({c: 100_000 for c in "ABCDE"})
        assert allocate(sizes, total=50_000, floor=0).sum() == pytest.approx(50_000, abs=5)


def _fake_parcels(n_dept=3, n_region=6, per_region=40, seed=0):
    """Parcels laid out in well-separated 5 km clusters so region ids are predictable."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dept):
        for r in range(n_region):
            # clusters 20 km apart in x, departments 200 km apart in y
            lon = -76.0 + r * 0.2
            lat = -10.0 - d * 2.0
            for i in range(per_region):
                rows.append({
                    "COD_PREDIO": f"d{d}_r{r}_{i}",
                    "dept": f"DEPT{d}",
                    "label": ["ANNUAL", "PERENNIAL", "PASTURE_FALLOW"][i % 3],
                    "year": 1998 + (i % 5),
                    "geometry": Point(lon + rng.normal(0, 0.002),
                                      lat + rng.normal(0, 0.002)),
                })
    return gpd.GeoDataFrame(rows, crs=4326)


class TestSample:
    def test_region_grid_separates_distant_clusters(self):
        """The invariant the sampler needs: parcels ~22 km apart never share a 5 km cell,
        so 'sample a region' really does mean 'sample one place'."""
        g = _fake_parcels(n_dept=1, n_region=3, per_region=10)
        reg = region_grid(g, region_km=5.0)
        cluster = g.COD_PREDIO.str.extract(r"_r(\d+)_")[0]
        # no cell may contain parcels from two different clusters
        assert reg.groupby(reg).apply(lambda s: cluster[s.index].nunique()).max() == 1

    def test_caps_parcels_per_region(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CC_PROC", str(tmp_path))
        src = tmp_path / "src"
        src.mkdir()
        _fake_parcels().to_parquet(src / "modeling_parcels.parquet", index=False)
        out = sample(src, target_n=180, max_per_region=10, floor=0, save=False)
        reg = region_grid(out, 5.0)
        assert reg.value_counts().max() <= 10   # the cap is what buys spatial spread

    def test_covers_every_department(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CC_PROC", str(tmp_path))
        src = tmp_path / "src"
        src.mkdir()
        _fake_parcels().to_parquet(src / "modeling_parcels.parquet", index=False)
        out = sample(src, target_n=180, max_per_region=10, floor=0, save=False)
        assert out.dept.nunique() == 3

    def test_population_weights_expand_to_population(self, tmp_path, monkeypatch):
        """weight = stratum population / stratum sample, so weights recover the total.

        Named `population_weight`, not `sample_weight`: `perennial/panel.py` computes its own
        `sample_weight` for a different stratification, and the two must compose rather than
        collide (they did — `build_panel` raised KeyError on the suffixed columns)."""
        monkeypatch.setenv("CC_PROC", str(tmp_path))
        src = tmp_path / "src"
        src.mkdir()
        full = _fake_parcels()
        full.to_parquet(src / "modeling_parcels.parquet", index=False)
        out = sample(src, target_n=180, max_per_region=10, floor=0, save=False)
        assert out.population_weight.sum() == pytest.approx(len(full), rel=0.02)
        assert "sample_weight" not in out.columns

    def test_rejects_source_without_dept(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CC_PROC", str(tmp_path))
        src = tmp_path / "src"
        src.mkdir()
        _fake_parcels().drop(columns="dept").to_parquet(
            src / "modeling_parcels.parquet", index=False)
        with pytest.raises(KeyError):
            sample(src, target_n=10, save=False)


class TestCleanGeometry:
    """La Libertad's cadastre is 3D and Earth Engine rejects 3D GeoJSON outright —
    `EEException: Invalid GeoJSON geometry` — 5,411 parcels' worth, three chunks into a
    multi-hour extraction. Shapely calls those polygons perfectly valid, so nothing
    upstream catches it."""

    def test_strips_z(self):
        from shapely.geometry import Polygon
        g = gpd.GeoSeries([Polygon([(0, 0, 5), (1, 0, 5), (1, 1, 5), (0, 0, 5)])], crs=4326)
        assert shapely.has_z(g.values).all()
        out = clean_geometry(g)
        assert not shapely.has_z(out.values).any()
        assert out.iloc[0].equals(shapely.force_2d(g.iloc[0]))

    def test_repairs_self_intersection(self):
        from shapely.geometry import Polygon
        bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
        g = gpd.GeoSeries([bowtie], crs=4326)
        assert not g.is_valid.all()
        assert clean_geometry(g).is_valid.all()

    def test_leaves_clean_2d_polygons_alone(self):
        from shapely.geometry import Polygon
        p = Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])
        g = gpd.GeoSeries([p], crs=4326)
        assert clean_geometry(g).iloc[0].equals(p)


class TestChunkTodo:
    """Chunk extent is what a `filterBounds` runs against, so it drives GEE cost directly.
    A year-only grouping spans up to 110 deg^2 nationally."""

    def _parcels(self, n_dept=4, per_dept=300):
        rng = np.random.default_rng(0)
        rows = []
        for d in range(n_dept):
            for i in range(per_dept):
                rows.append({"COD_PREDIO": f"d{d}_{i}", "dept": f"D{d}",
                             "year": 1999 + (i % 2),
                             "centroid_lon": -76.0 + d * 3 + rng.normal(0, 0.05),
                             "centroid_lat": -10.0 + rng.normal(0, 0.05)})
        g = gpd.GeoDataFrame(rows, geometry=[
            Point(r["centroid_lon"], r["centroid_lat"]) for r in rows], crs=4326)
        return g

    def test_no_parcel_lost_or_duplicated(self):
        g = self._parcels()
        todo = _chunk_todo(g, 100)
        cods = pd.concat([c for _, _, c in todo]).COD_PREDIO
        assert len(cods) == len(g) and cods.is_unique

    def test_chunk_ids_are_unique(self):
        todo = _chunk_todo(self._parcels(), 100)
        ids = [i for _, i, _ in todo]
        assert len(set(ids)) == len(ids)

    def test_respects_chunk_size(self):
        todo = _chunk_todo(self._parcels(), 100)
        assert max(len(c) for _, _, c in todo) <= 100

    def test_packs_distant_departments_apart(self):
        """Departments 3 deg apart must not be merged into one chunk under a 1 deg^2 cap."""
        g = self._parcels(n_dept=4, per_dept=20)      # 10 parcels per (year, dept)
        todo = _chunk_todo(g, chunk_size=400, max_bbox=1.0)
        for _, _, c in todo:
            assert _bbox_area(c) <= 1.0 or c.dept.nunique() == 1

    def test_packs_small_neighbours_together(self):
        """...but small adjacent groups SHOULD share a request, or overhead dominates."""
        rng = np.random.default_rng(0)
        rows = [{"COD_PREDIO": f"d{d}_{i}", "dept": f"D{d}", "year": 1999,
                 "centroid_lon": -76.0 + d * 0.1 + rng.normal(0, 0.01),
                 "centroid_lat": -10.0 + rng.normal(0, 0.01)}
                for d in range(5) for i in range(10)]
        g = gpd.GeoDataFrame(rows, geometry=[
            Point(r["centroid_lon"], r["centroid_lat"]) for r in rows], crs=4326)
        todo = _chunk_todo(g, chunk_size=400)
        assert len(todo) == 1          # 5 tiny neighbouring depts -> one request


class TestLodo:
    """Leave-one-department-out is the headline spatial-generalisation claim, so the two
    ways it could quietly cheat are pinned here: the held-out department leaking into
    training, and the held-out department being used as the early-stopping validation set.

    Exercised with the `rules` model, not LightGBM: importing lightgbm into a process that
    has already imported torch segfaults on macOS (libomp clash — see data.py). `rules` goes
    through the identical make_dataset/class_weights/fit/predict_proba path.
    """

    def _workspace(self, tmp_path, monkeypatch):
        rng = np.random.default_rng(0)
        rows = []
        for d, lon0 in enumerate([-76.0, -73.0, -70.0]):
            for i in range(150):
                lab = i % 3
                rows.append({
                    "COD_PREDIO": f"D{d}_{i}", "dept": f"D{d}",
                    "label": ["ANNUAL", "PASTURE_FALLOW", "PERENNIAL"][lab],
                    "label_id": lab, "year": 1999 + (i % 3),
                    "region_id": f"r{d}_{i // 25}", "quality_ok": True,
                    "centroid_lon": lon0 + rng.normal(0, 0.05),
                    "centroid_lat": -10 + rng.normal(0, 0.05),
                })
        g = gpd.GeoDataFrame(rows, geometry=[
            Point(r["centroid_lon"], r["centroid_lat"]) for r in rows], crs=4326)

        ws, fs = tmp_path / "ws", tmp_path / "feat"
        ws.mkdir()
        fs.mkdir()
        g.to_parquet(ws / "modeling_parcels.parquet", index=False)
        (ws / "label_map.json").write_text(
            '{"ANNUAL": 0, "PASTURE_FALLOW": 1, "PERENNIAL": 2}')
        # a feature that genuinely carries the label, so a fit is meaningful
        pd.DataFrame({"COD_PREDIO": g.COD_PREDIO,
                      "NDVI_p25": g.label_id * 0.3 + rng.normal(0, 0.02, len(g)),
                      "NDVI_amp": rng.normal(0, 0.05, len(g)),
                      "NDVI_max": g.label_id * 0.3 + rng.normal(0, 0.02, len(g)),
                      "frac_l7": rng.random(len(g))}
                     ).to_parquet(fs / FN_LGBM, index=False)
        monkeypatch.setenv("CC_PROC", str(ws))
        monkeypatch.setenv("CC_FEAT", str(fs))
        monkeypatch.setenv("CC_RUNS", str(tmp_path / "runs"))
        return g

    def test_scores_every_department_on_unseen_data(self, tmp_path, monkeypatch):
        from crop_classifier.allperu.lodo import run
        self._workspace(tmp_path, monkeypatch)
        res = run(model_name="rules", min_parcels=10, save=False)
        assert set(res.dept) == {"D0", "D1", "D2"}
        assert (res.n_test == 150).all()
        # the `rules` control is weak by design; only assert it produced real scores
        assert res.macro_f1.notna().all() and (res.macro_f1 > 0).all()

    def test_held_out_department_never_appears_in_training(self, tmp_path, monkeypatch):
        """The leakage that would invalidate the whole result."""
        import crop_classifier.allperu.lodo as L
        g = self._workspace(tmp_path, monkeypatch)
        seen = {}
        orig = L.make_dataset

        def spy(kind, parcels, idx, **kw):
            seen.setdefault("calls", []).append(set(parcels.loc[idx, "dept"]))
            return orig(kind, parcels, idx, **kw)

        monkeypatch.setattr(L, "make_dataset", spy)
        L.run(model_name="rules", min_parcels=10, save=False)
        # calls come in (train, inner-val, test) triples, one triple per department
        for i in range(0, len(seen["calls"]), 3):
            train, val, test = seen["calls"][i:i + 3]
            assert len(test) == 1
            held = next(iter(test))
            assert held not in train, f"{held} leaked into its own training set"
            assert held not in val, f"{held} used as its own early-stopping val set"
        assert len(seen["calls"]) == 3 * g.dept.nunique()

    def test_drop_features_is_honoured(self, tmp_path, monkeypatch):
        from crop_classifier.allperu.lodo import run
        self._workspace(tmp_path, monkeypatch)
        res = run(model_name="rules", drop_features="meta", min_parcels=10, save=False)
        assert len(res) == 3          # frac_l7 withheld; the run still completes

    def test_tag_keeps_two_architectures_from_overwriting_each_other(
            self, tmp_path, monkeypatch):
        """Untagged LODO wrote fixed filenames, so running a second architecture silently
        replaced the first one's metrics *and* its per-department fits. The comparison
        between architectures is the reason a second run exists, so it must survive."""
        import os

        from crop_classifier.allperu.lodo import run
        self._workspace(tmp_path, monkeypatch)
        ws, rd = Path(os.environ["CC_PROC"]), Path(os.environ["CC_RUNS"])

        run(model_name="rules", min_parcels=10, save=True)
        run(model_name="rules", min_parcels=10, save=True, tag="ltae")

        for stem, ext in [("lodo_metrics", "csv"), ("lodo_predictions", "parquet"),
                          ("lodo_summary", "json")]:
            assert (ws / f"{stem}.{ext}").exists(), f"untagged {stem} was clobbered"
            assert (ws / f"{stem}_ltae.{ext}").exists(), f"tagged {stem} not written"
        assert (rd / "lodo" / "D0").is_dir()          # untagged fits still in place
        assert (rd / "lodo" / "ltae" / "D0").is_dir()  # tagged fits nested beside them

    def test_rejects_a_workspace_without_dept(self, tmp_path, monkeypatch):
        from crop_classifier.allperu.lodo import run
        g = self._workspace(tmp_path, monkeypatch)
        import os
        ws = Path(os.environ["CC_PROC"])
        g.drop(columns="dept").to_parquet(ws / "modeling_parcels.parquet", index=False)
        with pytest.raises(KeyError):
            run(model_name="rules", min_parcels=10, save=False)


class TestSources:
    def test_norm_reconciles_the_three_spellings(self):
        assert _norm("LA_LIBERTAD") == _norm("LA LIBERTAD") == _norm("La Libertad")
        assert _norm("  huánuco ") == "HUANUCO"

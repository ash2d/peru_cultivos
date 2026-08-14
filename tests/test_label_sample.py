"""Unit tests for the S2 labelling draw (docs/all_peru/s2_labelling_plan.md §12).

**There is one round and no second pass**, so a design bug here is not correctable
downstream — it is the campaign. Four things are therefore pinned:

* the allocation hits its floor and its exact total (a 50-parcel overshoot is 5 % of the
  budget);
* the 2-per-region cap holds *globally*, not per cell — otherwise two classes each spend
  the same region's budget and the sample collapses back onto neighbours;
* the crop cap holds wherever the population allows it;
* the weights reconstruct the eligible population exactly, which is the only thing that
  makes any reported share a population share rather than a sample mean.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from crop_classifier.allperu.label_sample import (
    CLASS_RATIO,
    _draw_cell,
    allocate_classes,
    allocate_departments,
    allocation_table,
    draw,
    eligible_population,
    sensitivity_sweep,
)


def _universe(n_per_cell=60, depts=("A", "B", "C"), seed=0) -> gpd.GeoDataFrame:
    """A synthetic eligible universe: 3 departments x 3 classes, spread over regions."""
    rng = np.random.default_rng(seed)
    rows = []
    for di, dept in enumerate(depts):
        for ci, cls in enumerate(CLASS_RATIO):
            for k in range(n_per_cell):
                rows.append({
                    "COD_PREDIO": f"{dept}{ci}{k:04d}", "dept": dept, "label": cls,
                    "area_ha": 1.0 + rng.random(),
                    "crop_set": ["MANGO", "ARROZ", "MAIZ"][k % 3],
                    # many regions so the cap is not the binding constraint by accident
                    "region_id": f"r{di}_{ci * n_per_cell + k // 2}",
                    "geometry": Point(-79 + di + 0.001 * k, -5 - ci - 0.001 * k),
                })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)


class TestAllocation:
    def test_floor_is_respected_and_total_is_exact(self):
        sizes = pd.Series({"BIG": 100_000, "MID": 10_000, "TINY": 300})
        q = allocate_departments(sizes, total=1_000, floor=55)
        assert q.min() >= 55
        assert int(q.sum()) == 1_000

    def test_floor_is_capped_by_availability(self):
        """A department with fewer parcels than the floor cannot be over-drawn."""
        sizes = pd.Series({"BIG": 100_000, "TINY": 12})
        q = allocate_departments(sizes, total=200, floor=55)
        assert q["TINY"] == 12

    def test_larger_departments_get_more_but_not_proportionally(self):
        sizes = pd.Series({"BIG": 160_000, "SMALL": 10_000})
        q = allocate_departments(sizes, total=1_000, floor=55)
        assert q["BIG"] > q["SMALL"]
        # sqrt allocation: 4x the parcels buys ~2x the sample, nowhere near 4x
        assert q["BIG"] / q["SMALL"] < 4.0

    def test_class_split_is_2_1_1_and_sums(self):
        assert allocate_classes(55) == {"PERENNIAL": 28, "ANNUAL": 14,
                                        "PASTURE_FALLOW": 13}
        assert allocate_classes(110) == {"PERENNIAL": 55, "ANNUAL": 28,
                                         "PASTURE_FALLOW": 27}
        for n in range(1, 200):
            assert sum(allocate_classes(n).values()) == n

    def test_cell_shortfall_spills_within_the_department(self):
        """A department short of perennials keeps its department total."""
        sizes = pd.DataFrame({"PERENNIAL": [2], "ANNUAL": [500],
                              "PASTURE_FALLOW": [500]}, index=["A"])
        t = allocation_table(sizes, total=100, floor=100)
        assert t["n_target"].sum() == 100
        assert t.loc[t.declared_class == "PERENNIAL", "n_target"].iloc[0] <= 2


class TestSensitivitySweep:
    def test_perennial_survives_the_whole_plausible_range(self):
        """The plan's claim: 43-54 % PERENNIAL, no class below ~20 %, across 5-30 %."""
        s = sensitivity_sweep()
        assert s["PERENNIAL"].between(0.40, 0.58).all()
        assert s[["PERENNIAL", "ANNUAL", "OTHER"]].min().min() >= 0.18

    def test_shares_sum_to_one(self):
        s = sensitivity_sweep()
        assert np.allclose(s[["PERENNIAL", "ANNUAL", "OTHER"]].sum(axis=1), 1.0)


class TestDraw:
    @pytest.fixture(scope="class")
    def drawn(self):
        u = _universe()
        return draw(u, total=90, floor=30, pilot_n=12, overlap_n=6, save=False)

    def test_region_cap_holds_globally(self, drawn):
        assert drawn.groupby("region_id").size().max() <= 2

    def test_pilot_and_main_are_disjoint(self, drawn):
        pilot = set(drawn.loc[drawn.batch == "pilot", "COD_PREDIO"])
        main = set(drawn.loc[drawn.batch == "main", "COD_PREDIO"])
        assert not (pilot & main)

    def test_overlap_is_drawn_from_the_main_batch_only(self, drawn):
        assert (drawn.loc[drawn.overlap, "batch"] == "main").all()
        assert int(drawn.overlap.sum()) == 6

    def test_item_ids_are_unique(self, drawn):
        assert drawn["item_id"].is_unique

    def test_declared_class_is_renamed_not_leaked_as_label(self, drawn):
        """`label` must not survive into the sample: the HTML builder allow-lists on name."""
        assert "declared_class" in drawn.columns
        assert "label" not in drawn.columns

    def test_crop_cap_holds_where_the_population_allows(self):
        """A cell of 20 from 3 equally-common crops: no crop may exceed 8 (ceil(0.4*20))."""
        u = _universe(n_per_cell=90)
        cell = u[(u.dept == "A") & (u.label == "PERENNIAL")].copy()
        cell["_crop"] = cell["crop_set"]
        got = _draw_cell(cell, 20, np.random.default_rng(0), {}, 2, 0.40)
        assert len(got) == 20
        assert got["_crop"].value_counts().max() <= 8

    def test_crop_cap_relaxes_rather_than_underfilling(self):
        """A cell where every parcel is mango still has to deliver its quota."""
        u = _universe(n_per_cell=90)
        cell = u[(u.dept == "A") & (u.label == "PERENNIAL")].copy()
        cell["_crop"] = "MANGO"
        got = _draw_cell(cell, 20, np.random.default_rng(0), {}, 2, 0.40)
        assert len(got) == 20

    def test_weights_reconstruct_the_eligible_population(self):
        u = _universe()
        pop = (u.groupby(["dept", "label"]).size()
               .rename("N_eligible").reset_index()
               .rename(columns={"label": "declared_class"}))
        s = draw(u, total=90, floor=30, pilot_n=0, overlap_n=0,
                 pop_eligible=pop, save=False)
        got = s.groupby("declared_class")["weight"].sum().round(6)
        want = pop.groupby("declared_class")["N_eligible"].sum()
        assert np.allclose(got.reindex(want.index).values, want.values)

    def test_pilot_rows_carry_no_weight(self, drawn):
        assert drawn.loc[drawn.batch == "pilot", "weight"].isna().all()


class TestEligiblePopulation:
    def test_scales_the_universe_by_the_probed_rate(self):
        u = pd.DataFrame({"dept": ["A"] * 1000, "label": ["ANNUAL"] * 1000})
        probed = pd.DataFrame({"dept": ["A"] * 10, "declared_class": ["ANNUAL"] * 10})
        elig = pd.Series([True] * 6 + [False] * 4)
        out = eligible_population(u, probed, elig)
        assert out["elig_rate"].iloc[0] == pytest.approx(0.6)
        assert out["N_eligible"].iloc[0] == 600


class TestEsriProbeSelection:
    """Which of several overlapping Esri footprints supplies `imagery_date`.

    The layer loop already fixes resolution (it stops at the finest layer that answers), so
    the only remaining choice is between dated footprints overlapping the same point. For a
    2019+ endpoint campaign that must be the **most recent** — and the code once did the
    opposite of the comment sitting next to it.
    """

    def _fake(self, monkeypatch, dates_by_layer):
        import datetime as dt

        from crop_classifier.allperu import esri_dates as E

        class _R:
            def __init__(self, ds):
                self._ds = ds

            def json(self):
                return {"features": [
                    {"attributes": {"SRC_DATE2": int(
                        dt.datetime.fromisoformat(d).timestamp() * 1000),
                        "SRC_RES": 1.2, "SRC_DESC": d}} for d in self._ds]}

        def _get(url, **kw):
            layer = int(url.rstrip("/query").rsplit("/", 1)[1])
            return _R(dates_by_layer.get(layer, []))

        monkeypatch.setattr(E.requests, "get", _get)
        return E

    def test_takes_the_most_recent_overlapping_footprint(self, monkeypatch):
        E = self._fake(monkeypatch, {11: ["2019-03-01", "2024-07-15", "2021-11-02"]})
        out = E.probe(("P1", -79.0, -5.0, "PIURA"))
        assert out["date"] == "2024-07-15" and out["year"] == 2024

    def test_finest_resolution_still_wins_over_a_newer_coarse_one(self, monkeypatch):
        """Resolution is decided by the layer order, never by the date."""
        E = self._fake(monkeypatch, {10: ["2019-01-01"], 11: ["2025-01-01"]})
        out = E.probe(("P1", -79.0, -5.0, "PIURA"))
        assert out["res"] == "60cm" and out["year"] == 2019

    def test_no_metadata_anywhere_returns_nulls(self, monkeypatch):
        E = self._fake(monkeypatch, {})
        out = E.probe(("P1", -79.0, -5.0, "PIURA"))
        assert out["res"] is None and out["year"] is None

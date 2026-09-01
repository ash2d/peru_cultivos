"""The demo workspace runs on a clone with no data and no Earth Engine account.

This is the promise `README.md` makes in its quickstart, and it is the one most likely to rot
silently: the demo is not on anyone's daily path, so the first person to find it broken is a
new collaborator on their first hour with the repository.

The training arm runs in a **subprocess**. LightGBM and torch each bundle their own libomp and
co-loading them on macOS segfaults the interpreter (exit 139), which in a test session takes
the whole run down rather than failing one test.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"


def test_the_demo_workspace_is_committed():
    """`data/` is gitignored; the demo is the deliberate exception. If this fails, a fresh
    clone has nothing to run."""
    for f in ("modeling_parcels.parquet", "label_map.json", "README.md",
              "features/features_lightgbm.parquet"):
        assert (DEMO / f).exists(), f"data/demo/{f} missing"
    out = subprocess.run(["git", "ls-files", "data/demo"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    assert any(f.endswith("modeling_parcels.parquet") for f in out), (
        "data/demo is not tracked by git — check the !data/demo/ rules in .gitignore")


def test_it_is_stamped_synthetic_on_the_table_itself():
    """A README nobody opens is not a safeguard. The stamp rides on the data, so anything
    derived from it carries the mark."""
    g = gpd.read_parquet(DEMO / "modeling_parcels.parquet")
    assert g["synthetic"].all()
    assert set(g["dept"]) == {"DEMOLANDIA", "EJEMPLIA"}      # not Peruvian department names


def test_it_has_the_columns_the_real_loader_requires():
    g = gpd.read_parquet(DEMO / "modeling_parcels.parquet")
    required = {"COD_PREDIO", "label", "label_id", "split", "fold", "year", "quality_ok",
                "buffer_excl_test", "region_id", "block_id", "geometry", "centroid_lat"}
    assert required <= set(g.columns)
    assert {f"buffer_excl_fold{k}" for k in range(5)} <= set(g.columns)


def test_no_region_is_split_across_train_and_test():
    """The discipline the demo is supposed to teach: whole regions move together. Splitting
    parcels at random puts a parcel's neighbours in its own training set, which is the
    cheapest way there is to manufacture an inflated CV score."""
    g = gpd.read_parquet(DEMO / "modeling_parcels.parquet")
    per_region = g.groupby("region_id")["split"].nunique()
    assert (per_region == 1).all(), "a region appears in more than one split"
    tv = g[g["split"] == "trainval"]
    assert (tv.groupby("region_id")["fold"].nunique() == 1).all()


def test_the_feature_store_carries_the_real_column_groups():
    """`--drop-features meta,location` must have something to drop, or the demo teaches a
    pipeline that differs from the real one in exactly the place that matters most."""
    from crop_classifier.data import META_FEATURES, ORDER_FEATURES
    f = pd.read_parquet(DEMO / "features" / "features_lightgbm.parquet")
    g = gpd.read_parquet(DEMO / "modeling_parcels.parquet")
    # META_FEATURES straddles the two tables in the real store too: `n_valid_obs` and
    # `max_gap` are parcel columns, the rest are feature columns. The demo must match that
    # split, or `--drop-features meta` behaves differently here than on real data.
    assert set(META_FEATURES) <= set(f.columns) | set(g.columns)
    assert {"frac_l7", "n_dates", "n_valid_pixels"} <= set(f.columns)
    assert {"n_valid_obs", "max_gap"} <= set(g.columns)
    assert set(ORDER_FEATURES) <= set(f.columns)


def test_the_generator_is_deterministic(tmp_path, monkeypatch):
    """Committed generated data plus a nondeterministic generator means an unreviewable diff
    every time anyone regenerates."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mk", ROOT / "tools" / "make_demo_workspace.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    import numpy as np
    a = mk.build_parcels(np.random.default_rng(mk.SEED))
    b = mk.build_parcels(np.random.default_rng(mk.SEED))
    assert a.drop(columns="geometry").equals(b.drop(columns="geometry"))


@pytest.mark.slow
def test_train_then_evaluate_runs_end_to_end(tmp_path):
    """⭐ The quickstart itself. Subprocess, because LightGBM must not share this process
    with torch."""
    env = {"CC_PROC": str(DEMO), "CC_FEAT": str(DEMO / "features"),
           "CC_RUNS": str(tmp_path)}
    import os
    e = {**os.environ, **env}

    r = subprocess.run([sys.executable, "-m", "crop_classifier.cli", "train",
                        "--model", "lightgbm", "--run-name", "t"],
                       cwd=ROOT, env=e, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "t" / "cv_metrics.json").exists()
    m = json.loads((tmp_path / "t" / "cv_metrics.json").read_text())
    assert 0.2 < m["cv_macro_f1_mean"] < 0.95        # a working model, not a perfect one

    r = subprocess.run([sys.executable, "-m", "crop_classifier.cli", "evaluate",
                        str(tmp_path / "t")],
                       cwd=ROOT, env=e, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    # the protocol's two non-negotiables, on a run that has only CV
    assert "floor" in r.stdout and "skill" in r.stdout
    assert "not computed: LODO" in r.stdout

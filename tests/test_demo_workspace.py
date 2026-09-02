"""The demo workspace runs on a clone with no data and no Earth Engine account.

The promise `README.md` makes in its quickstart, and the one most likely to rot silently:
the demo is not on anyone's daily path, so the first to find it broken is a new collaborator.

It is a real sample of the national build, so these tests also guard the two properties that
make a small sample honest — whole regions move together into a split, and more than one
department is present so that leave-one-department-out can actually run.

The training arm runs in a **subprocess**: LightGBM and torch each bundle their own libomp
and co-loading them on macOS segfaults the interpreter (exit 139), which in a test session
takes the whole run down.
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
    for f in ("modeling_parcels.parquet", "label_map.json", "README.md", "expected_cv.json",
              "features/features_lightgbm.parquet"):
        assert (DEMO / f).exists(), f"data/demo/{f} missing"
    out = subprocess.run(["git", "ls-files", "data/demo"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    assert any(f.endswith("modeling_parcels.parquet") for f in out), (
        "data/demo is not tracked by git — check the !data/demo/ rules in .gitignore")


def test_more_than_one_department_is_present():
    """⭐ The demo's whole point is that someone can see CV and LODO disagree before they have
    any data. Leave-one-department-out needs more than one department to hold out; a
    single-department demo would teach the pipeline while hiding the split that decides."""
    g = gpd.read_parquet(DEMO / "modeling_parcels.parquet")
    assert g["dept"].nunique() >= 4
    assert (g.groupby("dept").size() >= 100).all(), "a department too small to hold out"


def test_a_completed_lodo_run_is_committed():
    """So `cc evaluate` prints the CV-vs-LODO gap on a fresh clone, instead of telling a new
    collaborator to go and run something that needs the data they do not have."""
    assert (DEMO / "lodo_summary_demo.json").exists()
    m = json.loads((DEMO / "lodo_summary_demo.json").read_text())
    assert m["n_departments"] >= 4
    cv = json.loads((DEMO / "expected_cv.json").read_text())["cv_macro_f1_mean"]
    assert m["mean_macro_f1"] < cv, (
        "the committed demo no longer shows LODO below CV — either the sample changed or "
        "something is wrong; that gap is what the demo exists to show")


def test_it_has_the_columns_the_real_loader_requires():
    g = gpd.read_parquet(DEMO / "modeling_parcels.parquet")
    required = {"COD_PREDIO", "label", "label_id", "split", "fold", "year", "quality_ok",
                "buffer_excl_test", "region_id", "block_id", "geometry", "centroid_lat"}
    assert required <= set(g.columns)
    assert {f"buffer_excl_fold{k}" for k in range(5)} <= set(g.columns)


def test_no_region_is_split_across_train_and_test():
    """The discipline the demo teaches: whole regions move together. Random parcel splits put
    a parcel's neighbours in its own training set — the cheapest way to inflate a CV score."""
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


def test_the_generator_refuses_to_run_without_the_source_data(monkeypatch, tmp_path):
    """Committed data plus a generator that silently produces something else is worse than no
    generator. Without the national workspace it must say so and stop — not write a partial
    demo over the committed one."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mk", ROOT / "tools" / "make_demo_workspace.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    monkeypatch.setattr(mk, "SRC_PROC", tmp_path / "absent")
    with pytest.raises(SystemExit) as e:
        mk._require_source()
    assert "DATA_ACCESS" in str(e.value)


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
    expected = json.loads((DEMO / "expected_cv.json").read_text())["cv_macro_f1_mean"]
    assert m["cv_macro_f1_mean"] == pytest.approx(expected, abs=0.05), (
        f"demo CV moved: {m['cv_macro_f1_mean']:.4f} against a recorded {expected:.4f}. "
        f"Training is seeded, so this is a real change in behaviour, not noise.")

    r = subprocess.run([sys.executable, "-m", "crop_classifier.cli", "evaluate",
                        str(tmp_path / "t")],
                       cwd=ROOT, env=e, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    # the protocol's two non-negotiables, on a run that has only CV
    assert "floor" in r.stdout and "skill" in r.stdout
    assert "not computed: LODO" in r.stdout

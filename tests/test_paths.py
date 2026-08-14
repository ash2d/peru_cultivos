"""``CC_PROC`` switches the tables directory; the feature store stays shared.

The whole 3-class workspace rests on this: if ``proc()`` were ever bound at import time
the perennial run would silently overwrite the 12-class tables (whose locked test set is
unspent). These tests pin the call-time behaviour.
"""

from __future__ import annotations

import importlib

import pytest

from crop_classifier import paths


def test_proc_defaults_to_shared(monkeypatch, tmp_path):
    monkeypatch.delenv("CC_PROC", raising=False)
    assert paths.proc() == paths.PROC_SHARED


def test_cc_proc_switches_tables_dir(monkeypatch, tmp_path):
    ws = tmp_path / "perennial"
    monkeypatch.setenv("CC_PROC", str(ws))
    assert paths.proc() == ws
    assert ws.exists()                      # created on demand


def test_feature_store_does_not_follow_cc_proc(monkeypatch, tmp_path):
    """``CC_PROC`` alone must NOT move the feature store: the 12-class and 3-class
    workspaces deliberately share one pixel store."""
    monkeypatch.delenv("CC_FEAT", raising=False)
    monkeypatch.setenv("CC_PROC", str(tmp_path / "perennial"))
    assert paths.FEAT == paths.PROC_SHARED / "features"
    assert paths.feat() == paths.PROC_SHARED / "features"


def test_cc_feat_switches_feature_store(monkeypatch, tmp_path):
    """...but ``CC_FEAT`` does, which is what keeps the all-Peru pixels out of the
    audited Piura store."""
    fs = tmp_path / "all_peru" / "features"
    monkeypatch.setenv("CC_FEAT", str(fs))
    assert paths.feat() == fs
    assert fs.exists()                      # created on demand


def test_feature_paths_follow_env_after_import(monkeypatch, tmp_path):
    """The import-time-binding regression, for the feature store this time: import the
    modules first, set ``CC_FEAT`` after, and every derived path must still move."""
    from crop_classifier.features import landsat_gee as lg
    from crop_classifier.perennial import panel as P

    fs = tmp_path / "ws_feat"
    monkeypatch.setenv("CC_FEAT", str(fs))
    assert lg.f_coverage() == fs / "coverage.parquet"
    assert P.panel_feat() == fs / "panel"
    assert P.panel_dirs(2001) == (fs / "panel", fs / "panel" / "2001")


def test_cc_runs_switches_run_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CC_RUNS", str(tmp_path / "runs_x"))
    assert paths.runs() == tmp_path / "runs_x"
    monkeypatch.delenv("CC_RUNS")
    assert paths.runs() == paths.ROOT / "runs"


@pytest.mark.parametrize("module,fn", [
    ("crop_classifier.labels", "out_paths"),
    ("crop_classifier.splits", "out_paths"),
])
def test_out_paths_follow_env_after_import(monkeypatch, tmp_path, module, fn):
    """The regression that matters: import first, set CC_PROC after — the module must
    still resolve into the new workspace."""
    mod = importlib.import_module(module)
    monkeypatch.setenv("CC_PROC", str(tmp_path / "ws"))
    for p in getattr(mod, fn)():
        assert p.parent == tmp_path / "ws"


def test_landsat_parcels_path_follows_env(monkeypatch, tmp_path):
    from crop_classifier.features import landsat_gee as lg
    monkeypatch.delenv("CC_FEAT", raising=False)
    monkeypatch.setenv("CC_PROC", str(tmp_path / "ws"))
    assert lg.f_parcels() == tmp_path / "ws" / "modeling_parcels.parquet"
    # coverage tracks the FEATURE store, not CC_PROC
    assert lg.f_coverage().parent == paths.feat() == paths.FEAT

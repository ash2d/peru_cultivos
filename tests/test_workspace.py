"""Named workspaces resolve, activate, and stay honest about what set what.

The point of the layer is that ``-w national`` cannot disagree with itself the way three
separate ``export`` lines could. These tests pin that, and pin that activation still works
*through* the environment variables, because every module downstream reads those at call
time and must keep doing so.
"""

from __future__ import annotations

import textwrap

import pytest

from crop_classifier import paths
from crop_classifier import workspace as W


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """A workspaces.yaml in a temp dir, with a deliberately shared feature store."""
    f = tmp_path / "workspaces.yaml"
    f.write_text(textwrap.dedent("""
        data_root: ./data
        runs_root: ./runs
        gee_project: some-gcp-project
        cenagro_source_dir: null
        workspaces:
          alpha:
            proc: processed
            feat: processed/features
            runs: .
            about: the first one
          beta:
            proc: processed/beta
            feat: processed/features      # shared with alpha on purpose
            runs: beta
    """))
    monkeypatch.setenv("CC_WORKSPACES_FILE", str(f))
    for v in W.ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(W, "_ACTIVATED", None)
    return tmp_path


def test_names_preserve_file_order(cfg):
    assert W.names() == ["alpha", "beta"]


def test_relative_roots_resolve_against_the_config_file_not_the_cwd(cfg, monkeypatch, tmp_path):
    """`data_root: ./data` must mean "next to workspaces.yaml". If it were resolved against
    the process cwd, running a command from a subdirectory would silently target a
    different, probably empty, store — the exact failure this layer exists to remove."""
    sub = tmp_path / "somewhere" / "else"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    assert W.resolve("alpha").proc == cfg / "data" / "processed"


def test_runs_dot_means_the_runs_root(cfg):
    assert W.resolve("alpha").runs == cfg / "runs"


def test_activate_sets_all_three_env_vars(cfg):
    ws = W.activate("beta")
    import os
    assert os.environ["CC_PROC"] == str(ws.proc)
    assert os.environ["CC_FEAT"] == str(ws.feat)
    assert os.environ["CC_RUNS"] == str(ws.runs)


def test_paths_follow_an_activated_workspace(cfg):
    """The whole contract: activate, then the ordinary path helpers point into it."""
    ws = W.activate("beta")
    assert paths.proc() == ws.proc
    assert paths.feat() == ws.feat
    assert paths.runs() == ws.runs


def test_activate_does_not_create_directories(cfg):
    """`--help` and `cc workspaces` must not scatter empty directories. Creation stays in
    paths.proc()/feat()/runs(), which run only for a workspace a command actually touches."""
    ws = W.activate("alpha")
    assert not ws.proc.exists()
    assert not ws.feat.exists()


def test_activated_reports_the_name_only_when_we_set_it(cfg, monkeypatch):
    assert W.activated() is None
    monkeypatch.setenv("CC_PROC", "/somewhere/manual")
    assert W.activated() is None            # an export is not an activation
    W.activate("alpha")
    assert W.activated() == "alpha"


def test_unknown_workspace_names_the_alternatives(cfg):
    with pytest.raises(SystemExit) as e:
        W.resolve("gamma")
    assert "alpha" in str(e.value) and "beta" in str(e.value)


def test_missing_config_says_which_file_to_edit(tmp_path, monkeypatch):
    monkeypatch.setenv("CC_WORKSPACES_FILE", str(tmp_path / "nope.yaml"))
    with pytest.raises(SystemExit) as e:
        W.load_config()
    assert "nope.yaml" in str(e.value)


def test_gee_project_prefers_the_environment(cfg, monkeypatch):
    assert W.gee_project() == "some-gcp-project"
    monkeypatch.setenv("GEE_PROJECT", "mine")
    assert W.gee_project() == "mine"


def test_gee_project_refuses_to_guess(cfg, monkeypatch):
    """Earth Engine bills a project and another user cannot use this one's, so an unset
    value must stop the run rather than fall back to a default that is not theirs."""
    monkeypatch.delenv("GEE_PROJECT", raising=False)
    (cfg / "workspaces.yaml").write_text("gee_project: ''\nworkspaces: {a: {proc: p}}\n")
    with pytest.raises(SystemExit) as e:
        W.gee_project()
    assert "earthengine authenticate" in str(e.value)


def test_cenagro_source_dir_unset_points_at_the_docs(cfg):
    with pytest.raises(SystemExit) as e:
        W.cenagro_source_dir()
    assert "DATA_ACCESS" in str(e.value)


def test_the_repo_config_parses_and_every_workspace_resolves(monkeypatch):
    """The checked-in workspaces.yaml itself, so a typo in it fails here and not halfway
    through someone's first extraction."""
    monkeypatch.delenv("CC_WORKSPACES_FILE", raising=False)
    assert "national" in W.names()
    for name in W.names():
        ws = W.resolve(name)
        assert ws.proc.is_absolute() and ws.feat.is_absolute() and ws.runs.is_absolute()

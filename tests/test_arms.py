"""The S2 arm driver, and specifically the three things it refuses.

This logic used to live in the `s2-train` CLI body, where the only way to exercise it was to
type the command. Three of its branches exist to say *no* to a combination — and a refusal
that is never tested is a refusal that quietly stops refusing.
"""

from __future__ import annotations

import pytest

from crop_classifier.labelling import arms


def test_rules_is_refused_on_a_two_class_target():
    """⛔ In a two-class space the rule's fallback resolves PASTURE_FALLOW to PERENNIAL. It
    would run to completion and return a meaningless number — worse than a crash, because
    nothing in the output says so."""
    for target in ("t2", "t2w"):
        with pytest.raises(SystemExit) as e:
            arms.check_model_is_runnable("fit", "rules", target)
        assert "meaningless" in str(e.value)


def test_rules_is_allowed_where_the_label_space_supports_it():
    for target in ("t3", "t3w", "t4", "t5"):
        arms.check_model_is_runnable("fit", "rules", target)


def test_only_the_model_is_refused_not_the_step():
    """`report` and `prep` do not fit anything, so the incompatibility does not apply."""
    arms.check_model_is_runnable("report", "rules", "t2")
    arms.check_model_is_runnable("prep", "rules", "t2")


def test_a_learned_model_is_never_refused():
    for model in ("lightgbm", "ltae"):
        arms.check_model_is_runnable("fit", model, "t2")


def test_the_refusal_follows_the_config_files(tmp_path, monkeypatch):
    """The incompatible set is derived from `rules_compatible:` in config/labels/*.yaml, so
    a new two-class label set is covered without anyone remembering to update a literal."""
    from crop_classifier import label_sets as L
    monkeypatch.setattr(L, "CONFIG_DIR", tmp_path)
    L.load.cache_clear()
    (tmp_path / "mine.yaml").write_text(
        "name: mine\ncollapse: {ANNUAL: NON_PERENNIAL}\nrules_compatible: false\n")
    with pytest.raises(SystemExit):
        arms.check_model_is_runnable("fit", "rules", "mine")
    L.load.cache_clear()


def test_an_unbuilt_workspace_names_the_command_that_builds_it(monkeypatch, tmp_path):
    from crop_classifier.labelling import train_prep as P
    monkeypatch.setattr(P, "LABELS_DIR", tmp_path)
    with pytest.raises(SystemExit) as e:
        arms.resolve_workspace("t4", "none", False)
    assert "prep" in str(e.value)


def test_an_unknown_climate_arm_lists_the_real_ones(monkeypatch):
    with pytest.raises(SystemExit) as e:
        arms.resolve_workspace("t4", "humidity", False)
    assert "humidity" in str(e.value)


def test_an_unknown_step_is_rejected_before_anything_is_touched():
    with pytest.raises(SystemExit) as e:
        arms.run_step("frobnicate")
    assert "frobnicate" in str(e.value)


def test_baseline_without_a_run_says_which_flag_is_missing(monkeypatch, tmp_path):
    """`baseline` transfers a saved Landsat model; with no --run it used to fail deep inside
    the loader on a None path."""
    monkeypatch.setattr(arms, "resolve_workspace", lambda *a, **k: tmp_path)
    monkeypatch.setattr(arms, "check_model_is_runnable", lambda *a, **k: None)
    with pytest.raises(SystemExit) as e:
        arms.run_step("baseline", run=None)
    assert "--run" in str(e.value)


def test_eval_test_is_refused_for_the_rules_control(monkeypatch, tmp_path):
    """The locked test is 161 parcels and spending it is one-way. `rules` is a floor
    exercise, never a candidate, so it must not be what spends it."""
    monkeypatch.setattr(arms, "resolve_workspace", lambda *a, **k: tmp_path)
    with pytest.raises(SystemExit) as e:
        arms.run_step("fit", model="rules", target="t4", eval_test=True)
    assert "control" in str(e.value)

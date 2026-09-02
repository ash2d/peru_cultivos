"""Label spaces are files, not code — and a new one needs no Python change.

This is the promise the repository makes to whoever inherits it: "new labels arrive, and you
add a label set." A test is the only thing that keeps that promise true, because the moment
one modelling decision has to be made in Python again, the next five will be too.
"""

from __future__ import annotations

import pandas as pd
import pytest

from crop_classifier import label_sets as L


def test_every_shipped_label_set_loads():
    assert set(L.available()) >= {"t2", "t2w", "t3", "t3w", "t4", "t5"}
    for name in L.available():
        L.load(name)


def test_the_full_space_is_the_five_recorded_classes():
    assert L.load("t5").classes == sorted(L.SOURCE_CLASSES)


def test_collapse_merges_and_null_drops():
    t3 = L.load("t3")
    assert t3.classes == ["ANNUAL", "OTHER", "PERENNIAL"]
    got = t3.apply(pd.Series(["PERENNIAL", "WOODY_NON_CROP", "NON_AGRICULTURE", "ANNUAL"]))
    # a dropped class comes back missing (pandas normalises None to NaN on map), which is
    # what `build_workspace`'s `.notna()` filter removes
    assert list(got[[0, 3]]) == ["PERENNIAL", "ANNUAL"]
    assert got[[1, 2]].isna().all()


def test_t3_and_t3w_differ_only_in_where_woody_lands():
    """The bracket that decides more than most features do: this one choice moves PERENNIAL
    F1 by 0.190, so a result read at only one end of it has not been read."""
    t3, t3w = L.load("t3"), L.load("t3w")
    assert t3.collapse["WOODY_NON_CROP"] is None
    assert t3w.collapse["WOODY_NON_CROP"] == "PERENNIAL"
    assert t3.classes == t3w.classes


def test_the_two_class_sets_refuse_the_rules_baseline():
    """`rules` maps three semantic groups onto label ids; in a two-class space its fallback
    resolves PASTURE_FALLOW to PERENNIAL — a meaningless number, worse than an error."""
    assert L.rules_incompatible() == {"t2", "t2w"}


def test_an_unknown_name_names_the_directory_to_add_to():
    with pytest.raises(SystemExit) as e:
        L.load("t9_does_not_exist")
    assert "config/labels" in str(e.value).replace("\\", "/")


def test_a_typo_in_collapse_is_an_error_not_a_silent_no_op(tmp_path, monkeypatch):
    """A key that is not a recorded class would otherwise be ignored, leaving the arm
    training in a different label space than its author believes."""
    monkeypatch.setattr(L, "CONFIG_DIR", tmp_path)
    L.load.cache_clear()
    (tmp_path / "typo.yaml").write_text(
        "name: typo\ncollapse:\n  WOODY_NONCROP: PERENNIAL\n")     # missing underscore
    with pytest.raises(SystemExit) as e:
        L.load("typo")
    assert "WOODY_NONCROP" in str(e.value)
    L.load.cache_clear()


def test_a_name_that_disagrees_with_its_filename_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "CONFIG_DIR", tmp_path)
    L.load.cache_clear()
    (tmp_path / "t7.yaml").write_text("name: t8\ncollapse: {}\n")
    with pytest.raises(SystemExit):
        L.load("t7")
    L.load.cache_clear()


def test_a_brand_new_label_set_needs_no_code_change(tmp_path, monkeypatch):
    """⭐ The acceptance test for "adapt when new labels come later".

    Drop a YAML file in the directory; it is loadable, it appears in `available()`, its
    collapse map reaches `train_prep.TARGETS`, and nothing under src/ was edited.
    """
    monkeypatch.setattr(L, "CONFIG_DIR", tmp_path)
    L.load.cache_clear()
    (tmp_path / "orchard2.yaml").write_text(
        "name: orchard2\n"
        "about: perennial vs everything farmable, dropping non-agriculture\n"
        "collapse:\n"
        "  ANNUAL: FARMED_NOT_PERENNIAL\n"
        "  OTHER: FARMED_NOT_PERENNIAL\n"
        "  WOODY_NON_CROP: PERENNIAL\n"
        "  NON_AGRICULTURE: null\n")

    ls = L.load("orchard2")
    assert ls.classes == ["FARMED_NOT_PERENNIAL", "PERENNIAL"]
    assert "orchard2" in L.available()

    from crop_classifier.labelling import train_prep as P
    assert P.TARGETS["orchard2"] == ls.collapse       # resolved live, not bound at import

    got = ls.apply(pd.Series(["PERENNIAL", "WOODY_NON_CROP", "ANNUAL", "NON_AGRICULTURE"]))
    assert list(got[:3]) == ["PERENNIAL", "PERENNIAL", "FARMED_NOT_PERENNIAL"]
    assert pd.isna(got[3])
    L.load.cache_clear()


def test_train_prep_still_exposes_the_names_it_always_did():
    """`P.TARGETS` and `P.RULES_INCOMPATIBLE` were a literal; they are now resolved from
    disk. Anything that imported them keeps working, which is why the move was safe."""
    from crop_classifier.labelling import train_prep as P
    assert set(P.TARGETS) == set(L.available())
    assert P.RULES_INCOMPATIBLE == L.rules_incompatible()

"""Guards on the climate-covariate arms (docs/RESULTS.md §8.8).

The comparison in §8.8 is only meaningful if the arms differ in **exactly one** thing — the
extra columns — so what is pinned here is sameness, not accuracy:

* every arm's parcels, folds and dead-zones come from the *same* file, by symlink. A rebuilt
  copy would let a fold drift and the delta would then be unattributable;
* a store with no ``statics.npz`` gives ``n_static == 0`` and a batch with no ``stat`` key, so
  the sequence models are byte-for-byte the ones fitted in §8.2 — this is what made the
  ``none`` arms reproduce their recorded numbers exactly;
* statics are standardised on **train only**. A static normalised against the validation
  fold's own mean is a leak, and with two columns over ~550 parcels it would be a large one;
* the ``latlon`` control really is coordinates and really is the same width as ``both``,
  because the whole reading of §8.8 rests on comparing the two;
* the majority-class floor is the one printed beside every macro-F1 (CLAUDE.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crop_classifier.data import FN_STATICS, Normalizer
from crop_classifier.labelling import climate_arms as C


# ------------------------------------------------------------------------------------
# arm definitions
# ------------------------------------------------------------------------------------
def test_the_control_arm_is_coordinates_and_carries_no_climate():
    assert C.CLIMATE_SETS["latlon"] == ["centroid_lat", "centroid_lon"]
    assert not set(C.CLIMATE_SETS["latlon"]) & set(C.CLIMATE_ONLY)


def test_the_control_is_the_same_width_as_the_arm_it_controls():
    # a 1-column control against a 2-column arm would confound content with count
    assert len(C.CLIMATE_SETS["latlon"]) == len(C.CLIMATE_SETS["both"])


def test_both_is_exactly_the_union_of_the_single_covariate_arms():
    assert C.CLIMATE_SETS["both"] == C.CLIMATE_SETS["temp"] + C.CLIMATE_SETS["rain"]


def test_the_no_climate_arm_adds_nothing_and_reuses_the_base_workspace():
    assert C.CLIMATE_SETS["none"] == []
    assert C.ws_dir("t4", "none", True).name == "ws_t4_pilot"


@pytest.mark.parametrize("arm", [a for a in C.ARMS if a != "none"])
def test_every_other_arm_gets_its_own_workspace(arm):
    assert C.ws_dir("t4", arm, True).name == f"ws_t4_pilot__clim_{arm}"


# ------------------------------------------------------------------------------------
# the majority-class floor printed beside every macro-F1
# ------------------------------------------------------------------------------------
def test_the_floor_is_the_always_guess_the_largest_class_score():
    counts = pd.Series({"a": 50, "b": 30, "c": 20})
    p = 0.5
    assert C._majority_floor(counts) == pytest.approx((2 * p / (1 + p)) / 3)


def test_the_floor_rises_when_classes_are_pooled():
    """§8.2c's trap: fewer classes is a higher floor, so bare macro-F1s are not comparable."""
    four = pd.Series({"a": 444, "b": 165, "c": 141, "d": 115})
    two = pd.Series({"a": 444, "rest": 421})
    assert C._majority_floor(two) > C._majority_floor(four)
    # RESULTS.md §8.2c/§8.8 print 0.171 for `t4`; these are the 865-usable counts rather
    # than the 704 trainval ones, so the check is on the same value to 2 dp
    assert C._majority_floor(four) == pytest.approx(0.170, abs=0.002)


# ------------------------------------------------------------------------------------
# statics: absent must mean absent
# ------------------------------------------------------------------------------------
def test_a_store_without_a_statics_file_yields_no_statics(tmp_path, monkeypatch):
    ds = _seq_dataset(tmp_path, monkeypatch, statics=None)
    assert ds.statics is None and ds.n_static == 0
    assert "stat" not in ds[0]


def test_a_store_with_statics_exposes_them_on_every_item(tmp_path, monkeypatch):
    ds = _seq_dataset(tmp_path, monkeypatch, statics=np.array([[1.0, 2.0], [3.0, 4.0]],
                                                             dtype="float32"))
    assert ds.n_static == 2 and ds.static_names == ["tmean_c", "precip_mm_yr"]
    assert tuple(ds[0]["stat"].shape) == (2,)


def test_statics_are_standardised_with_the_train_normalizer_not_refit(tmp_path, monkeypatch):
    """A validation set must inherit the train mean/std, or the comparison leaks."""
    train = _seq_dataset(tmp_path, monkeypatch,
                         statics=np.array([[10.0, 100.0], [20.0, 200.0]], dtype="float32"))
    val = _seq_dataset(tmp_path, monkeypatch,
                       statics=np.array([[10.0, 100.0], [20.0, 200.0]], dtype="float32"),
                       normalizer=train.normalizer)
    assert val.normalizer.smean is train.normalizer.smean
    np.testing.assert_allclose(val.statics, train.statics)


def test_a_missing_parcel_in_the_statics_file_raises_rather_than_dropping(tmp_path,
                                                                         monkeypatch):
    with pytest.raises(KeyError):
        _seq_dataset(tmp_path, monkeypatch,
                     statics=np.array([[1.0, 2.0]], dtype="float32"),
                     statics_cods=["p0"])


def test_the_normalizer_state_round_trips_the_static_block():
    n = Normalizer()
    n.fit(np.arange(20, dtype="float32").reshape(10, 2), np.ones((10, 2), bool))
    n.fit_static(np.array([[1.0, 10.0], [3.0, 30.0]], dtype="float32"))
    back = Normalizer.from_state(n.state())
    np.testing.assert_allclose(back.smean, n.smean)
    np.testing.assert_allclose(back.sstd, n.sstd)


def test_a_state_written_before_statics_existed_still_loads():
    n = Normalizer()
    n.fit(np.arange(20, dtype="float32").reshape(10, 2), np.ones((10, 2), bool))
    back = Normalizer.from_state(n.state())
    assert back.smean is None


# ------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------
def _seq_dataset(tmp_path, monkeypatch, statics, statics_cods=None, normalizer=None):
    """A two-parcel `SeqDataset` over a synthetic feature directory."""
    import geopandas as gpd
    from shapely.geometry import Point

    from crop_classifier.data import SeqDataset

    fd = tmp_path / "features"
    fd.mkdir(exist_ok=True)
    cods = np.array(["p0", "p1"], dtype=object)
    np.savez(fd / "tensor_perdate.npz", cod_predio=cods,
             X=np.zeros((2, 4, 11), dtype="float32"),
             doy=np.zeros((2, 4), dtype="float32"),
             mask=np.ones((2, 4), dtype=bool),
             pixmask=np.ones((2, 4), dtype=bool))
    sf = fd / FN_STATICS
    if statics is None:
        sf.unlink(missing_ok=True)
    else:
        np.savez(sf, cod_predio=np.array(statics_cods or ["p0", "p1"], dtype=object),
                 X=statics, names=np.array(["tmean_c", "precip_mm_yr"], dtype=object))
    parcels = gpd.GeoDataFrame(
        {"COD_PREDIO": cods, "label_id": [0, 1]},
        geometry=[Point(0, 0), Point(1, 1)], crs=4326)
    return SeqDataset("sequence", parcels, parcels.index, normalizer, feat_dir=fd)


# ------------------------------------------------------------------------------------
# §8.8b: the bracket, the control comparison and the WOODY boundary probe
# ------------------------------------------------------------------------------------
def test_the_control_is_never_one_of_the_arms_it_is_compared_against():
    """`control_paired` must compare temp/rain/both against latlon, never latlon to itself.

    ARMS is ordered `none, temp, rain, both, latlon` and the slice `ARMS[1:-1]` relies on
    the control being last. If a sixth arm were appended, that slice would silently start
    testing `latlon` against the new arm instead — a comparison that would still print.
    """
    assert C.ARMS[0] == "none"
    assert C.ARMS[-1] == "latlon"
    assert C.ARMS[1:-1] == ["temp", "rain", "both"]


def test_the_woody_probe_returns_nothing_on_a_target_that_has_no_woody_class(tmp_path,
                                                                             monkeypatch):
    """`t3w` folds WOODY_NON_CROP into PERENNIAL, so the sub-problem does not exist there.

    It has to come back empty rather than raise or, worse, index some other class into
    `iW` — §8.8b reads the `t4` table as evidence about `t3w`, so a silently wrong `t3w`
    row would be read as a comparison.
    """
    import json

    ws = tmp_path / "ws_t3w_pilot"
    ws.mkdir()
    (ws / "label_map.json").write_text(json.dumps({"ANNUAL": 0, "OTHER": 1,
                                                   "PERENNIAL": 2}))
    monkeypatch.setattr(C.P, "ws_dir", lambda *a, **k: ws)
    assert C.woody_boundary("t3w", True).empty


def test_the_bracket_needs_two_targets_to_be_a_bracket(monkeypatch):
    """One target is not a bracket. Returning a half-table would read as a comparison."""
    monkeypatch.setattr(C, "collect", lambda t, inc: pd.DataFrame()
                        if t == "t3w" else pd.DataFrame({"model": ["lightgbm"],
                                                         "arm": ["none"]}))
    assert C.bracket(("t4", "t3w")).empty


def test_the_bracket_suffixes_every_metric_with_its_target():
    """The floor moves between targets (0.171 -> 0.228), so no column may be shared.

    An unsuffixed metric column would silently carry one target's value for both.
    """
    cols = ["cv_macro_f1", "lodo_macro_f1", "lodo_sd", "n_dept", "cv_skill", "lodo_skill",
            "cv_perennial_f1", "lodo_perennial_f1", "d_cv_macro_f1", "d_lodo_macro_f1",
            "majority_floor"]
    frame = pd.DataFrame([{"model": "lightgbm", "arm": "none",
                           **{c: 1.0 for c in cols}}])
    by_target = {"t4": frame, "t3w": frame.copy()}
    orig = C.collect
    try:
        C.collect = lambda t, inc: by_target[t]
        out = C.bracket(("t4", "t3w"))
    finally:
        C.collect = orig
    assert set(out.columns) == {"model", "arm", *[f"{c}_{t}" for c in cols
                                                  for t in ("t4", "t3w")]}
    assert "majority_floor" not in out.columns

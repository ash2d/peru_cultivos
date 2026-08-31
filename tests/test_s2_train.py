"""Guards on the S2 label -> trainable workspace path and the LTAE sequence tensor.

Two of these pin traps that return a plausible wrong answer rather than an error, which is
this project's recurring failure shape:

* **the position axis wraps.** `ag_year` is Aug 1 - Jul 31, so a day-of-year encoding runs
  365 -> 1 in the middle of every parcel's series and the sinusoidal encoder then places
  midwinter observations next to the first week of August. Nothing raises; the model just
  learns worse.
* **`quality_ok` is NA for every parcel in this campaign** and `load_parcels` filters on
  `== True`, so a workspace that forwards the column untouched trains on zero rows and
  reports it as an empty dataset, not as an error.
"""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from crop_classifier.features.s2_assemble import build_sequence_tensor
from crop_classifier.labelling.train_prep import TARGETS, TO_LANDSAT, _reassign_folds


# ------------------------------------------------------------------------------------
# the sequence tensor
# ------------------------------------------------------------------------------------
def _px(cod: str, dates: list[str]) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame({
        "COD_PREDIO": [cod] * n, "date": pd.to_datetime(dates),
        "n_px": [20] * n, "eroded": [True] * n,
        **{b: np.linspace(500, 3000, n) for b in
           ("B", "G", "R", "NIR", "SWIR1", "SWIR2")},
    })


def _parcels(cod: str, imagery_date: str) -> pd.DataFrame:
    return pd.DataFrame({"COD_PREDIO": [cod], "imagery_date": [imagery_date],
                         "area_ha": [1.0]})


def test_positions_are_days_since_aug1_and_never_wrap():
    # a window that straddles the New Year: Aug 1 2021 - Jul 31 2022
    dates = ["2021-08-15", "2021-11-20", "2021-12-28", "2022-01-05", "2022-04-10",
             "2022-07-20"]
    t = build_sequence_tensor(_px("A", dates), _parcels("A", "2022-03-01"))
    pos = t["doy"][0][t["mask"][0]]
    assert len(pos) == len(dates)
    # monotone across the New Year — a day-of-year encoding would drop 362 -> 5 here
    assert (np.diff(pos) > 0).all(), pos
    assert pos[0] == 14                      # Aug 15 is 14 days after Aug 1
    assert 0 <= pos.min() and pos.max() <= 364


def test_observations_outside_the_agricultural_year_are_dropped():
    dates = ["2020-09-01", "2021-08-15", "2022-04-10", "2023-01-01"]
    t = build_sequence_tensor(_px("A", dates), _parcels("A", "2022-03-01"))
    assert t["mask"][0].sum() == 2           # only the two inside Aug 2021 - Jul 2022


def test_padded_positions_are_masked_out():
    t = build_sequence_tensor(_px("A", ["2021-08-15", "2022-04-10"]),
                              _parcels("A", "2022-03-01"))
    assert t["mask"][0][:2].all() and not t["mask"][0][2:].any()
    assert (t["X"][0][~t["mask"][0]] == 0).all()


def test_tensor_carries_the_eleven_channels_in_order():
    from crop_classifier.features.indices import CHANNELS
    t = build_sequence_tensor(_px("A", ["2021-08-15", "2022-04-10"]),
                              _parcels("A", "2022-03-01"))
    assert list(t["channels"]) == CHANNELS
    assert t["X"].shape[2] == len(CHANNELS) == 11


# ------------------------------------------------------------------------------------
# label targets
# ------------------------------------------------------------------------------------
def _apply(target: str, labels: list[str]) -> list[str]:
    m = TARGETS[target]
    return [v for v in (m.get(x, x) for x in labels) if v is not None]


ALL5 = ["PERENNIAL", "ANNUAL", "OTHER", "WOODY_NON_CROP", "NON_AGRICULTURE"]


def test_t5_is_the_identity_over_the_five_real_classes():
    assert sorted(set(_apply("t5", ALL5))) == sorted(ALL5)


def test_t4_pools_non_agriculture_into_other():
    out = _apply("t4", ALL5)
    assert "NON_AGRICULTURE" not in out
    assert sorted(set(out)) == sorted(["PERENNIAL", "ANNUAL", "OTHER", "WOODY_NON_CROP"])


def test_t3_drops_both_classes_the_landsat_model_has_no_word_for():
    out = _apply("t3", ALL5)
    assert sorted(set(out)) == ["ANNUAL", "OTHER", "PERENNIAL"]
    assert len(out) == 3                     # dropped, not remapped


def test_t3w_keeps_every_parcel_and_folds_woody_into_perennial():
    out = _apply("t3w", ALL5)
    assert len(out) == len(ALL5)             # nothing dropped
    assert sorted(set(out)) == ["ANNUAL", "OTHER", "PERENNIAL"]
    assert TARGETS["t3w"]["WOODY_NON_CROP"] == "PERENNIAL"


def test_only_the_three_landsat_mappable_classes_have_a_counterpart():
    # the baseline must never silently score WOODY_NON_CROP or NON_AGRICULTURE against a
    # model whose label space cannot express them
    assert set(TO_LANDSAT) == {"PERENNIAL", "ANNUAL", "OTHER"}
    assert set(TO_LANDSAT.values()) == {"PERENNIAL", "ANNUAL", "PASTURE_FALLOW"}


# ------------------------------------------------------------------------------------
# folding the pilot in
# ------------------------------------------------------------------------------------
def _fold_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "region_id": ["r1", "r1", "r2", "r3", "r1", "r9", "r9", "r8"],
        "fold":      [  0,    0,    1,    2,   -1,   -1,   -1,   -1],
    })


def test_a_pilot_parcel_inherits_the_fold_of_a_region_already_assigned():
    f = _reassign_folds(_fold_frame(), seed=42)
    assert f.iloc[4] == 0, "r1 is already fold 0; its pilot parcel must not split the region"


def test_frozen_folds_are_never_moved():
    df = _fold_frame()
    f = _reassign_folds(df, seed=42)
    had = df["fold"] >= 0
    assert (f[had].values == df.loc[had, "fold"].values).all()


def test_every_region_ends_in_exactly_one_fold():
    df = _fold_frame()
    df["fold"] = _reassign_folds(df, seed=42)
    assert (df.groupby("region_id")["fold"].nunique() == 1).all()


def test_no_parcel_is_left_unassigned():
    assert (_reassign_folds(_fold_frame(), seed=42) >= 0).all()


# ------------------------------------------------------------------------------------
# workspace schema
# ------------------------------------------------------------------------------------
REQUIRED = ["COD_PREDIO", "label", "label_id", "split", "fold", "quality_ok",
            "year", "area_ha", "n_valid_obs", "max_gap", "buffer_excl_test",
            *[f"buffer_excl_fold{k}" for k in range(5)]]


@pytest.mark.parametrize("target", sorted(TARGETS))
def test_built_workspace_has_what_the_trainer_reads(target):
    from crop_classifier.labelling.train_prep import ws_dir
    ws = ws_dir(target)
    if not (ws / "modeling_parcels.parquet").exists():
        pytest.skip(f"{ws.name} not built")
    df = gpd.read_parquet(ws / "modeling_parcels.parquet")
    assert not set(REQUIRED) - set(df.columns)
    # `quality_ok` is NA on every parcel in the campaign and `load_parcels` filters on
    # `== True`; forwarding it untouched trains on nothing and says so as an empty dataset
    assert (df["quality_ok"] == True).all()  # noqa: E712
    labels = json.loads((ws / "label_map.json").read_text())
    assert set(df["label"]) == set(labels)
    assert df["label_id"].map({v: k for k, v in labels.items()}).equals(df["label"])
    assert set(df["fold"][df["split"] == "trainval"]) <= set(range(5))


def test_geometry_survives_the_workspace_build():
    from crop_classifier.labelling.train_prep import ws_dir
    ws = ws_dir("t4")
    if not (ws / "modeling_parcels.parquet").exists():
        pytest.skip("t4 not built")
    df = gpd.read_parquet(ws / "modeling_parcels.parquet")
    assert isinstance(df, gpd.GeoDataFrame) and df.geometry.notna().all()
    assert isinstance(df.geometry.iloc[0].representative_point(), Point)


# ------------------------------------------------------------------------------------
# leave-one-department-out must not spend the locked test
# ------------------------------------------------------------------------------------
def test_lodo_predictions_contain_no_locked_test_parcel():
    """The one invariant that cannot be recovered if it is broken.

    `dept_transfer` drops `split == "test"` before it does anything else, so no held-out
    department can contain a locked-test parcel and no training department can either. If
    this ever fails the test set is spent and there is no second one.
    """

    from crop_classifier.labelling.train_prep import LABELS_DIR

    preds = sorted(LABELS_DIR.glob("ws_*/lodo_*_preds.parquet"))
    if not preds:
        pytest.skip("no LODO run in this workspace")
    lab = gpd.read_parquet(LABELS_DIR / "labelled_parcels.parquet")
    locked = set(lab.loc[lab["split"] == "test", "COD_PREDIO"].astype(str))
    assert locked, "the frozen split has no test parcels — that is itself a bug"
    for f in preds:
        got = set(pd.read_parquet(f)["COD_PREDIO"].astype(str))
        assert not (got & locked), f"{f.name} scored {len(got & locked)} locked-test parcels"


def test_every_department_is_held_out_exactly_once():
    """Each parcel appears in exactly one held-out department, so the mean is unweighted
    over departments and every parcel contributes once."""
    from crop_classifier.labelling.train_prep import LABELS_DIR

    preds = sorted(LABELS_DIR.glob("ws_*/lodo_*_preds.parquet"))
    if not preds:
        pytest.skip("no LODO run in this workspace")
    d = pd.read_parquet(preds[0])
    assert d.groupby("COD_PREDIO").size().max() == 1
    assert d.groupby("dept")["COD_PREDIO"].nunique().min() >= 35


# ------------------------------------------------------------------------------------
# the two-class collapse
# ------------------------------------------------------------------------------------
def test_t2_is_perennial_versus_everything_else():
    assert TARGETS["t2"] == {"ANNUAL": "NON_PERENNIAL", "OTHER": "NON_PERENNIAL",
                             "NON_AGRICULTURE": "NON_PERENNIAL",
                             "WOODY_NON_CROP": "NON_PERENNIAL"}
    # PERENNIAL is absent from the map, so it maps to itself and no parcel is dropped
    assert "PERENNIAL" not in TARGETS["t2"]
    assert None not in TARGETS["t2"].values()


def test_t2w_puts_woody_on_the_perennial_side_and_t2_does_not():
    """The only difference between the two, and it is the whole decision."""
    assert TARGETS["t2"]["WOODY_NON_CROP"] == "NON_PERENNIAL"
    assert TARGETS["t2w"]["WOODY_NON_CROP"] == "PERENNIAL"
    for t in ("t2", "t2w"):
        assert set(TARGETS[t]) == {"ANNUAL", "OTHER", "NON_AGRICULTURE", "WOODY_NON_CROP"}


def test_the_two_class_targets_never_reuse_the_name_OTHER():
    """`OTHER` means the specific state "farmable but not cropped" in t3/t4/t5.

    Reusing it for "not perennial" would silently redefine a class name across workspaces,
    so the negative class is called NON_PERENNIAL.
    """
    for t in ("t2", "t2w"):
        assert "OTHER" not in set(TARGETS[t].values())
        assert "NON_PERENNIAL" in set(TARGETS[t].values())


def test_the_rule_model_is_refused_on_the_two_class_targets():
    """It would run and return a meaningless number, which is worse than an error.

    `RuleModel._class_ids` maps three semantic names onto label ids and falls back to
    position when a name is absent; in a two-class space `PASTURE_FALLOW` falls back to
    id 1, which is `PERENNIAL`.
    """
    from crop_classifier.labelling.train_prep import RULES_INCOMPATIBLE
    assert RULES_INCOMPATIBLE == {"t2", "t2w"}
    from crop_classifier.perennial.rules import RuleModel
    m = RuleModel()
    m.set_class_names(["NON_PERENNIAL", "PERENNIAL"])
    per, ann, pas = m._class_ids()
    assert per == 1
    assert pas == per, ("the collision this guard exists for: with two classes the "
                        "PASTURE_FALLOW fallback resolves to PERENNIAL")


def test_majority_baseline_rises_as_classes_are_removed():
    """macro-F1 is not comparable across targets, and this is why.

    Averaging F1 over 2 classes when only one is ever predicted still collects a full score
    on that one and divides by 2. The floor roughly triples from t5 to t2, so a macro-F1
    that goes up when the label space is collapsed may be a model that got worse.
    """
    from crop_classifier.labelling.report_s2 import majority_baseline
    from crop_classifier.labelling.train_prep import ws_dir
    if not (ws_dir("t2", True) / "modeling_parcels.parquet").exists():
        pytest.skip("workspaces not built here")
    b = {t: majority_baseline(t)["majority_macro_f1"] for t in ("t5", "t4", "t3w", "t2")}
    assert b["t5"] < b["t4"] < b["t3w"] < b["t2"]
    assert b["t2"] > 2 * b["t5"]

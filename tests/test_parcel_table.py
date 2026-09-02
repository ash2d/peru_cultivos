"""Guards on the per-parcel export (`analysis parcel-table`).

The table is what someone leaves with, so the tests are on the two things a reader cannot
check by looking at it:

* a prediction in it was **held out** — the model was fitted on the campaign parcels, so a
  column silently filled from `cc predict` would be the model grading its own training rows;
* the four observations are in **one vocabulary** — `OTHER` and `PASTURE_FALLOW` are the same
  class under two instruments' names, and a table that keeps both cannot be crosstabbed.

The join itself is exercised once against the committed panel, and skipped without it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from crop_classifier.allperu import parcel_table as T


def _preds(ids: list[str], best: str = "PERENNIAL") -> pd.DataFrame:
    """A predictions frame in the shape `train.py` writes."""
    p = pd.DataFrame({"COD_PREDIO": ids, "y_true": 0})
    for c in ("ANNUAL", "OTHER", "PERENNIAL"):
        p[f"prob_{c}"] = 0.9 if c == best else 0.05
    return p


def _run(tmp_path, folds: dict[int, list[str]], test: list[str] | None = None):
    for k, ids in folds.items():
        (tmp_path / f"fold{k}").mkdir()
        _preds(ids).to_parquet(tmp_path / f"fold{k}" / "preds_val.parquet", index=False)
    if test is not None:
        _preds(test, best="ANNUAL").to_parquet(tmp_path / "preds_test.parquet", index=False)
    return tmp_path


def test_a_locked_test_parcel_is_kept_from_the_refit_not_the_fold(tmp_path):
    """Test parcels appear in a validation fold **and** in the locked-test file.

    They carry a fold id, and `data.fold_split` takes every row of that fold as validation
    while the train side is `split == "trainval"` only — so both readings are held out, and
    the one to keep is the final refit's. Getting this wrong would double the parcel.
    """
    out = T.held_out_predictions(_run(tmp_path, {0: ["a", "t1"], 1: ["b"]}, test=["t1"]))
    assert len(out) == 3 and not out["COD_PREDIO"].duplicated().any()
    row = out.loc[out["COD_PREDIO"] == "t1"].iloc[0]
    assert row["s2_pred_source"] == "locked_test"
    assert row["s2_pred_label"] == "ANNUAL"          # the refit's read, not the fold's


def test_two_folds_holding_the_same_trainval_parcel_still_raises(tmp_path):
    """The overlap above is expected; overlapping *folds* are a split bug and must not pass
    silently as one parcel with two answers."""
    with pytest.raises(ValueError, match="overlapping folds"):
        T.held_out_predictions(_run(tmp_path, {0: ["a"], 1: ["a"]}))


def test_an_abstention_is_not_a_prediction(tmp_path):
    """`cc predict` abstains by writing a null label and a reason; carrying the row through
    with its `pred_proba` would put a number in the column for a parcel the model refused."""
    f = tmp_path / "applied.parquet"
    pd.DataFrame({"COD_PREDIO": ["a", "b"], "pred_label": ["PERENNIAL", None],
                  "pred_proba": [0.9, 0.4], "abstained": [False, True]}).to_parquet(f)
    out = T.applied_predictions(f)
    assert list(out["COD_PREDIO"]) == ["a"]
    assert set(out["s2_pred_source"]) == {"applied"}


def test_the_four_observations_share_one_vocabulary():
    """`OTHER` (imagery) and `PASTURE_FALLOW` (declared) are the same class. The export maps
    both sides onto the declared names, so `pett_class`, `cen_class`, `s2_class` and
    `s2_pred_class` can be crosstabbed against each other without a rename."""
    declared = {"PERENNIAL", "ANNUAL", "PASTURE_FALLOW"}
    assert set(T.PRED_TO_DECLARED.values()) <= declared
    assert {v for v in T.S2_TO_DECLARED.values() if v is not None} <= declared
    # ...and the two classes with no declared counterpart stay unmapped, never guessed
    assert T.S2_TO_DECLARED["WOODY_NON_CROP"] is None
    assert T.S2_TO_DECLARED["NON_AGRICULTURE"] is None


@pytest.mark.skipif(not T.PANEL.exists(), reason="needs the committed national panel")
def test_the_export_is_one_row_per_parcel_over_the_committed_panel(tmp_path):
    """The universe is the census-linked panel and every later observation is a LEFT join:
    the row count must not move, and a parcel no instrument reached must still be there with
    nulls rather than dropped."""
    panel = pd.read_parquet(T.PANEL, columns=["COD_PREDIO"])
    df = T.build(out=tmp_path / "t.parquet", verbose=False)
    assert len(df) == len(panel)
    assert not df["COD_PREDIO"].duplicated().any()
    assert list(df.columns) == T.COLUMNS
    assert df["pett_class"].notna().all() and df["cen_class"].notna().all()
    # the imagery columns are sparse by construction — the campaign drew from the PETT
    # population, not from this subset, so a full column here would mean a bad join
    assert 0 < df["s2_class"].notna().sum() < len(df)
    assert (df["n_observations"] >= 2).all()

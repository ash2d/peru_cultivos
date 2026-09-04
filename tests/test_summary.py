"""Guards on `analysis summary` — the one-table descriptive export.

It restates published numbers, so the test that matters is that it still agrees with them:
the national row is the §7 headline (+9.9 pp, gap −2.1 pp) and the per-department rows are
`national_by_dept.csv`. Two ways it could silently stop agreeing are pinned separately —
comparing all parcels instead of crop-on-both-sides, and dropping the post-stratification.
"""

from __future__ import annotations

import pandas as pd
import pytest

from crop_classifier.allperu import summary as S
from crop_classifier.allperu.cenagro_shift import poststratify

pytestmark = pytest.mark.skipif(not (S.PANEL.exists() and S.TENURE.exists()),
                                reason="needs the committed panel and tenure tables")


def test_the_national_row_is_the_published_headline():
    row = S.build().iloc[0]
    assert row["perennial_before_pct"] == 16.5 and row["perennial_after_pct"] == 26.4
    assert row["perennial_change_pp"] == 9.9
    assert row["tenure_gap_pp"] == -2.1
    assert row["n_linked"] == 63766          # crop-on-both-sides, not all 95,941


def test_the_department_rows_match_the_published_table():
    got = S.build(by_dept=True).set_index("dept")
    want = pd.read_csv(S.ROOT / "data" / "processed" / "cenagro" /
                       "national_by_dept.csv").set_index("dept")
    for d, r in want.iterrows():
        assert got.loc[d, "n_linked"] == r["n"]
        assert got.loc[d, "perennial_change_pp"] == r["change_pp"]
        assert got.loc[d, "tenure_gap_pp"] == r["tenure_diff_pp"]


def test_the_change_is_conditional_on_a_crop_recorded_on_both_sides():
    """The census asks which crop is grown, so a fallow parcel leaves the frame. Comparing
    all parcels reads that instrument difference as land change."""
    panel = pd.read_parquet(S.PANEL, columns=["pett_class", "cen_class"])
    assert len(S.crop_panel()) < len(panel)
    assert not S.crop_panel()[["pett_class", "cen_class"]].isin(["PASTURE_FALLOW"]).any().any()


def test_post_stratification_moves_the_national_change():
    """The name link over-selects perennial parcels, so the raw panel change is not the
    national one — if these ever agree, the reweighting has stopped being applied."""
    crop = S.crop_panel()
    assert S._change(crop)["perennial_change_pp"] == 8.4            # raw linked panel
    assert S._change(poststratify(crop), weight="ps_weight")["perennial_change_pp"] == 9.9

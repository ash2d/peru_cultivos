"""Guards on the CENAGRO 2012 'before' observation.

The trap here is not silent data corruption but a **silent change of question**. PETT
recorded a land *state* and can say "EN DESCANSO"; CENAGRO question 024 asks which crop is
grown, so a fallow parcel contributes no row. Crossing the two raw makes `PASTURE_FALLOW`
appear to collapse 17.9 % → 1.1 % — a difference of instruments read as a change in land.
These tests hold the like-for-like restriction, and the class-priority collapse that turns a
farmer-level link into one row per parcel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crop_classifier.allperu import cenagro as C


def test_a_census_crop_list_is_classified_by_the_projects_own_lexicon():
    """Both sides must be classified by the same rules or part of the 'change' is a
    change of definition."""
    s = pd.Series(["LIMON ACIDO | MELON", "ARROZ", "MANGO", "MAIZ AMARILLO DURO", ""])
    got = C.classify_crop_list(s)
    assert got.iloc[0] == "PERENNIAL"      # lime wins over melon by group_priority
    assert got.iloc[1] == "ANNUAL"
    assert got.iloc[2] == "PERENNIAL"
    assert got.iloc[3] == "ANNUAL"
    assert pd.isna(got.iloc[4])   # a blank cell resolves to nothing, not to ANNUAL


def test_the_separator_is_the_pipe_the_census_export_uses():
    a = C.classify_crop_list(pd.Series(["ARROZ | MANGO"])).iloc[0]
    assert a == "PERENNIAL", "a multi-crop cell must not be treated as one unknown token"


def test_share_table_reports_change_in_percentage_points_and_a_ratio():
    df = pd.DataFrame({"b": ["PERENNIAL"] * 20 + ["ANNUAL"] * 80,
                       "a": ["PERENNIAL"] * 30 + ["ANNUAL"] * 70})
    t = C._share_table(df, "b", "a", classes=("PERENNIAL", "ANNUAL"))
    per = t[t["class"] == "PERENNIAL"].iloc[0]
    assert per["b_pct"] == 20.0 and per["a_pct"] == 30.0
    assert per["change_pp"] == 10.0
    assert per["ratio"] == 1.5


def test_the_paired_ci_uses_discordant_pairs_not_the_marginals():
    """A parcel that is PERENNIAL on both sides carries no information about the change;
    only the parcels that moved do. A CI built from the marginal shares would be too wide
    and would let a real change look like noise."""
    n = 1000
    # identical on both sides: zero discordant pairs -> zero-width interval
    same = pd.DataFrame({"b": ["PERENNIAL"] * n, "a": ["PERENNIAL"] * n})
    t = C._share_table(same, "b", "a", classes=("PERENNIAL",))
    assert t.iloc[0]["ci95_pp"] == 0.0
    assert t.iloc[0]["change_pp"] == 0.0


def test_shares_of_all_three_classes_sum_to_the_resolved_total():
    df = pd.DataFrame({"b": ["PERENNIAL", "ANNUAL", "PASTURE_FALLOW"] * 10,
                       "a": ["ANNUAL", "ANNUAL", "PERENNIAL"] * 10})
    t = C._share_table(df, "b", "a")
    # shares are rounded to 0.1 pp for reading, so allow that much slack
    assert t["b_pct"].sum() == pytest.approx(100.0, abs=0.2)
    assert t["a_pct"].sum() == pytest.approx(100.0, abs=0.2)
    assert t["change_pp"].sum() == pytest.approx(0.0, abs=0.2)


# --- the built tables, if they exist in this workspace ---
def test_the_like_for_like_table_drops_fallow_from_both_sides():
    f = C.OUT_DIR / "pett_to_cenagro_cropped_only.csv"
    if not f.exists():
        pytest.skip("cenagro tables not built in this workspace")
    t = pd.read_csv(f)
    assert set(t["class"]) == {"PERENNIAL", "ANNUAL"}, (
        "PASTURE_FALLOW must not appear: the census cannot record it, so including it "
        "measures the instrument, not the land")
    assert t["change_pp"].sum() == pytest.approx(0.0, abs=0.2)


def test_the_headline_change_is_smaller_than_the_naive_one():
    """Pinning the actual finding: crossing the two raw overstates the perennial shift."""
    naive = C.OUT_DIR / "pett_to_cenagro.csv"
    fair = C.OUT_DIR / "pett_to_cenagro_cropped_only.csv"
    if not (naive.exists() and fair.exists()):
        pytest.skip("cenagro tables not built in this workspace")
    n = pd.read_csv(naive).set_index("class").loc["PERENNIAL", "change_pp"]
    f = pd.read_csv(fair).set_index("class").loc["PERENNIAL", "change_pp"]
    assert f < n
    assert np.sign(f) == np.sign(n), "the sign should survive; only the size is inflated"

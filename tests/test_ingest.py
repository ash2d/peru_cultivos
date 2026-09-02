"""Unit tests for label ingest (docs/s2_labelling/plan.md).

The three things that would corrupt the labelled table without raising anything:

* a label outside the four-value scheme silently becoming a class of its own;
* a wrong kappa in a gate report, which is how a bad codebook gets signed off;
* an ``item_id`` resolving to more than one ``COD_PREDIO``, which attaches a human's
  judgement to the wrong parcel.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from crop_classifier.labelling.ingest import (
    ABSTAIN,
    CLASSES,
    CSV_COLUMNS,
    LABELS,
    agreement,
    cohens_kappa,
    ingest,
    read_csvs,
    resolve,
    transition_matrix,
)


def _csv(tmp_path, name, rows):
    f = tmp_path / name
    pd.DataFrame(rows).to_csv(f, index=False)
    return f


def _row(item, labeller, label, bm=False):
    """A returned CSV row. **No `confidence`** — the control was removed; see
    :class:`TestNoConfidenceField`."""
    return {"item_id": item, "labeller": labeller, "label": label,
            "crop_guess": "", "boundary_mismatch": bm, "seconds_spent": 90,
            "timestamp": "2026-08-12T10:00:00Z"}


class TestReadCsvs:
    def test_accepts_every_value_in_the_scheme(self, tmp_path):
        f = _csv(tmp_path, "a.csv", [_row(f"M{i:05d}", "A", lab)
                                     for i, lab in enumerate(LABELS)])
        assert len(read_csvs([f])) == len(LABELS) == 6

    def test_rejects_a_label_outside_the_scheme(self, tmp_path):
        f = _csv(tmp_path, "a.csv", [_row("M00001", "A", "PASTURE")])
        with pytest.raises(ValueError, match="outside the label scheme"):
            read_csvs([f])

    def test_rejects_a_csv_missing_the_key(self, tmp_path):
        f = tmp_path / "b.csv"
        pd.DataFrame({"label": ["ANNUAL"]}).to_csv(f, index=False)
        with pytest.raises(ValueError, match="missing required column"):
            read_csvs([f])

    def test_boundary_mismatch_parses_from_strings(self, tmp_path):
        f = _csv(tmp_path, "a.csv", [_row("M00001", "A", "ANNUAL", bm="True"),
                                     _row("M00002", "A", "ANNUAL", bm="False")])
        out = read_csvs([f]).set_index("item_id")["boundary_mismatch"]
        assert bool(out["M00001"]) and not bool(out["M00002"])


class TestKappa:
    def test_matches_a_hand_worked_example(self):
        """Classic 2x2: a=[20 yes,5 no...] -> po=0.70, pe=0.50, kappa=0.40."""
        a = ["Y"] * 25 + ["N"] * 25
        b = ["Y"] * 20 + ["N"] * 5 + ["Y"] * 10 + ["N"] * 15
        # po = (20 + 15)/50 = 0.70 ; pe = .5*.6 + .5*.4 = 0.50 ; k = .2/.5 = 0.40
        assert cohens_kappa(pd.Series(a), pd.Series(b)) == pytest.approx(0.40)

    def test_perfect_agreement_is_one(self):
        a = ["PERENNIAL", "ANNUAL", "OTHER", "WOODY_NON_CROP"] * 5
        assert cohens_kappa(pd.Series(a), pd.Series(a), LABELS) == pytest.approx(1.0)

    def test_reports_both_4way_and_perennial_vs_rest(self):
        """Both numbers are needed because they can diverge in either direction.

        Here the labellers agree perfectly on what is perennial and disagree only on OTHER vs
        WOODY_NON_CROP *inside* the rest — the 4-way number looks bad while the distinction
        the research question turns on is unanimous.
        """
        rows = ([_row(f"M{i:05d}", lab, "PERENNIAL")
                 for i in range(10) for lab in ("A", "B")]
                + [_row(f"M{i:05d}", "A", "OTHER") for i in range(10, 20)]
                + [_row(f"M{i:05d}", "B", "WOODY_NON_CROP") for i in range(10, 20)])
        out = agreement(pd.DataFrame(rows))
        assert out["n_overlap"] == 20
        assert out["kappa_called"] == pytest.approx(1 / 3)
        assert out["kappa_perennial"] == pytest.approx(1.0)

    def test_woody_confusion_can_also_sink_the_perennial_number(self):
        """The opposite case: one labeller calls woody cover a crop, the other does not."""
        rows = ([_row(f"M{i:05d}", "A", "PERENNIAL") for i in range(10)]
                + [_row(f"M{i:05d}", "B", "WOODY_NON_CROP") for i in range(10)]
                + [_row(f"M{i:05d}", lab, "ANNUAL")
                   for i in range(10, 20) for lab in ("A", "B")])
        out = agreement(pd.DataFrame(rows))
        assert out["kappa_perennial"] < out["kappa_called"]


class TestResolve:
    def test_agreement_collapses_to_one_row(self):
        df = pd.DataFrame([_row("M1", "A", "ANNUAL"), _row("M1", "B", "ANNUAL")])
        out = resolve(df)
        assert len(out) == 1 and out.resolution.iloc[0] == "agreed"

    def test_unadjudicated_disagreement_is_flagged_not_hidden(self):
        df = pd.DataFrame([_row("M1", "A", "ANNUAL"), _row("M1", "B", "PERENNIAL")])
        out = resolve(df)
        assert out.resolution.iloc[0] == "unresolved"

    def test_adjudication_wins(self):
        df = pd.DataFrame([_row("M1", "A", "ANNUAL"), _row("M1", "B", "PERENNIAL")])
        out = resolve(df, {"M1": "OTHER"})
        assert out.label.iloc[0] == "OTHER" and out.resolution.iloc[0] == "adjudicated"


class TestNoConfidenceField:
    """Confidence was a second, softer abstain beside UNSURE.

    Two ways to record doubt split the signal: the hard one the gates read, and a graded one
    nobody moved off its default. Removing it is safe only if nothing downstream still
    reaches for the column.
    """

    def test_resolve_emits_no_confidence_column(self):
        df = pd.DataFrame([_row("M1", "A", "ANNUAL"), _row("M1", "B", "ANNUAL")])
        assert "confidence" not in resolve(df).columns

    def test_it_is_not_in_the_expected_csv_schema(self):
        assert "confidence" not in CSV_COLUMNS

    def test_a_csv_that_still_carries_one_is_read_not_rejected(self, tmp_path):
        """Any CSV downloaded before the change must still ingest — the column is simply
        surplus, and failing on it would strand real work."""
        rows = [dict(_row(f"M{i}", "A", "ANNUAL"), confidence=1) for i in range(3)]
        out = read_csvs([_csv(tmp_path, "old.csv", rows)])
        assert len(out) == 3

    def test_the_gate_report_no_longer_carries_the_low_confidence_share(self, tmp_path):
        s = _sample()
        key = pd.DataFrame({"item_id": [f"M{i}" for i in range(6)],
                            "COD_PREDIO": [f"P{i}" for i in range(6)]})
        _csv(tmp_path, "a.csv", [_row(f"M{i}", "A", "ANNUAL") for i in range(6)])
        ingest(tmp_path, s, key, out_dir=tmp_path)
        import json
        g = json.loads((tmp_path / "kappa_report.json").read_text())["gates"]
        assert "low_confidence_share_among_called" not in g["G2_unsure_share"]
        assert set(g["G2_unsure_share"]) == {"value", "criterion", "pass"}


def _sample(n=6):
    return gpd.GeoDataFrame({
        "COD_PREDIO": [f"P{i}" for i in range(n)],
        "dept": ["PIURA"] * n,
        "declared_class": ["ANNUAL"] * (n // 2) + ["PERENNIAL"] * (n - n // 2),
        "weight": [100.0] * n, "area_ha": [1.0] * n,
        "imagery_date": ["2022-06-15"] * n, "imagery_res": ["1.2m"] * n,
        "stratum": ["PIURA|ANNUAL"] * n, "region_id": [f"r{i}" for i in range(n)],
        "split": ["trainval"] * n, "fold": [0] * n, "batch": ["main"] * n,
        "geometry": [Point(-80 + i * 0.01, -5) for i in range(n)],
    }, geometry="geometry", crs=4326)


class TestIngest:
    def test_item_id_must_resolve_to_exactly_one_parcel(self, tmp_path):
        s = _sample()
        key = pd.DataFrame({"item_id": ["M1", "M1"], "COD_PREDIO": ["P0", "P1"]})
        _csv(tmp_path, "a.csv", [_row("M1", "A", "ANNUAL")])
        with pytest.raises(ValueError, match="map to >1"):
            ingest(tmp_path, s, key, out_dir=tmp_path)

    def test_unknown_item_id_is_an_error_not_a_silent_drop(self, tmp_path):
        s = _sample()
        key = pd.DataFrame({"item_id": ["M1"], "COD_PREDIO": ["P0"]})
        _csv(tmp_path, "a.csv", [_row("M9", "A", "ANNUAL")])
        with pytest.raises(ValueError, match="not in item_key"):
            ingest(tmp_path, s, key, out_dir=tmp_path)

    def test_boundary_mismatch_is_excluded_but_kept_in_the_table(self, tmp_path):
        s = _sample()
        key = pd.DataFrame({"item_id": [f"M{i}" for i in range(6)],
                            "COD_PREDIO": [f"P{i}" for i in range(6)]})
        _csv(tmp_path, "a.csv",
             [_row(f"M{i}", "A", "ANNUAL", bm=(i == 0)) for i in range(6)])
        out = ingest(tmp_path, s, key, out_dir=tmp_path)
        assert len(out) == 6
        assert int((~out.usable).sum()) == 1
        assert out.loc[~out.usable, "exclude_reason"].iloc[0] == "boundary_mismatch"

    def test_label_is_stored_verbatim(self, tmp_path):
        """WOODY_NON_CROP must survive ingest; folding it in is a later decision."""
        s = _sample()
        key = pd.DataFrame({"item_id": [f"M{i}" for i in range(6)],
                            "COD_PREDIO": [f"P{i}" for i in range(6)]})
        _csv(tmp_path, "a.csv", [_row(f"M{i}", "A", "WOODY_NON_CROP")
                                 for i in range(6)])
        out = ingest(tmp_path, s, key, out_dir=tmp_path)
        assert set(out["label"]) == {"WOODY_NON_CROP"}


class TestUnsure:
    """UNSURE is an abstain: stored verbatim, never trained on, and it drives gate G2."""

    def test_unsure_is_kept_in_the_table_but_excluded_from_usable(self, tmp_path):
        s = _sample()
        key = pd.DataFrame({"item_id": [f"M{i}" for i in range(6)],
                            "COD_PREDIO": [f"P{i}" for i in range(6)]})
        _csv(tmp_path, "a.csv",
             [_row(f"M{i}", "A", "UNSURE" if i < 2 else "ANNUAL") for i in range(6)])
        out = ingest(tmp_path, s, key, out_dir=tmp_path)
        assert (out.label == "UNSURE").sum() == 2          # stored verbatim
        assert set(out.loc[out.label == "UNSURE", "exclude_reason"]) == {"unsure"}
        assert int(out.usable.sum()) == 4

    def test_one_abstain_plus_one_call_resolves_to_the_call(self):
        """Not a disagreement about land cover — throwing the call away wastes it."""
        df = pd.DataFrame([_row("M1", "A", "UNSURE"), _row("M1", "B", "PERENNIAL")])
        out = resolve(df)
        assert out.label.iloc[0] == "PERENNIAL"
        assert out.resolution.iloc[0] == "one_abstained"

    def test_two_different_calls_still_count_as_unresolved(self):
        df = pd.DataFrame([_row("M1", "A", "OTHER"), _row("M1", "B", "PERENNIAL")])
        assert resolve(df).resolution.iloc[0] == "unresolved"

    def test_kappa_called_ignores_parcels_either_labeller_abstained_on(self):
        """A codebook is not responsible for where someone chose to abstain."""
        # two classes among the called parcels, or kappa is undefined by construction
        rows = ([_row(f"M{i:05d}", lab, "PERENNIAL" if i < 5 else "ANNUAL")
                 for i in range(10) for lab in ("A", "B")]
                + [_row("M00099", "A", "UNSURE"), _row("M00099", "B", "ANNUAL")])
        out = agreement(pd.DataFrame(rows))
        assert out["n_both_called"] == 10
        assert out["kappa_called"] == pytest.approx(1.0)   # perfect on what was called
        assert out["one_abstained_other_did_not"] == 1

    def test_g2_reads_the_unsure_share(self, tmp_path):
        s = _sample()
        key = pd.DataFrame({"item_id": [f"M{i}" for i in range(6)],
                            "COD_PREDIO": [f"P{i}" for i in range(6)]})
        _csv(tmp_path, "a.csv",
             [_row(f"M{i}", "A", "UNSURE" if i < 3 else "ANNUAL") for i in range(6)])
        ingest(tmp_path, s, key, out_dir=tmp_path)
        import json
        g = json.loads((tmp_path / "kappa_report.json").read_text())["gates"]
        assert g["G2_unsure_share"]["value"] == pytest.approx(0.5)
        assert g["G2_unsure_share"]["pass"] is False

    def test_g3_does_not_demand_a_quota_of_abstains(self, tmp_path):
        """UNSURE is not a class, so it must not fail the >=150-per-class gate."""
        s = _sample()
        key = pd.DataFrame({"item_id": [f"M{i}" for i in range(6)],
                            "COD_PREDIO": [f"P{i}" for i in range(6)]})
        _csv(tmp_path, "a.csv", [_row(f"M{i}", "A", "ANNUAL") for i in range(6)])
        ingest(tmp_path, s, key, out_dir=tmp_path)
        import json
        g = json.loads((tmp_path / "kappa_report.json").read_text())["gates"]
        assert "UNSURE" not in g["G3_min_per_class"]["classes_below"]


class TestNonAgriculture:
    """The sixth value, carved out of `OTHER`.

    `OTHER` used to mean both "farmable land not currently cropped" (pasture, fallow, prepared
    ground) and "not farmland at all" (water, built-up, road). A fallow field can convert to
    a perennial and a road cannot, so pooling them puts a structurally impossible outcome in
    the class the project is measuring.
    """

    def test_it_is_a_real_class_not_an_abstain(self):
        assert "NON_AGRICULTURE" in CLASSES
        assert ABSTAIN == "UNSURE" and ABSTAIN not in CLASSES

    def test_it_survives_ingest_verbatim_and_is_usable(self, tmp_path):
        """Whether it trains as its own class or is pooled is a later modelling call."""
        s = _sample()
        key = pd.DataFrame({"item_id": [f"M{i}" for i in range(6)],
                            "COD_PREDIO": [f"P{i}" for i in range(6)]})
        _csv(tmp_path, "a.csv", [_row(f"M{i}", "A", "NON_AGRICULTURE")
                                 for i in range(6)])
        out = ingest(tmp_path, s, key, out_dir=tmp_path)
        assert set(out["label"]) == {"NON_AGRICULTURE"}
        assert bool(out.usable.all())

    def test_it_counts_toward_the_per_class_gate(self, tmp_path):
        """It is a class with a floor, unlike UNSURE — so a thin one must be visible."""
        s = _sample()
        key = pd.DataFrame({"item_id": [f"M{i}" for i in range(6)],
                            "COD_PREDIO": [f"P{i}" for i in range(6)]})
        _csv(tmp_path, "a.csv", [_row(f"M{i}", "A", "ANNUAL") for i in range(6)])
        ingest(tmp_path, s, key, out_dir=tmp_path)
        import json
        g = json.loads((tmp_path / "kappa_report.json").read_text())["gates"]
        assert "NON_AGRICULTURE" in g["G3_min_per_class"]["classes_below"]

    def test_disagreeing_with_other_is_a_real_disagreement(self):
        """If the codebook's boundary rule fails, this is the shape it fails in."""
        df = pd.DataFrame([_row("M1", "A", "OTHER"),
                           _row("M1", "B", "NON_AGRICULTURE")])
        assert resolve(df).resolution.iloc[0] == "unresolved"

    def test_the_split_can_lower_kappa_and_that_is_visible(self):
        """Two labellers who agree it is 'not a crop' but split 3-vs-6 must not read as
        agreement — that is the exact failure the boundary rule exists to prevent."""
        rows = ([_row(f"M{i:05d}", lab, "PERENNIAL")
                 for i in range(10) for lab in ("A", "B")]
                + [_row(f"M{i:05d}", "A", "OTHER") for i in range(10, 20)]
                + [_row(f"M{i:05d}", "B", "NON_AGRICULTURE") for i in range(10, 20)])
        out = agreement(pd.DataFrame(rows))
        assert out["kappa_called"] < 0.75          # would fail G1, correctly
        assert out["kappa_perennial"] == pytest.approx(1.0)


class TestDegenerateKappa:
    def test_a_single_class_overlap_gives_nan_and_therefore_FAILS_g1(self):
        """Kappa is undefined when only one class appears; it must not pass by accident."""
        import numpy as np
        rows = [_row(f"M{i:05d}", lab, "PERENNIAL")
                for i in range(10) for lab in ("A", "B")]
        out = agreement(pd.DataFrame(rows))
        assert np.isnan(out["kappa_called"])
        assert not (out["kappa_called"] >= 0.75)      # the gate expression itself


class TestTransitionMatrix:
    def test_rows_sum_to_one_and_are_weighted(self):
        lab = pd.DataFrame({
            "declared_class": ["ANNUAL"] * 4 + ["PERENNIAL"] * 4,
            "label": ["ANNUAL", "ANNUAL", "PERENNIAL", "OTHER"] + ["PERENNIAL"] * 4,
            "weight": [10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0],
            "usable": [True] * 8})
        t = transition_matrix(lab)
        assert t["share"].sum(axis=1).round(6).eq(1.0).all()
        assert t.loc["ANNUAL", ("share", "PERENNIAL")] == pytest.approx(0.25)

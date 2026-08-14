"""Unit tests for the Phase-7 gate logic (perennial/diagnostics.py).

Only the pure decision logic is exercised here — the pass/fail rules that decide whether a
trend may be produced at all. The El Nino arm fits LightGBM and is deliberately **not**
tested in-process: ``tests/test_pipeline.py`` imports torch, and on macOS torch and
lightgbm each bundle their own libomp, so loading both in one pytest process segfaults.
"""

import numpy as np
import pandas as pd

from crop_classifier.perennial import diagnostics as D


def _preds(rows):
    """Minimal panel_predictions frame: (COD_PREDIO, year, pred_label, pett_label)."""
    df = pd.DataFrame(rows, columns=["COD_PREDIO", "year", "pred_label", "pett_label"])
    df["abstained"] = df["pred_label"].isna()
    df["pred_proba"] = 0.8
    for c in ("ANNUAL", "PASTURE_FALLOW", "PERENNIAL"):
        df[f"prob_{c}"] = 1 / 3
    return df


class TestEras:
    def test_era_of_matches_panel_composition(self):
        assert D.era_of(1996) == "L5 only"
        assert D.era_of(1998) == "L5 only"
        assert D.era_of(1999) == "L5 + L7"
        assert D.era_of(2011) == "L5 + L7"
        assert D.era_of(2012) == "L7 only"
        assert D.era_of(2023) == "L7 only"

    def test_flag_years_are_the_measured_ones(self):
        assert set(D.FLAG_YEARS) == {1997, 2009, 2011, 2012}


class TestS4:
    def _tt(self, accs):
        return pd.DataFrame({"k": list(range(-5, 6)), "n": [100] * 11,
                             "accuracy": accs, "macro_f1": accs})

    def test_pass_when_within_tolerance(self, monkeypatch):
        acc = [0.5, 0.6, 0.72, 0.74, 0.75, 0.76, 0.75, 0.74, 0.72, 0.6, 0.5]
        monkeypatch.setattr(D.P, "temporal_transfer",
                            lambda *a, **k: self._tt(acc))
        _, v = D.s4_temporal_transfer(_preds([]))
        assert v["pass"] and v["accuracy_at_k0"] == 0.76
        # the |k|>3 tails are outside the criterion and must not fail it
        assert abs(v["max_abs_deviation_within_3"] - 0.04) < 1e-9

    def test_a_rise_above_k0_is_reported_separately_from_degradation(self, monkeypatch):
        acc = [0.5, 0.6, 0.7, 0.72, 0.74, 0.60, 0.74, 0.90, 0.7, 0.6, 0.5]
        monkeypatch.setattr(D.P, "temporal_transfer",
                            lambda *a, **k: self._tt(acc))
        _, v = D.s4_temporal_transfer(_preds([]))
        assert not v["pass"]                    # two-sided, as the criterion is written
        assert v["degradation_only_pass"]       # nothing actually got worse
        assert abs(v["max_degradation_within_3"]) < 1e-9

    def test_fail_on_a_cliff_within_k3(self, monkeypatch):
        acc = [0.5, 0.6, 0.7, 0.72, 0.74, 0.76, 0.74, 0.30, 0.7, 0.6, 0.5]
        monkeypatch.setattr(D.P, "temporal_transfer",
                            lambda *a, **k: self._tt(acc))
        _, v = D.s4_temporal_transfer(_preds([]))
        assert not v["pass"] and v["worst_k_within_3"] == 2

    def test_fail_when_panel_misses_the_label_year(self, monkeypatch):
        monkeypatch.setattr(D.P, "temporal_transfer",
                            lambda *a, **k: pd.DataFrame(columns=["k", "n", "accuracy",
                                                                  "macro_f1"]))
        _, v = D.s4_temporal_transfer(_preds([]))
        assert not v["pass"]


class TestS5:
    def _series_preds(self, per_parcel):
        rows = []
        for cod, (lab, seq) in per_parcel.items():
            for i, c in enumerate(seq):
                rows.append([cod, 2000 + i, c, lab])
        return _preds(rows)

    def test_stable_perennial_passes(self):
        stable = ["PERENNIAL"] * 10
        p = self._series_preds({f"p{i}": ("PERENNIAL", stable) for i in range(20)})
        _, v, _ = D.s5_flicker(p, ["ANNUAL", "PASTURE_FALLOW", "PERENNIAL"])
        assert v["pass"] and v["flicker_PERENNIAL"] == 0.0

    def test_alternating_perennial_fails(self):
        noisy = ["PERENNIAL", "ANNUAL"] * 5
        p = self._series_preds({f"p{i}": ("PERENNIAL", noisy) for i in range(20)})
        _, v, _ = D.s5_flicker(p, ["ANNUAL", "PASTURE_FALLOW", "PERENNIAL"])
        assert not v["pass"] and v["flicker_PERENNIAL"] == 1.0

    def test_threshold_is_15_percent_of_perennial_parcels(self):
        stable, noisy = ["PERENNIAL"] * 10, ["PERENNIAL", "ANNUAL"] * 5
        mix = {f"s{i}": ("PERENNIAL", stable) for i in range(90)}
        mix.update({f"n{i}": ("PERENNIAL", noisy) for i in range(10)})
        _, v, _ = D.s5_flicker(self._series_preds(mix),
                               ["ANNUAL", "PASTURE_FALLOW", "PERENNIAL"])
        assert abs(v["flicker_PERENNIAL"] - 0.10) < 1e-9 and v["pass"]


class TestEraSteps:
    def _drift(self, medians):
        rows = [{"year": y, "era": D.era_of(y), "feature": "NDVI_median", "n": 100,
                 "mean": m, "std": 0.1, "p10": m - 0.1, "median": m, "p90": m + 0.1,
                 "flagged_year": y in D.FLAG_YEARS}
                for y, m in medians.items()]
        return pd.DataFrame(rows)

    def test_flagged_years_are_excluded_from_era_statistics(self):
        # 1997 is a coverage failure, not radiometry — a wild value there must not move
        # the era means at all.
        base = {y: 0.5 for y in range(1996, 2012)}
        base.update({y: 0.5 for y in range(2012, 2024)})
        clean = D.era_steps(self._drift(base))
        base[1997] = 99.0
        assert np.allclose(clean["step"], D.era_steps(self._drift(base))["step"])

    def test_step_is_measured_against_within_era_movement(self):
        m = {y: 0.50 for y in range(1996, 1999)}
        m.update({y: 0.50 + 0.01 * (y % 2) for y in range(1999, 2012)})
        m.update({y: 0.80 for y in range(2012, 2024)})
        s = D.era_steps(self._drift(m)).set_index("boundary")
        late = [i for i in s.index if "2012" in i][0]
        assert s.loc[late, "step"] > 0.25
        assert s.loc[late, "step_over_yoy"] > 10       # far beyond year-to-year noise

"""Unit tests for the modelling pipeline (labels, splits, features, models, device)."""

import threading
import time

import numpy as np
import pandas as pd
import pytest
import torch

from crop_classifier.data import (
    FN_LGBM,
    META_FEATURES,
    Normalizer,
    class_weights,
    final_split,
    fold_split,
    make_flat,
    parse_year_spec,
    resolve_drop_features,
    restrict_years,
)
from crop_classifier.device import pick_device, seed_everything
from crop_classifier.features.assemble import _subsample_dates, _summaries
from crop_classifier.features.indices import BANDS, CHANNELS, add_indices, scale_sr
from crop_classifier.features.landsat_gee import (
    ChunkTimeout,
    _chunk_id,
    _max_gap,
    _retry,
    _run_chunk,
    missions_for_year,
)
from crop_classifier.labels import assign_raw_label, crop_set_key
from crop_classifier.models.ltae import LTAENet
from crop_classifier.models.psetae import PSETAENet
from crop_classifier.splits import buffer_exclusions

CFG = {"keep_categories": ["crop"], "landcover_classes": ["pasture", "fallow"],
       "drop_categories": ["land_prep", "unspecified"],
       "merge": {"CAFE+PLATANO": "CAFE"}}


class TestLabels:
    def _row(self, crops, cats):
        return pd.Series({"crops": crops, "crop_categories": cats})

    def test_single_crop(self):
        assert assign_raw_label(self._row(["ARROZ"], ["crop"]), CFG) == ("ARROZ", "single_crop")

    def test_landcover(self):
        assert assign_raw_label(self._row(["DESCANSO"], ["fallow"]), CFG)[0] == "FALLOW"
        assert assign_raw_label(self._row(["PASTO"], ["pasture"]), CFG)[0] == "PASTURE"

    def test_dropped_category(self):
        lab, reason = assign_raw_label(self._row(["ARADO"], ["land_prep"]), CFG)
        assert lab is None and reason.startswith("dropped_category")

    def test_merge_map(self):
        lab, reason = assign_raw_label(
            self._row(["CAFE", "PLATANO"], ["crop", "crop"]), CFG)
        assert (lab, reason) == ("CAFE", "merged")
        # order/duplicates don't matter for the key
        assert crop_set_key(["PLATANO", "CAFE", "CAFE"]) == "CAFE+PLATANO"

    def test_unmerged_multicrop_excluded(self):
        lab, reason = assign_raw_label(
            self._row(["ARROZ", "MAIZ"], ["crop", "crop"]), CFG)
        assert lab is None and reason == "multicrop_unmerged"


class TestSplits:
    def test_buffer_exclusions(self):
        xy = np.array([[0, 0], [100, 0], [5000, 0], [10000, 0]], dtype=float)
        held = np.array([True, False, False, False])
        cand = ~held
        excl = buffer_exclusions(xy, held, cand, buffer_m=1500)
        assert excl.tolist() == [False, True, False, False]

    def test_buffer_empty_masks(self):
        xy = np.zeros((3, 2))
        assert not buffer_exclusions(xy, np.zeros(3, bool), np.ones(3, bool), 100).any()


class TestFeatures:
    def test_max_gap(self):
        assert _max_gap(np.array([1] * 12)) == 0
        assert _max_gap(np.array([0] * 12)) == 12
        assert _max_gap(np.array([1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1])) == 3

    def test_missions_for_year(self):
        assert set(missions_for_year(1998)) == {"L5"}
        assert set(missions_for_year(2000)) == {"L5", "L7"}
        assert set(missions_for_year(2013)) == {"L5", "L7", "L8"}

    def test_indices_range(self):
        df = pd.DataFrame({b: np.random.randint(7000, 30000, 50) for b in BANDS})
        out = add_indices(scale_sr(df))
        assert set(CHANNELS) <= set(out.columns)
        assert out["NDVI"].between(-1.5, 2.5).all()

    def test_harmonic_recovers_phase(self):
        # a pure annual cosine peaking at doy 120 must be recovered by the fit
        doy = np.linspace(10, 360, 24)
        t = doy / 365.25
        peak = 120 / 365.25
        y = 0.5 + 0.3 * np.cos(2 * np.pi * (t - peak))
        s = _summaries(doy, y)
        fitted_peak = np.arctan2(s["h_sin"], s["h_cos"]) / (2 * np.pi) * 365.25
        assert abs(fitted_peak - 120) < 3
        assert abs(s["h_mean"] - 0.5) < 0.02

    def test_summaries_sparse_is_nan(self):
        s = _summaries(np.array([10.0, 50.0]), np.array([0.1, 0.2]))
        assert np.isnan(s["h_cos"]) and np.isnan(s["slope"])
        assert not np.isnan(s["median"])

    def test_subsample_dates(self):
        assert len(_subsample_dates(np.arange(100), 64)) <= 64
        assert (_subsample_dates(np.arange(10), 64) == np.arange(10)).all()


def _raise_value_error():
    raise ValueError("some genuine bug")


class TestGeeFaultTolerance:
    def test_retry_recovers_from_transient(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("Connection reset by peer")
            return 42

        assert _retry(flaky, tries=5, base_wait=0.001) == 42
        assert calls["n"] == 3

    def test_deadline_turns_a_silent_hang_into_a_retry(self):
        """The measured failure this exists for: a GEE call that never returns and never
        raises. Before the deadline, `_retry` had nothing to catch and the run stalled —
        once for 13.4 h. The first call hangs; the deadline must abandon it and retry."""
        calls = {"n": 0}

        def hangs_once():
            calls["n"] += 1
            if calls["n"] == 1:
                time.sleep(30)          # never reached to completion: deadline fires first
            return "ok"

        assert _retry(hangs_once, tries=3, base_wait=0.001, deadline_s=0.2) == "ok"
        assert calls["n"] == 2

    def test_deadline_gives_up_after_all_tries(self):
        with pytest.raises(ChunkTimeout):
            _retry(lambda: time.sleep(30), tries=2, base_wait=0.001, deadline_s=0.1)

    def test_deadline_can_be_disabled(self):
        assert _retry(lambda: 7, tries=1, deadline_s=None) == 7

    def test_abandoned_thread_is_daemon_so_the_process_can_exit(self):
        """Regression: the timeout must not strand the interpreter at shutdown.

        The first implementation used a ``ThreadPoolExecutor``. Its threads are non-daemon
        and ``concurrent.futures.thread`` joins them via ``atexit``, so a thread parked in a
        hung GEE read kept the process alive *after all its work was done* — measured at up
        to 3 h across all five national panel workers on 2026-08-08. A daemon thread is
        abandoned at exit instead.
        """
        seen: list[threading.Thread] = []

        def hangs():
            seen.append(threading.current_thread())
            time.sleep(30)

        with pytest.raises(ChunkTimeout):
            _retry(hangs, tries=1, base_wait=0.001, deadline_s=0.2)

        assert seen and all(t.daemon for t in seen), "abandoned GEE thread must be a daemon"
        assert not any(
            type(t).__module__.startswith("concurrent.futures")
            for t in threading.enumerate()
        ), "no futures worker may outlive the call — atexit would join it"

    def test_deadline_propagates_the_real_exception(self):
        """A genuine error inside the worker thread must surface unchanged, not as a
        timeout — otherwise real bugs would be misclassified as transient and retried."""
        with pytest.raises(ValueError, match="some genuine bug"):
            _retry(_raise_value_error, tries=3, base_wait=0.001, deadline_s=5)

    def test_retry_raises_split_error_immediately(self):
        calls = {"n": 0}

        def too_big():
            calls["n"] += 1
            raise RuntimeError("Computation timed out.")

        with pytest.raises(RuntimeError):
            _retry(too_big, tries=5, base_wait=0.001)
        assert calls["n"] == 1  # no pointless retries on a deterministic error

    def test_retry_raises_unknown_error_immediately(self):
        calls = {"n": 0}

        def broken():
            calls["n"] += 1
            raise ValueError("some genuine bug")

        with pytest.raises(ValueError):
            _retry(broken, tries=5, base_wait=0.001)
        assert calls["n"] == 1

    def test_run_chunk_splits_oversized(self):
        """A 'too big' chunk is recursively halved until the server accepts it."""
        def fn(chunk, year):
            if len(chunk) > 6:
                raise RuntimeError("User memory limit exceeded.")
            return pd.DataFrame({"COD_PREDIO": chunk["COD_PREDIO"].values, "year": year})

        chunk = pd.DataFrame({"COD_PREDIO": [f"p{i}" for i in range(20)]})
        out = _run_chunk(fn, chunk, 1999, min_size=2)
        assert len(out) == 20  # every parcel survives the splitting
        assert set(out["COD_PREDIO"]) == set(chunk["COD_PREDIO"])

    def test_chunk_id_is_order_invariant(self):
        a = pd.DataFrame({"COD_PREDIO": ["x", "y", "z"]})
        b = pd.DataFrame({"COD_PREDIO": ["z", "x", "y"]})
        c = pd.DataFrame({"COD_PREDIO": ["x", "y"]})
        assert _chunk_id(a) == _chunk_id(b)
        assert _chunk_id(a) != _chunk_id(c)


class TestModels:
    def test_ltae_forward_shapes_and_mask(self):
        seed_everything(0)
        net = LTAENet(n_channels=11, n_classes=5)
        x = torch.randn(4, 20, 11)
        doy = torch.randint(1, 366, (4, 20)).float()
        mask = torch.ones(4, 20, dtype=torch.bool)
        mask[0, 5:] = False   # sparse parcel
        mask[1, :] = False    # fully-masked parcel must not produce NaN
        out = net(x, doy, mask)
        assert out.shape == (4, 5)
        assert torch.isfinite(out).all()

    def test_psetae_forward_shapes(self):
        seed_everything(0)
        net = PSETAENet(n_channels=11, n_classes=5)
        x = torch.randn(3, 12, 8, 11)
        doy = torch.randint(1, 366, (3, 12)).float()
        mask = torch.ones(3, 12, dtype=torch.bool)
        pixmask = torch.rand(3, 12, 8) > 0.5
        out = net(x, doy, mask, pixmask)
        assert out.shape == (3, 5)
        assert torch.isfinite(out).all()

    def test_psetae_gradients_finite_single_pixel_dates(self):
        """Regression: single-pixel dates give var=0; sqrt'(0)=inf -> NaN grads unless
        eps is inside the sqrt (the pilot-smoke bug)."""
        seed_everything(0)
        net = PSETAENet(n_channels=11, n_classes=5)
        x = torch.randn(4, 10, 8, 11)
        doy = torch.randint(1, 366, (4, 10)).float()
        mask = torch.ones(4, 10, dtype=torch.bool)
        pixmask = torch.zeros(4, 10, 8, dtype=torch.bool)
        pixmask[:, :, 0] = True        # exactly one valid pixel per date -> var == 0
        loss = torch.nn.functional.cross_entropy(
            net(x, doy, mask, pixmask), torch.randint(0, 5, (4,)))
        loss.backward()
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        assert all(torch.isfinite(g).all() for g in grads)

    def test_mask_actually_masks_ltae(self):
        """Changing values at masked positions must not change the output."""
        seed_everything(0)
        net = LTAENet(n_channels=11, n_classes=5).eval()
        x = torch.randn(1, 10, 11)
        doy = torch.arange(1, 11).float().unsqueeze(0)
        mask = torch.ones(1, 10, dtype=torch.bool)
        mask[0, 7:] = False
        x2 = x.clone()
        x2[0, 7:] = 999.0
        with torch.no_grad():
            assert torch.allclose(net(x, doy, mask), net(x2, doy, mask), atol=1e-5)


# LightGBM training must run in a torch-free process (torch + lightgbm each bundle a
# libomp that segfaults when co-loaded on macOS). pytest imports torch for the model
# tests, so we exercise lgb.train in a subprocess with a clean interpreter.
_LGBM_SUBPROC = """
import numpy as np, pandas as pd
from crop_classifier.data import FlatData, class_weights
from crop_classifier.models.trees import LightGBMModel
rng = np.random.default_rng(0)
n_classes = 6
ytr = rng.integers(0, 5, 120)                       # class 5 absent from train
Xtr = pd.DataFrame(rng.standard_normal((120, 8)), columns=[f"f{i}" for i in range(8)])
yva = rng.integers(0, 6, 40)                        # val introduces class 5
Xva = pd.DataFrame(rng.standard_normal((40, 8)), columns=Xtr.columns)
tr = FlatData(Xtr, ytr, np.arange(120), list(Xtr.columns))
va = FlatData(Xva, yva, np.arange(40), list(Xva.columns))
m = LightGBMModel(n_estimators=20); m.set_n_classes(n_classes)
import tempfile, pathlib
m.fit(tr, va, class_weights(ytr, n_classes), pathlib.Path(tempfile.mkdtemp()))
p = m.predict_proba(va)
assert p.shape == (40, n_classes), p.shape        # full width despite absent class 5
assert np.allclose(p.sum(1), 1.0, atol=1e-5)
import sys
assert "torch" not in sys.modules                 # the flat path stays torch-free
print("OK")
"""


class TestLightGBM:
    def test_full_class_space_and_torch_free(self):
        import subprocess
        import sys
        r = subprocess.run([sys.executable, "-c", _LGBM_SUBPROC],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        assert "OK" in r.stdout

    def test_input_kind_without_import(self):
        from crop_classifier.models.base import model_input_kind
        assert model_input_kind("lightgbm") == "flat"
        assert model_input_kind("psetae") == "pixelset"


class TestFeatureExclusion:
    """RESULTS.md §4.6 — acquisition metadata must be withholdable, and must *stay*
    withheld at inference (the panel store still contains the columns)."""

    def _store(self, tmp_path):
        cod = ["a", "b", "c"]
        pd.DataFrame({"COD_PREDIO": cod, "NDVI_median": [0.1, 0.2, 0.3],
                      "frac_l7": [0.0, 0.5, 1.0], "n_valid_obs": [4, 8, 9],
                      "n_dates": [4, 8, 9], "max_gap": [30, 20, 10],
                      "n_valid_pixels": [5, 6, 7], "centroid_lat": [-5.1, -5.2, -5.3]}
                     ).to_parquet(tmp_path / FN_LGBM, index=False)
        parcels = pd.DataFrame({"COD_PREDIO": cod, "label_id": [0, 1, 2]})
        return parcels, parcels.index

    def test_alias_resolution(self):
        assert resolve_drop_features("meta") == META_FEATURES
        assert resolve_drop_features("meta,location") == META_FEATURES + ["centroid_lat"]
        assert resolve_drop_features(["frac_l7", "frac_l7"]) == ["frac_l7"]
        assert resolve_drop_features(None) == []

    def test_drop_removes_columns(self, tmp_path):
        parcels, idx = self._store(tmp_path)
        full = make_flat(parcels, idx, feat_dir=tmp_path)
        assert "frac_l7" in full.feature_names and "centroid_lat" in full.feature_names

        nometa = make_flat(parcels, idx, feat_dir=tmp_path, drop_features="meta")
        assert not set(META_FEATURES) & set(nometa.feature_names)
        assert "centroid_lat" in nometa.feature_names   # location kept unless asked

        nolat = make_flat(parcels, idx, feat_dir=tmp_path, drop_features="meta,location")
        assert nolat.feature_names == ["NDVI_median"]

    def test_feature_names_pin_survives_a_store_that_still_has_them(self, tmp_path):
        """The inference contract: an ablated model scores on its own column list even
        though the panel feature store carries every original column."""
        parcels, idx = self._store(tmp_path)
        fitted = make_flat(parcels, idx, feat_dir=tmp_path, drop_features="meta")
        scored = make_flat(parcels, idx, feat_dir=tmp_path,
                           feature_names=fitted.feature_names)
        assert list(scored.X.columns) == fitted.feature_names
        assert "frac_l7" not in scored.X.columns

    def test_missing_model_feature_raises(self, tmp_path):
        parcels, idx = self._store(tmp_path)
        with pytest.raises(KeyError):
            make_flat(parcels, idx, feat_dir=tmp_path,
                      feature_names=["NDVI_median", "not_a_column"])


class TestTrainYears:
    """``--train-years`` must subset the TRAIN cohort and leave val/test alone
    (RESULTS.md §8.1 — the >=1999 no-El-Nino retrain must stay CV-comparable)."""

    def _parcels(self):
        # 8 parcels: folds 0/1 in trainval, 2 test rows; label years straddle 1999.
        return pd.DataFrame({
            "COD_PREDIO": list("abcdefgh"),
            "year": [1998, 1999, 1998, 2000, 1998, 1999, 1998, 2001],
            "split": ["trainval"] * 6 + ["test"] * 2,
            "fold": [0, 0, 1, 1, 1, 1, -1, -1],
            "buffer_excl_fold0": [False] * 8,
            "buffer_excl_fold1": [False] * 8,
            "buffer_excl_test": [False] * 8,
        })

    def test_parse_year_spec(self):
        assert parse_year_spec("1999-2001") == [1999, 2000, 2001]
        assert parse_year_spec("1999,2005") == [1999, 2005]
        assert parse_year_spec("1999-2000,2005") == [1999, 2000, 2005]
        assert parse_year_spec(None) is None and parse_year_spec("") is None

    def test_restricts_train_only_in_cv(self):
        df = self._parcels()
        tr, va = fold_split(df, 0)
        tr_r = restrict_years(df, tr, parse_year_spec("1999-2023"))
        # train side: c,e drop (1998); d,f stay
        assert sorted(df.loc[tr, "COD_PREDIO"]) == ["c", "d", "e", "f"]
        assert sorted(df.loc[tr_r, "COD_PREDIO"]) == ["d", "f"]
        # val side untouched, 1998 rows included
        assert sorted(df.loc[va, "COD_PREDIO"]) == ["a", "b"]

    def test_restricts_final_refit_not_test(self):
        df = self._parcels()
        tr, te = final_split(df)
        tr_r = restrict_years(df, tr, parse_year_spec("1999-2023"))
        assert sorted(df.loc[tr_r, "COD_PREDIO"]) == ["b", "d", "f"]
        assert sorted(df.loc[te, "COD_PREDIO"]) == ["g", "h"]   # 1998 test row kept

    def test_none_spec_is_identity(self):
        df = self._parcels()
        tr, _ = fold_split(df, 0)
        assert restrict_years(df, tr, None).equals(tr)

    def test_missing_year_is_dropped(self):
        df = self._parcels()
        df.loc[3, "year"] = np.nan
        tr, _ = fold_split(df, 0)
        tr_r = restrict_years(df, tr, [1999, 2000])
        assert sorted(df.loc[tr_r, "COD_PREDIO"]) == ["f"]


class TestData:
    def test_class_weights_missing_class(self):
        w = class_weights(np.array([0, 0, 1]), n_classes=3)
        assert w[2] == 0.0 and w[1] > w[0] > 0

    def test_normalizer_roundtrip(self):
        X = np.random.randn(10, 6, 11).astype(np.float32) * 5 + 3
        valid = np.random.rand(10, 6) > 0.3
        n = Normalizer().fit(X, valid)
        Z = n.apply(X)
        assert abs(Z[valid].mean()) < 0.1
        n2 = Normalizer.from_state(n.state())
        assert np.allclose(n2.apply(X), Z)


class TestDevice:
    def test_pick_device_returns_valid(self):
        d = pick_device()
        assert d.type in ("cuda", "mps", "cpu")

    def test_prefer_cpu(self):
        assert pick_device("cpu").type == "cpu"

    def test_prefer_unavailable_raises(self):
        if not torch.cuda.is_available():
            with pytest.raises(RuntimeError):
                pick_device("cuda")

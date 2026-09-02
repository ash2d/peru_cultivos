"""Tests for the observation-density work (docs/RESULTS.md §6).

Every test pins a **decision**, not an implementation:

* degradation removes whole *acquisition dates*, never observations within a date, and never
  invents observations a parcel did not have;
* the target date count is **rank-matched** to the endpoint distribution, because an i.i.d.
  draw plus the unavoidable ``min(target, have)`` truncation lands well below the endpoint —
  the plan asks to match the distribution, not the median;
* the SLC-off stripe is a *contiguous band of pixel columns* within one acquisition, and its
  default width is the **measured** within-parcel loss, not the nominal scene-level 22 %;
* degraded rows are appended to the **train** side only;
* the density-conditional temperature composes with an already-applied scalar, so a panel can
  be recalibrated post hoc without re-inference;
* the within-parcel fixed-effect slope with clustered SEs matches statsmodels exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crop_classifier.allperu import density as D
from crop_classifier.data import DROP_SETS, ORDER_FEATURES, resolve_drop_features


# --- helpers ---
def make_px(n_parcels: int = 4, n_dates: int = 20, n_px: int = 10,
            seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_parcels):
        for d in range(n_dates):
            for k in range(n_px):
                rows.append({"COD_PREDIO": f"p{p}", "year": 1999,
                             "doy": 5 + d * 17, "mission": 5,
                             "lon": -80.0 + k * 0.0003, "lat": -5.0,
                             **{b: rng.uniform(0.05, 0.4)
                                for b in ("B", "G", "R", "NIR", "SWIR1", "SWIR2")}})
    return pd.DataFrame(rows)


# --- degradation ---
def test_degradation_removes_whole_dates_not_observations_within_a_date():
    """An acquisition either happened and was clear or it did not. Thinning a date's pixels
    would model cloud, not archive depth — only the SLC-off stripe may do that."""
    px = make_px()
    rng = np.random.default_rng(0)
    deg = D.degrade_pixels(px, {"p0": 5, "p1": 5, "p2": 5, "p3": 5}, rng,
                           slc_gap_frac=0.0)
    per_parcel = deg.groupby("COD_PREDIO")["doy"].nunique()
    assert (per_parcel == 5).all()
    # every surviving date keeps all of its pixels when no stripe is applied
    counts = deg.groupby(["COD_PREDIO", "doy"]).size()
    assert (counts == 10).all()


def test_degradation_never_invents_observations():
    px = make_px(n_dates=6)
    rng = np.random.default_rng(0)
    deg = D.degrade_pixels(px, {f"p{i}": 50 for i in range(4)}, rng, slc_gap_frac=0.0)
    assert deg.groupby("COD_PREDIO")["doy"].nunique().eq(6).all()
    assert len(deg) == len(px)


def test_slc_stripe_is_a_contiguous_band_of_pixel_columns():
    """SLC-off gaps are wedges; over one parcel they are locally parallel, so what is lost is
    a contiguous run of pixel columns, not a random scatter."""
    px = make_px(n_parcels=1, n_dates=1, n_px=20)
    rng = np.random.default_rng(3)
    deg = D.degrade_pixels(px, {"p0": 1}, rng, slc_gap_frac=0.25)
    kept = np.sort(deg["lon"].to_numpy())
    all_lon = np.sort(px["lon"].unique())
    missing = np.setdiff1d(all_lon, kept)
    assert len(missing) == 5                       # round(0.25 * 20)
    pos = np.searchsorted(all_lon, missing)
    assert (np.diff(pos) == 1).all()               # contiguous


def test_slc_default_is_the_measured_within_parcel_loss_not_the_scene_figure():
    """The nominal 22 % is a *scene* statistic. Measured on the national panel the mean
    per-date pixel count relative to a parcel's best date falls 0.968 -> 0.861, i.e. ~11 pp.
    A default of 0.22 would over-degrade training by a factor of two."""
    assert D.SLC_OFF_FRAC == pytest.approx(0.11)
    assert D.SLC_OFF_SCENE_FRAC == pytest.approx(0.22)
    assert D.SLC_OFF_FRAC < D.SLC_OFF_SCENE_FRAC


def test_targets_are_rank_matched_so_the_degraded_marginal_matches_the_endpoint():
    """The decision: rank-match, don't i.i.d.-draw. Because degradation can only remove
    dates, an i.i.d. pairing sends rich targets to poor parcels where they are wasted and the
    realised distribution falls short of the endpoint."""
    rng = np.random.default_rng(0)
    current = np.arange(5, 45, dtype=float)                   # training: 5..44 dates
    pool = np.arange(3, 25, dtype=float)                      # endpoint: 3..24
    cods = np.array([f"p{i}" for i in range(len(current))])

    ranked = D.draw_targets(cods, pool, rng, current=current)
    iid = D.draw_targets(cods, pool, rng)
    realised_rank = np.array([min(ranked[c], int(v)) for c, v in zip(cods, current)])
    realised_iid = np.array([min(iid[c], int(v)) for c, v in zip(cods, current)])

    assert abs(realised_rank.mean() - pool.mean()) < abs(realised_iid.mean() - pool.mean())
    # and the rank-matched assignment is monotone in the parcel's own density
    vals = np.array([ranked[c] for c in cods])
    assert (np.diff(vals) >= 0).all()


# --- the 1a audit ---
def test_fe_cluster_slopes_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(0)
    n = 1500
    g = rng.integers(0, 200, n)
    x = rng.normal(size=n) + g * 0.01
    y = 0.3 * x + g * 0.02 + rng.normal(size=n)
    beta, se, _ = D._fe_cluster_slopes(y[:, None], x, g)
    d = pd.DataFrame({"y": y, "x": x, "g": g})
    xd = d.groupby("g")["x"].transform(lambda s: s - s.mean())
    yd = d.groupby("g")["y"].transform(lambda s: s - s.mean())
    r = sm.OLS(yd, sm.add_constant(xd)).fit(cov_type="cluster",
                                            cov_kwds={"groups": pd.factorize(g)[0]})
    assert beta[0] == pytest.approx(r.params.iloc[1], rel=1e-9)
    assert se[0] == pytest.approx(r.bse.iloc[1], rel=1e-9)


def test_feature_family_separates_order_statistics_from_fitted_ones():
    """The audit's whole point: a max over 24 draws is a different estimator from a max over
    13, while a harmonic coefficient is not."""
    assert D.feature_family("NDVI_max") == "order_extreme"
    assert D.feature_family("NDVI_amp") == "order_extreme"
    assert D.feature_family("NDVI_p25") == "order_quantile"
    assert D.feature_family("NDVI_median") == "central"
    assert D.feature_family("NDVI_h_cos") == "fitted"
    assert D.feature_family("centroid_lat") == "static_or_meta"


def test_order_drop_set_covers_every_channel_extreme_and_nothing_else():
    drop = set(resolve_drop_features("order"))
    assert "NDVI_max" in drop and "BSI_amp" in drop and "SWIR2_min" in drop
    assert "NDVI_median" not in drop and "NDVI_p25" not in drop
    assert len(ORDER_FEATURES) == 11 * 3
    assert DROP_SETS["order"] is ORDER_FEATURES


# --- 1c: density-conditional temperature ---
def test_density_temperature_is_monotone_in_log_n():
    p = np.array([[0.7, 0.2, 0.1]] * 4)
    n = np.array([4.0, 10.0, 20.0, 40.0])
    params = {"intercept": 0.5, "slope": 0.4}
    out = D.apply_density_temperature(p, n, params)
    # larger T softens: the top class falls monotonically as n (hence T) grows
    assert (np.diff(out[:, 0]) < 0).all()


def test_recalibration_composes_with_an_already_applied_scalar():
    """Temperature scaling composes, which is what lets a panel be recalibrated post hoc
    instead of re-inferred: apply(apply(p, T1), T2) == apply(p, T1*T2)."""
    from crop_classifier.perennial.calibration import apply_temperature

    rng = np.random.default_rng(0)
    raw = rng.dirichlet(np.ones(3), size=50)
    base = 1.3
    already = apply_temperature(raw, base)
    preds = pd.DataFrame({"COD_PREDIO": [f"p{i}" for i in range(50)],
                          "n_valid_obs": rng.integers(5, 30, 50).astype(float)})
    for j, c in enumerate(["ANNUAL", "PASTURE_FALLOW", "PERENNIAL"]):
        preds[f"prob_{c}"] = already[:, j]
    params = {"intercept": 0.8, "slope": 0.3}
    out = D.recalibrate_predictions(preds, params, base_temperature=base)

    want = D.apply_density_temperature(raw, preds["n_valid_obs"].to_numpy(), params)
    got = out[["prob_ANNUAL", "prob_PASTURE_FALLOW", "prob_PERENNIAL"]].to_numpy()
    assert np.allclose(got, want, atol=1e-10)


def test_recalibration_updates_pred_label_to_match_the_new_probabilities():
    preds = pd.DataFrame({
        "COD_PREDIO": ["a"], "n_valid_obs": [10.0],
        "prob_ANNUAL": [0.4], "prob_PASTURE_FALLOW": [0.1], "prob_PERENNIAL": [0.5],
        "pred_label": ["PERENNIAL"]})
    out = D.recalibrate_predictions(preds, {"intercept": 1.0, "slope": 0.0})
    # a pure scalar temperature cannot change an argmax — the label must be preserved
    assert out.loc[0, "pred_label"] == "PERENNIAL"


# --- augmentation plumbing ---
def test_augmented_features_are_appended_not_merged(tmp_path):
    """A degraded copy must add a *row* for the parcel, not overwrite or widen it: the
    training signal is the same label seen at two densities."""
    from crop_classifier.data import make_flat

    feat_dir = tmp_path / "feat"
    feat_dir.mkdir()
    base = pd.DataFrame({"COD_PREDIO": ["a", "b"], "NDVI_median": [0.5, 0.6],
                         "n_dates": [20, 22]})
    base.to_parquet(feat_dir / "features_lightgbm.parquet", index=False)
    deg = pd.DataFrame({"COD_PREDIO": ["a", "b"], "NDVI_median": [0.45, 0.55],
                        "n_dates": [12, 12]})
    deg_path = tmp_path / "deg.parquet"
    deg.to_parquet(deg_path, index=False)

    parcels = pd.DataFrame({"COD_PREDIO": ["a", "b"], "label_id": [0, 1]})
    plain = make_flat(parcels, parcels.index, feat_dir=feat_dir)
    aug = make_flat(parcels, parcels.index, feat_dir=feat_dir, augment_feat=deg_path)
    assert len(plain.y) == 2 and len(aug.y) == 4
    assert list(aug.feature_names) == list(plain.feature_names)
    assert sorted(aug.y) == [0, 0, 1, 1]

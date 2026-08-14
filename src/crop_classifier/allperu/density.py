"""Observation-density robustness — the temporal-OOD work (docs/RESULTS.md §6).

The measured problem (RESULTS.md §8.3): clear Landsat observations per parcel-year fall from
~24 (W04) to ~13 (W19) as L5 retires and L7 goes SLC-off, and within parcel the predicted
probability tracks that density (+0.052 per log-observation on PETT-``PERENNIAL`` parcels,
−0.022 on PETT-``ANNUAL`` ones). Both classes revert toward the base rate as evidence thins,
which at the level of a share is indistinguishable from real land-use change.

Nothing here touches Earth Engine. The panel pixel store keeps **per-date observations**, so
every experiment in this module is re-assembly plus retraining:

* :func:`feature_density_audit` (1a) — regress every LightGBM feature on
  ``log(n_valid_obs)`` with a parcel fixed effect, cluster SEs by parcel, rank by the
  within-parcel standardised slope. Generalises ``windows.density_confound_test``, which did
  the same thing for the single quantity ``prob_PERENNIAL``.
* :func:`degrade_pixels` / :func:`build_degraded_features` (1b) — subsample a pixel store's
  acquisition **dates** down to the endpoint density and, optionally, punch SLC-off-shaped
  stripes through the surviving pixels, then assemble the LightGBM feature table from the
  degraded copy. Used both to augment training and to build the calibration ladder.
* :func:`fit_density_temperature` / :func:`apply_density_temperature` (1c) — temperature as
  a monotone function of ``log n`` instead of one scalar.

**Design note.** ``n_valid_obs`` is a *coverage-stage* count (GEE's clear-observation count)
while ``n_dates`` is what the pixel store actually holds. Degradation operates on dates —
the thing that can be removed — and the target distribution is therefore drawn from the
endpoint years' ``n_dates``. The two track each other closely (nationally 12.6 vs 13.6 in
2019) but they are not the same number and must not be swapped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from crop_classifier.features import assemble as asm
from crop_classifier.paths import feat, proc

# Endpoint = the last window (W19). Its observation density is what training has to survive.
ENDPOINT_YEARS = (2019, 2020, 2021, 2022, 2023)

# Landsat 7's scan-line corrector failed 2003-05-31 and ~22 % of every ETM+ *scene* is lost
# to wedge-shaped gaps. **A parcel does not lose 22 %.** Measured on the national panel, the
# mean per-date pixel count relative to a parcel's own best date falls from 0.968 (1999-2002)
# to 0.861 (2019-23) — an ~11 pp within-parcel loss, half the scene-level figure, because the
# gaps widen toward the swath edge and a 0.5 ha parcel samples one place in that gradient.
# The default is therefore the measured number, not the literature one.
SLC_OFF_FRAC = 0.11
SLC_OFF_SCENE_FRAC = 0.22   # the nominal scene-level figure, kept for reference only

FN_DEGRADED = "features_lightgbm_degraded.parquet"


# ------------------------------------------------------------------------------------
# 1a — feature density-sensitivity audit
# ------------------------------------------------------------------------------------
def _fe_cluster_slopes(y: np.ndarray, x: np.ndarray, gid: np.ndarray) -> tuple:
    """Univariate within-group slope of ``y`` on ``x`` with cluster-robust SEs.

    Vectorised over the columns of ``y`` so 135 features cost one pass instead of 135
    statsmodels fits. Equivalent to ``OLS(y_demeaned ~ x_demeaned).fit(cov_type='cluster')``
    including its small-sample correction — pinned against statsmodels in the tests.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    codes, _ = pd.factorize(gid)
    n_g = codes.max() + 1
    cnt = np.bincount(codes, minlength=n_g).astype(float)
    xbar = np.bincount(codes, weights=x, minlength=n_g) / cnt
    xd = x - xbar[codes]
    sxx = float((xd ** 2).sum())
    if sxx <= 0:
        nan = np.full(y.shape[1], np.nan)
        return nan, nan, nan
    ybar = np.stack([np.bincount(codes, weights=y[:, j], minlength=n_g) / cnt
                     for j in range(y.shape[1])], axis=1)
    yd = y - ybar[codes]
    beta = (xd[:, None] * yd).sum(0) / sxx
    resid = yd - xd[:, None] * beta
    score = xd[:, None] * resid
    gsum = np.stack([np.bincount(codes, weights=score[:, j], minlength=n_g)
                     for j in range(y.shape[1])], axis=1)
    meat = (gsum ** 2).sum(0)
    n, k = len(x), 2                       # intercept + slope, after absorbing the FE
    corr = (n_g / max(n_g - 1, 1)) * ((n - 1) / max(n - k, 1))
    var = corr * meat / sxx ** 2
    se = np.sqrt(var)
    return beta, se, yd.std(0)


def _panel_feature_frame(feat_dir: Path | None = None,
                         years: tuple[int, ...] | None = None) -> pd.DataFrame:
    """Stack the per-year panel LightGBM bundles into one parcel-year table."""
    feat_dir = feat_dir or (feat() / "panel")
    yrs = years or tuple(range(1999, 2024))
    frames = []
    for y in yrs:
        p = Path(feat_dir) / str(y) / asm.FN_LGBM
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        d["year"] = y
        frames.append(d)
    if not frames:
        raise FileNotFoundError(f"no assembled panel bundles under {feat_dir}")
    return pd.concat(frames, ignore_index=True)


# Feature families. Order statistics are the ones a shrinking sample changes *by
# construction*: a max over 24 draws is a different estimator from a max over 13, even if
# the underlying series is identical. Central and fitted statistics are not.
def feature_family(col: str) -> str:
    if col.endswith(("_max", "_min", "_amp")):
        return "order_extreme"
    if col.endswith(("_p25", "_p75")):
        return "order_quantile"
    if col.endswith(("_median", "_mean", "_std")):
        return "central"
    if col.endswith(("_slope", "_h_mean", "_h_cos", "_h_sin")):
        return "fitted"
    return "static_or_meta"


def feature_density_audit(feat_dir: Path | None = None,
                          years: tuple[int, ...] | None = None,
                          density_col: str = "n_valid_obs",
                          save: bool = True, tag: str = "") -> pd.DataFrame:
    """1a: how much does each feature move when only the *number of looks* changes?

    Within parcel (parcel FE), regress each feature on ``log(density_col)`` with SEs
    clustered by parcel. The reported ``coef_sd`` is in units of the feature's own
    within-parcel SD, so features on different scales are comparable, and it is the number
    to rank on: a feature with |coef_sd| ≈ 0.3 moves a third of a within-parcel standard
    deviation for an e-fold change in observation count, entirely without the land changing.
    """
    df = _panel_feature_frame(feat_dir, years)
    df = df[df[density_col].notna() & (df[density_col] > 0)]
    x = np.log(df[density_col].to_numpy(float))
    gid = df["COD_PREDIO"].to_numpy()

    skip = {"COD_PREDIO", "year", density_col}
    cols = [c for c in df.columns
            if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    # NaN pattern differs per column (harmonics need >= 4 obs), so group columns by their
    # missingness mask and run one vectorised fit per distinct mask.
    masks: dict[bytes, list[str]] = {}
    for c in cols:
        m = df[c].notna().to_numpy()
        masks.setdefault(m.tobytes(), []).append(c)
    for _, group in masks.items():
        m = df[group[0]].notna().to_numpy()
        if m.sum() < 100 or len(np.unique(gid[m])) < 30:
            continue
        beta, se, sd = _fe_cluster_slopes(df.loc[m, group].to_numpy(float), x[m], gid[m])
        for j, c in enumerate(group):
            if not np.isfinite(sd[j]) or sd[j] == 0:
                continue
            z = beta[j] / se[j] if se[j] > 0 else np.nan
            rows.append({"feature": c, "family": feature_family(c),
                         "coef": float(beta[j]), "se": float(se[j]),
                         "within_sd": float(sd[j]),
                         "coef_sd": float(beta[j] / sd[j]),
                         "t": float(z), "n": int(m.sum()),
                         "n_parcels": int(len(np.unique(gid[m])))})
    out = pd.DataFrame(rows)
    out["abs_coef_sd"] = out["coef_sd"].abs()
    out = out.sort_values("abs_coef_sd", ascending=False).reset_index(drop=True)
    if save:
        suf = f"_{tag}" if tag else ""
        p = proc() / f"density_feature_audit{suf}.csv"
        out.to_csv(p, index=False)
        print(f"wrote {p}: {len(out)} features")
    return out


def audit_summary(audit: pd.DataFrame) -> pd.DataFrame:
    """Family-level roll-up — the number that decides whether 1a scopes anything."""
    g = (audit.groupby("family")
         .agg(n=("feature", "size"), mean_abs_coef_sd=("abs_coef_sd", "mean"),
              max_abs_coef_sd=("abs_coef_sd", "max"),
              frac_sig=("t", lambda s: float((s.abs() > 1.96).mean())))
         .sort_values("mean_abs_coef_sd", ascending=False))
    return g.reset_index()


# ------------------------------------------------------------------------------------
# 1b — degradation: make a training year look like an endpoint year
# ------------------------------------------------------------------------------------
def endpoint_date_counts(feat_dir: Path | None = None,
                         years: tuple[int, ...] = ENDPOINT_YEARS) -> np.ndarray:
    """Empirical distribution of ``n_dates`` in the endpoint window.

    The **distribution**, not the median: matching only the central tendency would leave the
    training data with a tail of richly-observed parcels that the endpoint never has, and it
    is precisely the tail that order statistics live in.
    """
    df = _panel_feature_frame(feat_dir, years)
    return df["n_dates"].to_numpy(float)


def degrade_pixels(px: pd.DataFrame, targets: dict[str, int],
                   rng: np.random.Generator,
                   slc_gap_frac: float = SLC_OFF_FRAC) -> pd.DataFrame:
    """Subsample each parcel's acquisition dates to ``targets[COD_PREDIO]``.

    Dates are dropped, not pixels-within-dates: an acquisition either happened and was clear
    or it did not, and thinning a date's pixels would model cloud, not archive depth. The
    stripes are applied *afterwards*, to the surviving dates, because SLC-off removes part of
    a scene the satellite *did* acquire.

    L7's SLC-off gaps are wedges that are locally parallel over a single parcel, so within a
    parcel they remove a contiguous band of pixel columns, in a position that varies per
    acquisition. That is what makes the loss look like noise in a whole-year summary but not
    in a per-date one. This is an approximation of the geometry, not a simulation of it.

    Parcels already at or below their target are returned untouched — degradation never
    invents observations. Fully vectorised: a per-parcel Python loop over 54 k parcels x
    ~1 M parcel-dates is hours, and this is run once per experiment arm.
    """
    if px.empty:
        return px.copy()
    d = px.reset_index(drop=True)

    # --- 1. keep a random subset of each parcel's acquisition dates -------------------
    pairs = d[["COD_PREDIO", "doy"]].drop_duplicates().copy()
    pairs["_u"] = rng.random(len(pairs))
    pairs = pairs.sort_values(["COD_PREDIO", "_u"])
    pairs["_rank"] = pairs.groupby("COD_PREDIO", sort=False).cumcount()
    tgt = pairs["COD_PREDIO"].map(targets)
    n_have = pairs.groupby("COD_PREDIO", sort=False)["_rank"].transform("size")
    tgt = tgt.fillna(n_have).clip(lower=1)
    pairs = pairs[pairs["_rank"] < tgt.to_numpy()]
    d = d.merge(pairs[["COD_PREDIO", "doy"]], on=["COD_PREDIO", "doy"], how="inner")
    if d.empty or slc_gap_frac <= 0:
        return d

    # --- 2. punch one contiguous lon-band out of each surviving acquisition ----------
    g = d.groupby(["COD_PREDIO", "doy"], sort=False)
    size = g["lon"].transform("size").to_numpy()
    rank = g["lon"].rank(method="first").to_numpy() - 1
    n_drop = np.round(slc_gap_frac * size).astype(int)
    # one random band start per (parcel, date) group, broadcast back to its rows
    codes = g.ngroup().to_numpy()
    starts = rng.random(codes.max() + 1)
    start = np.floor(starts[codes] * np.maximum(size - n_drop, 1)).astype(int)
    drop = (n_drop > 0) & (rank >= start) & (rank < start + n_drop)
    return d[~drop].reset_index(drop=True)


def draw_targets(cods: np.ndarray, pool: np.ndarray, rng: np.random.Generator,
                 current: np.ndarray | None = None) -> dict[str, int]:
    """One target date-count per parcel, from the endpoint's empirical distribution.

    With ``current`` (the parcel's own date count) the assignment is **rank-matched**: a
    parcel at the 90th percentile of the training density gets the 90th percentile of the
    endpoint density. An i.i.d. draw instead pairs rich parcels with poor targets and poor
    parcels with rich ones, and since degradation can only ever remove dates
    (``min(target, have)``) the result lands well *below* the endpoint — matching the
    distribution requires matching the order too. Plan §1b: "match the *distribution* of
    ``n_valid_obs`` in W19, not its median."
    """
    if current is None:
        draws = rng.choice(pool, size=len(cods), replace=True)
    else:
        cur = np.asarray(current, dtype=float)
        # percentile of each parcel within its own cohort, ties broken randomly
        order = np.lexsort((rng.random(len(cur)), cur))
        pct = np.empty(len(cur))
        pct[order] = (np.arange(len(cur)) + 0.5) / len(cur)
        draws = np.quantile(pool, pct)
    return {str(c): int(max(round(d), 1)) for c, d in zip(cods, draws)}


def _degraded_coverage(pd_med: pd.DataFrame) -> pd.DataFrame:
    """``n_valid_obs`` / ``max_gap`` recomputed from a degraded parcel-date table.

    Mirrors ``landsat_gee``'s definitions as closely as the pixel store allows:
    ``n_valid_obs`` is the clear-observation count (here the surviving date count) and
    ``max_gap`` the longest run of empty months over the 12-month year.
    """
    rows = []
    for cid, grp in pd_med.groupby("COD_PREDIO", sort=False):
        months = np.unique(((grp["doy"].to_numpy() - 1) // 30.44).astype(int).clip(0, 11))
        pres = np.zeros(12, dtype=bool)
        pres[months] = True
        gap = best = 0
        for m in pres:
            gap = 0 if m else gap + 1
            best = max(best, gap)
        rows.append({"COD_PREDIO": cid, "n_valid_obs": float(len(grp)),
                     "max_gap": float(best)})
    return pd.DataFrame(rows)


def attach_panel_n_dates(preds: pd.DataFrame,
                         panel_feat_dir: Path | None = None) -> pd.DataFrame:
    """Join each parcel-year's realised ``n_dates`` from its assembled panel bundle.

    The panel prediction table carries ``n_valid_obs`` (the coverage-stage clear count) but
    not ``n_dates`` (what the pixel store actually held, and therefore what the features
    were computed from). Degradation acts on dates, so the calibration ladder is indexed on
    dates and the panel must be too — mixing the two would apply a correction fitted at one
    density to a row labelled with another.
    """
    feat_dir = Path(panel_feat_dir or (feat() / "panel"))
    parts = []
    for y in sorted(preds["year"].unique()):
        p = feat_dir / str(int(y)) / asm.FN_LGBM
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["COD_PREDIO", "n_dates"])
        d["year"] = int(y)
        parts.append(d)
    if not parts:
        raise FileNotFoundError(f"no assembled panel bundles under {feat_dir}")
    nd = pd.concat(parts, ignore_index=True)
    return preds.drop(columns=["n_dates"], errors="ignore").merge(
        nd, on=["COD_PREDIO", "year"], how="left")


def build_degraded_features(src_feat_dir: Path | None = None,
                            out_path: Path | None = None,
                            target_pool: np.ndarray | None = None,
                            parcels: pd.DataFrame | None = None,
                            slc_gap_frac: float = SLC_OFF_FRAC,
                            seed: int = 42,
                            years: list[int] | None = None,
                            only_cods: set[str] | None = None,
                            fixed_target: int | None = None,
                            quiet: bool = False) -> pd.DataFrame:
    """Assemble a LightGBM feature table from a **degraded** copy of a pixel store.

    One row per parcel, exactly the schema of ``features_lightgbm.parquet``, so it can be
    concatenated onto the training matrix (``data.make_flat(augment_feat=…)``) or scored by
    an existing model without any change to the model code.

    ``fixed_target`` overrides the sampled distribution with one date count for every parcel
    — that is the calibration ladder of 1c, where the point is to sweep density, not to
    imitate it.

    Parcels whose degraded copy would fall under the ``n_valid_obs >= 4`` quality gate are
    **dropped, not imputed** (plan §1b): a parcel that would have been abstained on at
    endpoint density must not contribute a training row pretending otherwise.
    """
    src = Path(src_feat_dir or feat())
    rng = np.random.default_rng(seed)
    if parcels is None:
        import geopandas as gpd
        parcels = gpd.read_parquet(proc() / "modeling_parcels.parquet")
    if target_pool is None and fixed_target is None:
        target_pool = endpoint_date_counts()

    files = sorted(src.glob("pixels_*.parquet"))
    if years is not None:
        want = {str(y) for y in years}
        files = [f for f in files if f.stem.split("_")[1] in want]
    frames = []
    for f in files:
        px = pd.read_parquet(f)
        if only_cods is not None:
            px = px[px["COD_PREDIO"].isin(only_cods)]
        if px.empty:
            continue
        px = asm.scale_sr(px)
        px["px_id"] = (px["lon"].round(4).astype(str) + "_"
                       + px["lat"].round(4).astype(str))
        have = px.groupby("COD_PREDIO", sort=False)["doy"].nunique()
        cods = have.index.to_numpy()
        if fixed_target is not None:
            targets = {str(c): int(fixed_target) for c in cods}
        else:
            targets = draw_targets(cods, target_pool, rng,
                                   current=have.to_numpy(float))
        deg = degrade_pixels(px, targets, rng, slc_gap_frac=slc_gap_frac)
        if deg.empty:
            continue
        pd_med = asm.per_date_medians(deg)
        # the extraction-time quality gate, re-applied at the degraded density
        n_dates = pd_med.groupby("COD_PREDIO")["doy"].nunique()
        keep = set(n_dates[n_dates >= 4].index)
        pd_med = pd_med[pd_med["COD_PREDIO"].isin(keep)]
        if pd_med.empty:
            continue
        feats = asm.build_lightgbm_features(pd_med, parcels)
        # `n_valid_obs` and `max_gap` are statics copied from the parcel table, so they
        # still describe the FULL year — every other column now describes the degraded one.
        # Recompute them from the surviving dates so the row is internally consistent; an
        # arm that keeps `meta` would otherwise train on a density label contradicting its
        # own features (the `frac_l7` failure mode, one level down).
        feats = feats.drop(columns=["n_valid_obs", "max_gap"], errors="ignore").merge(
            _degraded_coverage(pd_med), on="COD_PREDIO", how="left")
        frames.append(feats)
        if not quiet:
            print(f"  {f.stem}: {px.COD_PREDIO.nunique():,} -> {len(feats):,} parcels, "
                  f"median n_dates {feats['n_dates'].median():.0f}", flush=True)
    out = (pd.concat(frames, ignore_index=True) if frames
           else pd.DataFrame(columns=["COD_PREDIO"]))
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)
        print(f"wrote {out_path}: {len(out):,} degraded parcel rows")
    return out


# ------------------------------------------------------------------------------------
# 1c — density-conditional temperature
# ------------------------------------------------------------------------------------
CALIBRATION_TARGETS = (6, 8, 10, 12, 16, 22, None)   # None = the undegraded store


def calibration_ladder(run_dir: Path, fold: int = 0,
                       targets: tuple = CALIBRATION_TARGETS,
                       seed: int = 7, save: bool = True) -> pd.DataFrame:
    """Score one fold's **held-out** parcels at a ladder of artificial densities.

    The fold model is refit here rather than loaded because the pipeline only persists the
    final refit, and the final refit has seen every fold's validation parcels. Calibrating
    on data the model trained on measures nothing.

    Returns one row per (parcel, rung) with the realised ``n_dates`` and the predicted
    probabilities — the input to :func:`fit_density_temperature`.
    """
    import json as _json

    from crop_classifier import models  # noqa: F401  (registers the models)
    from crop_classifier.data import (
        class_weights,
        fold_split,
        load_parcels,
        make_flat,
        resolve_drop_features,
    )
    from crop_classifier.models.base import get_model

    run_dir = Path(run_dir)
    with open(run_dir / "cv_metrics.json") as f:
        cfg = _json.load(f)
    if cfg["model"] != "lightgbm":
        raise ValueError("the ladder refits a flat model; run it on a LightGBM arm")
    drop = resolve_drop_features(cfg.get("drop_features"))
    aug = cfg.get("augment_feat")
    parcels = load_parcels(require_quality=True)
    with open(proc() / "label_map.json") as f:
        label_map = _json.load(f)
    n_classes = len(label_map)

    tr_idx, va_idx = fold_split(parcels, fold)
    train_ds = make_flat(parcels, tr_idx, drop_features=drop, augment_feat=aug)
    val_ds = make_flat(parcels, va_idx, drop_features=drop)
    model = get_model("lightgbm")
    model.set_n_classes(n_classes)
    tmp = run_dir / "_ladder"
    tmp.mkdir(parents=True, exist_ok=True)
    model.fit(train_ds, val_ds, class_weights(train_ds.y, n_classes), tmp)
    names = model.feature_names
    classes = [c for c, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    va_cods = set(parcels.loc[va_idx, "COD_PREDIO"])
    rows = []
    for t in targets:
        if t is None:
            ds, tag_n = val_ds, None
            feats = pd.read_parquet(feat() / asm.FN_LGBM,
                                    columns=["COD_PREDIO", "n_dates"])
            nd = dict(zip(feats["COD_PREDIO"], feats["n_dates"]))
        else:
            from crop_classifier.data import FlatData
            deg = build_degraded_features(only_cods=va_cods, fixed_target=t, seed=seed + t,
                                          parcels=parcels, quiet=True)
            nd = dict(zip(deg["COD_PREDIO"], deg["n_dates"]))
            sub = parcels.loc[va_idx, ["COD_PREDIO", "label_id"]].merge(
                deg, on="COD_PREDIO", how="inner")
            # column order pinned to the fitted model's, exactly as inference does
            ds = FlatData(X=sub[names], y=sub["label_id"].to_numpy(),
                          cod_predio=sub["COD_PREDIO"].to_numpy(), feature_names=names)
            tag_n = t
        prob = model.predict_proba(ds)
        df = pd.DataFrame({"COD_PREDIO": ds.cod_predio, "y_true": ds.y,
                           "target": -1 if tag_n is None else tag_n,
                           "n_dates": [nd.get(c, np.nan) for c in ds.cod_predio]})
        for j, c in enumerate(classes):
            df[f"prob_{c}"] = prob[:, j]
        rows.append(df)
        print(f"  rung target={t}: {len(df):,} parcels, "
              f"median n_dates {np.nanmedian(df['n_dates']):.0f}", flush=True)
    out = pd.concat(rows, ignore_index=True)
    if save:
        p = run_dir / "density_ladder.parquet"
        out.to_parquet(p, index=False)
        print(f"wrote {p}: {len(out):,} rows")
    return out


def fit_density_temperature(prob: np.ndarray, y: np.ndarray, n_obs: np.ndarray,
                            n_bands: int = 5) -> dict:
    """Temperature as a linear function of ``log n_obs``, fitted on held-out predictions.

    One scalar ``T`` assumes the model is equally over-confident whatever it was shown. It is
    not: with fewer looks the evidence is weaker, and if the probabilities do not soften to
    match, a share computed from them drifts as the archive thins. Bands are equal-count
    quantiles of ``log n``; the reported ``slope``/``intercept`` are an OLS fit of the
    per-band temperatures on the band-mean ``log n``, so the correction is monotone and
    defined between bands.
    """
    from crop_classifier.perennial.calibration import (
        expected_calibration_error,
        fit_temperature,
        nll,
    )

    ln = np.log(np.clip(n_obs, 1, None))
    qs = np.quantile(ln, np.linspace(0, 1, n_bands + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    bands = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (ln >= lo) & (ln < hi)
        if m.sum() < 200:
            continue
        T = fit_temperature(prob[m], y[m])
        bands.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                      "mean_log_n": float(ln[m].mean()),
                      "mean_n": float(np.exp(ln[m]).mean()),
                      "temperature": float(T),
                      "nll": nll(prob[m], y[m]),
                      "ece": expected_calibration_error(prob[m], y[m])})
    if len(bands) < 2:
        return {"bands": bands, "slope": 0.0,
                "intercept": float(bands[0]["temperature"]) if bands else 1.0}
    xb = np.array([b["mean_log_n"] for b in bands])
    tb = np.array([b["temperature"] for b in bands])
    wb = np.array([b["n"] for b in bands], dtype=float)
    slope, intercept = np.polyfit(xb, tb, 1, w=np.sqrt(wb))
    return {"bands": bands, "slope": float(slope), "intercept": float(intercept),
            "global_temperature": float(fit_temperature(prob, y)),
            "n": int(len(y))}


def apply_density_temperature(prob: np.ndarray, n_obs: np.ndarray, params: dict,
                              t_min: float = 0.5, t_max: float = 6.0) -> np.ndarray:
    """Apply ``T(log n) = intercept + slope · log n`` row-wise, clipped to a sane range."""
    from crop_classifier.perennial.calibration import apply_temperature

    ln = np.log(np.clip(np.asarray(n_obs, dtype=float), 1, None))
    T = np.clip(params["intercept"] + params["slope"] * ln, t_min, t_max)
    out = np.empty_like(prob, dtype=float)
    for t in np.unique(T):
        m = T == t
        out[m] = apply_temperature(prob[m], float(t))
    return out


def recalibrate_predictions(preds: pd.DataFrame, params: dict,
                            base_temperature: float = 1.0,
                            density_col: str = "n_valid_obs") -> pd.DataFrame:
    """Re-scale a prediction table's ``prob_*`` columns by ``T(n)``.

    Temperature scaling composes — ``apply(apply(p, T1), T2) == apply(p, T1·T2)`` — so a
    panel that was already scaled by the run's scalar ``T`` can be moved onto a
    density-conditional temperature *post hoc*, with no re-inference. ``base_temperature``
    is that already-applied scalar; the extra factor is ``T(n)/base``.
    """
    cols = [c for c in preds.columns if c.startswith("prob_")]
    classes = [c[len("prob_"):] for c in cols]
    out = preds.copy()
    p = out[cols].to_numpy(float)
    extra = dict(params)
    extra["intercept"] = params["intercept"] / base_temperature
    extra["slope"] = params["slope"] / base_temperature
    p2 = apply_density_temperature(p, out[density_col].to_numpy(float), extra)
    for j, c in enumerate(cols):
        out[c] = p2[:, j]
    out["pred_label"] = np.array(classes)[p2.argmax(1)]
    return out


# ------------------------------------------------------------------------------------
# 2c — per-year distribution alignment (a SENSITIVITY arm, never a default)
# ------------------------------------------------------------------------------------
def quantile_align(values: np.ndarray, reference: np.ndarray,
                   n_knots: int = 201) -> np.ndarray:
    """Map ``values`` onto ``reference``'s distribution by matching quantiles."""
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < 10 or not np.isfinite(reference).any():
        return v
    qs = np.linspace(0, 1, n_knots)
    src = np.quantile(v[ok], qs)
    dst = np.quantile(reference[np.isfinite(reference)], qs)
    out = v.copy()
    # np.interp needs a strictly increasing x; ties in a degenerate feature collapse it
    keep = np.r_[True, np.diff(src) > 0]
    if keep.sum() < 2:
        return v
    out[ok] = np.interp(v[ok], src[keep], dst[keep])
    return out


def write_aligned_panel_bundles(run_feature_names: list[str],
                                reference: pd.DataFrame,
                                years: list[int],
                                panel_feat_dir: Path | None = None,
                                suffix: str = "_qmap") -> list[int]:
    """Quantile-map each panel year's features onto the training distribution.

    ⚠️ **This erases genuine aggregate change along with the artefact.** If perennial area
    really grew, the aligned panel cannot show it: forcing each year's marginal onto the
    training years' marginal removes any shift in the mean of a feature, whatever caused it.
    It is therefore a *sensitivity arm* — it answers "how much of the series survives if we
    assume no aggregate change in the features?" — and must never be the default panel
    (temporal_ood_plan.md §2c).

    Writes ``<panel>/<year><suffix>/features_lightgbm.parquet`` so nothing existing moves.
    """
    feat_dir = Path(panel_feat_dir or (feat() / "panel"))
    written = []
    for y in years:
        src = feat_dir / str(y) / asm.FN_LGBM
        if not src.exists():
            continue
        d = pd.read_parquet(src)
        for c in run_feature_names:
            if c in d.columns and c in reference.columns:
                d[c] = quantile_align(d[c].to_numpy(float),
                                      reference[c].to_numpy(float))
        out_dir = feat_dir / f"{y}{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        d.to_parquet(out_dir / asm.FN_LGBM, index=False)
        written.append(y)
    print(f"wrote {len(written)} aligned bundles to {feat_dir}/<year>{suffix}/")
    return written


def save_params(params: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"wrote {path}")

"""Two-period tenure difference-in-differences — ``docs/RESULTS.md §7`` v2.

Four estimands failed because each needed the classifier to deliver a defensible *level* or
*trend*, and the predicted probability drifts as the Landsat archive thins (§8.3). This design
stops trying to fix that. Each parcel carries **two dated observations of its titling status**
— status at its PETT declaration (parcel-specific, 1996-2009) and status at its cadastre
transaction (mostly 2011-12). Comparing parcels that became registered against parcels that did
not, before against after, **differences out any drift common to both arms**, which is the
class of artefact that killed the window pivot.

The classifier still supplies the outcome, with all of its problems. What changes is that its
errors now land on *both* sides of the comparison.

**The two observations are snapshots of a rolling programme, not two shared dates** (plan §1.1).
Titling ran as continuous departmental campaigns, so registration happened at an unobserved
moment inside each parcel's own interval. That forces the restrictions in :func:`restrict`:

* **R1** at-risk pool only (PETT label ``ANNUAL``) — both arms then start at a true perennial
  share of ~0, so the class-specific component of the drift applies equally to both. Dropping
  it rebuilds the design that already failed (§8.2).
* **R2** both observations dated — 24.6 % of parcels carry a status with no date, and an
  undated change cannot be placed on either side of a window.
* **R3** cadastre date at or after the declaration, and strictly before the post-windows.
* **R4** ⭐ the pre-window must end **before that parcel's own declaration year**, because
  before the declaration is the only interval in which the parcel is *known* untreated. This is
  the restriction that decides whether the design is identified, and it is the one the pilot
  did not apply.
* **R5** treated and control compared within department x declaration-year cohort — the
  registration rate is non-monotone in the gap between observations, so the gap marks a
  campaign wave rather than a duration.

**N-D9**: the control is ``NO INSCRITO`` at *both* observations. Already-``INSCRITO`` parcels
are a different kind of parcel carrying T1's differential false-positive rate (§8.5); they are
a sensitivity arm, never the primary control.

Run with::

    CC_PROC=data/processed/all_peru uv run python -m crop_classifier.cli allperu tenure-did \\
        --preds data/processed/all_peru/panel_predictions_nolat_aug_yleak10.parquet \\
        --tag nolat_aug_yleak10_pilot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crop_classifier.paths import proc

PROB_COL = "prob_PERENNIAL"

# Calendar windows, inherited from windows.py so the two modules cannot drift apart.
WINDOWS: dict[str, tuple[int, int]] = {
    "W99": (1999, 2003), "W04": (2004, 2008), "W09": (2009, 2013),
    "W14": (2014, 2018), "W19": (2019, 2023),
}
# The pre-period split used for the R4-valid placebo. The powered cohort (reg_year >= 2004)
# has exactly ONE window entirely before declaration (W99), so a pre-trend test has to be
# built by splitting it. Both halves are still entirely pre-declaration.
PRE_SUBWINDOWS: dict[str, tuple[int, int]] = {"P1": (1999, 2000), "P2": (2001, 2003)}

MIN_YEARS = 3           # a 5-year window needs this many observed years to qualify
MIN_YEARS_SUB = 2       # a 2-3 year sub-window needs this many
POST_WINDOWS = ["W14", "W19"]

# --- registered gate thresholds (plan §4). Fixed before the study; do not tune. ---
G1_BAND_PER_DECADE = 0.010   # equals the registered +/-0.005 over a 5-year step
G2_MAX_SMD = 0.25            # common support
G2_MAX_RATE_GAP = 0.02       # differential attrition / training-set membership
G2_MIN_SHARED_REGION = 0.80  # share of parcels in regions holding both arms
G3A_MAX_SE = 0.0025          # placebo resolution — the binding criterion
G3B_MDE = 0.02               # headline effect the study must be powered for

# --- the v3 reopening (plan §9): subtract the pre-trend instead of proving it away ---
# The registered sensitivity grid. M = 0 is the uncorrected headline; the derived M (7 on
# this window grid) assumes the pre-trend continues in a straight line for the whole
# headline horizon and is the pessimistic end. **Report the curve, never one point.**
SENSITIVITY_M: tuple[float, ...] = (0.0, 1.0, 3.0, 5.0, 7.0)


# ---------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------
def load_inputs(preds: Path | str, parcels: Path | str,
                tenure: Path | str) -> pd.DataFrame:
    """Panel predictions + parcel geography + the two tenure observations, one frame.

    ``tenure_two_period.parquet`` carries duplicate ``COD_PREDIO`` rows (726,812 join hits
    against 726,808 parcels); de-duplicating here is not optional — without it a handful of
    parcels are silently double-weighted.
    """
    import geopandas as gpd

    p = pd.read_parquet(preds)
    try:
        g = gpd.read_parquet(parcels)
    except Exception:                                            # pragma: no cover
        g = pd.read_parquet(parcels)
    t = pd.read_parquet(tenure).drop_duplicates("COD_PREDIO")

    geo_cols = [c for c in ("COD_PREDIO", "dept", "region_id", "split", "area_ha")
                if c in g.columns]
    ten_cols = ["COD_PREDIO", "tenure", "cadastre_status", "became_registered",
                "reg_year", "cadastre_date"]
    out = (p.merge(g[geo_cols], on="COD_PREDIO", how="left", suffixes=("", "_geo"))
             .merge(t[ten_cols], on="COD_PREDIO", how="left"))
    out["cadastre_year"] = pd.to_datetime(out["cadastre_date"]).dt.year
    return out


# ---------------------------------------------------------------------------------
# R1-R5
# ---------------------------------------------------------------------------------
def restrict(df: pd.DataFrame, cohort_min_year: int, control: str = "no_inscrito",
             post_windows: list[str] | None = None,
             apply_r2r3: bool = True) -> tuple[pd.DataFrame, dict]:
    """Apply R1-R4 and the N-D9 control definition. Returns ``(kept, attrition_report)``.

    ``cohort_min_year`` implements **R4**: a parcel is kept only if every pre-window used for
    identification ends before its declaration year. For the default pre-window ``W99``
    (1999-2003) that means ``reg_year >= 2004``.

    ``control`` selects the comparison arm:

    * ``no_inscrito`` (**N-D9, the primary**) — ``NO INSCRITO`` at both observations;
    * ``any`` — the pilot's definition, everything not treated, which mixes in
      already-registered parcels and reintroduces T1's differential false-positive rate.
    """
    post = post_windows or POST_WINDOWS
    first_post_year = min(WINDOWS[w][0] for w in post)
    rep: dict = {"n_start": int(df["COD_PREDIO"].nunique())}

    d = df[~df.get("abstained", pd.Series(False, index=df.index)).astype(bool)]
    d = d[d[PROB_COL].notna()]
    rep["n_after_abstain"] = int(d["COD_PREDIO"].nunique())

    d = d[d["pett_label"] == "ANNUAL"]                                   # R1
    rep["n_R1_at_risk"] = int(d["COD_PREDIO"].nunique())

    if apply_r2r3:
        d = d[d["cadastre_date"].notna() & d["reg_year"].notna()]        # R2
        rep["n_R2_dated"] = int(d["COD_PREDIO"].nunique())
        d = d[(d["cadastre_year"] >= d["reg_year"])                      # R3
              & (d["cadastre_year"] < first_post_year)]
        rep["n_R3_ordered"] = int(d["COD_PREDIO"].nunique())

    if cohort_min_year:
        d = d[d["reg_year"] >= cohort_min_year]                          # R4
        rep["n_R4_clean_pre"] = int(d["COD_PREDIO"].nunique())

    if control == "no_inscrito":                                         # N-D9
        d = d[d["tenure"] == "NO INSCRITO"]
    elif control != "any":
        raise ValueError(f"unknown control policy {control!r}")
    d = d.copy()
    d["treat"] = d["became_registered"].astype(float)
    rep["control_policy"] = control
    rep["cohort_min_year"] = int(cohort_min_year)
    rep["n_final"] = int(d["COD_PREDIO"].nunique())
    rep["n_treated"] = int(d.loc[d.treat == 1, "COD_PREDIO"].nunique())
    rep["n_control"] = int(d.loc[d.treat == 0, "COD_PREDIO"].nunique())
    return d, rep


def window_means(df: pd.DataFrame, spec: dict[str, tuple[int, int]],
                 min_years: int) -> pd.DataFrame:
    """Parcel x window mean probability, dropping windows with too few observed years.

    A window that does not qualify is **dropped, not filled**: an abstained year and an
    unchanged year are not the same thing, and filling makes them look alike (W-D2).
    """
    parts = []
    carry = [c for c in ("treat", "region_id", "dept", "reg_year", "cohort", "area_ha",
                         "sample_weight", "split") if c in df.columns]
    for name, (lo, hi) in spec.items():
        s = df[(df["year"] >= lo) & (df["year"] <= hi)]
        if s.empty:
            continue
        agg = {"p_mean": (PROB_COL, "mean"), "n_years": (PROB_COL, "size")}
        agg.update({c: (c, "first") for c in carry})
        g = s.groupby("COD_PREDIO", observed=True).agg(**agg).reset_index()
        g = g[g["n_years"] >= min_years]
        g["window"] = name
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------------------------
# the estimator
# ---------------------------------------------------------------------------------
def did(tab: pd.DataFrame, pre: str, post: list[str], cluster: str = "region_id",
        dept_window_fe: bool = True, extra_post_fe: str | None = None) -> dict:
    """Parcel FE + window FE (+ department x window FE) DiD, SEs clustered by ``cluster``.

    Parcel FE absorbs every time-invariant parcel property, **including T1's differential
    false-positive rate** (§8.5) — a level difference between kinds of parcel, which therefore
    drops out. Department x window FE absorbs region-specific drift, the artefact class that
    killed M2. Only parcels observed on **both** sides contribute, which is what makes the
    comparison within-parcel.
    """
    import statsmodels.api as sm

    s = tab[tab["window"].isin([pre] + post)].copy()
    s["post"] = s["window"].isin(post).astype(float)
    both = s.groupby("COD_PREDIO")["post"].nunique()
    s = s[s["COD_PREDIO"].isin(both[both == 2].index)]
    # Degenerate inputs return nan rather than raising. A single cluster makes the
    # cluster-robust correction divide by (n_groups - 1) = 0 deep inside statsmodels, which
    # surfaces as a ZeroDivisionError from a sandwich-estimator internal — unreadable, and it
    # aborts any per-department loop on its smallest department instead of reporting nan
    # for that one.
    if (s.empty or s["treat"].nunique() < 2
            or (cluster in s.columns and s[cluster].nunique() < 2)):
        return {"coef": float("nan"), "se": float("nan"), "p": float("nan"),
                "n_treated": 0, "n_control": 0, "n_obs": 0}
    s["tp"] = s["treat"] * s["post"]

    X = s[["post", "tp"]].astype(float)
    if extra_post_fe and extra_post_fe in s.columns and s[extra_post_fe].nunique() > 1:
        # R5's second half as a regression: declaration-year cohort gets its own time
        # effect. Interacted with POST for the same rank reason as `dept` below.
        ex = pd.get_dummies(s[extra_post_fe].astype(str), drop_first=True, dtype=float)
        X = pd.concat([X, ex.mul(s["post"].to_numpy(), axis=0)
                       .add_prefix(f"post_x_{extra_post_fe}_")], axis=1)
    if dept_window_fe and "dept" in s.columns and s["dept"].nunique() > 1:
        # Department-specific time effects. Interacting `dept` with the POST indicator is the
        # only full-rank way to do this in a two-period design: a dept x window dummy set is
        # collinear with the window dummies once the parcel FE is swept out (within a parcel
        # the pre and post indicators are negatives of each other), which silently returns
        # nan standard errors rather than an error.
        dep = pd.get_dummies(s["dept"].astype(str), drop_first=True, dtype=float)
        dep = dep.mul(s["post"].to_numpy(), axis=0).add_prefix("post_x_")
        X = pd.concat([X, dep], axis=1)
    g = s["COD_PREDIO"].to_numpy()
    Xd = X.set_index(s.index).groupby(g).transform(lambda z: z - z.mean())
    yd = s["p_mean"].astype(float).groupby(g).transform(lambda z: z - z.mean())
    keep = Xd.columns[Xd.std().to_numpy() > 1e-12]
    r = sm.OLS(yd.to_numpy(), sm.add_constant(Xd[keep], has_constant="add")).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(s[cluster])[0]})
    return {"coef": float(r.params["tp"]), "se": float(r.bse["tp"]),
            "p": float(r.pvalues["tp"]),
            "n_treated": int(s.loc[s.treat == 1, "COD_PREDIO"].nunique()),
            "n_control": int(s.loc[s.treat == 0, "COD_PREDIO"].nunique()),
            "n_obs": int(len(s)), "n_clusters": int(s[cluster].nunique())}


def _midpoint(w: str, spec: dict[str, tuple[int, int]]) -> float:
    lo, hi = spec[w]
    return (lo + hi) / 2.0


def horizon_years(pre: str, post: list[str], spec: dict[str, tuple[int, int]]) -> float:
    """Years between the pre-window midpoint and the (mean) post-window midpoint."""
    return float(np.mean([_midpoint(w, spec) for w in post]) - _midpoint(pre, spec))


# ---------------------------------------------------------------------------------
# G1 — the placebo, as an equivalence test with three outcomes
# ---------------------------------------------------------------------------------
def placebo(tab: pd.DataFrame, pre: str, post: str,
            spec: dict[str, tuple[int, int]],
            band_per_decade: float = G1_BAND_PER_DECADE,
            dept_window_fe: bool = True) -> dict:
    """G1. Both windows are entirely pre-treatment, so the true coefficient is zero.

    **Equivalence, not significance** (plan §4). The old criterion paired an equivalence bound
    with "the CI contains 0"; those pull in opposite directions, because the second rewards an
    imprecise estimate and punishes a precise one. Here:

    * **PASS** — the whole 95 % CI lies inside the band;
    * **FAIL** — the CI excludes 0 *and* the point estimate is outside the band;
    * **INCONCLUSIVE** — anything else, i.e. the CI is wider than the band.

    The band is expressed **per decade** so placebo contrasts of different lengths are held to
    the same standard. ``+/-0.005`` over a 5-year step is ``0.010`` per decade.
    """
    r = did(tab, pre, [post], dept_window_fe=dept_window_fe)
    h = horizon_years(pre, [post], spec)
    band = band_per_decade * h / 10.0
    lo, hi = r["coef"] - 1.96 * r["se"], r["coef"] + 1.96 * r["se"]
    inside = bool(lo >= -band and hi <= band)
    excludes_zero = bool(lo > 0 or hi < 0)
    outside_band = bool(abs(r["coef"]) > band)
    verdict = "PASS" if inside else ("FAIL" if (excludes_zero and outside_band)
                                     else "INCONCLUSIVE")
    return {**r, "contrast": f"{pre}->{post}", "horizon_years": h,
            "band": band, "band_per_decade": band_per_decade,
            "ci_low": lo, "ci_high": hi,
            "coef_per_decade": r["coef"] / h * 10.0 if h else float("nan"),
            "verdict": verdict}


# ---------------------------------------------------------------------------------
# The v3 correction — measure the pre-trend and subtract it (plan §9)
# ---------------------------------------------------------------------------------
def amplification_factor(pre: str, post: list[str], placebo_pre: str, placebo_post: str,
                         spec: dict[str, tuple[int, int]]) -> float:
    """``M`` — how many placebo horizons fit inside the headline horizon.

    A pre-trend measured over a 2.5-year step contaminates a 17.5-year headline by **seven
    times** its coefficient, not by its coefficient. ``M`` is therefore a property of the
    **window grid**, not a constant, and is derived from window midpoints here so that
    changing ``WINDOWS`` or ``PRE_SUBWINDOWS`` cannot silently leave a stale 7 behind.
    """
    h_head = horizon_years(pre, post, spec)
    h_plac = horizon_years(placebo_pre, [placebo_post], spec)
    if h_plac == 0:
        raise ValueError("placebo horizon is zero — M is undefined")
    return float(h_head / h_plac)


def corrected_effect(headline: dict, placebo: dict, m: float,
                     cov: float = 0.0) -> dict:
    """``headline - M * placebo``, carrying the placebo's uncertainty into the CI.

    This is the whole point of the reopening. The v2 gate demanded *proof* that the
    pre-trend was negligible (an equivalence test), and the sample that would need does not
    exist in Peru — 25,202 parcels per arm against 6,559 (RESULTS.md §10.4). Measuring the
    pre-trend and subtracting it is strictly less demanding and *is* achievable, at the cost
    of a much wider interval::

        corrected = headline - M * placebo
        se        = sqrt(se_h^2 + M^2 * se_p^2 - 2 * M * cov)

    ``cov`` defaults to **zero — the registered formula**, which treats the two coefficients
    as independent. They are not: both are estimated on overlapping parcels and the placebo's
    windows sit inside the headline's pre-period. A *positive* covariance would make the
    registered SE conservative; ``bootstrap_covariance`` measures the sign and size and is
    reported as a diagnostic. The registered number never moves on it.
    """
    from scipy.stats import norm

    coef = float(headline["coef"] - m * placebo["coef"])
    var = float(headline["se"] ** 2 + (m ** 2) * placebo["se"] ** 2 - 2.0 * m * cov)
    se = float(np.sqrt(var)) if var > 0 else float("nan")
    lo, hi = coef - 1.96 * se, coef + 1.96 * se
    z = coef / se if se and se == se and se > 0 else float("nan")
    return {"m": float(m), "coef": coef, "se": se, "ci_low": float(lo),
            "ci_high": float(hi), "z": float(z),
            "p": float(2 * (1 - norm.cdf(abs(z)))) if z == z else float("nan"),
            "excludes_zero": bool(se == se and se > 0 and (lo > 0 or hi < 0)),
            "cov_used": float(cov),
            "headline_coef": float(headline["coef"]), "headline_se": float(headline["se"]),
            "placebo_coef": float(placebo["coef"]), "placebo_se": float(placebo["se"])}


def sensitivity_curve(headline: dict, placebo: dict,
                      ms: tuple[float, ...] = SENSITIVITY_M,
                      cov: float = 0.0) -> pd.DataFrame:
    """The corrected effect and its CI at every registered ``M``.

    Registered as **always reported**: ``M = 0`` is the uncorrected headline and the derived
    ``M`` assumes the pre-trend runs in a straight line for the whole headline horizon.
    Neither is "the" answer — publishing the curve lets a reader choose their assumption,
    and publishing one point without it invites the reader to assume the author chose.
    """
    return pd.DataFrame([corrected_effect(headline, placebo, m, cov=cov) for m in ms])


def decision(corrected: dict) -> dict:
    """The registered primary decision rule (plan §9, fixed before extraction).

    **REPORTED** — the 95 % CI of ``headline - M * placebo`` excludes zero.
    **NOT-SEPARABLE** — anything else: the effect cannot be told apart from pre-existing
    drift. That is a *complete outcome*, not a failure — it bounds the titling effect and
    measures the anticipation trend, and it is written up to the same standard.
    """
    ok = bool(corrected.get("excludes_zero"))
    return {"decision": "REPORTED" if ok else "NOT-SEPARABLE",
            "rule": "95 % CI of (headline - M * placebo) excludes zero",
            "m": corrected.get("m"), "coef": corrected.get("coef"),
            "ci": [corrected.get("ci_low"), corrected.get("ci_high")],
            "se": corrected.get("se")}


def bootstrap_covariance(tab: pd.DataFrame, pre: str, post: list[str],
                         placebo_pre: str, placebo_post: str,
                         n_boot: int = 200, seed: int = 42,
                         cluster: str = "region_id",
                         dept_window_fe: bool = True) -> dict:
    """Cluster bootstrap of ``cov(headline, placebo)`` — a diagnostic, never the estimate.

    Resamples **whole regions** (the clustering unit, and the spatial-autocorrelation range),
    refits both coefficients on each replicate and reports their covariance and correlation.
    The registered SE assumes ``cov = 0``; this says how wrong that is and in which
    direction. It is deliberately *not* wired into the reported number — a registered formula
    that moves after the data are seen is not registered.
    """
    rng = np.random.default_rng(seed)
    regions = tab[cluster].dropna().unique()
    hs, ps = [], []
    for _ in range(n_boot):
        pick = rng.choice(regions, size=len(regions), replace=True)
        parts = []
        for j, r in enumerate(pick):                 # re-key so a region drawn twice counts twice
            s = tab[tab[cluster] == r].copy()
            s["COD_PREDIO"] = s["COD_PREDIO"].astype(str) + f"__b{j}"
            s[cluster] = f"{r}__b{j}"
            parts.append(s)
        rep = pd.concat(parts, ignore_index=True)
        h = did(rep, pre, post, cluster=cluster, dept_window_fe=dept_window_fe)
        p = did(rep, placebo_pre, [placebo_post], cluster=cluster,
                dept_window_fe=dept_window_fe)
        if h["coef"] == h["coef"] and p["coef"] == p["coef"]:
            hs.append(h["coef"])
            ps.append(p["coef"])
    if len(hs) < 20:
        return {"n_boot_ok": len(hs), "cov": float("nan"), "corr": float("nan")}
    h_arr, p_arr = np.asarray(hs), np.asarray(ps)
    cov = float(np.cov(h_arr, p_arr)[0, 1])
    return {"n_boot_ok": len(hs), "cov": cov,
            "corr": float(np.corrcoef(h_arr, p_arr)[0, 1]),
            "boot_se_headline": float(h_arr.std(ddof=1)),
            "boot_se_placebo": float(p_arr.std(ddof=1))}


# ---------------------------------------------------------------------------------
# G2 — sample integrity
# ---------------------------------------------------------------------------------
def _smd(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference, the conventional common-support statistic."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return float(abs(a.mean() - b.mean()) / s) if s > 0 else 0.0


def integrity(df: pd.DataFrame, tab: pd.DataFrame,
              spec: dict[str, tuple[int, int]]) -> dict:
    """G2 — four cheap checks, all computable before any extraction is funded.

    Two of them are new to this plan and have never been run in this project: **differential
    attrition** (window qualification depends on observation density, which depends on parcel
    size, which correlates with titling) and **training-set membership** (the model was trained
    on some of these parcels *labelled* ``ANNUAL``, which holds their predicted perennial
    probability down in every window).

    ⚠️ Baseline ``p_mean`` is **reported and never acted on**. Trimming on the pre-period
    outcome induces regression to the mean, which looks exactly like the pre-trend G1 hunts
    for.
    """
    one = df.drop_duplicates("COD_PREDIO")
    t, c = one[one.treat == 1], one[one.treat == 0]
    out: dict = {"n_treated": int(len(t)), "n_control": int(len(c))}

    smds = {}
    if "area_ha" in one.columns:
        smds["log_area"] = _smd(np.log(t["area_ha"].clip(lower=1e-3)),
                                np.log(c["area_ha"].clip(lower=1e-3)))
    smds["reg_year"] = _smd(t["reg_year"], c["reg_year"])
    if "cadastre_year" in one.columns:
        smds["exposure_gap"] = _smd(t["cadastre_year"] - t["reg_year"],
                                    c["cadastre_year"] - c["reg_year"])
    if "dept" in one.columns:                # department: max SMD over one-hot shares
        dep = sorted(one["dept"].dropna().unique())
        smds["dept_max"] = max((_smd((t["dept"] == d).astype(float),
                                     (c["dept"] == d).astype(float)) for d in dep),
                               default=float("nan"))
    out["smd"] = smds
    out["max_smd"] = float(np.nanmax(list(smds.values()))) if smds else float("nan")
    out["pass_common_support"] = bool(out["max_smd"] < G2_MAX_SMD)

    # differential attrition: does a window qualify at the same rate in both arms?
    att = {}
    for w in spec:
        q = tab[tab["window"] == w]["COD_PREDIO"].unique()
        rt = float(t["COD_PREDIO"].isin(q).mean()) if len(t) else float("nan")
        rc = float(c["COD_PREDIO"].isin(q).mean()) if len(c) else float("nan")
        att[w] = {"treated": rt, "control": rc, "gap": rt - rc}
    out["attrition"] = att
    gaps = [abs(v["gap"]) for v in att.values() if v["gap"] == v["gap"]]
    out["max_attrition_gap"] = float(max(gaps)) if gaps else float("nan")
    out["pass_attrition"] = bool(out["max_attrition_gap"] < G2_MAX_RATE_GAP)

    # training-set membership: a parcel the model fitted is anchored in every window
    if "split" in one.columns and one["split"].notna().any():
        tr_t = float((t["split"] == "trainval").mean())
        tr_c = float((c["split"] == "trainval").mean())
        out["train_membership"] = {"treated": tr_t, "control": tr_c, "gap": tr_t - tr_c}
        out["pass_train_membership"] = bool(abs(tr_t - tr_c) < G2_MAX_RATE_GAP)
    else:
        out["train_membership"] = None
        out["pass_train_membership"] = True

    # arms must share regions, or treatment is collinear with the clustering unit
    if "region_id" in one.columns:
        by = one.groupby("region_id")["treat"].agg(["min", "max"])
        shared = set(by[by["min"] != by["max"]].index)
        out["shared_region_share"] = float(one["region_id"].isin(shared).mean())
        out["pass_shared_regions"] = bool(
            out["shared_region_share"] >= G2_MIN_SHARED_REGION)
    else:
        out["shared_region_share"] = float("nan")
        out["pass_shared_regions"] = False

    # reported, never gated, never trimmed on
    base = tab[tab["window"] == list(spec)[0]]
    if not base.empty:
        out["baseline_p_mean"] = {
            "treated": float(base.loc[base.treat == 1, "p_mean"].mean()),
            "control": float(base.loc[base.treat == 0, "p_mean"].mean())}
        out["baseline_p_mean"]["gap"] = (out["baseline_p_mean"]["treated"]
                                         - out["baseline_p_mean"]["control"])
    out["pass"] = bool(out["pass_common_support"] and out["pass_attrition"]
                       and out["pass_train_membership"] and out["pass_shared_regions"])
    return out


# ---------------------------------------------------------------------------------
# G3 — precision, sized for the gate that decides
# ---------------------------------------------------------------------------------
def precision(df: pd.DataFrame, spec: dict[str, tuple[int, int]],
              pre: str, post: str, min_years: int,
              targets: tuple[int, ...] = (1300, 2000, 4000, 6500, 10000, 20000),
              band_per_decade: float = G1_BAND_PER_DECADE) -> dict:
    """G3a — the expected placebo SE at a range of sample sizes, from measured variance.

    The old G4 powered the *headline* while N-D1 made the *placebo* decisive, so it would have
    reported a comfortable pass on a sample that then coin-flips the gate that ends the study.
    This computes the quantity that actually binds.

    The DiD SE on a two-window contrast is driven by the SD of the **within-parcel change** in
    window-mean probability, inflated by the intra-region correlation of that change::

        SE ~ SD(delta) * sqrt((1/n_t + 1/n_c) * deff_cluster)

    Both inputs are measured on the panel in hand rather than assumed — this project's own rule
    for GEE timing, applied to statistics.
    """
    a = window_means(df, {pre: spec[pre]}, min_years).set_index("COD_PREDIO")
    b = window_means(df, {post: spec[post]}, min_years).set_index("COD_PREDIO")
    j = a[["p_mean", "region_id"]].join(b[["p_mean"]], how="inner", rsuffix="_post")
    if len(j) < 30:
        return {"n_pairs": int(len(j)), "error": "too few paired parcels to measure"}
    delta = (j["p_mean_post"] - j["p_mean"]).astype(float)
    sd = float(delta.std(ddof=1))

    grp = j.assign(d=delta).groupby("region_id")["d"]
    m = float(grp.size().mean())
    var_w = float(delta.var(ddof=1))
    var_b = float(grp.mean().var(ddof=1))
    icc = float(max(0.0, (var_b - var_w / m) / var_w)) if var_w > 0 else 0.0
    deff = 1.0 + (m - 1.0) * icc

    h = horizon_years(pre, [post], spec)
    band = band_per_decade * h / 10.0
    rows = []
    for n in targets:
        se = sd * np.sqrt((1.0 / n + 1.0 / n) * deff)
        rows.append({"n_per_arm": int(n), "expected_se": float(se),
                     "ci_half_width": float(1.96 * se),
                     "fits_band": bool(1.96 * se <= band)})
    need = (1.96 * sd) ** 2 * 2 * deff / band ** 2 if band > 0 else float("inf")
    return {"n_pairs": int(len(j)), "sd_delta": sd, "mean_parcels_per_region": m,
            "icc": icc, "deff_cluster": float(deff), "horizon_years": h, "band": band,
            "ladder": rows, "n_per_arm_required": float(need),
            "max_se_allowed": float(band / 1.96)}


def population_ceiling(source: Path | str, tenure: Path | str,
                       cohort_min_year: int = 2004,
                       post_windows: list[str] | None = None) -> dict:
    """How many parcels in **all of Peru** could ever enter this design, after R1-R4.

    This is the number that decides whether G3a is reachable, and it is a property of the
    archive rather than of the budget. If the required sample exceeds it, no amount of GEE
    time buys the study — which is exactly the situation the plan's stop rule anticipates.
    """
    post = post_windows or POST_WINDOWS
    first_post_year = min(WINDOWS[w][0] for w in post)
    src = pd.read_parquet(source, columns=["COD_PREDIO", "dept", "label", "year"])
    t = pd.read_parquet(tenure).drop_duplicates("COD_PREDIO")
    d = (src[src["label"] == "ANNUAL"]
         .merge(t, on="COD_PREDIO", how="inner", suffixes=("", "_t"))
         .drop_duplicates("COD_PREDIO"))
    steps = {"R1_at_risk": int(len(d))}
    d = d[d["cadastre_date"].notna() & d["reg_year"].notna()]
    steps["R2_dated"] = int(len(d))
    cy = pd.to_datetime(d["cadastre_date"]).dt.year
    d = d[(cy >= d["reg_year"]) & (cy < first_post_year)]
    steps["R3_ordered"] = int(len(d))
    d = d[d["reg_year"] >= cohort_min_year]
    steps["R4_clean_pre"] = int(len(d))
    treated = d[d["became_registered"]]
    control = d[(d["tenure"] == "NO INSCRITO") & (~d["became_registered"])]
    eff = 1.0 / (1.0 / max(len(treated), 1) + 1.0 / max(len(control), 1))
    return {"cohort_min_year": cohort_min_year, "steps": steps,
            "n_treated_max": int(len(treated)), "n_control_max": int(len(control)),
            "effective_n_max": float(eff),
            "by_dept": treated.groupby("dept").size().sort_values(
                ascending=False).to_dict()}


def feasibility(precision_result: dict, ceiling: dict) -> dict:
    """N3/G3a: can the population supply the sample the placebo gate needs?

    This is the question that decides whether any GEE budget should be spent, and it is
    answerable **before** spending a second of it. ``required_effective_n`` comes from measured
    variance; ``available_effective_n`` from the archive after R1-R4. If the first exceeds the
    second, the study is not fundable at any budget and the plan's stop rule applies.
    """
    need_per_arm = precision_result.get("n_per_arm_required", float("nan"))
    need_eff = need_per_arm / 2.0
    have_eff = ceiling.get("effective_n_max", float("nan"))
    shortfall = need_eff / have_eff if have_eff else float("inf")
    return {
        "band": precision_result.get("band"),
        "required_n_per_arm": float(need_per_arm),
        "required_effective_n": float(need_eff),
        "available_treated": ceiling.get("n_treated_max"),
        "available_control": ceiling.get("n_control_max"),
        "available_effective_n": float(have_eff),
        "shortfall_factor": float(shortfall),
        "feasible": bool(need_eff <= have_eff),
    }


# ---------------------------------------------------------------------------------
# registration — written BEFORE the extraction, never rewritten afterwards
# ---------------------------------------------------------------------------------
def registration(cohort_min_year: int = 2004, control: str = "no_inscrito") -> dict:
    """The pre-registered analysis, as a dict. ``write_registration`` persists it.

    Everything here is decided before a single parcel-year is extracted. The project's whole
    track record is pre-registered predictions being confirmed or refuted on the record
    (plan.md §7b.1); a decision rule chosen after seeing the coefficient is not a rule.
    """
    full = {**PRE_SUBWINDOWS, **WINDOWS}
    pl_a, pl_b = list(PRE_SUBWINDOWS)
    m = amplification_factor("W99", POST_WINDOWS, pl_a, pl_b, full)
    return {
        "registered_on": "2026-08-11",
        "plan": "docs/RESULTS.md §7 (the v3 reopening)",
        "primary_decision_rule": (
            "An effect is REPORTED only if the 95 % CI of (headline - M * placebo) "
            "excludes zero, at the derived M. Otherwise the outcome is NOT-SEPARABLE: "
            "the effect cannot be separated from pre-existing trend. NOT-SEPARABLE is a "
            "complete, publishable outcome and is written up to the same standard."),
        "primary_outcome": (
            "window-mean prob_PERENNIAL (W-D9 — the probability, not the thresholded "
            "class). Thresholded share is secondary."),
        "primary_contrast": {"pre": "W99", "post": POST_WINDOWS,
                             "also_reported_separately": POST_WINDOWS},
        "placebo": {"pre": pl_a, "post": pl_b,
                    "years": {pl_a: PRE_SUBWINDOWS[pl_a], pl_b: PRE_SUBWINDOWS[pl_b]},
                    "why": ("both windows lie entirely before the declaration year of the "
                            f"reg_year >= {cohort_min_year} cohort (R4)")},
        "amplification_factor_M": m,
        "M_derivation": ("headline horizon / placebo horizon, from window midpoints — "
                         f"{horizon_years('W99', POST_WINDOWS, full)} y / "
                         f"{horizon_years(pl_a, [pl_b], full)} y"),
        "correction": "corrected = headline - M * placebo; "
                      "se = sqrt(se_h^2 + M^2 * se_p^2)  [cov assumed 0]",
        "sensitivity_curve_M": list(SENSITIVITY_M),
        "sensitivity_curve_is_always_reported": True,
        "restrictions": ["R1 at-risk (PETT ANNUAL)", "R2 both observations dated",
                         "R3 cadastre after declaration and before the post-windows",
                         f"R4 pre-window entirely before declaration (reg_year >= "
                         f"{cohort_min_year})",
                         "R5 compared within department x declaration-year cohort"],
        "control_definition": control,
        "estimator": ("parcel FE + window FE + department x POST, SEs clustered by 5 km "
                      "region_id"),
        "model": "runs/all_peru/selected_model.json -> lightgbm_nometa_nolat_aug_yleak10",
        "sensitivity_model_arms": ["lightgbm_nometa_nolat", "ltae (negative control only)"],
        "G2_is_a_diagnostic_not_a_gate": (
            "common support, training-set membership and shared-region coverage are known "
            "to fail on the existing panel (RESULTS.md §10.3). They are measured and "
            "reported, not remedied by redesigning the sample."),
        "shares_use_sample_weight": True,
        "locked_test": "UNSPENT — not touched by this study",
    }


def write_registration(path: Path | str | None = None, cohort_min_year: int = 2004,
                       control: str = "no_inscrito") -> Path:
    """Persist the registration. **Refuses to overwrite** — that is the point of it."""
    p = Path(path) if path is not None else proc() / "did2_registration.json"
    reg = registration(cohort_min_year=cohort_min_year, control=control)
    if p.exists():
        old = json.loads(p.read_text())
        # compare through JSON so a tuple/list round-trip is not read as an edit
        if old != json.loads(json.dumps(reg)):
            raise FileExistsError(
                f"{p} exists and differs from the registration being written. A "
                f"registration is not editable after the fact — use a new path if this is "
                f"a genuinely different study.")
        print(f"registration already written and identical: {p}")
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(reg, f, indent=2)
    print(f"wrote {p}  (M = {reg['amplification_factor_M']:.1f})")
    return p


# ---------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------
def run(preds: Path | str, parcels: Path | str, tenure: Path | str, tag: str,
        cohort_min_year: int = 2004, control: str = "no_inscrito",
        apply_r4: bool = True, save: bool = True) -> dict:
    """Run the whole design on one panel and gate it. Returns the verdict dict.

    ``apply_r4=False`` reproduces the **pilot** (calendar windows for every parcel, placebo
    W99->W04), which is kept only so §9.6's numbers stay reproducible under N-D7. It is not a
    valid specification: for a parcel declared in 1999, W04 may be entirely post-treatment.
    """
    df = load_inputs(preds, parcels, tenure)
    spec_pre = PRE_SUBWINDOWS if apply_r4 else {"W99": WINDOWS["W99"], "W04": WINDOWS["W04"]}
    pre_a, pre_b = list(spec_pre)
    min_sub = MIN_YEARS_SUB if apply_r4 else MIN_YEARS

    d, rep = restrict(df, cohort_min_year if apply_r4 else 0, control=control,
                      apply_r2r3=apply_r4)
    full_spec = {**spec_pre, **{w: WINDOWS[w] for w in ["W99"] + POST_WINDOWS}}

    tab_sub = window_means(d, spec_pre, min_sub)
    tab_main = window_means(d, {w: WINDOWS[w] for w in ["W99"] + POST_WINDOWS}, MIN_YEARS)
    tab = pd.concat([tab_sub, tab_main], ignore_index=True)

    g1 = placebo(tab, pre_a, pre_b, full_spec, dept_window_fe=apply_r4)
    g2 = integrity(d, tab, {w: WINDOWS[w] for w in ["W99"] + POST_WINDOWS})
    g3a = precision(d, full_spec, pre_a, pre_b, min_sub)
    head = did(tab, "W99", POST_WINDOWS, dept_window_fe=apply_r4)
    g3b = {"mde": G3B_MDE, "se_headline": head["se"],
           "powered": bool(head["se"] == head["se"] and 2.80 * head["se"] <= G3B_MDE)}

    per_window = (tab.groupby(["window", "treat"], observed=True)["p_mean"].mean()
                  .unstack().rename(columns={0.0: "control", 1.0: "treated"}))
    if {"control", "treated"}.issubset(per_window.columns):
        per_window["difference"] = per_window["treated"] - per_window["control"]

    verdict = {
        "tag": tag, "preds": str(preds), "apply_r4": apply_r4,
        "cohort_min_year": cohort_min_year, "control_policy": control,
        "restrictions": rep,
        "G1_placebo": g1,
        "G2_integrity": g2,
        "G3a_precision": g3a,
        "G3b_power": g3b,
        "headline_W99_to_post": head,
        "per_window_means": per_window.reset_index().to_dict("records"),
        "estimate_licensed": bool(g1["verdict"] == "PASS" and g2["pass"]),
    }
    if save:
        d_out = proc()
        with open(d_out / f"did_gates_{tag}.json", "w") as f:
            json.dump(verdict, f, indent=2, default=float)
        tab.to_parquet(d_out / f"did_window_table_{tag}.parquet", index=False)
        per_window.reset_index().to_csv(d_out / f"did_series_{tag}.csv", index=False)
        print(f"wrote did_gates_{tag}.json / did_window_table_{tag}.parquet / "
              f"did_series_{tag}.csv to {d_out}")
    return verdict


def _weighted_share(g: pd.DataFrame, col: str, thr: float = 0.5) -> float:
    """Sample-weighted thresholded share. Any *share* reported here must use the weight."""
    w = g["sample_weight"].astype(float) if "sample_weight" in g.columns else 1.0
    v = (g[col].astype(float) >= thr).astype(float)
    return float(np.average(v, weights=w)) if np.ndim(w) else float(v.mean())


def run_corrected(preds: Path | str, parcels: Path | str, tenure: Path | str, tag: str,
                  cohort_min_year: int = 2004, control: str = "no_inscrito",
                  n_boot: int = 0, save: bool = True,
                  export_share: tuple[float, float] = (0.573, 0.772)) -> dict:
    """The v3 estimate: placebo, headline, **corrected effect + sensitivity curve**, decision.

    Order is deliberate and enforced by the code path: **the placebo is estimated first**.
    A headline computed before its pre-trend is a number nobody can un-see.

    ``export_share`` is §6.4's (export-only, export+mixed) split of PERENNIAL, applied to the
    corrected coefficient so no sentence about *export* crops is written against an
    undiscounted number.

    Writes ``did2_*`` artifacts under ``proc()`` and **refuses to overwrite** — the v2 run's
    ``did_*`` files are the record of why the study stopped.
    """
    df = load_inputs(preds, parcels, tenure)
    d, rep = restrict(df, cohort_min_year, control=control, apply_r2r3=True)
    pl_a, pl_b = list(PRE_SUBWINDOWS)
    full_spec = {**PRE_SUBWINDOWS, **{w: WINDOWS[w] for w in ["W99"] + POST_WINDOWS}}

    tab = pd.concat([window_means(d, PRE_SUBWINDOWS, MIN_YEARS_SUB),
                     window_means(d, {w: WINDOWS[w] for w in ["W99"] + POST_WINDOWS},
                                  MIN_YEARS)], ignore_index=True)

    # 1. the placebo, first and on its own.
    g1 = placebo(tab, pl_a, pl_b, full_spec, dept_window_fe=True)
    # 2. the headline.
    head = did(tab, "W99", POST_WINDOWS, dept_window_fe=True)
    # 5. each post-window separately — W19 is where the density artefact is worst.
    per_post = {w: did(tab, "W99", [w], dept_window_fe=True) for w in POST_WINDOWS}

    # 3. the correction and the curve.
    m = amplification_factor("W99", POST_WINDOWS, pl_a, pl_b, full_spec)
    boot = (bootstrap_covariance(tab, "W99", POST_WINDOWS, pl_a, pl_b, n_boot=n_boot)
            if n_boot else {"n_boot_ok": 0, "cov": float("nan"), "corr": float("nan")})
    corr = corrected_effect(head, g1, m)
    curve = sensitivity_curve(head, g1, ms=tuple(SENSITIVITY_M) + (m,))
    curve = curve.drop_duplicates("m").sort_values("m").reset_index(drop=True)
    # 4. the decision under the registered rule, at the derived M.
    dec = decision(corr)

    # secondary outcome: the thresholded share, and the weighted series.
    thr_tab = tab.copy()
    thr_tab["p_mean"] = (thr_tab["p_mean"] >= 0.5).astype(float)
    head_thr = did(thr_tab, "W99", POST_WINDOWS, dept_window_fe=True)
    g1_thr = did(thr_tab, pl_a, [pl_b], dept_window_fe=True)
    corr_thr = corrected_effect(head_thr, g1_thr, m)

    # R5's regression form, as a robustness arm: the declaration-year cohort gets its own
    # time effect on top of the department's. The registered primary is the specification
    # above; this says whether the campaign-wave confound is doing any of the work.
    cohort_col = "cohort" if "cohort" in tab.columns else "reg_year"
    r5 = {"pre_post": did(tab, "W99", POST_WINDOWS, dept_window_fe=True,
                          extra_post_fe=cohort_col),
          "placebo": did(tab, pl_a, [pl_b], dept_window_fe=True,
                         extra_post_fe=cohort_col)}
    r5["corrected"] = corrected_effect(r5["pre_post"], r5["placebo"], m)

    series = (tab.groupby(["window", "treat"], observed=True)
              .apply(lambda g: pd.Series({
                  "p_mean": float(np.average(
                      g["p_mean"],
                      weights=g["sample_weight"].astype(float)
                      if "sample_weight" in g.columns else None)),
                  "share_thresholded_weighted": _weighted_share(g, "p_mean"),
                  "n": int(len(g))}), include_groups=False)
              .reset_index())

    out = {
        "tag": tag, "preds": str(preds), "cohort_min_year": cohort_min_year,
        "control_policy": control, "restrictions": rep,
        "M": m, "M_derivation": "headline horizon / placebo horizon, from window midpoints",
        "placebo": g1,
        "headline": head,
        "headline_by_post_window": per_post,
        "corrected": corr,
        "sensitivity_curve": curve.to_dict("records"),
        "decision": dec,
        "secondary_thresholded_share": {"placebo": g1_thr, "headline": head_thr,
                                        "corrected": corr_thr},
        "robustness_cohort_post_fe": r5,
        "export_discounted": {
            "export_only_share": export_share[0], "export_plus_mixed_share": export_share[1],
            "coef_export_only": corr["coef"] * export_share[0],
            "ci_export_only": [corr["ci_low"] * export_share[0],
                               corr["ci_high"] * export_share[0]],
            "coef_export_plus_mixed": corr["coef"] * export_share[1],
            "note": ("§6.4: PERENNIAL is 57.3 % export / 19.9 % mixed / 22.8 % domestic. "
                     "No sentence about export crops may use the undiscounted number.")},
        "G2_integrity_diagnostic": integrity(
            d, tab, {w: WINDOWS[w] for w in ["W99"] + POST_WINDOWS}),
        "bootstrap_covariance_diagnostic": boot,
        "series": series.to_dict("records"),
    }
    if save:
        dout = proc()
        gates_p = dout / f"did2_gates_{tag}.json"
        for p in (gates_p, dout / f"did2_window_table_{tag}.parquet",
                  dout / f"did2_curve_{tag}.csv", dout / f"did2_series_{tag}.csv"):
            if p.exists():
                raise FileExistsError(f"{p} exists — use a new tag, never overwrite a run")
        with open(gates_p, "w") as f:
            json.dump(out, f, indent=2, default=float)
        tab.to_parquet(dout / f"did2_window_table_{tag}.parquet", index=False)
        curve.to_csv(dout / f"did2_curve_{tag}.csv", index=False)
        series.to_csv(dout / f"did2_series_{tag}.csv", index=False)
        print(f"wrote did2_gates_{tag}.json / did2_window_table_{tag}.parquet / "
              f"did2_curve_{tag}.csv / did2_series_{tag}.csv to {dout}")
    return out


def cross_sectional_contrast(preds: Path | str, parcels: Path | str, tenure: Path | str,
                             tag: str, windows: tuple[str, str] = ("W99", "W19"),
                             save: bool = True) -> dict:
    """⚠️ **DESCRIPTIVE COMPANION, NOT AN ESTIMATE.** INSCRITO vs NO INSCRITO, by window.

    This is the design the project started with and does **not** use: compare parcels that were
    already registered at declaration against parcels that were not, and read the perennial
    share. It is computed and reported because the number is asked for, and because *why* it is
    uninterpretable is itself a result. Three reasons it cannot carry a causal claim:

    * **The classifier's error is differential by tenure.** Among at-risk (PETT-``ANNUAL``)
      parcels the true perennial share at the label year is ~0 by construction, so the measured
      W99 gap **is** the false-positive-rate gap (§8.5: −4.4 pp raw). Any endpoint gap is that
      artefact plus the effect, and the endpoint error structure is unmeasured.
    * **Registration at declaration is not random** — it is entangled with declaration year,
      which is entangled with baseline perennial propensity, and with region, which is
      entangled with crop suitability.
    * **The endpoint level is where the model is least trustworthy** (observation density
      collapses; both tenure groups roughly double into W19).

    The **per-department** breakdown is returned and printed alongside the national figure and
    is not optional: the contrast's *sign* is department-specific, so a pooled number hides a
    quantity that does not exist.
    """
    df = load_inputs(preds, parcels, tenure)
    d = df[~df.get("abstained", pd.Series(False, index=df.index)).astype(bool)]
    d = d[d[PROB_COL].notna() & (d["pett_label"] == "ANNUAL") & d["tenure"].notna()].copy()
    d["treat"] = (d["tenure"] == "INSCRITO").astype(float)      # NB: tenure, not treatment
    spec = {w: WINDOWS[w] for w in windows}
    tab = window_means(d, spec, MIN_YEARS)

    def _w(g: pd.DataFrame, col: str) -> float:
        w = (g["sample_weight"].astype(float) if "sample_weight" in g.columns
             else pd.Series(1.0, index=g.index))
        return float(np.average(g[col], weights=w))

    rows = []
    for (win, ten), g in tab.groupby(["window", "treat"], observed=True):
        rows.append({"window": win, "tenure": "INSCRITO" if ten else "NO INSCRITO",
                     "n": int(len(g)), "prob_mean": _w(g, "p_mean"),
                     "share_thresholded": _w(g.assign(
                         _t=(g["p_mean"] >= 0.5).astype(float)), "_t")})
    nat = pd.DataFrame(rows)

    gaps = {}
    for col in ("prob_mean", "share_thresholded"):
        p = nat.pivot(index="window", columns="tenure", values=col)
        gaps[col] = {w: float(p.loc[w, "INSCRITO"] - p.loc[w, "NO INSCRITO"])
                     for w in windows if w in p.index}
        gaps[col]["change"] = gaps[col].get(windows[1], float("nan")) - \
            gaps[col].get(windows[0], float("nan"))

    by_dept = []
    for dept, gd in tab.groupby("dept", observed=True):
        rec: dict = {"dept": dept, "n_parcels": int(gd["COD_PREDIO"].nunique())}
        ok = True
        for w in windows:
            s = gd[gd["window"] == w]
            if s["treat"].nunique() < 2:
                ok = False
                break
            rec[f"{w}_gap"] = _w(s[s.treat == 1], "p_mean") - _w(s[s.treat == 0], "p_mean")
        if ok:
            rec["gap_change"] = rec[f"{windows[1]}_gap"] - rec[f"{windows[0]}_gap"]
            by_dept.append(rec)
    dept_df = pd.DataFrame(by_dept).sort_values("n_parcels", ascending=False)

    fe = did(tab, windows[0], [windows[1]], dept_window_fe=True)
    out = {
        "tag": tag, "preds": str(preds), "windows": list(windows),
        "national": nat.to_dict("records"), "gaps": gaps,
        "by_dept": dept_df.to_dict("records"),
        "n_depts": int(len(dept_df)),
        "n_depts_inscrito_higher": {
            w: int((dept_df[f"{w}_gap"] > 0).sum()) for w in windows} if len(dept_df) else {},
        "n_depts_gap_moves_to_inscrito": int((dept_df["gap_change"] > 0).sum())
        if len(dept_df) else 0,
        "parcel_fe_did_on_tenure_at_declaration": fe,
        "IS_NOT_CAUSAL": (
            "Descriptive only. The W99 gap among at-risk parcels is the classifier's "
            "differential false-positive rate (§8.5), not agronomy; registration at "
            "declaration is not random; and the endpoint level is where observation density "
            "is worst. The sign is department-specific — read by_dept, not the pooled row."),
    }
    if save:
        dout = proc()
        for p in (dout / f"xsec_tenure_{tag}.json", dout / f"xsec_tenure_by_dept_{tag}.csv"):
            if p.exists():
                raise FileExistsError(f"{p} exists — use a new tag")
        with open(dout / f"xsec_tenure_{tag}.json", "w") as f:
            json.dump(out, f, indent=2, default=float)
        dept_df.to_csv(dout / f"xsec_tenure_by_dept_{tag}.csv", index=False)
        print(f"wrote xsec_tenure_{tag}.json / xsec_tenure_by_dept_{tag}.csv to {dout}")
    return out


def print_cross_sectional(v: dict) -> None:
    w0, w1 = v["windows"]
    print(f"\n=== ⚠️ DESCRIPTIVE cross-sectional tenure contrast: {v['tag']} ===")
    print("    NOT an estimate — see IS_NOT_CAUSAL in the JSON.")
    print(pd.DataFrame(v["national"]).to_string(index=False))
    g = v["gaps"]["prob_mean"]
    print(f"\nprobability  gap {w0} {g[w0]:+.4f} -> {w1} {g[w1]:+.4f}  "
          f"(change {g['change']:+.4f})")
    s = v["gaps"]["share_thresholded"]
    print(f"thresh share gap {w0} {s[w0]:+.4f} -> {w1} {s[w1]:+.4f}  "
          f"(change {s['change']:+.4f})")
    fe = v["parcel_fe_did_on_tenure_at_declaration"]
    print(f"\nparcel-FE DiD on tenure-at-declaration: {fe['coef']:+.4f} "
          f"(se {fe['se']:.4f}, p {fe['p']:.3f})")
    print(f"\nper department ({v['n_depts']} with both tenure groups): "
          f"INSCRITO reads higher in {v['n_depts_inscrito_higher'][w0]} at {w0} and "
          f"{v['n_depts_inscrito_higher'][w1]} at {w1}; "
          f"gap moves toward INSCRITO in {v['n_depts_gap_moves_to_inscrito']}.")
    print(pd.DataFrame(v["by_dept"]).to_string(index=False))


def print_corrected(v: dict) -> None:
    r = v["restrictions"]
    print(f"\n=== tenure DiD v3 (pre-trend corrected): {v['tag']} ===")
    print(f"cohort reg_year >= {v['cohort_min_year']}   control: {v['control_policy']}")
    print(f"  treated {r['n_treated']:,}   control {r['n_control']:,}")
    p, h, c = v["placebo"], v["headline"], v["corrected"]
    print(f"\n1. PLACEBO {p['contrast']} ({p['horizon_years']:.1f} y): {p['coef']:+.4f} "
          f"(se {p['se']:.4f}, p {p['p']:.3f})  CI [{p['ci_low']:+.4f}, {p['ci_high']:+.4f}]"
          f"  [{p['verdict']} under the v2 equivalence gate]")
    print(f"2. HEADLINE W99->{'+'.join(POST_WINDOWS)}: {h['coef']:+.4f} "
          f"(se {h['se']:.4f}, p {h['p']:.3f}, n_treated {h['n_treated']})")
    for w, r2 in v["headline_by_post_window"].items():
        print(f"     W99->{w}: {r2['coef']:+.4f} (se {r2['se']:.4f}, p {r2['p']:.3f})")
    print(f"3. CORRECTED at M = {v['M']:.1f}: {c['coef']:+.4f} (se {c['se']:.4f})  "
          f"95% CI [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]")
    print("   sensitivity curve:")
    for row in v["sensitivity_curve"]:
        print(f"     M={row['m']:>4.1f}  {row['coef']:+.4f}  se {row['se']:.4f}  "
              f"CI [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
              f"  {'excludes 0' if row['excludes_zero'] else ''}")
    print(f"4. DECISION: {v['decision']['decision']}   ({v['decision']['rule']})")
    e = v["export_discounted"]
    print(f"7. export-discounted (x{e['export_only_share']}): "
          f"{e['coef_export_only']:+.4f}  CI [{e['ci_export_only'][0]:+.4f}, "
          f"{e['ci_export_only'][1]:+.4f}]")


def print_verdict(v: dict) -> None:
    r = v["restrictions"]
    print(f"\n=== tenure DiD: {v['tag']} ===")
    print(f"R4 applied: {v['apply_r4']}   control: {v['control_policy']}   "
          f"cohort reg_year >= {v['cohort_min_year'] if v['apply_r4'] else 'n/a'}")
    print("attrition: " + "  ".join(
        f"{k.replace('n_', '')} {r[k]:,}" for k in
        ("n_start", "n_R1_at_risk", "n_R2_dated", "n_R3_ordered", "n_R4_clean_pre", "n_final")
        if k in r))
    print(f"  treated {r['n_treated']:,}  control {r['n_control']:,}")
    g1 = v["G1_placebo"]
    print(f"\nG1 PLACEBO {g1['contrast']} ({g1['horizon_years']:.1f} y): "
          f"{g1['coef']:+.4f} (se {g1['se']:.4f})  "
          f"95% CI [{g1['ci_low']:+.4f}, {g1['ci_high']:+.4f}]  band +/-{g1['band']:.4f}")
    print(f"  => {g1['verdict']}   (n_treated {g1['n_treated']}, "
          f"per decade {g1['coef_per_decade']:+.4f})")
    g2 = v["G2_integrity"]
    print(f"\nG2 INTEGRITY  max SMD {g2['max_smd']:.3f} (<{G2_MAX_SMD}) "
          f"{'PASS' if g2['pass_common_support'] else 'FAIL'}"
          f" | attrition gap {g2['max_attrition_gap']:.3f} (<{G2_MAX_RATE_GAP}) "
          f"{'PASS' if g2['pass_attrition'] else 'FAIL'}")
    tm = g2.get("train_membership")
    if tm:
        print(f"  train membership gap {tm['gap']:+.3f} "
              f"{'PASS' if g2['pass_train_membership'] else 'FAIL'}"
              f" | shared regions {g2['shared_region_share']:.3f} "
              f"{'PASS' if g2['pass_shared_regions'] else 'FAIL'}")
    if "baseline_p_mean" in g2:
        b = g2["baseline_p_mean"]
        print(f"  [reported, never trimmed on] baseline p_mean "
              f"treated {b['treated']:.4f} control {b['control']:.4f} gap {b['gap']:+.4f}")
    g3 = v["G3a_precision"]
    if "ladder" in g3:
        print(f"\nG3a PRECISION  SD(delta) {g3['sd_delta']:.4f}  deff_cluster "
              f"{g3['deff_cluster']:.2f}  band +/-{g3['band']:.4f}")
        for row in g3["ladder"]:
            print(f"   n/arm {row['n_per_arm']:6,d} -> SE {row['expected_se']:.5f}  "
                  f"CI +/-{row['ci_half_width']:.5f}  "
                  f"{'fits' if row['fits_band'] else 'TOO WIDE'}")
        print(f"   required n per arm to fit the band: {g3['n_per_arm_required']:,.0f}")
    h = v["headline_W99_to_post"]
    print(f"\nheadline W99->{'+'.join(POST_WINDOWS)}: {h['coef']:+.4f} "
          f"(se {h['se']:.4f}, p {h['p']:.3f}, n_treated {h['n_treated']})")
    print(f"\n=> estimate licensed: {v['estimate_licensed']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--parcels", type=Path, default=None)
    ap.add_argument("--tenure", type=Path, default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cohort-min-year", type=int, default=2004)
    ap.add_argument("--control", default="no_inscrito", choices=["no_inscrito", "any"])
    ap.add_argument("--no-r4", action="store_true", help="reproduce the pilot (invalid)")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    v = run(a.preds, a.parcels or proc() / "panel_parcels.parquet",
            a.tenure or proc() / "tenure_two_period.parquet", tag=a.tag,
            cohort_min_year=a.cohort_min_year, control=a.control,
            apply_r4=not a.no_r4, save=not a.no_save)
    print_verdict(v)
    raise SystemExit(0 if v["G1_placebo"]["verdict"] != "FAIL" else 1)


if __name__ == "__main__":
    main()

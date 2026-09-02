"""Multi-year **windows** — the estimand forced by the Phase-7 gate failure (RESULTS.md §5).

Both panels failed S5 flicker (0.43-0.98 vs a 0.15 criterion) on every architecture, which
killed the per-parcel annual trajectory but not the research question — that only ever
needed a state at two well-supported points and a between-group contrast. Four ideas, in
order of application:

* **M1 aggregate probabilities, not classes.** Per parcel per 5-year window take
  ``mean(prob_PERENNIAL)`` over the non-abstained years (≥ ``min_years``), then threshold
  once. The modal ``pred_label`` turns a parcel sitting stably at p ≈ 0.45 into a coin flip.
* **M2 the baseline is observed, not predicted.** The PETT declaration is the ~1997-2006
  state, so the at-risk pool is conditioned on it (``pett_label == "ANNUAL"``) rather than
  predicted from 1996-98 imagery — where the El Niño confound and the year↔label confound
  live.
* **M3 identify within window.** A difference between tenure groups in the same window, so
  year effects, sensor era and coverage difference out.
* **M4 the PETT-``PERENNIAL`` pool is the internal control.** Drift or an L5→L7 ramp moves
  it and the at-risk pool together.

Everything here is diagnostic — whether the window estimand is stable enough to carry the
deliverable (:func:`run_diagnostic`, T3), and whether classifier error is non-differential
with respect to tenure (:func:`tenure_error_test`, T1). Neither produces a trend or a
per-parcel conversion date.

Run with::

    CC_PROC=data/processed/all_peru uv run python -m crop_classifier.cli archive windows \\
        --preds data/processed/all_peru/panel_predictions_nolat.parquet --tag nolat
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crop_classifier.paths import proc

# Non-overlapping 5-year windows (W-D2). Start 1999: 1996-98 is where every documented
# failure lives, and W-D3 takes the baseline from the PETT label instead. End 2023: L7
# acquisitions stop and the panel is TM/ETM+ only (W-D7).
WINDOWS: tuple[tuple[str, int, int], ...] = (
    ("W99", 1999, 2003),
    ("W04", 2004, 2008),
    ("W09", 2009, 2013),
    ("W14", 2014, 2018),
    ("W19", 2019, 2023),
)
WINDOW_NAMES = [w[0] for w in WINDOWS]
WINDOW_MID = {name: (lo + hi) / 2 for name, lo, hi in WINDOWS}

MIN_YEARS = 3          # W-D2: a window needs this many observed years to qualify
THRESHOLD = 0.5        # the calibrated operating point; probabilities are temperature-scaled

PROB_COL = "prob_PERENNIAL"

# T3 acceptance criteria (RESULTS.md §5, gate T3).
W1_MAX_MULTI_CHANGE = 0.10      # P(>=2 window-state changes) on the selected model
W2_MAX_SLOPE_PER_DECADE = 0.01  # |slope| of the PETT-PERENNIAL control, share/decade
W3_MAX_ASYMMETRY = 0.01         # |up - down| adjacent-window disagreement
W4_MAX_NEG_STEP = 0.005         # largest allowed decrease in the at-risk pool's series


# --- window assignment + aggregation (M1) ---
def window_of(year: int | float) -> str | None:
    for name, lo, hi in WINDOWS:
        if lo <= year <= hi:
            return name
    return None


def window_table(preds: pd.DataFrame, min_years: int = MIN_YEARS,
                 threshold: float = THRESHOLD, prob_col: str = PROB_COL,
                 drop_abstained: bool = True) -> pd.DataFrame:
    """Parcel-year predictions -> one row per ``(COD_PREDIO, window)``.

    ``p_mean`` is the mean probability over the window's observed years (primary outcome,
    W-D9); ``perennial`` is it thresholded once. Windows with fewer than ``min_years``
    observed years are dropped, not filled — an abstained year and an unchanged year differ.
    """
    df = preds.copy()
    if drop_abstained and "abstained" in df.columns:
        df = df[~df["abstained"].astype(bool)]
    df = df[df[prob_col].notna()]
    df["window"] = df["year"].map(window_of)
    df = df[df["window"].notna()]

    carry = [c for c in ("pett_label", "sample_weight", "area_ha", "dept", "tenure",
                         "reg_year", "region_id", "split") if c in df.columns]
    agg = {"p_mean": (prob_col, "mean"), "n_years": (prob_col, "size")}
    agg.update({c: (c, "first") for c in carry})
    out = (df.groupby(["COD_PREDIO", "window"], observed=True)
             .agg(**agg).reset_index())
    out = out[out["n_years"] >= min_years].copy()
    out["perennial"] = out["p_mean"] >= threshold
    out["window_idx"] = out["window"].map({n: i for i, n in enumerate(WINDOW_NAMES)})
    if "sample_weight" not in out.columns:
        out["sample_weight"] = 1.0
    return out.sort_values(["COD_PREDIO", "window_idx"]).reset_index(drop=True)


# --- stability: annual flicker vs window-state changes ---
def _changes(states: np.ndarray) -> int:
    return int((states[1:] != states[:-1]).sum()) if len(states) > 1 else 0


def window_changes(wt: pd.DataFrame) -> pd.DataFrame:
    """Per parcel: how many times its window state changes across observed windows."""
    g = wt.sort_values("window_idx").groupby("COD_PREDIO", observed=True)
    out = g.agg(n_windows=("perennial", "size"),
                n_changes=("perennial", lambda s: _changes(s.to_numpy())),
                pett_label=("pett_label", "first") if "pett_label" in wt.columns
                else ("perennial", "first"))
    return out.reset_index()


def change_summary(wt: pd.DataFrame, min_windows: int = 2) -> dict:
    """Fraction of parcels with 0 / ≥1 / ≥2 window-state changes — the M1 stability claim.

    Restricted to parcels observed in ≥ ``min_windows`` windows; a parcel seen once cannot
    change and would dilute the rate toward zero.
    """
    ch = window_changes(wt)
    ch = ch[ch["n_windows"] >= min_windows]
    if ch.empty:
        return {"n_parcels": 0}
    return {
        "n_parcels": int(len(ch)),
        "mean_windows_observed": float(ch["n_windows"].mean()),
        "zero_changes": float((ch["n_changes"] == 0).mean()),
        "ge1_change": float((ch["n_changes"] >= 1).mean()),
        "ge2_changes": float((ch["n_changes"] >= 2).mean()),
    }


def annual_flicker(preds: pd.DataFrame, drop_abstained: bool = True) -> dict:
    """The S5-style annual flicker, for comparison with the window rate.

    Two variants: the 3-class series (the gate's definition) and the perennial-vs-rest
    binarisation (this plan's estimand). Reported for PERENNIAL-labelled parcels and for all.
    """
    from crop_classifier.perennial import trajectories as TR

    df = preds.copy()
    if drop_abstained and "abstained" in df.columns:
        df = df[~df["abstained"].astype(bool)]
    df = df[df["pred_label"].notna()]
    out: dict = {}
    for name, col in (("3class", "pred_label"), ("perennial_vs_rest", "_bin")):
        d = df.copy()
        if col == "_bin":
            d["_bin"] = np.where(d["pred_label"] == "PERENNIAL", "PERENNIAL", "OTHER")
        arr, _, _, _ = TR.to_series(d, class_col=col)
        parcels = sorted(d["COD_PREDIO"].unique())
        flick = pd.Series(TR.flicker_rate(arr), index=parcels)
        lab = d.drop_duplicates("COD_PREDIO").set_index("COD_PREDIO").get("pett_label")
        out[name] = {"ALL": float(flick.mean())}
        if lab is not None:
            for k, v in flick.groupby(lab.reindex(flick.index)).mean().items():
                out[name][str(k)] = float(v)
    return out


# --- shares (M2/M3) and the noise floor (W3) ---
def share_by(wt: pd.DataFrame, by: list[str] | None = None,
             weighted: bool = True) -> pd.DataFrame:
    """Weighted share perennial per window, optionally split by ``by`` columns.

    Weighted by default and must stay so: the national sample allocates departments
    sqrt-proportionally, doubling the raw PERENNIAL share, so an unweighted share is about
    the sample not Peru (CLAUDE.md §8).
    """
    by = list(by or [])
    keys = by + ["window"]
    w = wt["sample_weight"].to_numpy(float) if weighted else np.ones(len(wt))
    d = wt.assign(_w=w, _num=w * wt["perennial"].to_numpy(float),
                  _pw=w * wt["p_mean"].to_numpy(float))
    g = d.groupby(keys, observed=True)
    out = g.agg(n=("perennial", "size"), w_sum=("_w", "sum"), num=("_num", "sum"),
                pw=("_pw", "sum")).reset_index()
    out["share"] = out["num"] / out["w_sum"]
    out["p_mean"] = out["pw"] / out["w_sum"]
    # binomial-ish SE on the effective sample size (design effect from the weights)
    neff = out["w_sum"] ** 2 / g["_w"].apply(lambda s: (s ** 2).sum()).to_numpy()
    out["n_eff"] = neff
    out["se"] = np.sqrt(out["share"] * (1 - out["share"]) / np.maximum(neff, 1))
    out["window_idx"] = out["window"].map({n: i for i, n in enumerate(WINDOW_NAMES)})
    return out.drop(columns=["w_sum", "num", "pw"]).sort_values(
        by + ["window_idx"]).reset_index(drop=True)


def adjacent_disagreement(wt: pd.DataFrame, restrict: pd.Series | None = None) -> dict:
    """Up/down state changes between adjacent windows — the noise floor.

    Any change estimate has to clear this. Symmetry is the tell: a noisy classifier moves
    parcels up and down at the same rate; a real conversion (or a drifting model) is
    directional.
    """
    d = wt if restrict is None else wt[restrict]
    piv = d.pivot_table(index="COD_PREDIO", columns="window", values="perennial")
    rows, ups, downs, ns = [], 0, 0, 0
    for a, b in zip(WINDOW_NAMES[:-1], WINDOW_NAMES[1:]):
        if a not in piv.columns or b not in piv.columns:
            continue
        pair = piv[[a, b]].dropna()
        if pair.empty:
            continue
        up = float(((~pair[a].astype(bool)) & pair[b].astype(bool)).mean())
        dn = float((pair[a].astype(bool) & (~pair[b].astype(bool))).mean())
        rows.append({"from": a, "to": b, "n": int(len(pair)), "up": up, "down": dn,
                     "disagreement": up + dn, "asymmetry": up - dn})
        ups += up * len(pair)
        downs += dn * len(pair)
        ns += len(pair)
    pooled_up = ups / ns if ns else np.nan
    pooled_down = downs / ns if ns else np.nan
    return {"pairs": rows, "n_pairs_total": int(ns), "up": pooled_up, "down": pooled_down,
            "disagreement": pooled_up + pooled_down,
            "asymmetry": pooled_up - pooled_down}


def density_confound_test(preds: pd.DataFrame, min_years: int = MIN_YEARS) -> dict:
    """Does the window probability track observation density rather than the land?

    With L5 retired and L7 SLC-off, the national panel averages ~24 clear observations per
    parcel-year in 2004-08 and ~13 in 2019-23. A weaker signal reverts toward the prior —
    falling probability on true perennials, rising on true annuals — which at share level is
    indistinguishable from a real conversion.

    Within parcel: regress the window-mean probability on ``log(n_valid_obs)`` after a
    parcel FE, by PETT label, SEs clustered by parcel. Positive on PETT-``PERENNIAL`` and
    negative on PETT-``ANNUAL`` is the compression signature.
    """
    import statsmodels.api as sm

    d = preds.copy()
    if "abstained" in d.columns:
        d = d[~d["abstained"].astype(bool)]
    d["window"] = d["year"].map(window_of)
    d = d[d["window"].notna() & d["n_valid_obs"].notna()]
    pw = (d.groupby(["COD_PREDIO", "window"], observed=True)
            .agg(p=(PROB_COL, "mean"), nobs=("n_valid_obs", "mean"),
                 n_years=(PROB_COL, "size"), pett_label=("pett_label", "first"))
            .reset_index())
    pw = pw[pw["n_years"] >= min_years]
    out: dict = {"n_parcel_windows": int(len(pw))}
    for lab, sub in pw.groupby("pett_label"):
        sub = sub[sub["nobs"] > 0]
        if sub["COD_PREDIO"].nunique() < 30:
            continue
        X = pd.DataFrame({"log_nobs": np.log(sub["nobs"].to_numpy(float))})
        gid = sub["COD_PREDIO"].to_numpy()
        Xd = X.groupby(gid).transform(lambda s: s - s.mean())
        yd = sub["p"].astype(float).groupby(gid).transform(lambda s: s - s.mean())
        r = sm.OLS(yd.to_numpy(), sm.add_constant(Xd)).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(gid)[0]})
        out[str(lab)] = {"coef": float(r.params.iloc[1]), "se": float(r.bse.iloc[1]),
                         "p": float(r.pvalues.iloc[1]), "n": int(len(sub))}
    dens = pw.groupby("window")["nobs"].mean()
    out["mean_n_valid_obs_by_window"] = {k: float(v) for k, v in dens.items()}
    if len(dens) >= 2:
        out["log_density_change_W99_to_W19"] = float(
            np.log(dens.get("W19", np.nan)) - np.log(dens.get("W99", np.nan)))
    per = out.get("PERENNIAL", {}).get("coef", np.nan)
    ann = out.get("ANNUAL", {}).get("coef", np.nan)
    out["compression_signature"] = bool(per > 0 and ann < 0)
    out["implied_control_drift"] = (float(per * out.get("log_density_change_W99_to_W19",
                                                        np.nan))
                                    if per == per else float("nan"))
    return out


def slope_per_decade(shares: pd.DataFrame) -> float:
    """OLS slope of ``share`` on window mid-year, expressed per decade."""
    s = shares.dropna(subset=["share"])
    if len(s) < 2:
        return float("nan")
    x = s["window"].map(WINDOW_MID).to_numpy(float)
    y = s["share"].to_numpy(float)
    return float(np.polyfit(x, y, 1)[0] * 10.0)


# --- T3 — the window diagnostic ---
def run_diagnostic(preds_path: Path | str, tag: str = "", tenure: pd.DataFrame | None = None,
                   min_years: int = MIN_YEARS, threshold: float = THRESHOLD,
                   balanced: bool = False, save: bool = True) -> dict:
    """Task T3: replicate the §1 M1/M2 window result on a panel and gate it on W1-W4.

    A failure here means the pivot does not survive and no GEE budget should be spent.

    ``balanced`` restricts to parcels that qualify in every window, separating a genuine
    level change from a changing parcel set — a diagnostic, not the default.
    """
    preds = pd.read_parquet(preds_path)
    if tenure is not None:
        preds = preds.merge(tenure[["COD_PREDIO", "tenure"]], on="COD_PREDIO", how="left")
    wt = window_table(preds, min_years=min_years, threshold=threshold)
    if balanced:
        n_w = wt.groupby("COD_PREDIO", observed=True).size()
        wt = wt[wt["COD_PREDIO"].isin(n_w[n_w == len(WINDOWS)].index)]

    at_risk = wt[wt["pett_label"] == "ANNUAL"]
    control = wt[wt["pett_label"] == "PERENNIAL"]
    by_label = share_by(wt, ["pett_label"])
    ctrl_series = by_label[by_label["pett_label"] == "PERENNIAL"]
    risk_series = by_label[by_label["pett_label"] == "ANNUAL"]

    adj = adjacent_disagreement(wt)
    changes = change_summary(wt)
    slope_ctrl = slope_per_decade(ctrl_series)

    steps = risk_series.sort_values("window_idx")["share"].diff().dropna()
    worst_neg = float(steps.min()) if len(steps) else float("nan")
    net = (float(risk_series.sort_values("window_idx")["share"].iloc[-1]
                 - risk_series.sort_values("window_idx")["share"].iloc[0])
           if len(risk_series) >= 2 else float("nan"))

    w1 = bool(changes.get("ge2_changes", 1.0) < W1_MAX_MULTI_CHANGE)
    w2 = bool(abs(slope_ctrl) < W2_MAX_SLOPE_PER_DECADE)
    w3 = bool(abs(adj["asymmetry"]) < W3_MAX_ASYMMETRY and net > adj["disagreement"] * 0)
    w3_net_exceeds = bool(abs(net) > abs(adj["asymmetry"]))
    w4 = bool(worst_neg >= -W4_MAX_NEG_STEP)

    verdict = {
        "preds": str(preds_path), "tag": tag, "balanced": balanced,
        "density_confound": density_confound_test(preds, min_years=min_years),
        "n_parcels": int(wt["COD_PREDIO"].nunique()),
        "n_parcel_windows": int(len(wt)),
        "min_years": min_years, "threshold": threshold,
        "annual_flicker": annual_flicker(preds),
        "window_changes": changes,
        "adjacent_disagreement": adj,
        "at_risk_series": risk_series[["window", "n", "share", "p_mean", "se"]]
        .to_dict("records"),
        "control_series": ctrl_series[["window", "n", "share", "p_mean", "se"]]
        .to_dict("records"),
        "control_slope_per_decade": slope_ctrl,
        "at_risk_net_change": net,
        "at_risk_worst_negative_step": worst_neg,
        "W1_multi_change_lt_0.10": w1,
        "W2_control_flat": w2,
        "W3_symmetric_noise_floor": w3,
        "W3_net_change_exceeds_asymmetry": w3_net_exceeds,
        "W4_monotone_at_risk": w4,
        "pass": bool(w1 and w2 and w3 and w4),
        "n_at_risk_parcels": int(at_risk["COD_PREDIO"].nunique()),
        "n_control_parcels": int(control["COD_PREDIO"].nunique()),
    }
    if "tenure" in wt.columns and wt["tenure"].notna().any():
        ten = share_by(wt[wt["pett_label"] == "ANNUAL"], ["tenure"])
        verdict["at_risk_by_tenure"] = ten[["tenure", "window", "n", "share", "p_mean",
                                            "se"]].to_dict("records")

    if save:
        suf = f"_{tag}" if tag else ""
        d = proc()
        with open(d / f"window_diagnostic{suf}.json", "w") as f:
            json.dump(verdict, f, indent=2, default=float)
        wt.to_parquet(d / f"window_table{suf}.parquet", index=False)
        by_label.to_csv(d / f"window_shares{suf}.csv", index=False)
        print(f"wrote window_diagnostic{suf}.json / window_table{suf}.parquet / "
              f"window_shares{suf}.csv to {d}")
    return verdict


def print_diagnostic(v: dict) -> None:
    print(f"\n=== window diagnostic: {v['tag'] or v['preds']} ===")
    print(f"{v['n_parcels']:,} parcels, {v['n_parcel_windows']:,} parcel-windows "
          f"(>= {v['min_years']} observed years each)")
    af = v["annual_flicker"]
    print(f"annual flicker  3-class PERENNIAL {af['3class'].get('PERENNIAL', float('nan')):.3f}"
          f"   perennial-vs-rest {af['perennial_vs_rest'].get('PERENNIAL', float('nan')):.3f}")
    c = v["window_changes"]
    print(f"window-state changes over {c.get('mean_windows_observed', 0):.1f} windows: "
          f"zero {c.get('zero_changes', float('nan')):.3f}  "
          f">=1 {c.get('ge1_change', float('nan')):.3f}  "
          f">=2 {c.get('ge2_changes', float('nan')):.3f}")
    a = v["adjacent_disagreement"]
    print(f"adjacent-window disagreement: up {a['up']:.4f}  down {a['down']:.4f}  "
          f"asymmetry {a['asymmetry']:+.4f}")
    for name, key in (("at-risk (PETT ANNUAL)", "at_risk_series"),
                      ("control (PETT PERENNIAL)", "control_series")):
        s = " ".join(f"{r['window']} {r['share']:.3f}" for r in v[key])
        print(f"{name:26s} {s}")
    dc = v.get("density_confound", {})
    if dc.get("PERENNIAL"):
        print(f"density confound (within parcel, dp/dlog n_obs): "
              f"PERENNIAL {dc['PERENNIAL']['coef']:+.4f} (p {dc['PERENNIAL']['p']:.1e})  "
              f"ANNUAL {dc.get('ANNUAL', {}).get('coef', float('nan')):+.4f}  "
              f"| log-density W99->W19 {dc.get('log_density_change_W99_to_W19', float('nan')):+.3f}"
              f" => implied control drift {dc.get('implied_control_drift', float('nan')):+.4f}")
    print(f"control slope {v['control_slope_per_decade']:+.4f}/decade   "
          f"at-risk net {v['at_risk_net_change']:+.4f}   "
          f"worst negative step {v['at_risk_worst_negative_step']:+.4f}")
    for k in ("W1_multi_change_lt_0.10", "W2_control_flat", "W3_symmetric_noise_floor",
              "W4_monotone_at_risk"):
        print(f"  {'PASS' if v[k] else 'FAIL'}  {k}")
    print(f"  => {'PASS' if v['pass'] else 'FAIL'}")


# --- T1 — is classifier error non-differential with respect to tenure? ---
def tenure_error_test(preds_cv: pd.DataFrame, parcels: pd.DataFrame,
                      tenure: pd.DataFrame, target: str = "PERENNIAL",
                      save: bool = True, tag: str = "") -> dict:
    """Task T1. The estimator is a difference in predicted share between tenure groups.

    That equals the difference in true share only if the classifier errs the same way on
    both::

        observed_share = true_share · sensitivity + (1 − true_share) · (1 − specificity)

    Non-differential error attenuates the contrast toward zero but preserves its sign, and
    one error matrix corrects it; differential error leaves a bias of unknown sign.

    Two tests, since INSCRITO parcels are genuinely different (larger, more often perennial):

    1. marginal sensitivity / false-positive rate / predicted-vs-true share per group;
    2. ``correct ~ tenure + true_class + log(area)`` — an LPM with region FE absorbed, SEs
       clustered by region, plus a logit with department FE as a check. The tenure
       coefficient is the test.

    ⚠️ This tests error at the label year (~1999, TM/ETM+) while the estimand sits in
    2019-2023. With no endpoint ground truth, non-differential error at the endpoint is
    assumed, not verified.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    prob_cols = [c for c in preds_cv.columns if c.startswith("prob_")]
    classes = [c[len("prob_"):] for c in prob_cols]
    df = preds_cv.copy()
    df["pred_label"] = np.array(classes)[df[prob_cols].to_numpy().argmax(1)]
    keep = [c for c in ("COD_PREDIO", "region_id", "dept", "area_ha", "label")
            if c in parcels.columns]
    df = df.merge(parcels[keep], on="COD_PREDIO", how="left", suffixes=("", "_p"))
    df = df.merge(tenure[["COD_PREDIO", "tenure"]], on="COD_PREDIO", how="left")
    n_all = len(df)
    df = df[df["tenure"].notna()].copy()
    true_col = "label" if "label" in df.columns else "label_p"
    df["is_target"] = df[true_col] == target
    df["pred_target"] = df["pred_label"] == target
    df["correct"] = df["pred_label"] == df[true_col]
    df["log_area"] = np.log(df["area_ha"].clip(lower=1e-3))

    rows = []
    for grp, sub in df.groupby("tenure"):
        pos, neg = sub[sub["is_target"]], sub[~sub["is_target"]]
        rows.append({
            "tenure": grp, "n": len(sub),
            "true_share": float(sub["is_target"].mean()),
            "pred_share": float(sub["pred_target"].mean()),
            "sensitivity": float(pos["pred_target"].mean()) if len(pos) else np.nan,
            "fpr": float(neg["pred_target"].mean()) if len(neg) else np.nan,
            "specificity": float(1 - neg["pred_target"].mean()) if len(neg) else np.nan,
            "accuracy": float(sub["correct"].mean()),
            "mean_area_ha": float(sub["area_ha"].mean()),
        })
    marg = pd.DataFrame(rows)

    # stratified by area quartile — the likeliest confounder (clean-pixel count)
    df["area_q"] = pd.qcut(df["area_ha"], 4, labels=["q1", "q2", "q3", "q4"],
                           duplicates="drop")
    strat_rows = []
    for (q, grp), sub in df.groupby(["area_q", "tenure"], observed=True):
        pos, neg = sub[sub["is_target"]], sub[~sub["is_target"]]
        strat_rows.append({"area_q": str(q), "tenure": grp, "n": len(sub),
                           "sensitivity": float(pos["pred_target"].mean())
                           if len(pos) else np.nan,
                           "fpr": float(neg["pred_target"].mean()) if len(neg) else np.nan})
    strat = pd.DataFrame(strat_rows)
    piv = strat.pivot(index="area_q", columns="tenure", values="sensitivity")
    max_dsens = (float((piv.iloc[:, 0] - piv.iloc[:, 1]).abs().max())
                 if piv.shape[1] == 2 else float("nan"))

    # --- the operative bias: differential FALSE POSITIVES inside the at-risk pool ---
    # The estimand is the perennial share of PETT-ANNUAL parcels; their true perennial share
    # at the label year is ~0, so their measured share is essentially the classifier's
    # false-positive rate. A tenure gap in that rate transfers one-for-one into the contrast
    # — this is the number that decides interpretability, not the accuracy test.
    pool = df[df[true_col] == "ANNUAL"]
    fp_rows = []
    for grp, sub in pool.groupby("tenure"):
        fp_rows.append({"tenure": grp, "n": len(sub),
                        "fp_rate": float(sub["pred_target"].mean())})
    fp = pd.DataFrame(fp_rows)
    fp_gap = (float(fp.loc[fp.tenure == "INSCRITO", "fp_rate"].iloc[0]
                    - fp.loc[fp.tenure == "NO INSCRITO", "fp_rate"].iloc[0])
              if len(fp) == 2 else float("nan"))

    # --- conditional test: LPM with region FE absorbed, SEs clustered by region ---
    d = df.dropna(subset=["region_id", "log_area"]).copy()
    d["insc"] = (d["tenure"] == "INSCRITO").astype(float)
    X = pd.get_dummies(d[[true_col]], drop_first=True, dtype=float)
    X["insc"] = d["insc"].values
    X["log_area"] = d["log_area"].values
    y = d["correct"].astype(float)
    # within-transform on region (absorbs the FE without materialising dummies)
    grp = d["region_id"].values
    Xd = X.groupby(grp).transform(lambda s: s - s.mean())
    yd = y.groupby(grp).transform(lambda s: s - s.mean())
    lpm = sm.OLS(yd, sm.add_constant(Xd, has_constant="add")).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(grp)[0]})

    # logit with department FE as a robustness check on the linear approximation
    logit = None
    try:
        logit = smf.logit(f"correct ~ insc + C({true_col}) + log_area + C(dept)",
                          data=d.assign(correct=d["correct"].astype(int))).fit(disp=0)
    except Exception as e:                                   # pragma: no cover
        print(f"  logit did not converge ({e}); LPM only")

    # conditional FP ~ tenure within region and size
    pool_d = pool.dropna(subset=["region_id", "log_area"]).copy()
    Xf = pd.DataFrame({"insc": (pool_d["tenure"] == "INSCRITO").astype(float).values,
                       "log_area": pool_d["log_area"].values}, index=pool_d.index)
    gf = pool_d["region_id"].values
    Xfd = Xf.groupby(gf).transform(lambda s: s - s.mean())
    yfd = pool_d["pred_target"].astype(float).groupby(gf).transform(lambda s: s - s.mean())
    fp_lpm = sm.OLS(yfd, sm.add_constant(Xfd, has_constant="add")).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(gf)[0]})

    out = {
        "tag": tag, "target": target,
        "n_with_tenure": int(len(df)), "n_total": int(n_all),
        "tenure_coverage": float(len(df) / max(n_all, 1)),
        "marginal": marg.to_dict("records"),
        "at_risk_false_positive": fp.to_dict("records"),
        "at_risk_fp_gap_INSCRITO_minus_NO": fp_gap,
        "at_risk_fp_gap_conditional": float(fp_lpm.params["insc"]),
        "at_risk_fp_gap_conditional_se": float(fp_lpm.bse["insc"]),
        "at_risk_fp_gap_conditional_p": float(fp_lpm.pvalues["insc"]),
        "by_area_quartile": strat.to_dict("records"),
        "max_abs_dsensitivity_within_area_q": max_dsens,
        "lpm_tenure_coef": float(lpm.params["insc"]),
        "lpm_tenure_se": float(lpm.bse["insc"]),
        "lpm_tenure_p": float(lpm.pvalues["insc"]),
        "lpm_n": int(len(d)), "lpm_n_regions": int(d["region_id"].nunique()),
    }
    if logit is not None:
        out.update(logit_tenure_coef=float(logit.params["insc"]),
                   logit_tenure_p=float(logit.pvalues["insc"]))
    out["pass_accuracy"] = bool(out["lpm_tenure_p"] > 0.05)
    out["pass_sensitivity"] = bool(np.isnan(max_dsens) or max_dsens < 0.05)
    # The false-positive leg is not in the plan's original criterion; added because it is
    # the leg the estimand rests on. Criterion: the conditional gap must be small against
    # the ~2 pp differential the design is powered for.
    out["pass_false_positive"] = bool(out["at_risk_fp_gap_conditional_p"] > 0.05
                                      or abs(out["at_risk_fp_gap_conditional"]) < 0.01)
    out["pass"] = bool(out["pass_accuracy"] and out["pass_sensitivity"]
                       and out["pass_false_positive"])
    if save:
        suf = f"_{tag}" if tag else ""
        with open(proc() / f"tenure_error_test{suf}.json", "w") as f:
            json.dump(out, f, indent=2, default=float)
        marg.to_csv(proc() / f"tenure_error_marginal{suf}.csv", index=False)
        strat.to_csv(proc() / f"tenure_error_by_area{suf}.csv", index=False)
        print(f"wrote tenure_error_test{suf}.json (+2 csv) to {proc()}")
    return out


def print_tenure_error(v: dict) -> None:
    print(f"\n=== T1 non-differential error by tenure ({v['tag'] or 'cv'}) ===")
    print(f"tenure resolved for {v['n_with_tenure']:,} of {v['n_total']:,} held-out "
          f"parcels ({v['tenure_coverage']:.1%})")
    print(pd.DataFrame(v["marginal"]).to_string(index=False))
    print(f"max |Δsensitivity| within area quartile: "
          f"{v['max_abs_dsensitivity_within_area_q']:.4f}  (criterion < 0.05)")
    print("at-risk (true ANNUAL) false-positive rate: "
          + "  ".join(f"{r['tenure']} {r['fp_rate']:.4f} (n={r['n']:,})"
                      for r in v["at_risk_false_positive"]))
    print(f"  gap INSCRITO - NO INSCRITO {v['at_risk_fp_gap_INSCRITO_minus_NO']:+.4f}; "
          f"conditional on region+size {v['at_risk_fp_gap_conditional']:+.4f} "
          f"(se {v['at_risk_fp_gap_conditional_se']:.4f}, "
          f"p {v['at_risk_fp_gap_conditional_p']:.3f})")
    print(f"LPM (region FE, clustered SE): tenure coef {v['lpm_tenure_coef']:+.4f} "
          f"(se {v['lpm_tenure_se']:.4f}, p {v['lpm_tenure_p']:.3f})")
    if "logit_tenure_coef" in v:
        print(f"logit (dept FE):              tenure coef {v['logit_tenure_coef']:+.4f} "
              f"(p {v['logit_tenure_p']:.3f})")
    print(f"  => {'PASS' if v['pass'] else 'FAIL'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--min-years", type=int, default=MIN_YEARS)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--tenure", type=Path, default=None,
                    help="tenure_by_predio.parquet (allperu.tenure)")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    ten = pd.read_parquet(a.tenure) if a.tenure else None
    v = run_diagnostic(a.preds, tag=a.tag, tenure=ten, min_years=a.min_years,
                       threshold=a.threshold, save=not a.no_save)
    print_diagnostic(v)


if __name__ == "__main__":
    main()

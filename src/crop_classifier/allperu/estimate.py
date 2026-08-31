"""T5 — the tenure contrast on the window estimand (RESULTS.md §5, gate T5).

The estimand, written down before anything runs:

    Population   parcels in the 14 linkable departments whose PETT declaration resolves to
                 ANNUAL (the at-risk pool), with a non-null `ESTADO en RRPP`, passing the
                 area/quality gates.
    Outcome      perennial_W19 — window-mean prob_PERENNIAL over 2019-2023 (>= 3 observed
                 years), thresholded at the calibrated operating point. The continuous window
                 mean is reported alongside and is the *primary* outcome (W-D9).
    Primary      share perennial in W19 by tenure at titling, with department and
                 registration-year fixed effects, weighted by sample_weight, SEs clustered by
                 region_id.
    Secondary    Δ = share(W19) − share(W99) by tenure — the difference-in-differences arm.
    Controls     reported every time: (a) the PETT-PERENNIAL pool's window series, which must
                 be flat; (b) adjacent-window disagreement, the noise floor; (c) the T1
                 tenure-by-error test.

**This is descriptive, not causal.** Tenure is observed once, at titling, and a farmer
planning a 20-year orchard has strong reason to register title first. Say so in every
write-up (§6.2). ``allperu.tenure.tenure_two_period`` now supplies a *second* dated
observation (~2011) which would support a two-period difference-in-differences; that is a
design change, not something this module quietly assumes.

⚠️ **This module refuses to run while the gates fail.** T1, T2 and T3 all returned FAIL on
2026-08-10 (see docs/RESULTS.md §6), and producing a contrast from a panel that
failed validation is not a weaker finding, it is a wrong one. ``--force`` exists so the
refusal can be overridden deliberately and visibly, and it stamps ``gates_failed`` into every
artifact it writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crop_classifier.allperu import windows as W
from crop_classifier.paths import proc
from crop_classifier.perennial import area as AR

GATE_FILES = ("window_diagnostic_{tag}.json", "tenure_error_test_{tag}.json",
              "loyo_summary_{tag}.json")


def gate_status(tag: str = "nolat") -> dict:
    """Read the T1/T2/T3 verdicts from the workspace. Missing == not run == not passed."""
    out = {}
    for pat in GATE_FILES:
        f = proc() / pat.format(tag=tag)
        alt = proc() / pat.format(tag=f"{tag}_cv")
        f = f if f.exists() else alt
        if not f.exists():
            out[pat.format(tag=tag)] = None
            continue
        with open(f) as fh:
            d = json.load(fh)
        out[f.name] = bool(d.get("pass", d.get("verdict", {}).get("pass", False)))
    return out


def error_matrix_from_cv(preds_cv: pd.DataFrame, classes: list[str]) -> np.ndarray:
    """Olofsson error matrix from the **pooled spatial-CV** confusion of the model in use.

    ⚠️ CV-derived, never test-derived. The matrix on disk from earlier work
    (``lightgbm_3c_final/test_confusion.csv``) belongs to a disqualified model, and no
    endpoint-year matrix exists anywhere in the project. CV-derived CIs carry the residual
    spatial leakage of §4.1 and are mildly optimistic — label them as such.
    """
    cols = [f"prob_{c}" for c in classes]
    y_pred = np.asarray(preds_cv[cols].to_numpy().argmax(1))
    y_true = preds_cv["y_true"].to_numpy() if "y_true" in preds_cv.columns else \
        preds_cv["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    return AR.error_matrix(y_true, y_pred, len(classes))


def tenure_contrast(wt: pd.DataFrame, window: str = "W19", outcome: str = "perennial",
                    fe: tuple[str, ...] = ("dept", "reg_year")) -> dict:
    """Weighted linear model of the outcome on tenure with FE and region-clustered SEs.

    A linear probability model, deliberately: the estimand is a difference in *shares*, the
    weights are sampling weights (which a logit would not interpret the same way), and the
    coefficient is the contrast itself rather than an odds ratio needing a margins step.
    """
    import statsmodels.api as sm

    d = wt[wt["window"] == window].dropna(subset=["tenure", "region_id"]).copy()
    if d.empty:
        raise ValueError(f"no rows for window {window} with tenure and region_id")
    y = (d[outcome].astype(float) if outcome in d.columns else d["p_mean"].astype(float))
    X = pd.DataFrame({"insc": (d["tenure"] == "INSCRITO").astype(float).to_numpy()},
                     index=d.index)
    for f in fe:
        if f in d.columns and d[f].notna().any():
            dm = pd.get_dummies(d[f].astype(str), prefix=f, drop_first=True, dtype=float)
            X = pd.concat([X, dm.set_index(d.index)], axis=1)
    w = d["sample_weight"].to_numpy(float) if "sample_weight" in d.columns else None
    model = sm.WLS(y.to_numpy(), sm.add_constant(X.to_numpy(dtype=float)), weights=w)
    res = model.fit(cov_type="cluster",
                    cov_kwds={"groups": pd.factorize(d["region_id"])[0]})
    return {"window": window, "outcome": outcome, "n": int(len(d)),
            "n_clusters": int(d["region_id"].nunique()),
            "coef_INSCRITO": float(res.params[1]), "se": float(res.bse[1]),
            "p": float(res.pvalues[1]),
            "ci95": [float(res.conf_int()[1][0]), float(res.conf_int()[1][1])],
            "fixed_effects": [f for f in fe if f in d.columns]}


def did_contrast(wt: pd.DataFrame, pre: str = "W99", post: str = "W19",
                 outcome: str = "p_mean") -> dict:
    """Secondary: Δ(post − pre) by tenure, on parcels observed in **both** windows.

    Weaker than the primary because it needs the baseline window *predicted* rather than
    observed — which is what W-D3 was designed to avoid — but it controls for any
    tenure-correlated baseline propensity the fixed effects miss.
    """
    import statsmodels.api as sm

    piv = wt.pivot_table(index="COD_PREDIO", columns="window", values=outcome)
    meta = wt.drop_duplicates("COD_PREDIO").set_index("COD_PREDIO")
    both = piv[[pre, post]].dropna()
    d = meta.loc[both.index]
    keep = d["tenure"].notna() & d["region_id"].notna()
    both, d = both[keep.values], d[keep]
    y = (both[post] - both[pre]).to_numpy(float)
    X = pd.DataFrame({"insc": (d["tenure"] == "INSCRITO").astype(float).to_numpy()})
    w = d["sample_weight"].to_numpy(float) if "sample_weight" in d.columns else None
    res = sm.WLS(y, sm.add_constant(X.to_numpy(dtype=float)), weights=w).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["region_id"])[0]})
    return {"pre": pre, "post": post, "outcome": outcome, "n": int(len(y)),
            "mean_change_NO_INSCRITO": float(y[X["insc"].to_numpy() == 0].mean()),
            "mean_change_INSCRITO": float(y[X["insc"].to_numpy() == 1].mean()),
            "did_coef": float(res.params[1]), "se": float(res.bse[1]),
            "p": float(res.pvalues[1])}


def corrected_shares(wt: pd.DataFrame, preds_cv: pd.DataFrame, classes: list[str],
                     window: str = "W19") -> pd.DataFrame:
    """Olofsson-corrected class shares with 95 % CIs for one window, weighted."""
    d = wt[wt["window"] == window]
    prob = np.zeros((len(d), len(classes)), dtype=float)
    j = classes.index("PERENNIAL")
    prob[:, j] = d["p_mean"].to_numpy(float)
    rest = (1.0 - prob[:, j]) / max(len(classes) - 1, 1)
    for i in range(len(classes)):
        if i != j:
            prob[:, i] = rest
    n = error_matrix_from_cv(preds_cv, classes)
    area = (d["area_ha"].to_numpy(float) if "area_ha" in d.columns
            else np.ones(len(d), dtype=float))
    w = (d["sample_weight"].to_numpy(float) if "sample_weight" in d.columns
         else np.ones(len(d)))
    out = AR.estimate_all(prob, area, classes, n=n, weights=w)
    out["window"] = window
    out["error_matrix_source"] = "pooled spatial CV (label year) — CV-derived, not test"
    return out


def run(preds_path: Path, tenure_path: Path, run_dir: Path, tag: str = "nolat",
        force: bool = False, save: bool = True) -> dict:
    """The full T5 bundle. Refuses to run unless T1/T2/T3 all passed (see module docstring)."""
    gates = gate_status(tag)
    failed = [k for k, v in gates.items() if not v]
    if failed and not force:
        raise RuntimeError(
            "T5 refuses to run: the following gates are failed or missing — "
            f"{failed}. RESULTS.md §5 says stop and report. Re-run the gates, or pass "
            "force=True to override deliberately (the override is recorded in the output).")

    tenure = pd.read_parquet(tenure_path)
    preds = pd.read_parquet(preds_path).merge(
        tenure[["COD_PREDIO", "tenure", "reg_year"]], on="COD_PREDIO", how="left")
    parcels = pd.read_parquet(proc() / "modeling_parcels.parquet",
                              columns=["COD_PREDIO", "dept", "region_id"])
    preds = preds.merge(parcels, on="COD_PREDIO", how="left")
    wt = W.window_table(preds)
    at_risk = wt[wt["pett_label"] == "ANNUAL"]

    preds_cv = pd.read_parquet(Path(run_dir) / "preds_cv.parquet")
    with open(proc() / "label_map.json") as f:
        label_map = json.load(f)
    classes = [c for c, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    out = {
        "tag": tag, "gates": gates, "gates_failed": bool(failed),
        "forced": bool(force and failed),
        "n_at_risk": int(at_risk["COD_PREDIO"].nunique()),
        "primary_thresholded": tenure_contrast(at_risk, outcome="perennial"),
        "primary_continuous": tenure_contrast(at_risk, outcome="p_mean"),
        "secondary_did": did_contrast(at_risk),
        "control_series": W.share_by(wt[wt["pett_label"] == "PERENNIAL"]).to_dict("records"),
        "noise_floor": W.adjacent_disagreement(wt),
        "shares_by_tenure": W.share_by(at_risk, ["tenure"]).to_dict("records"),
        "olofsson_W19": corrected_shares(at_risk, preds_cv, classes).to_dict("records"),
        "limitations": [
            "Descriptive, not causal: tenure is observed once, at titling (window_plan §6.2).",
            "Non-differential classifier error is verified at the LABEL year only; at the "
            "2019-23 endpoint it is assumed, not verified — there is no endpoint ground "
            "truth anywhere in this project (T1's limit).",
            "The Olofsson error matrix is CV-derived, so its CIs carry the residual spatial "
            "leakage of §4.1 and are mildly optimistic.",
            "PERENNIAL is not the same as export — see allperu.export_crops.",
            "Declared woody non-crop parcels are excluded from training but not from the "
            "world; at inference they read PERENNIAL.",
        ],
    }
    if save:
        with open(proc() / f"window_estimate_{tag}.json", "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"wrote window_estimate_{tag}.json to {proc()}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--tenure", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True, help="model run dir with preds_cv")
    ap.add_argument("--tag", default="nolat")
    ap.add_argument("--force", action="store_true",
                    help="run even though a gate failed — recorded in the output")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    print(json.dumps(run(a.preds, a.tenure, a.run, tag=a.tag, force=a.force,
                         save=not a.no_save), indent=2, default=float))


if __name__ == "__main__":
    main()

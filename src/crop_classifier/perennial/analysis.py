"""End-to-end trend analysis: series -> smoothing -> transitions -> areas (plan §9).

Ties ``trajectories.py`` (series, smoothing, change detection) and ``area.py`` (the three
area estimators) together into the artifacts the report needs, and runs the §9.5
sensitivity battery. A trend that survives all five perturbations is a finding; a trend
that flips under any of them is an artefact, and this module's job is to make that
distinction visible rather than to produce one confident-looking line.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from crop_classifier.paths import proc
from crop_classifier.perennial import area as AR
from crop_classifier.perennial import trajectories as TR


def _prob_matrix(df: pd.DataFrame, classes: list[str]) -> np.ndarray:
    cols = [f"prob_{c}" for c in classes]
    return np.nan_to_num(df[cols].to_numpy(dtype=float))


def build_series(panel_preds: pd.DataFrame, classes: list[str],
                 smooth: bool = True) -> dict:
    """Wide class-id array + observed mask, raw and mode-filtered."""
    raw, observed, classes, years = TR.to_series(panel_preds, classes=classes)
    parcels = sorted(panel_preds["COD_PREDIO"].unique())
    out = {"raw": raw, "observed": observed, "classes": classes, "years": years,
           "parcels": parcels}
    if smooth:
        out["smoothed"] = TR.mode_filter(raw)
    return out


def flicker_report(series: dict, panel_preds: pd.DataFrame) -> pd.DataFrame:
    """Flicker rate per PETT label — criterion S5 (< 15 % for PERENNIAL parcels).

    A high flicker rate means the per-year features are too weak and the trend analysis
    must not proceed on raw predictions.
    """
    flick = TR.flicker_rate(series["raw"])
    lab = (panel_preds.drop_duplicates("COD_PREDIO")
           .set_index("COD_PREDIO")["pett_label"])
    df = pd.DataFrame({"COD_PREDIO": series["parcels"], "flicker": flick})
    df["pett_label"] = df["COD_PREDIO"].map(lab)
    rep = (df.groupby("pett_label", dropna=False)
           .agg(n=("flicker", "size"), flicker_rate=("flicker", "mean"))
           .reset_index())
    rep.loc[len(rep)] = ["ALL", len(df), float(df["flicker"].mean())]
    return rep


def area_trend(panel_preds: pd.DataFrame, classes: list[str],
               error_matrix: np.ndarray | None = None,
               use_weights: bool = True) -> pd.DataFrame:
    """Per-year area by all three estimators (§9.4), expanded to the population."""
    rows = []
    for y, sub in panel_preds.groupby("year"):
        sub = sub[~sub["abstained"].astype(bool)]
        if sub.empty:
            continue
        prob = _prob_matrix(sub, classes)
        w = sub["sample_weight"].to_numpy(float) if use_weights else None
        df = AR.estimate_all(prob, sub["area_ha"].to_numpy(float), classes,
                             n=error_matrix, weights=w)
        df["year"] = y
        df["n_parcels"] = len(sub)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def run(panel_preds: pd.DataFrame, classes: list[str],
        error_matrix: np.ndarray | None = None,
        min_duration: int = 3, save: bool = True) -> dict:
    """The full §9 bundle: series, flicker, transitions, area trend, realism checks."""
    out_dir = proc()
    series = build_series(panel_preds, classes)
    smoothed = series["smoothed"]

    flick = flicker_report(series, panel_preds)
    print("\nflicker rate by PETT label (criterion S5: < 0.15 for PERENNIAL):")
    print(flick.to_string(index=False))

    tr_raw = TR.detect_transitions(series["raw"], series["years"], classes,
                                   series["parcels"], min_duration)
    tr = TR.detect_transitions(smoothed, series["years"], classes, series["parcels"],
                               min_duration)
    tr = TR.add_confidence(tr, panel_preds)
    print(f"\ntransitions: {len(tr):,} (smoothed) vs {len(tr_raw):,} (raw), "
          f"min_duration={min_duration}")

    tm = TR.transition_matrix(tr, classes)
    print("\ntransition matrix (rows = from, cols = to):")
    print(tm.to_string())
    # realism check: the export-expansion narrative predicts asymmetry (§9.3)
    checks = {}
    if {"ANNUAL", "PERENNIAL"} <= set(classes):
        a2p = int(tm.loc["ANNUAL", "PERENNIAL"])
        p2a = int(tm.loc["PERENNIAL", "ANNUAL"])
        checks["annual_to_perennial"] = a2p
        checks["perennial_to_annual"] = p2a
        checks["asymmetry_ratio"] = float(a2p / p2a) if p2a else float("inf")
        print(f"\nANNUAL->PERENNIAL {a2p:,} vs PERENNIAL->ANNUAL {p2a:,} "
              f"(a symmetric matrix means you are measuring noise)")

    trend = area_trend(panel_preds, classes, error_matrix)
    share = TR.class_share_by_year(smoothed, series["years"], classes,
                                   area_ha=_align(panel_preds, series["parcels"],
                                                  "area_ha"),
                                   weights=_align(panel_preds, series["parcels"],
                                                  "sample_weight"))

    if save:
        tr.to_parquet(out_dir / "transitions.parquet", index=False)
        tm.to_csv(out_dir / "transition_matrix.csv")
        trend.to_csv(out_dir / "area_trend.csv", index=False)
        share.to_csv(out_dir / "class_share_by_year.csv", index=False)
        flick.to_csv(out_dir / "flicker_report.csv", index=False)
        with open(out_dir / "realism_checks.json", "w") as f:
            json.dump(checks, f, indent=2)
        print(f"\nwrote transitions/area_trend/class_share/flicker to {out_dir}")
    return {"series": series, "transitions": tr, "transition_matrix": tm,
            "area_trend": trend, "share": share, "flicker": flick, "checks": checks}


def _align(panel_preds: pd.DataFrame, parcels: list[str], col: str) -> np.ndarray:
    """Per-parcel column in the same order as the series rows."""
    s = panel_preds.drop_duplicates("COD_PREDIO").set_index("COD_PREDIO")[col]
    return s.reindex(parcels).to_numpy(dtype=float)


# --- §9.5 sensitivity battery ---
def perennial_share_trend(share: pd.DataFrame) -> dict:
    """Slope + direction of the PERENNIAL share over time (criterion S6 is *direction*)."""
    s = share[share["class"] == "PERENNIAL"].dropna(subset=["share"])
    if len(s) < 3:
        return {"slope_per_decade": np.nan, "direction": "insufficient"}
    slope = float(np.polyfit(s["year"], s["share"], 1)[0]) * 10
    return {"slope_per_decade": slope,
            "direction": "increasing" if slope > 0 else "decreasing",
            "first_year_share": float(s.iloc[0]["share"]),
            "last_year_share": float(s.iloc[-1]["share"])}


def sensitivity_table(variants: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per §9.5 variant: does the headline trend keep its sign?

    Variants to pass: sugarcane as PERENNIAL (D1), PASTURE_FALLOW split (D2), rule-based
    vs ML model, raw vs smoothed, MapBiomas class-21 handling.
    """
    rows = []
    for name, share in variants.items():
        t = perennial_share_trend(share)
        rows.append({"variant": name, **t})
    df = pd.DataFrame(rows)
    if len(df):
        signs = set(df["direction"]) - {"insufficient"}
        df.attrs["robust"] = len(signs) <= 1
    return df

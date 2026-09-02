"""The Phase-7 validation gate: four diagnostics that must pass before any trend.

The panel is a model trained on one ~1998 titling snapshot asked to predict 28 years;
these answer whether that transfer is real or whether the "trend" is the sensor record,
the El Nino, or classifier noise. Nothing downstream (trajectories, transitions, areas) is
meaningful unless they pass — a gate, not an appendix.

1. **S4 temporal transfer** (plan §7.3) — accuracy vs the PETT label at label-year ``+/-k``,
   locked-test only. Criterion: accuracy at ``|k| <= 3`` within 0.10 of ``k = 0``. Real
   land-use change also drives the decay, so this is a *lower bound* on stability.
2. **S5 flicker** (plan §9.2) — fraction of parcels whose raw series changes class more
   than ``n_observed / 5`` times. Criterion: under 15 % for PERENNIAL parcels. High flicker
   means the minimum-duration rule would be hiding the problem, not solving it.
3. **Sensor/feature drift** (plan §7.4) — per-year panel distributions of index features
   *and* raw bands. Raw bands are more sensor-sensitive than ratio indices and both models
   consume them, so the §4.6 metadata ablation does not remove this exposure.
4. **The 1999+2000 -> 1998 El Nino confound test** (RESULTS.md §8) — registration year is
   confounded with the label (1998 is 0.4 % perennial vs 2000's 33.8 %), 1997-98 was the
   Piura El Nino, and 1996-98 are the panel's baseline. Holds region fixed while varying
   year; reports per-class recall as **test arm minus control arm**.

Thin years to flag rather than interpolate over: 1997 (El Nino, 48.3 % gate pass), 2009
(49.2 %), 2011 (43.3 %), 2012 (83.6 %, L7 SLC-off alone).

⛔ Must stay torch-free: diagnostic 4 fits LightGBM, and co-loading torch + lightgbm on
macOS segfaults (each bundles its own libomp).
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from crop_classifier.paths import proc
from crop_classifier.perennial import analysis as AN
from crop_classifier.perennial import panel as P

# Materially degraded coverage — flagged in every figure, never interpolated (§6.2/§6.3).
FLAG_YEARS = {1997: "El Nino, 48.3 % gate", 2009: "49.2 % gate",
              2011: "43.3 % gate", 2012: "83.6 % gate, L7 SLC-off alone"}

# Mission composition of the panel (PANEL_MISSIONS = {L5, L7}).
ERAS = [("L5 only", 1996, 1998), ("L5 + L7", 1999, 2011), ("L7 only", 2012, 2023)]

# Drift: three index features plus level (median) and seasonal swing (amp) of six raw
# bands — raw bands are where a TM/ETM+ calibration step shows up, ratio indices cancel it.
INDEX_DRIFT = ["NDVI_median", "NDVI_amp", "BSI_max"]
RAW_BANDS = ["B", "G", "R", "NIR", "SWIR1", "SWIR2"]
RAW_DRIFT = [f"{b}_{s}" for b in RAW_BANDS for s in ("median", "amp")]

S4_TOLERANCE = 0.10      # accuracy at |k| <= 3 must stay within this of k = 0
S5_MAX_FLICKER = 0.15    # for PERENNIAL-labelled parcels


def era_of(year: int) -> str:
    for name, lo, hi in ERAS:
        if lo <= year <= hi:
            return name
    return "?"


# --- 1. S4 — temporal transfer ---
def _s4_composition_adjusted(panel_preds: pd.DataFrame, classes: list[str],
                             k_max: int = 5) -> pd.DataFrame:
    """Per-class recall vs ``k``, plus two composition-controlled accuracies.

    ``k`` bins are different parcel mixes (label year is confounded with label, §8), so a
    raw drop can be worse transfer or a harder mix. ``acc_std_k0prior`` re-weights recalls
    to the ``k = 0`` prior; ``bal_acc`` is unweighted mean recall. Neither replaces the
    criterion, which is on raw accuracy.
    """
    parcels = gpd.read_parquet(proc() / "modeling_parcels.parquet")
    ref = parcels[["COD_PREDIO", "label", "year", "split"]].rename(
        columns={"year": "label_year"})
    df = panel_preds.merge(ref, on="COD_PREDIO", how="inner")
    df = df[(df["split"] == "test") & ~df["abstained"].astype(bool)
            & df["pred_label"].notna()].copy()
    df["k"] = df["year"] - df["label_year"]
    df = df[df["k"].abs() <= k_max]
    if df.empty:
        return pd.DataFrame()
    p0 = (df[df["k"] == 0]["label"].value_counts(normalize=True)
          .reindex(classes).to_numpy())
    rows = []
    for k, g in df.groupby("k"):
        rec = np.array([(g.loc[g["label"] == c, "pred_label"] == c).mean()
                        if (g["label"] == c).any() else np.nan for c in classes])
        rows.append({"k": int(k), "n": len(g),
                     "acc_raw": float((g["pred_label"] == g["label"]).mean()),
                     "acc_std_k0prior": float(np.nansum(rec * p0)),
                     "bal_acc": float(np.nanmean(rec)),
                     **{f"recall_{c}": (float(v) if v == v else np.nan)
                        for c, v in zip(classes, rec)}})
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def _s4_supplementary(panel_preds: pd.DataFrame, classes: list[str] | None,
                      k_max: int) -> dict:
    """Composition-controlled S4 views, folded into the verdict as diagnosis only."""
    if classes is None:
        return {}
    adj = _s4_composition_adjusted(panel_preds, classes, k_max)
    if adj.empty or 0 not in set(adj["k"]):
        return {}
    near = adj[adj["k"].abs() <= 3]
    a0, b0 = (float(adj.loc[adj["k"] == 0, c].iloc[0])
              for c in ("acc_std_k0prior", "bal_acc"))
    d_std = float((near["acc_std_k0prior"] - a0).abs().max())
    d_bal = float((near["bal_acc"] - b0).abs().max())
    return {
        "supplementary_per_k": adj.to_dict("records"),
        "supplementary_max_dev_prior_standardised": d_std,
        "supplementary_prior_standardised_pass": bool(d_std <= S4_TOLERANCE),
        "supplementary_max_dev_balanced_accuracy": d_bal,
        "supplementary_balanced_accuracy_pass": bool(d_bal <= S4_TOLERANCE),
        "supplementary_note":
            "Which parcels have a panel year at distance k depends on their label year, "
            "and label year is confounded with label (§8) — so k bins are not the same "
            "parcel mix. `prior_standardised` re-weights per-class recalls to the k=0 "
            "class prior; `balanced_accuracy` weights all classes equally. Diagnosis "
            "only: the criterion is defined on raw accuracy.",
    }


def s4_temporal_transfer(panel_preds: pd.DataFrame, k_max: int = 5,
                         classes: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """plan §7.3 on locked-test parcels. Returns ``(per-k table, verdict)``."""
    tt = P.temporal_transfer(panel_preds, test_only=True, k_max=k_max)
    if tt.empty or 0 not in set(tt["k"]):
        return tt, {"criterion": "S4", "pass": False,
                    "reason": "no k=0 row — panel does not cover any label year"}

    a0 = float(tt.loc[tt["k"] == 0, "accuracy"].iloc[0])
    near = tt[tt["k"].abs() <= 3]
    worst_i = (near["accuracy"] - a0).abs().idxmax()
    worst_k = int(near.loc[worst_i, "k"])
    worst_drop = a0 - float(near.loc[worst_i, "accuracy"])

    # Criterion is two-sided as written, but a rise above k=0 is not a transfer failure —
    # report the one-sided degradation alongside rather than folding it in.
    worst_fall = float(max(0.0, (a0 - near["accuracy"]).max()))
    verdict = {
        "criterion": "S4", "tolerance": S4_TOLERANCE,
        "accuracy_at_k0": a0, "n_at_k0": int(tt.loc[tt["k"] == 0, "n"].iloc[0]),
        "worst_k_within_3": worst_k,
        "worst_accuracy_within_3": float(near.loc[worst_i, "accuracy"]),
        "max_abs_deviation_within_3": abs(worst_drop),
        "signed_drop_at_worst": worst_drop,
        "max_degradation_within_3": worst_fall,
        "pass": bool(abs(worst_drop) <= S4_TOLERANCE),
        "degradation_only_pass": bool(worst_fall <= S4_TOLERANCE),
        **_s4_supplementary(panel_preds, classes, k_max),
        "note": "Real land-use change also contributes to the decay, so this is a lower "
                "bound on model stability. A cliff at one specific year is a sensor "
                "artefact, not land use. `pass` is the criterion as written (two-sided); "
                "`degradation_only_pass` ignores accuracy that rises above k=0, which is "
                "odd but is not a transfer failure.",
    }
    return tt, verdict


# --- 2. S5 — flicker ---
def _flicker_by_label(series: dict, panel_preds: pd.DataFrame, arr: np.ndarray) -> dict:
    """Flicker rate per PETT label for an arbitrary series array."""
    from crop_classifier.perennial import trajectories as TR
    lab = (panel_preds.drop_duplicates("COD_PREDIO")
           .set_index("COD_PREDIO")["pett_label"])
    df = pd.DataFrame({"COD_PREDIO": series["parcels"],
                       "flicker": TR.flicker_rate(arr)})
    df["pett_label"] = df["COD_PREDIO"].map(lab)
    out = df.groupby("pett_label", dropna=False)["flicker"].mean().to_dict()
    out["ALL"] = float(df["flicker"].mean())
    return {k: float(v) for k, v in out.items()}


def s5_flicker(panel_preds: pd.DataFrame, classes: list[str]) -> tuple[pd.DataFrame, dict,
                                                                      dict]:
    """plan §9.2. Returns ``(per-label table, verdict, series bundle)``.

    Criterion is on the raw series. Two supplementary views make a failure diagnosable:
    *smoothed* (after the gap-aware 3-year mode filter — clean smoothed + failing raw means
    single-year noise) and *flagged years dropped* (1997/2009/2011/2012 — if that fixes it,
    the failure is a coverage artefact). Neither changes the verdict.

    The series bundle is returned so the caller can reuse it downstream *only if the gate
    passes* — building it is the expensive part.
    """
    series = AN.build_series(panel_preds, classes)
    rep = AN.flicker_report(series, panel_preds)
    row = rep[rep["pett_label"] == "PERENNIAL"]
    per = float(row["flicker_rate"].iloc[0]) if len(row) else float("nan")

    keep = [i for i, y in enumerate(series["years"]) if y not in FLAG_YEARS]
    verdict = {
        "criterion": "S5", "threshold": S5_MAX_FLICKER,
        "flicker_PERENNIAL": per,
        "n_PERENNIAL": int(row["n"].iloc[0]) if len(row) else 0,
        "flicker_ALL": float(rep.loc[rep["pett_label"] == "ALL",
                                     "flicker_rate"].iloc[0]),
        "pass": bool(per == per and per < S5_MAX_FLICKER),
        "supplementary_smoothed": _flicker_by_label(series, panel_preds,
                                                    series["smoothed"]),
        "supplementary_flagged_years_dropped": _flicker_by_label(
            series, panel_preds, series["raw"][:, keep]),
        "note": "Flicker is computed on the RAW series. If it fails, the minimum-duration "
                "rule would be hiding the problem, not solving it. The two supplementary "
                "views are diagnosis only and do not change the verdict.",
    }
    return rep, verdict, series


# --- 3. sensor / feature drift (plan §7.4) ---
def balanced_parcels(years: list[int] | None = None) -> list[str]:
    """Parcels present in every non-flagged assembled year.

    Composition changes year to year (the coverage gate drops ~half the panel in 1997,
    2009, 2011), so a per-year statistic over "whatever survived" mixes radiometric drift
    with a changing parcel set. Flagged years are excluded from the intersection (not the
    reported series), or they shrink it to their own survivors.
    """
    years = years or P.DEFAULT_YEARS
    common: set[str] | None = None
    for y in years:
        if y in FLAG_YEARS:
            continue
        f = P.panel_dirs(y)[1] / "features_lightgbm.parquet"
        if not f.exists():
            continue
        ids = set(pd.read_parquet(f, columns=["COD_PREDIO"])["COD_PREDIO"])
        common = ids if common is None else (common & ids)
    return sorted(common or [])


def feature_drift(years: list[int] | None = None,
                  features: list[str] | None = None,
                  balanced: bool = True) -> pd.DataFrame:
    """Per-year panel-wide distribution of each drift feature, with mission era attached.

    Read straight from the per-year assembled bundles — the features the model consumes,
    not a re-derivation. ``balanced`` restricts every year to the same parcels
    (``balanced_parcels``) so a mission-boundary step cannot be a composition artefact;
    ``False`` is the raw all-survivors view. Both reported, balanced primary.
    """
    years = years or P.DEFAULT_YEARS
    features = features or (INDEX_DRIFT + RAW_DRIFT)
    keep = set(balanced_parcels(years)) if balanced else None
    rows = []
    for y in years:
        _, fd = P.panel_dirs(y)
        f = fd / "features_lightgbm.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f, columns=["COD_PREDIO"] + list(features))
        if keep is not None:
            df = df[df["COD_PREDIO"].isin(keep)]
        for feat in features:
            if feat not in df:
                continue
            v = df[feat].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            rows.append({"year": y, "era": era_of(y), "feature": feat, "n": len(v),
                         "mean": float(v.mean()), "std": float(v.std()),
                         "p10": float(np.percentile(v, 10)),
                         "median": float(np.median(v)),
                         "p90": float(np.percentile(v, 90)),
                         "flagged_year": y in FLAG_YEARS,
                         "balanced": bool(balanced)})
    return pd.DataFrame(rows)


def dispersion_by_era(drift: pd.DataFrame) -> pd.DataFrame:
    """Cross-parcel spread (p90 - p10) per feature per era.

    ``era_steps`` compares levels; a sensor change can leave the level alone but change the
    noise (fewer clear acquisitions -> amplitude estimated from fewer dates -> population
    spreads without the centre moving). That per-parcel noise is what becomes S5 flicker.
    """
    d = drift[~drift["flagged_year"]].copy()
    d["spread"] = d["p90"] - d["p10"]
    d["kind"] = np.where(d["feature"].isin(INDEX_DRIFT), "index",
                         np.where(d["feature"].str.endswith("_amp"),
                                  "raw_band_amp", "raw_band_median"))
    out = (d.groupby(["kind", "era"], observed=True)["spread"].mean()
           .unstack("era").reindex(columns=[e[0] for e in ERAS]))
    out["ratio_last_over_first"] = out.iloc[:, -1] / out.iloc[:, 0]
    return out.reset_index()


def era_steps(drift: pd.DataFrame) -> pd.DataFrame:
    """Size of the jump at each mission boundary, in units of the within-era spread.

    A step much larger than year-to-year variation inside the adjacent eras is a sensor
    artefact; one comparable to it is not distinguishable from weather. Flagged years are
    excluded — coverage failures, not radiometry.
    """
    rows = []
    d = drift[~drift["flagged_year"]]
    for feat, g in d.groupby("feature"):
        by_era = {e: sub.sort_values("year") for e, sub in g.groupby("era")}
        for (e1, _, hi1), (e2, lo2, _) in zip(ERAS[:-1], ERAS[1:]):
            if e1 not in by_era or e2 not in by_era:
                continue
            a, b = by_era[e1]["median"], by_era[e2]["median"]
            # typical year-to-year movement within eras = the natural yardstick
            within = np.nanmean([a.diff().abs().mean(), b.diff().abs().mean()])
            step = float(b.mean() - a.mean())
            rows.append({"feature": feat, "boundary": f"{e1} -> {e2} ({hi1}/{lo2})",
                         "era_mean_before": float(a.mean()),
                         "era_mean_after": float(b.mean()),
                         "step": step,
                         "within_era_yoy": float(within),
                         "step_over_yoy": float(step / within) if within else np.nan,
                         "kind": "index" if feat in INDEX_DRIFT else "raw_band"})
    return pd.DataFrame(rows)


# --- 4. the 1999+2000 -> 1998 El Nino confound test (RESULTS.md §8) ---
def elnino_confound_test(seed: int = 42, drop_features: str | list[str] | None = "meta",
                         control_frac: float = 0.25,
                         region_years: tuple[int, ...] = (1999, 2000),
                         model_name: str = "lightgbm",
                         model_kw: dict | None = None,
                         val_frac: float = 0.15) -> dict:
    """Train on 1999+2000, test on 1998, within regions that contain both.

    Year and region are confounded (campaigns swept region by region), so restricting to
    shared regions holds region fixed while year varies. The control arm is required — a
    held-out slice of the same 1999+2000 cohort in the same regions — or a low 1998 score
    can't be told from "1998 parcels are harder"; the reported quantity is test arm minus
    control arm, per class. Not trained on 1998 by design: it has almost no perennial parcels.

    Per-class recall, not macro-F1: cohort priors differ enormously by construction, so an
    aggregate mostly measures prior shift. The control is drawn at random not spatially
    blocked, on purpose — the 1998 test parcels sit inside the training regions, so a
    blocked control would be handicapped and inflate the difference.

    ``region_years`` sets which regions count as shared; default ``(1999, 2000)`` matches
    the training arm. §8's ``(1999,)`` phrasing yields 18 regions not 24; both agree.

    ``model_name`` goes through the lazy registry (asks the actual architecture, not always
    LightGBM). ⛔ Only one of torch/lightgbm per process on macOS.
    """
    from crop_classifier.data import class_weights, make_dataset, resolve_drop_features
    from crop_classifier.models.base import get_model, model_input_kind
    from crop_classifier.paths import runs

    rng = np.random.default_rng(seed)
    parcels = gpd.read_parquet(proc() / "modeling_parcels.parquet")
    parcels = parcels[parcels["quality_ok"] == True]                      # noqa: E712
    with open(proc() / "label_map.json") as f:
        label_map = json.load(f)
    classes = [c for c, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    r98 = set(parcels.loc[parcels["year"] == 1998, "region_id"])
    r9900 = set(parcels.loc[parcels["year"].isin(region_years), "region_id"])
    shared = sorted(r98 & r9900)
    sub = parcels[parcels["region_id"].isin(shared)
                  & parcels["year"].isin([1998, 1999, 2000])].copy().reset_index(drop=True)

    kind = model_input_kind(model_name)
    is98 = (sub["year"] == 1998).to_numpy()
    pool = np.flatnonzero(~is98)                       # the 1999+2000 cohort
    perm = rng.permutation(pool)
    n_ctrl = int(round(control_frac * len(pool)))
    ctrl = np.zeros(len(sub), dtype=bool)
    ctrl[perm[:n_ctrl]] = True
    # val set carved out of TRAIN, never the control arm (must stay untouched).
    val = np.zeros(len(sub), dtype=bool)
    val[perm[n_ctrl:n_ctrl + int(round(val_frac * len(pool)))]] = True
    trn = ~is98 & ~ctrl & ~val

    def ds(mask, normalizer=None):
        return make_dataset(kind, sub, sub.index[mask], normalizer=normalizer,
                            drop_features=drop_features)

    train_ds = ds(trn)
    norm = getattr(train_ds, "normalizer", None)
    val_ds, test_ds, ctrl_ds = ds(val, norm), ds(is98, norm), ds(ctrl, norm)

    model = get_model(model_name, **(model_kw or {}))
    model.set_n_classes(len(classes))
    tmp = runs() / "_elnino_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    model.fit(train_ds, val_ds, class_weights(train_ds.y, len(classes)), tmp)

    def recalls(d):
        yp = model.predict_proba(d).argmax(1)
        yt = d.y
        out = {}
        for i, c in enumerate(classes):
            n = int((yt == i).sum())
            out[c] = {"n": n,
                      "recall": float((yp[yt == i] == i).mean()) if n else float("nan")}
        out["_accuracy"] = float((yp == yt).mean())
        out["_n"] = int(len(yt))
        return out

    test_arm, control_arm = recalls(test_ds), recalls(ctrl_ds)
    y = train_ds.y
    delta = {c: test_arm[c]["recall"] - control_arm[c]["recall"] for c in classes}

    # which reading of the §8 outcome table does this match?
    ann = delta.get("ANNUAL", np.nan)
    pas = delta.get("PASTURE_FALLOW", np.nan)
    finite = [v for v in delta.values() if v == v]
    if pas == pas and ann == ann and (ann - pas) > 0.15 and pas < -0.15:
        reading = ("SMOKING GUN — ANNUAL recall holds while PASTURE recall collapses: "
                   "1998 flood/bare radiometry is being read as the ANNUAL signature.")
    elif finite and max(finite) - min(finite) < 0.15 and np.mean(finite) < -0.05:
        reading = ("generic temporal-transfer decay — all classes degrade roughly "
                   "equally vs control; consistent with S4, far less alarming.")
    elif finite and abs(np.mean(finite)) < 0.05:
        reading = ("little degradation vs control — the confound is not operating "
                   "through radiometry; the panel baseline is safer than feared.")
    else:
        reading = "mixed — read the per-class table directly."

    return {
        "model": model_name, "model_kw": model_kw or {}, "input_kind": kind,
        "shared_regions": len(shared), "n_parcels_in_shared_regions": int(len(sub)),
        "n_train_1999_2000": int(len(train_ds.y)),
        "n_val_1999_2000": int(len(val_ds.y)),
        "n_control_1999_2000": int(len(ctrl_ds.y)),
        "n_test_1998": int(len(test_ds.y)),
        "drop_features": resolve_drop_features(drop_features),
        "train_label_mix": {classes[i]: int(n) for i, n in
                            enumerate(np.bincount(y, minlength=len(classes)))},
        "test_arm_1998": test_arm, "control_arm_1999_2000": control_arm,
        "delta_recall_test_minus_control": delta,
        "delta_accuracy": test_arm["_accuracy"] - control_arm["_accuracy"],
        "reading": reading,
        "limit": "One year cannot separate '1998 is a different year' from '1998 is an "
                 "El Nino year' — they are collinear here (1996, the natural non-El-Nino "
                 "L5-only comparator, has only 4 labelled parcels).",
    }


# --- the gate ---
def run_gate(panel_preds: pd.DataFrame | None = None, save: bool = True,
             preds_path: Path | str | None = None, tag: str = "",
             elnino_model: str = "lightgbm") -> dict:
    """All four diagnostics. S4 and S5 are the gate; 3 and 4 are context.

    Returns a dict with ``gate_pass``. If False, nothing downstream may be produced — a
    trend from a panel that failed validation is wrong, not weaker.

    ``preds_path`` selects the file to gate; ``tag`` suffixes every artifact, so a second
    model can be gated over the same panel without overwriting the first.
    """
    out_dir = proc()
    suf = f"_{tag}" if tag else ""
    if panel_preds is None:
        panel_preds = pd.read_parquet(
            Path(preds_path) if preds_path else out_dir / "panel_predictions.parquet")
    with open(out_dir / "label_map.json") as f:
        label_map = json.load(f)
    classes = [c for c, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    print("=" * 78, "\n1. S4 — temporal transfer (locked-test parcels)\n", "=" * 78)
    tt, v4 = s4_temporal_transfer(panel_preds, classes=classes)
    print(tt.to_string(index=False))
    if "supplementary_per_k" in v4:
        print("\ncomposition-controlled views (diagnosis only, criterion unchanged):")
        print(pd.DataFrame(v4["supplementary_per_k"]).to_string(
            index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"  prior-standardised worst |k|<=3 deviation "
              f"{v4['supplementary_max_dev_prior_standardised']:.4f} -> "
              f"{'pass' if v4['supplementary_prior_standardised_pass'] else 'FAIL'}")
        print(f"  balanced-accuracy  worst |k|<=3 deviation "
              f"{v4['supplementary_max_dev_balanced_accuracy']:.4f} -> "
              f"{'pass' if v4['supplementary_balanced_accuracy_pass'] else 'FAIL'}")
    print(f"\nS4: {'PASS' if v4['pass'] else 'FAIL'} — "
          f"k=0 accuracy {v4.get('accuracy_at_k0', float('nan')):.4f}, worst within "
          f"|k|<=3 is k={v4.get('worst_k_within_3')} at "
          f"{v4.get('worst_accuracy_within_3', float('nan')):.4f} "
          f"(deviation {v4.get('max_abs_deviation_within_3', float('nan')):.4f}, "
          f"tolerance {S4_TOLERANCE})")
    if v4.get("pass") != v4.get("degradation_only_pass"):
        print(f"     note: worst *degradation* is only "
              f"{v4['max_degradation_within_3']:.4f} — the two-sided criterion is "
              f"tripped by accuracy RISING above k=0, not by transfer decay.")

    print("\n" + "=" * 78, "\n2. S5 — flicker\n", "=" * 78)
    flick, v5, series = s5_flicker(panel_preds, classes)
    print(flick.to_string(index=False))
    print(f"\nS5: {'PASS' if v5['pass'] else 'FAIL'} — PERENNIAL flicker "
          f"{v5['flicker_PERENNIAL']:.4f} (threshold {S5_MAX_FLICKER})")
    print("  supplementary (diagnosis only, does not change the verdict):")
    print(f"    smoothed series          : "
          f"{ {k: round(v, 4) for k, v in v5['supplementary_smoothed'].items()} }")
    print(f"    flagged years dropped    : "
          f"{ {k: round(v, 4) for k, v in v5['supplementary_flagged_years_dropped'].items()} }")

    print("\n" + "=" * 78, "\n3. sensor / feature drift\n", "=" * 78)
    drift = feature_drift(balanced=True)
    drift_all = feature_drift(balanced=False)
    steps, steps_all = era_steps(drift), era_steps(drift_all)
    nbal = int(drift["n"].max()) if len(drift) else 0
    print(f"balanced panel: {nbal:,} parcels present in every non-flagged year "
          f"(all-survivors view reported alongside)")
    if len(steps):
        print(steps.sort_values("step_over_yoy", key=abs, ascending=False)
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        big = steps[steps["step_over_yoy"].abs() >= 2]
        print(f"\n{len(big)} of {len(steps)} boundary steps exceed 2x the within-era "
              f"year-to-year movement "
              f"({int((big['kind'] == 'raw_band').sum())} raw-band, "
              f"{int((big['kind'] == 'index').sum())} index)")
    disp = dispersion_by_era(drift)
    print("\ncross-parcel spread (p90-p10) by era — levels can be stable while the NOISE "
          "grows:")
    print(disp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n" + "=" * 78,
          f"\n4. 1999+2000 -> 1998 El Nino confound test ({elnino_model})\n", "=" * 78)
    en = elnino_confound_test(model_name=elnino_model)
    print(json.dumps({k: v for k, v in en.items() if k != "limit"}, indent=2))

    gate = {"gate_pass": bool(v4["pass"] and v5["pass"]), "S4": v4, "S5": v5,
            "elnino": en,
            "drift": {"balanced_n_parcels": nbal,
                      "n_boundary_steps_over_2x_yoy": int(
                          (steps["step_over_yoy"].abs() >= 2).sum()) if len(steps) else 0,
                      "n_boundary_steps": int(len(steps)),
                      "dispersion_by_era": disp.to_dict("records"),
                      "worst": (steps.reindex(steps["step_over_yoy"].abs()
                                              .sort_values(ascending=False).index)
                                .head(6).to_dict("records") if len(steps) else [])},
            "flagged_years": {str(k): v for k, v in FLAG_YEARS.items()}}
    print("\n" + "=" * 78)
    print(f"GATE: {'PASS' if gate['gate_pass'] else 'FAIL'} "
          f"(S4 {'pass' if v4['pass'] else 'FAIL'}, S5 "
          f"{'pass' if v5['pass'] else 'FAIL'})")
    print("=" * 78)

    if save:
        from crop_classifier.paths import runs
        from crop_classifier.perennial import figures as FG

        tt.to_csv(out_dir / f"s4_temporal_transfer{suf}.csv", index=False)
        flick.to_csv(out_dir / f"s5_flicker_report{suf}.csv", index=False)
        pd.concat([drift, drift_all]).to_csv(
            out_dir / f"feature_drift_by_year{suf}.csv", index=False)
        pd.concat([steps.assign(balanced=True), steps_all.assign(balanced=False)]).to_csv(
            out_dir / f"feature_drift_era_steps{suf}.csv", index=False)
        with open(out_dir / f"phase7_gate{suf}.json", "w") as f:
            json.dump(gate, f, indent=2)

        fig_dir = runs() / f"diagnostics{suf}"
        fig_dir.mkdir(parents=True, exist_ok=True)
        FG.temporal_transfer_figure(tt, fig_dir / "s4_temporal_transfer.png")
        bal = f"balanced panel, {nbal:,} parcels present in every non-flagged year"
        FG.drift_panel_figure(
            drift, fig_dir / "drift_indices.png", INDEX_DRIFT,
            f"Index-feature drift — ratio indices partially cancel TM/ETM+ calibration "
            f"differences ({bal})")
        FG.drift_panel_figure(
            drift, fig_dir / "drift_raw_bands.png", RAW_DRIFT, ncols=3,
            title=f"Raw-band drift — NOT removed by the §4.6 ablation: raw bands are more "
                  f"sensor-sensitive than indices and both models consume them ({bal})")
        FG.elnino_confound_figure(en, fig_dir / "elnino_confound.png")
        print(f"wrote s4/s5/drift tables + phase7_gate.json to {out_dir}")
        print(f"wrote figures to {fig_dir}")
    gate["_series"] = series
    return gate

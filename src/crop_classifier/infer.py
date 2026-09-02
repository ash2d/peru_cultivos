"""Batch inference from a run dir (plan.md §12). Same feature path as training.

Given a polygons parquet (needs ``COD_PREDIO``; geometry+year only needed for
cold-start), joins the cached feature store, predicts, and **abstains** where
``quality_ok`` is false/unknown or ``max proba < tau``. Parcels absent from the feature
store are abstained with reason ``no_features`` (re-run extraction to cold-start them).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from crop_classifier.data import Normalizer, load_parcels, make_dataset
from crop_classifier.models.base import MODEL_REGISTRY, _ensure_registered

ROOT = Path(__file__).resolve().parents[2]


def _load_model(run_dir: Path):
    with open(run_dir / "cv_metrics.json") as f:
        model_name = json.load(f)["model"]
    _ensure_registered(model_name)   # lazy import so a lightgbm run never loads torch
    cls = MODEL_REGISTRY[model_name]
    return cls.load(run_dir / "model.bin"), model_name


def infer(run_dir: Path, polygons: Path | None = None, out: Path | None = None,
          tau: float = 0.0, parcels: pd.DataFrame | None = None,
          feat_dir: Path | None = None, calibrate: bool = True) -> pd.DataFrame:
    """Batch prediction with abstention.

    ``parcels``/``feat_dir`` let the multi-year panel score one year at a time against its
    own feature bundle; both default to the workspace table and the shared store.
    ``calibrate`` applies the run's fitted temperature (D8) when one exists — required
    before any confidence threshold or probability-weighted area estimate is meaningful.
    """
    run_dir = Path(run_dir)
    model, model_name = _load_model(run_dir)
    with open(run_dir / "label_map.json") as f:
        label_map = json.load(f)
    classes = [c for c, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    # target parcels: user file or the full modelling table (quality filter OFF —
    # inference covers everything and abstains instead of dropping)
    base = load_parcels(require_quality=False) if parcels is None else parcels.copy()
    if polygons is not None:
        req = pd.read_parquet(polygons)[["COD_PREDIO"]]
        base = base.merge(req, on="COD_PREDIO", how="inner")
    base = base.reset_index(drop=True)

    ok = (base["quality_ok"] == True)  # noqa: E712
    scored = base[ok]
    norm = Normalizer.from_state(model.normalizer_state) \
        if getattr(model, "normalizer_state", None) else None
    # Score on exactly the columns the model was fitted on, in order. This keeps an ablated
    # model (RESULTS.md §4.6) ablated at inference: withheld columns are still in the panel
    # store and would otherwise silently return.
    feature_names = getattr(model, "feature_names", None) or None
    ds = make_dataset(model.input_kind, base, scored.index, normalizer=norm,
                      feat_dir=feat_dir, feature_names=feature_names)
    prob = model.predict_proba(ds)
    if calibrate:
        from crop_classifier.perennial.calibration import (
            apply_temperature,
            load_temperature,
        )
        T = load_temperature(run_dir)
        if T != 1.0:
            prob = apply_temperature(prob, T)
            print(f"applied temperature T={T:.3f}")
    got = pd.DataFrame({"COD_PREDIO": ds.cod_predio if hasattr(ds, "cod_predio")
                        else scored["COD_PREDIO"].values})
    for i, c in enumerate(classes):
        got[f"prob_{c}"] = prob[:, i]

    res = base[["COD_PREDIO", "year", "n_valid_obs", "max_gap", "quality_ok"]].merge(
        got, on="COD_PREDIO", how="left")
    prob_cols = [f"prob_{c}" for c in classes]
    # explicit numpy bool masks (quality_ok is a nullable-boolean column -> np.select-safe)
    has_prob = res[prob_cols].notna().all(axis=1).to_numpy(dtype=bool)
    pmax = res[prob_cols].max(axis=1).to_numpy()
    pmax_nan = np.isnan(pmax)
    q = res["quality_ok"]
    q_unmeasured = q.isna().to_numpy(dtype=bool)
    q_fail = (q == False).fillna(False).to_numpy(dtype=bool)  # noqa: E712
    low_conf = has_prob & ~pmax_nan & (pmax < tau)
    argm = np.nan_to_num(res[prob_cols].to_numpy()).argmax(axis=1)

    res["pred_label"] = np.where(has_prob, np.array(classes)[argm], None)
    res["pred_proba"] = pmax
    res["abstained"] = ~has_prob | low_conf | q_unmeasured | q_fail
    res["abstain_reason"] = np.select(
        [q_unmeasured, q_fail, ~has_prob, low_conf],
        ["coverage_unmeasured", "quality_gate", "no_features", "low_confidence"],
        default="")
    res.loc[res["abstained"].to_numpy(dtype=bool), "pred_label"] = None

    n_ab = int(res["abstained"].sum())
    print(f"{len(res):,} parcels: {len(res) - n_ab:,} predicted, {n_ab:,} abstained "
          f"({res.loc[res.abstained, 'abstain_reason'].value_counts().to_dict()})")
    if out:
        res.to_parquet(out, index=False)
        print(f"wrote {out}")
    return res

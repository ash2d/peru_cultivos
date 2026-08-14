"""Admitting OLI — the temporal-OOD work (docs/RESULTS.md §6.4).

``perennial/panel.PANEL_MISSIONS = {"L5", "L7"}`` was chosen so every panel year is inferred
on radiometry the model trained on (training is 52.8 % L5 / 47.2 % L7 / 0 % OLI). Defensible
— but it is **also what creates the endpoint density collapse**: from 2013 two OLI sensors
are flying, and the same parcel-years the panel reads with ~13 clear ETM+ observations could
be read with roughly three times as many.

Step 3a is the test that decides whether that trade is available. 2013-2023 has L7 and L8
flying together, so for the **same parcel-year** the model can be scored twice — once on
L7-only features (what the panel already has) and once on harmonised-OLI-only features — and
the two compared directly. No assumption about radiometry is needed; the overlap measures it.

Pass criteria (plan §3a), fixed before running:

* median per-parcel ``|Δp_PERENNIAL|`` **< 0.05** — inside the adjacent-window disagreement
  noise floor of 3-4 pp already measured (RESULTS.md §8.2);
* class agreement **≥ 0.95**;
* **no systematic shift in the PETT-``PERENNIAL`` control pool** — the pool whose drift broke
  M2. A mean shift there is exactly the failure this test exists to catch, so it is checked
  as a signed mean, not an absolute one.

Fail ⇒ the mission policy stands, step 3 stops, and that is itself a result (T-D6).

GEE hygiene (plan §3a, CLAUDE.md §8): probe before extracting, size from the measured rate,
and **verify every year by counting output parcels** — never by "the process ended".

Run with::

    CC_PROC=data/processed/all_peru CC_FEAT=data/processed/all_peru/features \\
      uv run python -m crop_classifier.cli allperu oli probe --years 2015,2019
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from crop_classifier.features import assemble as asm
from crop_classifier.features import landsat_gee as lg
from crop_classifier.paths import proc
from crop_classifier.perennial.panel import panel_feat

OLI_MISSIONS = {"L8", "L9"}
DEFAULT_YEARS = (2015, 2019)     # both squarely inside the L7+L8 overlap
MIN_OBS = 4                      # the same quality gate the panel uses

# T3a acceptance, registered before the test is run.
MAX_MEDIAN_ABS_DP = 0.05
MIN_CLASS_AGREEMENT = 0.95
MAX_CONTROL_SHIFT = 0.02


def verdict(median_abs_dp: float, class_agreement: float,
            control_shift: float) -> dict:
    """The 3a pass/fail, as a pure function of the three registered legs.

    Separated from :func:`compare` so the *criterion* can be tested without an extraction.
    All three must hold: passing the per-parcel noise leg is not passing 3a, and the
    control-pool shift — the leg the estimand actually rests on (§8.2) — must be able to fail
    the whole test on its own.
    """
    out = {"pass_median_dp": bool(median_abs_dp < MAX_MEDIAN_ABS_DP),
           "pass_agreement": bool(class_agreement >= MIN_CLASS_AGREEMENT),
           "pass_control": bool(abs(control_shift) < MAX_CONTROL_SHIFT)}
    out["pass"] = bool(out["pass_median_dp"] and out["pass_agreement"]
                       and out["pass_control"])
    return out


def oli_dir() -> Path:
    """Raw OLI pixel store — a sibling of the panel store, never mixed into it."""
    p = panel_feat() / "oli"
    p.mkdir(parents=True, exist_ok=True)
    return p


def bundle_dir(year: int) -> Path:
    return panel_feat() / f"{year}_oli"


def probe(years: tuple[int, ...] = DEFAULT_YEARS, n_parcels: int = 150,
          seed: int = 42) -> pd.DataFrame:
    """Measure s/parcel-year for an OLI extraction before committing to the full job.

    Deliberately not reusing ``panel.timing_probe``: that one pins ``PANEL_MISSIONS`` (L5+L7)
    and the whole point here is the *other* mission set, which returns more observations per
    parcel-year and is therefore slower and bigger. Never extrapolate the L5/L7 rate to OLI.
    """
    lg.MISSION_FILTER = OLI_MISSIONS
    panel = gpd.read_parquet(proc() / "panel_parcels.parquet")
    sub = panel.sample(min(n_parcels, len(panel)), random_state=seed)
    rows = []
    for y in years:
        d = oli_dir() / "_probe" / str(y)
        d.mkdir(parents=True, exist_ok=True)
        p = sub.copy()
        p["year"] = y
        t0 = time.time()
        cov = lg.run_coverage(parcels=p, out=d / f"coverage_{y}.parquet", years=[y])
        t_cov = time.time() - t0
        surv = p[p["COD_PREDIO"].isin(cov.loc[cov["n_valid_obs"] >= MIN_OBS, "COD_PREDIO"])]
        t0 = time.time()
        lg.run_pixels(parcels=surv, years=[y], feat_dir=d, only_quality_ok=False)
        t_px = time.time() - t0
        mb = sum(f.stat().st_size for f in d.rglob("*.parquet")) / 1e6
        rows.append({"year": y, "n_parcels": len(p),
                     "gate_pass": len(surv) / len(p),
                     "mean_n_valid_obs": float(cov["n_valid_obs"].mean()),
                     "s_per_parcel_year": (t_cov + t_px) / len(p),
                     "mb_per_parcel_year": mb / len(p)})
        print(f"  {y}: {(t_cov + t_px) / len(p):.3f} s/parcel-year, gate "
              f"{len(surv) / len(p):.1%}, mean n_valid_obs "
              f"{cov['n_valid_obs'].mean():.1f}", flush=True)
    df = pd.DataFrame(rows)
    n_panel = len(gpd.read_parquet(proc() / "panel_parcels.parquet"))
    df["projected_hours_full_year"] = df["s_per_parcel_year"] * n_panel / 3600
    print(df.round(3).to_string(index=False))
    return df


def extract(years: tuple[int, ...] = DEFAULT_YEARS, chunk_size: int = 400,
            pixel_chunk_size: int = 40) -> None:
    """Coverage -> per-year gate -> pixels, OLI only, into ``<panel>/oli/``. Resumable."""
    lg.MISSION_FILTER = OLI_MISSIONS
    panel = gpd.read_parquet(proc() / "panel_parcels.parquet")
    d = oli_dir()
    for y in years:
        p = panel.copy()
        p["year"] = y
        print(f"\n=== OLI {y}: coverage ===", flush=True)
        cov = lg.run_coverage(parcels=p, out=d / f"coverage_{y}.parquet", years=[y],
                              chunk_size=chunk_size)
        cov["quality_ok"] = cov["n_valid_obs"] >= MIN_OBS
        cov.to_parquet(proc() / f"oli_coverage_{y}.parquet", index=False)
        surv = p[p["COD_PREDIO"].isin(cov.loc[cov["quality_ok"], "COD_PREDIO"])]
        print(f"OLI {y}: gate {len(surv):,}/{len(p):,} ({len(surv) / len(p):.1%})",
              flush=True)
        print(f"=== OLI {y}: pixels ===", flush=True)
        lg.run_pixels(parcels=surv, years=[y], feat_dir=d, only_quality_ok=False,
                      chunk_size=pixel_chunk_size)


def assemble(years: tuple[int, ...] = DEFAULT_YEARS, harmonize: bool = True,
             suffix: str | None = None, coefficients: dict | None = None) -> None:
    """One bundle per year in ``<panel>/<year>_oli/`` (or ``_oliraw`` unharmonised).

    ``harmonize=True`` is the point of step 3: the Roy et al. OLI->ETM+ coefficients have
    been implemented since the panel was designed and have never once been used. The
    unharmonised arm exists to attribute a failure — if raw and harmonised OLI disagree with
    L7 by the same amount, the harmonisation is not what is wrong.
    """
    suffix = suffix if suffix is not None else ("_oli" if harmonize else "_oliraw")
    panel = gpd.read_parquet(proc() / "panel_parcels.parquet")
    for y in years:
        if not (oli_dir() / f"pixels_{y}.parquet").exists():
            print(f"{y}: no OLI pixel store — skipping")
            continue
        statics = panel.copy()
        cov_path = proc() / f"oli_coverage_{y}.parquet"
        if cov_path.exists():
            cov = pd.read_parquet(cov_path)[["COD_PREDIO", "n_valid_obs", "max_gap"]]
            statics = statics.drop(columns=["n_valid_obs", "max_gap"],
                                   errors="ignore").merge(cov, on="COD_PREDIO", how="left")
        print(f"\n=== assembling OLI {y} "
              f"({'harmonised' if harmonize else 'RAW'}) ===", flush=True)
        asm.assemble(years=[y], feat_dir=oli_dir(),
                     out_dir=panel_feat() / f"{y}{suffix}",
                     parcels=statics, harmonize_oli=harmonize,
                     harmonize_coefficients=coefficients)


def verify(years: tuple[int, ...] = DEFAULT_YEARS) -> pd.DataFrame:
    """Count the outputs. A finished process is not evidence that a job completed."""
    rows = []
    for y in years:
        px = oli_dir() / f"pixels_{y}.parquet"
        bundle = bundle_dir(y) / asm.FN_LGBM
        n_px = n_par = n_feat = 0
        if px.exists():
            d = pd.read_parquet(px, columns=["COD_PREDIO"])
            n_px, n_par = len(d), d["COD_PREDIO"].nunique()
        if bundle.exists():
            n_feat = len(pd.read_parquet(bundle, columns=["COD_PREDIO"]))
        rows.append({"year": y, "pixel_obs": n_px, "parcels_with_pixels": n_par,
                     "parcels_in_bundle": n_feat})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def compare(run_dir: Path, years: tuple[int, ...] = DEFAULT_YEARS,
            l7_preds: Path | None = None, tag: str = "", save: bool = True,
            suffix: str = "_oli") -> dict:
    """3a: same parcel-year, L7-only features vs harmonised-OLI-only features.

    Returns the verdict dict. Everything is paired within parcel-year, so nothing here
    depends on the two arms covering the same parcels — only on the overlap between them.
    """
    from crop_classifier.perennial.panel import infer_panel

    run_dir = Path(run_dir)
    l7 = pd.read_parquet(l7_preds or (proc() / "panel_predictions_nolat.parquet"))
    l7 = l7[l7["year"].isin(years)]
    oli = infer_panel(run_dir, years=list(years), save=False, bundle_suffix=suffix)
    # `infer_panel` attaches the L7 panel coverage, so its `n_valid_obs` is the L7 count.
    # Replace it with the OLI one for reporting — otherwise the table claims the two arms
    # saw the same number of observations, which is the opposite of the point.
    ocov = []
    for y in years:
        p = proc() / f"oli_coverage_{y}.parquet"
        if p.exists():
            c = pd.read_parquet(p)[["COD_PREDIO", "n_valid_obs"]].copy()
            c["year"] = y
            ocov.append(c)
    if ocov:
        oli = oli.drop(columns=["n_valid_obs"]).merge(
            pd.concat(ocov, ignore_index=True), on=["COD_PREDIO", "year"], how="left")

    keys = ["COD_PREDIO", "year"]
    m = l7.merge(oli, on=keys, suffixes=("_l7", "_oli"))
    m = m[~m["abstained_l7"].astype(bool) & ~m["abstained_oli"].astype(bool)]
    m["dp"] = m["prob_PERENNIAL_oli"] - m["prob_PERENNIAL_l7"]
    agree = float((m["pred_label_l7"] == m["pred_label_oli"]).mean())
    med_abs = float(m["dp"].abs().median())

    per_year = (m.groupby("year")
                .agg(n=("dp", "size"), median_abs_dp=("dp", lambda s: s.abs().median()),
                     mean_dp=("dp", "mean"),
                     agreement=("pred_label_l7",
                                lambda s: float((s == m.loc[s.index,
                                                            "pred_label_oli"]).mean())))
                .reset_index())

    ctrl = m[m["pett_label_l7"] == "PERENNIAL"]
    risk = m[m["pett_label_l7"] == "ANNUAL"]
    ctrl_shift = float(ctrl["dp"].mean()) if len(ctrl) else float("nan")

    out = {
        "tag": tag, "run": str(run_dir), "years": list(years),
        "n_paired_parcel_years": int(len(m)),
        "median_abs_dp": med_abs, "mean_dp": float(m["dp"].mean()),
        "p95_abs_dp": float(m["dp"].abs().quantile(0.95)),
        "class_agreement": agree,
        "control_mean_dp": ctrl_shift,
        "at_risk_mean_dp": float(risk["dp"].mean()) if len(risk) else float("nan"),
        "mean_n_valid_obs_l7": float(m["n_valid_obs_l7"].mean()),
        "mean_n_valid_obs_oli": float(m["n_valid_obs_oli"].mean()),
        "per_year": per_year.to_dict("records"),
        "criteria": {"max_median_abs_dp": MAX_MEDIAN_ABS_DP,
                     "min_class_agreement": MIN_CLASS_AGREEMENT,
                     "max_control_shift": MAX_CONTROL_SHIFT},
    }
    out.update(verdict(med_abs, agree, ctrl_shift))
    if save:
        suf = f"_{tag}" if tag else ""
        with open(proc() / f"oli_overlap{suf}.json", "w") as f:
            json.dump(out, f, indent=2, default=float)
        m[keys + ["prob_PERENNIAL_l7", "prob_PERENNIAL_oli", "dp", "pred_label_l7",
                  "pred_label_oli", "pett_label_l7", "n_valid_obs_l7",
                  "n_valid_obs_oli"]].to_parquet(
            proc() / f"oli_overlap_pairs{suf}.parquet", index=False)
        print(f"wrote oli_overlap{suf}.json + oli_overlap_pairs{suf}.parquet to {proc()}")
    return out


def split_half_control(run_dir: Path, years: tuple[int, ...] = DEFAULT_YEARS,
                       seed: int = 11, tag: str = "", save: bool = True) -> dict:
    """The control 3a needs: how well does the model agree with **itself**?

    Split each parcel-year's L7 acquisitions into two disjoint halves, build features from
    each, and score both. Same sensor, same year, same parcel, same model — the only
    difference is *which* clear observations were used. Whatever agreement that produces is
    the ceiling any cross-sensor comparison can reach, and without it a criterion of 0.95 is
    a number with no referent.

    ⚠️ Each half has half the observations, so this is a slightly pessimistic ceiling: the
    OLI arm is compared at full density on both sides. It bounds the criterion from below,
    which is the direction that matters — if split-half agreement is already far under 0.95,
    the registered threshold was never achievable and the 3a failure must be read on the
    *control shift*, not on agreement.
    """
    from crop_classifier.data import FlatData
    from crop_classifier.models.trees import LightGBMModel
    from crop_classifier.perennial.calibration import apply_temperature, load_temperature

    run_dir = Path(run_dir)
    model = LightGBMModel.load(run_dir / "model.bin")
    names = model.feature_names
    T = load_temperature(run_dir)
    with open(run_dir / "label_map.json") as f:
        classes = [c for c, _ in sorted(json.load(f).items(), key=lambda kv: kv[1])]
    p_idx = classes.index("PERENNIAL")
    panel = gpd.read_parquet(proc() / "panel_parcels.parquet")
    rng = np.random.default_rng(seed)

    rows = []
    for y in years:
        px_path = panel_feat() / f"pixels_{y}.parquet"
        if not px_path.exists():
            continue
        px = asm.scale_sr(pd.read_parquet(px_path))
        px["px_id"] = (px["lon"].round(4).astype(str) + "_"
                       + px["lat"].round(4).astype(str))
        pairs = px[["COD_PREDIO", "doy"]].drop_duplicates().copy()
        pairs["_half"] = rng.integers(0, 2, len(pairs))
        halves = {}
        for h in (0, 1):
            sel = pairs.loc[pairs["_half"] == h, ["COD_PREDIO", "doy"]]
            sub = px.merge(sel, on=["COD_PREDIO", "doy"], how="inner")
            med = asm.per_date_medians(sub)
            nd = med.groupby("COD_PREDIO")["doy"].nunique()
            med = med[med["COD_PREDIO"].isin(nd[nd >= MIN_OBS].index)]
            feats = asm.build_lightgbm_features(med, panel)
            ds = FlatData(X=feats[names], y=np.zeros(len(feats), dtype=int),
                          cod_predio=feats["COD_PREDIO"].to_numpy(),
                          feature_names=names)
            prob = apply_temperature(model.predict_proba(ds), T)
            halves[h] = pd.DataFrame({
                "COD_PREDIO": ds.cod_predio,
                f"p{h}": prob[:, p_idx],
                f"lab{h}": np.array(classes)[prob.argmax(1)]})
        m = halves[0].merge(halves[1], on="COD_PREDIO")
        m["year"] = y
        rows.append(m)
        print(f"  {y}: {len(m):,} parcels, agreement "
              f"{(m.lab0 == m.lab1).mean():.4f}, median |dp| "
              f"{(m.p1 - m.p0).abs().median():.4f}", flush=True)

    allm = pd.concat(rows, ignore_index=True)
    out = {"tag": tag, "years": list(years), "n": int(len(allm)),
           "class_agreement": float((allm.lab0 == allm.lab1).mean()),
           "median_abs_dp": float((allm.p1 - allm.p0).abs().median()),
           "p95_abs_dp": float((allm.p1 - allm.p0).abs().quantile(0.95)),
           "mean_dp": float((allm.p1 - allm.p0).mean()),
           "per_year": [{"year": int(y), "n": int(len(s)),
                         "agreement": float((s.lab0 == s.lab1).mean()),
                         "median_abs_dp": float((s.p1 - s.p0).abs().median())}
                        for y, s in allm.groupby("year")]}
    print(f"\nsplit-half self-agreement {out['class_agreement']:.4f}, "
          f"median |dp| {out['median_abs_dp']:.4f}  "
          f"(the ceiling for any cross-sensor comparison)")
    if save:
        suf = f"_{tag}" if tag else ""
        with open(proc() / f"oli_split_half{suf}.json", "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"wrote oli_split_half{suf}.json to {proc()}")
    return out


def print_compare(v: dict) -> None:
    print(f"\n=== 3a OLI overlap: {v['tag'] or v['run']} ===")
    print(f"{v['n_paired_parcel_years']:,} paired parcel-years over {v['years']}")
    print(f"observations per parcel-year: L7-only {v['mean_n_valid_obs_l7']:.1f}  "
          f"OLI-only {v['mean_n_valid_obs_oli']:.1f}")
    print(pd.DataFrame(v["per_year"]).round(4).to_string(index=False))
    print(f"median |dp| {v['median_abs_dp']:.4f} (< {v['criteria']['max_median_abs_dp']})  "
          f"p95 {v['p95_abs_dp']:.4f}")
    print(f"class agreement {v['class_agreement']:.4f} "
          f"(>= {v['criteria']['min_class_agreement']})")
    print(f"control-pool mean dp {v['control_mean_dp']:+.4f} "
          f"(|.| < {v['criteria']['max_control_shift']})   "
          f"at-risk {v['at_risk_mean_dp']:+.4f}")
    for k in ("pass_median_dp", "pass_agreement", "pass_control"):
        print(f"  {'PASS' if v[k] else 'FAIL'}  {k}")
    print(f"  => {'PASS' if v['pass'] else 'FAIL'}")

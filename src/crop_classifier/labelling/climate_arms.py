"""Climate covariates as extra model inputs on the S2 endpoint labels.

Four feature sets are fitted per model class, so that "does climate help?" is asked
against the *same* parcels, the *same* frozen folds and the *same* seeds:

| arm | columns added |
|---|---|
| ``none`` | — (the arm already in ``RESULTS.md`` §8.2) |
| ``temp`` | ``tmean_c`` |
| ``rain`` | ``precip_mm_yr`` |
| ``both`` | both |

``tmean_c`` / ``precip_mm_yr`` are the WorldClim 2.1 1970–2000 normals sampled at each
parcel centroid (``allperu.climate``): annual mean temperature in °C and annual rainfall
total in mm/yr.

⚠️ **Two warnings that decide how the result must be read**, both already on this
project's record:

1. A 1 km climate surface is a **smooth function of location**, so these columns are a
   ``centroid_lat`` proxy, and ``centroid_lat`` is the project's canonical example of a
   feature that buys cross-validation skill and *loses* out-of-department skill
   (``RESULTS.md`` §4.2). Every arm here is therefore reported CV **and** LODO, and
   ``location_proxy_audit`` measures directly how much of the department identity the two
   columns carry.
2. The normals are **time-invariant**. That is fine here and only here: this is a
   single-epoch endpoint classifier, so there is no trend for a constant to manufacture
   (``RESULTS.md`` §4.4/§5 is about the panel). Do not carry these columns into a panel.

Each variant gets its own ``CC_PROC`` workspace, built by **symlinking the base
workspace's ``modeling_parcels.parquet`` and ``label_map.json``** rather than rebuilding
them — the split, the folds and the 3 km dead-zones are then identical across arms by
construction, so a difference between arms cannot be a difference in fold membership.

The flat (LightGBM / rules) path reads the extra columns straight out of the feature
parquet. The sequence (LTAE) path has no static input at all — its tensor is spectral +
doy + mask — so the variant also writes ``statics.npz``, which ``data.SeqDataset`` picks
up and ``models.ltae`` concatenates into the classifier head.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from crop_classifier.labelling import train_prep as P
from crop_classifier.paths import ROOT

CLIMATE_PARQUET = ROOT / "data" / "processed" / "climate" / "parcel_climate_normals.parquet"
CLIMATE_SETS: dict[str, list[str]] = {
    "none": [],
    "temp": ["tmean_c"],
    "rain": ["precip_mm_yr"],
    "both": ["tmean_c", "precip_mm_yr"],
}
ARMS = list(CLIMATE_SETS)


def ws_dir(target: str, climate: str, include_pilot: bool = True) -> Path:
    """Workspace for one (target, climate) arm. ``none`` is the existing base workspace."""
    base = P.ws_dir(target, include_pilot)
    return base if climate == "none" else base.parent / f"{base.name}__clim_{climate}"


def _climate_table(cods: pd.Series) -> pd.DataFrame:
    """The climate columns for these parcels, with the coverage actually achieved."""
    if not CLIMATE_PARQUET.exists():
        raise SystemExit(
            f"missing {CLIMATE_PARQUET} — build it once with\n"
            f"  uv run python -m crop_classifier.cli allperu climate normals")
    cl = pd.read_parquet(CLIMATE_PARQUET,
                         columns=["COD_PREDIO", *CLIMATE_SETS["both"]])
    cl["COD_PREDIO"] = cl["COD_PREDIO"].astype(str)
    want = pd.DataFrame({"COD_PREDIO": cods.astype(str).unique()})
    out = want.merge(cl, on="COD_PREDIO", how="left")
    miss = int(out[CLIMATE_SETS["both"]].isna().any(axis=1).sum())
    if miss:
        # a climate NaN would silently drop the parcel from LightGBM's split logic in a
        # way that differs between arms, which is exactly the comparison being made
        raise SystemExit(f"{miss} of {len(out)} labelled parcels have no climate value; "
                         f"rebuild the normals over a parcel table that covers them")
    return out


def build_variant(target: str = "t4", climate: str = "both",
                  include_pilot: bool = True, verbose: bool = True) -> Path:
    """Base workspace + climate columns -> a ``CC_PROC`` workspace for one arm."""
    if climate not in CLIMATE_SETS:
        raise SystemExit(f"unknown climate arm {climate!r}; expected {ARMS}")
    base = P.ws_dir(target, include_pilot)
    if not (base / "modeling_parcels.parquet").exists():
        raise SystemExit(f"{base} not built — run `allperu s2-train prep` first")
    if climate == "none":
        return base

    out = ws_dir(target, climate, include_pilot)
    (out / "features").mkdir(parents=True, exist_ok=True)
    for fn in ("modeling_parcels.parquet", "label_map.json"):
        link = out / fn
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(base / fn)

    cols = CLIMATE_SETS[climate]
    parcels = gpd.read_parquet(base / "modeling_parcels.parquet")
    clim = _climate_table(parcels["COD_PREDIO"])

    feats = pd.read_parquet(P.FEAT_S2 / "s2_features_lightgbm.parquet")
    feats["COD_PREDIO"] = feats["COD_PREDIO"].astype(str)
    feats = feats.merge(clim[["COD_PREDIO", *cols]], on="COD_PREDIO", how="left")
    feats.to_parquet(out / "features" / "features_lightgbm.parquet", index=False)

    # the sequence tensor is unchanged; the statics ride alongside it
    link = out / "features" / "tensor_perdate.npz"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(P.FEAT_S2 / "tensor_perdate.npz")

    st = clim.set_index("COD_PREDIO").loc[:, cols]
    np.savez(out / "features" / "statics.npz",
             cod_predio=np.array(st.index, dtype=object),
             X=st.to_numpy(dtype="float32"),
             names=np.array(cols, dtype=object))

    if verbose:
        j = parcels[["COD_PREDIO", "label", "dept"]].merge(clim, on="COD_PREDIO")
        print(f"[{target}{'+pilot' if include_pilot else ''} | clim={climate}] "
              f"{len(parcels)} parcels, +{len(cols)} column(s) {cols} -> {out}")
        print(j.groupby("label")[CLIMATE_SETS["both"]].agg(["mean", "std"]).round(1)
              .to_string())
    return out


def build_all(target: str = "t4", include_pilot: bool = True) -> None:
    for c in ARMS:
        build_variant(target, c, include_pilot)
        print()


# ---------------------------------------------------------------------------------
# Is climate just latitude again?
# ---------------------------------------------------------------------------------
def location_proxy_audit(target: str = "t4", include_pilot: bool = True) -> dict:
    """How much of *where the parcel is* do the two climate columns carry?

    ``centroid_lat`` gains +0.047 on CV and loses 0.060 on LODO (``RESULTS.md`` §4.2), and
    the mechanism is that it lets the model name the department. Climate is a smooth
    surface over the same space, so the honest question is not "is it correlated with
    latitude" but "can a model recover the department from these two numbers alone". A
    5-fold accuracy well above the department prior means the same failure mode is
    available, and the LODO column of the results table is where it would show up.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    base = P.ws_dir(target, include_pilot)
    parcels = gpd.read_parquet(base / "modeling_parcels.parquet")
    parcels = parcels[parcels["split"] != "test"].copy()
    clim = _climate_table(parcels["COD_PREDIO"])
    df = parcels[["COD_PREDIO", "dept", "label"]].merge(clim, on="COD_PREDIO")
    pt = parcels.geometry.representative_point().to_crs(4326)
    df["centroid_lat"] = pt.y.values
    df["centroid_lon"] = pt.x.values

    cols = CLIMATE_SETS["both"]
    corr = df[[*cols, "centroid_lat", "centroid_lon"]].corr(method="spearman")
    prior = df["dept"].value_counts(normalize=True).max()
    acc = float(cross_val_score(RandomForestClassifier(300, random_state=0),
                                df[cols], df["dept"], cv=5, scoring="accuracy").mean())
    acc_lat = float(cross_val_score(RandomForestClassifier(300, random_state=0),
                                    df[["centroid_lat"]], df["dept"], cv=5,
                                    scoring="accuracy").mean())
    out = {"n": len(df), "n_dept": int(df["dept"].nunique()),
           "dept_prior": float(prior),
           "dept_acc_from_climate": acc, "dept_acc_from_centroid_lat": acc_lat,
           "spearman_tmean_lat": float(corr.loc["tmean_c", "centroid_lat"]),
           "spearman_precip_lat": float(corr.loc["precip_mm_yr", "centroid_lat"]),
           "spearman_tmean_precip": float(corr.loc["tmean_c", "precip_mm_yr"])}
    print("\nIs climate a location proxy?")
    print(f"  department recovered from (tmean_c, precip_mm_yr) alone: "
          f"{acc:.3f} 5-fold accuracy over {out['n_dept']} departments "
          f"(prior {prior:.3f}); from centroid_lat alone: {acc_lat:.3f}")
    print(f"  Spearman  tmean~lat {out['spearman_tmean_lat']:+.3f}   "
          f"precip~lat {out['spearman_precip_lat']:+.3f}   "
          f"tmean~precip {out['spearman_tmean_precip']:+.3f}")
    print("\n  class means:")
    print(df.groupby("label")[cols].agg(["mean", "std"]).round(1).to_string())
    print("\n  department means:")
    print(df.groupby("dept")[cols].mean().round(1).sort_values("precip_mm_yr").to_string())
    (base / "climate_location_audit.json").write_text(json.dumps(out, indent=2))
    return out

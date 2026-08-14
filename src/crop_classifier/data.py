"""Datasets serving the three model input kinds (plan.md §10): flat | sequence | pixelset.

Selection logic is centralised here so every model sees identical splits:

* CV fold ``k``:   val = ``fold == k``;  train = other trainval folds minus
  ``buffer_excl_fold{k}`` (the §5 dead-zone) — both restricted to ``quality_ok``.
* Final/test:      train = all trainval minus ``buffer_excl_test``; test = ``split=='test'``.

Channel normalisation (sequence/pixelset) is fit on the TRAIN subset only and applied to
val/test — never fit on held-out data (leakage checklist §5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# NB: torch is imported lazily inside SeqDataset only. A LightGBM run must not import
# torch (macOS libomp clash with lightgbm -> segfault); the flat path never needs it.
from crop_classifier.paths import feat, proc

# Feature-store file names inside a feature directory. The directory is FEAT by default
# (the shared training store) but is overridable per call so the multi-year panel can read
# its own per-year store (perennial/panel.py) without touching the audited one.
FN_LGBM = "features_lightgbm.parquet"
FN_PERDATE = "tensor_perdate.npz"
FN_PIXELSET = "tensor_pixelset.npz"


def load_parcels(require_quality: bool = True,
                 path: Path | None = None) -> gpd.GeoDataFrame:
    """Load the workspace's parcel table (``CC_PROC`` selects the workspace)."""
    df = gpd.read_parquet(path or proc() / "modeling_parcels.parquet")
    if require_quality:
        df = df[df["quality_ok"] == True]  # noqa: E712  (boolean dtype, NA-safe)
    return df


def fold_split(df: pd.DataFrame, fold: int) -> tuple[pd.Index, pd.Index]:
    """(train_idx, val_idx) for CV fold ``fold``, buffer dead-zone applied to train."""
    val = df["fold"] == fold
    train = (df["split"] == "trainval") & ~val & ~df[f"buffer_excl_fold{fold}"]
    return df.index[train], df.index[val]


def final_split(df: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    """(train_idx, test_idx) for the final refit + single locked-test evaluation."""
    train = (df["split"] == "trainval") & ~df["buffer_excl_test"]
    test = df["split"] == "test"
    return df.index[train], df.index[test]


def parse_year_spec(spec: str | list[int] | None) -> list[int] | None:
    """``"1999-2023"`` / ``"1999,2000"`` / ``"1999-2001,2005"`` -> sorted year list.

    ``None``/empty means "no restriction". Ranges are inclusive on both ends.
    """
    if spec is None:
        return None
    if isinstance(spec, (list, tuple, set)):
        return sorted(int(y) for y in spec) or None
    spec = spec.strip()
    if not spec:
        return None
    years: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            years.update(range(lo, hi + 1))
        else:
            years.add(int(part))
    return sorted(years) or None


def restrict_years(df: pd.DataFrame, idx: pd.Index,
                   years: list[int] | None) -> pd.Index:
    """Subset ``idx`` to parcels whose label ``year`` is in ``years``.

    Applied to the **train** side only (train.py). Validation/test membership must not
    change, or CV stops being comparable with runs that used the full label cohort —
    which is the whole point of the ≥1999 (no-El-Nino) experiment (RESULTS.md §8.1).
    Parcels with a missing ``year`` are dropped by the restriction, since they cannot be
    shown to satisfy it.
    """
    if not years:
        return idx
    keep = df.loc[idx, "year"].isin(years).fillna(False)
    return idx[keep.values]


def class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """'balanced' weights, safe for classes absent from a fold."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = np.where(counts > 0, len(y) / (n_classes * np.maximum(counts, 1)), 0.0)
    return w


# ------------------------------------------------------------------------------------
# flat (LightGBM)
# ------------------------------------------------------------------------------------
@dataclass
class FlatData:
    X: pd.DataFrame          # feature matrix (NaN allowed)
    y: np.ndarray
    cod_predio: np.ndarray
    feature_names: list[str] = field(default_factory=list)


# Acquisition metadata: describes *which satellite was overhead*, not the land. These vary
# across a multi-year panel for reasons that have nothing to do with land use — `frac_l7`
# ramps 0 -> 1 as L7 replaces L5 — and in training they are confounded with the label
# because titling year and mission availability move together (RESULTS.md §4.6). A model
# that uses them manufactures a trend. `centroid_lat` is listed separately: it is
# time-invariant so it cannot manufacture a trend, but it is heavy spatial memorisation.
META_FEATURES = ["frac_l7", "n_valid_obs", "n_dates", "max_gap", "n_valid_pixels"]
LOCATION_FEATURES = ["centroid_lat"]

# Order statistics over a parcel-year's observations. A max over 24 clear looks is not the
# same estimator as a max over 13, so these move when only the *number of looks* changes —
# measured, within parcel: mean |slope| 0.35 within-SD per e-fold of observation count for
# min/max/amplitude against 0.08 for the harmonic/slope fits (`allperu.density`, the step-1a
# audit in docs/RESULTS.md §6.1). Landsat density is not stationary — ~24 clear
# observations per parcel-year in 2004-08 against ~13 in 2019-23 — so a model leaning on
# them reads archive depth as land-use change.
ORDER_FEATURES = [f"{ch}_{k}" for ch in
                  ["B", "G", "R", "NIR", "SWIR1", "SWIR2",
                   "NDVI", "EVI", "NDWI", "NDMI", "BSI"]
                  for k in ("min", "max", "amp")]

DROP_SETS = {"meta": META_FEATURES, "location": LOCATION_FEATURES,
             "order": ORDER_FEATURES}


def resolve_drop_features(spec: str | list[str] | None) -> list[str]:
    """Feature-exclusion spec -> explicit column list.

    Accepts a list, or a comma-separated string whose entries are either column names or
    the group aliases in ``DROP_SETS`` (``meta``, ``location``).
    """
    if not spec:
        return []
    items = [s.strip() for s in spec.split(",")] if isinstance(spec, str) else list(spec)
    out: list[str] = []
    for it in items:
        if not it:
            continue
        out.extend(DROP_SETS.get(it, [it]))
    return list(dict.fromkeys(out))


def make_flat(parcels: pd.DataFrame, idx: pd.Index,
              feat_dir: Path | None = None,
              drop_features: str | list[str] | None = None,
              feature_names: list[str] | None = None,
              augment_feat: Path | str | None = None) -> FlatData:
    """Flat feature matrix for LightGBM.

    ``drop_features`` excludes columns at *training* time (names or ``DROP_SETS`` aliases).
    ``feature_names`` instead pins the exact ordered column list — used at inference so a
    model scores on precisely the columns it was fitted on, whatever the store contains.

    ``augment_feat`` points at a second feature table keyed on the same ``COD_PREDIO`` —
    in practice a **degraded** copy of the store, assembled at endpoint observation density
    (``allperu.density.build_degraded_features``) — whose rows for the same ``idx`` are
    appended, doubling the parcel's representation at two densities under one label. It must
    only ever be passed for a **train** dataset: adding degraded copies to a validation set
    would score the model on rows it also trained on, and would change what CV means.
    """
    feats = pd.read_parquet((feat_dir or feat()) / FN_LGBM)
    if augment_feat is not None:
        aug = pd.read_parquet(augment_feat)
        aug = aug[[c for c in feats.columns if c in aug.columns]]
        feats = pd.concat([feats, aug], ignore_index=True)
    sub = parcels.loc[idx, ["COD_PREDIO", "label_id"]].merge(
        feats, on="COD_PREDIO", how="inner")
    if feature_names is not None:
        missing = [c for c in feature_names if c not in sub.columns]
        if missing:
            raise KeyError(f"feature store is missing {len(missing)} model features: "
                           f"{missing[:10]}")
        X = sub[list(feature_names)]
    else:
        drop = ["COD_PREDIO", "label_id"] + resolve_drop_features(drop_features)
        X = sub.drop(columns=[c for c in drop if c in sub.columns])
    return FlatData(X=X, y=sub["label_id"].values, cod_predio=sub["COD_PREDIO"].values,
                    feature_names=list(X.columns))


# ------------------------------------------------------------------------------------
# sequence (LTAE) / pixelset (PSE-LTAE)
# ------------------------------------------------------------------------------------
class Normalizer:
    """Per-channel mean/std over *valid* entries; fit on train only."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray, valid: np.ndarray) -> Normalizer:
        flat = X[valid]                       # [n_valid, C]
        self.mean = flat.mean(axis=0).astype(np.float32)
        self.std = (flat.std(axis=0) + 1e-6).astype(np.float32)
        return self

    def apply(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state(cls, s: dict) -> Normalizer:
        n = cls()
        n.mean = np.array(s["mean"], dtype=np.float32)
        n.std = np.array(s["std"], dtype=np.float32)
        return n


class SeqDataset:
    """Per-date median sequences (LTAE) or pixel sets (PSE-LTAE).

    Deliberately not subclassing ``torch.utils.data.Dataset`` so this module imports
    without torch; a plain ``__len__``/``__getitem__`` is all ``DataLoader`` needs. torch
    is imported here (only reached on a torch-model run).
    """

    def __init__(self, kind: str, parcels: pd.DataFrame, idx: pd.Index,
                 normalizer: Normalizer | None = None,
                 feat_dir: Path | None = None):
        import torch
        self._torch = torch
        assert kind in ("sequence", "pixelset")
        self.kind = kind
        fd = feat_dir or feat()
        z = np.load(fd / (FN_PERDATE if kind == "sequence" else FN_PIXELSET),
                    allow_pickle=True)
        pos = {c: i for i, c in enumerate(z["cod_predio"])}
        sub = parcels.loc[idx]
        keep = sub["COD_PREDIO"].isin(pos).values
        sub = sub[keep]
        rows = np.array([pos[c] for c in sub["COD_PREDIO"]])
        self.cod_predio = sub["COD_PREDIO"].values
        self.y = sub["label_id"].values.astype(np.int64)
        self.X = z["X"][rows]                # [N,T,C] or [N,T,P,C]
        self.doy = z["doy"][rows].astype(np.float32)
        self.mask = z["mask"][rows]
        self.pixmask = z["pixmask"][rows] if kind == "pixelset" else None

        if normalizer is None:
            normalizer = Normalizer()
            valid = self.mask if kind == "sequence" else self.pixmask
            normalizer.fit(self.X, valid)
        self.normalizer = normalizer
        self.X = self.normalizer.apply(self.X)
        # re-zero padded entries so masked positions stay neutral after normalisation
        if kind == "sequence":
            self.X[~self.mask] = 0.0
        else:
            self.X[~self.pixmask] = 0.0

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        torch = self._torch
        item = {"x": torch.from_numpy(self.X[i]),
                "doy": torch.from_numpy(self.doy[i]),
                "mask": torch.from_numpy(self.mask[i]),
                "y": torch.tensor(self.y[i])}
        if self.kind == "pixelset":
            item["pixmask"] = torch.from_numpy(self.pixmask[i])
        return item


def make_dataset(input_kind: str, parcels: pd.DataFrame, idx: pd.Index,
                 normalizer: Normalizer | None = None,
                 feat_dir: Path | None = None,
                 drop_features: str | list[str] | None = None,
                 feature_names: list[str] | None = None,
                 augment_feat: Path | str | None = None):
    """Factory used by the trainer: one call site for all three input kinds.

    ``drop_features``/``feature_names``/``augment_feat`` apply to the flat path only — the
    sequence and pixel-set tensors carry no statics at all (only spectral + doy + mask), so
    the attention models are structurally free of the metadata confound, and degradation
    augmentation for them would need degraded *tensors*, which is a separate build.
    """
    if input_kind == "flat":
        return make_flat(parcels, idx, feat_dir=feat_dir, drop_features=drop_features,
                         feature_names=feature_names, augment_feat=augment_feat)
    return SeqDataset(input_kind, parcels, idx, normalizer, feat_dir=feat_dir)

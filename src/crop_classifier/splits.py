"""Spatial blocks, locked test set, CV folds, buffered dead-zone (plan.md §5, A2).

Test/fold units are **contiguous super-regions** (``region_km`` grid, default 5 km), not
scattered 1 km blocks: the buffer dead-zone costs training data in proportion to the
held-out *perimeter*, and 329 scattered test blocks + a 1.5 km buffer sterilised 69 % of
all parcels (33,245/48,289). A few large regions hold out the same test share for a
fraction of that cost. 1 km ``block_id`` is kept for reporting/fine structure.

Adds to ``modeling_parcels.parquet``:

* ``block_id``      — 1 km grid cell (EPSG:32717) of the parcel centroid
* ``region_id``     — ``region_km`` super-region cell; the unit of test/fold assignment
* ``split``         — ``test`` (locked, whole regions, ~15 %) or ``trainval``
* ``fold``          — 0..k-1 for trainval rows (StratifiedGroupKFold, groups=region), -1 for test
* ``buffer_excl_test``     — trainval parcel within ``buffer_m`` of a test parcel ->
                             excluded from *all* training (dead-zone)
* ``buffer_excl_fold{k}``  — parcel excluded from fold-k *training* (within ``buffer_m``
                             of a fold-k val parcel)

Also writes ``splits_meta.json`` (autocorrelation audit + settings) and
``class_block_counts.csv`` (per-class distinct-block/region report — the
honest-evaluability check).

Run with::

    uv run python -m crop_classifier.splits           # or: cli splits assign
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedGroupKFold

from crop_classifier.paths import proc

CONFIG_DIR = Path(__file__).resolve().parent / "config"

# Metric CRS for block gridding, the autocorrelation audit and the buffer dead-zone.
# Piura sits inside UTM 17S, so 32717 is exact there and stays the default. The all-Peru
# workspace spans zones 17S-19S and overrides it via `metric_crs:` in its split config —
# one zone for the whole country, because block ids must come from a single continuous
# grid. UTM 18S is the choice there: Peru reaches ~6.4 deg either side of its central
# meridian, a scale error under ~0.7 %, i.e. <11 m on the 1.5 km buffer.
DEFAULT_METRIC_CRS = 32717  # WGS84 / UTM 17S
METRIC_CRS = DEFAULT_METRIC_CRS  # back-compat alias; assign() reads the config


def out_paths() -> tuple[Path, Path, Path]:
    """``(modeling_parcels, splits_meta, class_block_counts)`` in the current workspace.

    Resolved at call time so ``CC_PROC`` selects the workspace (see paths.py).
    """
    p = proc()
    return (p / "modeling_parcels.parquet", p / "splits_meta.json",
            p / "class_block_counts.csv")


def load_config(path: Path | None = None) -> dict[str, Any]:
    with open(path or CONFIG_DIR / "split.yaml") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------------------------
# Autocorrelation audit (sets/justifies block size + buffer)
# ------------------------------------------------------------------------------------
def autocorrelation_audit(xy: np.ndarray, label_id: np.ndarray, cfg: dict) -> dict:
    """Neighbour label-agreement vs distance -> decorrelation range estimate.

    Samples parcels, finds neighbours within ``audit_max_dist_m``, bins pair agreement by
    distance. Baseline = expected agreement of two random parcels (sum p_i^2). The
    decorrelation range r is the first bin whose agreement is within 10 % of baseline.
    """
    rng = np.random.default_rng(cfg["seed"])
    n = len(xy)
    take = min(cfg["audit_sample"], n)
    idx = rng.choice(n, take, replace=False)
    tree = cKDTree(xy)
    max_d = float(cfg["audit_max_dist_m"])
    bins = np.array([0, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, max_d])

    pairs_d, pairs_same = [], []
    for i in idx:
        js = tree.query_ball_point(xy[i], max_d)
        js = [j for j in js if j > i]
        if len(js) > 40:  # cap per-parcel pair count to keep it fast & unbiased-ish
            js = list(rng.choice(js, 40, replace=False))
        for j in js:
            pairs_d.append(np.hypot(*(xy[i] - xy[j])))
            pairs_same.append(label_id[i] == label_id[j])
    pairs_d, pairs_same = np.array(pairs_d), np.array(pairs_same, dtype=float)

    p = np.bincount(label_id) / n
    baseline = float((p**2).sum())
    curve = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (pairs_d >= lo) & (pairs_d < hi)
        curve.append({"lo_m": float(lo), "hi_m": float(hi), "n_pairs": int(m.sum()),
                      "agreement": float(pairs_same[m].mean()) if m.any() else None})
    r = None
    for c in curve:
        if c["agreement"] is not None and c["agreement"] <= baseline * 1.10:
            r = c["hi_m"]
            break
    return {"baseline_agreement": baseline, "curve": curve, "decorrelation_range_m": r}


# ------------------------------------------------------------------------------------
# Test-region selection + folds + buffer
# ------------------------------------------------------------------------------------
def _joint_tv_distance(df: pd.DataFrame, test: set[str], unit_col: str,
                       cols: list[str]) -> float:
    """Total-variation distance between test and trainval on the joint ``cols`` histogram.

    Half the L1 distance between the two normalised joint distributions: 0 = identical
    composition, 1 = disjoint. Used to score candidate test draws (§4.4).
    """
    is_test = df[unit_col].isin(test)
    if not is_test.any() or is_test.all():
        return 1.0
    key = df[cols].astype(str).agg("|".join, axis=1)
    a = key[is_test].value_counts(normalize=True)
    b = key[~is_test].value_counts(normalize=True)
    idx = a.index.union(b.index)
    return float(0.5 * (a.reindex(idx, fill_value=0.0)
                        - b.reindex(idx, fill_value=0.0)).abs().sum())


def _draw_test_units(df: pd.DataFrame, unit_col: str, test_frac: float,
                     rng: np.random.Generator) -> set[str]:
    """One shuffled draw of whole units up to ``test_frac`` of parcels, class-repaired."""
    # np.asarray(..., object): `unique()` on an Arrow-backed column returns an ArrowStringArray,
    # which numpy shuffles unsafely (it warns that the result may contain duplicates).
    units = np.asarray(df[unit_col].unique(), dtype=object)
    rng.shuffle(units)
    target = test_frac * len(df)
    sizes = df.groupby(unit_col).size()
    test, acc = set(), 0
    for b in units:
        if acc >= target:
            break
        test.add(b)
        acc += int(sizes[b])
    # ensure every class appears in test: move in the smallest unit containing it
    for cls, sub in df.groupby("label"):
        if not sub[unit_col].isin(test).any():
            test.add(sub.groupby(unit_col).size().idxmin())
    # ensure every class still appears in trainval (never move a class's only units out)
    train_classes = set(df.loc[~df[unit_col].isin(test), "label"])
    for cls in set(df["label"]) - train_classes:   # give the largest unit back
        test.discard(df[df["label"] == cls].groupby(unit_col).size().idxmax())
    return test


def pick_test_units(df: pd.DataFrame, unit_col: str, test_frac: float, seed: int,
                    balance_cols: list[str] | None = None,
                    n_candidates: int = 1) -> set[str]:
    """Choose the locked-test units, optionally **balanced on ``balance_cols``** (§4.4).

    The original draw was a single shuffle, and ``splits.py`` never read ``year``. Titling
    swept region by region, so a purely spatial draw is also a temporal draw: Piura's locked
    test came out **48.3 % label-year 1998 against trainval's 32.9 %** — an accidental
    temporal split nobody designed, and one that interacts with the 1997-98 El Niño and with
    the year↔label confound (1998 is 0.4 % perennial, 2000 is 33.8 %).

    The fix is cheap: draw ``n_candidates`` independent region sets at the same target
    fraction and keep the one minimising the total-variation distance between test and
    trainval on the joint ``year × label`` distribution. Spatial contiguity, the class
    repairs and the test fraction are all unchanged — only *which* equally-valid draw is
    taken. ``n_candidates=1`` reproduces the original behaviour exactly.
    """
    rng = np.random.default_rng(seed)
    cols = [c for c in (balance_cols or []) if c in df.columns]
    best, best_d = None, np.inf
    for _ in range(max(1, n_candidates)):
        cand = _draw_test_units(df, unit_col, test_frac, rng)
        if not cols:
            return cand
        d = _joint_tv_distance(df, cand, unit_col, cols)
        if d < best_d:
            best, best_d = cand, d
    return best


def buffer_exclusions(xy: np.ndarray, held_mask: np.ndarray, cand_mask: np.ndarray,
                      buffer_m: float) -> np.ndarray:
    """True for candidate parcels within ``buffer_m`` of any held-out parcel."""
    out = np.zeros(len(xy), dtype=bool)
    if not held_mask.any() or not cand_mask.any():
        return out
    tree = cKDTree(xy[held_mask])
    d, _ = tree.query(xy[cand_mask], k=1)
    out[np.where(cand_mask)[0][d <= buffer_m]] = True
    return out


def assign(config_path: Path | None = None, save: bool = True) -> gpd.GeoDataFrame:
    cfg = load_config(config_path)
    f_parcels, f_meta, f_blocks = out_paths()
    df = gpd.read_parquet(f_parcels)
    print(f"loaded {len(df):,} parcels")

    # ---- blocks (1 km, reporting) + super-regions (test/fold units) ----
    metric_crs = cfg.get("metric_crs", DEFAULT_METRIC_CRS)
    m = df.geometry.representative_point().to_crs(metric_crs)
    xy = np.c_[m.x.values, m.y.values]
    size = cfg["block_km"] * 1000.0
    df["block_id"] = [f"{int(x // size)}_{int(y // size)}" for x, y in xy]
    rsize = cfg["region_km"] * 1000.0
    df["region_id"] = [f"r{int(x // rsize)}_{int(y // rsize)}" for x, y in xy]
    n_blocks = df["block_id"].nunique()
    n_regions = df["region_id"].nunique()
    print(f"{n_blocks:,} populated {cfg['block_km']:g} km blocks; "
          f"{n_regions:,} populated {cfg['region_km']:g} km regions "
          f"(median {df.groupby('region_id').size().median():.0f} parcels/region)")

    # ---- autocorrelation audit ----
    print("running autocorrelation audit …")
    audit = autocorrelation_audit(xy, df["label_id"].values, cfg)
    r = audit["decorrelation_range_m"]
    print(f"  baseline agreement {audit['baseline_agreement']:.3f}; "
          f"decorrelation range ≈ {r if r else '> ' + str(cfg['audit_max_dist_m'])} m")
    for c in audit["curve"]:
        if c["agreement"] is not None:
            print(f"    {c['lo_m']:>5.0f}-{c['hi_m']:>5.0f} m: {c['agreement']:.3f} "
                  f"(n={c['n_pairs']:,})")
    if r is None or r > cfg["buffer_m"]:
        print(f"  WARNING: buffer_m={cfg['buffer_m']} < decorrelation range — "
              f"neighbour leakage is only partially controlled. FLAGGED for review.")

    # ---- locked test (contiguous regions: buffer cost scales with perimeter) ----
    # `balance_test_on` + `n_test_candidates` add §4.4 of docs/all_peru/window_plan.md: pick
    # the best of N equally-valid draws on the joint year x label composition. Defaults keep
    # the original single-draw behaviour so existing splits stay reproducible.
    balance_cols = cfg.get("balance_test_on") or []
    n_cand = int(cfg.get("n_test_candidates", 1))
    test_regions = pick_test_units(df, "region_id", cfg["test_frac"], cfg["seed"],
                                   balance_cols=balance_cols, n_candidates=n_cand)
    df["split"] = np.where(df["region_id"].isin(test_regions), "test", "trainval")
    print(f"test: {int((df['split'] == 'test').sum()):,} parcels in "
          f"{len(test_regions):,} contiguous {cfg['region_km']:g} km regions "
          f"({(df['split'] == 'test').mean():.1%})")
    # Always measured and recorded, whether or not it was optimised — an accidental temporal
    # split is invisible unless someone writes the number down.
    tv_dist = {}
    for cols in ([balance_cols] if balance_cols else []) + [["year", "label"], ["year"],
                                                            ["label"]]:
        cols = [c for c in cols if c in df.columns]
        if cols:
            tv_dist["x".join(cols)] = _joint_tv_distance(df, test_regions, "region_id",
                                                         cols)
    for k, v in tv_dist.items():
        print(f"  test-vs-trainval total-variation distance on {k}: {v:.4f}")
    if "year" in df.columns:
        yr = pd.crosstab(df["year"], df["split"], normalize="columns")
        top = yr.reindex(yr["test"].sort_values(ascending=False).index).head(5)
        print("  label-year composition (top 5 test years):")
        print(top.round(3).to_string())

    # ---- CV folds on trainval (grouped by region -> contiguous folds too) ----
    tv = df[df["split"] == "trainval"]
    sgkf = StratifiedGroupKFold(n_splits=cfg["n_folds"], shuffle=True,
                                random_state=cfg["seed"])
    df["fold"] = -1
    for k, (_, val_idx) in enumerate(
            sgkf.split(tv, tv["label_id"], groups=tv["region_id"])):
        df.loc[tv.index[val_idx], "fold"] = k
    print("fold sizes:", df[df.fold >= 0].fold.value_counts().sort_index().tolist())

    # ---- buffered dead-zones ----
    is_test = (df["split"] == "test").values
    is_tv = ~is_test
    df["buffer_excl_test"] = buffer_exclusions(xy, is_test, is_tv, cfg["buffer_m"])
    for k in range(cfg["n_folds"]):
        held = (df["fold"] == k).values
        cand = is_tv & ~held
        df[f"buffer_excl_fold{k}"] = buffer_exclusions(xy, held, cand, cfg["buffer_m"])
    n_bt = int(df["buffer_excl_test"].sum())
    n_bf = [int(df[f"buffer_excl_fold{k}"].sum()) for k in range(cfg["n_folds"])]
    print(f"buffer ({cfg['buffer_m']} m) excludes {n_bt:,} parcels from final training; "
          f"per-fold {n_bf}")

    # ---- per-class block/region report ----
    rep = (df.groupby("label")
             .agg(n_parcels=("label", "size"),
                  n_blocks=("block_id", "nunique"),
                  n_regions=("region_id", "nunique"),
                  n_test=("split", lambda s: int((s == "test").sum())))
             .sort_values("n_parcels", ascending=False))
    rep["n_test_regions"] = df[df.split == "test"].groupby("label")["region_id"].nunique()
    rep["n_test_regions"] = rep["n_test_regions"].fillna(0).astype(int)
    print("\nper-class block coverage:")
    print(rep.to_string())

    if save:
        df.to_parquet(f_parcels, index=False)
        rep.to_csv(f_blocks)
        meta = {"config": cfg, "audit": audit, "n_blocks": int(n_blocks),
                "n_regions": int(n_regions), "n_test_regions": len(test_regions),
                "buffer_excluded_test": n_bt, "buffer_excluded_folds": n_bf,
                "test_balance": {"balance_test_on": balance_cols,
                                 "n_candidates": n_cand,
                                 "tv_distance": tv_dist}}
        with open(f_meta, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\nupdated {f_parcels}; wrote {f_meta.name}, {f_blocks.name}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Assign spatial blocks/folds/test (plan §5)")
    ap.add_argument("--config", type=Path, default=None, help="path to split.yaml")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    assign(config_path=args.config, save=not args.no_save)


if __name__ == "__main__":
    main()

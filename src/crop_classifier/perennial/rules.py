"""Rule-based phenology classifier — the scientific control (plan §4, decision D4).

An explicit, auditable depth-2 decision rule on NDVI level and NDVI seasonal amplitude.
If a transparent rule does nearly as well as an attention network, the paper should say
so — and the rule is far easier to defend when applied to years with no ground truth.

**The physics it encodes.** Over one Landsat year in Piura:

* *perennial* — canopy present all year: high NDVI **p25** (an orchard never goes bare),
  **low seasonal amplitude**, low BSI maximum;
* *annual* — one or two green peaks separated by bare soil: low NDVI p25, **high
  amplitude**, high BSI maximum;
* *pasture/fallow* — low-to-moderate NDVI throughout, low amplitude but a *low* mean.
  Low amplitude with a **high** mean is what separates perennial from this.

So the discriminating plane is (NDVI level, NDVI amplitude), and the rule is::

    if NDVI_p25 >= t_hi and NDVI_amp <= t_amp:   PERENNIAL
    elif NDVI_amp > t_amp or NDVI_max >= t_peak: ANNUAL
    else:                                        PASTURE_FALLOW

NDVI **p25** rather than min: the minimum is one cloud-edge pixel away from garbage.

Registered as a model so ``train.py``/``evaluate.py``/``infer.py`` drive it unchanged and
it goes through the identical spatial-CV protocol. Thresholds are **fitted on the training
fold only**, never on validation data. numpy/pandas only — no torch, no lightgbm (macOS
libomp).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from crop_classifier.models.base import register

DEFAULT_FEATURES = ("NDVI_p25", "NDVI_amp", "NDVI_max")


def _logistic(x: np.ndarray, scale: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x / max(scale, 1e-6)))


@register("rules")
class RuleModel:
    """Depth-2 phenology rule with grid-searched thresholds.

    ``predict_proba`` returns **soft** scores (a logistic squash of each margin), not
    one-hot: the reliability curve, temperature scaling and the probability-weighted area
    estimate all need a spread of confidences to mean anything.
    """

    input_kind = "flat"

    def __init__(self, features: tuple[str, ...] = DEFAULT_FEATURES,
                 grid_steps: int = 25, seed: int = 42,
                 softness: float = 0.05,
                 climate_features: tuple[str, ...] = (),
                 min_stratum: int = 60, **kw):
        self.features = tuple(features)
        self.grid_steps = grid_steps
        self.seed = seed
        self.softness = softness           # logistic scale for the soft scores
        self.thresholds: dict[str, float] = {}
        self.class_names: list[str] | None = None
        self._n_classes = 0
        self.fallback_class = 0            # training-majority class for NaN rows
        self.n_nan_rows = 0
        # climate stratification (see `_strata` for what it does and why)
        self.climate_features = tuple(climate_features)
        self.min_stratum = min_stratum
        self.climate_cuts: dict[str, float] = {}
        self.strata_thresholds: dict[str, dict[str, float]] = {}

    # -- registry plumbing -------------------------------------------------------------
    def set_n_classes(self, n: int) -> None:
        self._n_classes = n

    def set_class_names(self, names: list[str]) -> None:
        """Optional: lets the rule map its three semantic groups onto label ids."""
        self.class_names = list(names)

    # -- feature access ----------------------------------------------------------------
    def _columns(self, ds) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(level, amp, peak, nan_mask)`` for a flat dataset."""
        X: pd.DataFrame = ds.X
        missing = [f for f in self.features if f not in X.columns]
        if missing:
            raise KeyError(f"rule features missing from the feature store: {missing}")
        cols = [X[f].to_numpy(dtype=float) for f in self.features]
        level, amp, peak = cols[0], cols[1], cols[2]
        nan = np.isnan(level) | np.isnan(amp) | np.isnan(peak)
        return level, amp, peak, nan

    def _class_ids(self) -> tuple[int, int, int]:
        """Label ids of (PERENNIAL, ANNUAL, PASTURE_FALLOW).

        Falls back to alphabetical order — which for these three names *is*
        ANNUAL=0, PASTURE_FALLOW=1, PERENNIAL=2 — when class names were not supplied.
        """
        names = self.class_names or ["ANNUAL", "PASTURE_FALLOW", "PERENNIAL"]
        idx = {n: i for i, n in enumerate(names)}
        return (idx.get("PERENNIAL", 2), idx.get("ANNUAL", 0),
                idx.get("PASTURE_FALLOW", 1))

    # -- climate stratification ----------------------------------------------------------
    def _strata(self, ds) -> np.ndarray:
        """Stratum id per row from the climate columns, using the fitted median cuts.

        A rule model cannot take a covariate the way a booster does — there is no
        coefficient to give it. What climate *can* do inside a depth-2 phenology rule is
        move the thresholds: 'high NDVI all year' means something different in a 1,600 mm
        valley and on the 20 mm coastal desert, so the rule is fitted **separately either
        side of the training median** of each climate column. One column gives 2 strata,
        two columns give a 2x2 = 4.

        ⚠️ This is the one place in this comparison where the extra input also multiplies
        the number of fitted parameters — 3 thresholds per stratum instead of 3 in total —
        so an improvement here is not the same kind of evidence as an improvement in
        LightGBM, and a *loss* here can be plain overfitting on ~140 rows per cell.
        ``min_stratum`` sends any thinner cell back to the global thresholds.
        """
        if not self.climate_features:
            return np.zeros(len(ds.y) if hasattr(ds, "y") else len(ds.X), dtype=int)
        X = ds.X
        missing = [f for f in self.climate_features if f not in X.columns]
        if missing:
            raise KeyError(f"climate features missing from the feature store: {missing}")
        sid = np.zeros(len(X), dtype=int)
        for b, f in enumerate(self.climate_features):
            v = X[f].to_numpy(dtype=float)
            sid |= (np.nan_to_num(v, nan=self.climate_cuts[f]) > self.climate_cuts[f]
                    ).astype(int) << b
        return sid

    # -- the rule ----------------------------------------------------------------------
    def _apply(self, level, amp, peak, t_hi, t_amp, t_peak) -> np.ndarray:
        per, ann, pas = self._class_ids()
        out = np.full(len(level), pas, dtype=np.int64)
        is_annual = (amp > t_amp) | (peak >= t_peak)
        out[is_annual] = ann
        out[(level >= t_hi) & (amp <= t_amp)] = per
        return out

    def fit(self, train_ds, val_ds, class_weight: np.ndarray, run_dir: Path):
        """Grid-search the three thresholds on TRAIN ONLY, maximising class-weighted
        macro-F1. ``val_ds`` is accepted for interface compatibility and deliberately
        never read."""
        level, amp, peak, nan = self._columns(train_ds)
        y = np.asarray(train_ds.y)
        self.n_nan_rows = int(nan.sum())
        ok = ~nan
        counts = np.bincount(y, minlength=max(self._n_classes, 1))
        self.fallback_class = int(counts.argmax())

        def q(v: np.ndarray, n: int) -> np.ndarray:
            """Grid over quantiles of the training distribution, not of an arbitrary
            fixed range — thresholds stay meaningful whatever the radiometry."""
            return np.unique(np.nanquantile(v[ok], np.linspace(0.02, 0.98, n)))

        g_hi, g_amp, g_peak = (q(level, self.grid_steps), q(amp, self.grid_steps),
                               q(peak, self.grid_steps))
        yo, lo, ao, po = y[ok], level[ok], amp[ok], peak[ok]
        n_cls = max(self._n_classes, int(yo.max()) + 1)

        def search(sel: np.ndarray) -> tuple[float, float, float, float]:
            yy, ll, aa, pp = yo[sel], lo[sel], ao[sel], po[sel]
            best = (-1.0, float(np.median(g_hi)), float(np.median(g_amp)),
                    float(np.nanmax(g_peak)))
            for t_hi in g_hi:
                for t_amp in g_amp:
                    for t_peak in g_peak:
                        f1 = _macro_f1(yy, self._apply(ll, aa, pp, t_hi, t_amp, t_peak),
                                       n_cls)
                        if f1 > best[0]:
                            best = (f1, float(t_hi), float(t_amp), float(t_peak))
            return best

        best = search(np.ones(len(yo), dtype=bool))
        self.thresholds = {"NDVI_level_hi": best[1], "NDVI_amp_max": best[2],
                           "NDVI_peak_min": best[3], "train_macro_f1": best[0]}
        print(f"  rules: t_level>={best[1]:.3f}, t_amp<={best[2]:.3f}, "
              f"t_peak>={best[3]:.3f}  (train macro-F1 {best[0]:.3f}, "
              f"{self.n_nan_rows:,} NaN rows -> class {self.fallback_class})")

        # --- one threshold set per climate stratum, fitted on train only ---
        self.climate_cuts, self.strata_thresholds = {}, {}
        if self.climate_features:
            for f in self.climate_features:
                self.climate_cuts[f] = float(np.nanmedian(
                    train_ds.X[f].to_numpy(dtype=float)[ok]))
            sid = self._strata(train_ds)[ok]
            for k in np.unique(sid):
                sel = sid == k
                if int(sel.sum()) < self.min_stratum:
                    print(f"  rules[clim stratum {k}]: n={int(sel.sum())} "
                          f"< min_stratum {self.min_stratum} -> global thresholds")
                    continue
                b = search(sel)
                self.strata_thresholds[str(int(k))] = {
                    "NDVI_level_hi": b[1], "NDVI_amp_max": b[2],
                    "NDVI_peak_min": b[3], "train_macro_f1": b[0], "n": int(sel.sum())}
                print(f"  rules[clim stratum {k}]: n={int(sel.sum())} "
                      f"t_level>={b[1]:.3f}, t_amp<={b[2]:.3f}, t_peak>={b[3]:.3f} "
                      f"(train macro-F1 {b[0]:.3f})")
            cuts = {k: round(v, 1) for k, v in self.climate_cuts.items()}
            print(f"  rules: climate cuts {cuts}")

        if run_dir is not None:
            run_dir = Path(run_dir)
            with open(run_dir / "thresholds.json", "w") as f:
                json.dump({**self.thresholds, "features": list(self.features),
                           "n_nan_rows": self.n_nan_rows,
                           "fallback_class": self.fallback_class,
                           "climate_features": list(self.climate_features),
                           "climate_cuts": self.climate_cuts,
                           "strata_thresholds": self.strata_thresholds}, f, indent=2)
            try:
                self._boundary_png(lo, ao, yo, run_dir / "rule_boundary.png")
            except Exception as e:  # plotting must never fail a training run
                print(f"  (boundary plot skipped: {e})")
        return self

    def predict_proba(self, ds) -> np.ndarray:
        level, amp, peak, nan = self._columns(ds)
        n = len(level)
        # per-row thresholds: the global set, overridden inside any climate stratum that
        # was thick enough to fit its own
        t_hi = np.full(n, self.thresholds["NDVI_level_hi"], dtype=float)
        t_amp = np.full(n, self.thresholds["NDVI_amp_max"], dtype=float)
        t_peak = np.full(n, self.thresholds["NDVI_peak_min"], dtype=float)
        if self.climate_features and self.strata_thresholds:
            sid = self._strata(ds)
            for k, th in self.strata_thresholds.items():
                m = sid == int(k)
                t_hi[m], t_amp[m], t_peak[m] = (th["NDVI_level_hi"], th["NDVI_amp_max"],
                                                th["NDVI_peak_min"])
        per, ann, pas = self._class_ids()
        s = self.softness

        # soft membership of each rule branch; each is in (0,1) and calibratable
        hi = _logistic(np.nan_to_num(level - t_hi), s)          # NDVI level is high
        flat = _logistic(np.nan_to_num(t_amp - amp), s)         # amplitude is low
        tall = _logistic(np.nan_to_num(peak - t_peak), s)       # a strong green peak

        score = np.zeros((n, max(self._n_classes, 3)), dtype=float)
        score[:, per] = hi * flat
        score[:, ann] = np.maximum((1 - flat), tall) * (1 - hi * flat)
        score[:, pas] = (1 - hi) * flat
        score += 1e-3                                           # never a hard zero
        prob = score / score.sum(axis=1, keepdims=True)

        # rows with NaN rule features (fewer than 4 observations -> no harmonic fit)
        # fall through to the documented default; they are counted, never silent.
        if nan.any():
            prob[nan] = 1e-3
            prob[nan, self.fallback_class] = 1.0
            prob[nan] /= prob[nan].sum(axis=1, keepdims=True)
        return prob

    # -- persistence -------------------------------------------------------------------
    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump({"thresholds": self.thresholds, "features": list(self.features),
                       "n_classes": self._n_classes, "softness": self.softness,
                       "class_names": self.class_names,
                       "fallback_class": self.fallback_class,
                       "n_nan_rows": self.n_nan_rows,
                       "climate_features": list(self.climate_features),
                       "climate_cuts": self.climate_cuts,
                       "strata_thresholds": self.strata_thresholds}, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> RuleModel:
        with open(path) as f:
            s = json.load(f)
        m = cls(features=tuple(s["features"]), softness=s["softness"],
                climate_features=tuple(s.get("climate_features", ())))
        m.thresholds = s["thresholds"]
        m.climate_cuts = s.get("climate_cuts", {})
        m.strata_thresholds = s.get("strata_thresholds", {})
        m._n_classes = s["n_classes"]
        m.class_names = s.get("class_names")
        m.fallback_class = s.get("fallback_class", 0)
        m.n_nan_rows = s.get("n_nan_rows", 0)
        return m

    # -- diagnostics -------------------------------------------------------------------
    def _boundary_png(self, level, amp, y, path: Path) -> None:
        """The decision boundary in (NDVI level, amplitude) space over the training data."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = self.class_names or ["c0", "c1", "c2"]
        rng = np.random.default_rng(self.seed)
        take = rng.choice(len(level), min(6000, len(level)), replace=False)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for c in range(max(self._n_classes, 3)):
            m = y[take] == c
            ax.scatter(level[take][m], amp[take][m], s=3, alpha=0.25,
                       label=names[c] if c < len(names) else str(c))
        ax.axvline(self.thresholds["NDVI_level_hi"], color="k", lw=1)
        ax.axhline(self.thresholds["NDVI_amp_max"], color="k", lw=1)
        ax.set(xlabel=self.features[0], ylabel=self.features[1],
               title="rule decision boundary (train fold)")
        ax.legend(markerscale=4, fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Vectorised macro-F1 — the grid search calls this ~15k times per fold."""
    f1s = np.empty(n_classes)
    for c in range(n_classes):
        tp = np.count_nonzero((y_pred == c) & (y_true == c))
        fp = np.count_nonzero((y_pred == c) & (y_true != c))
        fn = np.count_nonzero((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        f1s[c] = (2 * tp / denom) if denom else 0.0
    return float(f1s.mean())

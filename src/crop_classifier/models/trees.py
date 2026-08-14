"""LightGBM baseline (plan.md §8 rung 1) — flat whole-year summary features, NaN-native.

Uses the native ``lgb.train`` API (not the sklearn wrapper) so training happens on the
**full** class space ``0..n_classes-1``: spatially-blocked folds routinely contain a class
absent from train (or vice versa), which breaks the sklearn wrapper's internal label
encoder. Native ``Dataset`` + ``num_class`` sidesteps that and always returns full-width
probabilities. CPU on every platform (A6); no interpolation anywhere. XGBoost would be a
drop-in alternative behind the same ``flat`` interface.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from crop_classifier.data import FlatData
from crop_classifier.models.base import register


@register("lightgbm")
class LightGBMModel:
    input_kind = "flat"

    def __init__(self, n_estimators: int = 2000, learning_rate: float = 0.05,
                 num_leaves: int = 63, min_child_samples: int = 20,
                 feature_fraction: float = 0.9, early_stopping_rounds: int = 100,
                 seed: int = 42, **kw):
        self.params = {"objective": "multiclass", "learning_rate": learning_rate,
                       "num_leaves": num_leaves, "min_child_samples": min_child_samples,
                       "feature_fraction": feature_fraction, "seed": seed,
                       "num_threads": 0, "verbosity": -1, **kw}
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.booster: lgb.Booster | None = None
        self.feature_names: list[str] = []
        self._n_classes: int = 0

    def set_n_classes(self, n: int) -> None:
        self._n_classes = n

    def fit(self, train_ds: FlatData, val_ds: FlatData, class_weight: np.ndarray,
            run_dir: Path):
        self.feature_names = train_ds.feature_names
        w = class_weight[train_ds.y]
        params = {**self.params, "num_class": self._n_classes, "metric": "multi_logloss"}
        dtrain = lgb.Dataset(train_ds.X, label=train_ds.y, weight=w,
                             feature_name=self.feature_names, free_raw_data=False)
        dval = lgb.Dataset(val_ds.X, label=val_ds.y, reference=dtrain,
                           feature_name=self.feature_names, free_raw_data=False)
        evals: dict = {}
        self.booster = lgb.train(
            params, dtrain, num_boost_round=self.n_estimators,
            valid_sets=[dtrain, dval], valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                       lgb.record_evaluation(evals), lgb.log_evaluation(0)],
        )
        # boosting curve = the loss-curve analogue (§9)
        pd.DataFrame({"train_logloss": evals["train"]["multi_logloss"],
                      "val_logloss": evals["val"]["multi_logloss"]}).to_csv(
            run_dir / "curves.csv", index=False)
        return self

    def predict_proba(self, ds: FlatData) -> np.ndarray:
        return self.booster.predict(ds.X)  # already [N, n_classes]

    def feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame({"feature": self.feature_names,
                             "gain": self.booster.feature_importance("gain")}
                            ).sort_values("gain", ascending=False)

    def save(self, path: Path) -> None:
        self.booster.save_model(str(path))
        np.save(path.with_suffix(".meta.npy"),
                {"n_classes": self._n_classes, "feature_names": self.feature_names},
                allow_pickle=True)

    @classmethod
    def load(cls, path: Path) -> LightGBMModel:
        m = cls()
        m.booster = lgb.Booster(model_file=str(path))
        meta = np.load(path.with_suffix(".meta.npy"), allow_pickle=True).item()
        m._n_classes = meta["n_classes"]
        m.feature_names = meta["feature_names"]
        return m

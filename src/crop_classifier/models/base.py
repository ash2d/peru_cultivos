"""Common model interface + registry (plan.md §10).

Trainer, evaluator and inference talk only to this interface, so the three models are
three files + three registry entries; adding a model = one file + one ``@register``.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np

InputKind = Literal["flat", "sequence", "pixelset"]

MODEL_REGISTRY: dict[str, type] = {}

# name -> module that defines & @register-s it. Imported on first use only, so a LightGBM
# run never imports torch and a torch run never imports lightgbm (macOS libomp clash).
_LAZY_MODULES = {
    "lightgbm": "crop_classifier.models.trees",
    "ltae": "crop_classifier.models.ltae",
    "psetae": "crop_classifier.models.psetae",
    # the phenology rule model (docs/RESULTS.md §2) — numpy/pandas only, so it is
    # safe alongside either of the above
    "rules": "crop_classifier.perennial.rules",
}

INPUT_KIND = {"lightgbm": "flat", "ltae": "sequence", "psetae": "pixelset",
              "rules": "flat"}


def register(name: str):
    def _wrap(cls):
        MODEL_REGISTRY[name] = cls
        cls.name = name
        return cls
    return _wrap


def _ensure_registered(name: str) -> None:
    if name not in MODEL_REGISTRY:
        if name not in _LAZY_MODULES:
            raise KeyError(f"unknown model '{name}' — available: {sorted(_LAZY_MODULES)}")
        importlib.import_module(_LAZY_MODULES[name])  # triggers @register


def get_model(name: str, **kwargs):
    _ensure_registered(name)
    return MODEL_REGISTRY[name](**kwargs)


def model_input_kind(name: str) -> str:
    """Input kind without importing the model's module (avoids loading torch)."""
    return INPUT_KIND[name]


@runtime_checkable
class CropModel(Protocol):
    name: str
    input_kind: InputKind

    def fit(self, train_ds, val_ds, class_weight: np.ndarray, run_dir: Path) -> CropModel: ...
    def predict_proba(self, ds) -> np.ndarray: ...           # [N, n_classes]
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> CropModel: ...

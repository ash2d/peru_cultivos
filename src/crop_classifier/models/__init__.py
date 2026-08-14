"""Model registry (plan.md §8/§10).

Models are registered **lazily** — importing this package does NOT import torch or
lightgbm. ``get_model(name)`` imports only the one module needed. This is essential on
macOS: torch and lightgbm each bundle their own ``libomp`` and loading both in one
process segfaults, so a LightGBM run must never import torch (and vice-versa).
"""

from crop_classifier.models.base import MODEL_REGISTRY, get_model  # noqa: F401

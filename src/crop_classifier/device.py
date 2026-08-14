"""Device selection + seeding (plan.md A6).

The two Torch models must run unchanged on Apple-Silicon MPS (M3 Air), CUDA (JASMIN
Orchid) and CPU. ``pick_device()`` selects ``cuda -> mps -> cpu``; nothing in the code
base may hard-code ``.cuda()``. Keep tensors fp32 (MPS has no fp64).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

# Any op MPS lacks silently falls back to CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def pick_device(prefer: str | None = None) -> torch.device:
    """Return the best available device (``cuda -> mps -> cpu``).

    ``prefer`` forces a specific device ("cuda" | "mps" | "cpu") and raises if it is
    unavailable — useful on the cluster to fail loudly rather than train on CPU.
    """
    if prefer:
        prefer = prefer.lower()
        if prefer == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if prefer == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    """Pin every RNG we use. Results should match across devices within fp tolerance."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

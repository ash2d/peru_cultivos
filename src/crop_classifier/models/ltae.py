"""LTAE — Lightweight Temporal Attention Encoder (plan.md §8 rung 2).

Garnot & Landrieu (2020)-style: per-date median vectors -> linear embedding + sinusoidal
DOY positions -> multi-head attention with *learned master queries* over the (masked,
irregular) date axis -> MLP head. No binning, no interpolation; the padding mask keeps
attention on real acquisitions only. The core (``LTAECore``) is shared with PSE-LTAE, so
rung 3 differs from rung 2 only by the pixel-set encoder — a controlled ablation.
"""

from __future__ import annotations

import torch
from torch import nn

from crop_classifier.models.base import register
from crop_classifier.models.torch_common import TorchModelBase, sinusoidal_position


class LTAECore(nn.Module):
    """Masked master-query attention over dates: [B,T,E] + doy + mask -> [B,E]."""

    def __init__(self, d_in: int, d_model: int = 128, n_head: int = 8):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model, self.n_head, self.d_k = d_model, n_head, d_model // n_head
        self.inproj = nn.Linear(d_in, d_model)
        self.norm_in = nn.LayerNorm(d_model)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.query = nn.Parameter(torch.randn(n_head, self.d_k) / self.d_k**0.5)
        self.norm_out = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
                                 nn.Linear(d_model, d_model))

    def forward(self, x: torch.Tensor, doy: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        e = self.norm_in(self.inproj(x) + sinusoidal_position(doy, self.d_model))
        k = self.key(e).view(B, T, self.n_head, self.d_k).transpose(1, 2)  # [B,h,T,dk]
        v = e.view(B, T, self.n_head, self.d_k).transpose(1, 2)            # [B,h,T,dk]
        att = torch.einsum("hd,bhtd->bht", self.query, k) / self.d_k**0.5  # [B,h,T]
        att = att.masked_fill(~mask.unsqueeze(1), float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = torch.nan_to_num(att, nan=0.0)  # rows with no valid dates
        out = torch.einsum("bht,bhtd->bhd", att, v).reshape(B, self.d_model)
        return self.norm_out(out + self.mlp(out))


class LTAENet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, d_model: int = 128,
                 n_head: int = 8, dropout: float = 0.2):
        super().__init__()
        self.core = LTAECore(n_channels, d_model, n_head)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, n_classes))

    def forward(self, x, doy, mask):
        return self.head(self.core(x, doy, mask))


@register("ltae")
class LTAEModel(TorchModelBase):
    input_kind = "sequence"

    def __init__(self, d_model: int = 128, n_head: int = 8, dropout: float = 0.2,
                 n_channels: int = 11, **kw):
        super().__init__(d_model=d_model, n_head=n_head, dropout=dropout,
                         n_channels=n_channels, **kw)

    def build_net(self, n_classes: int) -> nn.Module:
        kw = self.model_kw
        return LTAENet(kw["n_channels"], n_classes, kw["d_model"], kw["n_head"],
                       kw["dropout"])

    def forward_batch(self, batch: dict) -> torch.Tensor:
        return self.net(batch["x"], batch["doy"], batch["mask"])

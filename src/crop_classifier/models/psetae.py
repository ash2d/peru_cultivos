"""PSE-LTAE — Pixel-Set Encoder + LTAE (plan.md §8 rung 3, the SOTA candidate).

Per date, an MLP embeds each clear pixel; masked mean+std pooling over the pixel set
*learns* the per-date distribution summary (the principled version of "use percentiles",
§4); the pooled per-date embeddings feed the **same** ``LTAECore`` as rung 2, so the
LTAE -> PSE-LTAE delta isolates the value of the pixel-set encoder.
"""

from __future__ import annotations

import torch
from torch import nn

from crop_classifier.models.base import register
from crop_classifier.models.ltae import LTAECore
from crop_classifier.models.torch_common import TorchModelBase


class PSE(nn.Module):
    """Pixel-set encoder: [B,T,P,C] + pixmask -> per-date embeddings [B,T,d_out]."""

    def __init__(self, n_channels: int, d_pix: int = 64, d_out: int = 128):
        super().__init__()
        self.mlp1 = nn.Sequential(nn.Linear(n_channels, 32), nn.ReLU(),
                                  nn.Linear(32, d_pix), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Linear(2 * d_pix, d_out), nn.ReLU())

    def forward(self, x: torch.Tensor, pixmask: torch.Tensor) -> torch.Tensor:
        h = self.mlp1(x)                                   # [B,T,P,d]
        m = pixmask.unsqueeze(-1).float()
        n = m.sum(dim=2).clamp(min=1.0)                    # valid pixels per date
        mean = (h * m).sum(dim=2) / n
        var = ((h - mean.unsqueeze(2)) ** 2 * m).sum(dim=2) / n
        # eps INSIDE the sqrt: without it, single-pixel dates give var=0 and sqrt'(0)=inf
        # -> NaN gradients that poison training (found in the pilot smoke test).
        std = (var + 1e-6).sqrt()
        return self.mlp2(torch.cat([mean, std], dim=-1))


class PSETAENet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, d_pix: int = 64,
                 d_model: int = 128, n_head: int = 8, dropout: float = 0.2):
        super().__init__()
        self.pse = PSE(n_channels, d_pix, d_model)
        self.core = LTAECore(d_model, d_model, n_head)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, n_classes))

    def forward(self, x, doy, mask, pixmask):
        e = self.pse(x, pixmask)              # [B,T,d_model]
        return self.head(self.core(e, doy, mask))


@register("psetae")
class PSETAEModel(TorchModelBase):
    input_kind = "pixelset"

    def __init__(self, d_pix: int = 64, d_model: int = 128, n_head: int = 8,
                 dropout: float = 0.2, n_channels: int = 11, **kw):
        super().__init__(d_pix=d_pix, d_model=d_model, n_head=n_head,
                         dropout=dropout, n_channels=n_channels, **kw)

    def build_net(self, n_classes: int) -> nn.Module:
        kw = self.model_kw
        return PSETAENet(kw["n_channels"], n_classes, kw["d_pix"], kw["d_model"],
                         kw["n_head"], kw["dropout"])

    def forward_batch(self, batch: dict) -> torch.Tensor:
        return self.net(batch["x"], batch["doy"], batch["mask"], batch["pixmask"])

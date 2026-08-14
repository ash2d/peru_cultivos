"""Shared Torch training machinery for the two attention models (plan.md §8/§9, A6).

One generic fit loop: AdamW, class-weighted cross-entropy, early stopping on val
macro-F1, per-epoch curves.csv + curves.png. Device via ``pick_device()`` (cuda|mps|cpu);
everything fp32; no ``.cuda()`` anywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from crop_classifier.device import pick_device, seed_everything


class TorchModelBase:
    """fit/predict/save/load over an ``nn.Module`` built by the subclass."""

    input_kind = "sequence"  # overridden

    def __init__(self, epochs: int = 100, batch_size: int = 128, lr: float = 1e-3,
                 weight_decay: float = 1e-4, patience: int = 12, seed: int = 42,
                 grad_clip: float = 1.0, device: str | None = None, **model_kw):
        self.hp = dict(epochs=epochs, batch_size=batch_size, lr=lr,
                       weight_decay=weight_decay, patience=patience, seed=seed,
                       grad_clip=grad_clip)
        self.model_kw = model_kw
        self.device = pick_device(device)
        self.net: nn.Module | None = None
        self.normalizer_state: dict | None = None

    # subclass hooks -----------------------------------------------------------------
    def build_net(self, n_classes: int) -> nn.Module:
        raise NotImplementedError

    def forward_batch(self, batch: dict) -> torch.Tensor:
        raise NotImplementedError

    # generic loop -------------------------------------------------------------------
    def fit(self, train_ds, val_ds, class_weight: np.ndarray, run_dir: Path):
        seed_everything(self.hp["seed"])
        n_classes = len(class_weight)
        self.net = self.build_net(n_classes).to(self.device).float()
        self.normalizer_state = train_ds.normalizer.state()

        opt = torch.optim.AdamW(self.net.parameters(), lr=self.hp["lr"],
                                weight_decay=self.hp["weight_decay"])
        lossf = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weight, dtype=torch.float32, device=self.device))
        tl = DataLoader(train_ds, batch_size=self.hp["batch_size"], shuffle=True)
        vl = DataLoader(val_ds, batch_size=256)

        best_f1, best_state, patience_left = -1.0, None, self.hp["patience"]
        curves = []
        for epoch in range(self.hp["epochs"]):
            self.net.train()
            tr_loss, tr_pred, tr_true = 0.0, [], []
            for batch in tl:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.forward_batch(batch)
                loss = lossf(logits, batch["y"])
                if not torch.isfinite(loss):
                    continue  # skip a diverged batch rather than poison the weights
                opt.zero_grad()
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(self.net.parameters(),
                                                       self.hp["grad_clip"])
                if not torch.isfinite(gnorm):
                    continue  # non-finite grads must not reach the optimizer
                opt.step()
                tr_loss += float(loss) * len(batch["y"])
                tr_pred.append(logits.argmax(1).cpu().numpy())
                tr_true.append(batch["y"].cpu().numpy())
            tr_loss /= len(train_ds)
            if tr_true:
                tr_f1 = f1_score(np.concatenate(tr_true), np.concatenate(tr_pred),
                                 average="macro")
            else:  # every batch diverged (NaN loss) — record and keep going
                tr_f1 = 0.0

            va_loss, va_prob, va_true = self._eval_epoch(vl, lossf)
            va_f1 = f1_score(va_true, va_prob.argmax(1), average="macro")
            curves.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                           "train_macro_f1": tr_f1, "val_macro_f1": va_f1})
            if va_f1 > best_f1:
                best_f1, patience_left = va_f1, self.hp["patience"]
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.net.state_dict().items()}
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)

        cdf = pd.DataFrame(curves)
        cdf.to_csv(run_dir / "curves.csv", index=False)
        self._plot_curves(cdf, run_dir / "curves.png")
        return self

    @torch.no_grad()
    def _eval_epoch(self, loader, lossf):
        self.net.eval()
        loss, probs, trues, n = 0.0, [], [], 0
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            logits = self.forward_batch(batch)
            loss += float(lossf(logits, batch["y"])) * len(batch["y"])
            n += len(batch["y"])
            probs.append(torch.softmax(logits, 1).cpu().numpy())
            trues.append(batch["y"].cpu().numpy())
        return loss / max(n, 1), np.concatenate(probs), np.concatenate(trues)

    @torch.no_grad()
    def predict_proba(self, ds) -> np.ndarray:
        self.net.eval()
        out = []
        for batch in DataLoader(ds, batch_size=256):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            logits = torch.nan_to_num(self.forward_batch(batch))
            out.append(torch.softmax(logits, 1).cpu().numpy())
        return np.concatenate(out)

    @staticmethod
    def _plot_curves(cdf: pd.DataFrame, path: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        axes[0].plot(cdf.epoch, cdf.train_loss, label="train")
        axes[0].plot(cdf.epoch, cdf.val_loss, label="val")
        axes[0].set(title="loss", xlabel="epoch")
        axes[1].plot(cdf.epoch, cdf.train_macro_f1, label="train")
        axes[1].plot(cdf.epoch, cdf.val_macro_f1, label="val")
        axes[1].set(title="macro-F1", xlabel="epoch")
        for ax in axes:
            ax.legend()
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)

    # persistence --------------------------------------------------------------------
    def save(self, path: Path) -> None:
        torch.save({"state_dict": self.net.state_dict(), "hp": self.hp,
                    "model_kw": self.model_kw, "n_classes": self._n_classes,
                    "normalizer": self.normalizer_state}, path)

    def set_n_classes(self, n: int) -> None:
        self._n_classes = n

    @classmethod
    def load(cls, path: Path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(**blob["hp"], **blob["model_kw"])
        m._n_classes = blob["n_classes"]
        m.normalizer_state = blob["normalizer"]
        m.net = m.build_net(blob["n_classes"]).float()
        m.net.load_state_dict(blob["state_dict"])
        m.net.to(m.device)
        return m


def sinusoidal_position(doy: torch.Tensor, d_model: int) -> torch.Tensor:
    """Sinusoidal encoding of day-of-year (period ~366) -> [B, T, d_model]."""
    device = doy.device
    i = torch.arange(d_model // 2, device=device, dtype=torch.float32)
    freq = (2 * torch.pi / 366.0) * (i + 1)
    ang = doy.unsqueeze(-1) * freq  # [B,T,d/2]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

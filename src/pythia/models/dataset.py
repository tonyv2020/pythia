"""Sliding-window Dataset over the P1 feature frame.

Given (features, targets), materialise one sample per valid anchor date:
    - X: features on the trailing ``encoder_length`` rows (already lagged by
      the covariate-lag gate — no leakage risk here).
    - y: multi-target vector at the anchor date.

The anchor's earliest valid date is index ``encoder_length - 1`` (need that
many trailing bars).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PythiaWindowDataset(Dataset):
    """PyTorch Dataset yielding (encoder-window, target) tensors for the TFT-lite family."""

    def __init__(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        encoder_length: int,
    ) -> None:
        # Align on shared index; drop rows with any NaN in features or targets.
        common = features.index.intersection(targets.index)
        f = features.loc[common].astype(np.float32).sort_index()
        t = targets.loc[common].astype(np.float32).sort_index()
        mask = f.notna().all(axis=1) & t.notna().all(axis=1)
        # A row is a VALID anchor if row i AND every row in (i-encoder_length+1, i)
        # has non-NaN features. Precompute:
        valid_flag = mask.values
        rolling_ok = np.zeros_like(valid_flag)
        for i in range(encoder_length - 1, len(valid_flag)):
            rolling_ok[i] = valid_flag[i - encoder_length + 1 : i + 1].all()

        self.features = f.values  # (N, F)
        self.targets = t.values  # (N, D)
        self.anchor_idx = np.where(rolling_ok)[0]
        self.encoder_length = encoder_length
        self.index = f.index  # for external referencing
        self.feature_names: list[str] = list(f.columns)
        self.target_names: list[str] = list(t.columns)

    def __len__(self) -> int:
        """Number of full encoder-length windows available in the frame."""
        return len(self.anchor_idx)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the (encoder tensor, target tensor) pair for the i-th window."""
        pos = int(self.anchor_idx[i])
        x = self.features[pos - self.encoder_length + 1 : pos + 1].copy()  # (T, F)
        y = self.targets[pos].copy()  # (D,)
        return torch.from_numpy(x), torch.from_numpy(y)

    def anchor_timestamp(self, i: int) -> pd.Timestamp:
        """Return the anchor (target) timestamp for the i-th window (last bar of the encoder)."""
        return self.index[int(self.anchor_idx[i])]

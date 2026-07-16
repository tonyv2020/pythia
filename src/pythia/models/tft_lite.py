"""TFT-lite — Temporal Fusion Transformer's essentials in ~150 lines.

The design mirrors the TFT paper (Lim et al., 2019) but is deliberately
lean so it (a) trains fast on a single RTX 2080 Ti and (b) has no external
dependency beyond PyTorch itself (avoids pytorch-forecasting's version pin
hell). We keep the three ideas that actually matter:

1. **Variable Selection Network** at the encoder input — a softmax over
   input channels that learns to zero-out low-signal covariates. This is
   the interpretability lever P1's inference API surfaces.
2. **Recurrent encoder** (LSTM) over the lagged window. TFT uses an
   LSTM+static-covariate-encoder stack; we keep the LSTM and drop the
   static path (all P0 features are dynamic).
3. **Attention pooling** over encoder outputs, followed by a **multi-quantile
   head** — the model outputs P05, P10, P25, P50, P75, P90, P95 for each
   target head (return + realized range).

Explicitly NOT here: the pytorch-forecasting TemporalFusionTransformer's
static-covariate encoding, its future-known feature separation (we hand
those in as regular inputs; the covariate-lag gate already prevents the
leak that separation was supposed to prevent), or its exogenous encoder.
P2/P3 can grow those if the P1 headline number justifies it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TFTLiteConfig:
    """Architecture + training hyperparameters."""

    n_features: int
    encoder_length: int = 60
    hidden_size: int = 16
    lstm_layers: int = 1
    dropout: float = 0.1
    n_targets: int = 2  # return + range
    quantiles: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

    def n_quantiles(self) -> int:
        return len(self.quantiles)


class VariableSelectionNetwork(nn.Module):
    """Learn per-channel softmax weights over the input channels.

    Returns weighted features + the weight tensor so downstream code can
    surface per-covariate importance.
    """

    def __init__(self, n_features: int, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_features),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, F)
        weights = torch.softmax(self.attention(x), dim=-1)  # (B, T, F)
        return x * weights * float(x.size(-1)), weights


class TFTLite(nn.Module):
    def __init__(self, cfg: TFTLiteConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vsn = VariableSelectionNetwork(cfg.n_features, cfg.hidden_size, cfg.dropout)
        self.encoder = nn.LSTM(
            input_size=cfg.n_features,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
        )
        # Multi-head attention pooling over encoder outputs.
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden_size,
            num_heads=1,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(cfg.hidden_size)
        # Per-target quantile heads.
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(cfg.hidden_size, cfg.hidden_size),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.hidden_size, cfg.n_quantiles()),
                )
                for _ in range(cfg.n_targets)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Args:
            x: (B, T, F) — lagged feature window.

        Returns:
            quantile predictions per target head: list of ``n_targets``
            tensors, each shape (B, K); AND variable-selection weights of
            shape (B, T, F) for interpretability.

        Also stashes the temporal attention weights on ``self._last_attn_w``
        (B, T) as a side-effect so downstream code can surface P5c
        attention-over-time viz without a signature change. The tensor is
        detached from the graph and lives on the same device as the input.
        """
        x_weighted, vsn_weights = self.vsn(x)
        h, _ = self.encoder(x_weighted)  # (B, T, H)
        # Attention pool: query = last step, keys = all steps.
        q = h[:, -1:, :]  # (B, 1, H)
        pooled, attn_w = self.attn(q, h, h)  # (B, 1, H), (B, 1, T)
        # attn_w shape (B, 1, T) — squeeze the query axis; detach so
        # readers don't hold a graph reference across the eval loop.
        self._last_attn_w = attn_w.detach().squeeze(1)  # (B, T)
        z = self.norm(pooled.squeeze(1) + h[:, -1, :])  # residual + LN
        quantile_preds = [head(z) for head in self.heads]
        return quantile_preds, vsn_weights

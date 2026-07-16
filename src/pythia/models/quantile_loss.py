"""Multi-quantile pinball loss (Wen et al., 2017; Wang et al., 2019).

For a batch of predictions ``q_hat`` of shape ``(B, K)`` and targets ``y`` of
shape ``(B,)`` where ``K = len(quantiles)``:

    L = mean_b mean_k L_q(y_b, q_hat[b,k])

with the standard tilted absolute error. Reduces to |y - median| at q=0.5.

Optionally penalises quantile-crossing (q_hat[k] > q_hat[k+1] for q_k < q_{k+1})
via a soft ReLU penalty — this is the honesty rail that prevents the model
from producing an invalid predictive posterior even when the pinball loss
is minimised locally.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def multi_quantile_pinball(
    q_hat: torch.Tensor,  # (B, K)
    y: torch.Tensor,  # (B,) or (B, 1)
    quantiles: torch.Tensor,  # (K,)
    crossing_penalty: float = 0.1,
) -> torch.Tensor:
    """Multi-quantile pinball loss with a penalty that discourages quantile crossing."""
    if y.dim() == 1:
        y = y.unsqueeze(1)
    diff = y - q_hat  # (B, K)
    q = quantiles.view(1, -1)
    pinball = torch.maximum(q * diff, (q - 1.0) * diff).mean()

    if crossing_penalty > 0 and q_hat.size(1) > 1:
        cross = F.relu(q_hat[:, :-1] - q_hat[:, 1:]).mean()
        return pinball + crossing_penalty * cross
    return pinball

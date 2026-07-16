"""Protocols every model / baseline must implement to be scoreable.

A probabilistic forecast is (mean, sigma) at minimum — from that we derive
P10, P50, P90 assuming Normal. If a model produces quantiles natively it can
subclass ``ProbForecast`` and override ``quantile``; the default
implementation is Normal.

Rationale: keeping the interface minimal (mean+sigma) makes it trivial to
score any P0 or later model. Miscalibrated Normal assumptions are handled
by the ``calibration`` metric — if a model claims narrow σ but P10-P90
coverage collapses, it fails eval loudly and honestly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd
from scipy.stats import norm  # type: ignore[import-not-found]


@dataclass(frozen=True)
class ProbForecast:
    """Probabilistic point forecast.

    ``mean`` and ``sigma`` are per-observation. Length must match the target
    horizon's index. ``sigma`` must be strictly positive; a σ = 0 model can
    add a floor (e.g. 1e-9) — but the calibration metric will then punish it.
    """

    mean: pd.Series
    sigma: pd.Series

    def __post_init__(self) -> None:
        if len(self.mean) != len(self.sigma):
            raise ValueError("mean and sigma must have identical length")
        if not (self.mean.index == self.sigma.index).all():
            raise ValueError("mean and sigma must share an index")
        if (self.sigma <= 0).any():
            raise ValueError("sigma must be strictly positive")

    def quantile(self, q: float) -> pd.Series:
        """P(q) forecast under the Normal(mean, sigma) posterior."""
        if not 0.0 < q < 1.0:
            raise ValueError("q must be in (0,1)")
        z = float(norm.ppf(q))
        return self.mean + z * self.sigma


class Model(Protocol):
    """A model that Pythia can walk-forward score.

    Every call to ``fit`` sees data STRICTLY up to ``train_end`` (walk-forward
    honesty). ``predict`` produces one row per eval index; ``sigma`` may vary
    per prediction. The harness passes the eval window's INDEX (dates) only —
    the model has already been ``fit`` on the training frame it needs.
    """

    def fit(self, train: pd.DataFrame) -> None:
        """Fit the model on the train frame (strictly ≤ split's train_end for walk-forward honesty)."""
        ...

    def predict(self, eval_index: pd.Index) -> ProbForecast:
        """Return a per-observation probabilistic forecast covering every timestamp in `eval_index`."""
        ...

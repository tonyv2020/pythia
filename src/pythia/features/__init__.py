"""P1 feature construction.

Two hard invariants (helen's covariate-lag gate for P1):

1. Every OBSERVED covariate at row `t` is sourced from data timestamped
   STRICTLY BEFORE the target's realization. Concretely: if the target is
   the return from close_{t} to close_{t+1}, all observed features on row t
   must come from data at times <= t (the close price at t is "at time t"
   and is safe for a next-step forecast, but any data at t+1 is not).
   The default P1 setup lags observed covariates by ``lag=1`` — so at row t
   we present the model with covariates at t-1 (their previous close) and
   the target is the return realised between t-1 and t.
2. Known-future calendar features (dow, month, days_to_fomc, is_earnings_season)
   are EXEMPT from the lag — they are causally future-safe by construction
   (you can compute Wednesday's day-of-week on Tuesday without seeing any
   market data).

The ``LagPolicy`` here is enforced structurally by ``build_features``: any
column not explicitly classified either gets lagged (default) or fails
loudly. See ``tests/test_feature_lag_no_within_row_leakage.py`` for the
gate test helen requires.
"""

from .lag import LagPolicy, build_features, default_policy_for
from .targets import realized_range_target, return_target

__all__ = [
    "LagPolicy",
    "build_features",
    "default_policy_for",
    "return_target",
    "realized_range_target",
]

"""Covariate-lag gate — the P1 hard-invariant.

At row ``t``, observed covariates (prices, volumes) must be lagged by AT
LEAST ``lag`` rows, so that they represent data from strictly before the
target's realization at ``t``. Known-future calendar features are exempt.

The ``LagPolicy`` classifies every column into one of three buckets:

- ``observed``   : must be lagged; typically prices and volumes.
- ``known_future``: not lagged; safe by construction (dow, is_month_end,
                    days_to_fomc, ...). Present at ``t`` when we ask for the
                    forecast for ``t``.
- ``target``     : dropped from the feature matrix (never a feature of
                   itself); target values are aligned separately.

Any column NOT in the policy raises at build time. This turns a silent
leakage bug into a loud one — the P1 acceptance test asserts this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


# Calendar features produced by ``pythia.data.calendar_features`` — these
# are known at forecast time and MUST NOT be lagged.
DEFAULT_KNOWN_FUTURE: frozenset[str] = frozenset(
    {
        "dow",
        "month",
        "dom",
        "is_monday",
        "is_friday",
        "is_month_end",
        "is_quarter_end",
        "days_to_fomc",
        "is_earnings_season",
        # Intraday calendar (if a downstream caller flattens intraday to daily):
        "minute_of_day",
        "minutes_to_close",
    }
)


@dataclass(frozen=True)
class LagPolicy:
    """Explicit classification of every column.

    Any column present in the input frame but absent from all three sets
    causes ``build_features`` to raise — silent inclusion is not permitted.
    """

    observed: frozenset[str]
    known_future: frozenset[str]
    targets: frozenset[str]

    def __post_init__(self) -> None:
        overlap = (
            (self.observed & self.known_future)
            | (self.observed & self.targets)
            | (self.known_future & self.targets)
        )
        if overlap:
            raise ValueError(f"column(s) appear in multiple LagPolicy buckets: {sorted(overlap)}")

    def classified(self) -> frozenset[str]:
        return self.observed | self.known_future | self.targets


def default_policy_for(
    columns: Iterable[str],
    target_cols: Iterable[str],
    known_future: Iterable[str] = DEFAULT_KNOWN_FUTURE,
) -> LagPolicy:
    """Build a LagPolicy from a column list — everything not calendar and
    not a target is treated as observed.

    Rationale: raptor's wide frame is ``{SYM}_close`` / ``{SYM}_volume`` plus
    the calendar suffix. All symbol columns are observed by construction,
    so a whitelist by NAMING CONVENTION is safe and easy to audit.
    """
    cols = set(columns)
    target_set = frozenset(cols & set(target_cols))
    known_set = frozenset(cols & set(known_future))
    observed = frozenset(cols - target_set - known_set)
    return LagPolicy(observed=observed, known_future=known_set, targets=target_set)


def build_features(
    daily_wide: pd.DataFrame,
    policy: LagPolicy,
    lag: int = 1,
) -> pd.DataFrame:
    """Produce the model-input feature frame.

    - Every ``observed`` column is shifted by ``lag`` rows (so row t carries
      the value from row t-lag).
    - Every ``known_future`` column is passed through unchanged.
    - Every ``target`` column is dropped.
    - Any column NOT classified by ``policy`` raises ``ValueError`` — the
      whole point of the policy is to be exhaustive.

    Rows whose observed columns become NaN due to the shift (the first
    ``lag`` rows) are DROPPED — the target for those rows can't have a
    fully-populated feature vector, and imputing zero would smuggle in
    look-ahead-like structure.
    """
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")

    cols = set(daily_wide.columns)
    stray = cols - policy.classified()
    if stray:
        raise ValueError(
            f"column(s) not in LagPolicy: {sorted(stray)} — classify them as "
            f"observed / known_future / target explicitly"
        )

    lagged = daily_wide[sorted(policy.observed)].shift(lag)
    lagged.columns = [f"{c}_lag{lag}" for c in lagged.columns]

    known = daily_wide[sorted(policy.known_future)]

    out = lagged.join(known, how="outer").sort_index()
    # Drop the first `lag` rows where lagged columns are all-NaN.
    if policy.observed:
        first_valid = out[lagged.columns].dropna(how="all").index.min()
        if first_valid is not None:
            out = out.loc[first_valid:]
    return out

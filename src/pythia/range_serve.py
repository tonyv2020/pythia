"""P5a serve helper — the range block for /latest (helen D25, twin's serve vote).

Computes the served realized-range forecast + its honest verdict so /latest can
return a ``range`` block alongside ``price``, keyed on the same model_version.
The served range model is a CONFORMAL-wrapped RollingRange (D25 decision:
rolling_range is the CRPS winner; conformal right-sizes its band; it lands
~0.88 = mildly over-dispersed → AMBER, disclosed honestly, not tight bands).

``mean`` is 0-free (range is positive) — the cone is [p10, p50, p90] of
``log(high/low)``. This is a dispersion diagnostic, NOT an alpha signal.
"""

from __future__ import annotations

from scipy.stats import norm  # type: ignore[import-not-found]

import pandas as pd

from .backtest.harness import run_backtest
from .backtest.splits import expanding_walk_forward
from .baselines import RollingRange
from .features.targets import realized_range_target
from .models.conformal import ConformalScaledModel

RANGE_MODEL = "conformal_rolling_range"
_Z10 = float(norm.ppf(0.10))
_Z90 = float(norm.ppf(0.90))


def _range_fn(high_col: str, low_col: str):
    def fn(frame: pd.DataFrame) -> pd.Series:
        return realized_range_target(frame[high_col], frame[low_col]).reindex(frame.index)

    return fn


def compute_range_block(
    wide: pd.DataFrame,
    symbol: str = "QQQ",
    window: int = 60,
    initial_train: int = 252,
    eval_size: int = 63,
) -> dict:
    """Return the range block: the LATEST conformal-rolling-range forecast cone
    (p10/p50/p90 of log(high/low)) + the honest walk-forward verdict
    (coverage_80, crps) + a calibration flag. ``wide`` must carry
    ``{symbol}_high`` / ``{symbol}_low`` (assemble with ``hl_symbols={symbol}``).
    """
    hi, lo = f"{symbol}_high", f"{symbol}_low"
    if hi not in wide.columns or lo not in wide.columns:
        raise ValueError(f"range block needs {hi}/{lo}; assemble with hl_symbols={{'{symbol}'}}")

    rfn = _range_fn(hi, lo)

    # 1) Honest verdict on the walk-forward (same machinery as the report).
    splits = list(
        expanding_walk_forward(wide.index, initial_train_size=initial_train, eval_size=eval_size)
    )
    verdict: dict = {}
    if splits:
        reports = run_backtest(
            wide,
            f"{symbol}_close",
            splits,
            {
                RANGE_MODEL: lambda: ConformalScaledModel(
                    base=RollingRange(hi, lo, window=window), target_fn=rfn, horizon=1
                )
            },
            rw_name=RANGE_MODEL,
            target_fn=rfn,
            horizon=1,
        )
        r = reports[RANGE_MODEL]
        verdict = {
            "coverage_80": r.coverage_80,
            "crps": r.crps,
            "n_eval_obs": r.n_eval_obs,
            "n_splits": r.n_splits,
        }

    # 2) The LATEST forecast cone: fit on ALL history, predict the next bar.
    model = ConformalScaledModel(base=RollingRange(hi, lo, window=window), target_fn=rfn, horizon=1)
    model.fit(wide)
    last_idx = wide.index[-1:]
    fc = model.predict(last_idx)
    mean = float(fc.mean.iloc[0])
    sigma = float(fc.sigma.iloc[0])
    p10 = max(mean + _Z10 * sigma, 0.0)  # range is positive
    p50 = max(mean, 0.0)
    p90 = mean + _Z90 * sigma

    cov = verdict.get("coverage_80", float("nan"))
    # amber = right-sized-ish but outside the strict gate (disclose, per D20/D25)
    calibrated = 0.75 <= cov <= 0.85 if cov == cov else False
    return {
        "target": "realized_range_pct",
        "model": RANGE_MODEL,
        "cone": {"p10": p10, "p50": p50, "p90": p90, "units": "log(high/low)"},
        "coverage_80": cov,
        "crps": verdict.get("crps"),
        "n_eval_obs": verdict.get("n_eval_obs"),
        "calibrated": calibrated,
        "badge": "green" if calibrated else "amber",
        "note": (
            "realized-range cone; conformal-calibrated rolling range. "
            "Disclosed AMBER when eval coverage drifts outside 0.75-0.85 "
            "(structural train->eval range-vol drift, D25). Dispersion "
            "diagnostic, not a trade signal."
        ),
    }

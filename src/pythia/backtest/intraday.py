"""P3 intraday walk-forward orchestration.

Ties the pieces together WITHOUT a forked harness: assembles nothing itself —
it takes an already-assembled intraday wide frame (``data.intraday``) plus the
raptor p_move / direction Series (``data.pmove_history``), builds the splits +
the forward-session mask, and calls the shared ``run_backtest`` with
``horizon`` + ``eval_mask``. So the leak-free invariant lives in one place.

Scored models: random_walk, last_return, and — when the raptor histories are
provided — raptor_p_move (calibrated dispersion) and raptor_direction (tilt).
"""

from __future__ import annotations

import pandas as pd

from ..baselines import LastReturn, RandomWalk, RaptorDirection, RaptorPMove
from ..data.intraday import forward_session_mask
from .harness import Report, run_backtest
from .splits import expanding_walk_forward


def run_intraday_backtest(
    bars_wide: pd.DataFrame,
    price_col: str = "QQQ_close",
    p_move: pd.Series | None = None,
    tilt: pd.Series | None = None,
    horizon: int = 3,
    initial_train: int = 200,
    eval_size: int = 39,
    rw_name: str = "random_walk",
    with_tft: bool = False,
    tft_kwargs: dict | None = None,
    tft_conformal: bool = True,
) -> dict[str, Report]:
    """Run the intraday walk-forward. ``horizon`` bars ≈ the forecast horizon
    (3 × 10-min = 30 min). ``eval_size`` defaults to ~1 session of 10-min bars.

    p_move / tilt are the raptor histories (bar-indexed); when absent the
    corresponding baseline is simply not scored (reported n reflects that).
    """
    if bars_wide.empty:
        return {}
    splits = list(
        expanding_walk_forward(
            bars_wide.index, initial_train_size=initial_train, eval_size=eval_size
        )
    )
    mask = forward_session_mask(bars_wide, horizon)

    factories = {
        "random_walk": lambda: RandomWalk(price_col),
        "last_return": lambda: LastReturn(price_col),
    }
    if p_move is not None and not p_move.empty:
        factories["raptor_p_move"] = lambda: RaptorPMove(price_col, p_move, horizon=horizon)
    if tilt is not None and not tilt.empty:
        factories["raptor_direction"] = lambda: RaptorDirection(price_col, tilt, horizon=horizon)
    if with_tft:
        # Lazy import: torch only needed on the model path (keeps the baseline
        # path torch-free). Horizon-consistent forward-h target subclass, wrapped
        # (default) in per-train-window conformal calibration so the cone is
        # honestly-sized + seed-independent (helen D18).
        from ..models.intraday_tft import IntradayTFTLiteModel

        kw = dict(tft_kwargs or {})

        def _make_tft():
            base = IntradayTFTLiteModel(target_col=price_col, horizon=horizon, **kw)
            if not tft_conformal:
                return base
            from ..models.conformal import ConformalScaledModel

            return ConformalScaledModel(base=base, target_col=price_col, horizon=horizon)

        factories["tft_lite"] = _make_tft

    return run_backtest(
        bars_wide,
        price_col,
        splits,
        factories,
        rw_name=rw_name,
        eval_mask=mask,
        horizon=horizon,
    )

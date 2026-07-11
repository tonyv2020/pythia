"""Documented stub for raptor's ``p_move`` baseline.

The task calls for scoring against raptor's ``p_move`` — but wiring the
live raptor endpoint into P0 would (a) make the harness depend on a running
raptor at eval time and (b) make replays non-reproducible without a
snapshot of p_move's history.

P0 decision (documented for helen): ship a stub that raises rather than a
fake number. Sha1: no accidentally-fabricated skill. P1 will add:
    (1) a persistent p_move history table in raptor,
    (2) an adapter here that reads from that snapshot,
so the harness can score against p_move on the SAME walk-forward splits.
"""

from __future__ import annotations

import pandas as pd

from ..backtest.protocols import Model, ProbForecast


class RaptorPMoveStub(Model):
    """Not-yet-implemented raptor-p_move baseline.

    fit() succeeds silently so a harness config can still list it as a
    planned baseline; predict() raises so no run silently omits it and
    reports as though everything passed.
    """

    def __init__(self, target_col: str) -> None:
        self.target_col = target_col

    def fit(self, train: pd.DataFrame) -> None:
        return None

    def predict(self, eval_index: pd.Index) -> ProbForecast:
        raise NotImplementedError(
            "raptor p_move adapter not wired in P0; land raptor p_move history "
            "snapshot + adapter in P1 (tracked as a follow-up task)."
        )

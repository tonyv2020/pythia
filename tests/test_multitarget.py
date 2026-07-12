"""P5a multi-target — the realized-range data contract (agent-2 backend).

The wide frame must expose {target}_high / {target}_low ONLY for the requested
symbols (target-only, so the covariate set + price model are unchanged), and the
realized-range target = log(high/low) must compute from them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pythia.data.assembler import _pivot_wide
from pythia.features.targets import realized_range_target


def _long():
    rows = []
    for sym, base in (("QQQ", 400.0), ("SPY", 470.0)):
        for i, d in enumerate(("2026-06-08", "2026-06-09", "2026-06-10")):
            c = base + i
            rows.append({"symbol": sym, "date": pd.Timestamp(d).date(),
                         "open": c - 0.5, "high": c + 1.0, "low": c - 1.0,
                         "close": c, "volume": 100 + i})
    return pd.DataFrame(rows)


def test_hl_added_only_for_requested_symbol():
    wide = _pivot_wide(_long(), hl_symbols={"QQQ"})
    # QQQ gets high/low; SPY does NOT (target-only → covariates unchanged).
    for col in ("QQQ_close", "QQQ_volume", "QQQ_high", "QQQ_low",
                "SPY_close", "SPY_volume"):
        assert col in wide.columns, col
    assert "SPY_high" not in wide.columns
    assert "SPY_low" not in wide.columns


def test_default_is_close_volume_only_no_hl():
    wide = _pivot_wide(_long())  # no hl_symbols → existing behaviour
    assert "QQQ_high" not in wide.columns
    assert "QQQ_low" not in wide.columns
    assert "QQQ_close" in wide.columns and "SPY_volume" in wide.columns


def test_realized_range_target_computes_from_wide_hl():
    wide = _pivot_wide(_long(), hl_symbols={"QQQ"})
    rng = realized_range_target(wide["QQQ_high"], wide["QQQ_low"])
    # log(high/low) with high=close+1, low=close-1 → positive, finite.
    assert (rng.dropna() > 0).all()
    assert np.isfinite(rng.dropna()).all()

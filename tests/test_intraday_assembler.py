"""P3 intraday assembler — hermetic tests (fixture ticks, no DB).

Validates tick→bar roll-up, the wide schema, session filtering, the
overnight session-open marker, and the mask helper. The backtest/splits/
baselines/metrics are horizon-agnostic and already covered by the daily
suite, so these tests target only the new intraday-specific code.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from pythia.data.intraday import (
    assemble_intraday_dataset,
    bars_from_ticks,
    overnight_mask,
)


def _ticks() -> pd.DataFrame:
    """Two symbols, two session dates, 30-min-spanning ticks + one
    extended-hours tick (08:00) that session_only must drop."""
    rows = []
    plan = [
        # (symbol, date, time, price, volume)
        ("QQQ", date(2026, 6, 8), "08:00:00", 400.0, 5),  # pre-market → dropped
        ("QQQ", date(2026, 6, 8), "09:31:00", 401.0, 10),  # bar 09:30
        ("QQQ", date(2026, 6, 8), "09:45:00", 403.0, 10),  # bar 09:30 (same 30m)
        ("QQQ", date(2026, 6, 8), "10:05:00", 402.0, 10),  # bar 10:00
        ("QQQ", date(2026, 6, 9), "09:31:00", 410.0, 10),  # next session, bar 09:30
        ("SPY", date(2026, 6, 8), "09:32:00", 500.0, 20),  # bar 09:30
        ("SPY", date(2026, 6, 8), "10:10:00", 505.0, 20),  # bar 10:00
    ]
    for s, d, tm, px, v in plan:
        rows.append({"symbol": s, "date": d, "time": tm, "price": px, "volume": v})
    return pd.DataFrame(rows)


def test_bars_roll_up_ohlcv_and_drop_extended_hours():
    bars = bars_from_ticks(_ticks(), bar_minutes=30, session_only=True)
    # QQQ 09:30 bar on 06-08 aggregates the 09:31 + 09:45 ticks (not 08:00).
    q0930 = bars[(bars.symbol == "QQQ") & (bars.bar_ts == pd.Timestamp("2026-06-08 09:30"))].iloc[0]
    assert q0930["open"] == 401.0
    assert q0930["close"] == 403.0
    assert q0930["high"] == 403.0
    assert q0930["low"] == 401.0
    assert q0930["volume"] == 20
    # The 08:00 pre-market tick is gone.
    assert not ((bars.symbol == "QQQ") & (bars.bar_ts == pd.Timestamp("2026-06-08 08:00"))).any()


def test_session_open_marks_first_bar_of_each_session():
    bars = bars_from_ticks(_ticks(), bar_minutes=30)
    qqq = bars[bars.symbol == "QQQ"].sort_values("bar_ts")
    # First QQQ bar of 06-08 and first of 06-09 are session-open; the 10:00
    # bar on 06-08 is not.
    opens = qqq.set_index("bar_ts")["is_session_open"].to_dict()
    assert (
        opens[pd.Timestamp("2026-06-08 09:30")] is True or opens[pd.Timestamp("2026-06-08 09:30")]
    )
    assert not opens[pd.Timestamp("2026-06-08 10:00")]
    assert opens[pd.Timestamp("2026-06-09 09:30")]


def test_wide_schema_and_calendar_features():
    res = assemble_intraday_dataset(
        date(2026, 6, 8),
        date(2026, 6, 9),
        bar_minutes=30,
        symbols=["QQQ", "SPY"],
        ticks_fn=lambda syms, s, e: _ticks(),
    )
    w = res.bars
    for col in (
        "QQQ_close",
        "QQQ_volume",
        "SPY_close",
        "SPY_volume",
        "is_session_open",
        "minute_of_day",
        "minutes_to_close",
        "dow",
    ):
        assert col in w.columns, col
    assert res.bar_minutes == 30
    assert set(res.symbols_included) == {"QQQ", "SPY"}
    # minutes_to_close at the 09:30 bar = 16:00 - 09:30 = 390.
    assert w.loc[pd.Timestamp("2026-06-08 09:30"), "minutes_to_close"] == 390
    # index is time-ordered bar timestamps
    assert list(w.index) == sorted(w.index)


def test_missing_symbol_recorded():
    res = assemble_intraday_dataset(
        date(2026, 6, 8),
        date(2026, 6, 9),
        symbols=["QQQ", "SPY", "NVDA"],
        ticks_fn=lambda syms, s, e: _ticks(),
    )
    assert "NVDA" in res.symbols_missing
    assert "QQQ" in res.symbols_included


def test_overnight_mask_excludes_session_open_rows():
    res = assemble_intraday_dataset(
        date(2026, 6, 8),
        date(2026, 6, 9),
        symbols=["QQQ", "SPY"],
        ticks_fn=lambda syms, s, e: _ticks(),
    )
    mask = overnight_mask(res.bars)
    # The 06-09 09:30 row is a session open → masked out (False = drop target).
    assert mask.loc[pd.Timestamp("2026-06-09 09:30")] == False  # noqa: E712
    # A mid-session bar is kept.
    assert mask.loc[pd.Timestamp("2026-06-08 10:00")] == True  # noqa: E712


def test_empty_ticks_yields_empty_frame():
    res = assemble_intraday_dataset(
        date(2026, 6, 8),
        date(2026, 6, 9),
        symbols=["QQQ"],
        ticks_fn=lambda syms, s, e: pd.DataFrame(
            columns=["symbol", "date", "time", "price", "volume"]
        ),
    )
    assert res.bars.empty
    assert res.symbols_missing == ("QQQ",)

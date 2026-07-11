"""D8 historical backfill — hermetic tests (no network).

The historical provider is injected as a deterministic fake, so these tests
exercise the normalization + stitch + schema-preservation logic without ever
hitting Yahoo/stooq. A separate opt-in live smoke (scripts/backfill_historical
--smoke) covers the real provider.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from pythia.data import assembler
from pythia.data.assembler import assemble_dataset
from pythia.data.historical import (
    DAILY_COLUMNS,
    _normalize_provider_frame,
    combine_daily,
    fetch_historical_daily_bars,
)


# --- fixtures -------------------------------------------------------------

def _yf_like_frame(dates: list[date], base: float) -> pd.DataFrame:
    """Mimic a yfinance single-ticker download: DatetimeIndex named 'Date',
    a (measure, ticker) MultiIndex on columns, capitalised measure names."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name="Date")
    cols = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["QQQ"]]
    )
    data = {
        ("Open", "QQQ"): [base + i for i in range(len(dates))],
        ("High", "QQQ"): [base + i + 1 for i in range(len(dates))],
        ("Low", "QQQ"): [base + i - 1 for i in range(len(dates))],
        ("Close", "QQQ"): [base + i + 0.5 for i in range(len(dates))],
        ("Volume", "QQQ"): [1_000_000 + i for i in range(len(dates))],
    }
    return pd.DataFrame(data, index=idx).reindex(columns=cols)


def _fake_provider(frames_by_symbol: dict[str, pd.DataFrame]):
    def provider(symbol, start, end, adjust):
        return frames_by_symbol.get(symbol, pd.DataFrame())
    return provider


# --- normalization --------------------------------------------------------

def test_normalize_handles_yfinance_multiindex():
    raw = _yf_like_frame([date(2024, 1, 2), date(2024, 1, 3)], base=400.0)
    out = _normalize_provider_frame(raw)
    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert out["date"].tolist() == [date(2024, 1, 2), date(2024, 1, 3)]
    assert out["close"].tolist() == [400.5, 401.5]


def test_normalize_drops_rows_without_close():
    raw = pd.DataFrame(
        {"Date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
         "Open": [1.0, 2.0], "High": [1.0, 2.0], "Low": [1.0, 2.0],
         "Close": [1.0, None], "Volume": [10, 20]}
    )
    out = _normalize_provider_frame(raw)
    assert len(out) == 1
    assert out["date"].tolist() == [date(2024, 1, 2)]


def test_normalize_missing_column_raises():
    raw = pd.DataFrame({"Date": [pd.Timestamp("2024-01-02")], "Close": [1.0]})
    with pytest.raises(ValueError):
        _normalize_provider_frame(raw)


# --- fetch ----------------------------------------------------------------

def test_fetch_returns_long_daily_columns_and_skips_unavailable():
    qqq = _yf_like_frame([date(2024, 1, 2), date(2024, 1, 3)], base=400.0)
    provider = _fake_provider({"QQQ": qqq})  # SPY absent → skipped, not fatal
    out = fetch_historical_daily_bars(
        ["QQQ", "SPY"], date(2024, 1, 1), date(2024, 1, 5), provider_fn=provider
    )
    assert list(out.columns) == DAILY_COLUMNS
    assert set(out["symbol"].unique()) == {"QQQ"}
    assert len(out) == 2


def test_fetch_provider_exception_is_swallowed_per_symbol():
    def boom(symbol, start, end, adjust):
        if symbol == "SPY":
            raise RuntimeError("network down")
        return _yf_like_frame([date(2024, 1, 2)], base=400.0)
    out = fetch_historical_daily_bars(
        ["QQQ", "SPY"], date(2024, 1, 1), date(2024, 1, 5), provider_fn=boom
    )
    assert set(out["symbol"].unique()) == {"QQQ"}


# --- combine --------------------------------------------------------------

def _long(symbol, rows):
    return pd.DataFrame(
        [{"symbol": symbol, "date": d, "open": c, "high": c, "low": c,
          "close": c, "volume": v} for d, c, v in rows]
    )


def test_combine_prefers_raptor_on_overlap():
    raptor = _long("QQQ", [(date(2026, 6, 10), 500.0, 9)])
    hist = _long("QQQ", [(date(2026, 6, 10), 111.0, 1),   # overlap → raptor wins
                         (date(2024, 1, 2), 400.0, 1)])    # old → historical fills
    out = combine_daily(raptor, hist, prefer="raptor")
    assert len(out) == 2
    overlap = out[out["date"] == date(2026, 6, 10)]["close"].iloc[0]
    old = out[out["date"] == date(2024, 1, 2)]["close"].iloc[0]
    assert overlap == 500.0   # raptor's value, not historical's 111.0
    assert old == 400.0


def test_combine_prefer_historical_flips_overlap():
    raptor = _long("QQQ", [(date(2026, 6, 10), 500.0, 9)])
    hist = _long("QQQ", [(date(2026, 6, 10), 111.0, 1)])
    out = combine_daily(raptor, hist, prefer="historical")
    assert out[out["date"] == date(2026, 6, 10)]["close"].iloc[0] == 111.0


def test_combine_unions_symbols():
    raptor = _long("QQQ", [(date(2026, 6, 10), 500.0, 9)])
    hist = _long("SPY", [(date(2024, 1, 2), 470.0, 1)])
    out = combine_daily(raptor, hist)
    assert set(out["symbol"].unique()) == {"QQQ", "SPY"}


def test_combine_bad_prefer_raises():
    with pytest.raises(ValueError):
        combine_daily(_long("QQQ", []), _long("QQQ", []), prefer="nonsense")


# --- assemble_dataset: schema preservation --------------------------------

def _raptor_intraday(symbols, dates):
    """Minimal intraday frame the assembler's daily roll-up understands."""
    rows = []
    for s in symbols:
        for i, d in enumerate(dates):
            rows.append({"symbol": s, "date": d, "time": "15:59:00",
                         "price": 500.0 + i, "volume": 100, "open_price": 499.0 + i})
    return pd.DataFrame(rows)


def test_backfill_preserves_wide_schema_and_adds_rows(monkeypatch):
    """Backfill must add ROWS (more history) without changing the wide COLUMN
    schema — so the covariate-lag gate + ffill-past-only downstream are
    untouched. Same symbol set in both sources → identical columns, more rows.
    """
    recent = [date(2026, 6, 8), date(2026, 6, 9), date(2026, 6, 10)]
    raptor_intraday = _raptor_intraday(["QQQ", "SPY"], recent)
    monkeypatch.setattr(assembler, "_fetch_intraday",
                        lambda *a, **k: raptor_intraday)

    hist = {
        "QQQ": _yf_like_frame([date(2024, 1, 2), date(2024, 1, 3)], base=400.0),
        "SPY": _yf_like_frame([date(2024, 1, 2), date(2024, 1, 3)], base=470.0),
    }
    provider = _fake_provider(hist)

    base = assemble_dataset(date(2026, 6, 1), date(2026, 6, 10),
                            engine=object(), symbols=["QQQ", "SPY"])
    fat = assemble_dataset(date(2026, 6, 1), date(2026, 6, 10), engine=object(),
                           symbols=["QQQ", "SPY"], historical_start=date(2024, 1, 1),
                           historical_provider_fn=provider)

    # Identical column schema — backfill did not alter the interface.
    assert list(base.daily.columns) == list(fat.daily.columns)
    # More rows (old history stitched under recent).
    assert len(fat.daily) > len(base.daily)
    assert set(fat.symbols_backfilled) == {"QQQ", "SPY"}
    # The known wide-frame price columns are present and unchanged in name.
    assert "QQQ_close" in fat.daily.columns
    assert "SPY_volume" in fat.daily.columns


def test_backfill_off_by_default_is_unchanged(monkeypatch):
    recent = [date(2026, 6, 9), date(2026, 6, 10)]
    monkeypatch.setattr(assembler, "_fetch_intraday",
                        lambda *a, **k: _raptor_intraday(["QQQ"], recent))
    res = assemble_dataset(date(2026, 6, 1), date(2026, 6, 10),
                           engine=object(), symbols=["QQQ"])
    assert res.symbols_backfilled == ()
    assert "QQQ_close" in res.daily.columns

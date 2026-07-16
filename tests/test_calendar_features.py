"""Calendar features: idempotent, no NaN blowups, days-to-FOMC monotone."""

from __future__ import annotations

import pandas as pd

from pythia.data.calendar_features import (
    FOMC_DATES,
    _days_to_next_fomc,
    add_calendar_features,
    intraday_calendar_features,
)


def _daily_frame() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    return pd.DataFrame({"QQQ_close": range(len(idx))}, index=idx)


def test_add_calendar_features_returns_expected_columns() -> None:
    df = add_calendar_features(_daily_frame())
    expected = {
        "dow",
        "month",
        "dom",
        "is_monday",
        "is_friday",
        "is_month_end",
        "is_quarter_end",
        "days_to_fomc",
        "is_earnings_season",
    }
    assert expected.issubset(df.columns)


def test_days_to_fomc_monotone_between_meetings() -> None:
    if len(FOMC_DATES) < 2:
        return  # nothing to test
    a, b = FOMC_DATES[0], FOMC_DATES[1]
    # For a date between two meetings, days_to_fomc must strictly decrease
    # day over day, and hit 0 on the meeting day itself.
    from datetime import timedelta

    prior = (b - a).days + 1
    d = a + timedelta(days=1)
    while d < b:
        cur = _days_to_next_fomc(d)
        assert cur < prior, f"days_to_fomc not monotone at {d}: {cur} !< {prior}"
        prior = cur
        d += timedelta(days=1)
    assert _days_to_next_fomc(b) == 0.0


def test_add_calendar_features_is_idempotent() -> None:
    once = add_calendar_features(_daily_frame())
    twice = add_calendar_features(once)
    # No duplicated columns; values match.
    assert list(once.columns) == list(twice.columns)


def test_intraday_helper_computes_minutes_to_close() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["QQQ", "QQQ"],
            "time": ["09:30:00", "15:59:00"],
        }
    )
    out = intraday_calendar_features(df)
    assert out["minute_of_day"].tolist() == [9 * 60 + 30, 15 * 60 + 59]
    assert out["minutes_to_close"].tolist() == [16 * 60 - (9 * 60 + 30), 1]

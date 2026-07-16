"""Deterministic known-future calendar features.

Only features that are *known at inference time from calendar alone* are
included here — no earnings-day realized volatility, no FOMC-decision text,
nothing that would leak. Each feature is a pure function of the date.

Features shipped in P0:
    - dow            : 0=Mon..6=Sun (integer)
    - month          : 1..12
    - dom            : 1..31
    - is_monday, is_friday : bool (weekday anchors)
    - is_month_end   : last trading day of the month? (approx: last date in
                       the frame within that month)
    - is_quarter_end : same for quarter
    - days_to_fomc   : signed days until the next scheduled FOMC decision;
                       negative on the day itself is 0; ``NaN`` if FOMC list
                       doesn't extend far enough forward.
    - is_earnings_season : bool, True during weeks 3-7 after each quarter-end
                       (a standard ~S&P 500 reporting window heuristic).

INTRADAY calendar features (time-of-day, minutes-to-close) live on the
intraday frame, not the daily one — the task calls them out but the P0
dataset is daily-granular so the intraday helper is provided but not
applied by ``add_calendar_features``. See ``intraday_calendar_features``.

FOMC list: hardcoded 2023-2026 published dates (source: FRB calendar
snapshot; see docs/data-schema.md). We do NOT scrape a live source in P0
because that would make the dataset non-reproducible.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


# Published FOMC statement dates. Update this list when the FRB publishes
# the next calendar. See docs/data-schema.md for provenance.
FOMC_DATES: tuple[date, ...] = (
    date(2023, 2, 1),
    date(2023, 3, 22),
    date(2023, 5, 3),
    date(2023, 6, 14),
    date(2023, 7, 26),
    date(2023, 9, 20),
    date(2023, 11, 1),
    date(2023, 12, 13),
    date(2024, 1, 31),
    date(2024, 3, 20),
    date(2024, 5, 1),
    date(2024, 6, 12),
    date(2024, 7, 31),
    date(2024, 9, 18),
    date(2024, 11, 7),
    date(2024, 12, 18),
    date(2025, 1, 29),
    date(2025, 3, 19),
    date(2025, 5, 7),
    date(2025, 6, 18),
    date(2025, 7, 30),
    date(2025, 9, 17),
    date(2025, 10, 29),
    date(2025, 12, 10),
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 5, 6),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 11, 4),
    date(2026, 12, 16),
)


def _days_to_next_fomc(d: date) -> float:
    """Signed days until the next FOMC date >= d. NaN if none found."""
    for f in FOMC_DATES:
        if f >= d:
            return float((f - d).days)
    return float("nan")


def _is_earnings_season(d: date) -> bool:
    """Weeks 3-7 after each quarter-end are heuristic reporting windows.

    Q1 (Jan-Mar) → reports mid-Apr to mid-May; Q2 → mid-Jul to mid-Aug;
    Q3 → mid-Oct to mid-Nov; Q4 → late-Jan to late-Feb.
    """
    # Days since last quarter-end (0-based).
    q_ends = [date(d.year, 3, 31), date(d.year, 6, 30), date(d.year, 9, 30), date(d.year, 12, 31)]
    # Days since PREVIOUS quarter end (so day-1 of Q1 refers back to Dec 31).
    last_q = max((q for q in q_ends if q < d), default=date(d.year - 1, 12, 31))
    delta = (d - last_q).days
    return 15 <= delta <= 49


def add_calendar_features(daily_wide: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features to a date-indexed wide frame.

    The frame is returned with the same shape plus the new columns. Idempotent:
    calling twice does not duplicate columns.
    """
    if daily_wide.empty:
        return daily_wide

    df = daily_wide.copy()
    # Robust to either a DatetimeIndex or a date-like index.
    idx = pd.to_datetime(df.index)
    dates = [d.date() for d in idx]

    df["dow"] = idx.weekday
    df["month"] = idx.month
    df["dom"] = idx.day
    df["is_monday"] = df["dow"] == 0
    df["is_friday"] = df["dow"] == 4

    # is_month_end / is_quarter_end computed WITHIN the frame's own coverage:
    # the last row in each month is that month's end. This is honest — it
    # doesn't require knowing the exchange calendar in advance.
    month_key = idx.strftime("%Y-%m")
    quarter_key = pd.PeriodIndex(idx, freq="Q").astype(str)
    df["is_month_end"] = pd.Series(month_key, index=df.index) != pd.Series(
        month_key, index=df.index
    ).shift(-1, fill_value="__END__")
    df["is_quarter_end"] = pd.Series(quarter_key, index=df.index) != pd.Series(
        quarter_key, index=df.index
    ).shift(-1, fill_value="__END__")

    df["days_to_fomc"] = np.array([_days_to_next_fomc(d) for d in dates])
    df["is_earnings_season"] = np.array([_is_earnings_season(d) for d in dates])

    return df


def intraday_calendar_features(intraday: pd.DataFrame) -> pd.DataFrame:
    """Add ``minute_of_day`` and ``minutes_to_close`` to an intraday frame.

    Regular-session close = 16:00 EDT/EST (assumed by raptor's clock; the
    frame is stored in ``time_zone='EDT'`` in staging.quote_raw). If the tick
    is after 16:00, minutes_to_close is negative (extended trading).
    """
    if intraday.empty:
        return intraday
    out = intraday.copy()
    t = pd.to_datetime(out["time"], format="%H:%M:%S", errors="coerce")
    out["minute_of_day"] = t.dt.hour * 60 + t.dt.minute
    close_min = 16 * 60
    out["minutes_to_close"] = close_min - out["minute_of_day"]
    return out

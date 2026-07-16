"""Load raptor's persisted p_move / direction history as bar-indexed Series.

Source (live, on achilles): raptor-intel Postgres, ``staging.qqq_pmove``
(date, bar_time, p_move, oos) and ``staging.qqq_direction`` (date, bar_time,
p_up, p_dn, ...). Each row is keyed by (date, bar_time) on a ~10-min grid;
we combine them into a bar-timestamp index that lines up with the intraday
assembler's ``bar_ts`` so the RaptorPMove / RaptorDirection baselines can
reindex onto the eval bars.

A ``rows_fn`` seam lets tests inject fixture rows and never touch Postgres.
"""

from __future__ import annotations

from typing import Callable, Iterable

import pandas as pd

# rows_fn(sql) -> iterable of tuples, in the SELECTed column order.
RowsFn = Callable[[str], Iterable[tuple]]

_PMOVE_SQL = "SELECT date, bar_time, p_move FROM staging.qqq_pmove"
_DIRECTION_SQL = "SELECT date, bar_time, p_up, p_dn FROM staging.qqq_direction"


def _bar_ts(date_txt: str, bar_time_txt: str) -> pd.Timestamp:
    """Convert a (date, bar_time) pair into a UTC bar timestamp for indexing p_move rows."""
    return pd.to_datetime(f"{date_txt} {bar_time_txt}")


def _default_rows_fn(dsn: str | None) -> RowsFn:
    """Return a callable that fetches raptor p_move rows for a date range from the default DSN."""

    def rows_fn(sql: str) -> Iterable[tuple]:
        # Lazy import; only needed on the live (achilles) path.
        from sqlalchemy import text

        from .source import get_engine

        engine = get_engine(dsn)
        with engine.connect() as conn:
            return list(conn.execute(text(sql)))

    return rows_fn


def load_pmove_series(rows_fn: RowsFn | None = None, dsn: str | None = None) -> pd.Series:
    """Return ``p_move`` as a Series indexed on bar timestamps (sorted, deduped
    keeping the last row per bar)."""
    rows_fn = rows_fn or _default_rows_fn(dsn)
    rows = list(rows_fn(_PMOVE_SQL))
    if not rows:
        return pd.Series(dtype="float64")
    rows = [r for r in rows if r[2] is not None]
    idx = [_bar_ts(str(d), str(t)) for d, t, _ in rows]
    vals = [float(p) for _, _, p in rows]
    s = pd.Series(vals, index=pd.DatetimeIndex(idx), name="p_move")
    return s[~s.index.duplicated(keep="last")].sort_index()


def load_direction_tilt(rows_fn: RowsFn | None = None, dsn: str | None = None) -> pd.Series:
    """Return the directional TILT ``p_up - p_dn`` as a bar-indexed Series."""
    rows_fn = rows_fn or _default_rows_fn(dsn)
    rows = list(rows_fn(_DIRECTION_SQL))
    if not rows:
        return pd.Series(dtype="float64")
    rows = [r for r in rows if r[2] is not None and r[3] is not None]
    idx = [_bar_ts(str(d), str(t)) for d, t, _, _ in rows]
    vals = [float(pu) - float(pd_) for _, _, pu, pd_ in rows]
    s = pd.Series(vals, index=pd.DatetimeIndex(idx), name="tilt")
    return s[~s.index.duplicated(keep="last")].sort_index()

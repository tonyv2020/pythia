"""Postgres source connection.

Reads DSN from ``PYTHIA_DB_DSN`` env var, falls back to the k3s in-cluster
raptor postgres DSN. Uses SQLAlchemy so pandas.read_sql can consume it.
"""

from __future__ import annotations

import os

from sqlalchemy import Engine, create_engine

from ..config import DEFAULT_DB_DSN


def get_engine(dsn: str | None = None) -> Engine:
    dsn = dsn or os.environ.get("PYTHIA_DB_DSN") or DEFAULT_DB_DSN
    return create_engine(dsn, pool_pre_ping=True, future=True)

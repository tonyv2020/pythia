"""Registry + serve unit tests using an in-memory SQLite fallback.

For CI without a live Postgres, we stub the registry with SQLite. The
real behaviour on Postgres is exercised in ``scripts/nightly_retrain.py``
integration; helen verified the schema in D3.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from pythia.registry.models import ModelRegistry, compute_dataset_hash
from pythia.serve.app import create_app


def _sqlite_registry() -> ModelRegistry:
    # StaticPool + a single shared connection = the ":memory:" DB persists
    # across every SQLAlchemy query (otherwise each connection gets a fresh
    # empty in-memory DB and inserts vanish before the next SELECT).
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # SQLite: swap Postgres-specific bits out. The registry uses ANSI SQL
    # for schema creation minus BIGSERIAL + JSONB — the test replaces those.
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE pythia_models ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " model_name TEXT NOT NULL,"
                " model_version TEXT NOT NULL,"
                " trained_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,"
                " dataset_hash TEXT NOT NULL,"
                " report_json TEXT NOT NULL,"
                " artifact_uri TEXT NOT NULL,"
                " git_sha TEXT NOT NULL,"
                " UNIQUE (model_name, model_version)"
                ")"
            )
        )
    reg = ModelRegistry(engine)
    reg._ensured = True  # bypass Postgres ensure_schema
    return reg


def _sqlite_register(reg: ModelRegistry, **kw) -> int:
    """Direct SQLite insert bypassing the Postgres-flavoured ON CONFLICT."""
    q = text(
        "INSERT INTO pythia_models "
        "(model_name, model_version, dataset_hash, report_json, artifact_uri, git_sha) "
        "VALUES (:name, :ver, :hash, :report, :uri, :sha)"
    )
    with reg.engine.begin() as conn:
        conn.execute(
            q,
            {
                "name": kw["model_name"],
                "ver": kw["model_version"],
                "hash": kw["dataset_hash"],
                "report": json.dumps(kw["report_json"]),
                "uri": kw["artifact_uri"],
                "sha": kw["git_sha"],
            },
        )
        row = conn.execute(text("SELECT last_insert_rowid() AS id")).one()
    return int(row.id)


def test_compute_dataset_hash_is_stable() -> None:
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        f.write(b"some bytes")
        p = Path(f.name)
    assert compute_dataset_hash(p) == hashlib.sha256(b"some bytes").hexdigest()


def test_registry_latest_returns_most_recent() -> None:
    reg = _sqlite_registry()
    _sqlite_register(
        reg,
        model_name="tft",
        model_version="v1",
        dataset_hash="a" * 64,
        report_json={"tft": {"coverage_80": 0.79}},
        artifact_uri="local://x",
        git_sha="deadbeef",
    )
    # Backdate the first so ORDER BY trained_at DESC has something to sort.
    with reg.engine.begin() as conn:
        conn.execute(
            text("UPDATE pythia_models SET trained_at = '2020-01-01' WHERE model_version='v1'")
        )
    _sqlite_register(
        reg,
        model_name="tft",
        model_version="v2",
        dataset_hash="b" * 64,
        report_json={"tft": {"coverage_80": 0.81}},
        artifact_uri="local://y",
        git_sha="cafebabe",
    )
    rec = reg.latest("tft")
    assert rec is not None
    assert rec.model_version == "v2"


def test_serve_latest_returns_calibration_verdict() -> None:
    reg = _sqlite_registry()
    _sqlite_register(
        reg,
        model_name="tft_lite_daily_qqq",
        model_version="v1",
        dataset_hash="c" * 64,
        report_json={"tft_lite_daily_qqq": {"coverage_80": 0.78, "mae": 0.012}},
        artifact_uri="local://z",
        git_sha="feedface",
    )

    app = create_app(registry=reg)
    client = TestClient(app)
    r = client.get("/latest")
    assert r.status_code == 200
    data = r.json()
    assert data["calibrated"] is True
    assert r.headers["x-pythia-calibrated"] == "true"


def test_serve_latest_flags_miscalibration() -> None:
    reg = _sqlite_registry()
    _sqlite_register(
        reg,
        model_name="tft_lite_daily_qqq",
        model_version="v1",
        dataset_hash="d" * 64,
        report_json={"tft_lite_daily_qqq": {"coverage_80": 0.55, "mae": 0.012}},
        artifact_uri="local://z",
        git_sha="feedface",
    )
    app = create_app(registry=reg)
    client = TestClient(app)
    r = client.get("/latest")
    assert r.status_code == 200
    assert r.json()["calibrated"] is False
    assert r.headers["x-pythia-calibrated"] == "false"


def test_serve_health() -> None:
    reg = _sqlite_registry()
    client = TestClient(create_app(registry=reg))
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_variable_importance_empty_when_absent() -> None:
    # helen D17: gracefully empty (200 + empty array) instead of 404 when the
    # trainer has not yet exported VSN weights. Panel already handles empty.
    reg = _sqlite_registry()
    _sqlite_register(
        reg,
        model_name="tft_lite_daily_qqq",
        model_version="v1",
        dataset_hash="e" * 64,
        report_json={"tft_lite_daily_qqq": {"coverage_80": 0.78}},
        artifact_uri="local://z",
        git_sha="feedface",
    )
    client = TestClient(create_app(registry=reg))
    r = client.get("/variable-importance")
    assert r.status_code == 200
    data = r.json()
    assert data["variable_importance"] == []
    assert data["model_name"] == "tft_lite_daily_qqq"


def test_variable_importance_returns_weights_when_present() -> None:
    reg = _sqlite_registry()
    _sqlite_register(
        reg,
        model_name="tft_lite_daily_qqq",
        model_version="v2",
        dataset_hash="f" * 64,
        report_json={
            "tft_lite_daily_qqq": {"coverage_80": 0.78},
            "variable_importance": [
                {"feature": "SPY_close_lag1", "weight": 0.42},
                {"feature": "QQQ_volume_lag1", "weight": 0.18},
            ],
        },
        artifact_uri="local://z",
        git_sha="deadbeef",
    )
    client = TestClient(create_app(registry=reg))
    r = client.get("/variable-importance")
    assert r.status_code == 200
    data = r.json()
    assert data["variable_importance"][0]["feature"] == "SPY_close_lag1"
    assert data["variable_importance"][0]["weight"] == 0.42

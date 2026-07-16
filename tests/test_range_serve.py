"""P5a serve helper — compute_range_block hermetic test (no DB, no GPU)."""

from __future__ import annotations
import numpy as np
import pandas as pd
from pythia.range_serve import compute_range_block, RANGE_MODEL


def _wide(n=400):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 400.0 + np.cumsum([((i * 17) % 7 - 3) / 50.0 for i in range(n)])
    spread = np.array([1.0 + ((i * 13) % 5) / 5.0 for i in range(n)])
    return pd.DataFrame(
        {"QQQ_close": close, "QQQ_high": close + spread, "QQQ_low": close - spread}, index=idx
    )


def test_range_block_shape_and_positive_cone():
    blk = compute_range_block(_wide(), symbol="QQQ", window=60, initial_train=120, eval_size=40)
    assert blk["model"] == RANGE_MODEL
    assert blk["target"] == "realized_range_pct"
    c = blk["cone"]
    # positive, ordered cone of log(high/low)
    assert 0.0 <= c["p10"] <= c["p50"] <= c["p90"]
    assert np.isfinite(blk["coverage_80"])
    assert blk["badge"] in ("green", "amber")
    assert "not a trade signal" in blk["note"]


def test_range_block_requires_hl():
    df = _wide().drop(columns=["QQQ_high", "QQQ_low"])
    try:
        compute_range_block(df)
        assert False
    except ValueError:
        pass


# --- /latest serve wiring (twin's serve vote: range block alongside price) ---
# Reuse the SQLite mock-registry helpers from the registry-serve suite so both
# blocks are exercised against the SAME create_app path prod uses.
from fastapi.testclient import TestClient  # noqa: E402

from test_registry_serve import _sqlite_register, _sqlite_registry  # noqa: E402
from pythia.serve.app import create_app  # noqa: E402


def _client_with(report_json: dict) -> TestClient:
    reg = _sqlite_registry()
    _sqlite_register(
        reg,
        model_name="tft_lite_daily_qqq",
        model_version="v1",
        dataset_hash="f" * 64,
        report_json=report_json,
        artifact_uri="local://x",
        git_sha="feedface",
    )
    return TestClient(create_app(registry=reg))


def test_latest_serves_range_block_when_present():
    # register_range_block writes report_json["range"]; /latest must surface it
    # alongside price, both from the same registered version.
    range_block = {
        "target": "realized_range_pct",
        "model": RANGE_MODEL,
        "cone": {"p10": 0.006, "p50": 0.009, "p90": 0.014, "units": "log(high/low)"},
        "coverage_80": 0.88,
        "crps": 0.006,
        "badge": "amber",
        "calibrated": False,
    }
    client = _client_with({"tft_lite_daily_qqq": {"coverage_80": 0.79}, "range": range_block})
    data = client.get("/latest").json()
    assert data["range"] == range_block  # served verbatim
    assert data["range"]["cone"]["p10"] <= data["range"]["cone"]["p90"]
    assert data["price"]["coverage_80"] == 0.79  # price block still keyed the same
    assert data["model_version"] == "v1"


def test_latest_range_block_null_when_absent():
    # Until register_range_block runs, no range key → /latest returns range=None
    # (the panel toggle hides the cone) but price is unaffected.
    client = _client_with({"tft_lite_daily_qqq": {"coverage_80": 0.79}})
    data = client.get("/latest").json()
    assert data["range"] is None
    assert data["price"]["coverage_80"] == 0.79

"""FastAPI app for the pythia inference API."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware

from ..registry import ModelRegistry, get_default_registry


DEFAULT_MODEL_NAME = "tft_lite_daily_qqq"
CALIBRATION_LOWER = 0.75
CALIBRATION_UPPER = 0.85


def create_app(registry: ModelRegistry | None = None) -> FastAPI:
    app = FastAPI(
        title="pythia",
        version="0.1.0",
        description=(
            "Probabilistic forecasting inference API. Read-only. Trained "
            "models come from pythia_models Postgres table (see D3)."
        ),
    )
    cors_origins = os.environ.get(
        "PYTHIA_CORS_ORIGINS", "https://raptor.tonyvigna.com,https://tonyvigna.com"
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in cors_origins if o.strip()],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    reg = registry
    def _reg() -> ModelRegistry:
        nonlocal reg
        if reg is None:
            reg = get_default_registry()
        return reg

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "pythia"}

    @app.get("/latest")
    def latest(
        response: Response,
        model: str = Query(DEFAULT_MODEL_NAME, min_length=1, max_length=100),
    ) -> dict:
        """Return the most recent registered model's walk-forward report + a
        calibration verdict header. Reports are the ACTUAL harness output —
        no fabrication."""
        rec = _reg().latest(model)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"no model registered under {model!r}")
        rep = rec.report_json.get(model) or next(iter(rec.report_json.values()), {})
        cov = float(rep.get("coverage_80", float("nan")))
        calibrated = CALIBRATION_LOWER <= cov <= CALIBRATION_UPPER
        response.headers["X-Pythia-Calibrated"] = "true" if calibrated else "false"
        return {
            "model_name": rec.model_name,
            "model_version": rec.model_version,
            "trained_at": rec.trained_at.isoformat(),
            "dataset_hash": rec.dataset_hash,
            "git_sha": rec.git_sha,
            "artifact_uri": rec.artifact_uri,
            "report": rec.report_json,
            "calibrated": calibrated,
            "calibration_band": [CALIBRATION_LOWER, CALIBRATION_UPPER],
            "notes": [
                "P1 verdict on daily QQQ: TFT-lite CALIBRATED but NULL SKILL vs random-walk.",
                "Data-limited (n=214 walk-forward eval obs); backfill (D8) firms this up.",
                "Do not size trades from these forecasts.",
            ],
        }

    @app.get("/variable-importance")
    def variable_importance(
        model: str = Query(DEFAULT_MODEL_NAME, min_length=1, max_length=100),
    ) -> dict:
        """TFT variable-selection weights. Loading a live checkpoint + running a
        forward pass is a phase-3 concern; P1 returns the STATIC weights logged
        into the registry alongside the report (if the trainer supplied them).
        Absent weights → 404 with a clear message."""
        rec = _reg().latest(model)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"no model registered under {model!r}")
        weights = rec.report_json.get("variable_importance")
        if weights is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "variable-importance not yet exported by trainer for this run — "
                    "P1 phase 2c/d task"
                ),
            )
        return {
            "model_name": rec.model_name,
            "model_version": rec.model_version,
            "trained_at": rec.trained_at.isoformat(),
            "variable_importance": weights,
        }

    return app


app = create_app()

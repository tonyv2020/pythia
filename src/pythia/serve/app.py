"""FastAPI app for the pythia inference API."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware

from ..registry import ModelRegistry, get_default_registry


DEFAULT_MODEL_NAME = "tft_lite_daily_qqq"
CALIBRATION_LOWER = 0.75
CALIBRATION_UPPER = 0.85


def _build_notes(tft: dict, rw: dict) -> list[str]:
    """Compose model-note bullets from the served rows.

    Everything is DERIVED from the report_json — no hardcoded n/coverage/skill
    strings — so a panel reload after a new nightly retrain shows the new
    numbers automatically. helen D17.
    """
    notes: list[str] = []
    n_obs = int(tft.get("n_eval_obs") or 0)
    n_splits = int(tft.get("n_splits") or 0)
    cov = tft.get("coverage_80")
    skill = tft.get("mae_skill_vs_rw")

    if cov is not None:
        cov_f = float(cov)
        gap = 0.80 - cov_f
        if CALIBRATION_LOWER <= cov_f <= CALIBRATION_UPPER:
            notes.append(
                f"P10-P90 coverage {cov_f:.3f} is CALIBRATED (in the "
                f"{CALIBRATION_LOWER}-{CALIBRATION_UPPER} gate)."
            )
        elif 0.65 <= cov_f < CALIBRATION_LOWER:
            # helen D20: near-miss under-coverage on daily QQQ = the systematic
            # train->eval drift (future modestly more uncertain than the recent
            # past). Honest label, not a lie about hitting 0.80.
            notes.append(
                f"eval cov80 ~{cov_f:.2f}; bands slightly tight — systematic "
                f"~{gap*100:.0f}pp train->eval drift on daily QQQ."
            )
        else:
            notes.append(
                f"P10-P90 coverage {cov_f:.3f} is MISCALIBRATED "
                f"(gate {CALIBRATION_LOWER}-{CALIBRATION_UPPER}); tail width is off."
            )

    if skill is not None:
        s = float(skill)
        if s > 0.02:
            notes.append(f"MAE-skill vs random-walk: +{100*s:.1f}% (better than RW).")
        elif abs(s) <= 0.05:
            notes.append(f"MAE-skill vs random-walk: {100*s:+.1f}% (NULL SKILL — indistinguishable from RW).")
        else:
            notes.append(f"MAE-skill vs random-walk: {100*s:+.1f}% (worse than RW).")

    if n_obs or n_splits:
        notes.append(f"Sample: n={n_obs} / {n_splits} walk-forward splits.")

    notes.append("Do not size trades from these forecasts.")
    return notes


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
        # helen D17: pull the tft_lite row + the random_walk row from the report
        # so notes reflect the ACTUAL served model, not a hardcoded D9-era string.
        tft = rec.report_json.get("tft_lite") or rec.report_json.get(model) or {}
        rw = rec.report_json.get("random_walk") or {}
        cov = float(tft.get("coverage_80", float("nan")))
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
            "notes": _build_notes(tft, rw),
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
        # helen D17: return gracefully-empty (200 with empty array) instead of
        # 404 when the trainer has not yet exported VSN weights — the panel
        # already hides the drivers strip on empty; 404 was cosmetic noise that
        # helped mask the raptor-white-screen incident. Real drivers get filled
        # once the TFT trainer starts writing report_json["variable_importance"].
        weights = rec.report_json.get("variable_importance") or []
        return {
            "model_name": rec.model_name,
            "model_version": rec.model_version,
            "trained_at": rec.trained_at.isoformat(),
            "variable_importance": weights,
        }

    return app


app = create_app()

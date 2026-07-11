"""Pythia inference API (FastAPI).

Exposes:
    GET /health                       — liveness probe.
    GET /latest?model={name}          — most recent registered forecast
                                        report + hydrated per-horizon quantiles.
    GET /variable-importance?model=…  — TFT VSN weights averaged across the
                                        most recent inference batch.

Design notes for downstream P2 (raptor panel):
- Read-only. No training here.
- Enforces the "MISCALIBRATED → don't trade" warning as a HTTP header
  (``X-Pythia-Calibrated: false``) so the panel can visually flag the state.
- Auth is deferred to a proxy (Authentik ProxyProvider like the raptor
  intel panel) — the API itself is trust-anything by default.
"""

from .app import create_app

__all__ = ["create_app"]

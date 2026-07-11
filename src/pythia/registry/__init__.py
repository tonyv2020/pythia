"""Pythia model registry — versioned models + their walk-forward reports.

D3 contract: `(model_name, model_version)` primary key, `trained_at`,
`dataset_hash`, `walk_forward_report_json`, `artifact_uri`, `git_sha`.
Latest-by-`trained_at` is "the current model."

Backend: Postgres. The trainer INSERTs; the inference API SELECTs the latest.
"""

from .models import (
    ModelRecord,
    ModelRegistry,
    compute_dataset_hash,
    current_git_sha,
    get_default_registry,
)

__all__ = [
    "ModelRecord",
    "ModelRegistry",
    "compute_dataset_hash",
    "current_git_sha",
    "get_default_registry",
]

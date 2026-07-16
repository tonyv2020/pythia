"""Postgres-backed model registry (D3).

Schema (idempotent — ``ensure_schema`` creates if missing):

    CREATE TABLE IF NOT EXISTS pythia_models (
        id             BIGSERIAL PRIMARY KEY,
        model_name     TEXT NOT NULL,
        model_version  TEXT NOT NULL,
        trained_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
        dataset_hash   TEXT NOT NULL,
        report_json    JSONB NOT NULL,
        artifact_uri   TEXT NOT NULL,
        git_sha        TEXT NOT NULL,
        UNIQUE (model_name, model_version)
    );

DSN is read from ``PYTHIA_REGISTRY_DSN`` env (default falls back to
``PYTHIA_DB_DSN``, then the in-cluster raptor DSN). Keeping the same DSN
as the source data by default is intentional — raptor's postgres is the
one always-on database on the cluster.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import Engine, text
from sqlalchemy.exc import ProgrammingError

from ..config import DEFAULT_DB_DSN
from ..data.source import get_engine


def _coerce_ts(v: object) -> datetime:
    """SQLite returns TIMESTAMP as ISO-8601 str; Postgres returns datetime.
    Accept either — the tests use SQLite, prod uses Postgres."""
    if isinstance(v, datetime):
        return v
    return datetime.fromisoformat(str(v).replace("Z", "+00:00"))


@dataclass(frozen=True)
class ModelRecord:
    """One registered model version: identity + metadata + the walk-forward report JSON."""
    id: int
    model_name: str
    model_version: str
    trained_at: datetime
    dataset_hash: str
    report_json: dict
    artifact_uri: str
    git_sha: str


def compute_dataset_hash(path: Path) -> str:
    """SHA-256 of the Parquet bytes. Deterministic across replays because
    ``pythia.data.assembler`` writes byte-identical output."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def current_git_sha(repo_root: Path | None = None) -> str:
    """Best-effort git SHA of the pythia repo. Falls back to ``unknown``
    when there's no .git (e.g. in-container without a mount)."""
    cwd = str(repo_root) if repo_root else None
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


class ModelRegistry:
    """Postgres-backed store of trained model versions and their walk-forward reports (D3)."""
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._ensured = False

    def ensure_schema(self) -> None:
        """Idempotent bootstrap.

        If the caller's role lacks CREATE on the target schema but the
        table already exists (the common prod path when the registry role
        is INSERT/UPDATE-only, e.g. raptor_ro), swallow the
        InsufficientPrivilege ONLY when we can verify the table is already
        there — a DBA created it out-of-band. Any other error still raises.
        """
        if self._ensured:
            return
        ddl = text(
            """
            CREATE TABLE IF NOT EXISTS pythia_models (
                id            BIGSERIAL PRIMARY KEY,
                model_name    TEXT NOT NULL,
                model_version TEXT NOT NULL,
                trained_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                dataset_hash  TEXT NOT NULL,
                report_json   JSONB NOT NULL,
                artifact_uri  TEXT NOT NULL,
                git_sha       TEXT NOT NULL,
                UNIQUE (model_name, model_version)
            );
            CREATE INDEX IF NOT EXISTS pythia_models_name_trained_idx
                ON pythia_models (model_name, trained_at DESC);
            """
        )
        try:
            with self.engine.begin() as conn:
                conn.execute(ddl)
        except ProgrammingError:
            # InsufficientPrivilege — verify the table already exists.
            with self.engine.connect() as conn:
                exists = conn.execute(
                    text("SELECT 1 FROM pg_class WHERE relname = 'pythia_models' AND relkind = 'r'")
                ).first()
            if exists is None:
                raise
        self._ensured = True

    def register(
        self,
        *,
        model_name: str,
        model_version: str,
        dataset_hash: str,
        report_json: dict,
        artifact_uri: str,
        git_sha: str,
    ) -> int:
        """Insert a new model version. Returns the row id.

        Idempotent on (model_name, model_version): if a duplicate arrives
        (e.g. rerun with same config) it UPDATES the report + hash instead
        of failing — this is how the nightly retrain refreshes numbers
        without inflating registry rows.
        """
        self.ensure_schema()
        q = text(
            """
            INSERT INTO pythia_models
                (model_name, model_version, dataset_hash, report_json, artifact_uri, git_sha)
            VALUES
                (:name, :ver, :hash, CAST(:report AS jsonb), :uri, :sha)
            ON CONFLICT (model_name, model_version) DO UPDATE
                SET dataset_hash = EXCLUDED.dataset_hash,
                    report_json  = EXCLUDED.report_json,
                    artifact_uri = EXCLUDED.artifact_uri,
                    git_sha      = EXCLUDED.git_sha,
                    trained_at   = now()
            RETURNING id;
            """
        ).bindparams(report=json.dumps(report_json))
        with self.engine.begin() as conn:
            row = conn.execute(
                q,
                {
                    "name": model_name,
                    "ver": model_version,
                    "hash": dataset_hash,
                    "report": json.dumps(report_json),
                    "uri": artifact_uri,
                    "sha": git_sha,
                },
            ).one()
        return int(row.id)

    def latest(self, model_name: str) -> ModelRecord | None:
        """Most recent registered record for ``model_name`` by trained_at, or None."""
        self.ensure_schema()
        q = text(
            """
            SELECT id, model_name, model_version, trained_at, dataset_hash,
                   report_json, artifact_uri, git_sha
            FROM pythia_models
            WHERE model_name = :name
            ORDER BY trained_at DESC
            LIMIT 1
            """
        )
        with self.engine.connect() as conn:
            r = conn.execute(q, {"name": model_name}).mappings().first()
        if r is None:
            return None
        report = (
            r["report_json"] if isinstance(r["report_json"], dict) else json.loads(r["report_json"])
        )
        return ModelRecord(
            id=int(r["id"]),
            model_name=r["model_name"],
            model_version=r["model_version"],
            trained_at=_coerce_ts(r["trained_at"]),
            dataset_hash=r["dataset_hash"],
            report_json=report,
            artifact_uri=r["artifact_uri"],
            git_sha=r["git_sha"],
        )

        self.ensure_schema()
        q = text(
            """
            SELECT id, model_name, model_version, trained_at, dataset_hash,
                   report_json, artifact_uri, git_sha
            FROM pythia_models
            WHERE model_name = :name
            ORDER BY trained_at DESC
            """
        )
        with self.engine.connect() as conn:
            rows = conn.execute(q, {"name": model_name}).mappings().all()
        out: list[ModelRecord] = []
        for r in rows:
            report = (
                r["report_json"]
                if isinstance(r["report_json"], dict)
                else json.loads(r["report_json"])
            )
            out.append(
                ModelRecord(
                    id=int(r["id"]),
                    model_name=r["model_name"],
                    model_version=r["model_version"],
                    trained_at=r["trained_at"],
                    dataset_hash=r["dataset_hash"],
                    report_json=report,
                    artifact_uri=r["artifact_uri"],
                    git_sha=r["git_sha"],
                )
            )
        return out


def _registry_dsn() -> str:
    return (
        os.environ.get("PYTHIA_REGISTRY_DSN") or os.environ.get("PYTHIA_DB_DSN") or DEFAULT_DB_DSN
    )


def get_default_registry() -> ModelRegistry:
    """ModelRegistry bound to the default DSN (PYTHIA_REGISTRY_DSN / PYTHIA_DB_DSN / in-cluster default)."""
    return ModelRegistry(get_engine(_registry_dsn()))

"""Walk-forward scoring harness — fits each model on each split's training
window, evaluates on the split's eval window, aggregates metrics.

Look-ahead enforcement is achieved structurally: the model NEVER sees the
eval frame. It receives (a) the train frame, and (b) the eval index only —
if it needs the actual y-values at eval time (for last_return-style logic)
it must extract them from the trailing edge of the train frame. The
harness passes ``target_col`` so the model knows which column is y.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from ..metrics import (
    coverage,
    crps_normal,
    directional_hit_rate,
    mae,
    mae_skill_vs,
    pinball_loss,
)
from .protocols import Model, ProbForecast
from .splits import WalkForwardSplit, slice_train_eval


@dataclass
class Report:
    """Aggregated per-model report, plus per-split rows for auditing."""

    model_name: str
    n_splits: int
    n_eval_obs: int
    mae: float
    hit_rate: float
    coverage_80: float          # empirical P10-P90 coverage on eval
    crps: float
    pinball_50: float
    pinball_10: float
    pinball_90: float
    mae_skill_vs_rw: float | None = None
    per_split: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["per_split"] = list(self.per_split)  # already dicts
        return d


ModelFactory = Callable[[], Model]


def _target_returns(frame: pd.DataFrame, target_col: str) -> pd.Series:
    """Log returns of the target series; NaN on the first row."""
    px = frame[target_col].astype(float)
    return np.log(px / px.shift(1))


def run_backtest(
    frame: pd.DataFrame,
    target_col: str,
    splits: Iterable[WalkForwardSplit],
    model_factories: Mapping[str, ModelFactory],
    rw_name: str = "random_walk",
) -> dict[str, Report]:
    """Fit each model on each split, score, aggregate.

    ``model_factories`` maps model_name → callable that RETURNS A FRESH
    model instance (needed because ``fit`` should not accumulate state
    across splits — that would be look-ahead by another name).

    Returns ``{model_name: Report}``. If a factory named ``rw_name`` is
    included, MAE-skill-vs-RW is computed for every other model.
    """
    frame = frame.sort_index()
    returns = _target_returns(frame, target_col)

    per_split_records: dict[str, list[dict]] = {name: [] for name in model_factories}

    splits = list(splits)
    for split in splits:
        train_frame, eval_frame = slice_train_eval(frame, split)
        if eval_frame.empty:
            continue
        y_true = returns.loc[eval_frame.index]
        # Skip splits where the target series has NaN (e.g., first row).
        y_true = y_true.dropna()
        if y_true.empty:
            continue
        y_true_idx = y_true.index

        for name, factory in model_factories.items():
            model = factory()
            model.fit(train_frame)
            fc: ProbForecast = model.predict(y_true_idx)
            if len(fc.mean) != len(y_true_idx):
                raise RuntimeError(
                    f"{name} produced {len(fc.mean)} predictions for "
                    f"{len(y_true_idx)} eval rows"
                )
            per_split_records[name].append(
                {
                    "eval_start": split.eval_start.isoformat(),
                    "eval_end": split.eval_end.isoformat(),
                    "y_true": y_true.to_list(),
                    "mean": fc.mean.to_list(),
                    "sigma": fc.sigma.to_list(),
                    "p10": fc.quantile(0.10).to_list(),
                    "p50": fc.quantile(0.50).to_list(),
                    "p90": fc.quantile(0.90).to_list(),
                }
            )

    return _aggregate(per_split_records, rw_name)


def _aggregate(
    per_split_records: dict[str, list[dict]], rw_name: str
) -> dict[str, Report]:
    reports: dict[str, Report] = {}

    # First pass: raw metrics per model. Second pass adds MAE-skill vs RW.
    for name, records in per_split_records.items():
        if not records:
            reports[name] = Report(
                model_name=name,
                n_splits=0,
                n_eval_obs=0,
                mae=float("nan"),
                hit_rate=float("nan"),
                coverage_80=float("nan"),
                crps=float("nan"),
                pinball_50=float("nan"),
                pinball_10=float("nan"),
                pinball_90=float("nan"),
                warnings=[f"{name}: no splits produced eval predictions"],
            )
            continue

        y_true = np.concatenate([np.asarray(r["y_true"]) for r in records])
        mean = np.concatenate([np.asarray(r["mean"]) for r in records])
        sigma = np.concatenate([np.asarray(r["sigma"]) for r in records])
        p10 = np.concatenate([np.asarray(r["p10"]) for r in records])
        p50 = np.concatenate([np.asarray(r["p50"]) for r in records])
        p90 = np.concatenate([np.asarray(r["p90"]) for r in records])

        reports[name] = Report(
            model_name=name,
            n_splits=len(records),
            n_eval_obs=len(y_true),
            mae=float(mae(y_true, mean)),
            hit_rate=float(directional_hit_rate(y_true, mean)),
            coverage_80=float(coverage(y_true, p10, p90)),
            crps=float(crps_normal(y_true, mean, sigma)),
            pinball_50=float(pinball_loss(y_true, p50, 0.50)),
            pinball_10=float(pinball_loss(y_true, p10, 0.10)),
            pinball_90=float(pinball_loss(y_true, p90, 0.90)),
        )

    # MAE-skill vs random-walk: only if RW report is available and non-NaN.
    rw = reports.get(rw_name)
    if rw is not None and np.isfinite(rw.mae) and rw.mae > 0:
        for name, r in reports.items():
            if name == rw_name:
                continue
            if not np.isfinite(r.mae):
                continue
            r.mae_skill_vs_rw = float(mae_skill_vs(r.mae, rw.mae))

    # Cross-check: if any model shows extreme miscalibration, flag it.
    for name, r in reports.items():
        if np.isfinite(r.coverage_80) and abs(r.coverage_80 - 0.80) > 0.05:
            r.warnings.append(
                f"P10-P90 coverage is {r.coverage_80:.3f} — MISCALIBRATED "
                f"(target 0.80 ± 0.05); do not trade against this forecast"
            )

    return reports

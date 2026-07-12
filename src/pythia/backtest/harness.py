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


def _target_returns(frame: pd.DataFrame, target_col: str, horizon: int = 1) -> pd.Series:
    """Log returns of the target series over ``horizon`` bars.

    ``horizon == 1`` (default): one-step TRAILING return ``log(px[t]/px[t-1])`` —
    the original behaviour, NaN on the first row (daily P1 no-op, bit-identical).
    ``horizon > 1``: FORWARD h-bar return ``log(px[t+h]/px[t])`` at row t — the
    move the model, forecasting AT bar t, is predicting; NaN on the last h rows.
    """
    px = frame[target_col].astype(float)
    if horizon == 1:
        return np.log(px / px.shift(1))
    return np.log(px.shift(-horizon) / px)


def run_backtest(
    frame: pd.DataFrame,
    target_col: str,
    splits: Iterable[WalkForwardSplit],
    model_factories: Mapping[str, ModelFactory],
    rw_name: str = "random_walk",
    eval_mask: "pd.Series | Callable[[pd.DataFrame], pd.Series] | None" = None,
    horizon: int = 1,
    purge_last_train_rows: int | None = None,
    target_fn: "Callable[[pd.DataFrame], pd.Series] | None" = None,
) -> dict[str, Report]:
    """Fit each model on each split, score, aggregate.

    ``model_factories`` maps model_name → callable that RETURNS A FRESH
    model instance (needed because ``fit`` should not accumulate state
    across splits — that would be look-ahead by another name).

    ``eval_mask`` (P3): restrict SCORING to a subset of eval rows without
    disturbing the walk-forward geometry. A boolean ``pd.Series`` indexed on
    ``frame.index`` (True = score this row), OR a callable ``frame -> Series``,
    OR ``None`` (default = score everything, so daily P1 is a no-op). The mask
    is applied ONLY at metric-compute time — every model still ``predict``s the
    FULL eval window, so baselines are not retrained on filtered data and
    metrics stay apples-to-apples. P3 uses it to score within-session bars only
    (drop the overnight-gap rows) via ``data.intraday.overnight_mask``.

    ``horizon`` (P3): bars ahead the target spans. ``horizon=1`` (default) is
    the one-step daily no-op. ``horizon>1`` scores the FORWARD h-bar return
    ``log(px[t+h]/px[t])``. To keep the walk-forward leak-free at h>1:
      * the last ``purge_last_train_rows`` rows (default ``h-1``) are sliced off
        each split's TRAIN frame before ``.fit`` — their forward target would
        reach past train_end into eval;
      * eval rows within ``h-1`` of the split's eval_end are dropped from
        scoring — their forward target reaches past the split window;
      * ``horizon > eval_size`` raises (a split with no clean eval is exactly
        the leak we purge against — fail loud, never silent).

    ``target_fn`` (P5a): compute the scored target from the frame instead of the
    default price log-return. ``None`` (default) → ``_target_returns`` (the
    price target, unchanged). Pass e.g. the realized-range target
    ``log(high/low)`` to score a multi-target (range) forecast on the SAME
    leak-free walk-forward machinery. The returned Series is indexed on
    ``frame.index``; NaNs (e.g. horizon edges) are dropped per split like the
    default path. ``horizon`` still governs purge/eval-cap geometry.

    Returns ``{model_name: Report}``. If a factory named ``rw_name`` is
    included, MAE-skill-vs-RW is computed for every other model.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    purge = purge_last_train_rows if purge_last_train_rows is not None else max(horizon - 1, 0)

    frame = frame.sort_index()
    returns = target_fn(frame) if target_fn is not None else _target_returns(
        frame, target_col, horizon
    )

    mask_series: pd.Series | None = None
    if eval_mask is not None:
        resolved = eval_mask(frame) if callable(eval_mask) else eval_mask
        mask_series = resolved.astype(bool)

    per_split_records: dict[str, list[dict]] = {name: [] for name in model_factories}

    splits = list(splits)
    for split in splits:
        train_frame, eval_frame = slice_train_eval(frame, split)
        if eval_frame.empty:
            continue
        # Rail 2: a horizon wider than the eval window can't be scored cleanly.
        if horizon > len(eval_frame):
            raise RuntimeError(
                f"horizon={horizon} exceeds eval window size {len(eval_frame)} — "
                f"no leak-free eval rows; widen eval_size or shrink horizon"
            )
        # Purge the last `purge` train rows (their forward target reaches eval).
        if purge > 0 and len(train_frame) > purge:
            train_frame = train_frame.iloc[:-purge]

        y_true = returns.loc[eval_frame.index]
        # Skip rows where the target is NaN (first row for h=1; last h rows for
        # forward-h at the very end of the frame).
        y_true = y_true.dropna()
        # Rail 1: drop eval rows within h-1 of eval_end — their forward target
        # spans past this split's window into the next split's data.
        cut = max(horizon - 1, 0)
        if cut > 0:
            y_true = y_true.iloc[:-cut] if len(y_true) > cut else y_true.iloc[:0]
        if y_true.empty:
            continue
        y_true_idx = y_true.index

        # Score-time mask (positional over y_true_idx). Models still predict
        # the FULL eval window below; we only drop masked rows from metrics.
        if mask_series is not None:
            keep = mask_series.reindex(y_true_idx).fillna(False).to_numpy(dtype=bool)
            if not keep.any():
                continue
        else:
            keep = np.ones(len(y_true_idx), dtype=bool)

        for name, factory in model_factories.items():
            model = factory()
            model.fit(train_frame)
            fc: ProbForecast = model.predict(y_true_idx)
            if len(fc.mean) != len(y_true_idx):
                raise RuntimeError(
                    f"{name} produced {len(fc.mean)} predictions for "
                    f"{len(y_true_idx)} eval rows"
                )
            yt = np.asarray(y_true.to_list())[keep]
            mean = np.asarray(fc.mean.to_list())[keep]
            sigma = np.asarray(fc.sigma.to_list())[keep]
            p10 = np.asarray(fc.quantile(0.10).to_list())[keep]
            p50 = np.asarray(fc.quantile(0.50).to_list())[keep]
            p90 = np.asarray(fc.quantile(0.90).to_list())[keep]
            per_split_records[name].append(
                {
                    "eval_start": split.eval_start.isoformat(),
                    "eval_end": split.eval_end.isoformat(),
                    "y_true": yt.tolist(),
                    "mean": mean.tolist(),
                    "sigma": sigma.tolist(),
                    "p10": p10.tolist(),
                    "p50": p50.tolist(),
                    "p90": p90.tolist(),
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

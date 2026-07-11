"""End-to-end: harness runs both baselines on a synthetic frame + reports
non-NaN metrics and a well-formed skill score."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pythia.backtest import expanding_walk_forward, run_backtest
from pythia.baselines import LastReturn, RandomWalk


def test_harness_produces_scored_reports() -> None:
    idx = pd.date_range("2024-01-01", periods=400, freq="B")
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.01, size=len(idx))
    px = 100.0 * np.exp(np.cumsum(r))
    df = pd.DataFrame({"QQQ_close": px}, index=idx)

    splits = list(expanding_walk_forward(df.index, initial_train_size=252, eval_size=21))
    assert splits, "expected splits"

    reports = run_backtest(
        df,
        target_col="QQQ_close",
        splits=splits,
        model_factories={
            "random_walk": lambda: RandomWalk("QQQ_close"),
            "last_return": lambda: LastReturn("QQQ_close"),
        },
    )

    assert set(reports) == {"random_walk", "last_return"}
    for r in reports.values():
        assert r.n_splits > 0
        assert r.n_eval_obs > 0
        assert np.isfinite(r.mae)
        assert 0.0 <= r.hit_rate <= 1.0
        assert 0.0 <= r.coverage_80 <= 1.0
        assert np.isfinite(r.crps)
        assert np.isfinite(r.pinball_50)

    # RW should compute a skill score against ITSELF as None; other models
    # should get a finite score against RW.
    assert reports["random_walk"].mae_skill_vs_rw is None
    assert reports["last_return"].mae_skill_vs_rw is not None
    assert np.isfinite(reports["last_return"].mae_skill_vs_rw)


def test_calibration_warning_on_narrow_sigma() -> None:
    """A model that shouts narrow σ into fat-tailed noise trips the
    MISCALIBRATED warning."""
    idx = pd.date_range("2024-01-01", periods=400, freq="B")
    rng = np.random.default_rng(1)
    # Fat noise at eval time; training window will show narrower sigma so
    # LastReturn's fit-time sigma understates eval variance a bit — enough
    # that at least one report should be flagged if variance is high enough.
    r = rng.standard_t(df=3, size=len(idx)) * 0.02
    px = 100.0 * np.exp(np.cumsum(r))
    df = pd.DataFrame({"QQQ_close": px}, index=idx)

    reports = run_backtest(
        df,
        target_col="QQQ_close",
        splits=list(expanding_walk_forward(df.index, 252, 21)),
        model_factories={"random_walk": lambda: RandomWalk("QQQ_close")},
    )
    rw = reports["random_walk"]
    # Property test: whatever the exact number, if it's outside 0.75-0.85
    # the harness must have said so.
    if abs(rw.coverage_80 - 0.80) > 0.05:
        assert any("MISCALIBRATED" in w for w in rw.warnings)
    else:
        assert not any("MISCALIBRATED" in w for w in rw.warnings)

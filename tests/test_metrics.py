"""Property + known-value tests for every metric."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pythia.metrics import (
    coverage,
    crps_normal,
    directional_hit_rate,
    mae,
    mae_skill_vs,
    pinball_loss,
    rmse,
)


# ---- point ----


def test_mae_zero_on_perfect_forecast() -> None:
    assert mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_mae_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        mae([1.0, 2.0], [1.0])


def test_mae_skips_non_finite() -> None:
    assert mae([1.0, np.nan, 3.0], [1.0, 5.0, 3.0]) == 0.0


def test_rmse_matches_manual() -> None:
    y, yp = np.array([0.0, 0.0, 0.0]), np.array([1.0, -1.0, 2.0])
    assert rmse(y, yp) == pytest.approx(math.sqrt((1 + 1 + 4) / 3))


def test_mae_skill_vs_semantics() -> None:
    assert mae_skill_vs(0.0, 1.0) == pytest.approx(1.0)
    assert mae_skill_vs(1.0, 1.0) == 0.0
    assert mae_skill_vs(2.0, 1.0) == pytest.approx(-1.0)
    assert math.isnan(mae_skill_vs(1.0, 0.0))


# ---- directional ----


def test_directional_hit_rate_all_right() -> None:
    assert directional_hit_rate([1, -1, 1, -1], [2, -2, 3, -3]) == 1.0


def test_directional_hit_rate_all_wrong() -> None:
    assert directional_hit_rate([1, -1, 1, -1], [-2, 2, -3, 3]) == 0.0


def test_directional_ignores_double_zero() -> None:
    # (0,0) doesn't count either way.
    assert directional_hit_rate([0, 1, -1], [0, 1, -1]) == 1.0


# ---- coverage ----


def test_coverage_perfect_interval() -> None:
    y = np.array([0.0, 0.5, -0.5])
    lo = np.array([-1.0, -1.0, -1.0])
    hi = np.array([1.0, 1.0, 1.0])
    assert coverage(y, lo, hi) == 1.0


def test_coverage_no_interval_contains_nothing() -> None:
    y = np.array([10.0, -10.0])
    lo = np.array([-1.0, -1.0])
    hi = np.array([1.0, 1.0])
    assert coverage(y, lo, hi) == 0.0


def test_coverage_swapped_bounds_dont_count() -> None:
    y = np.array([0.0])
    lo = np.array([1.0])
    hi = np.array([-1.0])
    # Only invalid entry → mask is empty → NaN.
    assert math.isnan(coverage(y, lo, hi))


# ---- CRPS ----


def test_crps_zero_when_sigma_tiny_and_correct() -> None:
    y = np.array([0.0])
    mu = np.array([0.0])
    sd = np.array([1e-6])
    v = crps_normal(y, mu, sd)
    assert v >= 0
    assert v < 1e-4


def test_crps_larger_when_bias() -> None:
    y = np.zeros(1000)
    biased = crps_normal(y, np.full(1000, 1.0), np.ones(1000))
    unbiased = crps_normal(y, np.zeros(1000), np.ones(1000))
    assert biased > unbiased


def test_crps_sigma_must_be_positive() -> None:
    y = np.zeros(3)
    mu = np.zeros(3)
    sd = np.array([1.0, -1.0, 0.0])
    # -1.0 and 0.0 rows are masked out; only the first counts.
    v = crps_normal(y, mu, sd)
    assert 0 < v < 1


# ---- pinball ----


def test_pinball_symmetric_at_median() -> None:
    y = np.array([1.0, -1.0])
    qh = np.array([0.0, 0.0])
    v = pinball_loss(y, qh, 0.5)
    # 0.5*1 + 0.5*1 average = 0.5
    assert v == pytest.approx(0.5)


def test_pinball_asymmetric_at_p10() -> None:
    # y=1, forecast p10=0: over-predicting the tail — small penalty (0.1 side).
    # y=-1, forecast p10=0: under-predicting — big penalty (0.9 side).
    y = np.array([1.0, -1.0])
    qh = np.array([0.0, 0.0])
    v = pinball_loss(y, qh, 0.1)
    assert v == pytest.approx((0.1 * 1 + 0.9 * 1) / 2)


def test_pinball_q_bounds_enforced() -> None:
    with pytest.raises(ValueError):
        pinball_loss([1.0], [0.5], 0.0)
    with pytest.raises(ValueError):
        pinball_loss([1.0], [0.5], 1.0)

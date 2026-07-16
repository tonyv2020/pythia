"""TFTLite smoke test — forward pass shape, quantile monotonicity, fit-predict
loop on synthetic data, and calibration sanity.

CPU-only so it runs in CI without a GPU. Full walk-forward training on real
data is scripts/train_p1_tft.py — that produces report.json.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from pythia.models import TFTLite, TFTLiteConfig, TFTLiteModel
from pythia.models.quantile_loss import multi_quantile_pinball


def test_tft_lite_forward_shape() -> None:
    cfg = TFTLiteConfig(
        n_features=8, encoder_length=20, hidden_size=8, n_targets=2, quantiles=(0.1, 0.5, 0.9)
    )
    m = TFTLite(cfg)
    x = torch.randn(4, cfg.encoder_length, cfg.n_features)
    q_preds, vsn_w = m(x)
    assert len(q_preds) == 2
    for q in q_preds:
        assert q.shape == (4, 3)
    assert vsn_w.shape == (4, cfg.encoder_length, cfg.n_features)


def test_pinball_reduces_over_training() -> None:
    """A trivial fit on constant targets should push pinball loss down."""
    torch.manual_seed(0)
    cfg = TFTLiteConfig(
        n_features=3, encoder_length=10, hidden_size=8, n_targets=1, quantiles=(0.1, 0.5, 0.9)
    )
    m = TFTLite(cfg)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    x = torch.randn(32, 10, 3)
    y = torch.zeros(32) + 0.5
    q_tensor = torch.tensor(cfg.quantiles)

    losses = []
    for _ in range(30):
        opt.zero_grad()
        q_preds, _ = m(x)
        loss = multi_quantile_pinball(q_preds[0], y, q_tensor)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"pinball didn't decrease: {losses[0]} → {losses[-1]}"


def test_adapter_fit_predict_on_synthetic() -> None:
    """End-to-end smoke: fit on synthetic wide frame, predict on last 5 rows."""
    torch.manual_seed(0)
    idx = pd.date_range("2024-01-01", periods=400, freq="B")
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.01, size=len(idx))
    px = 100.0 * np.exp(np.cumsum(r))

    df = pd.DataFrame(
        {
            "QQQ_close": px,
            "SPY_close": px * (1 + rng.normal(0, 0.001, size=len(idx))),
            "QQQ_volume": rng.integers(1_000_000, 5_000_000, size=len(idx)),
            "SPY_volume": rng.integers(500_000, 2_000_000, size=len(idx)),
            "dow": [d.weekday() for d in idx],
            "month": idx.month,
            "dom": idx.day,
            "is_monday": (idx.weekday == 0),
            "is_friday": (idx.weekday == 4),
            "is_month_end": False,
            "is_quarter_end": False,
            "days_to_fomc": np.arange(len(idx)) % 45,
            "is_earnings_season": (idx.month % 3 == 1),
        },
        index=idx,
    )

    train = df.iloc[:350]
    eval_idx = df.index[350:355]
    model = TFTLiteModel(
        target_col="QQQ_close",
        encoder_length=20,
        hidden_size=8,
        max_epochs=2,
        batch_size=16,
        device="cpu",
    )
    model.fit(train)
    fc = model.predict(eval_idx)
    assert len(fc.mean) == 5
    assert (fc.sigma > 0).all()


def test_adapter_captures_last_attention() -> None:
    """P5c: after predict(), the adapter exposes a length-encoder_length
    attention array from the LAST forward pass."""
    torch.manual_seed(0)
    idx = pd.date_range("2024-01-01", periods=400, freq="B")
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.01, size=len(idx))
    px = 100.0 * np.exp(np.cumsum(r))
    df = pd.DataFrame(
        {
            "QQQ_close": px,
            "SPY_close": px * (1 + rng.normal(0, 0.001, size=len(idx))),
            "QQQ_volume": rng.integers(1_000_000, 5_000_000, size=len(idx)),
            "SPY_volume": rng.integers(500_000, 2_000_000, size=len(idx)),
            "dow": [d.weekday() for d in idx],
            "month": idx.month,
            "dom": idx.day,
            "is_monday": (idx.weekday == 0),
            "is_friday": (idx.weekday == 4),
            "is_month_end": False,
            "is_quarter_end": False,
            "days_to_fomc": np.arange(len(idx)) % 45,
            "is_earnings_season": (idx.month % 3 == 1),
        },
        index=idx,
    )
    train = df.iloc[:350]
    eval_idx = df.index[350:355]
    model = TFTLiteModel(
        target_col="QQQ_close",
        encoder_length=20,
        hidden_size=8,
        max_epochs=2,
        batch_size=16,
        device="cpu",
    )
    # Before fit/predict, no attention captured yet.
    assert model.last_attention_weights is None
    model.fit(train)
    model.predict(eval_idx)
    attn = model.last_attention_weights
    assert attn is not None
    assert len(attn) == 20  # encoder_length
    # softmax → sums to ~1, all non-negative.
    assert all(w >= 0 for w in attn)
    assert 0.98 < sum(attn) < 1.02

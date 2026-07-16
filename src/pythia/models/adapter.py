"""Model adapter — plug ``TFTLite`` into the P0 backtest harness.

The P0 harness (``pythia.backtest.harness.run_backtest``) expects an object
that implements the ``pythia.backtest.protocols.Model`` protocol: ``fit(train)``
and ``predict(eval_index)``. TFTLite is a PyTorch module; this adapter is
the glue.

At fit time we:
  1. Build features via ``pythia.features.build_features`` with the lag
     policy (structural leakage prevention — proven by the P1 phase-1 test).
  2. Compute the two targets (return, realized-range) for training rows.
  3. Standardise features (train-only fit; NO look-ahead into eval).
  4. Train TFTLite with multi-quantile pinball loss + crossing penalty.

At predict time we:
  1. Take the last ``encoder_length`` rows STRICTLY before each eval date.
  2. Standardise using train stats (never eval stats — that would leak).
  3. Forward-pass TFTLite → per-eval-date quantile vectors.
  4. Convert the P10/P50/P90 into ``ProbForecast(mean=P50, sigma=(P90-P10)/2.563)``.

The 2.563 factor comes from Normal quantile scaling — under Normal(0,1),
P90 − P10 = 2 · Φ⁻¹(0.90) ≈ 2.5631. This is a Normal-approximation for
plumbing compatibility with the P0 harness; the underlying model outputs
the actual quantiles independently (i.e. it doesn't assume Normal).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..backtest.protocols import ProbForecast
from ..features.lag import LagPolicy, build_features
from ..features.targets import realized_range_target, return_target
from .dataset import PythiaWindowDataset
from .quantile_loss import multi_quantile_pinball
from .tft_lite import TFTLite, TFTLiteConfig


@dataclass
class TFTLiteModel:
    """Model-protocol adapter for TFTLite.

    ``target_col``: column name for the price series driving the return
    target. The realised-range target is derived from the SAME symbol's
    high/low if provided in the frame (columns ``{sym}_high`` /
    ``{sym}_low``); otherwise we approximate range as |return| — degraded
    but not fatal (harness logs a warning).
    """

    target_col: str = "QQQ_close"
    lag: int = 1
    encoder_length: int = 60
    hidden_size: int = 16
    lstm_layers: int = 1
    dropout: float = 0.1
    max_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    quantiles: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1337
    # Which trained head predict() serves: 0 = return (default, P1), 1 = range
    # (P5a multi-target — the model already trains both; this exposes the range
    # head as a ProbForecast). No-op for the price model.
    target_head: int = 0

    # Fitted state (populated by fit()):
    _model: TFTLite | None = field(default=None, init=False, repr=False)
    _feature_names: list[str] | None = field(default=None, init=False, repr=False)
    _feat_mean: np.ndarray | None = field(default=None, init=False, repr=False)
    _feat_std: np.ndarray | None = field(default=None, init=False, repr=False)
    _train_frame: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _lag_policy: LagPolicy | None = field(default=None, init=False, repr=False)

    # P5c attention snapshot: length-``encoder_length`` list of temporal
    # attention weights captured on the LAST predict() call. Panel renders
    # this as the "which past bars did the model weight" strip. None if the
    # model has not been used to predict yet, or if predict() ran on an
    # empty eval index.
    _last_attention: list[float] | None = field(default=None, init=False, repr=False)

    @property
    def last_attention_weights(self) -> list[float] | None:
        """Return the last predict()'s attention weights over the encoder window.

        Length = encoder_length; sums to ~1 (softmax over past bars). None if
        no forward pass has run yet.
        """
        return self._last_attention

    def _build_targets(self, frame: pd.DataFrame) -> pd.DataFrame:
        px = frame[self.target_col]
        r = return_target(px).rename("y_return")
        sym = self.target_col.split("_close")[0]
        hi_col, lo_col = f"{sym}_high", f"{sym}_low"
        if hi_col in frame.columns and lo_col in frame.columns:
            rng = realized_range_target(frame[hi_col], frame[lo_col]).rename("y_range")
        else:
            rng = r.abs().rename("y_range")
        return pd.concat([r, rng], axis=1)

    def fit(self, train: pd.DataFrame) -> None:
        """Train the TFT-lite network on the assembled frame (multi-target: return + realized-range heads)."""
        torch.manual_seed(self.seed)
        # Build lag policy from train's columns; targets are the raw price
        # (and high/low if present) — they don't enter the feature matrix.
        target_cols = {self.target_col}
        sym = self.target_col.split("_close")[0]
        if f"{sym}_high" in train.columns:
            target_cols.add(f"{sym}_high")
        if f"{sym}_low" in train.columns:
            target_cols.add(f"{sym}_low")
        from ..features.lag import default_policy_for

        # Drop columns that are entirely NaN in the train window before we
        # build the policy — otherwise ffill has nothing to fill and the join
        # kills every row. Common on intraday P3 where the raptor tick feed
        # only intermittently covers non-QQQ board symbols.
        all_nan = [c for c in train.columns if train[c].isna().all()]
        if all_nan:
            train = train.drop(columns=all_nan)
        policy = default_policy_for(train.columns, target_cols=target_cols)
        # Forward-fill observed covariates BEFORE lag to survive raptor's
        # intermittent ingestion. ffill uses only past values (no look-ahead)
        # so the covariate-lag gate is preserved. Rows still NaN after ffill
        # (leading rows before any symbol reported) are dropped by the join.
        train_filled = train.copy()
        train_filled[sorted(policy.observed)] = train_filled[sorted(policy.observed)].ffill()
        feat = build_features(train_filled, policy, lag=self.lag)

        targets = self._build_targets(train)

        # Fit-time-only standardisation. Compute stats on FEAT ONLY, never
        # on rows we don't have targets for.
        joined = feat.join(targets, how="inner").dropna()
        if joined.empty:
            raise RuntimeError("no training samples after lag + target alignment")
        feat_used = joined[feat.columns].astype(np.float32)
        mu = feat_used.mean(axis=0).values
        sd = feat_used.std(axis=0).replace(0, 1.0).values
        feat_norm = (feat_used.values - mu) / sd
        feat_norm_df = pd.DataFrame(feat_norm, index=joined.index, columns=feat.columns)

        target_df = joined[["y_return", "y_range"]].astype(np.float32)

        ds = PythiaWindowDataset(feat_norm_df, target_df, encoder_length=self.encoder_length)
        if len(ds) == 0:
            raise RuntimeError(
                f"no valid anchors for encoder_length={self.encoder_length} "
                f"on {len(joined)} training rows"
            )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        cfg = TFTLiteConfig(
            n_features=len(feat.columns),
            encoder_length=self.encoder_length,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            dropout=self.dropout,
            n_targets=2,
            quantiles=self.quantiles,
        )
        model = TFTLite(cfg).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        q_tensor = torch.tensor(self.quantiles, dtype=torch.float32, device=self.device)

        model.train()
        for _ in range(self.max_epochs):
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                opt.zero_grad()
                q_preds, _ = model(x)
                # Two heads → two pinball losses, averaged.
                # crossing_penalty=0 for the report run — the pinball loss
                # itself already discourages crossing, and a large penalty
                # collapses the quantile spread (miscalibration failure mode).
                l_ret = multi_quantile_pinball(q_preds[0], y[:, 0], q_tensor, crossing_penalty=0.0)
                l_rng = multi_quantile_pinball(q_preds[1], y[:, 1], q_tensor, crossing_penalty=0.0)
                loss = 0.5 * (l_ret + l_rng)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        self._model = model
        self._feature_names = list(feat.columns)
        self._feat_mean = mu
        self._feat_std = sd
        self._train_frame = train
        self._lag_policy = policy

    def predict(self, eval_index: pd.Index) -> ProbForecast:
        """Return the (mean, sigma) forecast for ``eval_index`` from the selected target head."""
        assert self._model is not None, "TFTLiteModel not fit"
        assert self._train_frame is not None
        assert self._lag_policy is not None
        model = self._model
        model.eval()

        # Concat train + eval-anchor placeholder so we can build the lagged
        # feature window for each eval date without look-ahead: the LAGGED
        # value for eval date t is the value at t-1, which lives entirely in
        # train history. So we build features on the train frame and slice.
        train = self._train_frame
        # Same ffill as in fit — past-only imputation, structurally leak-free.
        train_filled = train.copy()
        train_filled[sorted(self._lag_policy.observed)] = train_filled[
            sorted(self._lag_policy.observed)
        ].ffill()
        feat = build_features(train_filled, self._lag_policy, lag=self.lag)
        feat_norm = (feat[self._feature_names].values - self._feat_mean) / self._feat_std
        feat_norm_df = pd.DataFrame(feat_norm, index=feat.index, columns=self._feature_names)

        # For each eval date t, we need the trailing encoder_length rows of
        # feat up to and including date t-1 (or the last train row if t is
        # the first eval date).
        preds = []
        q_idx_50 = self.quantiles.index(0.50)
        q_idx_10 = self.quantiles.index(0.10)
        q_idx_90 = self.quantiles.index(0.90)

        for t in eval_index:
            # last train date < t: closest anchor.
            anchor_pos = int(np.searchsorted(feat_norm_df.index, t, side="left")) - 1
            if anchor_pos < self.encoder_length - 1:
                # Fall back to a flat "unknown" prediction: median 0, wide σ.
                preds.append((0.0, -1.0, 1.0))
                continue
            window = feat_norm_df.values[anchor_pos - self.encoder_length + 1 : anchor_pos + 1]
            x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_preds, _ = model(x)
            q_ret = q_preds[self.target_head].squeeze(0).detach().cpu().numpy()
            preds.append((float(q_ret[q_idx_50]), float(q_ret[q_idx_10]), float(q_ret[q_idx_90])))
        # P5c: after the loop, stash the LAST forecast's temporal attention
        # (length encoder_length, sums to ~1). Only meaningful if at least one
        # real (non-fallback) forecast ran; otherwise leave None.
        last_attn = getattr(model, "_last_attn_w", None)
        if last_attn is not None and last_attn.ndim >= 2 and last_attn.size(0) >= 1:
            # Take the last batch item (== the last-forecast anchor because
            # we run predict one-at-a-time in a for loop).
            self._last_attention = last_attn[-1].cpu().numpy().astype(float).tolist()

        # ProbForecast wants a Normal(mean, sigma) surrogate — collapse
        # P10/P90 to sigma via Normal quantile scaling (2*inv_norm_cdf(0.9)).
        Z90_MINUS_Z10 = 2.5631031310892007
        means = np.array([p[0] for p in preds])
        sigmas = np.array([max((p[2] - p[1]) / Z90_MINUS_Z10, 1e-6) for p in preds])
        return ProbForecast(
            mean=pd.Series(means, index=eval_index),
            sigma=pd.Series(sigmas, index=eval_index),
        )

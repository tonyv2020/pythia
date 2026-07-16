"""IntradayTFTLiteModel must build a FORWARD-h target (matches the harness)."""

from __future__ import annotations
import numpy as np
import pandas as pd
from pythia.models.intraday_tft import IntradayTFTLiteModel


def test_intraday_tft_target_is_forward_h():
    idx = pd.date_range("2026-06-08 09:30", periods=12, freq="10min")
    px = pd.Series(np.linspace(400, 411, 12), index=idx)
    df = pd.DataFrame({"QQQ_close": px})
    m = IntradayTFTLiteModel(target_col="QQQ_close", horizon=3)
    tg = m._build_targets(df)
    # y_return[t] == log(px[t+3]) - log(px[t]); last 3 rows NaN.
    assert tg["y_return"].iloc[0] == np.log(px.iloc[3]) - np.log(px.iloc[0])
    assert tg["y_return"].iloc[5] == np.log(px.iloc[8]) - np.log(px.iloc[5])
    assert tg["y_return"].iloc[-3:].isna().all()
    assert (tg["y_range"].dropna() >= 0).all()

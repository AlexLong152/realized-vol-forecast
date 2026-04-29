"""Sanity tests for realized volatility estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from rvforecast.features.realized_vol import (
    close_to_close_vol,
    garman_klass_vol,
    parkinson_vol,
    rogers_satchell_vol,
)


def _flat(n: int = 60, price: float = 100.0) -> dict[str, pd.Series]:
    idx = pd.bdate_range("2020-01-01", periods=n)
    o = pd.Series(price, index=idx)
    return {"open": o, "high": o.copy(), "low": o.copy(), "close": o.copy()}


def test_constant_prices_zero_vol():
    prices = _flat()
    log_ret = np.log(prices["close"] / prices["close"].shift(1)).fillna(0.0)
    assert close_to_close_vol(log_ret, 22).dropna().abs().max() == pytest.approx(0.0)
    assert parkinson_vol(prices["high"], prices["low"], 22).dropna().abs().max() == pytest.approx(
        0.0
    )
    assert garman_klass_vol(
        prices["open"], prices["high"], prices["low"], prices["close"], 22
    ).dropna().abs().max() == pytest.approx(0.0)
    assert rogers_satchell_vol(
        prices["open"], prices["high"], prices["low"], prices["close"], 22
    ).dropna().abs().max() == pytest.approx(0.0)


def test_bigger_range_implies_bigger_parkinson():
    idx = pd.bdate_range("2020-01-01", periods=60)
    base = pd.Series(100.0, index=idx)
    narrow_high = base + 0.5
    narrow_low = base - 0.5
    wide_high = base + 2.0
    wide_low = base - 2.0
    narrow = parkinson_vol(narrow_high, narrow_low, 22).dropna()
    wide = parkinson_vol(wide_high, wide_low, 22).dropna()
    assert (wide > narrow).all()


def test_garman_klass_nonnegative():
    rng = np.random.default_rng(0)
    n = 200
    idx = pd.bdate_range("2020-01-01", periods=n)
    open_ = pd.Series(100 + rng.normal(0, 1, n).cumsum(), index=idx)
    intraday = pd.Series(rng.normal(0, 1, n), index=idx)
    high = pd.concat([open_, open_ + intraday.abs()], axis=1).max(axis=1) + 0.5
    low = pd.concat([open_, open_ - intraday.abs()], axis=1).min(axis=1) - 0.5
    close = open_ + intraday
    gk = garman_klass_vol(open_, high, low, close, 22).dropna()
    assert (gk >= 0).all()

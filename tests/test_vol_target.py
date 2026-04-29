"""Tests for the vol-targeting strategy summary metrics.

``rvforecast.extension.vol_target._summarize`` returns one dict packed
with several different calculations: annualized return, vol, Sharpe,
max drawdown, turnover, transaction-cost drag, and net-of-cost Sharpe.
Easy place for a refactor to flip a sign or swap Sharpe's denominator
without anyone noticing, so the arithmetic gets pinned here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from rvforecast.config import TRADING_DAYS
from rvforecast.extension.vol_target import (
    _build_position,
    _max_drawdown,
    _summarize,
)


def _series(values, dates) -> pd.Series:
    return pd.Series(values, index=pd.to_datetime(dates), dtype=float)


def test_constant_position_summary_matches_hand_calc():
    """Position = 1.0, returns = constant 0.001/day, no turnover.

    Annualized return = 0.001 * 252.
    Annualized vol    = 0 (zero-variance series).
    Sharpe             = NaN (zero-vol degenerate case).
    Turnover          = 0.
    """
    n = 50
    dates = pd.bdate_range("2020-01-01", periods=n)
    rets = pd.Series(0.001, index=dates)
    pos = pd.Series(1.0, index=dates)
    out = _summarize(rets, pos, "test")
    assert out["ann_return"] == pytest.approx(0.001 * TRADING_DAYS)
    assert out["ann_vol"] == pytest.approx(0.0)
    assert np.isnan(out["sharpe"])
    assert out["turnover"] == pytest.approx(0.0)
    assert out["tc_drag_1bp"] == pytest.approx(0.0)


def test_summary_sharpe_sign_and_scale():
    """Positive mean returns yield positive Sharpe; magnitude tracks ann/vol."""
    rng = np.random.default_rng(0)
    n = 252 * 5
    dates = pd.bdate_range("2020-01-01", periods=n)
    rets = pd.Series(rng.normal(0.0005, 0.01, size=n), index=dates)
    pos = pd.Series(1.0, index=dates)
    out = _summarize(rets, pos, "test")
    expected_sharpe = (rets.mean() * TRADING_DAYS) / (rets.std() * np.sqrt(TRADING_DAYS))
    assert out["sharpe"] == pytest.approx(expected_sharpe)
    assert out["sharpe"] > 0


def test_turnover_and_cost_drag():
    """Turnover is the sum of |Δposition|; 1bp cost is 0.0001 per unit turn."""
    dates = pd.bdate_range("2020-01-01", periods=4)
    pos = _series([1.0, 0.5, 0.5, 1.5], dates)
    rets = _series([0.01, 0.01, 0.01, 0.01], dates)
    out = _summarize(rets, pos, "test")
    # diff().abs() = [NaN -> 0, 0.5, 0.0, 1.0] -> sum = 1.5
    assert out["turnover"] == pytest.approx(1.5)
    assert out["tc_drag_1bp"] == pytest.approx(1.5 * 0.0001)
    assert out["tc_drag_5bp"] == pytest.approx(1.5 * 0.0005)


def test_max_drawdown_known_path():
    """A monotone-rise then 50% drop has drawdown = -0.5 on the price path."""
    # log returns that yield price path 1 -> 2 -> 4 -> 2
    log_rets = pd.Series(
        [np.log(2), np.log(2), np.log(0.5)],
        index=pd.bdate_range("2020-01-01", periods=3),
    )
    dd = _max_drawdown(log_rets)
    assert dd == pytest.approx(-0.5)


def test_max_drawdown_empty_returns_nan():
    assert np.isnan(_max_drawdown(pd.Series(dtype=float)))


def test_max_drawdown_monotone_rise_is_zero():
    log_rets = pd.Series(
        [0.01, 0.01, 0.01, 0.01],
        index=pd.bdate_range("2020-01-01", periods=4),
    )
    assert _max_drawdown(log_rets) == pytest.approx(0.0)


def test_build_position_clips_to_leverage_cap():
    """Position = clip(target / forecast, 0, leverage_cap).

    If the forecast vol is much smaller than the target, the unclipped
    ratio blows up — the cap is the only thing keeping the position
    sane. That's the case the test exercises.
    """
    from rvforecast.config import LEVERAGE_CAP, VOL_TARGET_ANNUAL

    dates = pd.bdate_range("2020-01-01", periods=3)
    # Tiny forecast vol → unclipped target/forecast >> LEVERAGE_CAP.
    forecast = _series([1e-6, 0.10, 0.30], dates)
    pos = _build_position(forecast)
    assert pos.iloc[0] == pytest.approx(LEVERAGE_CAP)
    assert pos.iloc[1] == pytest.approx(VOL_TARGET_ANNUAL / 0.10)
    assert pos.iloc[2] == pytest.approx(VOL_TARGET_ANNUAL / 0.30)
    assert (pos >= 0).all()
    assert (pos <= LEVERAGE_CAP).all()


def test_summary_empty_returns_all_nans():
    """An empty strategy series must propagate NaN, not raise.

    Hit when a model has no predictions overlapping the SPY return calendar.
    """
    out = _summarize(pd.Series(dtype=float), pd.Series(dtype=float), "empty")
    for key in (
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_drawdown",
        "turnover",
        "sharpe_net_1bp",
        "sharpe_net_5bp",
    ):
        assert np.isnan(out[key]), f"expected NaN for {key}, got {out[key]}"

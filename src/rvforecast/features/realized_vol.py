"""Realized volatility estimators.

Each function returns an annualized volatility (multiplied by ``sqrt(252)``)
expressed in the same units as the input return. The output is the sample
standard deviation under the corresponding estimator, not the variance.

Underlying model
----------------
All estimators assume the log-price follows Brownian motion with drift,

    d log p_t = mu dt + sigma dW_t,

and invert an exact identity for some functional of one day's path. With
``T`` = one trading day and ``(O, H, L, C)`` the open/high/low/close of
``log p`` over the day:

    E[(C - O)^2]                            = sigma^2 T           (driftless)
    E[(H - L)^2]                            = 4 log(2) sigma^2 T  (driftless)
    E[0.5 (H-L)^2 - (2 log 2 - 1)(C-O)^2]   = sigma^2 T           (driftless)
    E[(H-O)(H-C) + (L-O)(L-C)]              = sigma^2 T           (any drift)

Each estimator below is the right-hand side evaluated on one day's OHLC,
treated as a one-sample estimate of ``sigma^2 T``; annualization multiplies
by ``sqrt(252)`` to convert per-day to per-year. Estimators that exploit
the intraday range (Parkinson, Garman-Klass, Rogers-Satchell) have lower
variance than close-to-close because the daily H/L carries more information
about sigma than the close-to-close increment alone.

Estimators
----------
- :func:`close_to_close_vol`: rolling standard deviation of log returns.
- :func:`parkinson_vol`: range-based estimator using high and low only;
  efficient under zero drift.
- :func:`garman_klass_vol`: range-based estimator using OHLC; lower variance
  than Parkinson under zero drift.
- :func:`rogers_satchell_vol`: range-based estimator that is unbiased in the
  presence of nonzero drift.

References
----------
Garman, M. B., & Klass, M. J. (1980). Rogers, L. C. G., & Satchell, S. E.
(1991). Parkinson, M. (1980).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rvforecast.config import TRADING_DAYS

# Floor applied to one-day Garman-Klass realized variance before logging.
# A vol of 1e-8 corresponds to log(1e-8) ~= -18.4, well outside any realistic
# observed value, so capping at this level keeps perfectly-flat days alive
# in the panel (where they would otherwise become NaN under log(0)).
_RV_FLOOR: float = 1e-8


def _annualize(daily: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return daily * np.sqrt(TRADING_DAYS)


def close_to_close_vol(returns: pd.Series, window: int) -> pd.Series:
    """Annualized rolling close-to-close volatility from log returns.

    Inverts ``E[(log p_{t+1} - log p_t)^2] = sigma^2 dt``: a rolling sample
    standard deviation of log returns estimates ``sigma`` per ``sqrt(dt)``,
    which is then annualized.
    """
    daily = returns.rolling(window).std()
    return _annualize(daily).rename(f"cc_vol_{window}d")


def parkinson_vol(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """Annualized Parkinson volatility from log(high/low).

    Inverts the Brownian range-squared identity
    ``E[(log H - log L)^2] = 4 log(2) sigma^2 T`` (Parkinson 1980, driftless).
    Roughly 5x more efficient than close-to-close: the daily range sees the
    largest excursion of the path, not just its endpoints.
    """
    log_hl = np.log(high / low)
    daily_var = (1.0 / (4.0 * np.log(2.0))) * (log_hl**2).rolling(window).mean()
    return _annualize(np.sqrt(daily_var)).rename(f"parkinson_vol_{window}d")


def garman_klass_vol(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    """Annualized Garman-Klass volatility from OHLC.

    Minimum-variance unbiased linear combination of ``(H-L)^2`` and
    ``(C-O)^2`` under driftless Brownian motion. The coefficients ``0.5``
    and ``-(2 log 2 - 1) ~= -0.386`` are chosen so the expectation equals
    ``sigma^2 T`` with the smallest variance among such combinations
    (Garman & Klass 1980, eq. 16). Roughly 7-8x more efficient than
    close-to-close; the project's preferred estimator.
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    # Garman & Klass (1980), eq. 16.
    daily_var_components = 0.5 * log_hl**2 - (2.0 * np.log(2.0) - 1.0) * log_co**2
    daily_var = daily_var_components.rolling(window).mean()
    return _annualize(np.sqrt(daily_var.clip(lower=0.0))).rename(f"gk_vol_{window}d")


def rogers_satchell_vol(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    """Annualized Rogers-Satchell volatility (drift-independent).

    Uses the identity ``E[(H-O)(H-C) + (L-O)(L-C)] = sigma^2 T`` for
    Brownian motion with *arbitrary* drift mu: the drift contributions to
    the two cross-products cancel (Rogers & Satchell 1991). Slightly less
    efficient than Garman-Klass at zero drift, but unbiased when drift is
    non-negligible -- preferred over longer horizons where ``mu * T`` is
    no longer small.
    """
    log_ho = np.log(high / open_)
    log_hc = np.log(high / close)
    log_lo = np.log(low / open_)
    log_lc = np.log(low / close)
    daily_var = (log_ho * log_hc + log_lo * log_lc).rolling(window).mean()
    return _annualize(np.sqrt(daily_var.clip(lower=0.0))).rename(f"rs_vol_{window}d")


def garman_klass_one_day(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Annualized one-day Garman-Klass estimate (no rolling window).

    This is the modeling target. Each row is a single-day annualized vol; over
    long stretches it is noisy day-to-day, hence the value of forecasting it.
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    # Garman & Klass (1980), eq. 16.
    daily_var = 0.5 * log_hl**2 - (2.0 * np.log(2.0) - 1.0) * log_co**2
    daily_var = daily_var.clip(lower=0.0)
    return _annualize(np.sqrt(daily_var)).rename("gk_vol_1d")


def add_target_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """Add realized vol columns and the modeling target to a long panel.

    Expects columns ``open, high, low, close`` indexed by ``(date, ticker)``.
    Adds:
        - ``rv_1d``: one-day Garman-Klass annualized volatility for the row date.
        - ``log_rv_1d``: log of the above (the regression target after a shift).
        - rolling versions for HAR features computed downstream.

    ``garman_klass_one_day`` is row-wise, so no per-ticker groupby is needed.
    The target used in modeling is the *next* day's ``rv_1d`` for the same
    ticker; alignment is handled in :mod:`rvforecast.features.build_features`.
    """
    out = panel.copy()
    out["rv_1d"] = garman_klass_one_day(out["open"], out["high"], out["low"], out["close"])
    # Floor at _RV_FLOOR before logging so days with strictly-zero observed
    # range (rare but possible under flat synthetic data or stale quotes)
    # remain in the panel rather than being silently dropped via log(NaN).
    out["log_rv_1d"] = np.log(out["rv_1d"].clip(lower=_RV_FLOOR)).astype(float)
    return out

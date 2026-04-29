"""Build the feature matrix for one-day-ahead log realized volatility.

The output is a long panel indexed by ``(date, ticker)``. Every feature on
row ``t`` is a function of data with index strictly less than ``t`` for that
ticker, with cross-sectional ranks computed from same-day lagged values
(which are themselves functions of strictly past data per ticker). The
``shift(1)`` calls are aggressive on purpose; the look-ahead test in
``tests/test_features.py`` injects a future spike into a single row and
confirms it does not propagate backwards, and a same-day perturbation test
confirms the feature vector at row ``t`` is independent of the OHLC of day
``t`` itself.

The target ``y_target`` on row ``t`` is ``log_rv_1d`` for date ``t`` itself.
Combined with the strictly-past features, this gives a one-day-ahead
forecast: at the close of day ``t-1`` the model uses everything known up to
that close to predict the realized vol of day ``t``.

The contemporaneous return ``ret_1d = log(close_t / close_{t-1})`` is needed
internally to construct the lagged HAR-style return features, and is also
needed by the GARCH baseline's daily recursion. It is intentionally kept out
of the persisted feature matrix because, as a same-day measurement that uses
``close_t``, it would leak directly into models that consume the feature
matrix without re-shifting (notably LightGBM). GARCH sources returns from
the cached price panel directly; the LSTM builds its own raw panel from
prices and macro and never reads ``ret_1d`` from this matrix.
"""

from __future__ import annotations

import argparse
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from rvforecast.config import (
    DATA_PROCESSED,
    DATA_RAW,
    HAR_HORIZONS,
    ROOT,
    TRADING_DAYS,
    ensure_output_dirs,
)
from rvforecast.features.realized_vol import add_target_columns

# Internal column name for the contemporaneous (same-day) close-to-close return.
# Prefixed with an underscore to signal it is an intermediate and is dropped
# from the persisted feature matrix at the end of ``build_feature_matrix``.
_RAW_RET_COL = "_ret_1d_raw"


def _add_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach the contemporaneous log return per ticker.

    Stored under ``_RAW_RET_COL`` so it is visibly intermediate; downstream
    HAR-lag construction uses it after a ``shift(1)``, and it is dropped
    from the persisted feature matrix.
    """
    panel = panel.copy()
    panel[_RAW_RET_COL] = np.log(panel["adj_close"]).groupby(level="ticker").diff()
    return panel


def _shift_roll_mean(d: pd.DataFrame, col: str, h: int) -> pd.Series:
    return d[col].shift(1).rolling(h).mean()


def _shift_roll_abs_mean(d: pd.DataFrame, col: str, h: int) -> pd.Series:
    return d[col].abs().shift(1).rolling(h).mean()


def _per_ticker_apply(panel: pd.DataFrame, fn) -> pd.Series:
    """Apply ``fn`` per ticker, returning a Series aligned to ``panel``'s index.

    Used only for operations with temporal dependencies (``shift``,
    ``rolling``); row-wise transforms should not go through this wrapper.
    """
    return panel.groupby(level="ticker", group_keys=False).apply(fn)


def _add_har_lags(panel: pd.DataFrame, horizons: tuple[int, ...] = HAR_HORIZONS) -> pd.DataFrame:
    panel = panel.copy()
    for h in horizons:
        panel[f"log_rv_lag_{h}d"] = _per_ticker_apply(
            panel, partial(_shift_roll_mean, col="log_rv_1d", h=h)
        )
        panel[f"abs_ret_lag_{h}d"] = _per_ticker_apply(
            panel, partial(_shift_roll_abs_mean, col=_RAW_RET_COL, h=h)
        )
        panel[f"ret_lag_{h}d"] = _per_ticker_apply(
            panel, partial(_shift_roll_mean, col=_RAW_RET_COL, h=h)
        )
    return panel


def _add_range_lags(panel: pd.DataFrame, horizons: tuple[int, ...] = HAR_HORIZONS) -> pd.DataFrame:
    """Add lagged Parkinson and Garman-Klass annualized vols at each HAR horizon.

    For each horizon ``h`` and each ticker, computes per row ``t``::

        sigma_hat(t) = sqrt( 252 * mean{ daily_var(t-h), ..., daily_var(t-1) } )

    using the Parkinson and Garman-Klass daily variance estimators from each
    day's OHLC. Adds two columns per horizon: ``parkinson_vol_lag_{h}d`` and
    ``gk_vol_lag_{h}d``. The ``shift(1)`` excludes day ``t`` itself, keeping
    the features strictly causal.

    Complements ``_add_har_lags``, which works in *log-vol space*
    (mean of ``log_rv``); this function works in *vol-level space*
    (sqrt of mean variance). By Jensen's inequality these are not the same
    transform, so they carry different signal and the model gets both.

    Parkinson and Garman-Klass are different OHLC combinations estimating the
    same sigma with different idiosyncratic noise; including both gives the
    model a tiny ensemble of estimators. Daily *variances* are averaged
    (not daily *stds*) because variance is additive across independent
    increments while std is not.

    The intermediate same-day daily-variance columns are dropped before
    return so they cannot leak into downstream models.
    """
    panel = panel.copy()
    log_hl = np.log(panel["high"] / panel["low"])
    log_co = np.log(panel["close"] / panel["open"])
    parkinson_daily_var = (1.0 / (4.0 * np.log(2.0))) * log_hl**2
    # Garman & Klass  0.5*(ln H/L)^2 - (2*ln 2 - 1)*(ln C/O)^2.
    # clip(lower=0) handles days where a large |C-O| drives the GK estimate
    # negative -- a known finite-sample artifact of the negative coefficient
    # on log_co**2; sqrt of a negative would be NaN.
    gk_daily_var = (0.5 * log_hl**2 - (2.0 * np.log(2.0) - 1.0) * log_co**2).clip(lower=0.0)
    panel = panel.assign(_parkinson_var=parkinson_daily_var, _gk_var=gk_daily_var)

    def _annualized_sqrt_lag(d: pd.DataFrame, col: str, h: int) -> pd.Series:
        return np.sqrt(d[col].shift(1).rolling(h).mean() * TRADING_DAYS)

    for h in horizons:
        panel[f"parkinson_vol_lag_{h}d"] = _per_ticker_apply(
            panel, partial(_annualized_sqrt_lag, col="_parkinson_var", h=h)
        )
        panel[f"gk_vol_lag_{h}d"] = _per_ticker_apply(
            panel, partial(_annualized_sqrt_lag, col="_gk_var", h=h)
        )
    return panel.drop(columns=["_parkinson_var", "_gk_var"])


def _add_macro(panel: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Merge macro series, lagged by one day, plus rolling transforms.

    The rolling stats are computed on the macro frame itself rather than
    per-ticker because every ticker sees the same macro path on a given
    date; the per-ticker version was N-fold redundant.

    The 5-business-day ``ffill`` cap matches the one in
    :mod:`rvforecast.data.fetch_macro`. Weekend/holiday gaps are absorbed;
    multi-week outages stay NaN so they show up in the feature matrix
    instead of getting filled with a stale value across many trading days.
    """
    macro = macro.sort_index().ffill(limit=5)
    macro_lag = macro.shift(1)
    macro_lag = macro_lag.rename(columns={c: f"{c.lower()}_lag" for c in macro_lag.columns})
    macro_lag["vix_change_5d"] = macro_lag["vix_lag"].diff(5)
    rolling_mean = macro_lag["vix_lag"].rolling(252).mean()
    rolling_std = macro_lag["vix_lag"].rolling(252).std()
    macro_lag["vix_z_252"] = (macro_lag["vix_lag"] - rolling_mean) / rolling_std
    macro_lag["term_spread_change_5d"] = macro_lag["term_spread_lag"].diff(5)

    panel = (
        panel.reset_index().merge(macro_lag, on="date", how="left").set_index(["date", "ticker"])
    )
    return panel


def _add_calendar(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    dates = panel.index.get_level_values("date")
    panel["dow"] = dates.dayofweek.astype("int8")
    panel["month"] = dates.month.astype("int8")
    return panel


def _add_sector_rank(panel: pd.DataFrame, sector_map: pd.Series) -> pd.DataFrame:
    """Cross-sectional rank of trailing 22d Garman-Klass vol within sector.

    Uses ``gk_vol_lag_22d``, which is already a function of strictly past data,
    so the rank is leakage-free. Singletons (a single ticker mapped to its
    own sector, e.g., SPY mapped to ``ETF``) yield a degenerate constant
    rank of 1.0; we set those rows to NaN so the column does not introduce a
    spurious "rank" signal.
    """
    panel = panel.copy()
    tickers = panel.index.get_level_values("ticker")
    sectors = tickers.map(sector_map)
    if sectors.isna().any():
        unmapped = sorted(set(tickers[sectors.isna()]))
        warnings.warn(
            f"Sector map missing entries for {len(unmapped)} ticker(s); "
            f"falling back to 'Unknown'. First few: {unmapped[:10]}",
            stacklevel=2,
        )
    panel["sector"] = pd.Series(sectors, index=panel.index).fillna("Unknown")
    dates = panel.index.get_level_values("date")
    grouped = panel.groupby([dates, "sector"])["gk_vol_lag_22d"]
    panel["sector_vol_rank_22d"] = grouped.rank(pct=True).values
    sector_size = grouped.transform("size")
    panel.loc[sector_size.values <= 1, "sector_vol_rank_22d"] = np.nan
    return panel


def build_feature_matrix(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    sector_map: pd.Series,
) -> pd.DataFrame:
    """Construct the modeling feature matrix.

    Parameters
    ----------
    prices : DataFrame
        Long panel indexed by ``(date, ticker)`` with OHLC and ``adj_close``.
    macro : DataFrame
        Macro time series indexed by ``date`` with columns ``vix``, ``DGS10``,
        ``DGS2``, ``term_spread``.
    sector_map : Series
        Maps ticker to sector name.

    Returns
    -------
    DataFrame
        Feature matrix indexed by ``(date, ticker)`` with the regression
        target ``y_target`` (log realized vol of the row's date) appended.
        Features at row ``t`` are functions of data with index strictly less
        than ``t``, so the row predicts day ``t`` from data through ``t-1``.

    Notes
    -----
    Long-window features are NaN early in the panel where the rolling window
    is not yet warm: 22-day HAR/range lags need ~22 prior observations per
    ticker, and ``vix_z_252`` needs ~252 prior macro observations. Only
    rows with NaN ``y_target`` are dropped before returning, so the caller
    must handle NaN feature columns. LightGBM tolerates NaN natively;
    ``models/har.py`` fills with zero; the LSTM uses a dropna mask while
    constructing sequences.
    """
    panel = add_target_columns(prices.copy())
    panel = _add_returns(panel)
    panel = _add_har_lags(panel)
    panel = _add_range_lags(panel)
    panel = _add_macro(panel, macro)
    panel = _add_calendar(panel)
    panel = _add_sector_rank(panel, sector_map)

    # Target is the row's own day log realized vol; combined with shift(1)
    # features above this gives a one-day-ahead forecast.
    panel["y_target"] = panel["log_rv_1d"].astype(float)

    # Drop raw OHLCV, the unshifted realized-vol columns used to build the
    # target, and the contemporaneous return intermediate. ``_ret_1d_raw``
    # MUST NOT survive into the persisted matrix: at row ``t`` it equals
    # ``log(close_t / close_{t-1})``, which depends on ``close_t`` and would
    # leak directly into any model that does not re-shift it.
    drop = [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "rv_1d",
        "log_rv_1d",
        _RAW_RET_COL,
    ]
    panel = panel.drop(columns=[c for c in drop if c in panel.columns])

    panel = panel.dropna(subset=["y_target"])
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, default=DATA_RAW / "prices_long.parquet")
    parser.add_argument("--macro", type=Path, default=DATA_RAW / "macro.parquet")
    parser.add_argument("--sector-map", type=Path, default=ROOT / "configs" / "sector_map.csv")
    parser.add_argument("--out", type=Path, default=DATA_PROCESSED / "features.parquet")
    args = parser.parse_args()

    ensure_output_dirs()
    prices = pd.read_parquet(args.prices)
    macro = pd.read_parquet(args.macro)
    sector = pd.read_csv(args.sector_map).set_index("ticker")["sector"]

    features = build_feature_matrix(prices, macro, sector)
    features.to_parquet(args.out)
    print(f"Wrote feature matrix with shape {features.shape} to {args.out}")


if __name__ == "__main__":
    main()

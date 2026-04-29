"""GARCH(1,1) baseline (Bollerslev 1986), per ticker.

For each ticker, fit a GARCH(1,1) on percent log returns (returns multiplied
by 100 for numerical stability; the variance is rescaled accordingly). To
control compute, parameters ``(omega, alpha, beta)`` are refit at most
monthly within each test window. Between refits, the conditional variance
is propagated forward daily using the GARCH(1,1) recursion

    sigma2_{t+1} = omega + alpha * r_t^2 + beta * sigma2_t,

so the within-month forecast reacts to observed test returns rather than
being held constant at the start-of-month value. At each refit, the
parameter vector is updated and ``sigma2`` is re-anchored to the new fit's
last in-sample conditional variance. Convergence failures retain the prior
fit's parameter vector; tickers absent from training (cold-start) yield
NaN predictions.

To keep refits flat-cost across long folds, the in-memory return history
used by per-month refits is capped at :data:`MAX_HISTORY_DAYS` (~5 years).
Without the cap, a 17-year fold's last refit would fit on roughly four
thousand observations; the cap keeps that work bounded.

Returns are sourced from the cached price panel (``data/raw/prices_long.parquet``)
rather than from the feature matrix. The feature matrix intentionally
does not expose the contemporaneous one-day return because it is a same-day
measurement and would leak into any model that consumes the matrix without
re-shifting; GARCH's recursion implicitly shifts it via ``last_r``, so it is
safe here.

Citation
--------
Bollerslev, T. (1986). Generalized autoregressive conditional
heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

from rvforecast.config import DATA_RAW, TRADING_DAYS, ensure_output_dirs
from rvforecast.models._runner import load_features, run_walk_forward

# Cap on the in-memory return history retained between monthly refits.
# Five trading-year-equivalents keeps the per-refit cost bounded while
# preserving enough history for stable GARCH(1,1) estimation.
MAX_HISTORY_DAYS: int = 5 * TRADING_DAYS


def _fit_garch(returns_pct: pd.Series) -> ARCHModelResult | None:
    """Fit GARCH(1,1) on percent returns; return ``None`` on failure.

    The ``arch`` package raises ``LinAlgError`` on singular Hessians,
    ``ValueError`` on degenerate inputs, and ``RuntimeError`` when the
    optimizer fails to converge. Each of these means we should fall back
    to the prior fit's parameters rather than crash the walk-forward run.

    Warnings are silenced narrowly: only the categories that arch and
    scipy.optimize emit on the optimization path (``RuntimeWarning``
    from numpy under overflow/underflow, and ``UserWarning`` from arch
    around rescale/convergence) are filtered. Other warning classes
    propagate so genuine surprises are not swallowed.
    """
    series = returns_pct.dropna()
    if len(series) < 100:
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="arch.*")
        try:
            am = arch_model(series, vol="GARCH", p=1, q=1, mean="Zero", rescale=False)
            res = am.fit(disp="off", show_warning=False)
            if not np.all(np.isfinite(res.params.values)):
                return None
            return res
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            return None


def _annualized_log_vol(sigma2_pct_sq: float) -> float:
    """Return log of annualized volatility from a daily conditional variance.

    Input is the GARCH conditional variance in (percent return)^2 units;
    output is ``log(sqrt(annual_var_in_decimal_units))``, i.e., log of the
    annualized standard deviation in the same decimal units as the modeling
    target. The ``max(..., 1e-12)`` floor protects ``log`` from a zero
    conditional variance under degenerate estimates.
    """
    var_decimal = sigma2_pct_sq / (100.0**2)
    annual_var = var_decimal * TRADING_DAYS
    return float(np.log(np.sqrt(max(annual_var, 1e-12))))


def _extract_params(res: ARCHModelResult) -> tuple[float, float, float]:
    """Pull (omega, alpha, beta) out of a fitted GARCH(1,1) result.

    The ``arch`` package names parameters ``omega``, ``alpha[1]``, ``beta[1]``
    for GARCH(1,1) under a zero-mean specification. KeyError surfaces here if
    a future arch version renames them so the failure is loud rather than
    silently zeroing out the dynamics. ``tests/test_garch.py`` pins this
    contract against the installed arch version.
    """
    p = res.params
    return float(p["omega"]), float(p["alpha[1]"]), float(p["beta[1]"])


def _per_ticker_predict(
    test_index: pd.MultiIndex,
    train_returns_pct: pd.Series,
    test_returns_pct: np.ndarray,
    test_dates: pd.DatetimeIndex,
) -> pd.Series:
    """Daily one-step-ahead log vol forecasts with monthly parameter refits.

    On each test day ``t``, the conditional variance is

        sigma2_t = omega + alpha * r_{t-1}^2 + beta * sigma2_{t-1},

    where ``r_{t-1}`` is the return observed on the previous trading day
    (training return for the very first test day, then the prior test return
    thereafter). Parameters are refit when the calendar month changes and
    ``sigma2`` is re-anchored to the new fit's last in-sample value to absorb
    any drift accumulated under stale parameters.

    Missing returns
    ---------------
    A NaN test return on day ``t`` is treated as a missing innovation: day
    ``t+1``'s forecast is NaN (because ``r_t`` is unobserved), ``sigma2`` is
    not advanced, and the recursion resumes with the next observed return
    using the last finite ``sigma2``. We do **not** substitute zero for the
    missing return, which would inject a "no-movement" innovation and bias
    the conditional variance toward its unconditional level. Train and test
    paths thus reject missing observations symmetrically: ``_fit_garch``
    drops NaNs from training input, and the recursion above emits NaN.

    Memory note: ``history`` is truncated to the trailing :data:`MAX_HISTORY_DAYS`
    finite observations before each refit so per-refit cost stays bounded.
    """
    res = _fit_garch(train_returns_pct)
    if res is None:
        return pd.Series(np.nan, index=test_index, dtype=float)

    omega, alpha, beta = _extract_params(res)
    sigma2 = float(res.conditional_volatility.iloc[-1]) ** 2
    last_r = float(train_returns_pct.iloc[-1])
    history: list[float] = list(map(float, train_returns_pct.values))

    preds: list[float] = []
    last_month: tuple[int, int] | None = None
    for i, date in enumerate(test_dates):
        month = (date.year, date.month)
        if last_month is not None and month != last_month:
            arr = np.asarray(history, dtype=float)
            clean = arr[np.isfinite(arr)]
            if clean.size > MAX_HISTORY_DAYS:
                clean = clean[-MAX_HISTORY_DAYS:]
            new_res = _fit_garch(pd.Series(clean))
            if new_res is not None:
                omega, alpha, beta = _extract_params(new_res)
                sigma2 = float(new_res.conditional_volatility.iloc[-1]) ** 2
                last_r = float(clean[-1]) if clean.size else float("nan")
        last_month = month

        if np.isfinite(last_r) and np.isfinite(sigma2):
            sigma2 = omega + alpha * last_r**2 + beta * sigma2
            preds.append(_annualized_log_vol(sigma2))
        else:
            preds.append(float("nan"))

        last_r = float(test_returns_pct[i])
        history.append(last_r)

    return pd.Series(preds, index=test_index, dtype=float)


def predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    returns: pd.Series,
) -> pd.Series:
    """Per-fold GARCH predictions, public entry point.

    Parameters
    ----------
    train, test : DataFrame
        Slices of the feature panel for one walk-forward fold or for the
        pre-holdout / holdout split. Must be indexed by ``(date, ticker)``.
    returns : Series
        Log close-to-close returns indexed by ``(date, ticker)`` covering at
        least every ``(date, ticker)`` pair present in ``train`` and
        ``test``. Sourced from the cached price panel rather than the
        feature matrix to keep the same-day return out of the feature
        matrix entirely.

    Cold-start tickers (in test, absent from train) yield NaN.
    """
    parts: list[pd.Series] = []
    test_tickers = test.index.get_level_values("ticker").unique()
    train_tickers = set(train.index.get_level_values("ticker").unique())
    for t in test_tickers:
        test_t = test.xs(t, level="ticker", drop_level=False).sort_index()
        if t not in train_tickers or test_t.empty:
            parts.append(pd.Series(np.nan, index=test_t.index))
            continue
        train_t = train.xs(t, level="ticker", drop_level=False).sort_index()
        if train_t.empty:
            parts.append(pd.Series(np.nan, index=test_t.index))
            continue

        ticker_rets = returns.xs(t, level="ticker") if "ticker" in returns.index.names else returns
        train_dates = train_t.index.get_level_values("date")
        test_dates = test_t.index.get_level_values("date")
        train_returns_pct = (ticker_rets.reindex(train_dates) * 100.0).dropna()
        if len(train_returns_pct) < 100:
            parts.append(pd.Series(np.nan, index=test_t.index))
            continue
        # Missing test returns flow through as NaN; ``_per_ticker_predict``
        # emits NaN forecasts on the days that depend on them. Train uses
        # ``dropna``; both paths therefore refuse to fabricate observations.
        test_returns_pct = (ticker_rets.reindex(test_dates) * 100.0).to_numpy(dtype=float)
        parts.append(
            _per_ticker_predict(
                test_t.index,
                train_returns_pct,
                test_returns_pct,
                test_dates,
            )
        )
    if not parts:
        return pd.Series(index=test.index, dtype=float)
    return pd.concat(parts).reindex(test.index)


def load_returns_panel() -> pd.Series:
    """Load close-to-close log returns per ticker from the cached price panel."""
    prices = pd.read_parquet(DATA_RAW / "prices_long.parquet")
    return np.log(prices["adj_close"]).groupby(level="ticker").diff().rename("ret_1d_raw")


def main() -> None:
    ensure_output_dirs()
    panel = load_features()
    returns = load_returns_panel()

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
        return predict(train, test, returns)

    out = run_walk_forward(panel, fit_predict, out_name="garch")
    print(f"GARCH(1,1) predictions written to {out}")


if __name__ == "__main__":
    main()

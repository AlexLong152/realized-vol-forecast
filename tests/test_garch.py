"""Check the parameter-name contract between this project and ``arch``.

A zero-mean GARCH(1,1) in the current ``arch`` version exposes its
parameters as ``omega``, ``alpha[1]``, ``beta[1]``.
``rvforecast.models.garch._extract_params`` reads those keys by name, so
if a future ``arch`` release renames them this test fails loudly. Without
it, GARCH would silently zero out its own dynamics and we'd never know.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("arch")

from arch import arch_model  # noqa: E402
from rvforecast.models.garch import _extract_params, _fit_garch  # noqa: E402


def _synthetic_garch_returns(n: int = 1500, seed: int = 0) -> pd.Series:
    """Simulate a zero-mean GARCH(1,1) percent-return series.

    Stationarity requires ``alpha + beta < 1``; we pick (0.05, 0.9) which
    is well inside the simplex and broadly representative of equity-index
    GARCH(1,1) estimates.
    """
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 0.02, 0.05, 0.9
    sigma2 = omega / max(1.0 - alpha - beta, 1e-9)
    returns = np.empty(n, dtype=float)
    for t in range(n):
        eps = rng.standard_normal()
        returns[t] = np.sqrt(sigma2) * eps
        sigma2 = omega + alpha * returns[t] ** 2 + beta * sigma2
    return pd.Series(returns)


def test_arch_param_names_match_extract_params():
    res = _fit_garch(_synthetic_garch_returns())
    assert res is not None, "synthetic series should always produce a fit"
    params = res.params
    for key in ("omega", "alpha[1]", "beta[1]"):
        assert key in params.index, f"expected key '{key}' in arch params, got {list(params.index)}"
    omega, alpha, beta = _extract_params(res)
    assert all(np.isfinite([omega, alpha, beta]))
    # Sanity-check stationarity of the recovered estimates; not strict
    # equality to the simulation parameters because 1500 obs is not
    # enough to recover them tightly.
    assert alpha >= 0.0 and beta >= 0.0
    assert alpha + beta < 1.0


def test_fit_garch_returns_none_on_too_short_series():
    # 50 observations is well under the 100-row floor in ``_fit_garch``.
    short = pd.Series(np.random.default_rng(0).standard_normal(50))
    assert _fit_garch(short) is None


def test_arch_model_zero_mean_fit_smoke():
    """Sanity check that the bare ``arch_model`` invocation we use works."""
    series = _synthetic_garch_returns(n=500)
    am = arch_model(series, vol="GARCH", p=1, q=1, mean="Zero", rescale=False)
    res = am.fit(disp="off", show_warning=False)
    assert {"omega", "alpha[1]", "beta[1]"}.issubset(set(res.params.index))


def test_per_ticker_predict_nan_test_return_yields_nan_forecast_no_zero_substitution():
    """A missing test return must not get silently treated as a zero return.

    The behavior is documented on ``_per_ticker_predict``: a NaN test
    return on day ``t`` makes day ``t+1``'s forecast NaN (because ``r_t``
    is unobserved), the recursion does not advance ``sigma2``, and once
    finite returns come back the recursion resumes. The NaN forecast is
    *different* from what the old ``fillna(0.0)`` code did — substituting
    zero pulls the conditional variance toward its unconditional level
    and quietly biases the forecast. That's the bug this test catches.
    """
    from rvforecast.models.garch import _per_ticker_predict

    train_returns = _synthetic_garch_returns(n=400)
    train_returns_pct = train_returns * 100.0  # the public predict() multiplies by 100

    rng = np.random.default_rng(42)
    n_test = 8
    test_dates = pd.bdate_range("2020-01-02", periods=n_test)
    ticker = "AAA"
    test_index = pd.MultiIndex.from_arrays(
        [test_dates, [ticker] * n_test], names=["date", "ticker"]
    )
    test_returns_pct = rng.standard_normal(n_test) * 1.0

    # Path A: a clean NaN-free run.
    preds_clean = _per_ticker_predict(
        test_index, train_returns_pct, test_returns_pct.copy(), test_dates
    )
    assert preds_clean.notna().all(), "clean test returns should produce all-finite forecasts"

    # Path B: drop in a missing return on day index 3.
    test_returns_with_nan = test_returns_pct.copy()
    test_returns_with_nan[3] = np.nan
    preds_nan = _per_ticker_predict(
        test_index, train_returns_pct, test_returns_with_nan, test_dates
    )

    # Day 4's forecast uses r_3 (the missing return). It must be NaN, not the
    # value the old fillna(0.0) code would have produced.
    assert np.isnan(preds_nan.iloc[4]), (
        "missing return at day 3 must propagate NaN to day 4's forecast; "
        "got a finite forecast instead, which suggests fillna(0.0) is back."
    )
    # Days 0-3 are unaffected (they depend on returns at indices -1 through 2).
    assert preds_nan.iloc[:4].equals(preds_clean.iloc[:4])
    # The recursion resumes once finite returns return: day 5 onward is finite.
    assert preds_nan.iloc[5:].notna().all()

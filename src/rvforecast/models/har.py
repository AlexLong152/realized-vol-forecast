"""HAR-RV baseline (Corsi 2009).

We regress one-day-ahead log realized volatility on the 1-day, 5-day, and
22-day lagged log realized volatility, pooling across tickers with ticker
fixed effects. Ticker fixed effects are absorbed by ``pd.get_dummies`` plus
``sm.add_constant`` (one ticker is dropped as the reference, and its
intercept is captured by the constant); this is more efficient than
per-ticker OLS when the cross section is wide and the time series within
each ticker is relatively short.

A test-fold ticker that was absent from the training fold cannot be
matched to a training fixed effect. The implementation falls back to the
reference ticker's intercept (zero-filling the absent dummies), which is
the conventional handling but is silently arbitrary because the choice of
reference is alphabetical. We log a warning to stderr when this happens
so the behavior is observable.

Citation
--------
Corsi, F. (2009). A simple approximate long-memory model of realized
volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

from rvforecast.config import HAR_HORIZONS, ensure_output_dirs
from rvforecast.models._runner import load_features, run_walk_forward

HAR_REGRESSORS: list[str] = [f"log_rv_lag_{h}d" for h in HAR_HORIZONS]


def _design_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[HAR_REGRESSORS].copy()
    tickers = df.index.get_level_values("ticker")
    fe = pd.get_dummies(tickers, prefix="ticker", drop_first=True).astype(float)
    fe.index = df.index
    X = pd.concat([X, fe], axis=1)
    X = sm.add_constant(X, has_constant="add")
    return X, df["y_target"].astype(float)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Public per-fold HAR-RV predictions.

    Pooled OLS with ticker fixed effects, refit per fold. A test ticker
    absent from the training set falls back to the reference ticker's
    intercept; this is logged to stderr.
    """
    train = train.dropna(subset=HAR_REGRESSORS + ["y_target"])
    X_train, y_train = _design_matrix(train)
    model = sm.OLS(y_train, X_train).fit()

    test_tickers = set(test.index.get_level_values("ticker").unique())
    train_tickers = set(train.index.get_level_values("ticker").unique())
    new_tickers = test_tickers - train_tickers
    if new_tickers:
        print(
            f"har: {len(new_tickers)} ticker(s) in test absent from train "
            f"({sorted(new_tickers)!r}); falling back to reference-ticker intercept",
            file=sys.stderr,
        )

    X_test, _ = _design_matrix(test)
    # Align test design matrix to the training column set; unseen ticker
    # dummies are zero-filled, which falls back to the reference ticker's
    # intercept and is the conventional handling for new categories.
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
    pred = pd.Series(model.predict(X_test), index=test.index, dtype=float)
    return pred.replace([np.inf, -np.inf], np.nan)


def main() -> None:
    ensure_output_dirs()
    panel = load_features()
    out = run_walk_forward(panel, fit_predict, out_name="har")
    print(f"HAR-RV predictions written to {out}")


if __name__ == "__main__":
    main()

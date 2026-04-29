"""Naive baseline: predict ``log(rv)_t = log(rv_lag_1d)_t = log(rv)_{t-1}``.

This is the floor every other model must beat. It uses only the most recent
one-day lagged log realized volatility for the same ticker.
"""

from __future__ import annotations

import pandas as pd

from rvforecast.config import ensure_output_dirs
from rvforecast.models._runner import load_features, run_walk_forward


def fit_predict(_train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    # ``_train`` is unused: the naive model reads the row's own lagged-1d
    # column from the test slice directly.
    return test["log_rv_lag_1d"]


def main() -> None:
    ensure_output_dirs()
    panel = load_features()
    out = run_walk_forward(panel, fit_predict, out_name="naive")
    print(f"Naive predictions written to {out}")


if __name__ == "__main__":
    main()

"""Common utilities for running models through the walk-forward loop.

A "model" here is anything with a ``fit_predict(train_df, test_df)``
function (optionally taking ``fold`` / ``n_folds`` kwargs) that returns
a Series of predictions aligned to ``test_df.index``. This module loads
the feature matrix and splits, runs every model on the same folds, and
writes predictions in a single shape: a long DataFrame indexed by
``(date, ticker)`` with columns ``y_true`` (log realized vol) and
``y_pred`` (forecast in log space).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

from rvforecast.config import DATA_PROCESSED, RESULTS_PREDICTIONS, ensure_output_dirs
from rvforecast.validation.walk_forward import (
    iter_fold_dates,
    load_splits,
    persist_splits,
    trading_dates_from_panel,
)

FEATURES_PATH = DATA_PROCESSED / "features.parquet"

FitPredict = Callable[..., pd.Series]


def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def ensure_splits(panel: pd.DataFrame) -> Path:
    ensure_output_dirs()
    dates = trading_dates_from_panel(panel)
    return persist_splits(dates)


def _slice_by_dates(panel: pd.DataFrame, dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    mask = panel.index.get_level_values("date").isin(list(dates))
    return panel.loc[mask]


def _accepts_fold_kwarg(fn: Callable) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = sig.parameters.values()
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return True
    return "fold" in sig.parameters


def _count_folds(_dates: pd.DatetimeIndex) -> int:
    """Read the precomputed fold count from the persisted splits file.

    ``persist_splits`` is called immediately before this in
    :func:`run_walk_forward`, so the file is fresh.
    """
    return int(load_splits()["n_folds"])


def run_walk_forward(
    panel: pd.DataFrame,
    fit_predict: FitPredict,
    out_name: str,
) -> Path:
    """Run a model across all walk-forward folds and persist predictions.

    Parameters
    ----------
    panel : DataFrame
        Output of the feature builder, must include ``y_target``.
    fit_predict : callable
        Takes ``(train_df, test_df)`` and optionally the keyword arguments
        ``fold`` (the zero-indexed fold id) and ``n_folds`` (total fold
        count). Must return a Series of predictions aligned to
        ``test_df.index``. The fold metadata lets a model checkpoint per
        fold or branch on the final fold; baselines that do not need it
        can simply omit the kwargs.
    out_name : str
        Used to name the output parquet file under
        ``results/predictions/<out_name>.parquet``.
    """
    ensure_output_dirs()
    ensure_splits(panel)
    panel = panel.sort_index()
    dates = trading_dates_from_panel(panel)
    n_folds = _count_folds(dates)
    accepts_fold = _accepts_fold_kwarg(fit_predict)
    chunks: list[pd.DataFrame] = []
    for fold, train_dates, test_dates in iter_fold_dates(dates):
        train = _slice_by_dates(panel, train_dates).dropna(subset=["y_target"])
        test = _slice_by_dates(panel, test_dates).dropna(subset=["y_target"])
        if train.empty or test.empty:
            continue
        if accepts_fold:
            preds = fit_predict(train, test, fold=fold, n_folds=n_folds)
        else:
            preds = fit_predict(train, test)
        out = pd.DataFrame(
            {
                "y_true": test["y_target"].astype(float),
                "y_pred": preds.astype(float).reindex(test.index),
                "fold": fold,
            },
            index=test.index,
        )
        chunks.append(out)
    if not chunks:
        raise RuntimeError("No predictions produced; check fold setup and feature matrix.")
    full = pd.concat(chunks).sort_index()
    out_path = RESULTS_PREDICTIONS / f"{out_name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_path)
    return out_path

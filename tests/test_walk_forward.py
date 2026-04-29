"""Tests for the walk-forward splitter."""

from __future__ import annotations

import pandas as pd
import pytest
from rvforecast.config import TRADING_DAYS
from rvforecast.validation.walk_forward import (
    WalkForwardSplitter,
    load_holdout_cutoff,
    load_holdout_dates,
    load_pre_holdout_train_dates,
)


def _dates(years: int = 12) -> pd.DatetimeIndex:
    return pd.bdate_range("2010-01-04", periods=252 * years)


def test_no_train_test_overlap():
    dates = _dates()
    splitter = WalkForwardSplitter(
        initial_train_years=5, test_window_months=6, purge_days=5, embargo_days=5
    )
    for train, test in splitter.split(dates):
        assert train.intersection(test).empty


def test_purge_gap_respected():
    dates = _dates()
    splitter = WalkForwardSplitter(
        initial_train_years=5, test_window_months=6, purge_days=5, embargo_days=5
    )
    for train, test in splitter.split(dates):
        gap = (test.min() - train.max()).days
        assert gap >= 5, f"Purge gap {gap} days < required 5"


def test_embargo_excluded_from_immediate_next_fold():
    """The ``embargo_days`` business days after fold k's test window must not
    appear in fold k+1's training or test set.

    Following López de Prado, embargo is local to consecutive folds: it
    prevents leakage at the boundary between fold k and fold k+1. Older
    embargo zones (from much earlier folds) become valid training data
    once they are sufficiently far from the current test boundary, and
    expanding-window walk-forward does not exclude them permanently.
    """
    dates = _dates()
    embargo = 5
    splitter = WalkForwardSplitter(
        initial_train_years=5, test_window_months=6, purge_days=5, embargo_days=embargo
    )
    folds = list(splitter.split(dates))
    assert len(folds) >= 2
    for i in range(len(folds) - 1):
        _, test_i = folds[i]
        end_pos = dates.get_loc(test_i.max())
        embargo_zone = dates[end_pos + 1 : end_pos + 1 + embargo]
        train_next, test_next = folds[i + 1]
        assert train_next.intersection(embargo_zone).empty, (
            f"Fold {i+1} train overlaps embargo zone after fold {i}: "
            f"{train_next.intersection(embargo_zone).tolist()}"
        )
        assert test_next.intersection(embargo_zone).empty, (
            f"Fold {i+1} test overlaps embargo zone after fold {i}: "
            f"{test_next.intersection(embargo_zone).tolist()}"
        )


def test_embargo_plus_purge_gap_between_consecutive_test_windows():
    """The gap between consecutive test windows is at least ``embargo + purge``.

    The first business day of fold k+1's test must be at least
    ``embargo_days + purge_days`` business days after the last day of
    fold k's test.
    """
    dates = _dates()
    embargo = 5
    purge = 5
    splitter = WalkForwardSplitter(
        initial_train_years=5,
        test_window_months=6,
        purge_days=purge,
        embargo_days=embargo,
    )
    folds = list(splitter.split(dates))
    assert len(folds) >= 2
    for (_, test_prev), (_, test_next) in zip(folds, folds[1:], strict=False):
        prev_end_pos = dates.get_loc(test_prev.max())
        next_start_pos = dates.get_loc(test_next.min())
        assert next_start_pos - prev_end_pos >= embargo + purge


def test_holdout_never_in_any_fold():
    dates = _dates()
    holdout = load_holdout_dates(dates)
    splitter = WalkForwardSplitter(
        initial_train_years=5, test_window_months=6, purge_days=5, embargo_days=5
    )
    for train, test in splitter.split(dates):
        assert train.intersection(holdout).empty
        assert test.intersection(holdout).empty


def test_holdout_cutoff_is_two_years_before_max():
    dates = _dates()
    cutoff = load_holdout_cutoff(dates)
    expected = dates.max() - pd.DateOffset(years=2)
    assert cutoff == expected


def test_at_least_one_fold():
    dates = _dates()
    splitter = WalkForwardSplitter()
    folds = list(splitter.split(dates))
    assert len(folds) >= 1


def test_initial_train_too_large_raises():
    dates = pd.bdate_range("2020-01-01", periods=200)
    splitter = WalkForwardSplitter(initial_train_years=5)
    with pytest.raises(ValueError):
        list(splitter.split(dates))


def test_rolling_mode_keeps_train_size_constant():
    """Rolling mode must hold the training window length fixed.

    Earlier the start index was advanced by the wrong offset and the
    rolling window grew by ``purge + test_size`` per fold.
    """
    dates = _dates()
    splitter = WalkForwardSplitter(
        initial_train_years=5,
        test_window_months=6,
        purge_days=5,
        embargo_days=5,
        mode="rolling",
    )
    expected = 5 * TRADING_DAYS
    folds = list(splitter.split(dates))
    assert len(folds) >= 2
    # First fold may be shorter only if the start was clamped to 0; here
    # the training history is long enough that every fold is exactly
    # ``initial_train_years * TRADING_DAYS`` long.
    for train, _ in folds:
        assert len(train) == expected


def test_pre_holdout_train_dates_purge_gap():
    dates = _dates()
    pre = load_pre_holdout_train_dates(dates, purge_days=5)
    holdout = load_holdout_dates(dates)
    cutoff = load_holdout_cutoff(dates)
    assert pre.max() < cutoff
    # The purge keeps a five-business-day gap before the holdout window.
    pre_full = dates[dates < cutoff]
    assert len(pre) == len(pre_full) - 5
    assert pre.intersection(holdout).empty


def test_rolling_mode_short_test_window_emits_many_folds():
    """A short test window in rolling mode should produce many same-length folds.

    Exercises the rolling-mode start-index advance over many iterations,
    which is a different code path from the single test-window-per-fold
    case covered by ``test_rolling_mode_keeps_train_size_constant``.
    """
    dates = _dates()
    splitter = WalkForwardSplitter(
        initial_train_years=5,
        test_window_months=1,
        purge_days=2,
        embargo_days=2,
        mode="rolling",
    )
    expected = 5 * TRADING_DAYS
    folds = list(splitter.split(dates))
    assert len(folds) >= 10
    for train, test in folds:
        assert len(train) == expected
        assert train.intersection(test).empty
        assert (test.min() - train.max()).days >= 2


def test_no_train_dates_overlap_holdout_in_rolling_mode():
    dates = _dates()
    holdout = load_holdout_dates(dates)
    splitter = WalkForwardSplitter(mode="rolling")
    for train, test in splitter.split(dates):
        assert train.intersection(holdout).empty
        assert test.intersection(holdout).empty

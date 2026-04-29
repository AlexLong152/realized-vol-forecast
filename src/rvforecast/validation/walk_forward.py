"""Walk-forward splitter with purge and embargo.

The splitter operates on the unique set of dates in the panel index. The last
``HOLDOUT_YEARS`` of dates are reserved as a one-shot final test set; the
splitter never yields any train or test fold containing those dates. Use
:func:`load_holdout_dates` to retrieve them at the very end of the project.

Reference: López de Prado, M. (2018), *Advances in Financial Machine
Learning*, ch. 7, on purged and embargoed cross-validation for time series.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from rvforecast.config import (
    EMBARGO_DAYS,
    HOLDOUT_YEARS,
    INITIAL_TRAIN_YEARS,
    PURGE_DAYS,
    RESULTS,
    TEST_WINDOW_MONTHS,
    TRADING_DAYS,
)


@dataclass
class WalkForwardSplitter:
    """Yield ``(train_dates, test_dates)`` pairs in walk-forward order.

    Parameters
    ----------
    initial_train_years : int
        Length of the first training window in years (calendar, approximated
        as ``years * TRADING_DAYS``).
    test_window_months : int
        Length of each test window in months (approximated as months * 21).
    purge_days : int
        Number of trailing training days dropped before each test window to
        reduce leakage from overlapping target horizons.
    embargo_days : int
        Number of dates after each test window that are excluded from the
        *immediately following* fold (its training and test alike). The
        gap between fold ``k``'s last test date and fold ``k+1``'s first
        test date is ``embargo_days + purge_days`` business days. Following
        López de Prado, the exclusion is local to consecutive folds: in
        expanding mode, fold ``k+2``'s training treats the dates after
        ``test_k`` as ordinary training data, since they are no longer
        adjacent to a current test boundary. Embargo does not apply before
        the first fold (there is no preceding test window).
    mode : {'expanding', 'rolling'}
        Whether the training window expands across folds or rolls forward
        with a fixed length.

    Notes
    -----
    Year and month parameters are sliced against the *trading-day index*
    (252 trading days per year, 21 per month). Calendar quirks (early
    closes, leap years, month-end timing) cause the actual block length
    to drift by a small handful of days over a decade. The drift is
    small relative to ``purge_days`` and ``embargo_days`` and does not
    affect the leakage-control properties of the splitter.
    """

    initial_train_years: int = INITIAL_TRAIN_YEARS
    test_window_months: int = TEST_WINDOW_MONTHS
    purge_days: int = PURGE_DAYS
    embargo_days: int = EMBARGO_DAYS
    mode: Literal["expanding", "rolling"] = "expanding"

    def split(self, dates: pd.DatetimeIndex) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        usable = self._strip_holdout(dates)
        train_size = self.initial_train_years * TRADING_DAYS
        test_size = max(1, self.test_window_months * 21)
        if train_size >= len(usable):
            raise ValueError(
                f"Initial training window ({train_size}) exceeds usable dates ({len(usable)})"
            )

        start = 0
        train_end = train_size
        first = True
        while True:
            # First fold: no embargo before; only the train→test purge applies.
            # Subsequent folds: embargo_days from the prior test are excluded
            # entirely (never appearing as train or test), then purge_days.
            gap_before_test = self.purge_days if first else (self.embargo_days + self.purge_days)
            test_start = train_end + gap_before_test
            test_end = test_start + test_size
            if test_end > len(usable):
                break
            yield usable[start:train_end], usable[test_start:test_end]
            # The next fold's training ends where this fold's test ended; the
            # embargo zone [test_end : test_end + embargo_days) sits between
            # consecutive folds and is permanently excluded.
            train_end = test_end
            if self.mode == "expanding":
                start = 0
            else:
                # Rolling: keep the training window length fixed at train_size.
                start = max(0, train_end - train_size)
            first = False

    @staticmethod
    def _strip_holdout(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        normalized = _normalize_dates(dates)
        cutoff = _cutoff_from_normalized(normalized)
        return normalized[normalized < cutoff]


def _normalize_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Sort, deduplicate, and return ``dates`` as a ``DatetimeIndex``.

    Centralized so each public helper only does the normalization once;
    upstream callers (e.g., the splitter, the runner) frequently chain
    several of these together.
    """
    return pd.DatetimeIndex(dates).unique().sort_values()


def _cutoff_from_normalized(dates: pd.DatetimeIndex) -> pd.Timestamp:
    return dates.max() - pd.DateOffset(years=HOLDOUT_YEARS)


def load_holdout_cutoff(dates: pd.DatetimeIndex) -> pd.Timestamp:
    """Return the date at which the final two-year holdout begins."""
    return _cutoff_from_normalized(_normalize_dates(dates))


def load_holdout_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the dates reserved for the one-shot final test."""
    normalized = _normalize_dates(dates)
    cutoff = _cutoff_from_normalized(normalized)
    return normalized[normalized >= cutoff]


def load_pre_holdout_train_dates(
    dates: pd.DatetimeIndex, purge_days: int = PURGE_DAYS
) -> pd.DatetimeIndex:
    """Pre-holdout training dates with a purge gap before the holdout window.

    Mirrors the walk-forward purge: the last ``purge_days`` business days
    before the holdout cutoff are excluded from training, so the holdout
    evaluation uses the same train/test separation discipline as the
    expanding folds.
    """
    normalized = _normalize_dates(dates)
    cutoff = _cutoff_from_normalized(normalized)
    pre = normalized[normalized < cutoff]
    if purge_days > 0 and len(pre) > purge_days:
        pre = pre[:-purge_days]
    return pre


def persist_splits(
    dates: pd.DatetimeIndex,
    splitter: WalkForwardSplitter | None = None,
    out: Path | None = None,
) -> Path:
    """Materialize and persist all (train, test) date ranges to JSON.

    Every model loads from this file so that all evaluations share identical
    folds.
    """
    splitter = splitter or WalkForwardSplitter()
    out = out or (RESULTS / "splits.json")
    folds: list[dict[str, list[str]]] = []
    for i, (train, test) in enumerate(splitter.split(dates)):
        folds.append(
            {
                "fold": i,
                "train_start": str(train.min().date()),
                "train_end": str(train.max().date()),
                "test_start": str(test.min().date()),
                "test_end": str(test.max().date()),
            }
        )
    payload = {
        "holdout_start": str(load_holdout_cutoff(dates).date()),
        "n_folds": len(folds),
        "folds": folds,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    return out


def load_splits(path: Path | None = None) -> dict:
    path = path or (RESULTS / "splits.json")
    with open(path) as f:
        return json.load(f)


def iter_fold_dates(
    panel_dates: pd.DatetimeIndex, splits_path: Path | None = None
) -> Iterator[tuple[int, pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Iterate ``(fold_id, train_dates, test_dates)`` from persisted splits.

    The dates in the JSON are date strings; this helper resolves them back to
    the actual sorted, unique trading dates present in ``panel_dates``.
    """
    payload = load_splits(splits_path)
    panel_dates = _normalize_dates(panel_dates)
    for f in payload["folds"]:
        ts = panel_dates[
            (panel_dates >= pd.Timestamp(f["train_start"]))
            & (panel_dates <= pd.Timestamp(f["train_end"]))
        ]
        te = panel_dates[
            (panel_dates >= pd.Timestamp(f["test_start"]))
            & (panel_dates <= pd.Timestamp(f["test_end"]))
        ]
        yield int(f["fold"]), ts, te


def trading_dates_from_panel(panel: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(np.sort(panel.index.get_level_values("date").unique()))

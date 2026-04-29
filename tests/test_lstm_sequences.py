"""Sequence alignment tests for the LSTM data pipeline.

The LSTM used to pair a sequence ending at date ``t`` with the target at
the same row, which (once the target convention changed) meant the
sequence's last element was the prediction date — a one-day leak. The
fix: end the sequence at row ``t-1`` and pair it with the target at
row ``t``. These tests check that.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not installed; skipping LSTM tests", allow_module_level=True)

from rvforecast.models.lstm import _build_sequences  # noqa: E402


def _make_panel(n: int = 50, ticker: str = "AAA") -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n)
    idx = pd.MultiIndex.from_product([dates, [ticker]], names=["date", "ticker"])
    rng = np.random.default_rng(0)
    cols = ["log_rv_1d", "ret_1d", "log_hl", "log_co", "vix", "term_spread"]
    data = rng.normal(size=(n, len(cols)))
    return pd.DataFrame(data, index=idx, columns=cols)


def test_sequence_does_not_include_prediction_date():
    """The last row of a sequence must precede the row whose target it predicts."""
    raw = _make_panel(n=40)
    target = pd.Series(np.arange(len(raw), dtype=float), index=raw.index, name="y_target")
    seq_len = 5
    X, y, idx = _build_sequences(raw, target, seq_len=seq_len)

    # The first sequence covers raw rows 0..seq_len-1 inclusive and predicts
    # the target for raw row seq_len. ``y[0]`` must equal target at row
    # seq_len, NOT seq_len-1.
    assert y[0] == pytest.approx(seq_len)
    assert idx[0] == raw.index[seq_len]

    # The last element of the first sequence is raw row seq_len-1.
    np.testing.assert_allclose(X[0, -1, 0], raw.iloc[seq_len - 1, 0])


def test_sequence_count_matches_n_minus_seq_len():
    raw = _make_panel(n=40)
    target = pd.Series(np.zeros(len(raw)), index=raw.index, name="y_target")
    seq_len = 7
    X, y, _ = _build_sequences(raw, target, seq_len=seq_len)
    assert len(X) == len(raw) - seq_len
    assert len(y) == len(raw) - seq_len

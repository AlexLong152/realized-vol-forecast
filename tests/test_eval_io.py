"""Tests for the evaluation I/O alignment.

The check that matters most in :mod:`rvforecast.evaluation._io` is the
``y_true`` agreement check across prediction parquets. If a stale
parquet from an older feature matrix sneaks in, every aggregated metric
gets quietly polluted — so the check needs to fire on real disagreements
and *not* fire on the float noise that parquet round-trips introduce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from rvforecast.evaluation._io import align_outer, load_predictions


def _frame(values: dict[str, list[float]], dates: list[str], ticker: str = "AAA") -> pd.DataFrame:
    idx = pd.MultiIndex.from_product([pd.to_datetime(dates), [ticker]], names=["date", "ticker"])
    return pd.DataFrame(values, index=idx)


def test_align_outer_combines_disjoint_models():
    a = _frame({"y_true": [0.1, 0.2], "y_pred": [0.11, 0.21]}, ["2020-01-02", "2020-01-03"])
    b = _frame({"y_true": [0.3], "y_pred": [0.31]}, ["2020-01-06"])
    merged = align_outer({"a": a, "b": b})
    assert {"y_true", "a", "b"} <= set(merged.columns)
    # All three rows survive the outer join; y_true pulled from each frame.
    assert merged["y_true"].notna().all()
    # Each model contributes a prediction only on its own row.
    assert merged.loc[merged.index[:2], "a"].notna().all()
    assert merged.loc[merged.index[:2], "b"].isna().all()
    assert merged.loc[merged.index[-1:], "b"].notna().all()


def test_align_outer_tolerates_floating_point_noise():
    """Parquet round-trip can introduce ~1e-15 float noise; that must not raise."""
    dates = ["2020-01-02", "2020-01-03"]
    a = _frame({"y_true": [0.1, 0.2], "y_pred": [0.11, 0.21]}, dates)
    b = _frame({"y_true": [0.1 + 1e-15, 0.2 - 1e-15], "y_pred": [0.12, 0.22]}, dates)
    merged = align_outer({"a": a, "b": b})
    np.testing.assert_allclose(merged["y_true"].to_numpy(), [0.1, 0.2], rtol=1e-12)


def test_align_outer_raises_on_y_true_mismatch():
    """A real disagreement in ``y_true`` must raise, not silently overwrite.

    This catches the stale-parquet case (predictions generated from an
    earlier feature matrix). The threshold sits well above parquet
    round-trip noise (1e-9 rtol / 1e-12 atol).
    """
    dates = ["2020-01-02", "2020-01-03"]
    a = _frame({"y_true": [0.10, 0.20], "y_pred": [0.11, 0.21]}, dates)
    b = _frame({"y_true": [0.15, 0.20], "y_pred": [0.12, 0.22]}, dates)
    with pytest.raises(ValueError, match="y_true mismatch"):
        align_outer({"a": a, "b": b})


def test_align_outer_empty_input_returns_empty_frame():
    assert align_outer({}).empty


def test_load_predictions_strips_only_trailing_holdout_suffix(tmp_path):
    """``load_predictions`` keys frames by the parquet stem with a trailing
    ``_holdout`` removed. ``str.removesuffix`` (not ``str.replace``) means
    a stem like ``foo_holdout_bar`` keeps its mid-string occurrence intact."""
    df = pd.DataFrame(
        {"y_true": [0.0], "y_pred": [0.0]},
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2020-01-02"), "AAA")], names=["date", "ticker"]
        ),
    )
    df.to_parquet(tmp_path / "har_holdout.parquet")
    df.to_parquet(tmp_path / "lstm.parquet")
    df.to_parquet(tmp_path / "weird_holdout_inner.parquet")
    out = load_predictions(tmp_path)
    assert set(out.keys()) == {"har", "lstm", "weird_holdout_inner"}

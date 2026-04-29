"""Look-ahead bias tests for the feature builder.

Two things to check:

1. A huge spike injected at row ``t`` must not move any feature at an
   earlier row. Every feature on row ``t`` is supposed to depend only on
   data strictly before ``t``, so this is just stating that rule.

2. Perturbing the OHLC of day ``t`` itself must not move any feature at
   row ``t``. This is the harder bug to spot: a same-day measurement
   (today's close-to-close return, today's range) sneaks into the
   feature matrix and gets consumed verbatim by a tabular model, which
   then "predicts" a target that's a function of the same day's OHLC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rvforecast.features.build_features import build_feature_matrix


def _synthetic_panel(n: int = 400, tickers: tuple[str, ...] = ("AAA", "BBB")) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2018-01-01", periods=n)
    rows = []
    for t in tickers:
        open_ = 100 + rng.normal(0, 0.5, n).cumsum()
        intraday = rng.normal(0, 0.5, n)
        close = open_ + intraday
        high = np.maximum.reduce([open_, close]) + np.abs(rng.normal(0.5, 0.2, n))
        low = np.minimum.reduce([open_, close]) - np.abs(rng.normal(0.5, 0.2, n))
        d = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "adj_close": close,
                "volume": 1_000_000,
                "ticker": t,
            },
            index=pd.Index(dates, name="date"),
        )
        rows.append(d.reset_index())
    long = pd.concat(rows, ignore_index=True).set_index(["date", "ticker"]).sort_index()
    return long


def _synthetic_macro(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "vix": 15 + rng.normal(0, 1, len(dates)).cumsum() * 0.05,
            "DGS10": 2.5 + rng.normal(0, 0.01, len(dates)).cumsum(),
            "DGS2": 1.5 + rng.normal(0, 0.01, len(dates)).cumsum(),
            "term_spread": np.full(len(dates), 1.0),
        },
        index=pd.Index(dates, name="date"),
    )


def _sector_map(tickers: tuple[str, ...]) -> pd.Series:
    return pd.Series({t: "Test" for t in tickers}, name="sector")


def test_no_look_ahead_through_future_spike():
    panel = _synthetic_panel()
    dates = panel.index.get_level_values("date").unique().sort_values()
    macro = _synthetic_macro(dates)
    sector = _sector_map(("AAA", "BBB"))

    baseline = build_feature_matrix(panel.copy(), macro, sector)

    # Inject a future spike on the second-to-last date for ticker AAA.
    spike_date = dates[-2]
    perturbed = panel.copy()
    perturbed.loc[(spike_date, "AAA"), ["high", "close"]] = 1e6

    perturbed_features = build_feature_matrix(perturbed, macro, sector)

    earlier_dates = dates[:-3]
    feature_cols = [c for c in baseline.columns if c not in {"y_target", "sector"}]
    a = baseline.loc[(earlier_dates, "AAA"), feature_cols]
    b = perturbed_features.loc[(earlier_dates, "AAA"), feature_cols]
    common = a.index.intersection(b.index)
    diff = (a.loc[common].astype(float) - b.loc[common].astype(float)).abs().max().max()
    assert diff < 1e-9, f"Future spike leaked into earlier rows; max abs diff = {diff}"


def test_no_same_day_leakage_through_perturbed_ohlc():
    """Perturbing day ``t``'s OHLC must not move any feature at row ``t``.

    This catches the same-day-leakage class of bug: a column that is a
    measurement of day ``t`` (today's return, today's range, today's GK vol)
    must not survive into the feature matrix as an input, because the
    target on row ``t`` is itself a function of day ``t``'s OHLC.
    """
    n = 200
    panel = _synthetic_panel(n=n)
    dates = panel.index.get_level_values("date").unique().sort_values()
    macro = _synthetic_macro(dates)
    sector = _sector_map(("AAA", "BBB"))

    baseline = build_feature_matrix(panel.copy(), macro, sector)

    # Pick a date deep in the panel so HAR lags etc. are populated. Perturb
    # AAA's OHLC by a large multiplicative factor on that date only.
    target_date = dates[150]
    perturbed = panel.copy()
    perturbed.loc[(target_date, "AAA"), ["open", "high", "low", "close", "adj_close"]] = (
        perturbed.loc[(target_date, "AAA"), ["open", "high", "low", "close", "adj_close"]].astype(
            float
        )
        * 3.0
    )

    perturbed_features = build_feature_matrix(perturbed, macro, sector)

    feature_cols = [c for c in baseline.columns if c not in {"y_target", "sector"}]
    if (target_date, "AAA") not in baseline.index:
        return  # nothing to check
    a = baseline.loc[(target_date, "AAA"), feature_cols].astype(float)
    b = perturbed_features.loc[(target_date, "AAA"), feature_cols].astype(float)
    diff = (a - b).abs()
    assert diff.max() < 1e-9, (
        f"Same-day OHLC perturbation leaked into features at row t; "
        f"largest diff in column {diff.idxmax()} = {diff.max()}"
    )


def test_target_is_log_vol_for_row_date():
    """y_target on row t equals log_rv_1d at date t.

    Combined with the strictly-past features, this is a one-day-ahead
    forecast: features through t-1 predict day t. The earlier convention
    used ``shift(-1)`` on the target, which (combined with the same
    ``shift(1)`` on features) implicitly made every model predict two days
    ahead.
    """
    panel = _synthetic_panel(n=120)
    dates = panel.index.get_level_values("date").unique().sort_values()
    macro = _synthetic_macro(dates)
    sector = _sector_map(("AAA", "BBB"))
    feats = build_feature_matrix(panel, macro, sector)
    assert "y_target" in feats.columns
    assert feats["y_target"].notna().all()

    # Recompute log_rv_1d directly and confirm the target matches it row-for-row.
    open_ = panel["open"]
    high = panel["high"]
    low = panel["low"]
    close = panel["close"]
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    daily_var = (0.5 * log_hl**2 - (2.0 * np.log(2.0) - 1.0) * log_co**2).clip(lower=0.0)
    expected_log_rv = np.log(np.sqrt(daily_var * 252).clip(lower=1e-8))
    common = feats.index.intersection(expected_log_rv.index)
    assert np.allclose(
        feats.loc[common, "y_target"].to_numpy(),
        expected_log_rv.loc[common].to_numpy(),
    )


def test_macro_ffill_capped_at_five_business_days():
    """A multi-week macro outage must not get filled across every row.

    Routine weekend/holiday gaps (up to 5 business days) get absorbed.
    Longer outages stay NaN so they show up in the feature matrix
    instead of being papered over with a stale value. An earlier version
    of ``_add_macro`` did an unbounded ``ffill()`` that quietly undid the
    5-day cap from ``fetch_macro``; this test catches that regression.
    """
    n = 200
    panel = _synthetic_panel(n=n)
    dates = panel.index.get_level_values("date").unique().sort_values()
    macro = _synthetic_macro(dates)

    # Knock out a 12-business-day stretch in the middle of VIX (and term
    # spread) — well past the 5-day cap.
    gap_start, gap_end = 100, 112
    macro = macro.copy()
    macro.iloc[gap_start:gap_end, macro.columns.get_indexer(["vix", "term_spread"])] = np.nan

    sector = _sector_map(("AAA", "BBB"))
    feats = build_feature_matrix(panel, macro, sector)

    # The lagged VIX column at rows deep in the outage (e.g. 6+ business
    # days past gap_start) must be NaN — the 5-day cap exhausted itself.
    deep_dates = dates[gap_start + 7 : gap_end]
    for d in deep_dates:
        if (d, "AAA") not in feats.index:
            continue
        assert pd.isna(feats.loc[(d, "AAA"), "vix_lag"]), (
            f"vix_lag at {d} should be NaN past the 5-day ffill cap; "
            "the unbounded ffill regression has reappeared."
        )


def test_contemporaneous_return_is_not_a_feature():
    """The unshifted same-day return must not survive into the feature matrix.

    Lagged variants (``ret_lag_*d``, ``abs_ret_lag_*d``) are valid features
    because they are explicitly shifted; the raw same-day value is not.
    """
    panel = _synthetic_panel(n=120)
    dates = panel.index.get_level_values("date").unique().sort_values()
    macro = _synthetic_macro(dates)
    sector = _sector_map(("AAA", "BBB"))
    feats = build_feature_matrix(panel, macro, sector)

    # No same-day return columns; lagged ones are fine.
    assert "ret_1d" not in feats.columns
    assert "_ret_1d_raw" not in feats.columns
    assert any(c.startswith("ret_lag_") for c in feats.columns)
    assert any(c.startswith("abs_ret_lag_") for c in feats.columns)

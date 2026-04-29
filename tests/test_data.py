"""Tests for data ingestion correctness.

Two layers, both runnable offline:

1. OHLC structural invariants on synthetic data (high ≥ low, etc.). These
   pin the shape of the data the rest of the pipeline assumes.
2. Direct unit tests of the data-layer helpers (``load_universe``,
   ``_cache_covers``, manifest read/write). The previous version of this
   file exercised only (1) and never imported anything from
   :mod:`rvforecast.data`, so the cache-coverage logic — the most
   easily-broken part of the layer — went uncovered.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from rvforecast.data.fetch_prices import (
    _cache_covers,
    _read_manifest,
    _write_manifest,
    load_universe,
)


def _synthetic_ohlc(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    open_ = 100 + rng.normal(0, 0.5, n).cumsum()
    intraday = rng.normal(0, 0.5, n)
    high = open_ + np.abs(rng.normal(1.0, 0.3, n))
    low = open_ - np.abs(rng.normal(1.0, 0.3, n))
    close = open_ + intraday
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "adj_close": close},
        index=pd.Index(dates, name="date"),
    )


def test_ohlc_invariants_no_nans():
    df = _synthetic_ohlc()
    assert not df[["open", "high", "low", "close"]].isna().any().any()


def test_high_geq_low():
    df = _synthetic_ohlc()
    assert (df["high"] >= df["low"]).all()


def test_high_bounds_open_close():
    df = _synthetic_ohlc()
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()


def test_adjusted_close_monotone_relation_under_split():
    """A 2:1 split should leave adj_close at the same level while close halves.

    The relation ``adj_close / close`` is non-decreasing (in the absence of
    reverse splits) when we walk forward in time across split events. We test
    the simpler one-event invariant: after the split, the ratio jumps up but
    never down within a no-reverse-split sample.
    """
    df = _synthetic_ohlc(n=10)
    close = df["close"].copy()
    adj = close.copy()
    close.iloc[5:] = close.iloc[5:] / 2.0  # post-split observed close
    ratio = (adj / close).to_numpy()
    diffs = np.diff(ratio)
    assert np.all(diffs >= -1e-12), "adj/close ratio must be non-decreasing without reverse splits"


# ---------------------------------------------------------------------------
# Data-layer helpers
# ---------------------------------------------------------------------------


def test_load_universe_deduplicates_and_strips(tmp_path):
    """Repeated tickers must collapse to one; comments and blank lines drop."""
    p = tmp_path / "u.txt"
    p.write_text("AAPL\n# comment\nMSFT\nAAPL\n  \nMSFT\nNVDA\n")
    assert load_universe(p) == ["AAPL", "MSFT", "NVDA"]


def test_cache_covers_none_entry_is_never_covered():
    assert _cache_covers(None, "2020-01-01", "2020-12-31") is False


def test_cache_covers_wider_range_covers_narrower():
    entry = {"start": "2010-01-01", "end": "2020-12-31"}
    assert _cache_covers(entry, "2015-01-01", "2018-12-31") is True


def test_cache_covers_narrower_range_does_not_cover_wider():
    entry = {"start": "2015-01-01", "end": "2018-12-31"}
    assert _cache_covers(entry, "2010-01-01", "2020-12-31") is False


def test_cache_covers_open_ended_request_requires_open_ended_cache():
    """``end=None`` means 'today'; only an open-ended cache covers it.

    A cached range with a fixed ``end`` is by construction stale relative
    to a request with ``end=None``, so the splitter must refetch.
    """
    fixed_end = {"start": "2010-01-01", "end": "2020-12-31"}
    open_end = {"start": "2010-01-01", "end": None}
    assert _cache_covers(fixed_end, "2015-01-01", None) is False
    assert _cache_covers(open_end, "2015-01-01", None) is True


def test_cache_covers_open_ended_cache_covers_any_fixed_end():
    """An open-ended cache is wider than any fixed-end request."""
    entry = {"start": "2010-01-01", "end": None}
    assert _cache_covers(entry, "2015-01-01", "2018-12-31") is True


def test_manifest_round_trip(tmp_path):
    payload = {"AAPL": {"start": "2010-01-01", "end": "2020-12-31"}}
    _write_manifest(tmp_path, payload)
    assert _read_manifest(tmp_path) == payload


def test_manifest_missing_returns_empty_dict(tmp_path):
    assert _read_manifest(tmp_path) == {}


def test_manifest_corrupt_returns_empty_dict(tmp_path):
    """Corrupt manifest must not crash the pipeline; refetch from scratch.

    The contract is documented on ``_read_manifest``: a JSON parse error
    or read error is treated like a missing manifest, so the next
    ``fetch_prices`` call simply re-validates against on-disk parquets.
    """
    (tmp_path / "_manifest.json").write_text("{not valid json")
    assert _read_manifest(tmp_path) == {}


def test_manifest_write_creates_directory(tmp_path):
    nested = tmp_path / "nested" / "dir"
    _write_manifest(nested, {"AAPL": {}})
    assert (nested / "_manifest.json").exists()
    with open(nested / "_manifest.json") as f:
        assert json.load(f) == {"AAPL": {}}

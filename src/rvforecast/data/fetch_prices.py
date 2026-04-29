"""Pull daily OHLCV from Yahoo Finance and cache to parquet.

Both adjusted and unadjusted prices are stored. Adjusted close is appropriate for
return computation; unadjusted high/low/open/close is required by range-based
volatility estimators (Parkinson, Garman-Klass, Rogers-Satchell), since
adjusted prices distort the intraday range across split and dividend events.

Survivorship bias note
----------------------
The default universe is a static list of currently liquid S&P 500 names. Because
delisted tickers are excluded, in-sample volatility is biased downward relative
to the true historical universe, and any trading-strategy results inherit this
bias. Free data sources do not provide a clean point-in-time membership history
of the index. The bias direction is documented in the README and discussed
explicitly in the "What Didn't Work" section.

Caching strategy
----------------
Each ticker's parquet is keyed in a manifest by the ``(start, end)`` of the
range that fetched it, plus a ``fetched_at`` UTC timestamp. A subsequent
call with a wider range refetches; a call with a narrower range re-uses the
cache and slices in memory. Caches are not auto-refreshed on staleness:
delete ``_manifest.json`` (or the ticker's parquet) to force a refetch. The
``fetched_at`` field is recorded so callers needing freshness can inspect
the manifest themselves.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError

import pandas as pd
import yfinance as yf

from rvforecast.config import DATA_RAW, END_DATE, ROOT, START_DATE, ensure_output_dirs

_MANIFEST_NAME = "_manifest.json"
# ``requests`` ships transitively with yfinance; importing here keeps the
# transient-exception surface explicit without making it a hard dependency
# of this module if a future yfinance version drops it.
try:
    import requests as _requests
except ImportError:  # pragma: no cover - defensive
    _requests = None  # type: ignore[assignment]

_REQUESTS_EXC = (
    getattr(getattr(_requests, "exceptions", None), "RequestException", None)
    if _requests is not None
    else None
)
_TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = tuple(
    cls for cls in (_REQUESTS_EXC, URLError, ConnectionError, TimeoutError) if isinstance(cls, type)
)


def load_universe(universe_file: Path | None = None) -> list[str]:
    """Load tickers from a one-per-line text file, deduplicating in order.

    Duplicates in the universe file would otherwise produce duplicate rows
    in the long price panel and silently double-count downstream
    per-ticker aggregations.
    """
    path = universe_file or (ROOT / "configs" / "universe_sp50.txt")
    with open(path) as f:
        raw = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return list(dict.fromkeys(raw))


def _read_manifest(out_dir: Path) -> dict:
    manifest_path = out_dir / _MANIFEST_NAME
    if not manifest_path.exists():
        return {}
    try:
        with open(manifest_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _write_manifest(out_dir: Path, manifest: dict) -> None:
    manifest_path = out_dir / _MANIFEST_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def _cache_covers(entry: dict | None, start: str, end: str | None) -> bool:
    """Return True if a cached range covers the requested ``[start, end]``."""
    if entry is None:
        return False
    cached_start = entry.get("start")
    cached_end = entry.get("end")
    if cached_start is None or pd.Timestamp(cached_start) > pd.Timestamp(start):
        return False
    if end is None:
        return cached_end is None
    if cached_end is None:
        return True
    return pd.Timestamp(cached_end) >= pd.Timestamp(end)


def _download_one(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """Download a single ticker once with ``auto_adjust=False``.

    yfinance returns both ``Close`` and ``Adj Close`` in this mode; the
    earlier code did two separate downloads (one with ``auto_adjust=True``
    and one without) just to get the adjusted close, doubling Yahoo API
    pressure for no benefit. We rename the lower-cased ``adj close``
    column to ``adj_close`` to keep downstream snake_case.

    Specific transient exceptions (network errors and yfinance rate-limit
    surfaces) are retried with linear backoff; everything else propagates
    so unexpected failures aren't silently swallowed.
    """
    last_err: BaseException | None = None
    yfinance_transient: tuple[type[BaseException], ...] = ()
    rate_limit = getattr(getattr(yf, "exceptions", None), "YFRateLimitError", None)
    if rate_limit is not None:
        yfinance_transient = (rate_limit,)
    transient = _TRANSIENT_EXCEPTIONS + yfinance_transient
    for attempt in range(3):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
            df.index.name = "date"
            return df
        except transient as exc:
            last_err = exc
            time.sleep(1 + attempt)
    raise RuntimeError(f"Failed to download {ticker}: {last_err}")


def fetch_prices(
    tickers: list[str],
    start: str = START_DATE,
    end: str | None = END_DATE,
    out_dir: Path = DATA_RAW,
) -> pd.DataFrame:
    """Fetch raw and adjusted OHLCV. Returns a long-format DataFrame.

    Parameters
    ----------
    tickers : list of str
        Yahoo Finance tickers (e.g., ``"AAPL"``, ``"BRK-B"``).
    start, end : str or None
        ISO dates. ``end=None`` means today.
    out_dir : Path
        Cache location. Each ticker is written to ``<out_dir>/<TICKER>.parquet``;
        the date range that fetched it is recorded in
        ``<out_dir>/_manifest.json``.

    Returns
    -------
    DataFrame indexed by ``(date, ticker)`` with columns ``open, high, low,
    close, adj_close, volume``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(out_dir)
    rows: list[pd.DataFrame] = []
    for ticker in tickers:
        cache = out_dir / f"{ticker}.parquet"
        entry = manifest.get(ticker)
        if cache.exists() and _cache_covers(entry, start, end):
            df = pd.read_parquet(cache)
        else:
            raw = _download_one(ticker, start, end)
            if raw.empty:
                continue
            df = raw[["open", "high", "low", "close", "adj_close", "volume"]].copy()
            df = df.dropna(subset=["open", "high", "low", "close", "adj_close"])
            df.to_parquet(cache)
            manifest[ticker] = {
                "start": start,
                "end": end,
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            _write_manifest(out_dir, manifest)

        # Slice to the requested window in memory; the on-disk cache may
        # be wider when the manifest indicates a broader prior fetch.
        idx = df.index
        mask = idx >= pd.Timestamp(start)
        if end is not None:
            mask &= idx < pd.Timestamp(end)
        df = df.loc[mask].copy()
        df["ticker"] = ticker
        rows.append(df.reset_index())
    if not rows:
        raise RuntimeError("No data fetched. Check network or universe file.")
    long = pd.concat(rows, ignore_index=True)
    long = long.set_index(["date", "ticker"]).sort_index()
    if not long.index.is_unique:
        raise RuntimeError(
            "Duplicate (date, ticker) rows in the long panel; check the universe "
            "file for repeated tickers."
        )
    return long


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--universe", type=Path, default=None)
    args = parser.parse_args()

    ensure_output_dirs()
    tickers = load_universe(args.universe)
    df = fetch_prices(tickers, start=args.start, end=args.end)
    out = DATA_RAW / "prices_long.parquet"
    df.to_parquet(out)
    print(
        f"Wrote {len(df):,} rows for {df.index.get_level_values('ticker').nunique()} tickers to {out}"
    )


if __name__ == "__main__":
    main()

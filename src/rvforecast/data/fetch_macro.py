"""Pull macro series: VIX (Yahoo) and Treasury yields (FRED).

The term spread is just ``DGS10 - DGS2``. Series get forward-filled with
a one-business-week cap (``ffill(limit=5)``) and then aligned to the
equity panel's trading-day index before merging. The cap covers normal
weekend and holiday gaps. Anything longer stays NaN so it shows up in
the feature matrix instead of being filled with a stale value.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from urllib.error import URLError

import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

from rvforecast.config import DATA_RAW, END_DATE, START_DATE, ensure_output_dirs

FRED_SERIES: tuple[str, ...] = ("DGS10", "DGS2")

_TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (URLError, ConnectionError, TimeoutError)


def fetch_vix(start: str = START_DATE, end: str | None = END_DATE) -> pd.Series:
    """Daily VIX close from yfinance.

    Retries with linear backoff on the same transient exceptions
    :mod:`rvforecast.data.fetch_prices` retries on, so a single network
    blip during macro pull does not crash the whole pipeline. Other
    failures propagate.
    """
    rate_limit = getattr(getattr(yf, "exceptions", None), "YFRateLimitError", None)
    transient = _TRANSIENT_EXCEPTIONS + ((rate_limit,) if rate_limit is not None else ())
    last_err: BaseException | None = None
    for attempt in range(3):
        try:
            df = yf.download("^VIX", start=start, end=end, auto_adjust=False, progress=False)
            break
        except transient as exc:  # type: ignore[misc]
            last_err = exc
            time.sleep(1 + attempt)
    else:
        raise RuntimeError(f"Failed to download ^VIX: {last_err}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].rename("vix")
    s.index.name = "date"
    return s


def fetch_fred(
    series: tuple[str, ...] = FRED_SERIES, start: str = START_DATE, end: str | None = END_DATE
) -> pd.DataFrame:
    df = pdr.DataReader(list(series), "fred", start=start, end=end)
    df.index.name = "date"
    return df


def build_macro(start: str = START_DATE, end: str | None = END_DATE) -> pd.DataFrame:
    vix = fetch_vix(start, end)
    fred = fetch_fred(start=start, end=end)
    macro = pd.concat([vix, fred], axis=1).sort_index()
    # Cap forward-fill at one business week. Weekend/holiday gaps get
    # absorbed; multi-week outages stay NaN so they show up downstream
    # instead of being filled with a stale value.
    macro = macro.ffill(limit=5)
    macro["term_spread"] = macro["DGS10"] - macro["DGS2"]
    return macro


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--out", type=Path, default=DATA_RAW / "macro.parquet")
    args = parser.parse_args()

    ensure_output_dirs()
    macro = build_macro(args.start, args.end)
    macro.to_parquet(args.out)
    print(f"Wrote {len(macro):,} rows of macro data to {args.out}")


if __name__ == "__main__":
    main()

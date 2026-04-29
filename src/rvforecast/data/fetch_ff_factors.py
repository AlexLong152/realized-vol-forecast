"""Optional: pull Fama-French daily factors for sanity checks.

Not required by the volatility pipeline. Provided so a reviewer can quickly
confirm the equity panel covers a familiar return space, and so factor
regressions can be added without restructuring data ingestion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas_datareader import data as pdr

from rvforecast.config import DATA_RAW, END_DATE, START_DATE, ensure_output_dirs


def fetch_ff3(start: str = START_DATE, end: str | None = END_DATE) -> pd.DataFrame:
    df = pdr.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start=start, end=end)[0]
    df.index.name = "date"
    df = df.div(100.0)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--out", type=Path, default=DATA_RAW / "ff3_daily.parquet")
    args = parser.parse_args()

    ensure_output_dirs()
    ff = fetch_ff3(args.start, args.end)
    ff.to_parquet(args.out)
    print(f"Wrote {len(ff):,} rows of FF3 daily factors to {args.out}")


if __name__ == "__main__":
    main()

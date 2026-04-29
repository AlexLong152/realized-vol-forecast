"""Shared loaders for evaluation and plotting.

Both ``run_eval`` and ``plots`` need to load every model's prediction parquet
from a directory and align them. The alignment used previously was an inner
join on ``y_true`` plus ``y_pred`` per model, which dropped rows whenever
*any* model was missing a prediction (notably GARCH cold-start tickers).
That biased every model's metric to the GARCH-valid intersection.

Here we instead load each model's predictions independently and let callers
slice to the per-model valid rows for single-model metrics, and to pairwise
intersections for r2_oos / Diebold-Mariano comparisons.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_predictions(directory: Path) -> dict[str, pd.DataFrame]:
    """Load every parquet under ``directory`` keyed by model name.

    The trailing ``_holdout`` suffix is stripped so walk-forward and
    holdout parquets share the same canonical model name. Using
    ``str.removesuffix`` rather than ``str.replace`` means a model name
    that happens to contain ``_holdout`` mid-string (none today, but
    defensible) is not mangled.
    """
    out: dict[str, pd.DataFrame] = {}
    for parquet in sorted(directory.glob("*.parquet")):
        name = parquet.stem.removesuffix("_holdout")
        df = pd.read_parquet(parquet)
        out[name] = df
    return out


def align_outer(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Outer join all models on ``(date, ticker)``.

    Each model contributes a column named after the model holding its
    ``y_pred``. ``y_true`` is taken from the first model and asserted to
    match across models on overlapping rows; a mismatch raises
    ``ValueError`` so a stale parquet (e.g., generated against an earlier
    feature matrix) does not silently skew downstream metrics.
    """
    if not preds:
        return pd.DataFrame()
    items = list(preds.items())
    first_name, first_df = items[0]
    merged = first_df[["y_true", "y_pred"]].rename(columns={"y_pred": first_name})
    for name, df in items[1:]:
        # Assert y_true agreement on the index intersection where both
        # frames have a non-NaN truth label. Using ``np.allclose`` is
        # appropriate because parquet round-trip can introduce trivial
        # float noise; a real mismatch is many orders of magnitude bigger.
        common = merged.index.intersection(df.index)
        if len(common):
            yt_existing = merged.loc[common, "y_true"]
            yt_new = df.loc[common, "y_true"]
            both_present = yt_existing.notna() & yt_new.notna()
            if both_present.any():
                a = yt_existing[both_present].to_numpy(dtype=float)
                b = yt_new[both_present].to_numpy(dtype=float)
                if not np.allclose(a, b, rtol=1e-9, atol=1e-12):
                    raise ValueError(
                        f"y_true mismatch between '{first_name}' and '{name}' "
                        f"on {int(both_present.sum())} overlapping rows; "
                        f"max abs diff = {float(np.abs(a - b).max())}. "
                        "Regenerate predictions against a single feature matrix."
                    )
        merged = merged.join(df[["y_pred"]].rename(columns={"y_pred": name}), how="outer")
        # Fill in y_true from later frames where the first frame did not cover.
        if merged["y_true"].isna().any():
            merged["y_true"] = merged["y_true"].fillna(df["y_true"])
    return merged

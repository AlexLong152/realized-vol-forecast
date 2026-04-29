"""Aggregate predictions, compute metrics, and write summary tables.

Reads every parquet file in ``results/predictions/`` (or ``results/holdout/``
under ``--holdout``), computes QLIKE, MSE in log space, and OOS R-squared
versus HAR, and writes ``results/tables/metrics.csv`` and the pairwise
Diebold-Mariano p-value matrix to ``results/tables/dm_pvalues.csv``.

Single-model metrics are reported in two forms: ``qlike_own`` is computed
on the model's own valid rows and ``qlike_intersection`` is computed on
the intersection of every loaded model's valid rows. The intersection
metric is the right one for ranking because it controls for sample-size
asymmetry between models (notably GARCH cold-start NaNs); the per-model
metric is preserved as a fairness check.

The DM p-value matrix and ``r2_oos_vs_<baseline>`` use the pairwise
intersection of valid rows, as before.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from rvforecast.config import (
    RESULTS_HOLDOUT,
    RESULTS_PREDICTIONS,
    RESULTS_TABLES,
    ensure_output_dirs,
)
from rvforecast.evaluation._io import align_outer, load_predictions
from rvforecast.evaluation.metrics import (
    diebold_mariano,
    mse_log,
    qlike,
    r2_oos,
    squared_errors_log,
)

BASELINE = "har"


def _intersection_mask(merged: pd.DataFrame, models: list[str]) -> pd.Series:
    """Boolean mask of rows where ``y_true`` and every model are non-NaN."""
    cols = ["y_true", *models]
    return merged[cols].notna().all(axis=1)


def _metrics_table(
    merged: pd.DataFrame,
    models: list[str],
    baseline: str | None,
) -> pd.DataFrame:
    intersection = _intersection_mask(merged, models)
    intersection_n = int(intersection.sum())
    r2_col = f"r2_oos_vs_{baseline}" if baseline else "r2_oos"
    rows = []
    for m in models:
        sub = merged[["y_true", m]].dropna()
        n_obs = int(len(sub))
        if n_obs == 0:
            rows.append(
                {
                    "model": m,
                    "qlike_own": np.nan,
                    "qlike_intersection": np.nan,
                    "mse_log": np.nan,
                    r2_col: np.nan,
                    "n_obs_own": 0,
                    "n_obs_intersection": intersection_n,
                }
            )
            continue
        yt_log = sub["y_true"].to_numpy()
        yp_log = sub[m].to_numpy()
        yt_var = np.exp(2.0 * yt_log)
        yp_var = np.exp(2.0 * yp_log)

        if intersection_n > 0:
            inter = merged.loc[intersection, ["y_true", m]]
            qlike_inter = qlike(
                np.exp(2.0 * inter["y_true"].to_numpy()),
                np.exp(2.0 * inter[m].to_numpy()),
            )
        else:
            qlike_inter = float("nan")

        if baseline is not None and m != baseline:
            pair = merged[["y_true", baseline, m]].dropna()
            r2 = r2_oos(
                pair["y_true"].to_numpy(),
                pair[m].to_numpy(),
                pair[baseline].to_numpy(),
            )
        else:
            r2 = float("nan")
        rows.append(
            {
                "model": m,
                "qlike_own": qlike(yt_var, yp_var),
                "qlike_intersection": qlike_inter,
                "mse_log": mse_log(yt_log, yp_log),
                r2_col: r2,
                "n_obs_own": n_obs,
                "n_obs_intersection": intersection_n,
            }
        )
    return pd.DataFrame(rows).sort_values("qlike_intersection", na_position="last")


def _dm_matrix(merged: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    p = pd.DataFrame(np.nan, index=models, columns=models, dtype=float)
    for i, a in enumerate(models):
        for j in range(i + 1, len(models)):
            b = models[j]
            sub = merged[["y_true", a, b]].dropna()
            if sub.empty:
                continue
            err_a = squared_errors_log(sub["y_true"], sub[a]).to_numpy()
            err_b = squared_errors_log(sub["y_true"], sub[b]).to_numpy()
            _, pval = diebold_mariano(err_a, err_b, h=1)
            # The DM two-sided p-value is symmetric in (a, b).
            p.loc[a, b] = pval
            p.loc[b, a] = pval
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="Evaluate the one-shot holdout instead of walk-forward.",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    src_dir = RESULTS_HOLDOUT if args.holdout else RESULTS_PREDICTIONS
    out_metrics = RESULTS_TABLES / ("metrics_holdout.csv" if args.holdout else "metrics.csv")
    out_dm = RESULTS_TABLES / ("dm_pvalues_holdout.csv" if args.holdout else "dm_pvalues.csv")

    preds = load_predictions(src_dir)
    if not preds:
        raise SystemExit(f"No predictions found in {src_dir}")
    merged = align_outer(preds)
    models = list(preds.keys())

    if BASELINE in models:
        baseline = BASELINE
    else:
        baseline = None
        print(
            f"warning: baseline '{BASELINE}' not in {sorted(models)}; "
            "r2_oos column will be NaN for every model.",
            file=sys.stderr,
        )

    metrics = _metrics_table(merged, models, baseline)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_metrics, index=False)

    dm = _dm_matrix(merged, models)
    dm.to_csv(out_dm)

    print(metrics.to_string(index=False))
    print(f"\nMetrics: {out_metrics}\nDiebold-Mariano p-values: {out_dm}")


if __name__ == "__main__":
    main()

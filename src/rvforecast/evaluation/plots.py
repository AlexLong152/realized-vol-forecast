"""Diagnostic figures for the evaluation report.

Reads aligned predictions from ``results/predictions/`` and writes:

- ``rolling_qlike.png``: 60-day rolling QLIKE per model with recession shading.
- ``pred_vs_actual_scatter.png``: scatter of predicted vs actual log vol.
- ``residuals_over_time.png``: residual time series per model with high-VIX
  shading. The high-VIX threshold is the 90th percentile of the *full*
  sample VIX, so it is computed post-hoc and is intended only as a visual
  reference.
- ``per_ticker_metrics.png``: bar chart of QLIKE per ticker for the best model.
- ``feature_importance.png`` is produced by the LightGBM trainer.

Each plot uses the rows where it has the data it needs: per-model curves
fall back on each model's own valid rows, while plots requiring multiple
models simultaneously (none currently) would intersect those models' valid
rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rvforecast.config import (
    DATA_RAW,
    RESULTS_FIGURES,
    RESULTS_HOLDOUT,
    RESULTS_PREDICTIONS,
    SEED,
    ensure_output_dirs,
)
from rvforecast.evaluation._io import align_outer, load_predictions
from rvforecast.evaluation.metrics import qlike
from rvforecast.evaluation.plot_style import PALETTE, RECESSIONS, apply_style


def _qlike_at_date(df: pd.DataFrame, model: str) -> pd.Series:
    sub = df[["y_true", model]].dropna()
    yt_var = np.exp(2.0 * sub["y_true"])
    yp_var = np.exp(2.0 * sub[model])
    ratio = yt_var / yp_var
    daily = ratio - np.log(ratio) - 1.0
    return daily.groupby(level="date").mean()


def _shade_recessions(ax) -> None:
    for start, end in RECESSIONS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="grey", alpha=0.15)


def plot_rolling_qlike(merged: pd.DataFrame, models: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in models:
        daily = _qlike_at_date(merged, m)
        ax.plot(daily.rolling(60).mean(), label=m, color=PALETTE.get(m, None), linewidth=1.5)
    _shade_recessions(ax)
    ax.set_title("60-day rolling QLIKE by model")
    ax.set_ylabel("QLIKE")
    ax.set_xlabel("")
    ax.legend(loc="upper right")
    fig.savefig(out)
    plt.close(fig)


def plot_pred_vs_actual(merged: pd.DataFrame, models: list[str], out: Path) -> None:
    n = len(models)
    cols = min(n, 2)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    truth_for_bounds = merged["y_true"].dropna()
    lo = float(truth_for_bounds.min())
    hi = float(truth_for_bounds.max())
    for i, m in enumerate(models):
        ax = axes[i // cols, i % cols]
        sub = merged[["y_true", m]].dropna()
        sample = sub.sample(min(len(sub), 20000), random_state=SEED)
        ax.scatter(sample["y_true"], sample[m], s=3, alpha=0.3, color=PALETTE.get(m, "C0"))
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{m}: predicted vs actual log vol")
        ax.set_xlabel("Actual log vol")
        ax.set_ylabel("Predicted log vol")
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_residuals_over_time(merged: pd.DataFrame, models: list[str], out: Path) -> None:
    macro_path = DATA_RAW / "macro.parquet"
    vix = pd.read_parquet(macro_path)["vix"] if macro_path.exists() else None
    # Post-hoc threshold: uses the full sample to compute the 90th percentile.
    # Visualization aid only.
    high_vix_dates = vix[vix > vix.quantile(0.9)].index if vix is not None else pd.DatetimeIndex([])

    fig, ax = plt.subplots(figsize=(10, 5))
    for m in models:
        sub = merged[["y_true", m]].dropna()
        resid = (sub["y_true"] - sub[m]).groupby(level="date").mean()
        ax.plot(resid.rolling(20).mean(), label=m, color=PALETTE.get(m, None), linewidth=1.2)
    if len(high_vix_dates):
        starts = []
        ends = []
        prev = None
        for d in high_vix_dates:
            if prev is None or (d - prev).days > 5:
                if prev is not None:
                    ends.append(prev)
                starts.append(d)
            prev = d
        if prev is not None:
            ends.append(prev)
        for s, e in zip(starts, ends, strict=False):
            ax.axvspan(s, e, color="orange", alpha=0.05)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(
        "Residuals over time (20-day mean per model); orange shading = post-hoc high-VIX regime"
    )
    ax.set_ylabel("Residual (actual − predicted)")
    ax.legend(loc="upper right")
    fig.savefig(out)
    plt.close(fig)


def plot_per_ticker_metrics(merged: pd.DataFrame, best_model: str, out: Path) -> None:
    sub = merged[["y_true", best_model]].dropna()
    yt_var = np.exp(2.0 * sub["y_true"])
    yp_var = np.exp(2.0 * sub[best_model])
    df = pd.DataFrame({"y_true_var": yt_var.values, "y_pred_var": yp_var.values}, index=sub.index)
    rows = []
    for ticker, group in df.groupby(level="ticker"):
        rows.append((ticker, qlike(group["y_true_var"].to_numpy(), group["y_pred_var"].to_numpy())))
    res = pd.DataFrame(rows, columns=["ticker", "qlike"]).sort_values("qlike")

    fig, ax = plt.subplots(figsize=(10, max(5, 0.2 * len(res))))
    ax.barh(res["ticker"], res["qlike"], color=PALETTE.get(best_model, "C0"))
    ax.set_title(f"Per-ticker QLIKE — {best_model}")
    ax.set_xlabel("QLIKE")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _pooled_qlike(merged: pd.DataFrame, model: str, models: list[str]) -> float:
    """Pooled QLIKE for ``model`` on the all-models intersection.

    Computing QLIKE on each model's own valid rows produces an unfair
    comparison: GARCH cold-starts on post-IPO tickers, so its sample is
    a smaller and more recent slice than LightGBM's, and the per-model
    means are not directly comparable. Restricting to the intersection
    of every model's valid rows controls for that asymmetry.
    """
    cols = ["y_true", *models]
    intersection = merged[cols].dropna()
    if intersection.empty:
        return float("nan")
    return qlike(
        np.exp(2.0 * intersection["y_true"]).to_numpy(),
        np.exp(2.0 * intersection[model]).to_numpy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", action="store_true")
    args = parser.parse_args()

    ensure_output_dirs()
    apply_style()
    src = RESULTS_HOLDOUT if args.holdout else RESULTS_PREDICTIONS
    suffix = "_holdout" if args.holdout else ""
    preds = load_predictions(src)
    if not preds:
        raise SystemExit(f"No predictions found in {src}")
    merged = align_outer(preds)
    models = list(preds.keys())

    plot_rolling_qlike(merged, models, RESULTS_FIGURES / f"rolling_qlike{suffix}.png")
    plot_pred_vs_actual(merged, models, RESULTS_FIGURES / f"pred_vs_actual_scatter{suffix}.png")
    plot_residuals_over_time(merged, models, RESULTS_FIGURES / f"residuals_over_time{suffix}.png")

    # Best model = lowest pooled QLIKE on the all-models intersection (so
    # cold-start gaps in any one model do not skew the per-model comparison).
    best = min(models, key=lambda m: _pooled_qlike(merged, m, models))
    plot_per_ticker_metrics(merged, best, RESULTS_FIGURES / f"per_ticker_metrics{suffix}.png")
    print(f"Diagnostic plots written to {RESULTS_FIGURES}")


if __name__ == "__main__":
    main()

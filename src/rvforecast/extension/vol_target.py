"""Vol-targeted SPY position sizing.

Compares five sizing rules on SPY:

1. ``constant``: hold 1.0x SPY at all times.
2. ``garch``: scale exposure to target a 15% annualized vol using GARCH
   one-step forecasts.
3. ``har``: same but using HAR-RV forecasts.
4. ``lgbm``: same but using LightGBM forecasts.
5. ``lstm``: same but using LSTM forecasts.

Each rule uses ``position_t = clip(target_vol / forecast_vol_t, 0, leverage_cap)``,
where the forecast is the model's prediction for the realized vol of day
``t`` based on data through the close of ``t-1``. The forecast is therefore
known at ``t-1`` close; we enter the position at ``t-1`` close and hold
through ``t`` close, so the strategy log return on day ``t`` is
``position_t * r_t``. No additional shift is applied. (An earlier version
shifted positions by one day; that introduced a one-day-stale signal where
the forecast for day ``t-1`` was used to size day ``t``'s exposure.)

Forecast sourcing: SPY must appear in each model's prediction parquet for
the sizing rule to be evaluated. Earlier versions averaged constituent log
forecasts as a proxy when SPY was missing, but the average of constituent
log vols is a biased estimator of basket vol (it ignores cross-correlation),
so add SPY to the universe and rerun the model rather than relying on a
proxy.

Reported metrics: annualized return, annualized vol, Sharpe ratio, max
drawdown of the price-equity curve, turnover (sum of |Δposition|), and
Sharpe net of 1bp and 5bp per-turn transaction costs (the net Sharpe uses
the std of the net return series, not the gross std).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rvforecast.config import (
    DATA_RAW,
    LEVERAGE_CAP,
    RESULTS_EXTENSION,
    RESULTS_PREDICTIONS,
    TRADING_DAYS,
    VOL_TARGET_ANNUAL,
    ensure_output_dirs,
)

# Cap on how many days a stale forecast can be carried forward when
# reindexing onto the SPY return calendar. If a model quietly stops
# producing predictions, we don't want to drag a stale position for
# months; this cap is the limit.
FFILL_MAX_DAYS: int = 5


def _load_spy_returns() -> pd.Series:
    cache = DATA_RAW / "SPY.parquet"
    if cache.exists():
        spy = pd.read_parquet(cache)
    else:
        from rvforecast.data.fetch_prices import fetch_prices

        df = fetch_prices(["SPY"])
        spy = df.xs("SPY", level="ticker")
    spy = spy.sort_index()
    rets = np.log(spy["adj_close"] / spy["adj_close"].shift(1)).rename("ret")
    return rets.dropna()


def _load_pred_vol(name: str) -> pd.Series:
    """Per-date SPY forecast vol for a given model.

    SPY must be present in the prediction parquet. The previous fallback
    (cross-sectional mean of constituent log forecasts) was a biased
    proxy for index vol and has been removed.
    """
    path = RESULTS_PREDICTIONS / f"{name}.parquet"
    df = pd.read_parquet(path)
    if "SPY" not in df.index.get_level_values("ticker"):
        raise SystemExit(
            f"{path} does not contain SPY predictions. Add SPY to "
            "configs/universe_sp50.txt and rerun the data and model steps."
        )
    s = df.xs("SPY", level="ticker")["y_pred"]
    return np.exp(s).rename(f"{name}_vol")


def _max_drawdown(strat_log_returns: pd.Series) -> float:
    """Maximum drawdown of the price-equity curve, as a fraction.

    ``strat_log_returns`` is the daily log return of the strategy. The
    price index is ``exp(cumsum)`` starting at 1.0; drawdown at date t is
    ``(price_t - cummax_t) / cummax_t``.
    """
    if strat_log_returns.empty:
        return float("nan")
    price = np.exp(strat_log_returns.cumsum())
    drawdown = (price - price.cummax()) / price.cummax()
    return float(drawdown.min())


def _summarize(returns: pd.Series, positions: pd.Series, label: str) -> dict:
    """Summary metrics on a strategy.

    The strategy log return on day ``t`` is ``position_t * r_t`` (no shift):
    the forecast for day ``t`` is known at ``t-1`` close and the position
    is entered at that close.
    """
    strat = (positions * returns).dropna()
    if strat.empty:
        nan = float("nan")
        return {
            "strategy": label,
            "ann_return": nan,
            "ann_vol": nan,
            "sharpe": nan,
            "max_drawdown": nan,
            "turnover": nan,
            "tc_drag_1bp": nan,
            "tc_drag_5bp": nan,
            "sharpe_net_1bp": nan,
            "sharpe_net_5bp": nan,
        }
    ann_ret = strat.mean() * TRADING_DAYS
    ann_vol = strat.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    drawdown = _max_drawdown(strat)
    abs_turn_per_day = positions.diff().abs().reindex(strat.index).fillna(0.0)
    turnover = float(abs_turn_per_day.sum())
    cost_1bp = float((abs_turn_per_day * 0.0001).sum())
    cost_5bp = float((abs_turn_per_day * 0.0005).sum())
    net_1bp = strat - abs_turn_per_day * 0.0001
    net_5bp = strat - abs_turn_per_day * 0.0005
    sharpe_net_1bp = (
        net_1bp.mean() * TRADING_DAYS / (net_1bp.std() * np.sqrt(TRADING_DAYS))
        if net_1bp.std() > 0
        else float("nan")
    )
    sharpe_net_5bp = (
        net_5bp.mean() * TRADING_DAYS / (net_5bp.std() * np.sqrt(TRADING_DAYS))
        if net_5bp.std() > 0
        else float("nan")
    )
    return {
        "strategy": label,
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": drawdown,
        "turnover": turnover,
        "tc_drag_1bp": cost_1bp,
        "tc_drag_5bp": cost_5bp,
        "sharpe_net_1bp": float(sharpe_net_1bp),
        "sharpe_net_5bp": float(sharpe_net_5bp),
    }


def _build_position(forecast_vol: pd.Series) -> pd.Series:
    pos = (VOL_TARGET_ANNUAL / forecast_vol).clip(lower=0.0, upper=LEVERAGE_CAP)
    return pos.rename("position")


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    rets = _load_spy_returns()
    sized: dict[str, pd.Series] = {"constant": pd.Series(1.0, index=rets.index)}

    for name in ("garch", "har", "lgbm", "lstm"):
        pred_path = RESULTS_PREDICTIONS / f"{name}.parquet"
        if not pred_path.exists():
            continue
        raw_forecast = _load_pred_vol(name).reindex(rets.index)
        forecast = raw_forecast.ffill(limit=FFILL_MAX_DAYS)
        # Print how many days got carried forward. The cap (FFILL_MAX_DAYS)
        # is what bounds the worst case; this log just lets you see whether
        # we're anywhere near it for a given model.
        n_filled = int(raw_forecast.isna().sum() - forecast.isna().sum())
        if n_filled:
            print(f"{name}: forward-filled {n_filled} day(s) of forecast (cap {FFILL_MAX_DAYS})")
        sized[name] = _build_position(forecast)

    metrics = []
    equity_curves = {}
    for label, position in sized.items():
        common = rets.index.intersection(position.dropna().index)
        r, p = rets.loc[common], position.loc[common]
        metrics.append(_summarize(r, p, label))
        equity_curves[label] = (p * r).cumsum()

    metrics_df = pd.DataFrame(metrics)
    curves_df = pd.DataFrame(equity_curves)
    return metrics_df, curves_df


def _plot_equity(curves: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    from rvforecast.evaluation.plot_style import apply_style

    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in curves.columns:
        ax.plot(curves[col], label=col, linewidth=1.5)
    ax.set_title("Vol-targeted SPY equity curves (cumulative log returns)")
    ax.set_ylabel("Cumulative log return")
    ax.legend(loc="upper left")
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_EXTENSION)
    args = parser.parse_args()

    ensure_output_dirs()
    metrics, curves = run()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out_dir / "vol_target_metrics.csv", index=False)
    curves.to_parquet(args.out_dir / "vol_target_equity.parquet")
    _plot_equity(curves, args.out_dir / "vol_target_equity.png")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()

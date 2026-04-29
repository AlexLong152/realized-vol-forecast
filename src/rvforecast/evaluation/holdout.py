"""One-shot holdout evaluation.

Trains every model once on the entire pre-holdout sample, predicts on the
last two years, and writes outputs to ``results/holdout/`` tagged with
``_holdout``. This is run exactly once at the end of the project; if the
holdout numbers diverge from the walk-forward numbers, the README discusses
the gap rather than retraining to close it.

A purge gap of ``PURGE_DAYS`` business days is applied between the end of
the pre-holdout training sample and the start of the holdout, mirroring the
walk-forward folds.

The holdout-fitted LightGBM uses parameters frozen during the walk-forward
phase (``results/models/lgbm_params.json``). HAR and GARCH are refit on the
full pre-holdout sample. The LSTM holdout step (optional, requires
``--include-lstm``) trains five seeds on the pre-holdout panel exactly as in
walk-forward and reports the seed-mean prediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rvforecast.config import (
    DATA_RAW,
    PURGE_DAYS,
    RESULTS_HOLDOUT,
    RESULTS_MODELS,
    SEED,
    ensure_output_dirs,
)
from rvforecast.models import garch as garch_module
from rvforecast.models import har as har_module
from rvforecast.models import lgbm as lgbm_module
from rvforecast.models._runner import load_features
from rvforecast.validation.walk_forward import (
    load_holdout_dates,
    load_pre_holdout_train_dates,
    trading_dates_from_panel,
)


def _split_holdout(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = trading_dates_from_panel(panel)
    train_dates = load_pre_holdout_train_dates(dates, purge_days=PURGE_DAYS)
    holdout_dates = load_holdout_dates(dates)
    panel_dates = panel.index.get_level_values("date")
    pre = panel[panel_dates.isin(train_dates)]
    held = panel[panel_dates.isin(holdout_dates)]
    return pre.dropna(subset=["y_target"]), held.dropna(subset=["y_target"])


def _persist(name: str, test: pd.DataFrame, preds: pd.Series) -> Path:
    out = RESULTS_HOLDOUT / f"{name}_holdout.parquet"
    df = pd.DataFrame(
        {
            "y_true": test["y_target"].astype(float),
            "y_pred": preds.astype(float).reindex(test.index),
        },
        index=test.index,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return out


def _lstm_holdout(panel: pd.DataFrame, n_seeds: int = 5) -> None:
    """Train ``n_seeds`` LSTMs on pre-holdout data and predict the holdout.

    Mirrors the walk-forward LSTM training loop: features, raw price + macro
    panel, the same six-month internal validation block carved from the end
    of the training window, identical normalization rules, identical seed
    averaging. Imports are local so the rest of the holdout pipeline does
    not require torch when the LSTM step is not requested.
    """
    from rvforecast.models import lstm as lstm_module

    print(f"LSTM holdout ({n_seeds} seeds)...")
    prices = pd.read_parquet(DATA_RAW / "prices_long.parquet")
    macro = pd.read_parquet(DATA_RAW / "macro.parquet")
    X, y, idx, ticker_ids, n_tickers = lstm_module.build_lstm_inputs(panel, prices, macro)

    seq_dates = idx.get_level_values("date")
    panel_dates = trading_dates_from_panel(panel)
    train_dates = load_pre_holdout_train_dates(panel_dates, purge_days=PURGE_DAYS)
    holdout_dates = load_holdout_dates(panel_dates)

    masks = lstm_module.partition_fold(seq_dates, train_dates, holdout_dates)
    if masks is None:
        raise SystemExit("LSTM holdout has no usable rows; rebuild features and raw panels.")

    mean_pred, std_pred, y_te = lstm_module.run_seed_ensemble(
        X, y, ticker_ids, masks, n_tickers=n_tickers, n_seeds=n_seeds, base_seed=SEED
    )
    test_mask = masks[2]
    out = pd.DataFrame(
        {
            "y_true": y_te.astype(float),
            "y_pred": mean_pred.astype(float),
            "y_pred_std": std_pred.astype(float),
        },
        index=idx[test_mask],
    )
    path = RESULTS_HOLDOUT / "lstm_holdout.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    print(f"  wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-lstm",
        action="store_true",
        help="Also retrain the LSTM (5 seeds) on the pre-holdout sample. Requires torch.",
    )
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    ensure_output_dirs()
    panel = load_features()
    train, holdout = _split_holdout(panel)

    print("Naive holdout...")
    _persist("naive", holdout, holdout["log_rv_lag_1d"])

    print("HAR holdout...")
    har_preds = har_module.fit_predict(train, holdout)
    _persist("har", holdout, har_preds)

    print("GARCH holdout...")
    returns = garch_module.load_returns_panel()
    garch_preds = garch_module.predict(train, holdout, returns)
    _persist("garch", holdout, garch_preds)

    print("LightGBM holdout...")
    params_path = RESULTS_MODELS / "lgbm_params.json"
    if not params_path.exists():
        raise SystemExit("lgbm_params.json missing; run the walk-forward LightGBM step first.")
    with open(params_path) as f:
        params = json.load(f)

    booster, lgbm_preds, _, _ = lgbm_module.fit_booster(train, holdout, params)
    _persist("lgbm", holdout, lgbm_preds)
    booster.save_model(str(RESULTS_MODELS / "lgbm_holdout.txt"))

    if args.include_lstm:
        _lstm_holdout(panel, n_seeds=args.seeds)

    print(f"Holdout predictions written to {RESULTS_HOLDOUT}")


if __name__ == "__main__":
    main()

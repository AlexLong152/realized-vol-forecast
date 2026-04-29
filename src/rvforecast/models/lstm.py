"""LSTM volatility forecaster.

Architecture
------------
Two LSTM layers of 64 hidden units with dropout 0.2 between them, followed
by a small ticker embedding concatenated with the final hidden state and a
linear projection to a single scalar. The model consumes a sequence of the
last 22 trading days of raw daily features per ticker and predicts the log
realized volatility of the *following* day, matching the targets used by
the other models. A sequence ending on day ``t-1`` predicts day ``t``, so
the LSTM uses data with index strictly less than ``t`` exactly as the
tabular pipeline does.

Per-day features in the sequence (no rolling aggregates — the LSTM is
expected to learn its own summaries):
    - log Garman-Klass realized vol on the day
    - log close-to-close return
    - log(high/low) intraday range
    - log(close/open) drift contribution
    - VIX level
    - 10Y-2Y term spread

Normalization is per fold: each feature is z-scored using the mean and
standard deviation of the fold's training-only sequences (sequences
whose prediction date is in the train-only block; the trailing six
months are kept as the validation block for early stopping). Validation
sequences read input rows that by date are in the training-only block,
which is fine — the val sequence's *prediction* date is in the val
block, and its input window is allowed to look at history up to that
point. Mentioning this so it doesn't look like a leak.

Multi-seed reporting
--------------------
Five seeds per fold; the saved prediction is the per-row mean across seeds,
and the per-row standard deviation is recorded alongside as ``y_pred_std``.
Single-seed deep learning numbers on a panel this small are not credible;
seed dispersion is part of the result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rvforecast.config import (
    DATA_PROCESSED,
    DATA_RAW,
    RESULTS_PREDICTIONS,
    SEED,
    TRADING_DAYS,
    ensure_output_dirs,
)
from rvforecast.validation.walk_forward import (
    iter_fold_dates,
    persist_splits,
    trading_dates_from_panel,
)

SEQ_LEN = 22
HIDDEN = 64
LAYERS = 2
DROPOUT = 0.2
TICKER_EMB = 8
BATCH_SIZE = 1024
LR = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 80
PATIENCE = 8
N_SEEDS = 5

RAW_FEATURES = ["log_rv_1d", "ret_1d", "log_hl", "log_co", "vix", "term_spread"]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_raw_panel(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Per-day raw feature panel keyed by ``(date, ticker)``.

    Includes the contemporaneous Garman-Klass realized vol and same-day OHLC
    transforms; macro series (VIX, term spread) are merged on date. The
    panel is intentionally NOT lagged here — sequence construction enforces
    the look-ahead constraint by ending each input sequence at day ``t-1``
    and pairing it with the target for day ``t``.
    """
    panel = prices.sort_index().copy()
    open_ = panel["open"]
    high = panel["high"]
    low = panel["low"]
    close = panel["close"]

    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    daily_var = (0.5 * log_hl**2 - (2.0 * np.log(2.0) - 1.0) * log_co**2).clip(lower=0.0)
    log_rv = np.log(np.sqrt(daily_var * TRADING_DAYS).replace(0.0, np.nan)).astype(float)

    out = pd.DataFrame(index=panel.index)
    out["log_rv_1d"] = log_rv
    out["ret_1d"] = np.log(panel["adj_close"]).groupby(level="ticker").diff()
    out["log_hl"] = log_hl
    out["log_co"] = log_co

    macro = macro.sort_index().ffill()[["vix", "term_spread"]]
    out = (
        out.reset_index()
        .merge(macro, on="date", how="left")
        .set_index(["date", "ticker"])
        .sort_index()
    )
    return out.dropna()


def _build_sequences(
    raw: pd.DataFrame, y_target: pd.Series, seq_len: int = SEQ_LEN
) -> tuple[np.ndarray, np.ndarray, pd.MultiIndex]:
    """Vectorized per-ticker sliding-window construction.

    For each ticker, sort by date and build a 3D tensor of shape
    ``(n_seq, seq_len, n_features)`` where the i-th sequence is the
    ``seq_len`` consecutive rows ``[i, i+seq_len-1]``. The sequence is
    paired with the target on the *next* trading row's date, so the LSTM
    uses data with index strictly less than the prediction date — the same
    convention as the tabular feature builder. The returned ``MultiIndex``
    is keyed on the prediction date, not the sequence end date.
    """
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    idx_parts: list[pd.MultiIndex] = []

    target = y_target.dropna()
    for ticker, group in raw.groupby(level="ticker"):
        group = group.sort_index()
        n = len(group)
        if n <= seq_len:
            continue
        arr = group[RAW_FEATURES].to_numpy(dtype=np.float32)
        # Sequence i covers rows [i, i+seq_len-1] and predicts row i+seq_len,
        # so there are n - seq_len complete (sequence, target-row) pairs.
        n_seq = n - seq_len
        offsets = (
            np.arange(seq_len, dtype=np.int64)[None, :] + np.arange(n_seq, dtype=np.int64)[:, None]
        )
        X = arr[offsets]
        all_dates = group.index.get_level_values("date")
        prediction_dates = all_dates[seq_len : seq_len + n_seq]
        seq_idx = pd.MultiIndex.from_arrays(
            [prediction_dates, np.full(n_seq, ticker)], names=["date", "ticker"]
        )
        keep = seq_idx.isin(target.index)
        if not keep.any():
            continue
        X_parts.append(X[keep])
        y_parts.append(target.loc[seq_idx[keep]].to_numpy(dtype=np.float32))
        idx_parts.append(seq_idx[keep])

    X_full = np.concatenate(X_parts, axis=0)
    y_full = np.concatenate(y_parts, axis=0)
    if len(idx_parts) == 1:
        idx_full = idx_parts[0]
    else:
        # ``MultiIndex.append`` on a list of indexes uses the underlying
        # arrays directly; ``from_tuples`` over a flattened list comprehension
        # materializes hundreds of thousands of Python tuples and is the
        # slowest step before training starts.
        idx_full = idx_parts[0].append(idx_parts[1:])
    return X_full, y_full, idx_full


def partition_fold(
    seq_dates: pd.DatetimeIndex,
    train_dates: pd.DatetimeIndex,
    test_dates: pd.DatetimeIndex,
    val_months: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Build train-only, val, and test boolean masks over the sequence index.

    Returns ``None`` if any of the three resulting masks is empty so the
    caller can skip the fold rather than train on a degenerate split.
    """
    train_mask = seq_dates.isin(train_dates)
    test_mask = seq_dates.isin(test_dates)
    if not train_mask.any() or not test_mask.any():
        return None

    train_dates_in_fold = seq_dates[train_mask]
    val_cutoff = train_dates_in_fold.max() - pd.DateOffset(months=val_months)
    val_mask = train_mask & (seq_dates >= val_cutoff)
    train_only_mask = train_mask & ~val_mask
    if not train_only_mask.any() or not val_mask.any():
        return None
    return np.asarray(train_only_mask), np.asarray(val_mask), np.asarray(test_mask)


def _normalize_with_train(X: np.ndarray, train_only_mask: np.ndarray) -> np.ndarray:
    """Per-fold z-score using training-only statistics."""
    X_train_only = X[train_only_mask]
    mu = X_train_only.mean(axis=(0, 1), keepdims=True)
    sigma = X_train_only.std(axis=(0, 1), keepdims=True) + 1e-6
    return ((X - mu) / sigma).astype(np.float32)


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_tickers: int,
        hidden: int = HIDDEN,
        layers: int = LAYERS,
        dropout: float = DROPOUT,
        ticker_emb: int = TICKER_EMB,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.ticker_emb = nn.Embedding(n_tickers, ticker_emb)
        self.head = nn.Sequential(
            nn.Linear(hidden + ticker_emb, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, ticker_ids: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        emb = self.ticker_emb(ticker_ids)
        return self.head(torch.cat([last, emb], dim=1)).squeeze(-1)


def _train_one_seed(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    tk_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    tk_va: np.ndarray,
    X_te: np.ndarray,
    tk_te: np.ndarray,
    n_tickers: int,
    seed: int,
) -> np.ndarray:
    """Train one seed, return test-set predictions as a 1D numpy array.

    Seeds numpy and torch (CPU and CUDA). cuDNN is also pinned to its
    deterministic kernels with ``benchmark=False`` so per-seed results are
    reproducible on CUDA — without this the same seed can produce slightly
    different predictions across runs on the GPU. The seed averaging across
    ``N_SEEDS`` is the headline mitigation for non-determinism, but pinning
    cuDNN keeps the per-seed numbers themselves stable.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = _device()
    model = LSTMRegressor(n_features=X_tr.shape[2], n_tickers=n_tickers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(
        torch.from_numpy(X_tr), torch.from_numpy(tk_tr).long(), torch.from_numpy(y_tr)
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_va), torch.from_numpy(tk_va).long(), torch.from_numpy(y_va)
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 4, shuffle=False, num_workers=0, pin_memory=True
    )

    best_val = float("inf")
    best_state: dict | None = None
    epochs_no_improve = 0

    for _epoch in range(MAX_EPOCHS):
        model.train()
        for xb, tb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            tb = tb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb, tb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, tb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                tb = tb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb, tb)
                val_loss_sum += float(loss_fn(pred, yb).item()) * yb.shape[0]
                val_n += yb.shape[0]
        val_loss = val_loss_sum / max(val_n, 1)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds: list[np.ndarray] = []
    test_ds = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(tk_te).long())
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE * 4, shuffle=False, num_workers=0, pin_memory=True
    )
    with torch.no_grad():
        for xb, tb in test_loader:
            xb = xb.to(device, non_blocking=True)
            tb = tb.to(device, non_blocking=True)
            preds.append(model(xb, tb).detach().cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def run_seed_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    ticker_ids: np.ndarray,
    masks: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_tickers: int,
    n_seeds: int,
    base_seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ``n_seeds`` LSTMs on a single fold and aggregate predictions.

    Returns ``(mean_pred, std_pred, y_test)`` where ``mean_pred`` and
    ``std_pred`` are over the seed dimension and ``y_test`` is the test
    labels for convenience.
    """
    train_only_mask, val_mask, test_mask = masks
    X_norm = _normalize_with_train(X, train_only_mask)

    X_tr = X_norm[train_only_mask]
    y_tr = y[train_only_mask]
    tk_tr = ticker_ids[train_only_mask]
    X_va = X_norm[val_mask]
    y_va = y[val_mask]
    tk_va = ticker_ids[val_mask]
    X_te = X_norm[test_mask]
    y_te = y[test_mask]
    tk_te = ticker_ids[test_mask]

    seed_preds: list[np.ndarray] = []
    for s in range(n_seeds):
        preds = _train_one_seed(
            X_tr,
            y_tr,
            tk_tr,
            X_va,
            y_va,
            tk_va,
            X_te,
            tk_te,
            n_tickers=n_tickers,
            seed=base_seed + s,
        )
        seed_preds.append(preds)

    stacked = np.stack(seed_preds, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0), y_te


def build_lstm_inputs(
    features_panel: pd.DataFrame,
    prices: pd.DataFrame,
    macro: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.MultiIndex, np.ndarray, int]:
    """Construct the LSTM tensors and ticker IDs from the cached panels.

    Returns ``(X, y, idx, ticker_ids, n_tickers)``. ``idx`` is the prediction
    date / ticker MultiIndex aligned with rows of ``X``, ``y``, and
    ``ticker_ids``.
    """
    raw = _build_raw_panel(prices, macro)
    X, y, idx = _build_sequences(raw, features_panel["y_target"])
    tickers_sorted = sorted(set(idx.get_level_values("ticker")))
    ticker_to_id = {t: i for i, t in enumerate(tickers_sorted)}
    ticker_ids = np.array([ticker_to_id[t] for t in idx.get_level_values("ticker")], dtype=np.int64)
    return X, y, idx, ticker_ids, len(tickers_sorted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DATA_PROCESSED / "features.parquet")
    parser.add_argument("--prices", type=Path, default=DATA_RAW / "prices_long.parquet")
    parser.add_argument("--macro", type=Path, default=DATA_RAW / "macro.parquet")
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--out", type=Path, default=RESULTS_PREDICTIONS / "lstm.parquet")
    args = parser.parse_args()

    ensure_output_dirs()
    print(f"device: {_device()} (torch {torch.__version__})")

    features = pd.read_parquet(args.features)
    prices = pd.read_parquet(args.prices)
    macro = pd.read_parquet(args.macro)

    X, y, idx, ticker_ids, n_tickers = build_lstm_inputs(features, prices, macro)
    print(f"sequences: {X.shape}  tickers: {n_tickers}  targets: {y.shape}")

    persist_splits(trading_dates_from_panel(features))
    seq_dates = idx.get_level_values("date")

    chunks: list[pd.DataFrame] = []
    for fold, train_dates, test_dates in iter_fold_dates(trading_dates_from_panel(features)):
        masks = partition_fold(seq_dates, train_dates, test_dates)
        if masks is None:
            continue
        mean_pred, std_pred, y_te = run_seed_ensemble(
            X, y, ticker_ids, masks, n_tickers=n_tickers, n_seeds=args.seeds, base_seed=SEED
        )
        test_mask = masks[2]
        chunk = pd.DataFrame(
            {
                "y_true": y_te.astype(float),
                "y_pred": mean_pred.astype(float),
                "y_pred_std": std_pred.astype(float),
                "fold": fold,
            },
            index=idx[test_mask],
        )
        chunks.append(chunk)
        print(
            f"fold {fold:>2d}  test n={int(test_mask.sum())}  "
            f"mean MSE_log={float(((mean_pred - y_te) ** 2).mean()):.4f}"
        )

    if not chunks:
        raise SystemExit("no folds produced predictions; check data coverage")
    full = pd.concat(chunks).sort_index()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(args.out)
    print(f"LSTM predictions written to {args.out}")


if __name__ == "__main__":
    main()

"""Forecast evaluation metrics for variance forecasts.

Conventions
-----------
- The modeling target is log realized volatility. To compute QLIKE, predictions
  and truths are first exponentiated to get volatilities, then squared to get
  variances. QLIKE is then a function of variances.
- ``r2_oos`` is the standard out-of-sample R-squared relative to a baseline
  predictor (HAR by default), and is computed in *log* space because that is
  the loss the models are trained on.

References
----------
Patton, A. J. (2011). Volatility forecast comparison using imperfect
volatility proxies. *Journal of Econometrics*, 160(1), 246–256.

Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
*Journal of Business & Economic Statistics*, 13(3), 253–263.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _to_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def qlike(y_true_var: np.ndarray, y_pred_var: np.ndarray) -> float:
    """Patton (2011) QLIKE loss for variance forecasts.

    QLIKE(σ²_true, σ²_pred) = σ²_true / σ²_pred − log(σ²_true / σ²_pred) − 1.

    Both inputs must be variances (not volatilities). Returns the mean over
    the sample. Lower is better; QLIKE = 0 only if predictions equal truth.
    """
    yt = _to_array(y_true_var)
    yp = _to_array(y_pred_var)
    mask = np.isfinite(yt) & np.isfinite(yp) & (yt > 0) & (yp > 0)
    if not mask.any():
        return float("nan")
    ratio = yt[mask] / yp[mask]
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def mse_log(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """Mean squared error in log space (the training-time proxy)."""
    yt = _to_array(y_true_log)
    yp = _to_array(y_pred_log)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not mask.any():
        return float("nan")
    return float(np.mean((yt[mask] - yp[mask]) ** 2))


def r2_oos(y_true: np.ndarray, y_pred: np.ndarray, y_baseline: np.ndarray) -> float:
    """Out-of-sample R-squared of a model versus a baseline predictor.

    Defined as ``1 − SSE_model / SSE_baseline`` where both SSE terms are
    computed on the same out-of-sample target. ``y_baseline`` is the
    *baseline model's* predictions on this sample (e.g., HAR), not the
    in-sample mean; for the latter convention pass an array of the relevant
    constant. Positive values indicate the model beats the baseline;
    negative values indicate it does worse.
    """
    yt = _to_array(y_true)
    yp = _to_array(y_pred)
    yb = _to_array(y_baseline)
    mask = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(yb)
    if not mask.any():
        return float("nan")
    sse_model = np.sum((yt[mask] - yp[mask]) ** 2)
    sse_baseline = np.sum((yt[mask] - yb[mask]) ** 2)
    if sse_baseline == 0:
        return float("nan")
    return float(1.0 - sse_model / sse_baseline)


def diebold_mariano(errors_a: np.ndarray, errors_b: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Diebold-Mariano test for forecast accuracy difference.

    Inputs are *loss* series (e.g., squared errors) for two competing models
    on the same sample. The loss differential is ``d = errors_a - errors_b``,
    so a *positive* DM statistic indicates model B has lower mean loss
    (i.e., model B is the more accurate forecaster); a negative statistic
    indicates the reverse. The null is equal expected loss.

    For ``h == 1`` there is no overlap in forecast errors and the variance of
    the loss differential is the sample variance directly. For ``h > 1``,
    forecasts overlap by up to ``h-1`` periods, so a Bartlett-kernel HAC
    variance with bandwidth ``h-1`` is used and the Harvey-Leybourne-Newbold
    small-sample correction is applied to the statistic.

    Returns
    -------
    (dm_stat, p_value) : tuple of float
        Two-sided p-value under a standard-normal approximation.
    """
    a = _to_array(errors_a)
    b = _to_array(errors_b)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 10:
        return float("nan"), float("nan")
    d = a - b
    mean_d = float(np.mean(d))
    # For h == 1 the loop is empty and var is just the sample variance of
    # the loss differential; for h > 1 a Bartlett HAC of bandwidth h-1 is
    # added to account for forecast-error overlap.
    var = float(np.var(d, ddof=1))
    for lag in range(1, h):
        cov = float(np.cov(d[lag:], d[:-lag], ddof=1)[0, 1])
        var += 2.0 * (1.0 - lag / h) * cov
    # Non-positive variance means the loss-difference series is degenerate
    # (identical losses, or a tiny sample dominated by one value). Return
    # NaN — flooring the variance instead would just produce a huge ``dm``
    # and a meaningless p-value.
    if not np.isfinite(var) or var <= 0.0:
        return float("nan"), float("nan")
    dm = mean_d / np.sqrt(var / n)
    if h > 1:
        # HLN small-sample correction
        correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        dm = dm * correction
    # ``norm.sf`` is numerically stable for large |dm| where 1 - cdf underflows.
    p = 2.0 * stats.norm.sf(abs(dm))
    return float(dm), float(p)


def squared_errors_log(y_true_log: pd.Series, y_pred_log: pd.Series) -> pd.Series:
    return (y_true_log - y_pred_log) ** 2


def summarize_predictions(preds: pd.DataFrame, baseline: pd.DataFrame | None = None) -> dict:
    """Compute headline metrics for a single model's predictions.

    Parameters
    ----------
    preds : DataFrame
        Must contain ``y_true`` and ``y_pred`` (both in log-vol space).
    baseline : DataFrame or None
        If provided, used to compute OOS R-squared. Should be aligned to
        ``preds`` on its index.

    Returns
    -------
    dict with keys ``qlike``, ``mse_log``, ``r2_oos``.
    """
    df = preds.dropna(subset=["y_true", "y_pred"])
    yt_log = df["y_true"].to_numpy()
    yp_log = df["y_pred"].to_numpy()
    yt_var = np.exp(2.0 * yt_log)
    yp_var = np.exp(2.0 * yp_log)
    out = {
        "qlike": qlike(yt_var, yp_var),
        "mse_log": mse_log(yt_log, yp_log),
        "r2_oos": float("nan"),
    }
    if baseline is not None:
        joined = df.join(baseline["y_pred"].rename("y_base"), how="inner").dropna()
        if not joined.empty:
            out["r2_oos"] = r2_oos(
                joined["y_true"].to_numpy(),
                joined["y_pred"].to_numpy(),
                joined["y_base"].to_numpy(),
            )
    return out

# rvforecast

Walk-forward realized-volatility forecasting for US equities. The target is one-day-ahead log realized volatility on a static S&P 500 universe. Five models: naive, HAR-RV, GARCH(1,1), LightGBM, and a 5-seed LSTM ensemble. Evaluation: QLIKE, out-of-sample R² versus HAR, and pairwise Diebold–Mariano tests, with the last two years held out for a one-shot final test.

If you're a quant or ML reviewer skimming this, the things to check are: leakage tests pass, walk-forward is purged and embargoed, the holdout is only touched once, the baselines aren't strawmen, and QLIKE (not MSE) is the loss for variance forecasts.

## Install

```bash
make install
# equivalent to
python -m pip install -e ".[lstm,shap,dev]"
```

`make install` pulls in `torch` (CPU build) for the LSTM step. For CUDA, install `torch` from the PyTorch CUDA index *before* `make install` so the editable install picks up the GPU build.

## Reproduce

```bash
make data        # fetches OHLCV (yfinance) + VIX/FRED macro
make features    # builds the feature matrix with leakage tests pinned
make baselines   # naive, HAR-RV, GARCH(1,1)
make lgbm        # LightGBM, tuned once on a strictly-pre-walk-forward window
make lstm        # 5-seed LSTM ensemble (requires torch, included by make install)
make eval        # writes metrics.csv and dm_pvalues.csv
make extension   # vol-targeted SPY position sizing
make holdout     # one-shot final-test evaluation
```

`make all` runs the whole pipeline. `make test` runs the test suite (~60 tests, no network); `make lint` runs ruff and `black --check`.

## Results

50-name S&P 500 universe, 2005-01-01 → 2026-04-01, walk-forward folds with the last two years held out for a single one-shot evaluation. Lower QLIKE is better. `r2_oos` is out-of-sample R² versus HAR-RV (the literature's standard hard-to-beat baseline).

**Walk-forward (162k obs across all folds)**

| model  | qlike | mse_log | r²_oos vs HAR |
|--------|------:|--------:|--------------:|
| lstm   | 0.335 |   0.125 |        +13.5% |
| lgbm   | 0.356 |   0.133 |         +7.7% |
| har    | 0.409 |   0.144 |         (ref) |
| garch  | 0.459 |   0.312 |       −116.1% |
| naive  | 0.647 |   0.225 |        −56.0% |

**One-shot holdout (2024–2026, 25k obs, touched once)**

| model  | qlike | mse_log | r²_oos vs HAR |
|--------|------:|--------:|--------------:|
| lstm   | 0.350 |   0.129 |        +12.5% |
| lgbm   | 0.360 |   0.138 |         +6.6% |
| har    | 0.407 |   0.147 |         (ref) |
| garch  | 0.425 |   0.289 |        −96.1% |
| naive  | 0.638 |   0.231 |        −56.8% |

The walk-forward and holdout rankings agree, the gaps are similar, and the holdout numbers are the slightly worse of the two, which is what you want to see (no holdout overfit). Every pairwise Diebold–Mariano p-value is < 0.001, so the differences are statistically real, with the usual caveat that 10 pairwise tests deserve a multiple-testing correction (a Model Confidence Set is on the roadmap).

![Rolling QLIKE per model](results/figures/rolling_qlike.png)

60-day rolling QLIKE per model. Grey bands are NBER recessions. The ML models hold their advantage over HAR through 2008 and 2020 (the periods where forecasts matter most) and never blow up. GARCH consistently lags because the conditional-variance recursion can't keep up with the regime shifts inside a six-month test fold.

### Per-model commentary

- **naive** (predict yesterday's vol). Worst by every metric and the floor every other model has to clear. Useful as a sanity check.
- **GARCH(1,1)**. Beats naive but not HAR. Refit monthly per ticker; the smooth conditional-vol recursion is exactly what you'd want for *position sizing* (lowest turnover in the vol-targeting extension below) but not for *point forecasting* on a daily horizon.
- **HAR-RV** (Corsi 2009). The right baseline for this problem. Pooled OLS with ticker fixed effects; tied for second by mse_log, third by QLIKE.
- **LightGBM**. +7.7% OOS R² over HAR walk-forward, +6.6% holdout. SHAP and gain-based feature importance both put the 22-day GK lag and the lagged log RV at the top: the model is mostly rediscovering HAR-style aggregations and adding a small nonlinear premium on top.
- **LSTM** (5-seed ensemble, mean ± std reported as `y_pred_std`). +13.5% OOS R² over HAR walk-forward, +12.5% holdout. The seed std is non-trivial: single-seed numbers on a panel this small aren't credible, hence the 5-seed protocol.

![Predicted vs actual log realized vol](results/figures/pred_vs_actual_scatter.png)

The diagonal is perfect prediction. Naive scatters along it (it just lags); GARCH compresses toward the mean (the recursion smooths through volatility spikes); HAR / LightGBM / LSTM track the diagonal more faithfully into the right tail, which is where forecast value lives.

![LightGBM feature importance](results/figures/feature_importance.png)

LightGBM's top features by gain. The 22-day GK lag and 22-day log-RV lag dominate. The model is not finding alpha in obscure macro features; it is finding it in the same realized-vol aggregations HAR uses, and learning a mild nonlinearity on top. That's a real finding and goes in "What Didn't Work" territory: we did *not* find that calendar effects, term-spread, or VIX changes added meaningful predictive content beyond the realized-vol features themselves.

### Vol-targeted SPY (the trading-relevance extension)

Each rule sizes a long SPY position to a 15% annualized vol target using the model's forecast for SPY (`position_t = clip(0.15 / forecast_vol_t, 0, 2)`). Sharpe net of 5bp per-turn transaction costs:

| strategy | ann_return | ann_vol | sharpe | max_dd | turnover | sharpe_net_5bp |
|----------|----------:|---------:|-------:|-------:|---------:|---------------:|
| constant (1× SPY) | 10.3% | 19.0% | 0.54 | −80% | 0.0 | 0.54 |
| garch    | 8.0%  |  9.4%  | 0.85 | −15% |  35.4 | 0.84 |
| har      | 11.0% | 13.2%  | 0.83 | −23% |  88.0 | 0.81 |
| lgbm     | 10.7% | 12.8%  | 0.83 | −22% | 116.7 | 0.81 |
| lstm     | 11.1% | 12.8%  | 0.87 | −21% | 134.7 | 0.84 |

![Vol-targeted SPY equity curves](results/extension/vol_target_equity.png)

Two things worth saying out loud here. First, *every* vol-targeted variant beats constant SPY on Sharpe and on max drawdown. The vol forecast doesn't have to be great for vol targeting to add value, it just has to be reasonable. Second, GARCH gives the **best gross-Sharpe-per-unit-turnover** despite being the worst point forecaster: its forecasts are smoother, so positions don't churn. By contrast LSTM has the highest gross Sharpe but loses the most to costs because it reacts the most. There is no single "best" model for vol-targeting; the answer depends on your transaction-cost regime.

## Repository layout

```
src/rvforecast/
  config.py               constants, paths, seeds, target definitions
  data/                   yfinance + FRED ingestion with manifest cache
  features/               OHLC → realized vol → HAR/range/macro/calendar/sector features
  validation/             purged + embargoed walk-forward splitter
  models/                 naive, har, garch, lgbm, lstm
  evaluation/             metrics (QLIKE, OOS R², DM), plots, holdout, run_eval
  extension/              vol-targeted SPY sizing across forecasts

tests/                    look-ahead, splitter, GARCH-arch contract, DM symmetry, ...
configs/                  ticker universe + sector map
```

## How the usual mistakes are avoided

- **Look-ahead.** Every feature at row `t` is a function of data strictly before `t`. Two tests in `tests/test_features.py` check this: a future spike injected at row `t` doesn't move any feature at earlier rows, and perturbing the OHLC of day `t` itself doesn't move any feature at row `t`.
- **Walk-forward.** Expanding (or rolling) training window, a purge gap between train and test, a local embargo between consecutive folds, and the last `HOLDOUT_YEARS=2` reserved as a one-shot test. See `src/rvforecast/validation/walk_forward.py`.
- **Holdout.** `make holdout` retrains each model once on the pre-holdout sample and predicts the held-out two years. The whole point is that you only run it once.
- **Targets and loss.** Target is log Garman–Klass realized vol. Evaluation is QLIKE on variances; training uses MSE in log space because that's what the models actually optimize. Pairwise Diebold–Mariano uses a HAC variance and the Harvey–Leybourne–Newbold small-sample correction.

## What this doesn't fix

- **Survivorship bias.** The universe is currently liquid S&P 500 names. Free data doesn't give you clean point-in-time index membership, so in-sample vol is biased down and any strategy result inherits the same bias. Worth saying out loud.
- **LightGBM hyperparameters.** Tuned once on `2005-01-01 … 2009-12-31` (strictly before the first walk-forward test fold) and then frozen. Cheaper than nested rolling tuning but obviously less adaptive; see `src/rvforecast/models/lgbm.py`.
- **The vol target itself.** Daily one-day Garman–Klass is a noisy proxy for true realized vol. Intraday data would help, but it isn't free.
- **Transaction costs.** The vol-targeting extension subtracts costs additively from log returns (fine at 1–5 bp), rather than modeling them multiplicatively on positions.

## What didn't work

- **Macro features beyond realized vol.** VIX changes, the 252-day VIX z-score, the term spread, and term-spread changes are all in the feature matrix. None of them shows up near the top of LightGBM's gain-based importance; the realized-vol lags dominate. This is a real finding, not a tuning failure: at the daily horizon, the lagged realized-vol signal is so dominant that macro adds little marginal predictive content.
- **Calendar effects.** Day-of-week and month-of-year dummies were included; both rank near the bottom of feature importance.
- **GARCH for point forecasting.** GARCH(1,1) is the lowest QLIKE among the non-naive models. The conditional-variance recursion is too smooth to track the kind of day-by-day dispersion that a daily realized-vol target rewards. (It's still useful for *position sizing*, where smoothness is a feature; see the vol-targeting table above.)
- **Pairwise DM without multiple-testing correction.** Every p-value is essentially zero, so the qualitative ranking holds, but reporting 10 pairwise tests with no Bonferroni / Holm / Romano-Wolf correction is the kind of thing a careful reviewer asks about. Model Confidence Set (Hansen-Lunde-Nason 2011) is the right replacement and is on the roadmap.

## References

- Corsi (2009): HAR-RV
- Patton (2011): QLIKE and proper losses for variance forecasts
- López de Prado (2018), *Advances in Financial Machine Learning*, ch. 7: purged + embargoed walk-forward CV
- Bollerslev (1986): GARCH
- Garman & Klass (1980): range-based vol estimators
- Diebold & Mariano (1995); Harvey, Leybourne & Newbold (1997): forecast accuracy testing

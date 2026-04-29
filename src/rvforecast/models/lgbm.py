"""LightGBM model, pooled across tickers with ``ticker`` as a categorical.

Training protocol
-----------------
- Within each walk-forward training window, the trailing six months are
  carved off as a contiguous validation block for early stopping. The
  validation block is contiguous, not random, to respect time order.
- Hyperparameters are tuned once on a fixed pre-walk-forward sample
  (2005-01-01 through 2009-12-31, strictly before the first walk-forward
  test fold under ``INITIAL_TRAIN_YEARS=5``) using Optuna, then frozen and
  reused for every walk-forward fold. Tuning never sees data that any test
  fold also evaluates on. This is the simpler alternative to nested
  walk-forward; the choice is documented in the README.
- ``num_boost_round=2000`` with ``early_stopping_rounds=50``.

The booster for each fold is saved to ``results/models/lgbm_fold_{fold}.txt``
via LightGBM's native ``Booster.save_model``. (Pickle of LightGBM boosters
is fragile across library versions; the text format is the supported
serialization.)
SHAP and feature importance plots are saved on the *final* walk-forward
fold, so the artifacts reflect the most recent training window. SHAP is
optional (``pip install rvforecast[shap]``); if unavailable the SHAP plot
step is skipped and a notice is printed.
"""

from __future__ import annotations

import json

import lightgbm as lgb
import optuna
import pandas as pd

from rvforecast.config import (
    RESULTS_FIGURES,
    RESULTS_MODELS,
    RESULTS_TABLES,
    SEED,
    ensure_output_dirs,
)
from rvforecast.models._runner import load_features, run_walk_forward

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Tuning window is strictly before the first walk-forward test fold
# (panel start 2005-01-01 + INITIAL_TRAIN_YEARS=5 ⇒ first test starts ~2010).
# Tuning therefore never sees data that any test fold also evaluates on.
TUNE_START = pd.Timestamp("2005-01-01")
TUNE_END = pd.Timestamp("2009-12-31")
PARAMS_PATH = RESULTS_MODELS / "lgbm_params.json"

CATEGORICAL = ["ticker", "sector", "dow", "month"]
DROP_COLS = {"y_target", "fold"}

# feature_pre_filter must be False so Optuna can vary min_data_in_leaf;
# otherwise the first trial's value freezes the pre-filtered feature set
# and subsequent trials with smaller min_data_in_leaf are rejected. We
# also pass it on every non-tuning fit for consistency, since Dataset
# parameters can otherwise diverge between training sites.
DATASET_PARAMS: dict[str, bool] = {"feature_pre_filter": False}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DROP_COLS]


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Split a slice of the feature panel into ``(X, y, categorical columns)``.

    Adds the ``ticker`` index level as a column and casts categorical
    columns to ``category`` dtype.
    """
    df = df.copy()
    df["ticker"] = df.index.get_level_values("ticker")
    feat_cols = feature_columns(df)
    cat_cols = [c for c in CATEGORICAL if c in feat_cols]
    for c in cat_cols:
        df[c] = df[c].astype("category")
    X = df[feat_cols]
    y = df["y_target"].astype(float)
    return X, y, cat_cols


def align_categoricals(frames: list[pd.DataFrame], cat_cols: list[str]) -> list[pd.DataFrame]:
    """Use the union of categories across all frames as the canonical levels.

    A ticker that appears in test but not train (e.g., a post-IPO ticker
    versus a pre-IPO training window) would otherwise receive a different
    integer code in each ``cat.codes`` mapping, since ``astype('category')``
    infers categories from each frame independently. Aligning to the union
    keeps codes consistent so the LightGBM categorical handling stays valid.
    """
    aligned = [f.copy() for f in frames]
    for c in cat_cols:
        levels = pd.api.types.union_categoricals(
            [f[c] for f in aligned], sort_categories=True
        ).categories
        for f in aligned:
            f[c] = f[c].cat.set_categories(levels)
    return aligned


def split_train_val(train: pd.DataFrame, val_months: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = train.index.get_level_values("date")
    cutoff = dates.max() - pd.DateOffset(months=val_months)
    train_part = train[dates < cutoff]
    val_part = train[dates >= cutoff]
    if val_part.empty:
        n = max(1, int(len(dates.unique()) * 0.1))
        last = dates.unique().sort_values()[-n:]
        val_part = train[dates.isin(last)]
        train_part = train[~dates.isin(last)]
    return train_part, val_part


def _tune(panel: pd.DataFrame, n_trials: int = 50) -> dict:
    dates = panel.index.get_level_values("date")
    tune_panel = panel[(dates >= TUNE_START) & (dates <= TUNE_END)].dropna(subset=["y_target"])
    if tune_panel.empty:
        raise RuntimeError("Tuning window is empty; verify feature matrix coverage.")
    train_part, val_part = split_train_val(tune_panel)
    X_tr, y_tr, cat_cols = prepare(train_part)
    X_va, y_va, _ = prepare(val_part)
    X_tr, X_va = align_categoricals([X_tr, X_va], cat_cols)

    dtrain = lgb.Dataset(
        X_tr, label=y_tr, categorical_feature=cat_cols, params=DATASET_PARAMS, free_raw_data=False
    )
    dval = lgb.Dataset(
        X_va,
        label=y_va,
        categorical_feature=cat_cols,
        params=DATASET_PARAMS,
        reference=dtrain,
        free_raw_data=False,
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "l2",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 500),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
            "seed": SEED,
        }
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        return booster.best_score["valid_0"]["l2"]

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    best.update({"objective": "regression", "metric": "l2", "verbosity": -1, "seed": SEED})
    PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PARAMS_PATH, "w") as f:
        json.dump(best, f, indent=2)
    return best


def _load_params(panel: pd.DataFrame) -> dict:
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH) as f:
            return json.load(f)
    return _tune(panel)


def fit_booster(
    train: pd.DataFrame,
    test: pd.DataFrame,
    params: dict,
    val_months: int = 6,
) -> tuple[lgb.Booster, pd.Series, pd.DataFrame, list[str]]:
    """Fit a LightGBM booster on ``train`` and produce predictions on ``test``.

    Returns the booster, the predictions Series indexed by ``test.index``,
    the (aligned) validation feature frame for downstream artifacts (SHAP,
    feature importance), and the categorical column list.
    """
    train = train.dropna(subset=["y_target"])
    train_part, val_part = split_train_val(train, val_months=val_months)
    X_tr, y_tr, cat_cols = prepare(train_part)
    X_va, y_va, _ = prepare(val_part)
    X_te, _y_te, _ = prepare(test)
    # Force the test feature set to match training column-by-column. Missing
    # columns (rare but possible if a fold drops a constant feature) are
    # filled with 0.0 rather than NaN so LightGBM sees them as zeros instead
    # of "missing" values; mirrors the explicit ``fill_value=0.0`` used in
    # ``models/har.py`` for the same reason.
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0.0)
    X_tr, X_va, X_te = align_categoricals([X_tr, X_va, X_te], cat_cols)

    dtrain = lgb.Dataset(
        X_tr,
        label=y_tr,
        categorical_feature=cat_cols,
        params=DATASET_PARAMS,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        X_va,
        label=y_va,
        categorical_feature=cat_cols,
        params=DATASET_PARAMS,
        reference=dtrain,
        free_raw_data=False,
    )

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    preds = pd.Series(booster.predict(X_te), index=test.index, dtype=float)
    return booster, preds, X_va, cat_cols


def _make_fit_predict(params: dict):
    """Return a fold-aware ``fit_predict`` closure."""

    def fit_predict(
        train: pd.DataFrame,
        test: pd.DataFrame,
        *,
        fold: int = 0,
        n_folds: int = 1,
    ) -> pd.Series:
        booster, preds, X_va, cat_cols = fit_booster(train, test, params)

        model_path = RESULTS_MODELS / f"lgbm_fold_{fold}.txt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(model_path))

        if fold == n_folds - 1:
            _save_shap(booster, X_va.iloc[: min(len(X_va), 5000)], cat_cols)
            _save_feature_importance(booster)

        return preds

    return fit_predict


def _save_shap(booster: lgb.Booster, X: pd.DataFrame, cat_cols: list[str]) -> None:
    """Use LightGBM's native ``pred_contrib`` and pass the values to shap's plot.

    ``shap.TreeExplainer(booster)`` re-runs feature filtering on the booster
    and trips on category mismatches; the booster already exposes Tree-SHAP
    values directly via ``predict(..., pred_contrib=True)``, returning an
    array of shape ``(n_samples, n_features + 1)`` where the trailing column
    is the bias (expected value). Strip the bias column and pass the rest to
    the plotter, which only needs numeric arrays.
    """
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError as exc:
        print(f"SHAP summary skipped (install rvforecast[shap]): {exc}")
        return
    try:
        contribs = booster.predict(X, pred_contrib=True)
        sv = contribs[:, :-1]
        X_plot = X.copy()
        for c in cat_cols:
            X_plot[c] = X_plot[c].cat.codes.astype(float)
        shap.summary_plot(
            sv,
            X_plot,
            feature_names=booster.feature_name(),
            show=False,
            max_display=20,
        )
        fig = plt.gcf()
        fig.tight_layout()
        fig.savefig(RESULTS_FIGURES / "shap_summary.png", dpi=150)
        plt.close(fig)
    except (ValueError, RuntimeError) as exc:
        print(f"SHAP summary skipped: {exc}")


def _save_feature_importance(booster: lgb.Booster) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"Feature importance plot skipped: {exc}")
        return
    gain = booster.feature_importance(importance_type="gain")
    names = booster.feature_name()
    df = pd.DataFrame({"feature": names, "gain": gain}).sort_values("gain", ascending=False)
    df.head(20).iloc[::-1].plot.barh(x="feature", y="gain", legend=False, figsize=(8, 8))
    plt.xlabel("Gain")
    plt.title("LightGBM feature importance (top 20)")
    plt.tight_layout()
    plt.savefig(RESULTS_FIGURES / "feature_importance.png", dpi=150)
    plt.close()
    df.to_csv(RESULTS_TABLES / "feature_importance.csv", index=False)


def main() -> None:
    ensure_output_dirs()
    panel = load_features()
    params = _load_params(panel)
    fit_predict = _make_fit_predict(params)
    out = run_walk_forward(panel, fit_predict, out_name="lgbm")
    print(f"LightGBM predictions written to {out}")


if __name__ == "__main__":
    main()

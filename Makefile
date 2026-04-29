.PHONY: all data features baselines lgbm lstm eval extension holdout report clean cleanall test format lint install

PY := python

PRICES := data/raw/prices_long.parquet
MACRO := data/raw/macro.parquet
FEATURES := data/processed/features.parquet
SPLITS := results/splits.json
PREDICTIONS_DIR := results/predictions

install:
	$(PY) -m pip install -e ".[lstm,shap,dev]"

data: $(PRICES) $(MACRO)

$(PRICES):
	$(PY) -m rvforecast.data.fetch_prices

$(MACRO):
	$(PY) -m rvforecast.data.fetch_macro

features: $(FEATURES)

$(FEATURES): $(PRICES) $(MACRO)
	$(PY) -m rvforecast.features.build_features

baselines: features
	$(PY) -m rvforecast.models.naive
	$(PY) -m rvforecast.models.har
	$(PY) -m rvforecast.models.garch

lgbm: features
	$(PY) -m rvforecast.models.lgbm

lstm: features
	$(PY) -m rvforecast.models.lstm

eval: baselines lgbm
	$(PY) -m rvforecast.evaluation.run_eval
	$(PY) -m rvforecast.evaluation.plots

extension: eval
	$(PY) -m rvforecast.extension.vol_target

holdout: features
	$(PY) -m rvforecast.evaluation.holdout
	$(PY) -m rvforecast.evaluation.run_eval --holdout
	$(PY) -m rvforecast.evaluation.plots --holdout

report:
	@echo "Results written to results/. README has the narrative."

test:
	pytest -q

format:
	black src tests
	ruff check --fix src tests

lint:
	ruff check src tests
	black --check src tests

all: data features baselines lgbm lstm eval extension holdout report

clean:
	rm -rf data/processed/*.parquet results/predictions/*.parquet results/models/* results/figures/* results/tables/* results/holdout/* results/extension/*

# `clean` keeps the (expensive) raw price/macro cache. `cleanall` wipes it
# too, forcing a full re-fetch on the next `make data`.
cleanall: clean
	rm -rf data/raw/*.parquet data/raw/_manifest.json

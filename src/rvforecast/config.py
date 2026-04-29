"""Project-wide constants and configuration.

Centralizes random seeds, date ranges, paths, and target definitions so every
module pulls from a single source of truth.
"""

from __future__ import annotations

from pathlib import Path

SEED: int = 42

# Repository layout. The package lives at <ROOT>/src/rvforecast/, so step up
# two parents from this file to reach the repository root.
ROOT: Path = Path(__file__).resolve().parents[2]
DATA_RAW: Path = ROOT / "data" / "raw"
DATA_PROCESSED: Path = ROOT / "data" / "processed"
RESULTS: Path = ROOT / "results"
RESULTS_FIGURES: Path = RESULTS / "figures"
RESULTS_TABLES: Path = RESULTS / "tables"
RESULTS_PREDICTIONS: Path = RESULTS / "predictions"
RESULTS_MODELS: Path = RESULTS / "models"
RESULTS_HOLDOUT: Path = RESULTS / "holdout"
RESULTS_EXTENSION: Path = RESULTS / "extension"
CONFIGS: Path = ROOT / "configs"

_OUTPUT_DIRS: tuple[Path, ...] = (
    DATA_RAW,
    DATA_PROCESSED,
    RESULTS_FIGURES,
    RESULTS_TABLES,
    RESULTS_PREDICTIONS,
    RESULTS_MODELS,
    RESULTS_HOLDOUT,
    RESULTS_EXTENSION,
)


def ensure_output_dirs() -> None:
    """Create every output directory the pipeline writes into.

    Called from each script's ``main()`` rather than at import time so that
    importing :mod:`rvforecast.config` (e.g., from a test) has no filesystem
    side effects.
    """
    for path in _OUTPUT_DIRS:
        path.mkdir(parents=True, exist_ok=True)


# Default sample window
START_DATE: str = "2005-01-01"
END_DATE: str | None = None  # None means "today" at fetch time

# Final holdout: last N years are touched exactly once
HOLDOUT_YEARS: int = 2

# Walk-forward defaults
INITIAL_TRAIN_YEARS: int = 5
TEST_WINDOW_MONTHS: int = 6
PURGE_DAYS: int = 5
EMBARGO_DAYS: int = 5

# Annualization factor (US trading days per year)
TRADING_DAYS: int = 252

# Default universe size when not loading the full S&P 500
DEFAULT_UNIVERSE_SIZE: int = 50

# Vol-targeting extension
VOL_TARGET_ANNUAL: float = 0.15
LEVERAGE_CAP: float = 2.0

# Feature horizons for HAR decomposition
HAR_HORIZONS: tuple[int, int, int] = (1, 5, 22)

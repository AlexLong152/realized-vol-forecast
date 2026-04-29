"""Shared matplotlib style for evaluation figures."""

from __future__ import annotations

import matplotlib as mpl

PALETTE: dict[str, str] = {
    "naive": "#7f7f7f",
    "har": "#1f77b4",
    "garch": "#2ca02c",
    "lgbm": "#d62728",
    "lstm": "#9467bd",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titleweight": "bold",
            "font.size": 10,
            "legend.frameon": False,
        }
    )


# US recession ranges (NBER) covered by the sample period
RECESSIONS: list[tuple[str, str]] = [
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
]

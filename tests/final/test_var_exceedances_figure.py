"""Tests for VaR exceedances plotting helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from nvda_risk_project.final.var_exceedances_figure import plot_var_exceedances


def test_plot_var_exceedances_returns_bar_figure() -> None:
    exceedances = pd.DataFrame(
        {
            "method": ["historical", "parametric_normal", "historical", "parametric_normal"],
            "alpha": [0.95, 0.95, 0.99, 0.99],
            "exceedance_rate": [0.05, 0.04, 0.01, 0.015],
            "status": ["ok", "ok", "ok", "ok"],
        },
    )
    fig = plot_var_exceedances(exceedances)
    assert len(fig.data) >= 1
    assert fig.data[0].type == "bar"


def test_plot_var_exceedances_rejects_missing_columns() -> None:
    exceedances = pd.DataFrame({"alpha": [0.95, 0.99]})
    with pytest.raises(ValueError, match="must contain columns"):
        plot_var_exceedances(exceedances)


def test_plot_var_exceedances_rejects_empty_input_after_cleaning() -> None:
    exceedances = pd.DataFrame(
        {
            "alpha": [0.95, 0.99],
            "exceedance_rate": [None, None],
            "status": ["fallback", "fallback"],
        },
    )
    with pytest.raises(ValueError, match="empty after cleaning"):
        plot_var_exceedances(exceedances)

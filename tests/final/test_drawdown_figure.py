"""Tests for drawdown plotting helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from nvda_risk_project.final.drawdown_figure import plot_drawdown_curve


def test_plot_drawdown_curve_returns_line_figure() -> None:
    panel_monthly = pd.DataFrame(
        {
            "month": pd.date_range("2024-01-31", periods=4, freq="ME"),
            "nvda_return": [0.05, -0.03, 0.02, -0.01],
        },
    )
    fig = plot_drawdown_curve(panel_monthly)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"


def test_plot_drawdown_curve_rejects_missing_columns() -> None:
    panel_monthly = pd.DataFrame({"month": pd.date_range("2024-01-31", periods=3, freq="ME")})
    with pytest.raises(ValueError, match="must contain columns"):
        plot_drawdown_curve(panel_monthly)


def test_plot_drawdown_curve_rejects_empty_input_after_cleaning() -> None:
    panel_monthly = pd.DataFrame(
        {
            "month": pd.date_range("2024-01-31", periods=3, freq="ME"),
            "nvda_return": [None, None, None],
        },
    )
    with pytest.raises(ValueError, match="empty after cleaning"):
        plot_drawdown_curve(panel_monthly)

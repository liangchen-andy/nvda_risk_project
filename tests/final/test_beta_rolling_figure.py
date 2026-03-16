"""Tests for rolling beta plotting helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from nvda_risk_project.final.beta_rolling_figure import plot_beta_rolling_curve


def test_plot_beta_rolling_curve_returns_line_figure() -> None:
    panel_monthly = pd.DataFrame(
        {
            "month": pd.date_range("2023-01-31", periods=15, freq="ME"),
            "nvda_return": [0.04, 0.02, -0.01, 0.03, 0.01] * 3,
            "market_return": [0.02, 0.01, -0.005, 0.015, 0.008] * 3,
        },
    )
    fig = plot_beta_rolling_curve(panel_monthly, window=12)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"


def test_plot_beta_rolling_curve_rejects_missing_columns() -> None:
    panel_monthly = pd.DataFrame(
        {
            "month": pd.date_range("2024-01-31", periods=3, freq="ME"),
            "nvda_return": [0.01, 0.02, 0.03],
        },
    )
    with pytest.raises(ValueError, match="must contain columns"):
        plot_beta_rolling_curve(panel_monthly)


def test_plot_beta_rolling_curve_rejects_invalid_window() -> None:
    panel_monthly = pd.DataFrame(
        {
            "month": pd.date_range("2024-01-31", periods=4, freq="ME"),
            "nvda_return": [0.01, 0.02, 0.03, 0.01],
            "market_return": [0.005, 0.01, 0.015, 0.01],
        },
    )
    with pytest.raises(ValueError, match="greater than 1"):
        plot_beta_rolling_curve(panel_monthly, window=1)


def test_plot_beta_rolling_curve_rejects_empty_after_window() -> None:
    panel_monthly = pd.DataFrame(
        {
            "month": pd.date_range("2024-01-31", periods=3, freq="ME"),
            "nvda_return": [0.01, 0.02, 0.03],
            "market_return": [0.005, 0.01, 0.015],
        },
    )
    with pytest.raises(ValueError, match="empty after applying window"):
        plot_beta_rolling_curve(panel_monthly, window=12)

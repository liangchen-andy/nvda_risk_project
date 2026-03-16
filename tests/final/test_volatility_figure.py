"""Tests for volatility plotting helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from nvda_risk_project.final.volatility_figure import plot_volatility_series


def test_plot_volatility_series_returns_line_figure() -> None:
    rolling_vol = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=4, freq="B"),
            "rolling_vol": [0.2, 0.21, 0.22, 0.19],
        },
    )
    fig = plot_volatility_series(rolling_vol)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"


def test_plot_volatility_series_rejects_missing_columns() -> None:
    rolling_vol = pd.DataFrame({"date": pd.date_range("2024-01-02", periods=3, freq="B")})
    with pytest.raises(ValueError, match="must contain columns"):
        plot_volatility_series(rolling_vol)


def test_plot_volatility_series_rejects_empty_input_after_cleaning() -> None:
    rolling_vol = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "rolling_vol": [None, None, None],
        },
    )
    with pytest.raises(ValueError, match="empty after cleaning"):
        plot_volatility_series(rolling_vol)

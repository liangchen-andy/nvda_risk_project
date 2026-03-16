"""Tests for market-risk overview plotting helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from nvda_risk_project.final.market_figure import plot_market_risk_overview


def test_plot_market_risk_overview_returns_bar_figure() -> None:
    summary = pd.DataFrame(
        {
            "risk_dimension": ["market", "market", "liquidity"],
            "metric": ["vol", "var", "amihud"],
            "value": [0.2, -0.03, 1.2e-9],
        },
    )
    fig = plot_market_risk_overview(summary)
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"


def test_plot_market_risk_overview_rejects_missing_columns() -> None:
    summary = pd.DataFrame({"risk_dimension": ["market"], "value": [0.2]})
    with pytest.raises(ValueError, match="must contain columns"):
        plot_market_risk_overview(summary)

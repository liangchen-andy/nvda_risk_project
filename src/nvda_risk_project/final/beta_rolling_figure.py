"""Helpers to build rolling-beta figures from monthly panel artifacts."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_beta_rolling_curve(panel_monthly: pd.DataFrame, window: int = 12) -> go.Figure:
    """Create rolling CAPM beta curve from monthly NVDA and market returns."""
    required_columns = {"month", "nvda_return", "market_return"}
    if not required_columns.issubset(panel_monthly.columns):
        msg = f"panel_monthly data must contain columns {sorted(required_columns)}."
        raise ValueError(msg)
    if window <= 1:
        msg = "window must be greater than 1."
        raise ValueError(msg)

    plot_data = panel_monthly.copy()
    plot_data["month"] = pd.to_datetime(plot_data["month"], errors="coerce")
    plot_data["nvda_return"] = pd.to_numeric(plot_data["nvda_return"], errors="coerce")
    plot_data["market_return"] = pd.to_numeric(plot_data["market_return"], errors="coerce")
    plot_data = plot_data.dropna(subset=["month", "nvda_return", "market_return"]).sort_values(
        "month",
    )
    if plot_data.empty:
        msg = "panel_monthly data is empty after cleaning."
        raise ValueError(msg)

    cov = plot_data["nvda_return"].rolling(window=window, min_periods=window).cov(
        plot_data["market_return"],
    )
    var = plot_data["market_return"].rolling(window=window, min_periods=window).var()
    plot_data["beta_rolling"] = cov / var.replace(0, pd.NA)
    plot_data = plot_data.dropna(subset=["beta_rolling"])
    if plot_data.empty:
        msg = "rolling beta is empty after applying window."
        raise ValueError(msg)

    fig = px.line(
        plot_data,
        x="month",
        y="beta_rolling",
        labels={"month": "Month", "beta_rolling": "Rolling Beta"},
        title="NVDA Rolling CAPM Beta",
    )
    fig.update_layout(template="plotly_white+presentation")
    return fig

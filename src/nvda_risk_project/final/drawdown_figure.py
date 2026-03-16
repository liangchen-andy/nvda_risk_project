"""Helpers to build drawdown figures from monthly panel artifacts."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_drawdown_curve(panel_monthly: pd.DataFrame) -> go.Figure:
    """Create a drawdown curve from monthly NVDA returns."""
    required_columns = {"month", "nvda_return"}
    if not required_columns.issubset(panel_monthly.columns):
        msg = f"panel_monthly data must contain columns {sorted(required_columns)}."
        raise ValueError(msg)

    plot_data = panel_monthly.copy()
    plot_data["month"] = pd.to_datetime(plot_data["month"], errors="coerce")
    plot_data["nvda_return"] = pd.to_numeric(plot_data["nvda_return"], errors="coerce")
    plot_data = plot_data.dropna(subset=["month", "nvda_return"]).sort_values("month")
    if plot_data.empty:
        msg = "panel_monthly data is empty after cleaning."
        raise ValueError(msg)

    cumulative = (1.0 + plot_data["nvda_return"]).cumprod()
    running_peak = cumulative.cummax()
    plot_data["drawdown"] = cumulative / running_peak - 1.0

    fig = px.line(
        plot_data,
        x="month",
        y="drawdown",
        labels={"month": "Month", "drawdown": "Drawdown"},
        title="NVDA Drawdown Curve",
    )
    fig.update_layout(template="plotly_white+presentation")
    return fig

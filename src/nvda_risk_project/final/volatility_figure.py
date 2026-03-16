"""Helpers to build volatility figures from rolling-volatility artifacts."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_volatility_series(rolling_vol: pd.DataFrame) -> go.Figure:
    """Create a line chart for rolling volatility."""
    required_columns = {"date", "rolling_vol"}
    if not required_columns.issubset(rolling_vol.columns):
        msg = f"rolling_vol data must contain columns {sorted(required_columns)}."
        raise ValueError(msg)

    plot_data = rolling_vol.copy()
    plot_data["date"] = pd.to_datetime(plot_data["date"], errors="coerce")
    plot_data = plot_data.dropna(subset=["date", "rolling_vol"]).sort_values("date")
    if plot_data.empty:
        msg = "rolling_vol data is empty after cleaning."
        raise ValueError(msg)

    fig = px.line(
        plot_data,
        x="date",
        y="rolling_vol",
        labels={"date": "Date", "rolling_vol": "Rolling Volatility"},
        title="NVDA Rolling Volatility",
    )
    fig.update_layout(template="plotly_white+presentation")
    return fig

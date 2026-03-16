"""Helpers to build market-risk overview figures from summary artifacts."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_market_risk_overview(summary: pd.DataFrame) -> go.Figure:
    """Create a bar chart summarizing average absolute metric values by dimension."""
    required_columns = {"risk_dimension", "metric", "value"}
    if not required_columns.issubset(summary.columns):
        msg = f"risk summary must contain columns {sorted(required_columns)}."
        raise ValueError(msg)

    plot_data = summary.copy()
    plot_data["risk_dimension"] = plot_data["risk_dimension"].astype(str).str.strip()
    plot_data["value"] = pd.to_numeric(plot_data["value"], errors="coerce")
    plot_data = plot_data.dropna(subset=["risk_dimension", "value"])
    plot_data = plot_data.loc[plot_data["risk_dimension"] != ""]
    if plot_data.empty:
        msg = "risk summary is empty after cleaning."
        raise ValueError(msg)

    overview = (
        plot_data.assign(abs_value=plot_data["value"].abs())
        .groupby("risk_dimension", as_index=False)["abs_value"]
        .mean()
        .rename(columns={"abs_value": "avg_abs_metric_value"})
        .sort_values("risk_dimension")
    )
    fig = px.bar(
        overview,
        x="risk_dimension",
        y="avg_abs_metric_value",
        labels={
            "risk_dimension": "Risk Dimension",
            "avg_abs_metric_value": "Average Absolute Metric Value",
        },
        title="NVDA Risk Overview by Dimension",
    )
    fig.update_layout(template="plotly_white+presentation")
    return fig

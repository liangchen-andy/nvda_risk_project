"""Helpers to build VaR exceedance figures from backtest artifacts."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_var_exceedances(exceedances: pd.DataFrame) -> go.Figure:
    """Create grouped bars for exceedance rates by alpha and method."""
    required_columns = {"alpha", "exceedance_rate"}
    if not required_columns.issubset(exceedances.columns):
        msg = f"exceedances data must contain columns {sorted(required_columns)}."
        raise ValueError(msg)

    plot_data = exceedances.copy()
    plot_data["alpha"] = pd.to_numeric(plot_data["alpha"], errors="coerce")
    plot_data["exceedance_rate"] = pd.to_numeric(plot_data["exceedance_rate"], errors="coerce")
    if "method" not in plot_data.columns:
        plot_data["method"] = "historical"
    plot_data["method"] = plot_data["method"].astype(str).str.strip().str.lower()
    if "status" in plot_data.columns:
        status = plot_data["status"].astype(str).str.strip().str.lower()
        plot_data = plot_data.loc[status == "ok"]

    plot_data = plot_data.dropna(subset=["alpha", "exceedance_rate"])
    if plot_data.empty:
        msg = "exceedances data is empty after cleaning."
        raise ValueError(msg)

    plot_data["alpha_label"] = plot_data["alpha"].map(lambda x: f"{x:.2f}")
    fig = px.bar(
        plot_data,
        x="alpha_label",
        y="exceedance_rate",
        color="method",
        barmode="group",
        labels={
            "alpha_label": "Confidence Level",
            "exceedance_rate": "Exceedance Rate",
            "method": "Method",
        },
        title="VaR Exceedance Rates by Method",
    )
    fig.update_layout(template="plotly_white+presentation")
    return fig

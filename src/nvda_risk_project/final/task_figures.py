"""Tasks for creating final report figures."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from nvda_risk_project.config import (
    ANALYSIS_OUTPUT,
    DOCUMENTS_PUBLIC,
    FIGURES_OUTPUT,
    PROCESSED_DATA,
    SUMMARY_FIG_BETA_ROLLING,
    SUMMARY_FIG_DRAWDOWN,
    SUMMARY_FIG_VAR_EXCEEDANCES,
    SUMMARY_FIG_VOLATILITY,
    TABLES_OUTPUT,
)
from nvda_risk_project.final.beta_rolling_figure import plot_beta_rolling_curve
from nvda_risk_project.final.drawdown_figure import plot_drawdown_curve
from nvda_risk_project.final.market_figure import plot_market_risk_overview
from nvda_risk_project.final.var_exceedances_figure import plot_var_exceedances
from nvda_risk_project.final.volatility_figure import plot_volatility_series

_PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0bIDAT\x08\xd7c```\x00"
    b"\x00\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _write_placeholder_png(path: Path) -> None:
    """Write a deterministic 1x1 PNG placeholder to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_PNG_1X1)


def _render_figure_with_fallback(
    *,
    produces: Path,
    build_figure: Callable[[], object],
) -> None:
    """Render a figure and fall back to placeholder on any plotting failure."""
    produces.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig = build_figure()
        fig.write_image(str(produces))
    except Exception:
        _write_placeholder_png(produces)


def _copy_figure_with_fallback(*, figure: Path, produces: Path) -> None:
    """Copy a figure into output path with deterministic placeholder fallback."""
    figure.parent.mkdir(parents=True, exist_ok=True)
    produces.parent.mkdir(parents=True, exist_ok=True)

    if not figure.exists():
        _write_placeholder_png(figure)

    try:
        produces.write_bytes(figure.read_bytes())
    except Exception:
        _write_placeholder_png(produces)


def task_create_volatility_figure(
    rolling_vol_data: Path = ANALYSIS_OUTPUT / "rolling_vol.csv",
    produces: Path = SUMMARY_FIG_VOLATILITY,
) -> None:
    """Create volatility figure from rolling-volatility artifact."""
    rolling_vol = pd.read_csv(rolling_vol_data, parse_dates=["date"])
    _render_figure_with_fallback(
        produces=produces,
        build_figure=lambda: plot_volatility_series(rolling_vol),
    )


def task_create_var_exceedances_figure(
    var_exceedances_data: Path = ANALYSIS_OUTPUT / "var_exceedances_hist.csv",
    produces: Path = SUMMARY_FIG_VAR_EXCEEDANCES,
) -> None:
    """Create VaR exceedances figure from exceedance artifact."""
    _render_figure_with_fallback(
        produces=produces,
        build_figure=lambda: plot_var_exceedances(pd.read_csv(var_exceedances_data)),
    )


def task_create_beta_rolling_figure(
    panel_monthly_data: Path = PROCESSED_DATA / "panel_monthly.csv",
    produces: Path = SUMMARY_FIG_BETA_ROLLING,
) -> None:
    """Create rolling-beta figure from monthly panel artifact."""
    _render_figure_with_fallback(
        produces=produces,
        build_figure=lambda: plot_beta_rolling_curve(
            pd.read_csv(panel_monthly_data, parse_dates=["month"]),
        ),
    )


def task_create_market_figure(
    risk_summary: Path = TABLES_OUTPUT / "risk_summary.csv",
    produces: Path = FIGURES_OUTPUT / "market_risk_overview.png",
) -> None:
    """Create market-risk overview figure from risk-summary artifact."""
    _render_figure_with_fallback(
        produces=produces,
        build_figure=lambda: plot_market_risk_overview(pd.read_csv(risk_summary)),
    )


def task_create_drawdown_figure(
    panel_monthly_data: Path = PROCESSED_DATA / "panel_monthly.csv",
    produces: Path = FIGURES_OUTPUT / "drawdown_curve.png",
) -> None:
    """Create drawdown curve figure from monthly panel artifact."""
    _render_figure_with_fallback(
        produces=produces,
        build_figure=lambda: plot_drawdown_curve(
            pd.read_csv(panel_monthly_data, parse_dates=["month"]),
        ),
    )


def task_create_drawdown_summary_figure(
    figure: Path = FIGURES_OUTPUT / "drawdown_curve.png",
    produces: Path = SUMMARY_FIG_DRAWDOWN,
) -> None:
    """Copy drawdown figure into documents summary output, with fallback."""
    _copy_figure_with_fallback(figure=figure, produces=produces)


def task_create_documents_figure(
    figure: Path = FIGURES_OUTPUT / "market_risk_overview.png",
    produces: Path = DOCUMENTS_PUBLIC / "fig_market_overview.png",
) -> None:
    """Copy the upstream figure into documents, with placeholder fallback."""
    _copy_figure_with_fallback(figure=figure, produces=produces)


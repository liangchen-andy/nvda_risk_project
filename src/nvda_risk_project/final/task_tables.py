"""Tasks for creating final summary tables."""

from pathlib import Path

import pandas as pd

from nvda_risk_project.config import ANALYSIS_OUTPUT, DOCUMENTS_TABLES, RISK_DIMENSIONS, TABLES_OUTPUT


def task_create_risk_summary_table(
    market: Path = ANALYSIS_OUTPUT / "market_risk.csv",
    liquidity: Path = ANALYSIS_OUTPUT / "liquidity_risk.csv",
    drawdown: Path = ANALYSIS_OUTPUT / "drawdown_risk.csv",
    systematic: Path = ANALYSIS_OUTPUT / "systematic_risk.csv",
    macro: Path = ANALYSIS_OUTPUT / "macro_risk.csv",
    produces: Path = TABLES_OUTPUT / "risk_summary.csv",
) -> None:
    """Combine risk module outputs into one analysis table."""
    produces.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.concat(
        [pd.read_csv(ANALYSIS_OUTPUT / f"{dim}_risk.csv") for dim in RISK_DIMENSIONS],
        ignore_index=True,
    )
    summary = summary[["risk_dimension", "metric", "value"]]
    summary.to_csv(produces, index=False)


def task_create_documents_table(
    summary: Path = TABLES_OUTPUT / "risk_summary.csv",
    produces: Path = DOCUMENTS_TABLES / "estimation_results.md",
) -> None:
    """Create a markdown table for the document build pipeline."""
    produces.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.read_csv(summary)
    with produces.open("w", encoding="utf-8") as file:
        file.write(summary_df.to_markdown(index=False))


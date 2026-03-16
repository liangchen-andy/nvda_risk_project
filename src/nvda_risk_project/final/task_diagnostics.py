"""Tasks for creating reproducibility diagnostics artifacts."""

from pathlib import Path

import pandas as pd

from nvda_risk_project.config import (
    ANALYSIS_OUTPUT,
    CHECKS_OUTPUT,
    DATA_PROVENANCE,
    DIAGNOSTICS_DATA,
    DIAGNOSTICS_TABLE,
    DOCUMENTS_TABLES,
    PANEL_DAILY_DATA,
    RAW_DATA,
    TABLES_OUTPUT,
    VOL_WINDOW_DAYS,
)
from nvda_risk_project.final.diagnostics import (
    build_diagnostics_table,
    render_diagnostics_markdown,
)


def task_create_diagnostics_data(
    risk_summary: Path = TABLES_OUTPUT / "risk_summary.csv",
    scorecard: Path = CHECKS_OUTPUT / "scorecard.json",
    nvda_raw: Path = RAW_DATA / "nvda_daily.csv",
    market_raw: Path = RAW_DATA / "sp500_daily.csv",
    rolling_vol: Path = ANALYSIS_OUTPUT / "rolling_vol.csv",
    var_es_hist: Path = ANALYSIS_OUTPUT / "var_es_hist.csv",
    var_exceedances_hist: Path = ANALYSIS_OUTPUT / "var_exceedances_hist.csv",
    var_backtest_hist: Path = ANALYSIS_OUTPUT / "var_backtest_hist.csv",
    macro_raw: Path = RAW_DATA / "macro_monthly.csv",
    provenance: Path = DATA_PROVENANCE,
    panel_daily: Path = PANEL_DAILY_DATA,
    produces: Path = DIAGNOSTICS_DATA,
) -> None:
    """Generate diagnostics key-value rows from upstream pipeline artifacts."""
    produces.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = build_diagnostics_table(
        risk_summary_path=risk_summary,
        scorecard_path=scorecard,
        nvda_raw_path=nvda_raw,
        market_raw_path=market_raw,
        rolling_vol_path=rolling_vol,
        var_es_hist_path=var_es_hist,
        var_exceedances_hist_path=var_exceedances_hist,
        var_backtest_hist_path=var_backtest_hist,
        vol_window_days=VOL_WINDOW_DAYS,
        provenance_path=provenance,
        panel_daily_path=panel_daily,
        macro_raw_path=macro_raw,
    )
    diagnostics.to_csv(produces, index=False)


def task_create_diagnostics_table(
    diagnostics_data: Path = DIAGNOSTICS_DATA,
    produces: Path = DIAGNOSTICS_TABLE,
) -> None:
    """Render diagnostics in markdown for the documents folder."""
    DOCUMENTS_TABLES.mkdir(parents=True, exist_ok=True)
    diagnostics = pd.read_csv(diagnostics_data)
    markdown = render_diagnostics_markdown(diagnostics)
    with produces.open("w", encoding="utf-8") as file:
        file.write(markdown)

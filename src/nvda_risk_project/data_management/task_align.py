"""Tasks for aligning cleaned data into analysis-ready panels."""

from pathlib import Path

import pandas as pd

from nvda_risk_project.config import INTERIM_DATA, PROCESSED_DATA


def task_build_monthly_panel(
    nvda_data: Path = INTERIM_DATA / "nvda_daily.csv",
    sp500_data: Path = INTERIM_DATA / "sp500_daily.csv",
    macro_data: Path = INTERIM_DATA / "macro_monthly.csv",
    produces: Path = PROCESSED_DATA / "panel_monthly.csv",
) -> None:
    """Create a month-end panel combining returns, liquidity proxies, and macro values."""
    produces.parent.mkdir(parents=True, exist_ok=True)

    nvda = pd.read_csv(nvda_data, parse_dates=["date"]).copy()
    sp500 = pd.read_csv(sp500_data, parse_dates=["date"]).copy()
    macro = pd.read_csv(macro_data, parse_dates=["date"]).copy()

    nvda["month"] = pd.to_datetime(nvda["date"]).dt.to_period("M").dt.to_timestamp("M")
    sp500["month"] = pd.to_datetime(sp500["date"]).dt.to_period("M").dt.to_timestamp("M")
    macro["month"] = pd.to_datetime(macro["date"]).dt.to_period("M").dt.to_timestamp("M")

    nvda_monthly = (
        nvda.groupby("month", as_index=False)
        .agg(
            nvda_return=("return", "sum"),
            amihud_illiq=("amihud_illiq", "mean"),
            dollar_volume=("dollar_volume", "mean"),
        )
    )
    sp500_monthly = sp500.groupby("month", as_index=False).agg(market_return=("return", "sum"))
    macro_monthly = macro.drop(columns=["date"])

    panel = nvda_monthly.merge(sp500_monthly, on="month", how="inner").merge(
        macro_monthly,
        on="month",
        how="inner",
    )
    panel.to_csv(produces, index=False)


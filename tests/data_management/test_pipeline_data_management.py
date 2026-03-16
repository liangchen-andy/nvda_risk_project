"""Tests for the Milestone 1 data-management skeleton."""

from pathlib import Path

import pandas as pd

from nvda_risk_project.data_management.task_align import task_build_monthly_panel


def test_monthly_panel_contains_expected_columns(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"
    raw.mkdir()
    interim.mkdir()

    dates = pd.date_range("2024-01-02", periods=40, freq="B")
    nvda = pd.DataFrame(
        {
            "date": dates,
            "return": 0.001,
            "amihud_illiq": 1e-9,
            "dollar_volume": 2_000_000.0,
        },
    )
    sp500 = pd.DataFrame({"date": dates, "return": 0.0005})
    macro = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-31", periods=3, freq="ME"),
            "gdp_growth": [0.02, 0.02, 0.02],
            "inflation_yoy": [0.03, 0.03, 0.03],
            "policy_rate": [0.05, 0.05, 0.05],
        },
    )

    nvda_path = interim / "nvda_daily.csv"
    sp500_path = interim / "sp500_daily.csv"
    macro_path = interim / "macro_monthly.csv"
    out_path = processed / "panel_monthly.csv"
    nvda.to_csv(nvda_path, index=False)
    sp500.to_csv(sp500_path, index=False)
    macro.to_csv(macro_path, index=False)

    task_build_monthly_panel(
        nvda_data=nvda_path,
        sp500_data=sp500_path,
        macro_data=macro_path,
        produces=out_path,
    )

    out = pd.read_csv(out_path)
    expected = {
        "month",
        "nvda_return",
        "amihud_illiq",
        "dollar_volume",
        "market_return",
        "gdp_growth",
        "inflation_yoy",
        "policy_rate",
    }
    assert expected.issubset(out.columns)
    assert not out.empty


"""Tests for macro risk frequency and schema validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nvda_risk_project.analysis.macro_risk import build_daily_dgs10_panel
from nvda_risk_project.analysis.macro_risk import estimate_macro_risk


def _panel(month: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month": month,
            "nvda_return": np.linspace(-0.02, 0.03, len(month)),
            "gdp_growth": np.linspace(0.02, 0.025, len(month)),
            "inflation_yoy": np.linspace(0.03, 0.028, len(month)),
            "policy_rate": np.linspace(0.055, 0.05, len(month)),
        },
    )


def test_estimate_macro_risk_rejects_non_month_end_dates() -> None:
    panel = _panel(pd.Series(pd.date_range("2024-01-01", periods=6, freq="B")))

    with pytest.raises(ValueError, match="month-end monthly data"):
        estimate_macro_risk(panel)


def test_estimate_macro_risk_rejects_duplicate_month_rows() -> None:
    month = pd.Series(pd.date_range("2024-01-31", periods=6, freq="ME"))
    panel = _panel(month)
    panel.loc[1, "month"] = panel.loc[0, "month"]

    with pytest.raises(ValueError, match="contains duplicates"):
        estimate_macro_risk(panel)


def test_estimate_macro_risk_rejects_short_monthly_sample() -> None:
    panel = _panel(pd.Series(pd.date_range("2024-01-31", periods=3, freq="ME")))

    with pytest.raises(ValueError, match="at least 4 monthly observations"):
        estimate_macro_risk(panel)


def test_build_daily_dgs10_panel_aligns_monthly_macro_to_daily_returns() -> None:
    panel_daily = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=6, freq="B"),
            "ret": [0.01, -0.02, 0.005, 0.01, -0.01, 0.02],
        },
    )
    macro_monthly = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-12-31", "2024-01-31"]),
            "policy_rate": [0.05, 0.052],
        },
    )

    out = build_daily_dgs10_panel(panel_daily=panel_daily, macro_monthly=macro_monthly)
    assert {"date", "ret", "dgs10"} == set(out.columns)
    assert out.shape[0] == panel_daily.shape[0]
    assert out["dgs10"].notna().all()


def test_estimate_macro_risk_daily_dgs10_path_returns_expected_metrics() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=80, freq="B"),
            "ret": np.linspace(-0.015, 0.02, 80),
            "dgs10": np.linspace(3.8, 4.2, 80),
        },
    )

    out = estimate_macro_risk(panel)
    assert {"metric", "value"} == set(out.columns)
    assert not out.empty
    assert set(out["metric"]) == {
        "intercept",
        "beta_dgs10",
        "beta_dgs10_tstat",
        "beta_dgs10_pvalue",
        "r_squared",
        "obs_count",
    }
    assert out["value"].notna().all()


def test_estimate_macro_risk_rejects_missing_monthly_columns() -> None:
    panel = pd.DataFrame(
        {
            "month": pd.date_range("2024-01-31", periods=6, freq="ME"),
            "nvda_return": np.linspace(-0.02, 0.03, 6),
            "gdp_growth": np.linspace(0.02, 0.025, 6),
            "inflation_yoy": np.linspace(0.03, 0.028, 6),
        },
    )

    with pytest.raises(ValueError, match="missing required columns"):
        estimate_macro_risk(panel)


def test_estimate_macro_risk_rejects_invalid_month_values() -> None:
    panel = _panel(pd.Series(["2024-01-31", "invalid", "2024-03-31", "2024-04-30"]))

    with pytest.raises(ValueError, match="valid datetime"):
        estimate_macro_risk(panel)


def test_estimate_macro_risk_daily_rejects_duplicate_dates() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"] * 12),
            "ret": np.linspace(-0.01, 0.01, 36),
            "dgs10": np.linspace(3.8, 4.0, 36),
        },
    )

    with pytest.raises(ValueError, match="contains duplicates"):
        estimate_macro_risk(panel)


def test_estimate_macro_risk_daily_rejects_short_sample() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=20, freq="B"),
            "ret": np.linspace(-0.01, 0.01, 20),
            "dgs10": np.linspace(3.8, 4.0, 20),
        },
    )

    with pytest.raises(ValueError, match="at least 30 observations"):
        estimate_macro_risk(panel)


def test_estimate_macro_risk_daily_rejects_empty_after_clean() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=40, freq="B"),
            "ret": [np.nan] * 40,
            "dgs10": [np.nan] * 40,
        },
    )

    with pytest.raises(ValueError, match="empty after cleaning"):
        estimate_macro_risk(panel)


def test_build_daily_dgs10_panel_rejects_missing_columns() -> None:
    panel_daily = pd.DataFrame({"date": pd.date_range("2024-01-02", periods=3, freq="B")})
    macro_monthly = pd.DataFrame({"date": pd.to_datetime(["2023-12-31"]), "policy_rate": [0.05]})

    with pytest.raises(ValueError, match="panel_daily is missing required columns"):
        build_daily_dgs10_panel(panel_daily=panel_daily, macro_monthly=macro_monthly)

    panel_daily = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "ret": [0.01, -0.01, 0.02],
        },
    )
    macro_monthly = pd.DataFrame({"date": pd.to_datetime(["2023-12-31"])})

    with pytest.raises(ValueError, match="macro_monthly is missing required columns"):
        build_daily_dgs10_panel(panel_daily=panel_daily, macro_monthly=macro_monthly)


def test_build_daily_dgs10_panel_rejects_empty_inputs() -> None:
    panel_daily = pd.DataFrame({"date": [pd.NaT], "ret": [np.nan]})
    macro_monthly = pd.DataFrame({"date": [pd.NaT], "policy_rate": [np.nan]})

    with pytest.raises(ValueError, match="cannot build daily DGS10 panel from empty inputs"):
        build_daily_dgs10_panel(panel_daily=panel_daily, macro_monthly=macro_monthly)

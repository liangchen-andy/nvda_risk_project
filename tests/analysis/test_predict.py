"""Tests for task wiring in the Milestone 1 analysis skeleton."""

from pathlib import Path

import numpy as np
import pandas as pd

from nvda_risk_project.analysis.task_risk_analysis import (
    RISK_FUNCTIONS,
    task_create_historical_var_exceedances,
    task_create_historical_var_es,
    task_create_rolling_volatility,
)
from nvda_risk_project.config import RISK_DIMENSIONS


def test_risk_function_mapping_covers_all_dimensions() -> None:
    assert set(RISK_FUNCTIONS) == set(RISK_DIMENSIONS)


def test_each_risk_function_returns_non_empty_frame() -> None:
    panel = pd.DataFrame(
        {
            "month": pd.date_range("2024-01-31", periods=6, freq="ME"),
            "nvda_return": [0.01, -0.02, 0.03, 0.01, -0.01, 0.02],
            "market_return": [0.005, -0.01, 0.02, 0.008, -0.007, 0.015],
            "amihud_illiq": [1e-9, 2e-9, 1.5e-9, 1.8e-9, 2.1e-9, 1.7e-9],
            "dollar_volume": [2e6, 2.2e6, 2.5e6, 2.1e6, 2.0e6, 2.3e6],
            "gdp_growth": [0.02, 0.021, 0.021, 0.022, 0.023, 0.023],
            "inflation_yoy": [0.035, 0.034, 0.033, 0.032, 0.031, 0.03],
            "policy_rate": [0.055, 0.055, 0.054, 0.053, 0.052, 0.051],
        },
    )

    for dimension in RISK_DIMENSIONS:
        output = RISK_FUNCTIONS[dimension](panel)
        assert not output.empty
        assert {"metric", "value"} == set(output.columns)


def test_task_create_rolling_volatility_writes_csv(tmp_path: Path) -> None:
    panel_daily = tmp_path / "panel_daily.parquet"
    output_path = tmp_path / "rolling_vol.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=5, freq="B"),
            "ret": [0.0, 0.01, -0.02, 0.015, 0.005],
        },
    ).to_parquet(panel_daily, index=False)

    task_create_rolling_volatility(panel_daily_data=panel_daily, produces=output_path)

    out = pd.read_csv(output_path)
    assert {"date", "rolling_vol"} == set(out.columns)
    assert out.shape[0] == 5


def test_task_create_historical_var_es_includes_garch_method_with_short_sample_fallback(tmp_path: Path) -> None:
    panel_daily = tmp_path / "panel_daily.parquet"
    output_path = tmp_path / "var_es_hist.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=6, freq="B"),
            "ret": [0.0, -0.01, 0.02, -0.03, 0.01, -0.02],
        },
    ).to_parquet(panel_daily, index=False)

    task_create_historical_var_es(panel_daily_data=panel_daily, produces=output_path)

    out = pd.read_csv(output_path)
    assert {
        "method",
        "alpha",
        "var",
        "es",
        "status",
        "fallback_reason",
        "sample_size",
        "nu",
        "sigma_next",
        "garch_converged",
    } == set(out.columns)
    assert set(out["alpha"]) == {0.95, 0.99}
    assert set(out["method"]) == {"historical", "parametric_normal", "garch_t"}
    assert set(out.loc[out["method"].isin(["historical", "parametric_normal"]), "status"]) == {"ok"}
    assert set(
        out.loc[out["method"].isin(["historical", "parametric_normal"]), "fallback_reason"],
    ) == {"none"}
    assert set(out.loc[out["method"] == "garch_t", "status"]) == {"fallback"}
    assert set(out.loc[out["method"] == "garch_t", "fallback_reason"]) == {"short_sample"}
    assert set(out.loc[out["method"] == "garch_t", "garch_converged"]) == {False}


def test_task_create_historical_var_es_garch_method_succeeds_on_long_sample(tmp_path: Path) -> None:
    panel_daily = tmp_path / "panel_daily.parquet"
    output_path = tmp_path / "var_es_hist.csv"
    returns = pd.Series(np.linspace(-0.03, 0.03, 260), dtype=float)
    pd.DataFrame(
        {
            "date": pd.date_range("2020-01-02", periods=260, freq="B"),
            "ret": returns,
        },
    ).to_parquet(panel_daily, index=False)

    task_create_historical_var_es(panel_daily_data=panel_daily, produces=output_path)

    out = pd.read_csv(output_path)
    garch_rows = out.loc[out["method"] == "garch_t"]
    assert garch_rows.shape[0] == 2
    assert set(garch_rows["status"]) == {"ok"}
    assert set(garch_rows["fallback_reason"]) == {"none"}
    assert set(garch_rows["garch_converged"]) == {True}
    assert garch_rows["nu"].gt(4.0).all()
    assert garch_rows["sigma_next"].gt(0.0).all()


def test_task_create_historical_var_exceedances_writes_expected_rows(tmp_path: Path) -> None:
    panel_daily = tmp_path / "panel_daily.parquet"
    var_es_hist = tmp_path / "var_es_hist.csv"
    output_path = tmp_path / "var_exceedances_hist.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=5, freq="B"),
            "ret": [0.0, -0.01, -0.02, -0.03, 0.01],
        },
    ).to_parquet(panel_daily, index=False)
    pd.DataFrame(
        {
            "method": ["historical", "historical"],
            "alpha": [0.95, 0.99],
            "var": [-0.02, -0.03],
            "es": [-0.025, -0.03],
            "status": ["ok", "ok"],
            "sample_size": [5, 5],
        },
    ).to_csv(var_es_hist, index=False)

    task_create_historical_var_exceedances(
        panel_daily_data=panel_daily,
        var_es_hist_data=var_es_hist,
        produces=output_path,
    )

    out = pd.read_csv(output_path).sort_values("alpha")
    assert {"method", "alpha", "var", "sample_size", "exceedance_count", "exceedance_rate", "status"} == set(
        out.columns,
    )
    assert out["exceedance_count"].tolist() == [1, 0]
    assert out["sample_size"].tolist() == [5, 5]


def test_task_create_historical_var_exceedances_preserves_method_labels(tmp_path: Path) -> None:
    panel_daily = tmp_path / "panel_daily.parquet"
    var_es_hist = tmp_path / "var_es_hist.csv"
    output_path = tmp_path / "var_exceedances_hist.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=5, freq="B"),
            "ret": [0.0, -0.01, -0.02, -0.03, 0.01],
        },
    ).to_parquet(panel_daily, index=False)
    pd.DataFrame(
        {
            "method": ["historical", "parametric_normal"],
            "alpha": [0.95, 0.95],
            "var": [-0.02, -0.018],
            "es": [-0.025, -0.022],
            "status": ["ok", "ok"],
            "sample_size": [5, 5],
            "fallback_reason": ["none", "none"],
        },
    ).to_csv(var_es_hist, index=False)

    task_create_historical_var_exceedances(
        panel_daily_data=panel_daily,
        var_es_hist_data=var_es_hist,
        produces=output_path,
    )

    out = pd.read_csv(output_path)
    assert set(out["method"]) == {"historical", "parametric_normal"}


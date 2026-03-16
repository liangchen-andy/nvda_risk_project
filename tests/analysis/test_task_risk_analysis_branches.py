"""Branch coverage tests for analysis task helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import nvda_risk_project.analysis.task_risk_analysis as task_module


def test_task_run_risk_module_covers_macro_and_non_macro_branches(tmp_path: Path) -> None:
    monthly_panel = tmp_path / "panel_monthly.csv"
    panel_daily = tmp_path / "panel_daily.parquet"
    macro_raw = tmp_path / "macro_monthly.csv"
    macro_output = tmp_path / "macro_risk.csv"
    liquidity_output = tmp_path / "liquidity_risk.csv"

    pd.DataFrame(
        {
            "month": pd.date_range("2020-01-31", periods=8, freq="ME"),
            "nvda_return": np.linspace(-0.02, 0.03, 8),
            "market_return": np.linspace(-0.01, 0.02, 8),
            "amihud_illiq": np.linspace(1e-9, 2e-9, 8),
            "dollar_volume": np.linspace(2_000_000, 3_000_000, 8),
            "gdp_growth": np.linspace(0.02, 0.025, 8),
            "inflation_yoy": np.linspace(0.03, 0.027, 8),
            "policy_rate": np.linspace(0.055, 0.05, 8),
        },
    ).to_csv(monthly_panel, index=False)

    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=45, freq="B"),
            "ret": np.linspace(-0.02, 0.02, 45),
        },
    ).to_parquet(panel_daily, index=False)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-12-31", "2024-01-31", "2024-02-29"]),
            "policy_rate": [0.05, 0.052, 0.051],
        },
    ).to_csv(macro_raw, index=False)

    task_module.task_run_risk_module(
        dimension="macro",
        panel_data=monthly_panel,
        panel_daily_data=panel_daily,
        macro_raw_data=macro_raw,
        produces=macro_output,
    )
    task_module.task_run_risk_module(
        dimension="liquidity",
        panel_data=monthly_panel,
        panel_daily_data=panel_daily,
        macro_raw_data=macro_raw,
        produces=liquidity_output,
    )

    macro_df = pd.read_csv(macro_output)
    liquidity_df = pd.read_csv(liquidity_output)
    assert set(macro_df["risk_dimension"]) == {"macro"}
    assert set(liquidity_df["risk_dimension"]) == {"liquidity"}


def test_task_create_historical_var_es_covers_valueerror_and_exception(tmp_path: Path, monkeypatch) -> None:
    panel_daily = tmp_path / "panel_daily.parquet"
    output_path = tmp_path / "var_es_hist.csv"

    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=10, freq="B"),
            "ret": np.linspace(-0.02, 0.02, 10),
        },
    ).to_parquet(panel_daily, index=False)

    def _raise_value_error(*, returns: pd.Series, confidence_level: float) -> tuple[float, float]:
        raise ValueError("invalid input")

    def _raise_exception(*, returns: pd.Series, confidence_level: float) -> tuple[float, float]:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(task_module, "compute_historical_var_es", _raise_value_error)
    monkeypatch.setattr(task_module, "compute_parametric_var_es", _raise_exception)
    monkeypatch.setattr(
        task_module,
        "compute_garch_t_var_es",
        lambda returns, confidence_level: {
            "var": float("nan"),
            "es": float("nan"),
            "status": "fallback",
            "fallback_reason": "short_sample",
            "sample_size": int(pd.to_numeric(returns, errors="coerce").notna().sum()),
            "nu": float("nan"),
            "sigma_next": float("nan"),
            "converged": False,
        },
    )

    task_module.task_create_historical_var_es(panel_daily_data=panel_daily, produces=output_path)

    out = pd.read_csv(output_path)
    historical = out.loc[out["method"] == "historical"]
    parametric = out.loc[out["method"] == "parametric_normal"]
    assert set(historical["status"]) == {"fallback"}
    assert set(historical["fallback_reason"]) == {"invalid_input"}
    assert set(parametric["status"]) == {"fallback"}
    assert set(parametric["fallback_reason"]) == {"exception"}


def test_task_create_historical_var_exceedances_handles_zero_sample(tmp_path: Path) -> None:
    panel_daily = tmp_path / "panel_daily.parquet"
    var_es_hist = tmp_path / "var_es_hist.csv"
    output_path = tmp_path / "var_exceedances_hist.csv"

    pd.DataFrame({"date": pd.date_range("2024-01-02", periods=3, freq="B"), "ret": [np.nan] * 3}).to_parquet(
        panel_daily,
        index=False,
    )
    pd.DataFrame(
        {
            "method": ["historical"],
            "alpha": [0.95],
            "var": [-0.02],
            "es": [-0.03],
            "status": ["ok"],
        },
    ).to_csv(var_es_hist, index=False)

    task_module.task_create_historical_var_exceedances(
        panel_daily_data=panel_daily,
        var_es_hist_data=var_es_hist,
        produces=output_path,
    )

    out = pd.read_csv(output_path)
    assert out.loc[0, "sample_size"] == 0
    assert out.loc[0, "exceedance_count"] == 0
    assert pd.isna(out.loc[0, "exceedance_rate"])


def test_task_create_historical_var_backtest_covers_upstream_and_nan_paths(tmp_path: Path) -> None:
    panel_daily = tmp_path / "panel_daily.parquet"
    var_es_hist = tmp_path / "var_es_hist.csv"
    output = tmp_path / "var_backtest_hist.csv"

    pd.DataFrame({"ret": [-0.03, -0.01, 0.01, -0.02, 0.02]}).to_parquet(panel_daily, index=False)
    pd.DataFrame(
        {
            "method": ["garch_t", "garch_t", "historical"],
            "alpha": [0.95, 0.99, 0.95],
            "var": [-0.03, -0.04, np.nan],
            "es": [-0.04, -0.05, np.nan],
            "status": ["fallback", "fallback", "ok"],
            "fallback_reason": ["none", "unknown", "none"],
            "nu": [11.0, 12.0, np.nan],
            "sigma_next": [0.02, 0.03, np.nan],
            "garch_converged": ["1", "0", "maybe"],
        },
    ).to_csv(var_es_hist, index=False)

    task_module.task_create_historical_var_backtest(
        panel_daily_data=panel_daily,
        var_es_hist_data=var_es_hist,
        produces=output,
    )

    out = pd.read_csv(output)
    assert out.loc[0, "fallback_reason"] == "upstream_fallback"
    assert out.loc[1, "fallback_reason"] == "upstream_unknown"
    assert out.loc[2, "fallback_reason"] == "nan_var"
    assert str(out.loc[0, "upstream_garch_converged"]).strip().lower() == "true"
    assert str(out.loc[1, "upstream_garch_converged"]).strip().lower() == "false"
    assert str(out.loc[2, "upstream_garch_converged"]).strip().lower() == "false"


def test_task_create_historical_var_backtest_covers_invalid_input_and_exception(
    tmp_path: Path,
    monkeypatch,
) -> None:
    panel_daily = tmp_path / "panel_daily.parquet"
    var_es_hist = tmp_path / "var_es_hist.csv"
    output = tmp_path / "var_backtest_hist.csv"

    pd.DataFrame({"ret": [-0.03, -0.01, 0.01, -0.02, 0.02]}).to_parquet(panel_daily, index=False)
    pd.DataFrame(
        {
            "method": ["historical", "historical"],
            "alpha": [0.95, 0.99],
            "var": [-0.02, -0.025],
            "es": [-0.03, -0.035],
            "status": ["ok", "ok"],
            "fallback_reason": ["none", "none"],
        },
    ).to_csv(var_es_hist, index=False)

    def _kupiec_raise(*, exceedance_count: int, sample_size: int, alpha: float) -> dict[str, float | bool]:
        if alpha == 0.95:
            raise ValueError("invalid")
        raise RuntimeError("boom")

    monkeypatch.setattr(task_module, "kupiec_pof_test", _kupiec_raise)

    task_module.task_create_historical_var_backtest(
        panel_daily_data=panel_daily,
        var_es_hist_data=var_es_hist,
        produces=output,
    )

    out = pd.read_csv(output).sort_values("alpha").reset_index(drop=True)
    assert out.loc[0, "status"] == "fallback"
    assert out.loc[0, "fallback_reason"] == "invalid_input"
    assert out.loc[1, "status"] == "fallback"
    assert out.loc[1, "fallback_reason"] == "exception"

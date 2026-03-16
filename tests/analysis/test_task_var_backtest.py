"""Tests for historical VaR backtest task output."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nvda_risk_project.analysis.task_risk_analysis import task_create_historical_var_backtest


def test_task_create_historical_var_backtest_writes_expected_columns(tmp_path: Path) -> None:
    panel_daily = tmp_path / "panel_daily.parquet"
    var_es_hist = tmp_path / "var_es_hist.csv"
    output = tmp_path / "var_backtest_hist.csv"

    pd.DataFrame({"ret": [-0.04, -0.02, 0.01, -0.01, -0.05]}).to_parquet(panel_daily, index=False)
    pd.DataFrame(
        {
            "method": ["historical", "historical"],
            "alpha": [0.95, 0.99],
            "var": [-0.03, -0.045],
            "es": [-0.04, -0.05],
            "status": ["ok", "ok"],
            "sample_size": [5, 5],
        },
    ).to_csv(var_es_hist, index=False)

    task_create_historical_var_backtest(
        panel_daily_data=panel_daily,
        var_es_hist_data=var_es_hist,
        produces=output,
    )

    out = pd.read_csv(output)
    expected_columns = {
        "method",
        "alpha",
        "var",
        "sample_size",
        "exceedance_count",
        "exceedance_rate",
        "kupiec_lr_stat",
        "kupiec_p_value",
        "kupiec_reject_5pct",
        "christoffersen_lr_stat",
        "christoffersen_p_value",
        "christoffersen_reject_5pct",
        "status",
        "fallback_reason",
        "upstream_nu",
        "upstream_sigma_next",
        "upstream_garch_converged",
    }
    assert expected_columns == set(out.columns)
    assert set(out["status"]) == {"ok"}
    assert set(out["fallback_reason"]) == {"none"}
    assert out["upstream_nu"].isna().all()
    assert out["upstream_sigma_next"].isna().all()
    assert set(out["upstream_garch_converged"].astype(str).str.lower()) == {"false"}
    assert out["kupiec_p_value"].between(0, 1).all()
    assert out["christoffersen_p_value"].between(0, 1).all()


def test_task_create_historical_var_backtest_fallback_on_short_sample(tmp_path: Path) -> None:
    panel_daily = tmp_path / "panel_daily.parquet"
    var_es_hist = tmp_path / "var_es_hist.csv"
    output = tmp_path / "var_backtest_hist.csv"

    pd.DataFrame({"ret": [-0.01]}).to_parquet(panel_daily, index=False)
    pd.DataFrame(
        {
            "method": ["historical"],
            "alpha": [0.95],
            "var": [-0.02],
            "es": [-0.02],
            "status": ["ok"],
            "sample_size": [1],
        },
    ).to_csv(var_es_hist, index=False)

    task_create_historical_var_backtest(
        panel_daily_data=panel_daily,
        var_es_hist_data=var_es_hist,
        produces=output,
    )

    out = pd.read_csv(output)
    assert out.loc[0, "status"] == "fallback"
    assert out.loc[0, "fallback_reason"] == "short_sample"
    assert pd.isna(out.loc[0, "upstream_nu"])
    assert pd.isna(out.loc[0, "upstream_sigma_next"])
    assert str(out.loc[0, "upstream_garch_converged"]).strip().lower() == "false"
    assert pd.isna(out.loc[0, "kupiec_p_value"])
    assert pd.isna(out.loc[0, "christoffersen_p_value"])


def test_task_create_historical_var_backtest_preserves_method_labels(tmp_path: Path) -> None:
    panel_daily = tmp_path / "panel_daily.parquet"
    var_es_hist = tmp_path / "var_es_hist.csv"
    output = tmp_path / "var_backtest_hist.csv"

    pd.DataFrame({"ret": [-0.04, -0.02, 0.01, -0.01, -0.05]}).to_parquet(panel_daily, index=False)
    pd.DataFrame(
        {
            "method": ["historical", "parametric_normal"],
            "alpha": [0.95, 0.95],
            "var": [-0.03, -0.028],
            "es": [-0.04, -0.036],
            "status": ["ok", "ok"],
            "fallback_reason": ["none", "none"],
            "sample_size": [5, 5],
        },
    ).to_csv(var_es_hist, index=False)

    task_create_historical_var_backtest(
        panel_daily_data=panel_daily,
        var_es_hist_data=var_es_hist,
        produces=output,
    )

    out = pd.read_csv(output)
    assert set(out["method"]) == {"historical", "parametric_normal"}


def test_task_create_historical_var_backtest_propagates_garch_upstream_fields(
    tmp_path: Path,
) -> None:
    panel_daily = tmp_path / "panel_daily.parquet"
    var_es_hist = tmp_path / "var_es_hist.csv"
    output = tmp_path / "var_backtest_hist.csv"

    pd.DataFrame({"ret": [-0.04, -0.02, 0.01, -0.01, -0.05]}).to_parquet(panel_daily, index=False)
    pd.DataFrame(
        {
            "method": ["garch_t"],
            "alpha": [0.95],
            "var": [-0.03],
            "es": [-0.04],
            "status": ["ok"],
            "fallback_reason": ["none"],
            "sample_size": [5],
            "nu": [11.5],
            "sigma_next": [0.02],
            "garch_converged": [True],
        },
    ).to_csv(var_es_hist, index=False)

    task_create_historical_var_backtest(
        panel_daily_data=panel_daily,
        var_es_hist_data=var_es_hist,
        produces=output,
    )

    out = pd.read_csv(output)
    assert out.loc[0, "method"] == "garch_t"
    assert out.loc[0, "upstream_nu"] == 11.5
    assert out.loc[0, "upstream_sigma_next"] == 0.02
    assert str(out.loc[0, "upstream_garch_converged"]).strip().lower() == "true"

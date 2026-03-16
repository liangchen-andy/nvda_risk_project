"""Tests for diagnostics markdown artifact generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from nvda_risk_project.final.task_diagnostics import (
    task_create_diagnostics_data,
    task_create_diagnostics_table,
)


def test_task_create_diagnostics_table_contains_provenance_checks(tmp_path: Path) -> None:
    risk_summary = tmp_path / "risk_summary.csv"
    scorecard = tmp_path / "scorecard.json"
    nvda_raw = tmp_path / "nvda_daily.csv"
    market_raw = tmp_path / "sp500_daily.csv"
    rolling_vol = tmp_path / "rolling_vol.csv"
    var_es_hist = tmp_path / "var_es_hist.csv"
    var_exceedances_hist = tmp_path / "var_exceedances_hist.csv"
    var_backtest_hist = tmp_path / "var_backtest_hist.csv"
    panel_daily = tmp_path / "panel_daily.parquet"
    provenance = tmp_path / "data_provenance.json"
    diagnostics_csv = tmp_path / "diagnostics.csv"
    diagnostics_md = tmp_path / "diagnostics.md"

    pd.DataFrame({"risk_dimension": ["market"], "metric": ["vol"], "value": [0.2]}).to_csv(
        risk_summary,
        index=False,
    )
    scorecard.write_text(
        json.dumps(
            {
                "metric_count": 1,
                "fallback_events": 0,
                "var_backtest_garch_rows": 1,
                "var_backtest_garch_fallback_events": 0,
                "var_backtest_garch_status_ok_count": 1,
            },
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"date": ["2024-01-01"], "close": [100], "volume": [1000], "data_source": ["snapshot"]}).to_csv(
        nvda_raw,
        index=False,
    )
    pd.DataFrame({"date": ["2024-01-01"], "close": [4000], "volume": [2000], "data_source": ["online"]}).to_csv(
        market_raw,
        index=False,
    )
    pd.DataFrame({"date": ["2024-01-01"], "rolling_vol": [0.2]}).to_csv(rolling_vol, index=False)
    pd.DataFrame({"alpha": [0.95], "var": [-0.03], "es": [-0.035], "status": ["ok"]}).to_csv(
        var_es_hist,
        index=False,
    )
    pd.DataFrame(
        {
            "alpha": [0.95],
            "exceedance_count": [5],
            "exceedance_rate": [0.05],
            "status": ["ok"],
        },
    ).to_csv(var_exceedances_hist, index=False)
    pd.DataFrame(
        {
            "method": ["garch_t"],
            "alpha": [0.95],
            "kupiec_p_value": [0.2],
            "christoffersen_p_value": [0.3],
            "status": ["ok"],
        },
    ).to_csv(var_backtest_hist, index=False)
    pd.DataFrame({"date": ["2024-01-01"], "adj_close": [100.0]}).to_parquet(panel_daily, index=False)
    provenance.write_text(
        json.dumps(
            {
                "tickers": ["NVDA", "^GSPC"],
                "sample_start": "2024-01-01",
                "sample_end": "2024-01-01",
                "source": {"NVDA": "snapshot", "^GSPC": "online"},
                "artifacts": {
                    "nvda_raw": {
                        "path": str(nvda_raw),
                        "sha256": hashlib.sha256(nvda_raw.read_bytes()).hexdigest(),
                    },
                    "sp500_raw": {
                        "path": str(market_raw),
                        "sha256": hashlib.sha256(market_raw.read_bytes()).hexdigest(),
                    },
                    "panel_daily": {
                        "path": str(panel_daily),
                        "sha256": hashlib.sha256(panel_daily.read_bytes()).hexdigest(),
                    },
                },
            },
        ),
        encoding="utf-8",
    )

    task_create_diagnostics_data(
        risk_summary=risk_summary,
        scorecard=scorecard,
        nvda_raw=nvda_raw,
        market_raw=market_raw,
        rolling_vol=rolling_vol,
        var_es_hist=var_es_hist,
        var_exceedances_hist=var_exceedances_hist,
        var_backtest_hist=var_backtest_hist,
        provenance=provenance,
        panel_daily=panel_daily,
        produces=diagnostics_csv,
    )
    task_create_diagnostics_table(diagnostics_data=diagnostics_csv, produces=diagnostics_md)

    output = diagnostics_md.read_text(encoding="utf-8")
    assert "# Diagnostics" in output
    assert "## Quality Gates" in output
    assert "## Full Checks" in output
    assert "provenance_exists" in output
    assert "provenance_panel_sha_match" in output
    assert "provenance_nvda_raw_sha_match" in output
    assert "provenance_market_raw_sha_match" in output
    assert "provenance_source_match" in output
    assert "panel_daily_required_cols_present" in output
    assert "var_backtest_garch_rows_consistent" in output
    assert "var_backtest_historical_rows_consistent" in output
    assert "var_backtest_garch_fallback_events_consistent" in output
    assert "var_backtest_garch_status_ok_count_consistent" in output
    assert "garch_backtest_upstream_converged_present" in output
    assert "historical_kupiec_reject_rate_5pct_consistent" in output
    assert "garch_kupiec_reject_rate_5pct_consistent" in output
    assert "systematic_beta_consistent" in output
    assert "systematic_beta_rolling_60m_valid_points_consistent" in output
    assert "garch_var_es_nu_finite" in output
    assert "quality_gates_failed_checks" in output
    assert "quality_gates_all_pass" in output
    assert "placeholder_source_detected" in output
    assert "True" in output

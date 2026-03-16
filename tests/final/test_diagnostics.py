"""Tests for diagnostics artifact helpers."""

import hashlib
from pathlib import Path
import json

import pandas as pd

from nvda_risk_project.final.diagnostics import build_diagnostics_table


def _as_map(frame: pd.DataFrame) -> dict[str, str]:
    """Convert diagnostics key-value rows into a dictionary."""
    return dict(zip(frame["check"], frame["value"], strict=True))


def test_build_diagnostics_table_tracks_systematic_metric_consistency(tmp_path: Path) -> None:
    """Compare scorecard-recorded systematic metrics against risk_summary observations."""
    risk_summary = tmp_path / "risk_summary.csv"
    scorecard = tmp_path / "scorecard.json"

    pd.DataFrame(
        {
            "risk_dimension": ["systematic"] * 8,
            "metric": [
                "beta",
                "alpha",
                "beta_rolling_60m_latest",
                "beta_rolling_60m_valid_points",
                "beta_rolling_60m_fallback",
                "beta_rolling_252m_latest",
                "beta_rolling_252m_valid_points",
                "beta_rolling_252m_fallback",
            ],
            "value": [1.2, 0.01, 1.1, 14.0, 0.0, 0.0, 0.0, 1.0],
        },
    ).to_csv(risk_summary, index=False)
    scorecard.write_text(
        json.dumps(
            {
                "systematic_beta": 1.2,
                "systematic_alpha": 0.01,
                "systematic_beta_rolling_60m_latest": 1.1,
                "systematic_beta_rolling_60m_valid_points": 14,
                "systematic_beta_rolling_60m_fallback": False,
                "systematic_beta_rolling_252m_latest": 0.0,
                "systematic_beta_rolling_252m_valid_points": 0,
                "systematic_beta_rolling_252m_fallback": True,
            },
        ),
        encoding="utf-8",
    )

    diagnostics = build_diagnostics_table(
        risk_summary_path=risk_summary,
        scorecard_path=scorecard,
        nvda_raw_path=tmp_path / "nvda.csv",
        market_raw_path=tmp_path / "market.csv",
        rolling_vol_path=tmp_path / "rolling_vol.csv",
        var_es_hist_path=tmp_path / "var_es_hist.csv",
        var_exceedances_hist_path=tmp_path / "var_exceedances_hist.csv",
        var_backtest_hist_path=tmp_path / "var_backtest_hist.csv",
        vol_window_days=252,
    )
    diagnostics_map = _as_map(diagnostics)

    assert diagnostics_map["systematic_metrics_present"] == "True"
    assert diagnostics_map["systematic_beta_consistent"] == "True"
    assert diagnostics_map["systematic_alpha_consistent"] == "True"
    assert diagnostics_map["systematic_beta_rolling_60m_latest_consistent"] == "True"
    assert diagnostics_map["systematic_beta_rolling_60m_valid_points_consistent"] == "True"
    assert diagnostics_map["systematic_beta_rolling_60m_fallback_consistent"] == "True"
    assert diagnostics_map["systematic_beta_rolling_252m_latest_consistent"] == "True"
    assert diagnostics_map["systematic_beta_rolling_252m_valid_points_consistent"] == "True"
    assert diagnostics_map["systematic_beta_rolling_252m_fallback_consistent"] == "True"


def test_build_diagnostics_table_contains_required_checks(tmp_path: Path) -> None:
    """Validate diagnostics contains expected checks and populated values."""
    risk_summary = tmp_path / "risk_summary.csv"
    scorecard = tmp_path / "scorecard.json"
    nvda_raw = tmp_path / "nvda_daily.csv"
    market_raw = tmp_path / "sp500_daily.csv"
    rolling_vol = tmp_path / "rolling_vol.csv"
    var_es_hist = tmp_path / "var_es_hist.csv"
    var_exceedances_hist = tmp_path / "var_exceedances_hist.csv"
    var_backtest_hist = tmp_path / "var_backtest_hist.csv"
    macro_raw = tmp_path / "macro_monthly.csv"
    panel_daily = tmp_path / "panel_daily.parquet"
    provenance = tmp_path / "data_provenance.json"

    pd.DataFrame({"risk_dimension": ["market"], "metric": ["vol"], "value": [0.2]}).to_csv(
        risk_summary,
        index=False,
    )
    scorecard.write_text(
        json.dumps(
            {
                "metric_count": 1,
                "fallback_events": 0,
                "var_backtest_fallback_events": 0,
                "var_backtest_historical_rows": 2,
                "var_backtest_historical_fallback_events": 0,
                "var_backtest_parametric_rows": 0,
                "var_backtest_parametric_fallback_events": 0,
                "var_backtest_other_rows": 0,
                "var_backtest_other_fallback_events": 0,
                "var_backtest_garch_rows": 0,
                "var_backtest_garch_fallback_events": 0,
                "var_backtest_garch_status_ok_count": 0,
                "historical_kupiec_reject_rate_5pct": 0.0,
                "historical_christoffersen_reject_rate_5pct": 0.0,
                "parametric_kupiec_reject_rate_5pct": float("nan"),
                "parametric_christoffersen_reject_rate_5pct": float("nan"),
                "sample_target_start": "2020-01-01",
                "sample_target_end": "2024-12-31",
                "window_matches_target": "False",
                "status": "fail",
                "status_reason": "window_not_target",
                "var_backtest_fallback_reason_short_sample_count": 0,
                "var_backtest_fallback_reason_nan_var_count": 0,
                "var_backtest_fallback_reason_invalid_input_count": 0,
                "var_backtest_fallback_reason_exception_count": 0,
                "var_backtest_fallback_reason_upstream_fallback_count": 0,
                "var_backtest_fallback_reason_unknown_count": 0,
            },
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "close": [100],
            "volume": [1000],
            "data_source": ["snapshot"],
        },
    ).to_csv(
        nvda_raw,
        index=False,
    )
    pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "close": [4000],
            "volume": [2000],
            "data_source": ["online"],
        },
    ).to_csv(
        market_raw,
        index=False,
    )
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "rolling_vol": [None, 0.21],
        },
    ).to_csv(rolling_vol, index=False)
    pd.DataFrame(
        {
            "method": ["historical", "historical"],
            "alpha": [0.95, 0.99],
            "var": [-0.03, -0.04],
            "es": [-0.035, -0.045],
            "status": ["ok", "ok"],
            "sample_size": [100, 100],
        },
    ).to_csv(var_es_hist, index=False)
    pd.DataFrame(
        {
            "method": ["historical", "historical"],
            "alpha": [0.95, 0.99],
            "exceedance_count": [5, 1],
            "exceedance_rate": [0.05, 0.01],
            "status": ["ok", "ok"],
            "sample_size": [100, 100],
        },
    ).to_csv(var_exceedances_hist, index=False)
    pd.DataFrame(
        {
            "method": ["historical", "historical"],
            "alpha": [0.95, 0.99],
            "kupiec_p_value": [0.12, 0.08],
            "christoffersen_p_value": [0.34, 0.21],
            "status": ["ok", "ok"],
        },
    ).to_csv(var_backtest_hist, index=False)
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-31", periods=3, freq="ME"),
            "gdp_growth": [0.02, 0.021, 0.022],
            "inflation_yoy": [0.03, 0.029, 0.028],
            "policy_rate": [0.055, 0.054, 0.053],
        },
    ).to_csv(macro_raw, index=False)
    pd.DataFrame({"date": ["2024-01-01"], "adj_close": [100.0]}).to_parquet(
        panel_daily,
        index=False,
    )
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

    diagnostics = build_diagnostics_table(
        risk_summary_path=risk_summary,
        scorecard_path=scorecard,
        nvda_raw_path=nvda_raw,
        market_raw_path=market_raw,
        rolling_vol_path=rolling_vol,
        var_es_hist_path=var_es_hist,
        var_exceedances_hist_path=var_exceedances_hist,
        var_backtest_hist_path=var_backtest_hist,
        macro_raw_path=macro_raw,
        vol_window_days=252,
        provenance_path=provenance,
        panel_daily_path=panel_daily,
    )

    required_checks = {
        "risk_summary_exists",
        "scorecard_exists",
        "scorecard_parse_error",
        "provenance_exists",
        "provenance_parse_error",
        "provenance_ticker_match",
        "provenance_sample_match",
        "provenance_panel_sha_match",
        "provenance_nvda_raw_sha_match",
        "provenance_market_raw_sha_match",
        "provenance_source_match",
        "panel_daily_exists",
        "panel_daily_rows",
        "panel_daily_required_cols_present",
        "panel_daily_date_monotonic",
        "panel_daily_duplicate_dates",
        "panel_daily_core_missing_rate",
        "nvda_raw_rows",
        "market_raw_rows",
        "nvda_sample_start",
        "nvda_sample_end",
        "market_sample_start",
        "market_sample_end",
        "sample_target_start",
        "sample_target_end",
        "nvda_window_matches_target",
        "market_window_matches_target",
        "panel_window_matches_target",
        "scorecard_sample_target_start_recorded",
        "scorecard_sample_target_end_recorded",
        "scorecard_window_matches_target_recorded",
        "scorecard_window_matches_target_consistent",
        "scorecard_status_recorded",
        "scorecard_status_expected",
        "scorecard_status_consistent",
        "scorecard_status_reason_recorded",
        "scorecard_status_reason_expected",
        "scorecard_status_reason_consistent",
        "scorecard_metric_count",
        "systematic_metrics_present",
        "systematic_beta_recorded",
        "systematic_beta_observed",
        "systematic_beta_consistent",
        "systematic_alpha_recorded",
        "systematic_alpha_observed",
        "systematic_alpha_consistent",
        "systematic_beta_rolling_60m_latest_recorded",
        "systematic_beta_rolling_60m_latest_observed",
        "systematic_beta_rolling_60m_latest_consistent",
        "systematic_beta_rolling_60m_valid_points_recorded",
        "systematic_beta_rolling_60m_valid_points_observed",
        "systematic_beta_rolling_60m_valid_points_consistent",
        "systematic_beta_rolling_60m_fallback_recorded",
        "systematic_beta_rolling_60m_fallback_observed",
        "systematic_beta_rolling_60m_fallback_consistent",
        "systematic_beta_rolling_252m_latest_recorded",
        "systematic_beta_rolling_252m_latest_observed",
        "systematic_beta_rolling_252m_latest_consistent",
        "systematic_beta_rolling_252m_valid_points_recorded",
        "systematic_beta_rolling_252m_valid_points_observed",
        "systematic_beta_rolling_252m_valid_points_consistent",
        "systematic_beta_rolling_252m_fallback_recorded",
        "systematic_beta_rolling_252m_fallback_observed",
        "systematic_beta_rolling_252m_fallback_consistent",
        "fallback_events_recorded",
        "var_backtest_fallback_events_recorded",
        "var_backtest_historical_rows_recorded",
        "var_backtest_historical_fallback_events_recorded",
        "var_backtest_historical_rows_observed",
        "var_backtest_historical_fallback_events_observed",
        "var_backtest_historical_rows_consistent",
        "var_backtest_historical_fallback_events_consistent",
        "var_backtest_parametric_rows_recorded",
        "var_backtest_parametric_fallback_events_recorded",
        "var_backtest_parametric_rows_observed",
        "var_backtest_parametric_fallback_events_observed",
        "var_backtest_parametric_rows_consistent",
        "var_backtest_parametric_fallback_events_consistent",
        "var_backtest_other_rows_recorded",
        "var_backtest_other_fallback_events_recorded",
        "var_backtest_other_rows_observed",
        "var_backtest_other_fallback_events_observed",
        "var_backtest_other_rows_consistent",
        "var_backtest_other_fallback_events_consistent",
        "var_backtest_garch_rows_recorded",
        "var_backtest_garch_fallback_events_recorded",
        "var_backtest_garch_status_ok_count_recorded",
        "var_backtest_garch_rows_observed",
        "var_backtest_garch_fallback_events_observed",
        "var_backtest_garch_status_ok_count_observed",
        "var_backtest_garch_rows_consistent",
        "var_backtest_garch_fallback_events_consistent",
        "var_backtest_garch_status_ok_count_consistent",
        "quality_gates_pass_count",
        "quality_gates_fail_count",
        "quality_gates_unknown_count",
        "quality_gates_failed_checks",
        "quality_gates_all_pass",
        "historical_kupiec_reject_rate_5pct_recorded",
        "historical_christoffersen_reject_rate_5pct_recorded",
        "parametric_kupiec_reject_rate_5pct_recorded",
        "parametric_christoffersen_reject_rate_5pct_recorded",
        "garch_kupiec_reject_rate_5pct_recorded",
        "garch_christoffersen_reject_rate_5pct_recorded",
        "historical_kupiec_reject_rate_5pct_observed",
        "historical_christoffersen_reject_rate_5pct_observed",
        "parametric_kupiec_reject_rate_5pct_observed",
        "parametric_christoffersen_reject_rate_5pct_observed",
        "garch_kupiec_reject_rate_5pct_observed",
        "garch_christoffersen_reject_rate_5pct_observed",
        "historical_kupiec_reject_rate_5pct_consistent",
        "historical_christoffersen_reject_rate_5pct_consistent",
        "parametric_kupiec_reject_rate_5pct_consistent",
        "parametric_christoffersen_reject_rate_5pct_consistent",
        "garch_kupiec_reject_rate_5pct_consistent",
        "garch_christoffersen_reject_rate_5pct_consistent",
        "fallback_reason_short_sample_count",
        "fallback_reason_nan_var_count",
        "fallback_reason_invalid_input_count",
        "fallback_reason_exception_count",
        "fallback_reason_upstream_fallback_count",
        "fallback_reason_unknown_count",
        "data_source",
        "placeholder_source_detected",
        "rolling_vol_window_days",
        "rolling_vol_total_rows",
        "rolling_vol_valid_points",
        "rolling_vol_fallback_used",
        "hist_var_es_exists",
        "hist_var_es_method_count",
        "parametric_var_es_present",
        "garch_var_es_present",
        "hist_var_es_alpha_count",
        "hist_var_es_status_ok_count",
        "hist_var_95",
        "hist_es_95",
        "hist_var_99",
        "hist_es_99",
        "param_var_95",
        "param_es_95",
        "param_var_99",
        "param_es_99",
        "garch_var_es_nu_finite",
        "garch_var_es_nu_gt4",
        "garch_var_es_sigma_next_positive",
        "macro_data_exists",
        "macro_rows",
        "macro_month_end_only",
        "macro_duplicate_month",
        "macro_frequency_ok",
        "hist_exceed_exists",
        "hist_exceed_method_count",
        "parametric_exceed_present",
        "garch_exceed_present",
        "hist_exceed_status_ok_count",
        "hist_exceed_95_count",
        "hist_exceed_95_rate",
        "hist_exceed_95_gap",
        "hist_exceed_99_count",
        "hist_exceed_99_rate",
        "hist_exceed_99_gap",
        "hist_backtest_exists",
        "hist_backtest_method_count",
        "parametric_backtest_present",
        "garch_backtest_present",
        "hist_backtest_alpha_count",
        "hist_backtest_status_ok_count",
        "hist_kupiec_lr_95",
        "hist_kupiec_lr_99",
        "hist_kupiec_pvalue_95",
        "hist_kupiec_pvalue_99",
        "param_kupiec_lr_95",
        "param_kupiec_lr_99",
        "param_kupiec_pvalue_95",
        "param_kupiec_pvalue_99",
        "hist_christoffersen_lr_95",
        "hist_christoffersen_lr_99",
        "hist_christoffersen_pvalue_95",
        "hist_christoffersen_pvalue_99",
        "param_christoffersen_lr_95",
        "param_christoffersen_lr_99",
        "param_christoffersen_pvalue_95",
        "param_christoffersen_pvalue_99",
        "garch_kupiec_lr_95",
        "garch_kupiec_lr_99",
        "garch_kupiec_pvalue_95",
        "garch_kupiec_pvalue_99",
        "garch_christoffersen_lr_95",
        "garch_christoffersen_lr_99",
        "garch_christoffersen_pvalue_95",
        "garch_christoffersen_pvalue_99",
        "garch_backtest_upstream_exists",
        "garch_backtest_upstream_converged_present",
        "garch_backtest_upstream_converged_true_count",
        "garch_backtest_upstream_nu_finite",
        "garch_backtest_upstream_sigma_positive",
    }
    assert required_checks.issubset(set(diagnostics["check"]))

    diagnostics_map = _as_map(diagnostics)
    assert diagnostics_map["scorecard_parse_error"] == "False"
    assert diagnostics_map["provenance_exists"] == "True"
    assert diagnostics_map["provenance_parse_error"] == "False"
    assert diagnostics_map["provenance_ticker_match"] == "True"
    assert diagnostics_map["provenance_sample_match"] == "True"
    assert diagnostics_map["provenance_panel_sha_match"] == "True"
    assert diagnostics_map["provenance_nvda_raw_sha_match"] == "True"
    assert diagnostics_map["provenance_market_raw_sha_match"] == "True"
    assert diagnostics_map["provenance_source_match"] == "True"
    assert diagnostics_map["panel_daily_exists"] == "True"
    assert diagnostics_map["panel_daily_rows"] == "1"
    assert diagnostics_map["panel_daily_required_cols_present"] == "False"
    assert diagnostics_map["panel_daily_date_monotonic"] == "True"
    assert diagnostics_map["panel_daily_duplicate_dates"] == "0"
    assert diagnostics_map["panel_daily_core_missing_rate"] == "0.0"
    assert diagnostics_map["nvda_sample_start"] == "2024-01-01"
    assert diagnostics_map["market_sample_end"] == "2024-01-01"
    assert diagnostics_map["sample_target_start"] == "2020-01-01"
    assert diagnostics_map["sample_target_end"] == "2024-12-31"
    assert diagnostics_map["nvda_window_matches_target"] == "False"
    assert diagnostics_map["market_window_matches_target"] == "False"
    assert diagnostics_map["panel_window_matches_target"] == "False"
    assert diagnostics_map["scorecard_sample_target_start_recorded"] == "2020-01-01"
    assert diagnostics_map["scorecard_sample_target_end_recorded"] == "2024-12-31"
    assert diagnostics_map["scorecard_window_matches_target_recorded"] == "false"
    assert diagnostics_map["scorecard_window_matches_target_consistent"] == "True"
    assert diagnostics_map["scorecard_status_recorded"] == "fail"
    assert diagnostics_map["scorecard_status_expected"] == "fail"
    assert diagnostics_map["scorecard_status_consistent"] == "True"
    assert diagnostics_map["scorecard_status_reason_recorded"] == "window_not_target"
    assert diagnostics_map["scorecard_status_reason_expected"] == "window_not_target"
    assert diagnostics_map["scorecard_status_reason_consistent"] == "True"
    assert diagnostics_map["systematic_metrics_present"] == "False"
    assert diagnostics_map["systematic_beta_recorded"] == "nan"
    assert diagnostics_map["systematic_beta_observed"] == "unknown"
    assert diagnostics_map["systematic_beta_consistent"] == "unknown"
    assert diagnostics_map["systematic_alpha_recorded"] == "nan"
    assert diagnostics_map["systematic_alpha_observed"] == "unknown"
    assert diagnostics_map["systematic_alpha_consistent"] == "unknown"
    assert diagnostics_map["systematic_beta_rolling_60m_latest_recorded"] == "nan"
    assert diagnostics_map["systematic_beta_rolling_60m_latest_observed"] == "unknown"
    assert diagnostics_map["systematic_beta_rolling_60m_latest_consistent"] == "unknown"
    assert diagnostics_map["systematic_beta_rolling_60m_valid_points_recorded"] == "0"
    assert diagnostics_map["systematic_beta_rolling_60m_valid_points_observed"] == "unknown"
    assert diagnostics_map["systematic_beta_rolling_60m_valid_points_consistent"] == "unknown"
    assert diagnostics_map["systematic_beta_rolling_60m_fallback_recorded"] == "unknown"
    assert diagnostics_map["systematic_beta_rolling_60m_fallback_observed"] == "unknown"
    assert diagnostics_map["systematic_beta_rolling_60m_fallback_consistent"] == "unknown"
    assert diagnostics_map["systematic_beta_rolling_252m_latest_recorded"] == "nan"
    assert diagnostics_map["systematic_beta_rolling_252m_latest_observed"] == "unknown"
    assert diagnostics_map["systematic_beta_rolling_252m_latest_consistent"] == "unknown"
    assert diagnostics_map["systematic_beta_rolling_252m_valid_points_recorded"] == "0"
    assert diagnostics_map["systematic_beta_rolling_252m_valid_points_observed"] == "unknown"
    assert diagnostics_map["systematic_beta_rolling_252m_valid_points_consistent"] == "unknown"
    assert diagnostics_map["systematic_beta_rolling_252m_fallback_recorded"] == "unknown"
    assert diagnostics_map["systematic_beta_rolling_252m_fallback_observed"] == "unknown"
    assert diagnostics_map["systematic_beta_rolling_252m_fallback_consistent"] == "unknown"
    assert diagnostics_map["rolling_vol_window_days"] == "252"
    assert diagnostics_map["rolling_vol_total_rows"] == "2"
    assert diagnostics_map["rolling_vol_valid_points"] == "1"
    assert diagnostics_map["rolling_vol_fallback_used"] == "False"
    assert diagnostics_map["data_source"] == "online+snapshot"
    assert diagnostics_map["hist_var_es_exists"] == "True"
    assert diagnostics_map["hist_var_es_method_count"] == "1"
    assert diagnostics_map["parametric_var_es_present"] == "False"
    assert diagnostics_map["garch_var_es_present"] == "False"
    assert diagnostics_map["hist_var_es_alpha_count"] == "2"
    assert diagnostics_map["hist_var_es_status_ok_count"] == "2"
    assert diagnostics_map["hist_var_95"] == "-0.03"
    assert diagnostics_map["hist_es_99"] == "-0.045"
    assert diagnostics_map["param_var_95"] == "nan"
    assert diagnostics_map["param_es_99"] == "nan"
    assert diagnostics_map["garch_var_es_nu_finite"] == "unknown"
    assert diagnostics_map["garch_var_es_nu_gt4"] == "unknown"
    assert diagnostics_map["garch_var_es_sigma_next_positive"] == "unknown"
    assert diagnostics_map["macro_data_exists"] == "True"
    assert diagnostics_map["macro_rows"] == "3"
    assert diagnostics_map["macro_month_end_only"] == "True"
    assert diagnostics_map["macro_duplicate_month"] == "False"
    assert diagnostics_map["macro_frequency_ok"] == "True"
    assert diagnostics_map["hist_exceed_exists"] == "True"
    assert diagnostics_map["hist_exceed_method_count"] == "1"
    assert diagnostics_map["parametric_exceed_present"] == "False"
    assert diagnostics_map["garch_exceed_present"] == "False"
    assert diagnostics_map["hist_exceed_status_ok_count"] == "2"
    assert diagnostics_map["hist_exceed_95_count"] == "5"
    assert diagnostics_map["hist_exceed_95_gap"] == "0.0"
    assert diagnostics_map["hist_exceed_99_rate"] == "0.01"
    assert diagnostics_map["hist_exceed_99_gap"] == "0.0"
    assert diagnostics_map["hist_backtest_exists"] == "True"
    assert diagnostics_map["hist_backtest_method_count"] == "1"
    assert diagnostics_map["parametric_backtest_present"] == "False"
    assert diagnostics_map["garch_backtest_present"] == "False"
    assert diagnostics_map["hist_backtest_alpha_count"] == "2"
    assert diagnostics_map["hist_backtest_status_ok_count"] == "2"
    assert diagnostics_map["hist_kupiec_lr_95"] == "nan"
    assert diagnostics_map["hist_kupiec_lr_99"] == "nan"
    assert diagnostics_map["hist_kupiec_pvalue_95"] == "0.12"
    assert diagnostics_map["hist_kupiec_pvalue_99"] == "0.08"
    assert diagnostics_map["param_kupiec_lr_95"] == "nan"
    assert diagnostics_map["param_kupiec_lr_99"] == "nan"
    assert diagnostics_map["param_kupiec_pvalue_95"] == "nan"
    assert diagnostics_map["param_kupiec_pvalue_99"] == "nan"
    assert diagnostics_map["hist_christoffersen_lr_95"] == "nan"
    assert diagnostics_map["hist_christoffersen_lr_99"] == "nan"
    assert diagnostics_map["hist_christoffersen_pvalue_95"] == "0.34"
    assert diagnostics_map["hist_christoffersen_pvalue_99"] == "0.21"
    assert diagnostics_map["param_christoffersen_lr_95"] == "nan"
    assert diagnostics_map["param_christoffersen_lr_99"] == "nan"
    assert diagnostics_map["param_christoffersen_pvalue_95"] == "nan"
    assert diagnostics_map["param_christoffersen_pvalue_99"] == "nan"
    assert diagnostics_map["garch_kupiec_lr_95"] == "nan"
    assert diagnostics_map["garch_kupiec_lr_99"] == "nan"
    assert diagnostics_map["garch_kupiec_pvalue_95"] == "nan"
    assert diagnostics_map["garch_kupiec_pvalue_99"] == "nan"
    assert diagnostics_map["garch_christoffersen_lr_95"] == "nan"
    assert diagnostics_map["garch_christoffersen_lr_99"] == "nan"
    assert diagnostics_map["garch_christoffersen_pvalue_95"] == "nan"
    assert diagnostics_map["garch_christoffersen_pvalue_99"] == "nan"
    assert diagnostics_map["garch_backtest_upstream_exists"] == "True"
    assert diagnostics_map["garch_backtest_upstream_converged_present"] == "unknown"
    assert diagnostics_map["garch_backtest_upstream_converged_true_count"] == "unknown"
    assert diagnostics_map["garch_backtest_upstream_nu_finite"] == "unknown"
    assert diagnostics_map["garch_backtest_upstream_sigma_positive"] == "unknown"
    assert diagnostics_map["var_backtest_fallback_events_recorded"] == "0"
    assert diagnostics_map["var_backtest_historical_rows_recorded"] == "2"
    assert diagnostics_map["var_backtest_historical_fallback_events_recorded"] == "0"
    assert diagnostics_map["var_backtest_historical_rows_observed"] == "2"
    assert diagnostics_map["var_backtest_historical_fallback_events_observed"] == "0"
    assert diagnostics_map["var_backtest_historical_rows_consistent"] == "True"
    assert diagnostics_map["var_backtest_historical_fallback_events_consistent"] == "True"
    assert diagnostics_map["var_backtest_parametric_rows_recorded"] == "0"
    assert diagnostics_map["var_backtest_parametric_fallback_events_recorded"] == "0"
    assert diagnostics_map["var_backtest_parametric_rows_observed"] == "0"
    assert diagnostics_map["var_backtest_parametric_fallback_events_observed"] == "0"
    assert diagnostics_map["var_backtest_parametric_rows_consistent"] == "True"
    assert diagnostics_map["var_backtest_parametric_fallback_events_consistent"] == "True"
    assert diagnostics_map["var_backtest_other_rows_recorded"] == "0"
    assert diagnostics_map["var_backtest_other_fallback_events_recorded"] == "0"
    assert diagnostics_map["var_backtest_other_rows_observed"] == "0"
    assert diagnostics_map["var_backtest_other_fallback_events_observed"] == "0"
    assert diagnostics_map["var_backtest_other_rows_consistent"] == "True"
    assert diagnostics_map["var_backtest_other_fallback_events_consistent"] == "True"
    assert diagnostics_map["var_backtest_garch_rows_recorded"] == "0"
    assert diagnostics_map["var_backtest_garch_fallback_events_recorded"] == "0"
    assert diagnostics_map["var_backtest_garch_status_ok_count_recorded"] == "0"
    assert diagnostics_map["var_backtest_garch_rows_observed"] == "0"
    assert diagnostics_map["var_backtest_garch_fallback_events_observed"] == "0"
    assert diagnostics_map["var_backtest_garch_status_ok_count_observed"] == "0"
    assert diagnostics_map["var_backtest_garch_rows_consistent"] == "True"
    assert diagnostics_map["var_backtest_garch_fallback_events_consistent"] == "True"
    assert diagnostics_map["var_backtest_garch_status_ok_count_consistent"] == "True"
    assert diagnostics_map["quality_gates_pass_count"] == "22"
    assert diagnostics_map["quality_gates_fail_count"] == "2"
    assert diagnostics_map["quality_gates_unknown_count"] == "8"
    assert (
        diagnostics_map["quality_gates_failed_checks"]
        == "panel_daily_required_cols_present,panel_window_matches_target"
    )
    assert diagnostics_map["quality_gates_all_pass"] == "False"
    assert diagnostics_map["historical_kupiec_reject_rate_5pct_recorded"] == "0.0"
    assert diagnostics_map["historical_christoffersen_reject_rate_5pct_recorded"] == "0.0"
    assert diagnostics_map["parametric_kupiec_reject_rate_5pct_recorded"] == "nan"
    assert diagnostics_map["parametric_christoffersen_reject_rate_5pct_recorded"] == "nan"
    assert diagnostics_map["garch_kupiec_reject_rate_5pct_recorded"] == "nan"
    assert diagnostics_map["garch_christoffersen_reject_rate_5pct_recorded"] == "nan"
    assert diagnostics_map["historical_kupiec_reject_rate_5pct_observed"] == "0.0"
    assert diagnostics_map["historical_christoffersen_reject_rate_5pct_observed"] == "0.0"
    assert diagnostics_map["parametric_kupiec_reject_rate_5pct_observed"] == "nan"
    assert diagnostics_map["parametric_christoffersen_reject_rate_5pct_observed"] == "nan"
    assert diagnostics_map["garch_kupiec_reject_rate_5pct_observed"] == "nan"
    assert diagnostics_map["garch_christoffersen_reject_rate_5pct_observed"] == "nan"
    assert diagnostics_map["historical_kupiec_reject_rate_5pct_consistent"] == "True"
    assert diagnostics_map["historical_christoffersen_reject_rate_5pct_consistent"] == "True"
    assert diagnostics_map["parametric_kupiec_reject_rate_5pct_consistent"] == "True"
    assert diagnostics_map["parametric_christoffersen_reject_rate_5pct_consistent"] == "True"
    assert diagnostics_map["garch_kupiec_reject_rate_5pct_consistent"] == "True"
    assert diagnostics_map["garch_christoffersen_reject_rate_5pct_consistent"] == "True"
    assert diagnostics_map["fallback_reason_short_sample_count"] == "0"
    assert diagnostics_map["fallback_reason_nan_var_count"] == "0"
    assert diagnostics_map["fallback_reason_invalid_input_count"] == "0"
    assert diagnostics_map["fallback_reason_exception_count"] == "0"
    assert diagnostics_map["fallback_reason_upstream_fallback_count"] == "0"
    assert diagnostics_map["fallback_reason_unknown_count"] == "0"
    assert diagnostics_map["placeholder_source_detected"] == "False"


def test_build_diagnostics_table_handles_missing_inputs(tmp_path: Path) -> None:
    """Return safe defaults when all upstream artifacts are missing."""
    diagnostics = build_diagnostics_table(
        risk_summary_path=tmp_path / "missing_risk_summary.csv",
        scorecard_path=tmp_path / "missing_scorecard.json",
        nvda_raw_path=tmp_path / "missing_nvda.csv",
        market_raw_path=tmp_path / "missing_market.csv",
        rolling_vol_path=tmp_path / "missing_rolling_vol.csv",
        var_es_hist_path=tmp_path / "missing_var_es_hist.csv",
        var_exceedances_hist_path=tmp_path / "missing_var_exceedances_hist.csv",
        var_backtest_hist_path=tmp_path / "missing_var_backtest_hist.csv",
        vol_window_days=252,
        provenance_path=tmp_path / "missing_data_provenance.json",
    )

    diagnostics_map = _as_map(diagnostics)
    assert diagnostics_map["risk_summary_exists"] == "False"
    assert diagnostics_map["scorecard_exists"] == "False"
    assert diagnostics_map["scorecard_parse_error"] == "False"
    assert diagnostics_map["provenance_exists"] == "False"
    assert diagnostics_map["provenance_parse_error"] == "False"
    assert diagnostics_map["provenance_ticker_match"] == "unknown"
    assert diagnostics_map["provenance_sample_match"] == "unknown"
    assert diagnostics_map["provenance_panel_sha_match"] == "unknown"
    assert diagnostics_map["provenance_nvda_raw_sha_match"] == "unknown"
    assert diagnostics_map["provenance_market_raw_sha_match"] == "unknown"
    assert diagnostics_map["provenance_source_match"] == "unknown"
    assert diagnostics_map["panel_daily_exists"] == "False"
    assert diagnostics_map["panel_daily_rows"] == "0"
    assert diagnostics_map["panel_daily_required_cols_present"] == "unknown"
    assert diagnostics_map["panel_daily_date_monotonic"] == "unknown"
    assert diagnostics_map["panel_daily_duplicate_dates"] == "0"
    assert diagnostics_map["panel_daily_core_missing_rate"] == "nan"
    assert diagnostics_map["nvda_raw_rows"] == "0"
    assert diagnostics_map["market_raw_rows"] == "0"
    assert diagnostics_map["sample_target_start"] == "2020-01-01"
    assert diagnostics_map["sample_target_end"] == "2024-12-31"
    assert diagnostics_map["nvda_window_matches_target"] == "unknown"
    assert diagnostics_map["market_window_matches_target"] == "unknown"
    assert diagnostics_map["panel_window_matches_target"] == "unknown"
    assert diagnostics_map["scorecard_sample_target_start_recorded"] == "2020-01-01"
    assert diagnostics_map["scorecard_sample_target_end_recorded"] == "2024-12-31"
    assert diagnostics_map["scorecard_window_matches_target_recorded"] == "unknown"
    assert diagnostics_map["scorecard_window_matches_target_consistent"] == "unknown"
    assert diagnostics_map["scorecard_status_recorded"] == "unknown"
    assert diagnostics_map["scorecard_status_expected"] == "unknown"
    assert diagnostics_map["scorecard_status_consistent"] == "unknown"
    assert diagnostics_map["scorecard_status_reason_recorded"] == "unknown"
    assert diagnostics_map["scorecard_status_reason_expected"] == "unknown"
    assert diagnostics_map["scorecard_status_reason_consistent"] == "unknown"
    assert diagnostics_map["scorecard_metric_count"] == "0"
    assert diagnostics_map["fallback_events_recorded"] == "0"
    assert diagnostics_map["var_backtest_fallback_events_recorded"] == "0"
    assert diagnostics_map["var_backtest_historical_rows_recorded"] == "0"
    assert diagnostics_map["var_backtest_historical_fallback_events_recorded"] == "0"
    assert diagnostics_map["var_backtest_historical_rows_observed"] == "unknown"
    assert diagnostics_map["var_backtest_historical_fallback_events_observed"] == "unknown"
    assert diagnostics_map["var_backtest_historical_rows_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_historical_fallback_events_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_parametric_rows_recorded"] == "0"
    assert diagnostics_map["var_backtest_parametric_fallback_events_recorded"] == "0"
    assert diagnostics_map["var_backtest_parametric_rows_observed"] == "unknown"
    assert diagnostics_map["var_backtest_parametric_fallback_events_observed"] == "unknown"
    assert diagnostics_map["var_backtest_parametric_rows_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_parametric_fallback_events_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_other_rows_recorded"] == "0"
    assert diagnostics_map["var_backtest_other_fallback_events_recorded"] == "0"
    assert diagnostics_map["var_backtest_other_rows_observed"] == "unknown"
    assert diagnostics_map["var_backtest_other_fallback_events_observed"] == "unknown"
    assert diagnostics_map["var_backtest_other_rows_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_other_fallback_events_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_garch_rows_recorded"] == "0"
    assert diagnostics_map["var_backtest_garch_fallback_events_recorded"] == "0"
    assert diagnostics_map["var_backtest_garch_status_ok_count_recorded"] == "0"
    assert diagnostics_map["var_backtest_garch_rows_observed"] == "unknown"
    assert diagnostics_map["var_backtest_garch_fallback_events_observed"] == "unknown"
    assert diagnostics_map["var_backtest_garch_status_ok_count_observed"] == "unknown"
    assert diagnostics_map["var_backtest_garch_rows_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_garch_fallback_events_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_garch_status_ok_count_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_historical_rows_observed"] == "unknown"
    assert diagnostics_map["var_backtest_historical_fallback_events_observed"] == "unknown"
    assert diagnostics_map["var_backtest_historical_rows_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_historical_fallback_events_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_parametric_rows_observed"] == "unknown"
    assert diagnostics_map["var_backtest_parametric_fallback_events_observed"] == "unknown"
    assert diagnostics_map["var_backtest_parametric_rows_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_parametric_fallback_events_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_other_rows_observed"] == "unknown"
    assert diagnostics_map["var_backtest_other_fallback_events_observed"] == "unknown"
    assert diagnostics_map["var_backtest_other_rows_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_other_fallback_events_consistent"] == "unknown"
    assert diagnostics_map["historical_kupiec_reject_rate_5pct_observed"] == "unknown"
    assert diagnostics_map["historical_christoffersen_reject_rate_5pct_observed"] == "unknown"
    assert diagnostics_map["parametric_kupiec_reject_rate_5pct_observed"] == "unknown"
    assert diagnostics_map["parametric_christoffersen_reject_rate_5pct_observed"] == "unknown"
    assert diagnostics_map["historical_kupiec_reject_rate_5pct_consistent"] == "unknown"
    assert diagnostics_map["historical_christoffersen_reject_rate_5pct_consistent"] == "unknown"
    assert diagnostics_map["parametric_kupiec_reject_rate_5pct_consistent"] == "unknown"
    assert diagnostics_map["parametric_christoffersen_reject_rate_5pct_consistent"] == "unknown"
    assert diagnostics_map["garch_kupiec_reject_rate_5pct_observed"] == "unknown"
    assert diagnostics_map["garch_christoffersen_reject_rate_5pct_observed"] == "unknown"
    assert diagnostics_map["garch_kupiec_reject_rate_5pct_consistent"] == "unknown"
    assert diagnostics_map["garch_christoffersen_reject_rate_5pct_consistent"] == "unknown"
    assert diagnostics_map["quality_gates_pass_count"] == "0"
    assert diagnostics_map["quality_gates_fail_count"] == "1"
    assert diagnostics_map["quality_gates_unknown_count"] == "31"
    assert diagnostics_map["quality_gates_failed_checks"] == "provenance_exists"
    assert diagnostics_map["quality_gates_all_pass"] == "False"
    assert diagnostics_map["historical_kupiec_reject_rate_5pct_recorded"] == "nan"
    assert diagnostics_map["historical_christoffersen_reject_rate_5pct_recorded"] == "nan"
    assert diagnostics_map["parametric_kupiec_reject_rate_5pct_recorded"] == "nan"
    assert diagnostics_map["parametric_christoffersen_reject_rate_5pct_recorded"] == "nan"
    assert diagnostics_map["garch_kupiec_reject_rate_5pct_recorded"] == "nan"
    assert diagnostics_map["garch_christoffersen_reject_rate_5pct_recorded"] == "nan"
    assert diagnostics_map["historical_kupiec_reject_rate_5pct_observed"] == "unknown"
    assert diagnostics_map["historical_christoffersen_reject_rate_5pct_observed"] == "unknown"
    assert diagnostics_map["parametric_kupiec_reject_rate_5pct_observed"] == "unknown"
    assert diagnostics_map["parametric_christoffersen_reject_rate_5pct_observed"] == "unknown"
    assert diagnostics_map["historical_kupiec_reject_rate_5pct_consistent"] == "unknown"
    assert diagnostics_map["historical_christoffersen_reject_rate_5pct_consistent"] == "unknown"
    assert diagnostics_map["parametric_kupiec_reject_rate_5pct_consistent"] == "unknown"
    assert diagnostics_map["parametric_christoffersen_reject_rate_5pct_consistent"] == "unknown"
    assert diagnostics_map["garch_kupiec_reject_rate_5pct_observed"] == "unknown"
    assert diagnostics_map["garch_christoffersen_reject_rate_5pct_observed"] == "unknown"
    assert diagnostics_map["garch_kupiec_reject_rate_5pct_consistent"] == "unknown"
    assert diagnostics_map["garch_christoffersen_reject_rate_5pct_consistent"] == "unknown"
    assert diagnostics_map["fallback_reason_short_sample_count"] == "0"
    assert diagnostics_map["fallback_reason_nan_var_count"] == "0"
    assert diagnostics_map["fallback_reason_invalid_input_count"] == "0"
    assert diagnostics_map["fallback_reason_exception_count"] == "0"
    assert diagnostics_map["fallback_reason_upstream_fallback_count"] == "0"
    assert diagnostics_map["fallback_reason_unknown_count"] == "0"
    assert diagnostics_map["data_source"] == "unknown"
    assert diagnostics_map["placeholder_source_detected"] == "unknown"
    assert diagnostics_map["rolling_vol_window_days"] == "252"
    assert diagnostics_map["rolling_vol_total_rows"] == "0"
    assert diagnostics_map["rolling_vol_valid_points"] == "0"
    assert diagnostics_map["rolling_vol_fallback_used"] == "unknown"
    assert diagnostics_map["hist_var_es_exists"] == "False"
    assert diagnostics_map["hist_var_es_method_count"] == "0"
    assert diagnostics_map["parametric_var_es_present"] == "False"
    assert diagnostics_map["garch_var_es_present"] == "False"
    assert diagnostics_map["hist_var_es_alpha_count"] == "0"
    assert diagnostics_map["hist_var_es_status_ok_count"] == "0"
    assert diagnostics_map["hist_var_95"] == "nan"
    assert diagnostics_map["hist_es_99"] == "nan"
    assert diagnostics_map["param_var_95"] == "nan"
    assert diagnostics_map["param_es_99"] == "nan"
    assert diagnostics_map["garch_var_es_nu_finite"] == "unknown"
    assert diagnostics_map["garch_var_es_nu_gt4"] == "unknown"
    assert diagnostics_map["garch_var_es_sigma_next_positive"] == "unknown"
    assert diagnostics_map["macro_data_exists"] == "False"
    assert diagnostics_map["macro_rows"] == "0"
    assert diagnostics_map["macro_month_end_only"] == "unknown"
    assert diagnostics_map["macro_duplicate_month"] == "unknown"
    assert diagnostics_map["macro_frequency_ok"] == "unknown"
    assert diagnostics_map["hist_exceed_exists"] == "False"
    assert diagnostics_map["hist_exceed_method_count"] == "0"
    assert diagnostics_map["parametric_exceed_present"] == "False"
    assert diagnostics_map["garch_exceed_present"] == "False"
    assert diagnostics_map["hist_exceed_status_ok_count"] == "0"
    assert diagnostics_map["hist_exceed_95_count"] == "0"
    assert diagnostics_map["hist_exceed_95_gap"] == "nan"
    assert diagnostics_map["hist_exceed_99_rate"] == "nan"
    assert diagnostics_map["hist_exceed_99_gap"] == "nan"
    assert diagnostics_map["hist_backtest_exists"] == "False"
    assert diagnostics_map["hist_backtest_method_count"] == "0"
    assert diagnostics_map["parametric_backtest_present"] == "False"
    assert diagnostics_map["garch_backtest_present"] == "False"
    assert diagnostics_map["hist_backtest_alpha_count"] == "0"
    assert diagnostics_map["hist_backtest_status_ok_count"] == "0"
    assert diagnostics_map["hist_kupiec_lr_95"] == "nan"
    assert diagnostics_map["hist_kupiec_lr_99"] == "nan"
    assert diagnostics_map["hist_kupiec_pvalue_95"] == "nan"
    assert diagnostics_map["hist_kupiec_pvalue_99"] == "nan"
    assert diagnostics_map["param_kupiec_lr_95"] == "nan"
    assert diagnostics_map["param_kupiec_lr_99"] == "nan"
    assert diagnostics_map["param_kupiec_pvalue_95"] == "nan"
    assert diagnostics_map["param_kupiec_pvalue_99"] == "nan"
    assert diagnostics_map["hist_christoffersen_lr_95"] == "nan"
    assert diagnostics_map["hist_christoffersen_lr_99"] == "nan"
    assert diagnostics_map["hist_christoffersen_pvalue_95"] == "nan"
    assert diagnostics_map["hist_christoffersen_pvalue_99"] == "nan"
    assert diagnostics_map["param_christoffersen_lr_95"] == "nan"
    assert diagnostics_map["param_christoffersen_lr_99"] == "nan"
    assert diagnostics_map["param_christoffersen_pvalue_95"] == "nan"
    assert diagnostics_map["param_christoffersen_pvalue_99"] == "nan"
    assert diagnostics_map["garch_kupiec_lr_95"] == "nan"
    assert diagnostics_map["garch_kupiec_lr_99"] == "nan"
    assert diagnostics_map["garch_kupiec_pvalue_95"] == "nan"
    assert diagnostics_map["garch_kupiec_pvalue_99"] == "nan"
    assert diagnostics_map["garch_christoffersen_lr_95"] == "nan"
    assert diagnostics_map["garch_christoffersen_lr_99"] == "nan"
    assert diagnostics_map["garch_christoffersen_pvalue_95"] == "nan"
    assert diagnostics_map["garch_christoffersen_pvalue_99"] == "nan"
    assert diagnostics_map["garch_backtest_upstream_exists"] == "False"
    assert diagnostics_map["garch_backtest_upstream_converged_present"] == "unknown"
    assert diagnostics_map["garch_backtest_upstream_converged_true_count"] == "unknown"
    assert diagnostics_map["garch_backtest_upstream_nu_finite"] == "unknown"
    assert diagnostics_map["garch_backtest_upstream_sigma_positive"] == "unknown"


def test_build_diagnostics_table_handles_invalid_scorecard_json(tmp_path: Path) -> None:
    """Flag scorecard parse errors without breaking diagnostics creation."""
    scorecard = tmp_path / "scorecard.json"
    scorecard.write_text("{invalid}", encoding="utf-8")

    diagnostics = build_diagnostics_table(
        risk_summary_path=tmp_path / "risk_summary.csv",
        scorecard_path=scorecard,
        nvda_raw_path=tmp_path / "nvda.csv",
        market_raw_path=tmp_path / "market.csv",
        rolling_vol_path=tmp_path / "rolling_vol.csv",
        var_es_hist_path=tmp_path / "var_es_hist.csv",
        var_exceedances_hist_path=tmp_path / "var_exceedances_hist.csv",
        vol_window_days=252,
    )

    diagnostics_map = _as_map(diagnostics)
    assert diagnostics_map["scorecard_exists"] == "True"
    assert diagnostics_map["scorecard_parse_error"] == "True"
    assert diagnostics_map["scorecard_metric_count"] == "0"
    assert diagnostics_map["var_backtest_fallback_events_recorded"] == "0"
    assert diagnostics_map["var_backtest_historical_rows_recorded"] == "0"
    assert diagnostics_map["var_backtest_historical_fallback_events_recorded"] == "0"
    assert diagnostics_map["var_backtest_historical_rows_observed"] == "unknown"
    assert diagnostics_map["var_backtest_historical_fallback_events_observed"] == "unknown"
    assert diagnostics_map["var_backtest_historical_rows_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_historical_fallback_events_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_parametric_rows_recorded"] == "0"
    assert diagnostics_map["var_backtest_parametric_fallback_events_recorded"] == "0"
    assert diagnostics_map["var_backtest_parametric_rows_observed"] == "unknown"
    assert diagnostics_map["var_backtest_parametric_fallback_events_observed"] == "unknown"
    assert diagnostics_map["var_backtest_parametric_rows_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_parametric_fallback_events_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_other_rows_recorded"] == "0"
    assert diagnostics_map["var_backtest_other_fallback_events_recorded"] == "0"
    assert diagnostics_map["var_backtest_other_rows_observed"] == "unknown"
    assert diagnostics_map["var_backtest_other_fallback_events_observed"] == "unknown"
    assert diagnostics_map["var_backtest_other_rows_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_other_fallback_events_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_garch_rows_recorded"] == "0"
    assert diagnostics_map["var_backtest_garch_fallback_events_recorded"] == "0"
    assert diagnostics_map["var_backtest_garch_status_ok_count_recorded"] == "0"
    assert diagnostics_map["var_backtest_garch_rows_observed"] == "unknown"
    assert diagnostics_map["var_backtest_garch_fallback_events_observed"] == "unknown"
    assert diagnostics_map["var_backtest_garch_status_ok_count_observed"] == "unknown"
    assert diagnostics_map["var_backtest_garch_rows_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_garch_fallback_events_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_garch_status_ok_count_consistent"] == "unknown"
    assert diagnostics_map["quality_gates_failed_checks"] == "provenance_exists"
    assert diagnostics_map["quality_gates_all_pass"] == "False"
    assert diagnostics_map["historical_kupiec_reject_rate_5pct_recorded"] == "nan"
    assert diagnostics_map["historical_christoffersen_reject_rate_5pct_recorded"] == "nan"
    assert diagnostics_map["parametric_kupiec_reject_rate_5pct_recorded"] == "nan"
    assert diagnostics_map["parametric_christoffersen_reject_rate_5pct_recorded"] == "nan"
    assert diagnostics_map["garch_kupiec_reject_rate_5pct_recorded"] == "nan"
    assert diagnostics_map["garch_christoffersen_reject_rate_5pct_recorded"] == "nan"
    assert diagnostics_map["historical_kupiec_reject_rate_5pct_observed"] == "unknown"
    assert diagnostics_map["historical_christoffersen_reject_rate_5pct_observed"] == "unknown"
    assert diagnostics_map["parametric_kupiec_reject_rate_5pct_observed"] == "unknown"
    assert diagnostics_map["parametric_christoffersen_reject_rate_5pct_observed"] == "unknown"
    assert diagnostics_map["garch_kupiec_reject_rate_5pct_observed"] == "unknown"
    assert diagnostics_map["garch_christoffersen_reject_rate_5pct_observed"] == "unknown"
    assert diagnostics_map["historical_kupiec_reject_rate_5pct_consistent"] == "unknown"
    assert diagnostics_map["historical_christoffersen_reject_rate_5pct_consistent"] == "unknown"
    assert diagnostics_map["parametric_kupiec_reject_rate_5pct_consistent"] == "unknown"
    assert diagnostics_map["parametric_christoffersen_reject_rate_5pct_consistent"] == "unknown"
    assert diagnostics_map["garch_kupiec_reject_rate_5pct_consistent"] == "unknown"
    assert diagnostics_map["garch_christoffersen_reject_rate_5pct_consistent"] == "unknown"
    assert diagnostics_map["fallback_reason_short_sample_count"] == "0"
    assert diagnostics_map["fallback_reason_unknown_count"] == "0"


def test_build_diagnostics_table_handles_invalid_provenance_json(tmp_path: Path) -> None:
    """Flag provenance parse errors and keep diagnostics resilient."""
    provenance = tmp_path / "data_provenance.json"
    provenance.write_text("{invalid}", encoding="utf-8")

    diagnostics = build_diagnostics_table(
        risk_summary_path=tmp_path / "risk_summary.csv",
        scorecard_path=tmp_path / "scorecard.json",
        nvda_raw_path=tmp_path / "nvda.csv",
        market_raw_path=tmp_path / "market.csv",
        rolling_vol_path=tmp_path / "rolling_vol.csv",
        var_es_hist_path=tmp_path / "var_es_hist.csv",
        var_exceedances_hist_path=tmp_path / "var_exceedances_hist.csv",
        vol_window_days=252,
        provenance_path=provenance,
    )
    diagnostics_map = _as_map(diagnostics)
    assert diagnostics_map["provenance_exists"] == "True"
    assert diagnostics_map["provenance_parse_error"] == "True"
    assert diagnostics_map["provenance_ticker_match"] == "unknown"
    assert diagnostics_map["provenance_sample_match"] == "unknown"
    assert diagnostics_map["provenance_panel_sha_match"] == "unknown"
    assert diagnostics_map["provenance_nvda_raw_sha_match"] == "unknown"
    assert diagnostics_map["provenance_market_raw_sha_match"] == "unknown"
    assert diagnostics_map["provenance_source_match"] == "unknown"


def test_build_diagnostics_table_handles_provenance_without_source(tmp_path: Path) -> None:
    """Leave source matching unknown when provenance has no source section."""
    nvda_raw = tmp_path / "nvda_raw.csv"
    market_raw = tmp_path / "market_raw.csv"
    provenance = tmp_path / "data_provenance.json"

    pd.DataFrame({"date": ["2024-01-01"], "data_source": ["snapshot"]}).to_csv(
        nvda_raw,
        index=False,
    )
    pd.DataFrame({"date": ["2024-01-01"], "data_source": ["online"]}).to_csv(
        market_raw,
        index=False,
    )
    provenance.write_text(json.dumps({"tickers": ["NVDA", "^GSPC"]}), encoding="utf-8")

    diagnostics = build_diagnostics_table(
        risk_summary_path=tmp_path / "risk_summary.csv",
        scorecard_path=tmp_path / "scorecard.json",
        nvda_raw_path=nvda_raw,
        market_raw_path=market_raw,
        rolling_vol_path=tmp_path / "rolling_vol.csv",
        var_es_hist_path=tmp_path / "var_es_hist.csv",
        var_exceedances_hist_path=tmp_path / "var_exceedances_hist.csv",
        vol_window_days=252,
        provenance_path=provenance,
    )
    diagnostics_map = _as_map(diagnostics)
    assert diagnostics_map["provenance_source_match"] == "unknown"


def test_build_diagnostics_table_flags_conflicting_provenance_source(tmp_path: Path) -> None:
    """Flag False when provenance source disagrees with observed raw data source."""
    nvda_raw = tmp_path / "nvda_raw.csv"
    market_raw = tmp_path / "market_raw.csv"
    provenance = tmp_path / "data_provenance.json"

    pd.DataFrame({"date": ["2024-01-01"], "data_source": ["snapshot"]}).to_csv(
        nvda_raw,
        index=False,
    )
    pd.DataFrame({"date": ["2024-01-01"], "data_source": ["online"]}).to_csv(
        market_raw,
        index=False,
    )
    provenance.write_text(
        json.dumps({"source": {"NVDA": "online", "^GSPC": "online"}}),
        encoding="utf-8",
    )

    diagnostics = build_diagnostics_table(
        risk_summary_path=tmp_path / "risk_summary.csv",
        scorecard_path=tmp_path / "scorecard.json",
        nvda_raw_path=nvda_raw,
        market_raw_path=market_raw,
        rolling_vol_path=tmp_path / "rolling_vol.csv",
        var_es_hist_path=tmp_path / "var_es_hist.csv",
        var_exceedances_hist_path=tmp_path / "var_exceedances_hist.csv",
        vol_window_days=252,
        provenance_path=provenance,
    )
    diagnostics_map = _as_map(diagnostics)
    assert diagnostics_map["provenance_source_match"] == "False"


def test_build_diagnostics_table_uses_cache_for_raw_without_source_column(tmp_path: Path) -> None:
    """Infer cache source when raw CSVs do not include a source column."""
    nvda_raw = tmp_path / "nvda_raw.csv"
    market_raw = tmp_path / "market_raw.csv"
    pd.DataFrame({"date": ["2024-01-01"], "close": [100], "volume": [1000]}).to_csv(
        nvda_raw,
        index=False,
    )
    pd.DataFrame({"date": ["2024-01-01"], "close": [4000], "volume": [2000]}).to_csv(
        market_raw,
        index=False,
    )

    diagnostics = build_diagnostics_table(
        risk_summary_path=tmp_path / "risk_summary.csv",
        scorecard_path=tmp_path / "scorecard.json",
        nvda_raw_path=nvda_raw,
        market_raw_path=market_raw,
        rolling_vol_path=tmp_path / "rolling_vol.csv",
        var_es_hist_path=tmp_path / "var_es_hist.csv",
        var_exceedances_hist_path=tmp_path / "var_exceedances_hist.csv",
        vol_window_days=252,
    )
    diagnostics_map = _as_map(diagnostics)
    assert diagnostics_map["data_source"] == "cache"
    assert diagnostics_map["placeholder_source_detected"] == "False"


def test_build_diagnostics_table_detects_placeholder_source(tmp_path: Path) -> None:
    """Flag placeholder source when either raw input uses placeholder rows."""
    nvda_raw = tmp_path / "nvda_raw.csv"
    market_raw = tmp_path / "market_raw.csv"
    pd.DataFrame({"date": ["2024-01-01"], "data_source": ["placeholder"]}).to_csv(
        nvda_raw,
        index=False,
    )
    pd.DataFrame({"date": ["2024-01-01"], "data_source": ["snapshot"]}).to_csv(
        market_raw,
        index=False,
    )

    diagnostics = build_diagnostics_table(
        risk_summary_path=tmp_path / "risk_summary.csv",
        scorecard_path=tmp_path / "scorecard.json",
        nvda_raw_path=nvda_raw,
        market_raw_path=market_raw,
        rolling_vol_path=tmp_path / "rolling_vol.csv",
        var_es_hist_path=tmp_path / "var_es_hist.csv",
        var_exceedances_hist_path=tmp_path / "var_exceedances_hist.csv",
        vol_window_days=252,
    )
    diagnostics_map = _as_map(diagnostics)
    assert diagnostics_map["placeholder_source_detected"] == "True"


def test_build_diagnostics_table_handles_malformed_var_es_hist(tmp_path: Path) -> None:
    """Use defaults when historical VaR/ES artifact columns are malformed."""
    var_es_hist = tmp_path / "var_es_hist.csv"
    pd.DataFrame({"bad_col": [1, 2]}).to_csv(var_es_hist, index=False)

    diagnostics = build_diagnostics_table(
        risk_summary_path=tmp_path / "risk_summary.csv",
        scorecard_path=tmp_path / "scorecard.json",
        nvda_raw_path=tmp_path / "nvda.csv",
        market_raw_path=tmp_path / "market.csv",
        rolling_vol_path=tmp_path / "rolling_vol.csv",
        var_es_hist_path=var_es_hist,
        var_exceedances_hist_path=tmp_path / "var_exceedances_hist.csv",
        vol_window_days=252,
    )
    diagnostics_map = _as_map(diagnostics)
    assert diagnostics_map["hist_var_es_exists"] == "True"
    assert diagnostics_map["hist_var_es_method_count"] == "0"
    assert diagnostics_map["parametric_var_es_present"] == "False"
    assert diagnostics_map["garch_var_es_present"] == "False"
    assert diagnostics_map["hist_var_es_alpha_count"] == "0"
    assert diagnostics_map["hist_var_95"] == "nan"
    assert diagnostics_map["param_var_95"] == "nan"
    assert diagnostics_map["garch_var_es_nu_finite"] == "unknown"
    assert diagnostics_map["garch_var_es_nu_gt4"] == "unknown"
    assert diagnostics_map["garch_var_es_sigma_next_positive"] == "unknown"


def test_build_diagnostics_table_handles_malformed_var_exceedances_hist(tmp_path: Path) -> None:
    """Use defaults when historical exceedance artifact columns are malformed."""
    var_exceedances_hist = tmp_path / "var_exceedances_hist.csv"
    pd.DataFrame({"bad_col": [1, 2]}).to_csv(var_exceedances_hist, index=False)

    diagnostics = build_diagnostics_table(
        risk_summary_path=tmp_path / "risk_summary.csv",
        scorecard_path=tmp_path / "scorecard.json",
        nvda_raw_path=tmp_path / "nvda.csv",
        market_raw_path=tmp_path / "market.csv",
        rolling_vol_path=tmp_path / "rolling_vol.csv",
        var_es_hist_path=tmp_path / "var_es_hist.csv",
        var_exceedances_hist_path=var_exceedances_hist,
        vol_window_days=252,
    )
    diagnostics_map = _as_map(diagnostics)
    assert diagnostics_map["hist_exceed_exists"] == "True"
    assert diagnostics_map["hist_exceed_method_count"] == "0"
    assert diagnostics_map["parametric_exceed_present"] == "False"
    assert diagnostics_map["garch_exceed_present"] == "False"
    assert diagnostics_map["hist_exceed_status_ok_count"] == "0"
    assert diagnostics_map["hist_exceed_95_count"] == "0"
    assert diagnostics_map["hist_exceed_95_rate"] == "nan"
    assert diagnostics_map["hist_exceed_95_gap"] == "nan"
    assert diagnostics_map["hist_exceed_99_gap"] == "nan"


def test_build_diagnostics_table_handles_malformed_var_backtest_hist(tmp_path: Path) -> None:
    """Use defaults when historical backtest artifact columns are malformed."""
    var_backtest_hist = tmp_path / "var_backtest_hist.csv"
    pd.DataFrame({"bad_col": [1, 2]}).to_csv(var_backtest_hist, index=False)

    diagnostics = build_diagnostics_table(
        risk_summary_path=tmp_path / "risk_summary.csv",
        scorecard_path=tmp_path / "scorecard.json",
        nvda_raw_path=tmp_path / "nvda.csv",
        market_raw_path=tmp_path / "market.csv",
        rolling_vol_path=tmp_path / "rolling_vol.csv",
        var_es_hist_path=tmp_path / "var_es_hist.csv",
        var_exceedances_hist_path=tmp_path / "var_exceedances_hist.csv",
        var_backtest_hist_path=var_backtest_hist,
        vol_window_days=252,
    )
    diagnostics_map = _as_map(diagnostics)
    assert diagnostics_map["hist_backtest_exists"] == "True"
    assert diagnostics_map["hist_backtest_method_count"] == "0"
    assert diagnostics_map["parametric_backtest_present"] == "False"
    assert diagnostics_map["garch_backtest_present"] == "False"
    assert diagnostics_map["hist_backtest_alpha_count"] == "0"
    assert diagnostics_map["hist_backtest_status_ok_count"] == "0"
    assert diagnostics_map["hist_kupiec_pvalue_95"] == "nan"
    assert diagnostics_map["hist_christoffersen_pvalue_99"] == "nan"
    assert diagnostics_map["param_kupiec_pvalue_95"] == "nan"
    assert diagnostics_map["param_christoffersen_pvalue_99"] == "nan"
    assert diagnostics_map["garch_backtest_upstream_exists"] == "True"
    assert diagnostics_map["garch_backtest_upstream_converged_present"] == "unknown"
    assert diagnostics_map["garch_backtest_upstream_converged_true_count"] == "unknown"
    assert diagnostics_map["garch_backtest_upstream_nu_finite"] == "unknown"
    assert diagnostics_map["garch_backtest_upstream_sigma_positive"] == "unknown"
    assert diagnostics_map["var_backtest_garch_rows_observed"] == "unknown"
    assert diagnostics_map["var_backtest_garch_fallback_events_observed"] == "unknown"
    assert diagnostics_map["var_backtest_garch_status_ok_count_observed"] == "unknown"
    assert diagnostics_map["var_backtest_garch_rows_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_garch_fallback_events_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_garch_status_ok_count_consistent"] == "unknown"


def test_build_diagnostics_table_reads_fallback_reason_counts_from_scorecard(tmp_path: Path) -> None:
    """Expose reason-level fallback counters from the scorecard payload."""
    scorecard = tmp_path / "scorecard.json"
    scorecard.write_text(
        json.dumps(
            {
                "var_backtest_fallback_events": 4,
                "var_backtest_historical_rows": 2,
                "var_backtest_historical_fallback_events": 1,
                "var_backtest_parametric_rows": 2,
                "var_backtest_parametric_fallback_events": 2,
                "var_backtest_other_rows": 0,
                "var_backtest_other_fallback_events": 0,
                "var_backtest_garch_rows": 2,
                "var_backtest_garch_fallback_events": 1,
                "var_backtest_garch_status_ok_count": 1,
                "historical_kupiec_reject_rate_5pct": 0.5,
                "historical_christoffersen_reject_rate_5pct": 0.0,
                "parametric_kupiec_reject_rate_5pct": 1.0,
                "parametric_christoffersen_reject_rate_5pct": 0.5,
                "var_backtest_fallback_reason_short_sample_count": 2,
                "var_backtest_fallback_reason_nan_var_count": 1,
                "var_backtest_fallback_reason_invalid_input_count": 0,
                "var_backtest_fallback_reason_exception_count": 1,
                "var_backtest_fallback_reason_upstream_fallback_count": 0,
                "var_backtest_fallback_reason_unknown_count": 0,
            },
        ),
        encoding="utf-8",
    )

    diagnostics = build_diagnostics_table(
        risk_summary_path=tmp_path / "risk_summary.csv",
        scorecard_path=scorecard,
        nvda_raw_path=tmp_path / "nvda.csv",
        market_raw_path=tmp_path / "market.csv",
        rolling_vol_path=tmp_path / "rolling_vol.csv",
        var_es_hist_path=tmp_path / "var_es_hist.csv",
        var_exceedances_hist_path=tmp_path / "var_exceedances_hist.csv",
        var_backtest_hist_path=tmp_path / "var_backtest_hist.csv",
        vol_window_days=252,
    )
    diagnostics_map = _as_map(diagnostics)

    assert diagnostics_map["var_backtest_fallback_events_recorded"] == "4"
    assert diagnostics_map["var_backtest_historical_rows_recorded"] == "2"
    assert diagnostics_map["var_backtest_historical_fallback_events_recorded"] == "1"
    assert diagnostics_map["var_backtest_historical_rows_observed"] == "unknown"
    assert diagnostics_map["var_backtest_historical_fallback_events_observed"] == "unknown"
    assert diagnostics_map["var_backtest_historical_rows_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_historical_fallback_events_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_parametric_rows_recorded"] == "2"
    assert diagnostics_map["var_backtest_parametric_fallback_events_recorded"] == "2"
    assert diagnostics_map["var_backtest_parametric_rows_observed"] == "unknown"
    assert diagnostics_map["var_backtest_parametric_fallback_events_observed"] == "unknown"
    assert diagnostics_map["var_backtest_parametric_rows_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_parametric_fallback_events_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_other_rows_recorded"] == "0"
    assert diagnostics_map["var_backtest_other_fallback_events_recorded"] == "0"
    assert diagnostics_map["var_backtest_other_rows_observed"] == "unknown"
    assert diagnostics_map["var_backtest_other_fallback_events_observed"] == "unknown"
    assert diagnostics_map["var_backtest_other_rows_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_other_fallback_events_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_garch_rows_recorded"] == "2"
    assert diagnostics_map["var_backtest_garch_fallback_events_recorded"] == "1"
    assert diagnostics_map["var_backtest_garch_status_ok_count_recorded"] == "1"
    assert diagnostics_map["var_backtest_garch_rows_observed"] == "unknown"
    assert diagnostics_map["var_backtest_garch_fallback_events_observed"] == "unknown"
    assert diagnostics_map["var_backtest_garch_status_ok_count_observed"] == "unknown"
    assert diagnostics_map["var_backtest_garch_rows_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_garch_fallback_events_consistent"] == "unknown"
    assert diagnostics_map["var_backtest_garch_status_ok_count_consistent"] == "unknown"
    assert diagnostics_map["historical_kupiec_reject_rate_5pct_recorded"] == "0.5"
    assert diagnostics_map["historical_christoffersen_reject_rate_5pct_recorded"] == "0.0"
    assert diagnostics_map["parametric_kupiec_reject_rate_5pct_recorded"] == "1.0"
    assert diagnostics_map["parametric_christoffersen_reject_rate_5pct_recorded"] == "0.5"
    assert diagnostics_map["garch_kupiec_reject_rate_5pct_recorded"] == "nan"
    assert diagnostics_map["garch_christoffersen_reject_rate_5pct_recorded"] == "nan"
    assert diagnostics_map["fallback_reason_short_sample_count"] == "2"
    assert diagnostics_map["fallback_reason_nan_var_count"] == "1"
    assert diagnostics_map["fallback_reason_exception_count"] == "1"
    assert diagnostics_map["fallback_reason_unknown_count"] == "0"


def test_build_diagnostics_table_flags_garch_backtest_count_inconsistency(tmp_path: Path) -> None:
    """Mark garch backtest count checks as False when scorecard and artifact disagree."""
    scorecard = tmp_path / "scorecard.json"
    var_backtest_hist = tmp_path / "var_backtest_hist.csv"
    scorecard.write_text(
        json.dumps(
            {
                "var_backtest_garch_rows": 1,
                "var_backtest_garch_fallback_events": 0,
                "var_backtest_garch_status_ok_count": 0,
            },
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "method": ["garch_t", "garch_t"],
            "alpha": [0.95, 0.99],
            "kupiec_p_value": [0.1, 0.2],
            "christoffersen_p_value": [0.3, 0.4],
            "status": ["ok", "fallback"],
        },
    ).to_csv(var_backtest_hist, index=False)

    diagnostics = build_diagnostics_table(
        risk_summary_path=tmp_path / "risk_summary.csv",
        scorecard_path=scorecard,
        nvda_raw_path=tmp_path / "nvda.csv",
        market_raw_path=tmp_path / "market.csv",
        rolling_vol_path=tmp_path / "rolling_vol.csv",
        var_es_hist_path=tmp_path / "var_es_hist.csv",
        var_exceedances_hist_path=tmp_path / "var_exceedances_hist.csv",
        var_backtest_hist_path=var_backtest_hist,
        vol_window_days=252,
    )
    diagnostics_map = _as_map(diagnostics)

    assert diagnostics_map["var_backtest_garch_rows_observed"] == "2"
    assert diagnostics_map["var_backtest_garch_fallback_events_observed"] == "1"
    assert diagnostics_map["var_backtest_garch_status_ok_count_observed"] == "1"
    assert diagnostics_map["var_backtest_historical_rows_observed"] == "0"
    assert diagnostics_map["var_backtest_historical_fallback_events_observed"] == "0"
    assert diagnostics_map["var_backtest_historical_rows_consistent"] == "True"
    assert diagnostics_map["var_backtest_historical_fallback_events_consistent"] == "True"
    assert diagnostics_map["var_backtest_parametric_rows_observed"] == "0"
    assert diagnostics_map["var_backtest_parametric_fallback_events_observed"] == "0"
    assert diagnostics_map["var_backtest_parametric_rows_consistent"] == "True"
    assert diagnostics_map["var_backtest_parametric_fallback_events_consistent"] == "True"
    assert diagnostics_map["var_backtest_other_rows_observed"] == "2"
    assert diagnostics_map["var_backtest_other_fallback_events_observed"] == "1"
    assert diagnostics_map["var_backtest_other_rows_consistent"] == "False"
    assert diagnostics_map["var_backtest_other_fallback_events_consistent"] == "False"
    assert diagnostics_map["var_backtest_garch_rows_consistent"] == "False"
    assert diagnostics_map["var_backtest_garch_fallback_events_consistent"] == "False"
    assert diagnostics_map["var_backtest_garch_status_ok_count_consistent"] == "False"
    assert diagnostics_map["garch_kupiec_reject_rate_5pct_observed"] == "0.0"
    assert diagnostics_map["garch_christoffersen_reject_rate_5pct_observed"] == "0.0"
    assert diagnostics_map["garch_kupiec_reject_rate_5pct_consistent"] == "False"
    assert diagnostics_map["garch_christoffersen_reject_rate_5pct_consistent"] == "False"


def test_build_diagnostics_table_detects_parametric_method_presence(tmp_path: Path) -> None:
    """Record method coverage when historical, parametric, and garch outputs coexist."""
    var_es_hist = tmp_path / "var_es_hist.csv"
    var_exceedances_hist = tmp_path / "var_exceedances_hist.csv"
    var_backtest_hist = tmp_path / "var_backtest_hist.csv"
    pd.DataFrame(
        {
            "method": [
                "historical",
                "historical",
                "parametric_normal",
                "parametric_normal",
                "garch_t",
                "garch_t",
            ],
            "alpha": [0.95, 0.99, 0.95, 0.99, 0.95, 0.99],
            "var": [-0.03, -0.04, -0.028, -0.038, -0.029, -0.039],
            "es": [-0.035, -0.045, -0.032, -0.041, -0.034, -0.043],
            "status": ["ok", "ok", "ok", "ok", "ok", "ok"],
            "nu": [float("nan"), float("nan"), float("nan"), float("nan"), 12.0, 12.5],
            "sigma_next": [float("nan"), float("nan"), float("nan"), float("nan"), 0.02, 0.021],
        },
    ).to_csv(var_es_hist, index=False)
    pd.DataFrame(
        {
            "method": [
                "historical",
                "historical",
                "parametric_normal",
                "parametric_normal",
                "garch_t",
                "garch_t",
            ],
            "alpha": [0.95, 0.99, 0.95, 0.99, 0.95, 0.99],
            "exceedance_count": [5, 1, 4, 1, 4, 1],
            "exceedance_rate": [0.05, 0.01, 0.04, 0.01, 0.04, 0.01],
            "status": ["ok", "ok", "ok", "ok", "ok", "ok"],
        },
    ).to_csv(var_exceedances_hist, index=False)
    pd.DataFrame(
        {
            "method": [
                "historical",
                "historical",
                "parametric_normal",
                "parametric_normal",
                "garch_t",
                "garch_t",
            ],
            "alpha": [0.95, 0.99, 0.95, 0.99, 0.95, 0.99],
            "kupiec_lr_stat": [0.5, 1.2, 0.6, 1.1, 0.7, 1.0],
            "kupiec_p_value": [0.12, 0.08, 0.2, 0.11, 0.19, 0.1],
            "christoffersen_lr_stat": [0.4, 0.9, 0.45, 0.85, 0.5, 0.8],
            "christoffersen_p_value": [0.34, 0.21, 0.41, 0.22, 0.39, 0.2],
            "status": ["ok", "ok", "ok", "ok", "ok", "ok"],
            "upstream_nu": [float("nan"), float("nan"), float("nan"), float("nan"), 12.0, 12.5],
            "upstream_sigma_next": [float("nan"), float("nan"), float("nan"), float("nan"), 0.02, 0.021],
            "upstream_garch_converged": [False, False, False, False, True, True],
        },
    ).to_csv(var_backtest_hist, index=False)

    diagnostics = build_diagnostics_table(
        risk_summary_path=tmp_path / "risk_summary.csv",
        scorecard_path=tmp_path / "scorecard.json",
        nvda_raw_path=tmp_path / "nvda.csv",
        market_raw_path=tmp_path / "market.csv",
        rolling_vol_path=tmp_path / "rolling_vol.csv",
        var_es_hist_path=var_es_hist,
        var_exceedances_hist_path=var_exceedances_hist,
        var_backtest_hist_path=var_backtest_hist,
        vol_window_days=252,
    )
    diagnostics_map = _as_map(diagnostics)

    assert diagnostics_map["hist_var_es_method_count"] == "3"
    assert diagnostics_map["parametric_var_es_present"] == "True"
    assert diagnostics_map["garch_var_es_present"] == "True"
    assert diagnostics_map["param_var_95"] == "-0.028"
    assert diagnostics_map["param_es_99"] == "-0.041"
    assert diagnostics_map["garch_var_es_nu_finite"] == "True"
    assert diagnostics_map["garch_var_es_nu_gt4"] == "True"
    assert diagnostics_map["garch_var_es_sigma_next_positive"] == "True"
    assert diagnostics_map["hist_exceed_method_count"] == "3"
    assert diagnostics_map["parametric_exceed_present"] == "True"
    assert diagnostics_map["garch_exceed_present"] == "True"
    assert diagnostics_map["hist_backtest_method_count"] == "3"
    assert diagnostics_map["parametric_backtest_present"] == "True"
    assert diagnostics_map["garch_backtest_present"] == "True"
    assert diagnostics_map["var_backtest_historical_rows_observed"] == "2"
    assert diagnostics_map["var_backtest_historical_fallback_events_observed"] == "0"
    assert diagnostics_map["var_backtest_historical_rows_consistent"] == "False"
    assert diagnostics_map["var_backtest_historical_fallback_events_consistent"] == "True"
    assert diagnostics_map["var_backtest_parametric_rows_observed"] == "2"
    assert diagnostics_map["var_backtest_parametric_fallback_events_observed"] == "0"
    assert diagnostics_map["var_backtest_parametric_rows_consistent"] == "False"
    assert diagnostics_map["var_backtest_parametric_fallback_events_consistent"] == "True"
    assert diagnostics_map["var_backtest_other_rows_observed"] == "2"
    assert diagnostics_map["var_backtest_other_fallback_events_observed"] == "0"
    assert diagnostics_map["var_backtest_other_rows_consistent"] == "False"
    assert diagnostics_map["var_backtest_other_fallback_events_consistent"] == "True"
    assert diagnostics_map["param_kupiec_lr_95"] == "0.6"
    assert diagnostics_map["param_kupiec_lr_99"] == "1.1"
    assert diagnostics_map["param_kupiec_pvalue_95"] == "0.2"
    assert diagnostics_map["garch_kupiec_lr_95"] == "0.7"
    assert diagnostics_map["garch_kupiec_lr_99"] == "1.0"
    assert diagnostics_map["garch_kupiec_pvalue_95"] == "0.19"
    assert diagnostics_map["garch_kupiec_pvalue_99"] == "0.1"
    assert diagnostics_map["param_christoffersen_lr_95"] == "0.45"
    assert diagnostics_map["param_christoffersen_lr_99"] == "0.85"
    assert diagnostics_map["param_christoffersen_pvalue_99"] == "0.22"
    assert diagnostics_map["garch_christoffersen_lr_95"] == "0.5"
    assert diagnostics_map["garch_christoffersen_lr_99"] == "0.8"
    assert diagnostics_map["garch_christoffersen_pvalue_95"] == "0.39"
    assert diagnostics_map["garch_christoffersen_pvalue_99"] == "0.2"
    assert diagnostics_map["garch_backtest_upstream_exists"] == "True"
    assert diagnostics_map["garch_backtest_upstream_converged_present"] == "True"
    assert diagnostics_map["garch_backtest_upstream_converged_true_count"] == "2"
    assert diagnostics_map["garch_backtest_upstream_nu_finite"] == "True"
    assert diagnostics_map["garch_backtest_upstream_sigma_positive"] == "True"
    assert diagnostics_map["historical_kupiec_reject_rate_5pct_observed"] == "0.0"
    assert diagnostics_map["historical_christoffersen_reject_rate_5pct_observed"] == "0.0"
    assert diagnostics_map["parametric_kupiec_reject_rate_5pct_observed"] == "0.0"
    assert diagnostics_map["parametric_christoffersen_reject_rate_5pct_observed"] == "0.0"
    assert diagnostics_map["garch_kupiec_reject_rate_5pct_observed"] == "0.0"
    assert diagnostics_map["garch_christoffersen_reject_rate_5pct_observed"] == "0.0"
    assert diagnostics_map["historical_kupiec_reject_rate_5pct_consistent"] == "False"
    assert diagnostics_map["historical_christoffersen_reject_rate_5pct_consistent"] == "False"
    assert diagnostics_map["parametric_kupiec_reject_rate_5pct_consistent"] == "False"
    assert diagnostics_map["parametric_christoffersen_reject_rate_5pct_consistent"] == "False"
    assert diagnostics_map["garch_kupiec_reject_rate_5pct_consistent"] == "False"
    assert diagnostics_map["garch_christoffersen_reject_rate_5pct_consistent"] == "False"


def test_build_diagnostics_table_flags_macro_frequency_issues(tmp_path: Path) -> None:
    """Flag macro frequency as invalid for non-month-end duplicated rows."""
    macro_raw = tmp_path / "macro_monthly.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-15", "2024-01-15"],
            "gdp_growth": [0.02, 0.021, 0.021],
            "inflation_yoy": [0.03, 0.029, 0.029],
            "policy_rate": [0.055, 0.054, 0.054],
        },
    ).to_csv(macro_raw, index=False)

    diagnostics = build_diagnostics_table(
        risk_summary_path=tmp_path / "risk_summary.csv",
        scorecard_path=tmp_path / "scorecard.json",
        nvda_raw_path=tmp_path / "nvda.csv",
        market_raw_path=tmp_path / "market.csv",
        rolling_vol_path=tmp_path / "rolling_vol.csv",
        var_es_hist_path=tmp_path / "var_es_hist.csv",
        var_exceedances_hist_path=tmp_path / "var_exceedances_hist.csv",
        vol_window_days=252,
        macro_raw_path=macro_raw,
    )
    diagnostics_map = _as_map(diagnostics)

    assert diagnostics_map["macro_data_exists"] == "True"
    assert diagnostics_map["macro_month_end_only"] == "False"
    assert diagnostics_map["macro_duplicate_month"] == "True"
    assert diagnostics_map["macro_frequency_ok"] == "False"

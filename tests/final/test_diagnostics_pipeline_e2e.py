"""End-to-end quality gate checks for diagnostics artifacts."""

from __future__ import annotations

import pandas as pd
import pytask
from _pytask.outcomes import ExitCode

from nvda_risk_project.config import DIAGNOSTICS_DATA, ROOT


def _as_map(frame: pd.DataFrame) -> dict[str, str]:
    """Convert diagnostics key-value rows into a dictionary."""
    return dict(zip(frame["check"], frame["value"], strict=True))


def test_diagnostics_pipeline_produces_reproducibility_quality_gates() -> None:
    """Run diagnostics tasks and assert reproducibility checks are present."""
    session = pytask.build(
        config=ROOT / "pyproject.toml",
        paths=[ROOT / "src" / "nvda_risk_project"],
        expression="diagnostics",
        force=True,
    )
    assert session.exit_code == ExitCode.OK

    assert DIAGNOSTICS_DATA.exists()
    diagnostics = pd.read_csv(DIAGNOSTICS_DATA)
    diagnostics_map = _as_map(diagnostics)

    assert diagnostics_map["provenance_exists"] == "True"
    assert diagnostics_map["provenance_panel_sha_match"] == "True"
    assert diagnostics_map["provenance_nvda_raw_sha_match"] == "True"
    assert diagnostics_map["provenance_market_raw_sha_match"] == "True"
    assert diagnostics_map["panel_daily_exists"] == "True"
    assert diagnostics_map["panel_daily_required_cols_present"] == "True"
    assert diagnostics_map["panel_daily_date_monotonic"] == "True"
    assert diagnostics_map["panel_daily_duplicate_dates"] == "0"
    assert diagnostics_map["sample_target_start"] == "2020-01-01"
    assert diagnostics_map["sample_target_end"] == "2024-12-31"
    assert diagnostics_map["nvda_window_matches_target"] == "True"
    assert diagnostics_map["market_window_matches_target"] == "True"
    assert diagnostics_map["panel_window_matches_target"] == "True"
    assert diagnostics_map["scorecard_sample_target_start_recorded"] == "2020-01-01"
    assert diagnostics_map["scorecard_sample_target_end_recorded"] == "2024-12-31"
    assert diagnostics_map["scorecard_window_matches_target_recorded"] == "true"
    assert diagnostics_map["scorecard_window_matches_target_consistent"] == "True"
    assert diagnostics_map["scorecard_status_recorded"] == "ok"
    assert diagnostics_map["scorecard_status_expected"] == "ok"
    assert diagnostics_map["scorecard_status_consistent"] == "True"
    assert diagnostics_map["scorecard_status_reason_recorded"] == "none"
    assert diagnostics_map["scorecard_status_reason_expected"] == "none"
    assert diagnostics_map["scorecard_status_reason_consistent"] == "True"
    assert diagnostics_map["systematic_metrics_present"] == "True"
    assert diagnostics_map["systematic_beta_consistent"] == "True"
    assert diagnostics_map["systematic_alpha_consistent"] == "True"
    assert diagnostics_map["systematic_beta_rolling_60m_latest_consistent"] == "True"
    assert diagnostics_map["systematic_beta_rolling_60m_valid_points_consistent"] == "True"
    assert diagnostics_map["systematic_beta_rolling_60m_fallback_consistent"] == "True"
    assert diagnostics_map["systematic_beta_rolling_252m_latest_consistent"] == "True"
    assert diagnostics_map["systematic_beta_rolling_252m_valid_points_consistent"] == "True"
    assert diagnostics_map["systematic_beta_rolling_252m_fallback_consistent"] == "True"
    assert diagnostics_map["data_source"] not in {"", "unknown"}
    assert diagnostics_map["placeholder_source_detected"] == "False"
    assert diagnostics_map["parametric_var_es_present"] == "True"
    assert diagnostics_map["garch_var_es_present"] == "True"
    assert diagnostics_map["parametric_exceed_present"] == "True"
    assert diagnostics_map["garch_exceed_present"] == "True"
    assert diagnostics_map["parametric_backtest_present"] == "True"
    assert diagnostics_map["garch_backtest_present"] == "True"
    assert diagnostics_map["garch_backtest_upstream_exists"] == "True"
    assert diagnostics_map["garch_backtest_upstream_converged_present"] == "True"
    assert int(diagnostics_map["garch_backtest_upstream_converged_true_count"]) >= 2
    assert diagnostics_map["garch_backtest_upstream_nu_finite"] == "True"
    assert diagnostics_map["garch_backtest_upstream_sigma_positive"] == "True"
    assert diagnostics_map["garch_var_es_nu_finite"] == "True"
    assert diagnostics_map["garch_var_es_nu_gt4"] == "True"
    assert diagnostics_map["garch_var_es_sigma_next_positive"] == "True"
    assert diagnostics_map["param_var_95"] != "nan"
    assert diagnostics_map["param_es_99"] != "nan"
    assert diagnostics_map["hist_kupiec_lr_95"] != "nan"
    assert diagnostics_map["param_kupiec_pvalue_95"] != "nan"
    assert diagnostics_map["garch_kupiec_pvalue_95"] != "nan"
    assert diagnostics_map["garch_christoffersen_pvalue_99"] != "nan"
    assert diagnostics_map["param_christoffersen_pvalue_99"] != "nan"
    assert int(diagnostics_map["var_backtest_historical_rows_recorded"]) >= 2
    assert int(diagnostics_map["var_backtest_parametric_rows_recorded"]) >= 2
    assert int(diagnostics_map["var_backtest_historical_rows_observed"]) >= 2
    assert int(diagnostics_map["var_backtest_parametric_rows_observed"]) >= 2
    assert int(diagnostics_map["var_backtest_garch_rows_recorded"]) >= 2
    assert int(diagnostics_map["var_backtest_garch_rows_observed"]) >= 2
    assert int(diagnostics_map["var_backtest_historical_fallback_events_recorded"]) >= 0
    assert int(diagnostics_map["var_backtest_parametric_fallback_events_recorded"]) >= 0
    assert int(diagnostics_map["var_backtest_historical_fallback_events_observed"]) >= 0
    assert int(diagnostics_map["var_backtest_parametric_fallback_events_observed"]) >= 0
    assert int(diagnostics_map["var_backtest_garch_fallback_events_recorded"]) >= 0
    assert int(diagnostics_map["var_backtest_garch_status_ok_count_recorded"]) >= 0
    assert int(diagnostics_map["var_backtest_garch_fallback_events_observed"]) >= 0
    assert int(diagnostics_map["var_backtest_garch_status_ok_count_observed"]) >= 0
    assert diagnostics_map["var_backtest_historical_rows_consistent"] == "True"
    assert diagnostics_map["var_backtest_historical_fallback_events_consistent"] == "True"
    assert diagnostics_map["var_backtest_parametric_rows_consistent"] == "True"
    assert diagnostics_map["var_backtest_parametric_fallback_events_consistent"] == "True"
    assert diagnostics_map["var_backtest_garch_rows_consistent"] == "True"
    assert diagnostics_map["var_backtest_garch_fallback_events_consistent"] == "True"
    assert diagnostics_map["var_backtest_garch_status_ok_count_consistent"] == "True"
    assert diagnostics_map["quality_gates_fail_count"] == "0"
    assert diagnostics_map["quality_gates_unknown_count"] == "0"
    assert diagnostics_map["quality_gates_failed_checks"] == "none"
    assert diagnostics_map["quality_gates_all_pass"] == "True"
    assert diagnostics_map["historical_kupiec_reject_rate_5pct_recorded"] != "nan"
    assert diagnostics_map["historical_christoffersen_reject_rate_5pct_recorded"] != "nan"
    assert diagnostics_map["parametric_kupiec_reject_rate_5pct_recorded"] != "nan"
    assert diagnostics_map["parametric_christoffersen_reject_rate_5pct_recorded"] != "nan"
    assert diagnostics_map["historical_kupiec_reject_rate_5pct_observed"] != "nan"
    assert diagnostics_map["historical_christoffersen_reject_rate_5pct_observed"] != "nan"
    assert diagnostics_map["parametric_kupiec_reject_rate_5pct_observed"] != "nan"
    assert diagnostics_map["parametric_christoffersen_reject_rate_5pct_observed"] != "nan"
    assert diagnostics_map["garch_kupiec_reject_rate_5pct_recorded"] != "nan"
    assert diagnostics_map["garch_christoffersen_reject_rate_5pct_recorded"] != "nan"
    assert diagnostics_map["historical_kupiec_reject_rate_5pct_consistent"] == "True"
    assert diagnostics_map["historical_christoffersen_reject_rate_5pct_consistent"] == "True"
    assert diagnostics_map["parametric_kupiec_reject_rate_5pct_consistent"] == "True"
    assert diagnostics_map["parametric_christoffersen_reject_rate_5pct_consistent"] == "True"
    assert diagnostics_map["garch_kupiec_reject_rate_5pct_observed"] != "nan"
    assert diagnostics_map["garch_christoffersen_reject_rate_5pct_observed"] != "nan"
    assert diagnostics_map["garch_kupiec_reject_rate_5pct_consistent"] == "True"
    assert diagnostics_map["garch_christoffersen_reject_rate_5pct_consistent"] == "True"
    assert diagnostics_map["macro_frequency_ok"] == "True"
    assert int(diagnostics_map["fallback_reason_unknown_count"]) >= 0
    assert int(diagnostics_map["hist_backtest_alpha_count"]) >= 2

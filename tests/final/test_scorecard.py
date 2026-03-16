"""Tests for scorecard fallback accounting."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from nvda_risk_project.final.task_scorecard import task_create_scorecard


def _risk_summary(path: Path) -> None:
    pd.DataFrame(
        {
            "risk_dimension": ["market", "liquidity"],
            "metric": ["volatility_annualized", "amihud_illiq_mean"],
            "value": [0.2, 1e-9],
        },
    ).to_csv(path, index=False)


def test_task_create_scorecard_records_systematic_metrics_from_summary(tmp_path: Path) -> None:
    risk_summary = tmp_path / "risk_summary.csv"
    rolling_vol = tmp_path / "rolling_vol.csv"
    var_backtest_hist = tmp_path / "var_backtest_hist.csv"
    panel_daily = tmp_path / "panel_daily.parquet"
    scorecard = tmp_path / "scorecard.json"
    pd.DataFrame(
        {
            "risk_dimension": ["market", "systematic", "systematic", "systematic", "systematic"],
            "metric": [
                "volatility_annualized",
                "beta",
                "alpha",
                "beta_rolling_60m_valid_points",
                "beta_rolling_60m_fallback",
            ],
            "value": [0.2, 1.3, 0.01, 12.0, 0.0],
        },
    ).to_csv(risk_summary, index=False)
    pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=3, freq="B"), "rolling_vol": [0.2, 0.21, 0.22]},
    ).to_csv(rolling_vol, index=False)
    pd.DataFrame({"alpha": [0.95], "status": ["ok"]}).to_csv(var_backtest_hist, index=False)
    pd.DataFrame({"date": ["2020-01-01", "2024-12-31"], "adj_close": [100.0, 200.0]}).to_parquet(
        panel_daily,
        index=False,
    )

    task_create_scorecard(
        risk_summary=risk_summary,
        rolling_vol=rolling_vol,
        var_backtest_hist=var_backtest_hist,
        panel_daily=panel_daily,
        produces=scorecard,
    )
    payload = json.loads(scorecard.read_text(encoding="utf-8"))

    assert payload["systematic_beta"] == 1.3
    assert payload["systematic_alpha"] == 0.01
    assert payload["systematic_beta_rolling_60m_valid_points"] == 12
    assert payload["systematic_beta_rolling_60m_fallback"] is False
    assert payload["systematic_beta_rolling_252m_valid_points"] == 0
    assert payload["systematic_beta_rolling_252m_fallback"] is True


def test_task_create_scorecard_marks_fallback_when_no_valid_rolling_vol(tmp_path: Path) -> None:
    risk_summary = tmp_path / "risk_summary.csv"
    rolling_vol = tmp_path / "rolling_vol.csv"
    var_backtest_hist = tmp_path / "var_backtest_hist.csv"
    scorecard = tmp_path / "scorecard.json"
    _risk_summary(risk_summary)
    pd.DataFrame(
        {"date": pd.date_range("2024-01-02", periods=3, freq="B"), "rolling_vol": [None, None, None]},
    ).to_csv(rolling_vol, index=False)
    pd.DataFrame(
        {"alpha": [0.95], "status": ["ok"]},
    ).to_csv(var_backtest_hist, index=False)

    task_create_scorecard(
        risk_summary=risk_summary,
        rolling_vol=rolling_vol,
        var_backtest_hist=var_backtest_hist,
        panel_daily=tmp_path / "missing_panel_daily.parquet",
        produces=scorecard,
    )
    payload = json.loads(scorecard.read_text(encoding="utf-8"))

    assert payload["rolling_vol_valid_points"] == 0
    assert payload["rolling_vol_fallback"] is True
    assert payload["var_backtest_fallback_events"] == 0
    assert payload["var_backtest_fallback_reason_short_sample_count"] == 0
    assert payload["var_backtest_fallback_reason_nan_var_count"] == 0
    assert payload["var_backtest_fallback_reason_invalid_input_count"] == 0
    assert payload["var_backtest_fallback_reason_exception_count"] == 0
    assert payload["var_backtest_fallback_reason_upstream_fallback_count"] == 0
    assert payload["var_backtest_fallback_reason_unknown_count"] == 0
    assert payload["var_backtest_historical_rows"] == 1
    assert payload["var_backtest_historical_fallback_events"] == 0
    assert payload["var_backtest_parametric_rows"] == 0
    assert payload["var_backtest_parametric_fallback_events"] == 0
    assert payload["var_backtest_other_rows"] == 0
    assert payload["var_backtest_other_fallback_events"] == 0
    assert payload["var_backtest_garch_rows"] == 0
    assert payload["var_backtest_garch_fallback_events"] == 0
    assert payload["var_backtest_garch_status_ok_count"] == 0
    assert math.isnan(payload["historical_kupiec_reject_rate_5pct"])
    assert math.isnan(payload["historical_christoffersen_reject_rate_5pct"])
    assert math.isnan(payload["parametric_kupiec_reject_rate_5pct"])
    assert math.isnan(payload["parametric_christoffersen_reject_rate_5pct"])
    assert math.isnan(payload["garch_kupiec_reject_rate_5pct"])
    assert math.isnan(payload["garch_christoffersen_reject_rate_5pct"])
    assert payload["sample_target_start"] == "2020-01-01"
    assert payload["sample_target_end"] == "2024-12-31"
    assert payload["window_matches_target"] == "unknown"
    assert payload["status"] == "fail"
    assert payload["status_reason"] == "window_not_target"
    assert payload["fallback_events"] == 1


def test_task_create_scorecard_no_fallback_with_valid_rolling_vol(tmp_path: Path) -> None:
    risk_summary = tmp_path / "risk_summary.csv"
    rolling_vol = tmp_path / "rolling_vol.csv"
    var_backtest_hist = tmp_path / "var_backtest_hist.csv"
    scorecard = tmp_path / "scorecard.json"
    _risk_summary(risk_summary)
    pd.DataFrame(
        {"date": pd.date_range("2024-01-02", periods=3, freq="B"), "rolling_vol": [None, 0.2, 0.21]},
    ).to_csv(rolling_vol, index=False)
    pd.DataFrame(
        {"alpha": [0.95, 0.99], "status": ["ok", "ok"]},
    ).to_csv(var_backtest_hist, index=False)

    task_create_scorecard(
        risk_summary=risk_summary,
        rolling_vol=rolling_vol,
        var_backtest_hist=var_backtest_hist,
        panel_daily=tmp_path / "missing_panel_daily.parquet",
        produces=scorecard,
    )
    payload = json.loads(scorecard.read_text(encoding="utf-8"))

    assert payload["rolling_vol_valid_points"] == 2
    assert payload["rolling_vol_fallback"] is False
    assert payload["var_backtest_fallback_events"] == 0
    assert payload["var_backtest_fallback_reason_short_sample_count"] == 0
    assert payload["var_backtest_fallback_reason_unknown_count"] == 0
    assert payload["var_backtest_historical_rows"] == 2
    assert payload["var_backtest_historical_fallback_events"] == 0
    assert payload["var_backtest_parametric_rows"] == 0
    assert payload["var_backtest_parametric_fallback_events"] == 0
    assert payload["var_backtest_garch_rows"] == 0
    assert payload["var_backtest_garch_fallback_events"] == 0
    assert payload["var_backtest_garch_status_ok_count"] == 0
    assert math.isnan(payload["historical_kupiec_reject_rate_5pct"])
    assert math.isnan(payload["historical_christoffersen_reject_rate_5pct"])
    assert math.isnan(payload["garch_kupiec_reject_rate_5pct"])
    assert math.isnan(payload["garch_christoffersen_reject_rate_5pct"])
    assert payload["window_matches_target"] == "unknown"
    assert payload["status"] == "fail"
    assert payload["status_reason"] == "window_not_target"
    assert payload["fallback_events"] == 0


def test_task_create_scorecard_counts_backtest_fallback_events(tmp_path: Path) -> None:
    risk_summary = tmp_path / "risk_summary.csv"
    rolling_vol = tmp_path / "rolling_vol.csv"
    var_backtest_hist = tmp_path / "var_backtest_hist.csv"
    scorecard = tmp_path / "scorecard.json"
    _risk_summary(risk_summary)
    pd.DataFrame(
        {"date": pd.date_range("2024-01-02", periods=3, freq="B"), "rolling_vol": [0.2, 0.21, 0.22]},
    ).to_csv(rolling_vol, index=False)
    pd.DataFrame(
        {"alpha": [0.95, 0.99], "status": ["fallback", "ok"]},
    ).to_csv(var_backtest_hist, index=False)

    task_create_scorecard(
        risk_summary=risk_summary,
        rolling_vol=rolling_vol,
        var_backtest_hist=var_backtest_hist,
        panel_daily=tmp_path / "missing_panel_daily.parquet",
        produces=scorecard,
    )
    payload = json.loads(scorecard.read_text(encoding="utf-8"))

    assert payload["rolling_vol_fallback"] is False
    assert payload["var_backtest_fallback_events"] == 1
    assert payload["var_backtest_fallback_reason_unknown_count"] == 1
    assert payload["var_backtest_historical_rows"] == 2
    assert payload["var_backtest_historical_fallback_events"] == 1
    assert payload["var_backtest_parametric_rows"] == 0
    assert payload["var_backtest_parametric_fallback_events"] == 0
    assert payload["var_backtest_garch_rows"] == 0
    assert payload["var_backtest_garch_fallback_events"] == 0
    assert payload["var_backtest_garch_status_ok_count"] == 0
    assert math.isnan(payload["historical_kupiec_reject_rate_5pct"])
    assert math.isnan(payload["parametric_kupiec_reject_rate_5pct"])
    assert math.isnan(payload["garch_kupiec_reject_rate_5pct"])
    assert math.isnan(payload["garch_christoffersen_reject_rate_5pct"])
    assert payload["status"] == "fail"
    assert payload["status_reason"] == "window_not_target"
    assert payload["fallback_events"] == 1


def test_task_create_scorecard_tracks_backtest_fallback_reasons(tmp_path: Path) -> None:
    risk_summary = tmp_path / "risk_summary.csv"
    rolling_vol = tmp_path / "rolling_vol.csv"
    var_backtest_hist = tmp_path / "var_backtest_hist.csv"
    scorecard = tmp_path / "scorecard.json"
    _risk_summary(risk_summary)
    pd.DataFrame(
        {"date": pd.date_range("2024-01-02", periods=3, freq="B"), "rolling_vol": [0.2, 0.21, 0.22]},
    ).to_csv(rolling_vol, index=False)
    pd.DataFrame(
        {
            "alpha": [0.95, 0.99, 0.975],
            "status": ["fallback", "fallback", "fallback"],
            "fallback_reason": ["short_sample", "upstream_invalid_input", "exception"],
        },
    ).to_csv(var_backtest_hist, index=False)

    task_create_scorecard(
        risk_summary=risk_summary,
        rolling_vol=rolling_vol,
        var_backtest_hist=var_backtest_hist,
        panel_daily=tmp_path / "missing_panel_daily.parquet",
        produces=scorecard,
    )
    payload = json.loads(scorecard.read_text(encoding="utf-8"))

    assert payload["var_backtest_fallback_events"] == 3
    assert payload["var_backtest_fallback_reason_short_sample_count"] == 1
    assert payload["var_backtest_fallback_reason_upstream_fallback_count"] == 1
    assert payload["var_backtest_fallback_reason_exception_count"] == 1
    assert payload["var_backtest_fallback_reason_unknown_count"] == 0
    assert payload["var_backtest_historical_rows"] == 3
    assert payload["var_backtest_historical_fallback_events"] == 3
    assert payload["var_backtest_parametric_rows"] == 0
    assert payload["var_backtest_parametric_fallback_events"] == 0
    assert payload["var_backtest_garch_rows"] == 0
    assert payload["var_backtest_garch_fallback_events"] == 0
    assert payload["var_backtest_garch_status_ok_count"] == 0
    assert math.isnan(payload["historical_christoffersen_reject_rate_5pct"])
    assert math.isnan(payload["garch_kupiec_reject_rate_5pct"])
    assert math.isnan(payload["garch_christoffersen_reject_rate_5pct"])
    assert payload["status"] == "fail"
    assert payload["status_reason"] == "window_not_target"


def test_task_create_scorecard_tracks_method_level_backtest_quality(tmp_path: Path) -> None:
    risk_summary = tmp_path / "risk_summary.csv"
    rolling_vol = tmp_path / "rolling_vol.csv"
    var_backtest_hist = tmp_path / "var_backtest_hist.csv"
    scorecard = tmp_path / "scorecard.json"
    _risk_summary(risk_summary)
    pd.DataFrame(
        {"date": pd.date_range("2024-01-02", periods=3, freq="B"), "rolling_vol": [0.2, 0.21, 0.22]},
    ).to_csv(rolling_vol, index=False)
    pd.DataFrame(
        {
            "method": [
                "historical",
                "historical",
                "parametric_normal",
                "garch_t",
                "garch_t",
                "other_model",
            ],
            "alpha": [0.95, 0.99, 0.95, 0.95, 0.99, 0.95],
            "status": ["ok", "fallback", "fallback", "ok", "fallback", "ok"],
            "fallback_reason": [
                "none",
                "short_sample",
                "invalid_input",
                "none",
                "exception",
                "none",
            ],
            "kupiec_reject_5pct": [True, False, True, False, False, False],
            "christoffersen_reject_5pct": [False, False, True, False, False, False],
        },
    ).to_csv(var_backtest_hist, index=False)

    task_create_scorecard(
        risk_summary=risk_summary,
        rolling_vol=rolling_vol,
        var_backtest_hist=var_backtest_hist,
        panel_daily=tmp_path / "missing_panel_daily.parquet",
        produces=scorecard,
    )
    payload = json.loads(scorecard.read_text(encoding="utf-8"))

    assert payload["var_backtest_historical_rows"] == 2
    assert payload["var_backtest_historical_fallback_events"] == 1
    assert payload["var_backtest_parametric_rows"] == 1
    assert payload["var_backtest_parametric_fallback_events"] == 1
    assert payload["var_backtest_other_rows"] == 3
    assert payload["var_backtest_other_fallback_events"] == 1
    assert payload["var_backtest_garch_rows"] == 2
    assert payload["var_backtest_garch_fallback_events"] == 1
    assert payload["var_backtest_garch_status_ok_count"] == 1
    assert payload["historical_kupiec_reject_rate_5pct"] == 1.0
    assert payload["historical_christoffersen_reject_rate_5pct"] == 0.0
    assert math.isnan(payload["parametric_kupiec_reject_rate_5pct"])
    assert math.isnan(payload["parametric_christoffersen_reject_rate_5pct"])
    assert payload["garch_kupiec_reject_rate_5pct"] == 0.0
    assert payload["garch_christoffersen_reject_rate_5pct"] == 0.0
    assert payload["status"] == "fail"
    assert payload["status_reason"] == "window_not_target"


def test_task_create_scorecard_records_window_match_from_panel(tmp_path: Path) -> None:
    risk_summary = tmp_path / "risk_summary.csv"
    rolling_vol = tmp_path / "rolling_vol.csv"
    var_backtest_hist = tmp_path / "var_backtest_hist.csv"
    panel_daily = tmp_path / "panel_daily.parquet"
    scorecard = tmp_path / "scorecard.json"
    _risk_summary(risk_summary)
    pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=3, freq="B"), "rolling_vol": [0.2, 0.21, 0.22]},
    ).to_csv(rolling_vol, index=False)
    pd.DataFrame(
        {"alpha": [0.95], "status": ["ok"]},
    ).to_csv(var_backtest_hist, index=False)
    pd.DataFrame(
        {"date": ["2020-01-01", "2024-12-31"], "adj_close": [100.0, 200.0]},
    ).to_parquet(panel_daily, index=False)

    task_create_scorecard(
        risk_summary=risk_summary,
        rolling_vol=rolling_vol,
        var_backtest_hist=var_backtest_hist,
        panel_daily=panel_daily,
        produces=scorecard,
    )
    payload = json.loads(scorecard.read_text(encoding="utf-8"))

    assert payload["sample_target_start"] == "2020-01-01"
    assert payload["sample_target_end"] == "2024-12-31"
    assert payload["window_matches_target"] == "True"
    assert payload["status"] == "ok"
    assert payload["status_reason"] == "none"


def test_task_create_scorecard_marks_warn_when_window_ok_but_backtest_has_fallbacks(tmp_path: Path) -> None:
    risk_summary = tmp_path / "risk_summary.csv"
    rolling_vol = tmp_path / "rolling_vol.csv"
    var_backtest_hist = tmp_path / "var_backtest_hist.csv"
    panel_daily = tmp_path / "panel_daily.parquet"
    scorecard = tmp_path / "scorecard.json"
    _risk_summary(risk_summary)
    pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=3, freq="B"), "rolling_vol": [0.2, 0.21, 0.22]},
    ).to_csv(rolling_vol, index=False)
    pd.DataFrame(
        {
            "method": ["historical", "historical"],
            "alpha": [0.95, 0.99],
            "status": ["ok", "fallback"],
            "fallback_reason": ["none", "short_sample"],
            "kupiec_reject_5pct": [False, False],
            "christoffersen_reject_5pct": [False, False],
        },
    ).to_csv(var_backtest_hist, index=False)
    pd.DataFrame(
        {"date": ["2020-01-01", "2024-12-31"], "adj_close": [100.0, 200.0]},
    ).to_parquet(panel_daily, index=False)

    task_create_scorecard(
        risk_summary=risk_summary,
        rolling_vol=rolling_vol,
        var_backtest_hist=var_backtest_hist,
        panel_daily=panel_daily,
        produces=scorecard,
    )
    payload = json.loads(scorecard.read_text(encoding="utf-8"))

    assert payload["window_matches_target"] == "True"
    assert payload["var_backtest_fallback_events"] == 1
    assert payload["status"] == "warn"
    assert payload["status_reason"] == "backtest_fallbacks"


def test_task_create_scorecard_marks_fail_for_rolling_vol_fallback_even_when_window_matches(tmp_path: Path) -> None:
    risk_summary = tmp_path / "risk_summary.csv"
    rolling_vol = tmp_path / "rolling_vol.csv"
    var_backtest_hist = tmp_path / "var_backtest_hist.csv"
    panel_daily = tmp_path / "panel_daily.parquet"
    scorecard = tmp_path / "scorecard.json"
    _risk_summary(risk_summary)
    pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=3, freq="B"), "rolling_vol": [None, None, None]},
    ).to_csv(rolling_vol, index=False)
    pd.DataFrame(
        {"alpha": [0.95], "status": ["ok"]},
    ).to_csv(var_backtest_hist, index=False)
    pd.DataFrame(
        {"date": ["2020-01-01", "2024-12-31"], "adj_close": [100.0, 200.0]},
    ).to_parquet(panel_daily, index=False)

    task_create_scorecard(
        risk_summary=risk_summary,
        rolling_vol=rolling_vol,
        var_backtest_hist=var_backtest_hist,
        panel_daily=panel_daily,
        produces=scorecard,
    )
    payload = json.loads(scorecard.read_text(encoding="utf-8"))

    assert payload["window_matches_target"] == "True"
    assert payload["rolling_vol_fallback"] is True
    assert payload["status"] == "fail"
    assert payload["status_reason"] == "rolling_vol_fallback"

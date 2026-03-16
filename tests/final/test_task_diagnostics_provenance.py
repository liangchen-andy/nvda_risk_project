"""Task-level tests for provenance checks in diagnostics artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from nvda_risk_project.final.task_diagnostics import task_create_diagnostics_data


def _as_map(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path)
    return dict(zip(frame["check"], frame["value"], strict=True))


def _write_common_inputs(tmp_path: Path) -> dict[str, Path]:
    risk_summary = tmp_path / "risk_summary.csv"
    scorecard = tmp_path / "scorecard.json"
    nvda_raw = tmp_path / "nvda_daily.csv"
    market_raw = tmp_path / "sp500_daily.csv"
    rolling_vol = tmp_path / "rolling_vol.csv"
    var_es_hist = tmp_path / "var_es_hist.csv"
    var_exceedances_hist = tmp_path / "var_exceedances_hist.csv"
    diagnostics = tmp_path / "diagnostics.csv"

    pd.DataFrame({"risk_dimension": ["market"], "metric": ["vol"], "value": [0.2]}).to_csv(
        risk_summary,
        index=False,
    )
    scorecard.write_text(json.dumps({"metric_count": 1, "fallback_events": 0}), encoding="utf-8")
    pd.DataFrame({"date": ["2024-01-01"], "close": [100], "volume": [1000], "data_source": ["snapshot"]}).to_csv(
        nvda_raw,
        index=False,
    )
    pd.DataFrame({"date": ["2024-01-01"], "close": [4000], "volume": [2000], "data_source": ["online"]}).to_csv(
        market_raw,
        index=False,
    )
    pd.DataFrame({"date": ["2024-01-01"], "rolling_vol": [0.2]}).to_csv(rolling_vol, index=False)
    pd.DataFrame(
        {
            "alpha": [0.95],
            "var": [-0.03],
            "es": [-0.035],
            "status": ["ok"],
        },
    ).to_csv(var_es_hist, index=False)
    pd.DataFrame(
        {
            "alpha": [0.95],
            "exceedance_count": [5],
            "exceedance_rate": [0.05],
            "status": ["ok"],
        },
    ).to_csv(var_exceedances_hist, index=False)

    return {
        "risk_summary": risk_summary,
        "scorecard": scorecard,
        "nvda_raw": nvda_raw,
        "market_raw": market_raw,
        "rolling_vol": rolling_vol,
        "var_es_hist": var_es_hist,
        "var_exceedances_hist": var_exceedances_hist,
        "diagnostics": diagnostics,
    }


def test_task_create_diagnostics_data_includes_provenance_checks(tmp_path: Path) -> None:
    inputs = _write_common_inputs(tmp_path)
    panel_daily = tmp_path / "panel_daily.parquet"
    provenance = tmp_path / "data_provenance.json"
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
                        "path": str(inputs["nvda_raw"]),
                        "sha256": hashlib.sha256(inputs["nvda_raw"].read_bytes()).hexdigest(),
                    },
                    "sp500_raw": {
                        "path": str(inputs["market_raw"]),
                        "sha256": hashlib.sha256(inputs["market_raw"].read_bytes()).hexdigest(),
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
        risk_summary=inputs["risk_summary"],
        scorecard=inputs["scorecard"],
        nvda_raw=inputs["nvda_raw"],
        market_raw=inputs["market_raw"],
        rolling_vol=inputs["rolling_vol"],
        var_es_hist=inputs["var_es_hist"],
        var_exceedances_hist=inputs["var_exceedances_hist"],
        var_backtest_hist=tmp_path / "var_backtest_hist.csv",
        provenance=provenance,
        produces=inputs["diagnostics"],
    )
    diagnostics_map = _as_map(inputs["diagnostics"])

    assert diagnostics_map["provenance_exists"] == "True"
    assert diagnostics_map["provenance_parse_error"] == "False"
    assert diagnostics_map["provenance_ticker_match"] == "True"
    assert diagnostics_map["provenance_sample_match"] == "True"
    assert diagnostics_map["provenance_panel_sha_match"] == "True"
    assert diagnostics_map["provenance_nvda_raw_sha_match"] == "True"
    assert diagnostics_map["provenance_market_raw_sha_match"] == "True"
    assert diagnostics_map["provenance_source_match"] == "True"


def test_task_create_diagnostics_data_defaults_when_provenance_missing(tmp_path: Path) -> None:
    inputs = _write_common_inputs(tmp_path)

    task_create_diagnostics_data(
        risk_summary=inputs["risk_summary"],
        scorecard=inputs["scorecard"],
        nvda_raw=inputs["nvda_raw"],
        market_raw=inputs["market_raw"],
        rolling_vol=inputs["rolling_vol"],
        var_es_hist=inputs["var_es_hist"],
        var_exceedances_hist=inputs["var_exceedances_hist"],
        var_backtest_hist=tmp_path / "missing_var_backtest_hist.csv",
        provenance=tmp_path / "missing_data_provenance.json",
        produces=inputs["diagnostics"],
    )
    diagnostics_map = _as_map(inputs["diagnostics"])

    assert diagnostics_map["provenance_exists"] == "False"
    assert diagnostics_map["provenance_parse_error"] == "False"
    assert diagnostics_map["provenance_ticker_match"] == "unknown"
    assert diagnostics_map["provenance_sample_match"] == "unknown"
    assert diagnostics_map["provenance_panel_sha_match"] == "unknown"
    assert diagnostics_map["provenance_nvda_raw_sha_match"] == "unknown"
    assert diagnostics_map["provenance_market_raw_sha_match"] == "unknown"
    assert diagnostics_map["provenance_source_match"] == "unknown"

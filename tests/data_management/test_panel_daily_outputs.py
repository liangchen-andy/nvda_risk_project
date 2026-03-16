"""Tests for daily panel and provenance artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from nvda_risk_project.data_management.task_clean import (
    task_build_daily_panel,
    task_write_data_provenance,
)


def test_daily_panel_is_non_empty_with_expected_columns_and_dates(tmp_path: Path) -> None:
    nvda_raw = tmp_path / "nvda_daily.csv"
    sp500_raw = tmp_path / "sp500_daily.csv"
    panel_daily = tmp_path / "panel_daily.parquet"

    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    pd.DataFrame(
        {
            "date": dates,
            "symbol": ["NVDA"] * 4,
            "close": [100.0, 101.0, 99.0, 102.0],
            "volume": [1_000_000, 1_100_000, 1_200_000, 1_300_000],
            "data_source": ["snapshot"] * 4,
        },
    ).to_csv(nvda_raw, index=False)
    pd.DataFrame(
        {
            "date": dates,
            "symbol": ["^GSPC"] * 4,
            "close": [4000.0, 4010.0, 3995.0, 4020.0],
            "volume": [3_000_000_000] * 4,
            "data_source": ["snapshot"] * 4,
        },
    ).to_csv(sp500_raw, index=False)

    task_build_daily_panel(
        nvda_raw_data=nvda_raw,
        sp500_raw_data=sp500_raw,
        produces=panel_daily,
    )

    panel = pd.read_parquet(panel_daily)
    expected_columns = {
        "date",
        "adj_close",
        "close",
        "volume",
        "ret",
        "logret",
        "market_ret",
    }
    assert not panel.empty
    assert expected_columns == set(panel.columns)
    assert panel["date"].is_monotonic_increasing
    assert panel["date"].is_unique


def test_data_provenance_contains_required_fields(tmp_path: Path) -> None:
    panel_daily = tmp_path / "panel_daily.parquet"
    nvda_raw = tmp_path / "nvda_daily.csv"
    sp500_raw = tmp_path / "sp500_daily.csv"
    provenance = tmp_path / "data_provenance.json"

    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    pd.DataFrame(
        {
            "date": dates,
            "adj_close": [100.0, 101.0, 102.0],
            "close": [100.0, 101.0, 102.0],
            "volume": [1_000_000, 1_100_000, 1_200_000],
            "ret": [0.0, 0.01, 0.0099],
            "logret": [0.0, 0.00995, 0.00985],
            "market_ret": [0.0, 0.002, 0.0015],
        },
    ).to_parquet(panel_daily, index=False)
    pd.DataFrame({"date": dates, "data_source": ["snapshot"] * 3}).to_csv(nvda_raw, index=False)
    pd.DataFrame({"date": dates, "data_source": ["online"] * 3}).to_csv(sp500_raw, index=False)

    task_write_data_provenance(
        panel_daily_data=panel_daily,
        nvda_raw_data=nvda_raw,
        sp500_raw_data=sp500_raw,
        produces=provenance,
    )

    payload = json.loads(provenance.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(panel_daily.read_bytes()).hexdigest()
    expected_nvda_hash = hashlib.sha256(nvda_raw.read_bytes()).hexdigest()
    expected_sp500_hash = hashlib.sha256(sp500_raw.read_bytes()).hexdigest()

    assert payload["tickers"] == ["NVDA", "^GSPC"]
    assert payload["sample_start"] == "2024-01-02"
    assert payload["sample_end"] == "2024-01-04"
    assert payload["source"]["NVDA"] == "snapshot"
    assert payload["source"]["^GSPC"] == "online"
    assert payload["artifacts"]["nvda_raw"]["sha256"] == expected_nvda_hash
    assert payload["artifacts"]["sp500_raw"]["sha256"] == expected_sp500_hash
    assert payload["artifacts"]["panel_daily"]["sha256"] == expected_hash
    assert "generated_at_utc" in payload

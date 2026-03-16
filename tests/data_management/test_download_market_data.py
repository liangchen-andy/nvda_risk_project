"""Tests for market data download and caching behavior."""

from pathlib import Path

import pandas as pd

from nvda_risk_project.data_management.task_download import (
    _download_and_cache_market_data,
    _format_market_data,
)


def test_format_market_data_standardizes_schema() -> None:
    raw = pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-01", periods=3, freq="B"),
            "Close": [100.0, 101.0, 102.0],
            "Volume": [1_000_000, 1_200_000, 1_100_000],
        },
    )

    out = _format_market_data(raw, symbol="NVDA")
    assert list(out.columns) == ["date", "symbol", "close", "volume"]
    assert (out["symbol"] == "NVDA").all()
    assert len(out) == 3


def test_download_and_cache_uses_downloader_when_no_cache(tmp_path: Path) -> None:
    target = tmp_path / "nvda_daily.csv"

    def fake_downloader(symbol: str, start_date: str) -> pd.DataFrame:
        assert symbol == "NVDA"
        assert start_date
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-02", periods=2, freq="B"),
                "symbol": [symbol, symbol],
                "close": [100.0, 101.0],
                "volume": [1_000_000, 1_500_000],
            },
        )

    _download_and_cache_market_data(
        symbol="NVDA",
        produces=target,
        seed=11,
        downloader=fake_downloader,
        start_date="2024-01-02",
        end_date="2024-01-03",
    )
    out = pd.read_csv(target)
    assert {"date", "symbol", "close", "volume", "data_source"} == set(out.columns)
    assert out.shape[0] == 2
    assert set(out["data_source"]) == {"online"}


def test_download_and_cache_falls_back_to_placeholder(tmp_path: Path) -> None:
    target = tmp_path / "sp500_daily.csv"

    def failing_downloader(symbol: str, start_date: str) -> pd.DataFrame:
        raise RuntimeError(f"failed for {symbol} from {start_date}")

    _download_and_cache_market_data(
        symbol="^GSPC",
        produces=target,
        seed=22,
        downloader=failing_downloader,
    )

    out = pd.read_csv(target)
    assert {"date", "symbol", "close", "volume", "data_source"} == set(out.columns)
    assert not out.empty
    assert set(out["data_source"]) == {"placeholder"}


def test_download_and_cache_uses_snapshot_before_online(tmp_path: Path) -> None:
    target = tmp_path / "nvda_daily.csv"
    snapshot = tmp_path / "nvda_snapshot.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=2, freq="B"),
            "symbol": ["NVDA", "NVDA"],
            "close": [100.0, 101.0],
            "volume": [1_000_000, 1_100_000],
        },
    ).to_csv(snapshot, index=False)

    def failing_downloader(symbol: str, start_date: str) -> pd.DataFrame:
        raise AssertionError(f"downloader should not run for {symbol} from {start_date}")

    _download_and_cache_market_data(
        symbol="NVDA",
        produces=target,
        seed=11,
        downloader=failing_downloader,
        snapshot_path=snapshot,
        start_date="2024-01-02",
        end_date="2024-01-03",
    )

    out = pd.read_csv(target)
    assert out.shape[0] == 2
    assert set(out["data_source"]) == {"snapshot"}


def test_download_and_cache_prefers_existing_cache(tmp_path: Path) -> None:
    target = tmp_path / "nvda_daily.csv"
    snapshot = tmp_path / "nvda_snapshot.csv"

    cached = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=2, freq="B"),
            "symbol": ["NVDA", "NVDA"],
            "close": [300.0, 301.0],
            "volume": [2_000_000, 2_100_000],
            "data_source": ["snapshot", "snapshot"],
        },
    )
    cached.to_csv(target, index=False)

    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=2, freq="B"),
            "symbol": ["NVDA", "NVDA"],
            "close": [100.0, 101.0],
            "volume": [1_000_000, 1_100_000],
        },
    ).to_csv(snapshot, index=False)

    def failing_downloader(symbol: str, start_date: str) -> pd.DataFrame:
        raise AssertionError(f"downloader should not run for {symbol} from {start_date}")

    _download_and_cache_market_data(
        symbol="NVDA",
        produces=target,
        seed=11,
        downloader=failing_downloader,
        snapshot_path=snapshot,
        start_date="2024-01-02",
        end_date="2024-01-03",
    )

    out = pd.read_csv(target)
    assert out["close"].tolist() == [300.0, 301.0]


def test_download_and_cache_rebuilds_when_snapshot_window_is_too_short(tmp_path: Path) -> None:
    target = tmp_path / "nvda_daily.csv"
    snapshot = tmp_path / "nvda_snapshot.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "symbol": ["NVDA"],
            "close": [100.0],
            "volume": [1_000_000],
        },
    ).to_csv(snapshot, index=False)

    def fake_downloader(symbol: str, start_date: str) -> pd.DataFrame:
        assert symbol == "NVDA"
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-02", periods=2, freq="B"),
                "symbol": [symbol, symbol],
                "close": [300.0, 301.0],
                "volume": [2_000_000, 2_100_000],
            },
        )

    _download_and_cache_market_data(
        symbol="NVDA",
        produces=target,
        seed=11,
        downloader=fake_downloader,
        snapshot_path=snapshot,
        start_date="2024-01-02",
        end_date="2024-01-03",
    )

    out = pd.read_csv(target)
    assert out["close"].tolist() == [300.0, 301.0]
    assert set(out["data_source"]) == {"online"}


def test_download_and_cache_ignores_corrupt_existing_cache(tmp_path: Path) -> None:
    target = tmp_path / "sp500_daily.csv"
    target.write_text("", encoding="utf-8")

    def fake_downloader(symbol: str, start_date: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-02", periods=2, freq="B"),
                "symbol": [symbol, symbol],
                "close": [400.0, 401.0],
                "volume": [2_000_000, 2_100_000],
            },
        )

    _download_and_cache_market_data(
        symbol="^GSPC",
        produces=target,
        seed=22,
        downloader=fake_downloader,
        start_date="2024-01-02",
        end_date="2024-01-03",
    )

    out = pd.read_csv(target)
    assert out["close"].tolist() == [400.0, 401.0]
    assert set(out["data_source"]) == {"online"}


def test_download_and_cache_skips_corrupt_snapshot(tmp_path: Path) -> None:
    target = tmp_path / "nvda_daily.csv"
    snapshot = tmp_path / "nvda_snapshot.csv"
    snapshot.write_text("", encoding="utf-8")

    def fake_downloader(symbol: str, start_date: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-02", periods=2, freq="B"),
                "symbol": [symbol, symbol],
                "close": [500.0, 501.0],
                "volume": [3_000_000, 3_100_000],
            },
        )

    _download_and_cache_market_data(
        symbol="NVDA",
        produces=target,
        seed=11,
        downloader=fake_downloader,
        snapshot_path=snapshot,
        start_date="2024-01-02",
        end_date="2024-01-03",
    )

    out = pd.read_csv(target)
    assert out["close"].tolist() == [500.0, 501.0]
    assert set(out["data_source"]) == {"online"}

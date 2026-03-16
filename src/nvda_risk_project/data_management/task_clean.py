"""Tasks for cleaning raw data artifacts."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nvda_risk_project.config import (
    ASSET_TICKER,
    BENCHMARK_TICKER,
    DATA_PROVENANCE,
    INTERIM_DATA,
    PANEL_DAILY_DATA,
    RAW_DATA,
)


def _clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Create cleaned price features used by downstream tasks."""
    out = df.sort_values("date").copy()
    out["return"] = out["close"].pct_change().fillna(0.0)
    out["dollar_volume"] = out["close"] * out["volume"]
    out["amihud_illiq"] = np.abs(out["return"]) / out["dollar_volume"].clip(lower=1.0)
    return out


def _build_panel_daily(nvda: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Build a daily NVDA-market panel with return columns."""
    asset = nvda.copy()
    benchmark = market.copy()
    asset["date"] = pd.to_datetime(asset["date"], errors="coerce")
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce")
    if "adj_close" not in asset.columns:
        asset["adj_close"] = asset["close"]

    panel = (
        asset[["date", "adj_close", "close", "volume"]]
        .merge(
            benchmark[["date", "close"]].rename(columns={"close": "market_close"}),
            on="date",
            how="inner",
        )
        .dropna(subset=["date", "adj_close", "close", "volume", "market_close"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
    )
    panel["ret"] = panel["adj_close"].pct_change().fillna(0.0)
    panel["logret"] = np.log(panel["adj_close"]).diff().fillna(0.0)
    panel["market_ret"] = panel["market_close"].pct_change().fillna(0.0)
    return panel[["date", "adj_close", "close", "volume", "ret", "logret", "market_ret"]]


def _file_sha256(path: Path) -> str:
    """Return the SHA256 checksum for a file path."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _get_data_source(raw: pd.DataFrame) -> str:
    """Extract a normalized data-source label from raw data."""
    if "data_source" not in raw.columns:
        return "cache"
    sources = sorted({str(value) for value in raw["data_source"].dropna().tolist()})
    return "+".join(sources) if sources else "cache"


def task_clean_nvda_data(
    raw_data: Path = RAW_DATA / "nvda_daily.csv",
    produces: Path = INTERIM_DATA / "nvda_daily.csv",
) -> None:
    """Clean daily NVDA raw data."""
    produces.parent.mkdir(parents=True, exist_ok=True)
    _clean_prices(pd.read_csv(raw_data, parse_dates=["date"])).to_csv(produces, index=False)


def task_clean_sp500_data(
    raw_data: Path = RAW_DATA / "sp500_daily.csv",
    produces: Path = INTERIM_DATA / "sp500_daily.csv",
) -> None:
    """Clean daily market benchmark raw data."""
    produces.parent.mkdir(parents=True, exist_ok=True)
    _clean_prices(pd.read_csv(raw_data, parse_dates=["date"])).to_csv(produces, index=False)


def task_clean_macro_data(
    raw_data: Path = RAW_DATA / "macro_monthly.csv",
    produces: Path = INTERIM_DATA / "macro_monthly.csv",
) -> None:
    """Clean monthly macro raw data."""
    produces.parent.mkdir(parents=True, exist_ok=True)
    macro = pd.read_csv(raw_data, parse_dates=["date"]).sort_values("date")
    macro = macro.ffill()
    macro.to_csv(produces, index=False)


def task_build_daily_panel(
    nvda_raw_data: Path = RAW_DATA / "nvda_daily.csv",
    sp500_raw_data: Path = RAW_DATA / "sp500_daily.csv",
    produces: Path = PANEL_DAILY_DATA,
) -> None:
    """Create a daily panel for risk modeling from aligned NVDA and market prices."""
    produces.parent.mkdir(parents=True, exist_ok=True)
    nvda = pd.read_csv(nvda_raw_data, parse_dates=["date"])
    sp500 = pd.read_csv(sp500_raw_data, parse_dates=["date"])
    _build_panel_daily(nvda=nvda, market=sp500).to_parquet(produces, index=False)


def task_write_data_provenance(
    panel_daily_data: Path = PANEL_DAILY_DATA,
    nvda_raw_data: Path = RAW_DATA / "nvda_daily.csv",
    sp500_raw_data: Path = RAW_DATA / "sp500_daily.csv",
    produces: Path = DATA_PROVENANCE,
) -> None:
    """Write provenance metadata for the daily panel artifact."""
    produces.parent.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(panel_daily_data)
    nvda_raw = pd.read_csv(nvda_raw_data)
    sp500_raw = pd.read_csv(sp500_raw_data)
    panel_dates = pd.to_datetime(panel["date"], errors="coerce").dropna()

    payload = {
        "tickers": [ASSET_TICKER, BENCHMARK_TICKER],
        "sample_start": panel_dates.min().date().isoformat(),
        "sample_end": panel_dates.max().date().isoformat(),
        "source": {
            ASSET_TICKER: _get_data_source(nvda_raw),
            BENCHMARK_TICKER: _get_data_source(sp500_raw),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifacts": {
            "nvda_raw": {
                "path": str(nvda_raw_data),
                "rows": int(nvda_raw.shape[0]),
                "sha256": _file_sha256(nvda_raw_data),
            },
            "sp500_raw": {
                "path": str(sp500_raw_data),
                "rows": int(sp500_raw.shape[0]),
                "sha256": _file_sha256(sp500_raw_data),
            },
            "panel_daily": {
                "path": str(panel_daily_data),
                "rows": int(panel.shape[0]),
                "sha256": _file_sha256(panel_daily_data),
            },
        },
    }

    produces.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


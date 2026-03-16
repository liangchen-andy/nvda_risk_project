"""Tasks for downloading and preparing raw inputs for the risk pipeline."""

from pathlib import Path
from collections.abc import Callable
import warnings

import numpy as np
import pandas as pd

from nvda_risk_project.config import (
    ASSET_TICKER,
    BENCHMARK_TICKER,
    RAW_DATA,
    SAMPLE_END_DATE,
    SAMPLE_START_DATE,
    SNAPSHOT_MARKET_DAILY,
    SNAPSHOT_NVDA_DAILY,
)

_REQUIRED_PRICE_COLUMNS: tuple[str, ...] = ("date", "symbol", "close", "volume")
_DOWNLOAD_START_DATE = SAMPLE_START_DATE
_DOWNLOAD_END_DATE = SAMPLE_END_DATE
_SOURCE_COLUMN = "data_source"


def _slice_to_window(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Return rows in a closed date window."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    out = out.loc[(out["date"] >= start_ts) & (out["date"] <= end_ts)]
    return out.sort_values("date")


def _covers_window(df: pd.DataFrame, start_date: str, end_date: str) -> bool:
    """Check whether observed data spans the full requested business-date window."""
    if df.empty or "date" not in df.columns:
        return False
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        return False
    expected = pd.date_range(start_date, end_date, freq="B")
    if expected.empty:
        return False
    return bool(dates.min() <= expected.min() and dates.max() >= expected.max())


def _build_price_data(symbol: str, seed: int, start_date: str, end_date: str) -> pd.DataFrame:
    """Create deterministic placeholder daily prices for a symbol."""
    rng = np.random.default_rng(seed=seed)
    dates = pd.date_range(start_date, end_date, freq="B")
    returns = rng.normal(loc=0.0007, scale=0.02, size=len(dates))
    close = 100 * np.cumprod(1 + returns)
    volume = rng.integers(1_000_000, 7_000_000, size=len(dates))
    out = pd.DataFrame({"date": dates, "symbol": symbol, "close": close, "volume": volume})
    return out


def _format_market_data(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize downloaded market data into the project schema."""
    rename_map = {
        "Date": "date",
        "Close": "close",
        "Adj Close": "close",
        "Volume": "volume",
    }
    out = raw.rename(columns=rename_map)
    missing = {col for col in ("date", "close", "volume") if col not in out.columns}
    if missing:
        msg = f"Downloaded data for {symbol} missing columns: {sorted(missing)}"
        raise ValueError(msg)

    out = out[["date", "close", "volume"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out["symbol"] = symbol
    out = out.dropna(subset=["date", "close", "volume"])
    out = out[out["volume"] >= 0].sort_values("date")
    out = out[list(_REQUIRED_PRICE_COLUMNS)]
    if out.empty:
        msg = f"Downloaded data for {symbol} is empty after cleaning."
        raise ValueError(msg)
    return out


def _download_from_yfinance(symbol: str, start_date: str) -> pd.DataFrame:
    """Download market data from yfinance and format it."""
    import yfinance as yf  # Imported lazily so tests can mock without dependency.

    raw = yf.download(
        symbol,
        start=start_date,
        progress=False,
        auto_adjust=False,
        actions=False,
    )
    if raw.empty:
        msg = f"yfinance returned empty data for {symbol}."
        raise ValueError(msg)
    return _format_market_data(raw.reset_index(), symbol=symbol)


def _validate_market_data(df: pd.DataFrame, symbol: str) -> None:
    """Validate required schema and non-empty content for market data."""
    if not set(_REQUIRED_PRICE_COLUMNS).issubset(df.columns):
        msg = f"{symbol} data does not contain required columns {_REQUIRED_PRICE_COLUMNS}."
        raise ValueError(msg)
    if df.empty:
        msg = f"{symbol} data is empty."
        raise ValueError(msg)


def _mark_data_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Attach a source marker to market data rows."""
    out = df.copy()
    out[_SOURCE_COLUMN] = source
    return out


def _download_and_cache_market_data(
    symbol: str,
    produces: Path,
    seed: int,
    downloader: Callable[[str, str], pd.DataFrame] = _download_from_yfinance,
    snapshot_path: Path | None = None,
    start_date: str = _DOWNLOAD_START_DATE,
    end_date: str = _DOWNLOAD_END_DATE,
) -> None:
    """Build raw market data using cache, snapshot, download, then fallback."""
    produces.parent.mkdir(parents=True, exist_ok=True)
    if produces.exists():
        try:
            cached = pd.read_csv(produces, parse_dates=["date"])
            _validate_market_data(cached, symbol=symbol)
            cached = _slice_to_window(cached, start_date=start_date, end_date=end_date)
            if _covers_window(cached, start_date=start_date, end_date=end_date):
                cached.to_csv(produces, index=False)
                return
        except Exception:
            # Corrupt or partially-written cache should not break reproducibility.
            pass

    if snapshot_path is not None and snapshot_path.exists():
        try:
            snapshot = pd.read_csv(snapshot_path, parse_dates=["date"])
            _validate_market_data(snapshot, symbol=symbol)
            snapshot = _slice_to_window(snapshot, start_date=start_date, end_date=end_date)
            if _covers_window(snapshot, start_date=start_date, end_date=end_date):
                _mark_data_source(snapshot, source="snapshot").to_csv(produces, index=False)
                return
        except Exception:
            # If snapshot is malformed, continue to online/fallback path.
            pass

    try:
        downloaded = downloader(symbol, start_date)
        downloaded = _slice_to_window(downloaded, start_date=start_date, end_date=end_date)
        _validate_market_data(downloaded, symbol=symbol)
        if not _covers_window(downloaded, start_date=start_date, end_date=end_date):
            msg = (
                f"Downloaded data for {symbol} does not cover requested window "
                f"{start_date}..{end_date}."
            )
            raise ValueError(msg)
        downloaded = _mark_data_source(downloaded, source="online")
    except Exception as exc:
        warnings.warn(
            f"Could not download {symbol} from yfinance ({exc}). Falling back to placeholder data.",
            stacklevel=2,
        )
        downloaded = _build_price_data(
            symbol=symbol,
            seed=seed,
            start_date=start_date,
            end_date=end_date,
        )
        downloaded = _mark_data_source(downloaded, source="placeholder")

    downloaded.to_csv(produces, index=False)


def _build_macro_data() -> pd.DataFrame:
    """Create deterministic macro input data at monthly frequency."""
    dates = pd.date_range(SAMPLE_START_DATE, SAMPLE_END_DATE, freq="ME")
    return pd.DataFrame(
        {
            "date": dates,
            "gdp_growth": np.linspace(0.01, 0.03, len(dates)),
            "inflation_yoy": np.linspace(0.04, 0.02, len(dates)),
            "policy_rate": np.linspace(0.055, 0.045, len(dates)),
        },
    )


def task_download_nvda_raw(
    produces: Path = RAW_DATA / "nvda_daily.csv",
) -> None:
    """Download and cache NVDA daily data."""
    _download_and_cache_market_data(
        symbol=ASSET_TICKER,
        produces=produces,
        seed=11,
        snapshot_path=SNAPSHOT_NVDA_DAILY,
    )


def task_download_sp500_raw(
    produces: Path = RAW_DATA / "sp500_daily.csv",
) -> None:
    """Download and cache S&P 500 daily data."""
    _download_and_cache_market_data(
        symbol=BENCHMARK_TICKER,
        produces=produces,
        seed=22,
        snapshot_path=SNAPSHOT_MARKET_DAILY,
    )


def task_download_macro_raw(
    produces: Path = RAW_DATA / "macro_monthly.csv",
) -> None:
    """Create deterministic monthly macro inputs."""
    produces.parent.mkdir(parents=True, exist_ok=True)
    _build_macro_data().to_csv(produces, index=False)


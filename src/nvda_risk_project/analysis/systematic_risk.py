"""Systematic risk helpers with static and rolling CAPM beta outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

_REQUIRED_COLUMNS = {"nvda_return", "market_return"}
_ROLLING_WINDOWS_MONTHS = (60, 252)


def _clean_return_pairs(panel: pd.DataFrame) -> pd.DataFrame:
    """Return paired non-missing asset/market returns."""
    missing = _REQUIRED_COLUMNS.difference(panel.columns)
    if missing:
        msg = f"panel is missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    out = pd.DataFrame(
        {
            "asset": pd.to_numeric(panel["nvda_return"], errors="coerce"),
            "market": pd.to_numeric(panel["market_return"], errors="coerce"),
        },
    ).dropna()
    return out


def compute_rolling_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    *,
    window: int,
) -> pd.Series:
    """Compute rolling CAPM beta as cov(asset, market) / var(market)."""
    if window <= 1:
        msg = "window must be greater than 1."
        raise ValueError(msg)

    paired = pd.DataFrame(
        {
            "asset": pd.to_numeric(asset_returns, errors="coerce"),
            "market": pd.to_numeric(market_returns, errors="coerce"),
        },
    ).dropna()
    if paired.empty:
        return pd.Series(dtype=float, name=f"beta_rolling_{window}m")

    covariance = paired["asset"].rolling(window=window, min_periods=window).cov(paired["market"])
    variance = paired["market"].rolling(window=window, min_periods=window).var()
    beta = covariance / variance.replace(0.0, np.nan)
    beta.name = f"beta_rolling_{window}m"
    return beta


def estimate_systematic_risk(panel: pd.DataFrame) -> pd.DataFrame:
    """Return static CAPM beta/alpha and rolling-beta diagnostics (60m/252m)."""
    paired = _clean_return_pairs(panel)

    beta = 0.0
    alpha = float(paired["asset"].mean()) if not paired.empty else 0.0
    if paired.shape[0] >= 2:
        asset = paired["asset"].to_numpy(dtype=float)
        market = paired["market"].to_numpy(dtype=float)
        market_var = float(np.var(market, ddof=1))
        if np.isfinite(market_var) and market_var > 0:
            beta = float(np.cov(asset, market, ddof=1)[0, 1] / market_var)
            alpha = float(asset.mean() - beta * market.mean())

    metrics: list[tuple[str, float]] = [("beta", beta), ("alpha", alpha)]
    for window in _ROLLING_WINDOWS_MONTHS:
        rolling = compute_rolling_beta(paired["asset"], paired["market"], window=window)
        valid_points = int(rolling.notna().sum())
        latest_beta = float(rolling.dropna().iloc[-1]) if valid_points > 0 else 0.0
        metrics.extend(
            [
                (f"beta_rolling_{window}m_latest", latest_beta),
                (f"beta_rolling_{window}m_valid_points", float(valid_points)),
                (f"beta_rolling_{window}m_fallback", float(valid_points == 0)),
            ],
        )

    return pd.DataFrame(metrics, columns=["metric", "value"])

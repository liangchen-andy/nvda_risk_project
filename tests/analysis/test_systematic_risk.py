"""Tests for systematic risk helper functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nvda_risk_project.analysis.systematic_risk import compute_rolling_beta
from nvda_risk_project.analysis.systematic_risk import estimate_systematic_risk


def _as_map(frame: pd.DataFrame) -> dict[str, float]:
    """Convert metric-value rows to a dictionary."""
    return dict(zip(frame["metric"], frame["value"], strict=True))


def test_compute_rolling_beta_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="greater than 1"):
        compute_rolling_beta(
            asset_returns=pd.Series([0.01, 0.02]),
            market_returns=pd.Series([0.01, 0.02]),
            window=1,
        )


def test_estimate_systematic_risk_returns_static_and_rolling_metrics() -> None:
    months = pd.date_range("2019-01-31", periods=72, freq="ME")
    market = pd.Series(np.linspace(-0.03, 0.04, len(months)))
    asset = 0.002 + 1.5 * market
    panel = pd.DataFrame(
        {
            "month": months,
            "nvda_return": asset,
            "market_return": market,
        },
    )

    out = estimate_systematic_risk(panel)
    result = _as_map(out)

    assert set(result).issuperset(
        {
            "beta",
            "alpha",
            "beta_rolling_60m_latest",
            "beta_rolling_60m_valid_points",
            "beta_rolling_60m_fallback",
            "beta_rolling_252m_latest",
            "beta_rolling_252m_valid_points",
            "beta_rolling_252m_fallback",
        },
    )
    assert result["beta"] == pytest.approx(1.5, rel=1e-6)
    assert result["alpha"] == pytest.approx(0.002, rel=1e-6)
    assert result["beta_rolling_60m_latest"] == pytest.approx(1.5, rel=1e-6)
    assert result["beta_rolling_60m_valid_points"] == 13.0
    assert result["beta_rolling_60m_fallback"] == 0.0
    assert result["beta_rolling_252m_latest"] == 0.0
    assert result["beta_rolling_252m_valid_points"] == 0.0
    assert result["beta_rolling_252m_fallback"] == 1.0


def test_estimate_systematic_risk_falls_back_when_market_variance_is_zero() -> None:
    panel = pd.DataFrame(
        {
            "month": pd.date_range("2020-01-31", periods=6, freq="ME"),
            "nvda_return": [0.01, 0.02, 0.03, -0.01, 0.0, 0.01],
            "market_return": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        },
    )

    out = estimate_systematic_risk(panel)
    result = _as_map(out)

    assert result["beta"] == 0.0
    assert result["alpha"] == pytest.approx(np.mean(panel["nvda_return"]))
    assert result["beta_rolling_60m_valid_points"] == 0.0
    assert result["beta_rolling_60m_fallback"] == 1.0

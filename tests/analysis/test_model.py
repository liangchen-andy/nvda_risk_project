"""Tests for risk module output shapes in the Milestone 1 skeleton."""

import numpy as np
import pandas as pd
import pytest

from nvda_risk_project.analysis.drawdown_risk import estimate_drawdown_risk
from nvda_risk_project.analysis.liquidity_risk import estimate_liquidity_risk
from nvda_risk_project.analysis.macro_risk import estimate_macro_risk
from nvda_risk_project.analysis.market_risk import estimate_market_risk
from nvda_risk_project.analysis.systematic_risk import estimate_systematic_risk


@pytest.fixture
def panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month": pd.date_range("2024-01-31", periods=8, freq="ME"),
            "nvda_return": np.array([0.01, 0.03, -0.02, 0.015, -0.01, 0.02, 0.01, -0.005]),
            "market_return": np.array([0.005, 0.02, -0.01, 0.01, -0.008, 0.015, 0.007, -0.003]),
            "amihud_illiq": np.linspace(1e-9, 3e-9, 8),
            "dollar_volume": np.linspace(2_000_000, 3_000_000, 8),
            "gdp_growth": np.linspace(0.02, 0.025, 8),
            "inflation_yoy": np.linspace(0.035, 0.03, 8),
            "policy_rate": np.linspace(0.055, 0.05, 8),
        },
    )


@pytest.mark.parametrize(
    ("fn", "kwargs"),
    [
        (estimate_market_risk, {"var_level": 0.05}),
        (estimate_liquidity_risk, {}),
        (estimate_drawdown_risk, {}),
        (estimate_systematic_risk, {}),
        (estimate_macro_risk, {}),
    ],
)
def test_risk_module_output_schema(
    panel: pd.DataFrame,
    fn: callable,
    kwargs: dict[str, float],
) -> None:
    result = fn(panel, **kwargs)
    assert {"metric", "value"} == set(result.columns)
    assert not result.empty
    assert result["value"].notna().all()


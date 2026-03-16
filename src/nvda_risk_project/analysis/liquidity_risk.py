"""Liquidity risk calculations for the NVDA risk pipeline."""

import pandas as pd


def estimate_liquidity_risk(panel: pd.DataFrame) -> pd.DataFrame:
    """Return summary liquidity proxy metrics."""
    return pd.DataFrame(
        {
            "metric": ["amihud_illiq_mean", "dollar_volume_median"],
            "value": [panel["amihud_illiq"].mean(), panel["dollar_volume"].median()],
        },
    )

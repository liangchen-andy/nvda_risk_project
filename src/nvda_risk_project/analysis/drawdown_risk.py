"""Drawdown risk calculations with magnitude and duration summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _drawdown_duration(drawdown: pd.Series) -> pd.Series:
    """Compute consecutive drawdown durations (in periods)."""
    in_drawdown = drawdown.lt(0).to_numpy(dtype=bool)
    duration = np.zeros(len(in_drawdown), dtype=int)
    streak = 0
    for idx, flag in enumerate(in_drawdown):
        if flag:
            streak += 1
        else:
            streak = 0
        duration[idx] = streak
    return pd.Series(duration, index=drawdown.index, dtype=int)


def estimate_drawdown_risk(panel: pd.DataFrame) -> pd.DataFrame:
    """Return drawdown magnitude and duration metrics from cumulative returns."""
    returns = pd.to_numeric(panel["nvda_return"], errors="coerce").fillna(0.0)
    cumulative = (1 + returns).cumprod()
    running_peak = cumulative.cummax()
    drawdown = cumulative / running_peak - 1
    duration = _drawdown_duration(drawdown)
    return pd.DataFrame(
        {
            "metric": [
                "max_drawdown",
                "latest_drawdown",
                "max_drawdown_duration",
                "latest_drawdown_duration",
            ],
            "value": [
                float(drawdown.min()),
                float(drawdown.iloc[-1]),
                float(duration.max()),
                float(duration.iloc[-1]),
            ],
        },
    )

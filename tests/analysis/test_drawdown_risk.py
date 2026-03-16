"""Tests for drawdown risk helper functions."""

from __future__ import annotations

import pandas as pd

from nvda_risk_project.analysis.drawdown_risk import estimate_drawdown_risk


def _as_map(frame: pd.DataFrame) -> dict[str, float]:
    return dict(zip(frame["metric"], frame["value"], strict=True))


def test_estimate_drawdown_risk_includes_duration_metrics() -> None:
    panel = pd.DataFrame(
        {
            "nvda_return": [0.10, -0.05, 0.06, -0.02, -0.02, 0.05],
        },
    )

    out = estimate_drawdown_risk(panel)
    metrics = _as_map(out)
    assert set(metrics) == {
        "max_drawdown",
        "latest_drawdown",
        "max_drawdown_duration",
        "latest_drawdown_duration",
    }
    assert metrics["max_drawdown_duration"] == 2.0
    assert metrics["latest_drawdown_duration"] == 0.0

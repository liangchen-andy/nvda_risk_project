"""End-to-end regression checks for rolling-beta artifacts."""

from __future__ import annotations

import pandas as pd
import pytask
from _pytask.outcomes import ExitCode

from nvda_risk_project.config import PROCESSED_DATA, ROOT, SUMMARY_FIG_BETA_ROLLING


def test_beta_pipeline_produces_expected_artifacts() -> None:
    session = pytask.build(
        config=ROOT / "pyproject.toml",
        paths=[ROOT / "src" / "nvda_risk_project"],
        expression="beta",
        force=True,
    )
    assert session.exit_code == ExitCode.OK

    panel_monthly_path = PROCESSED_DATA / "panel_monthly.csv"
    assert panel_monthly_path.exists()
    panel_monthly = pd.read_csv(panel_monthly_path)
    assert {"month", "nvda_return", "market_return"}.issubset(panel_monthly.columns)
    assert not panel_monthly.empty

    assert SUMMARY_FIG_BETA_ROLLING.exists()
    assert SUMMARY_FIG_BETA_ROLLING.stat().st_size > 0

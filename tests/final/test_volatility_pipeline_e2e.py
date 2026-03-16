"""End-to-end regression checks for volatility artifacts."""

from __future__ import annotations

import pandas as pd
import pytask
from _pytask.outcomes import ExitCode

from nvda_risk_project.config import ANALYSIS_OUTPUT, ROOT, SUMMARY_FIG_VOLATILITY


def test_volatility_pipeline_produces_expected_artifacts() -> None:
    session = pytask.build(
        config=ROOT / "pyproject.toml",
        paths=[ROOT / "src" / "nvda_risk_project"],
        expression="rolling or volatility",
        force=True,
    )
    assert session.exit_code == ExitCode.OK

    rolling_vol_path = ANALYSIS_OUTPUT / "rolling_vol.csv"
    assert rolling_vol_path.exists()
    rolling_vol = pd.read_csv(rolling_vol_path)
    assert {"date", "rolling_vol"} == set(rolling_vol.columns)
    assert not rolling_vol.empty

    assert SUMMARY_FIG_VOLATILITY.exists()
    assert SUMMARY_FIG_VOLATILITY.stat().st_size > 0

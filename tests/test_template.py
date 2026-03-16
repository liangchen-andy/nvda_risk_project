"""Integration tests for the Milestone 1 skeleton workflow."""

import pytask
from _pytask.outcomes import ExitCode

from nvda_risk_project.config import ROOT


def test_pytask_build_for_src_tasks_only() -> None:
    session = pytask.build(
        config=ROOT / "pyproject.toml",
        paths=[ROOT / "src" / "nvda_risk_project"],
        force=True,
    )
    assert session.exit_code == ExitCode.OK


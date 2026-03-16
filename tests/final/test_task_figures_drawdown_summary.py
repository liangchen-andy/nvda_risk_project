"""Tests for drawdown summary figure task wiring."""

from __future__ import annotations

from pathlib import Path

from nvda_risk_project.final.task_figures import _PNG_1X1, task_create_drawdown_summary_figure


def test_task_create_drawdown_summary_figure_copies_existing_figure(tmp_path: Path) -> None:
    figure = tmp_path / "drawdown_curve.png"
    output = tmp_path / "documents" / "public" / "fig_drawdown.png"
    figure.write_bytes(b"drawdown-figure")

    task_create_drawdown_summary_figure(figure=figure, produces=output)

    assert output.read_bytes() == b"drawdown-figure"


def test_task_create_drawdown_summary_figure_falls_back_when_source_missing(
    tmp_path: Path,
) -> None:
    figure = tmp_path / "missing" / "drawdown_curve.png"
    output = tmp_path / "documents" / "public" / "fig_drawdown.png"

    task_create_drawdown_summary_figure(figure=figure, produces=output)

    assert figure.read_bytes() == _PNG_1X1
    assert output.read_bytes() == _PNG_1X1

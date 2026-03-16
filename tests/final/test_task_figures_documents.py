"""Tests for document figure task wiring."""

from __future__ import annotations

from pathlib import Path

from nvda_risk_project.final.task_figures import _PNG_1X1, task_create_documents_figure


def test_task_create_documents_figure_copies_existing_figure(tmp_path: Path) -> None:
    figure = tmp_path / "market_risk_overview.png"
    output = tmp_path / "documents" / "public" / "figure.png"
    figure.write_bytes(b"real-figure")

    task_create_documents_figure(figure=figure, produces=output)

    assert output.read_bytes() == b"real-figure"


def test_task_create_documents_figure_falls_back_when_source_missing(tmp_path: Path) -> None:
    figure = tmp_path / "missing" / "market_risk_overview.png"
    output = tmp_path / "documents" / "public" / "figure.png"

    task_create_documents_figure(figure=figure, produces=output)

    assert figure.read_bytes() == _PNG_1X1
    assert output.read_bytes() == _PNG_1X1

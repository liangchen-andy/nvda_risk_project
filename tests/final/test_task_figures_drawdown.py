"""Tests for drawdown figure task wiring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nvda_risk_project.final.task_figures import _PNG_1X1, task_create_drawdown_figure


class _FakeFigure:
    def write_image(self, path: str) -> None:
        Path(path).write_bytes(b"drawdown-image")


def test_task_create_drawdown_figure_writes_image(tmp_path: Path, monkeypatch) -> None:
    panel_monthly = tmp_path / "panel_monthly.csv"
    output_png = tmp_path / "drawdown_curve.png"
    pd.DataFrame(
        {
            "month": pd.date_range("2024-01-31", periods=3, freq="ME"),
            "nvda_return": [0.05, -0.03, 0.02],
        },
    ).to_csv(panel_monthly, index=False)

    monkeypatch.setattr(
        "nvda_risk_project.final.task_figures.plot_drawdown_curve",
        lambda _: _FakeFigure(),
    )

    task_create_drawdown_figure(panel_monthly_data=panel_monthly, produces=output_png)
    assert output_png.read_bytes() == b"drawdown-image"


def test_task_create_drawdown_figure_falls_back_to_placeholder(tmp_path: Path) -> None:
    panel_monthly = tmp_path / "panel_monthly.csv"
    output_png = tmp_path / "drawdown_curve.png"
    pd.DataFrame({"month": pd.date_range("2024-01-31", periods=3, freq="ME")}).to_csv(
        panel_monthly,
        index=False,
    )

    task_create_drawdown_figure(panel_monthly_data=panel_monthly, produces=output_png)
    assert output_png.read_bytes() == _PNG_1X1

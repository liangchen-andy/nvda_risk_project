"""Tests for rolling beta figure task wiring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nvda_risk_project.final.task_figures import _PNG_1X1, task_create_beta_rolling_figure


class _FakeFigure:
    def write_image(self, path: str) -> None:
        Path(path).write_bytes(b"beta-rolling-image")


def test_task_create_beta_rolling_figure_writes_image(tmp_path: Path, monkeypatch) -> None:
    panel_monthly = tmp_path / "panel_monthly.csv"
    output_png = tmp_path / "fig_beta_rolling.png"
    pd.DataFrame(
        {
            "month": pd.date_range("2023-01-31", periods=15, freq="ME"),
            "nvda_return": [0.04, 0.02, -0.01, 0.03, 0.01] * 3,
            "market_return": [0.02, 0.01, -0.005, 0.015, 0.008] * 3,
        },
    ).to_csv(panel_monthly, index=False)

    monkeypatch.setattr(
        "nvda_risk_project.final.task_figures.plot_beta_rolling_curve",
        lambda _: _FakeFigure(),
    )

    task_create_beta_rolling_figure(panel_monthly_data=panel_monthly, produces=output_png)
    assert output_png.read_bytes() == b"beta-rolling-image"


def test_task_create_beta_rolling_figure_falls_back_to_placeholder(tmp_path: Path) -> None:
    panel_monthly = tmp_path / "panel_monthly.csv"
    output_png = tmp_path / "fig_beta_rolling.png"
    pd.DataFrame({"month": pd.date_range("2024-01-31", periods=3, freq="ME")}).to_csv(
        panel_monthly,
        index=False,
    )

    task_create_beta_rolling_figure(panel_monthly_data=panel_monthly, produces=output_png)
    assert output_png.read_bytes() == _PNG_1X1

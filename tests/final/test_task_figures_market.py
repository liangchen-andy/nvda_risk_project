"""Tests for market figure task wiring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nvda_risk_project.final.task_figures import _PNG_1X1, task_create_market_figure


class _FakeFigure:
    def write_image(self, path: str) -> None:
        Path(path).write_bytes(b"market-image")


def test_task_create_market_figure_writes_image(tmp_path: Path, monkeypatch) -> None:
    summary = tmp_path / "risk_summary.csv"
    output_png = tmp_path / "market_risk_overview.png"
    pd.DataFrame(
        {
            "risk_dimension": ["market", "liquidity"],
            "metric": ["volatility_annualized", "amihud_illiq_mean"],
            "value": [0.2, 1e-9],
        },
    ).to_csv(summary, index=False)

    monkeypatch.setattr(
        "nvda_risk_project.final.task_figures.plot_market_risk_overview",
        lambda _: _FakeFigure(),
    )

    task_create_market_figure(risk_summary=summary, produces=output_png)
    assert output_png.read_bytes() == b"market-image"


def test_task_create_market_figure_falls_back_to_placeholder(tmp_path: Path) -> None:
    summary = tmp_path / "risk_summary.csv"
    output_png = tmp_path / "market_risk_overview.png"
    pd.DataFrame({"risk_dimension": ["market"], "value": [0.2]}).to_csv(summary, index=False)

    task_create_market_figure(risk_summary=summary, produces=output_png)
    assert output_png.read_bytes() == _PNG_1X1

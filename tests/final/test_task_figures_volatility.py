"""Tests for volatility figure task wiring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nvda_risk_project.final.task_figures import _PNG_1X1, task_create_volatility_figure


class _FakeFigure:
    def write_image(self, path: str) -> None:
        Path(path).write_bytes(b"fake-image")


def test_task_create_volatility_figure_writes_image(tmp_path: Path, monkeypatch) -> None:
    rolling_vol = tmp_path / "rolling_vol.csv"
    output_png = tmp_path / "fig_volatility.png"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "rolling_vol": [0.2, 0.21, 0.22],
        },
    ).to_csv(rolling_vol, index=False)

    monkeypatch.setattr(
        "nvda_risk_project.final.task_figures.plot_volatility_series",
        lambda _: _FakeFigure(),
    )

    task_create_volatility_figure(rolling_vol_data=rolling_vol, produces=output_png)
    assert output_png.read_bytes() == b"fake-image"


def test_task_create_volatility_figure_falls_back_to_placeholder(tmp_path: Path) -> None:
    rolling_vol = tmp_path / "rolling_vol.csv"
    output_png = tmp_path / "fig_volatility.png"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "rolling_vol": [None, None, None],
        },
    ).to_csv(rolling_vol, index=False)

    task_create_volatility_figure(rolling_vol_data=rolling_vol, produces=output_png)
    assert output_png.read_bytes() == _PNG_1X1

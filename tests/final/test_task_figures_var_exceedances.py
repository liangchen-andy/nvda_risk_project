"""Tests for VaR exceedances figure task wiring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nvda_risk_project.final.task_figures import _PNG_1X1, task_create_var_exceedances_figure


class _FakeFigure:
    def write_image(self, path: str) -> None:
        Path(path).write_bytes(b"exceedances-image")


def test_task_create_var_exceedances_figure_writes_image(tmp_path: Path, monkeypatch) -> None:
    exceedances = tmp_path / "var_exceedances_hist.csv"
    output_png = tmp_path / "fig_var_exceedances.png"
    pd.DataFrame(
        {
            "method": ["historical", "parametric_normal"],
            "alpha": [0.95, 0.95],
            "exceedance_rate": [0.05, 0.04],
            "status": ["ok", "ok"],
        },
    ).to_csv(exceedances, index=False)

    monkeypatch.setattr(
        "nvda_risk_project.final.task_figures.plot_var_exceedances",
        lambda _: _FakeFigure(),
    )

    task_create_var_exceedances_figure(var_exceedances_data=exceedances, produces=output_png)
    assert output_png.read_bytes() == b"exceedances-image"


def test_task_create_var_exceedances_figure_falls_back_to_placeholder(tmp_path: Path) -> None:
    exceedances = tmp_path / "var_exceedances_hist.csv"
    output_png = tmp_path / "fig_var_exceedances.png"
    pd.DataFrame({"alpha": [0.95], "status": ["ok"]}).to_csv(exceedances, index=False)

    task_create_var_exceedances_figure(var_exceedances_data=exceedances, produces=output_png)
    assert output_png.read_bytes() == _PNG_1X1

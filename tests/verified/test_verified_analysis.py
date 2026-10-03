"""Regression checks for economically meaningful calculation and timing errors."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("verified_analysis", Path(__file__).resolve().parents[2] / "scripts/verified_analysis.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


def test_compounding_losing_round_trip():
    panel = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                          "ret": [.1, -.1], "market_ret": [.05, -.05], "dollar_volume": [100, 200]})
    monthly = workflow.monthly_panel(panel)
    assert monthly["ret"].iloc[0] == pytest.approx(-.01)
    assert monthly["market_ret"].iloc[0] == pytest.approx(-.0025)


def test_future_returns_cannot_change_prior_forecast():
    panel = pd.DataFrame({"date": pd.bdate_range("2023-01-02", periods=45),
                          "ret": np.random.default_rng(7).normal(0, .02, 45)})
    first = workflow.rolling_forecasts(panel, window=30)
    changed = panel.copy()
    changed.loc[30:, "ret"] = .9
    second = workflow.rolling_forecasts(changed, window=30)
    np.testing.assert_allclose(first.loc[first["date"] == panel["date"].iloc[30], ["var", "es"]],
                               second.loc[second["date"] == panel["date"].iloc[30], ["var", "es"]])
    assert (first["training_end"] < first["date"]).all()
    assert set(first["training_size"]) == {30}


def test_macro_does_not_use_future_or_stale_yields():
    panel = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-20"]), "ret": [0, 0, 0]})
    macro = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "dgs10": [4.]})
    result = workflow.align_yields(panel, macro)
    assert np.isnan(result["dgs10"].iloc[0])
    assert result["dgs10"].iloc[1] == 4.
    assert np.isnan(result["dgs10"].iloc[2])


def test_tampered_vendor_archive_is_rejected(tmp_path, monkeypatch):
    for file in workflow.INPUT.iterdir():
        (tmp_path / file.name).write_bytes(file.read_bytes())
    (tmp_path / "nvda.json").write_bytes(b"tampered")
    monkeypatch.setattr(workflow, "INPUT", tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        workflow.load_inputs()


def test_missing_exchange_session_and_missing_warmup_are_rejected():
    nvda, market, _, _ = workflow.load_inputs()
    with pytest.raises(ValueError, match="exchange sessions"):
        workflow.make_panel(nvda[nvda["date"] != pd.Timestamp("2020-01-03")], market)
    with pytest.raises(ValueError, match="first return"):
        workflow.make_panel(nvda[nvda["date"] >= workflow.START], market)


def test_duplicate_dates_and_nonfinite_forecast_history_are_rejected():
    panel = pd.DataFrame({"date": pd.bdate_range("2023-01-02", periods=40), "ret": np.zeros(40)})
    panel.loc[1, "date"] = panel.loc[0, "date"]
    with pytest.raises(ValueError, match="unique dates"):
        workflow.rolling_forecasts(panel, 30)
    panel["date"] = pd.bdate_range("2023-01-02", periods=40)
    panel.loc[2, "ret"] = np.nan
    with pytest.raises(ValueError, match="Finite returns"):
        workflow.rolling_forecasts(panel, 30)


def test_observed_build_has_consistent_forecast_dates(tmp_path):
    metrics = workflow.build(tmp_path)
    assert metrics["daily_observations"] == 1258
    assert metrics["monthly_observations"] == 60
    forecasts = pd.read_csv(tmp_path / "oos_forecasts.csv", parse_dates=["date", "training_start", "training_end"])
    assert forecasts["date"].nunique() == 1006
    assert set(forecasts.groupby(["method", "alpha"]).size()) == {1006}
    assert (forecasts["training_end"] < forecasts["date"]).all()
    assert json.loads((tmp_path / "validation.json").read_text())["all_pass"]
    assert (tmp_path / "risk_dashboard.png").stat().st_size > 10000

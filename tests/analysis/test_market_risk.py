"""Tests for market risk helper functions."""

import numpy as np
import pandas as pd
import pytest

from nvda_risk_project.analysis.market_risk import compute_rolling_volatility
from nvda_risk_project.analysis.market_risk import build_rolling_volatility_frame
from nvda_risk_project.analysis.market_risk import compute_historical_var_es
from nvda_risk_project.analysis.market_risk import compute_parametric_var_es
from nvda_risk_project.analysis.market_risk import compute_garch_t_var_es
from nvda_risk_project.analysis.market_risk import christoffersen_independence_test
from nvda_risk_project.analysis.market_risk import kupiec_pof_test


def test_compute_rolling_volatility_empty_series() -> None:
    out = compute_rolling_volatility(pd.Series(dtype=float), window=3)
    assert out.empty


def test_compute_rolling_volatility_short_sample_returns_nan() -> None:
    returns = pd.Series([0.01, -0.02], dtype=float)
    out = compute_rolling_volatility(returns, window=3)
    assert out.isna().all()


def test_compute_rolling_volatility_handles_nan_in_window() -> None:
    returns = pd.Series([0.01, np.nan, 0.03, 0.02], dtype=float)
    out = compute_rolling_volatility(returns, window=2, periods_per_year=252)
    assert pd.isna(out.iloc[0])
    assert pd.isna(out.iloc[1])
    assert pd.isna(out.iloc[2])
    assert out.iloc[3] > 0


def test_compute_rolling_volatility_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="window must be a positive integer"):
        compute_rolling_volatility(pd.Series([0.01, 0.02]), window=0)


def test_build_rolling_volatility_frame_returns_expected_schema() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-04"]),
            "ret": [0.01, 0.02, -0.01],
        },
    )
    out = build_rolling_volatility_frame(panel_daily=panel, window=2, periods_per_year=252)
    assert list(out.columns) == ["date", "rolling_vol"]
    assert out["date"].is_monotonic_increasing
    assert pd.isna(out.iloc[0]["rolling_vol"])
    assert out.iloc[1]["rolling_vol"] > 0


def test_compute_historical_var_es_rejects_invalid_confidence_level() -> None:
    with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
        compute_historical_var_es(pd.Series([0.01, -0.02]), confidence_level=1.0)


def test_compute_historical_var_es_rejects_empty_after_cleaning() -> None:
    with pytest.raises(ValueError, match="at least one valid observation"):
        compute_historical_var_es(pd.Series([np.nan, None]), confidence_level=0.95)


def test_compute_historical_var_es_handles_short_sample() -> None:
    var_95, es_95 = compute_historical_var_es(pd.Series([-0.03]), confidence_level=0.95)
    assert var_95 == pytest.approx(-0.03)
    assert es_95 == pytest.approx(-0.03)


def test_compute_historical_var_es_for_95_and_99() -> None:
    returns = pd.Series([-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03], dtype=float)
    var_95, es_95 = compute_historical_var_es(returns, confidence_level=0.95)
    var_99, es_99 = compute_historical_var_es(returns, confidence_level=0.99)

    assert var_95 <= 0
    assert es_95 <= var_95
    assert var_99 <= var_95
    assert es_99 <= var_99


def test_compute_parametric_var_es_rejects_invalid_confidence_level() -> None:
    with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
        compute_parametric_var_es(pd.Series([0.01, -0.02]), confidence_level=1.0)


def test_compute_parametric_var_es_rejects_empty_after_cleaning() -> None:
    with pytest.raises(ValueError, match="at least one valid observation"):
        compute_parametric_var_es(pd.Series([np.nan, None]), confidence_level=0.95)


def test_compute_parametric_var_es_handles_short_sample() -> None:
    var_95, es_95 = compute_parametric_var_es(pd.Series([-0.03]), confidence_level=0.95)
    assert var_95 == pytest.approx(-0.03)
    assert es_95 == pytest.approx(-0.03)


def test_compute_parametric_var_es_for_95_and_99() -> None:
    returns = pd.Series([-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03], dtype=float)
    var_95, es_95 = compute_parametric_var_es(returns, confidence_level=0.95)
    var_99, es_99 = compute_parametric_var_es(returns, confidence_level=0.99)

    assert var_95 <= 0
    assert es_95 <= var_95
    assert var_99 <= var_95
    assert es_99 <= var_99


def test_compute_garch_t_var_es_returns_ok_for_valid_sample() -> None:
    rng = np.random.default_rng(seed=123)
    returns = pd.Series(rng.normal(loc=0.0, scale=0.02, size=300), dtype=float)

    out = compute_garch_t_var_es(returns, confidence_level=0.95, min_sample_size=120)

    assert out["status"] == "ok"
    assert out["fallback_reason"] == "none"
    assert out["sample_size"] == 300
    assert float(out["nu"]) > 4.0
    assert float(out["sigma_next"]) > 0.0
    assert out["converged"] is True
    assert np.isfinite(float(out["var"]))
    assert np.isfinite(float(out["es"]))
    assert float(out["es"]) <= float(out["var"])


def test_compute_garch_t_var_es_falls_back_on_short_sample() -> None:
    returns = pd.Series([0.01, -0.02, 0.005], dtype=float)

    out = compute_garch_t_var_es(returns, confidence_level=0.95, min_sample_size=10)

    assert out["status"] == "fallback"
    assert out["fallback_reason"] == "short_sample"
    assert out["sample_size"] == 3
    assert out["converged"] is False
    assert np.isnan(float(out["var"]))
    assert np.isnan(float(out["es"]))
    assert np.isnan(float(out["nu"]))
    assert np.isnan(float(out["sigma_next"]))


def test_compute_garch_t_var_es_falls_back_on_internal_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    returns = pd.Series(np.linspace(-0.02, 0.02, 300), dtype=float)

    def _raise_runtime_error(*args: object, **kwargs: object) -> float:
        raise RuntimeError("boom")

    monkeypatch.setattr("nvda_risk_project.analysis.market_risk.np.var", _raise_runtime_error)
    out = compute_garch_t_var_es(returns, confidence_level=0.95, min_sample_size=120)

    assert out["status"] == "fallback"
    assert out["fallback_reason"] == "exception"
    assert out["sample_size"] == 300
    assert out["converged"] is False
    assert np.isnan(float(out["var"]))
    assert np.isnan(float(out["es"]))
    assert np.isnan(float(out["nu"]))
    assert np.isnan(float(out["sigma_next"]))


def test_kupiec_pof_test_returns_expected_fields() -> None:
    """Compute finite Kupiec statistics for a regular exceedance sample."""
    result = kupiec_pof_test(exceedance_count=7, sample_size=100, alpha=0.95)

    assert set(result) == {
        "lr_stat",
        "p_value",
        "reject_5pct",
        "expected_exceedance_rate",
        "observed_exceedance_rate",
    }
    assert result["expected_exceedance_rate"] == pytest.approx(0.05)
    assert result["observed_exceedance_rate"] == pytest.approx(0.07)
    assert result["lr_stat"] >= 0
    assert 0 <= result["p_value"] <= 1


def test_kupiec_pof_test_rejects_zero_sample_size() -> None:
    """Raise on zero sample size to prevent invalid backtests."""
    with pytest.raises(ValueError, match="sample_size must be a positive integer"):
        kupiec_pof_test(exceedance_count=0, sample_size=0, alpha=0.95)


def test_kupiec_pof_test_rejects_invalid_alpha() -> None:
    """Raise when alpha is outside (0, 1)."""
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        kupiec_pof_test(exceedance_count=1, sample_size=100, alpha=1.0)


def test_kupiec_pof_test_rejects_exceedance_count_larger_than_sample() -> None:
    """Raise when exceedance count exceeds sample size."""
    with pytest.raises(ValueError, match="cannot exceed sample_size"):
        kupiec_pof_test(exceedance_count=11, sample_size=10, alpha=0.95)


def test_christoffersen_independence_test_returns_expected_fields() -> None:
    """Compute finite Christoffersen independence statistics for mixed transitions."""
    exceedances = pd.Series([0, 1, 0, 0, 1, 1, 0, 1], dtype=float)
    result = christoffersen_independence_test(exceedances=exceedances)

    assert set(result) == {
        "lr_stat",
        "p_value",
        "reject_5pct",
        "n00",
        "n01",
        "n10",
        "n11",
    }
    assert result["lr_stat"] >= 0
    assert 0 <= result["p_value"] <= 1
    assert result["n00"] + result["n01"] + result["n10"] + result["n11"] == len(exceedances) - 1


@pytest.mark.parametrize("sequence", ([0, 0, 0, 0, 0], [1, 1, 1, 1, 1]))
def test_christoffersen_independence_test_handles_all_zero_or_one(sequence: list[int]) -> None:
    """Return a stable non-rejection for degenerate boundary sequences."""
    result = christoffersen_independence_test(exceedances=pd.Series(sequence, dtype=float))

    assert result["lr_stat"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)
    assert result["reject_5pct"] is False


def test_christoffersen_independence_test_rejects_short_sample() -> None:
    """Raise on insufficient sample length for transition-based testing."""
    with pytest.raises(ValueError, match="at least two valid observations"):
        christoffersen_independence_test(exceedances=pd.Series([1], dtype=float))

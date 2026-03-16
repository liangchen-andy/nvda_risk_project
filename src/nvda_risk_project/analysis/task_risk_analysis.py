"""Tasks for running risk analysis modules."""

from pathlib import Path

import pandas as pd
import pytask

from nvda_risk_project.analysis.drawdown_risk import estimate_drawdown_risk
from nvda_risk_project.analysis.liquidity_risk import estimate_liquidity_risk
from nvda_risk_project.analysis.macro_risk import build_daily_dgs10_panel
from nvda_risk_project.analysis.macro_risk import estimate_macro_risk
from nvda_risk_project.analysis.market_risk import (
    build_rolling_volatility_frame,
    christoffersen_independence_test,
    compute_garch_t_var_es,
    compute_historical_var_es,
    compute_parametric_var_es,
    estimate_market_risk,
    kupiec_pof_test,
)
from nvda_risk_project.analysis.systematic_risk import estimate_systematic_risk
from nvda_risk_project.config import (
    ANALYSIS_OUTPUT,
    PANEL_DAILY_DATA,
    PROCESSED_DATA,
    RAW_DATA,
    RISK_DIMENSIONS,
    VAR_LEVEL,
    VOL_WINDOW_DAYS,
)

RISK_FUNCTIONS = {
    "market": lambda panel: estimate_market_risk(panel, var_level=VAR_LEVEL),
    "liquidity": estimate_liquidity_risk,
    "drawdown": estimate_drawdown_risk,
    "systematic": estimate_systematic_risk,
    "macro": estimate_macro_risk,
}


for dimension in RISK_DIMENSIONS:

    @pytask.task(id=dimension)
    def task_run_risk_module(
        dimension: str = dimension,
        panel_data: Path = PROCESSED_DATA / "panel_monthly.csv",
        panel_daily_data: Path = PANEL_DAILY_DATA,
        macro_raw_data: Path = RAW_DATA / "macro_monthly.csv",
        produces: Path = ANALYSIS_OUTPUT / f"{dimension}_risk.csv",
    ) -> None:
        """Run one risk module and persist metrics as CSV."""
        produces.parent.mkdir(parents=True, exist_ok=True)
        if dimension == "macro":
            panel_daily = pd.read_parquet(panel_daily_data)
            macro_monthly = pd.read_csv(macro_raw_data, parse_dates=["date"])
            macro_daily_panel = build_daily_dgs10_panel(panel_daily=panel_daily, macro_monthly=macro_monthly)
            output = estimate_macro_risk(macro_daily_panel).assign(risk_dimension=dimension)
        else:
            panel = pd.read_csv(panel_data, parse_dates=["month"])
            output = RISK_FUNCTIONS[dimension](panel).assign(risk_dimension=dimension)
        output.to_csv(produces, index=False)


def task_create_rolling_volatility(
    panel_daily_data: Path = PANEL_DAILY_DATA,
    produces: Path = ANALYSIS_OUTPUT / "rolling_vol.csv",
) -> None:
    """Create rolling volatility artifact from daily return panel."""
    produces.parent.mkdir(parents=True, exist_ok=True)
    panel_daily = pd.read_parquet(panel_daily_data)
    rolling_vol = build_rolling_volatility_frame(
        panel_daily=panel_daily,
        window=VOL_WINDOW_DAYS,
    )
    rolling_vol.to_csv(produces, index=False)


def task_create_historical_var_es(
    panel_daily_data: Path = PANEL_DAILY_DATA,
    produces: Path = ANALYSIS_OUTPUT / "var_es_hist.csv",
) -> None:
    """Create historical/parametric/GARCH-t VaR-ES artifact for alpha 0.95 and 0.99."""
    produces.parent.mkdir(parents=True, exist_ok=True)
    panel_daily = pd.read_parquet(panel_daily_data)
    returns = pd.to_numeric(panel_daily["ret"], errors="coerce")
    sample_size = int(returns.notna().sum())

    rows: list[dict[str, str | float | int]] = []
    methods = {
        "historical": compute_historical_var_es,
        "parametric_normal": compute_parametric_var_es,
    }
    for method, compute_var_es in methods.items():
        for alpha in (0.95, 0.99):
            fallback_reason = "none"
            try:
                var, es = compute_var_es(returns=returns, confidence_level=alpha)
                status = "ok"
            except ValueError:
                var, es, status = float("nan"), float("nan"), "fallback"
                fallback_reason = "invalid_input"
            except Exception:
                var, es, status = float("nan"), float("nan"), "fallback"
                fallback_reason = "exception"

            rows.append(
                {
                    "method": method,
                    "alpha": alpha,
                    "var": var,
                    "es": es,
                    "status": status,
                    "fallback_reason": fallback_reason,
                    "sample_size": sample_size,
                    "nu": float("nan"),
                    "sigma_next": float("nan"),
                    "garch_converged": False,
                },
            )

    for alpha in (0.95, 0.99):
        result = compute_garch_t_var_es(returns=returns, confidence_level=alpha)
        rows.append(
            {
                "method": "garch_t",
                "alpha": alpha,
                "var": float(result["var"]),
                "es": float(result["es"]),
                "status": str(result["status"]),
                "fallback_reason": str(result["fallback_reason"]),
                "sample_size": int(result.get("sample_size", sample_size)),
                "nu": float(result.get("nu", float("nan"))),
                "sigma_next": float(result.get("sigma_next", float("nan"))),
                "garch_converged": bool(result.get("converged", False)),
            },
        )

    pd.DataFrame(rows).to_csv(produces, index=False)


def task_create_historical_var_exceedances(
    panel_daily_data: Path = PANEL_DAILY_DATA,
    var_es_hist_data: Path = ANALYSIS_OUTPUT / "var_es_hist.csv",
    produces: Path = ANALYSIS_OUTPUT / "var_exceedances_hist.csv",
) -> None:
    """Create VaR exceedance counts and rates by method and alpha."""
    produces.parent.mkdir(parents=True, exist_ok=True)
    returns = pd.to_numeric(pd.read_parquet(panel_daily_data)["ret"], errors="coerce").dropna()
    sample_size = int(returns.shape[0])
    var_es_hist = pd.read_csv(var_es_hist_data)

    rows: list[dict[str, str | float | int]] = []
    for row in var_es_hist.to_dict(orient="records"):
        method = str(row.get("method", "historical"))
        alpha = float(row["alpha"])
        var = float(row["var"]) if pd.notna(row["var"]) else float("nan")
        status = str(row.get("status", "fallback"))
        if status != "ok" or pd.isna(var) or sample_size == 0:
            exceedance_count = 0
            exceedance_rate = float("nan")
        else:
            exceedance_count = int((returns < var).sum())
            exceedance_rate = float(exceedance_count / sample_size)

        rows.append(
            {
                "method": method,
                "alpha": alpha,
                "var": var,
                "sample_size": sample_size,
                "exceedance_count": exceedance_count,
                "exceedance_rate": exceedance_rate,
                "status": status,
            },
        )

    pd.DataFrame(rows).to_csv(produces, index=False)


def task_create_historical_var_backtest(
    panel_daily_data: Path = PANEL_DAILY_DATA,
    var_es_hist_data: Path = ANALYSIS_OUTPUT / "var_es_hist.csv",
    produces: Path = ANALYSIS_OUTPUT / "var_backtest_hist.csv",
) -> None:
    """Create Kupiec and Christoffersen backtest metrics by method and alpha."""
    produces.parent.mkdir(parents=True, exist_ok=True)
    returns = pd.to_numeric(pd.read_parquet(panel_daily_data)["ret"], errors="coerce").dropna()
    sample_size = int(returns.shape[0])
    var_es_hist = pd.read_csv(var_es_hist_data)

    def _as_bool_like(value: object) -> bool:
        """Parse bool-like values; default to False when unknown."""
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
        return False

    rows: list[dict[str, str | float | int | bool]] = []
    for row in var_es_hist.to_dict(orient="records"):
        method = str(row.get("method", "historical"))
        alpha = float(row["alpha"])
        var = float(row["var"]) if pd.notna(row["var"]) else float("nan")
        status = str(row.get("status", "fallback"))
        upstream_reason = str(row.get("fallback_reason", "unknown"))
        upstream_nu = float(row["nu"]) if pd.notna(row.get("nu")) else float("nan")
        upstream_sigma_next = (
            float(row["sigma_next"]) if pd.notna(row.get("sigma_next")) else float("nan")
        )
        upstream_garch_converged = _as_bool_like(row.get("garch_converged", False))

        fallback = status != "ok" or pd.isna(var) or sample_size < 2
        exceedance_count = 0
        exceedance_rate = float("nan")
        kupiec_lr = float("nan")
        kupiec_p_value = float("nan")
        kupiec_reject_5pct = False
        christoffersen_lr = float("nan")
        christoffersen_p_value = float("nan")
        christoffersen_reject_5pct = False
        backtest_status = "fallback"
        fallback_reason = "unknown"

        if status != "ok":
            fallback_reason = f"upstream_{upstream_reason}" if upstream_reason != "none" else "upstream_fallback"
        elif pd.isna(var):
            fallback_reason = "nan_var"
        elif sample_size < 2:
            fallback_reason = "short_sample"

        if not fallback:
            exceedances = (returns < var).astype(int)
            exceedance_count = int(exceedances.sum())
            exceedance_rate = float(exceedance_count / sample_size)
            try:
                kupiec = kupiec_pof_test(
                    exceedance_count=exceedance_count,
                    sample_size=sample_size,
                    alpha=alpha,
                )
                christoffersen = christoffersen_independence_test(exceedances=exceedances)
                kupiec_lr = float(kupiec["lr_stat"])
                kupiec_p_value = float(kupiec["p_value"])
                kupiec_reject_5pct = bool(kupiec["reject_5pct"])
                christoffersen_lr = float(christoffersen["lr_stat"])
                christoffersen_p_value = float(christoffersen["p_value"])
                christoffersen_reject_5pct = bool(christoffersen["reject_5pct"])
                backtest_status = "ok"
                fallback_reason = "none"
            except ValueError:
                backtest_status = "fallback"
                fallback_reason = "invalid_input"
            except Exception:
                backtest_status = "fallback"
                fallback_reason = "exception"

        rows.append(
            {
                "method": method,
                "alpha": alpha,
                "var": var,
                "sample_size": sample_size,
                "exceedance_count": exceedance_count,
                "exceedance_rate": exceedance_rate,
                "kupiec_lr_stat": kupiec_lr,
                "kupiec_p_value": kupiec_p_value,
                "kupiec_reject_5pct": kupiec_reject_5pct,
                "christoffersen_lr_stat": christoffersen_lr,
                "christoffersen_p_value": christoffersen_p_value,
                "christoffersen_reject_5pct": christoffersen_reject_5pct,
                "status": backtest_status,
                "fallback_reason": fallback_reason,
                "upstream_nu": upstream_nu,
                "upstream_sigma_next": upstream_sigma_next,
                "upstream_garch_converged": upstream_garch_converged,
            },
        )

    pd.DataFrame(rows).to_csv(produces, index=False)


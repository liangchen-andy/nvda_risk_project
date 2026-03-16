"""Macro risk calculations with monthly and daily-DGS10 regression paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

_MONTHLY_REQUIRED = {"month", "nvda_return", "gdp_growth", "inflation_yoy", "policy_rate"}
_DAILY_REQUIRED = {"date", "ret", "dgs10"}


def _validate_macro_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Validate macro panel schema and enforce monthly frequency inputs."""
    missing = _MONTHLY_REQUIRED.difference(panel.columns)
    if missing:
        msg = f"panel is missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    month = pd.to_datetime(panel["month"], errors="coerce")
    if month.isna().any():
        msg = "month column must contain valid datetime values."
        raise ValueError(msg)
    if not month.dt.is_month_end.all():
        msg = (
            "macro risk requires month-end monthly data; "
            "aggregate higher-frequency data before regression."
        )
        raise ValueError(msg)
    if month.duplicated().any():
        msg = "month column contains duplicates; expected one row per month."
        raise ValueError(msg)
    if len(month) < 4:
        msg = "macro risk requires at least 4 monthly observations."
        raise ValueError(msg)
    return panel


def _validate_daily_dgs10_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Validate daily macro-risk inputs for DGS10 regression."""
    missing = _DAILY_REQUIRED.difference(panel.columns)
    if missing:
        msg = f"panel is missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["ret"] = pd.to_numeric(out["ret"], errors="coerce")
    out["dgs10"] = pd.to_numeric(out["dgs10"], errors="coerce")
    out = out.dropna(subset=["date", "ret", "dgs10"]).sort_values("date")
    if out.empty:
        msg = "daily DGS10 panel is empty after cleaning."
        raise ValueError(msg)
    if out["date"].duplicated().any():
        msg = "date column contains duplicates; expected one row per day."
        raise ValueError(msg)
    if len(out) < 30:
        msg = "daily DGS10 regression requires at least 30 observations."
        raise ValueError(msg)
    return out


def build_daily_dgs10_panel(panel_daily: pd.DataFrame, macro_monthly: pd.DataFrame) -> pd.DataFrame:
    """Build a daily regression panel by aligning monthly policy rate to daily returns."""
    required_daily = {"date", "ret"}
    required_macro = {"date", "policy_rate"}
    missing_daily = required_daily.difference(panel_daily.columns)
    missing_macro = required_macro.difference(macro_monthly.columns)
    if missing_daily:
        msg = f"panel_daily is missing required columns: {sorted(missing_daily)}"
        raise ValueError(msg)
    if missing_macro:
        msg = f"macro_monthly is missing required columns: {sorted(missing_macro)}"
        raise ValueError(msg)

    daily = panel_daily[["date", "ret"]].copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily["ret"] = pd.to_numeric(daily["ret"], errors="coerce")
    daily = daily.dropna(subset=["date", "ret"]).sort_values("date")

    macro = macro_monthly[["date", "policy_rate"]].copy()
    macro["date"] = pd.to_datetime(macro["date"], errors="coerce")
    macro["policy_rate"] = pd.to_numeric(macro["policy_rate"], errors="coerce")
    macro = macro.dropna(subset=["date", "policy_rate"]).sort_values("date")

    if daily.empty or macro.empty:
        msg = "cannot build daily DGS10 panel from empty inputs."
        raise ValueError(msg)

    aligned = pd.merge_asof(
        daily,
        macro.rename(columns={"policy_rate": "dgs10"}),
        on="date",
        direction="backward",
    )
    aligned["dgs10"] = aligned["dgs10"].ffill().bfill()
    return aligned[["date", "ret", "dgs10"]]


def _estimate_daily_dgs10_risk(panel: pd.DataFrame, *, hac_lags: int = 5) -> pd.DataFrame:
    """Estimate daily return sensitivity to DGS10 with HAC-robust inference."""
    panel = _validate_daily_dgs10_panel(panel)
    y = panel["ret"].to_numpy(dtype=float)
    x = sm.add_constant(panel["dgs10"].to_numpy(dtype=float))
    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": max(hac_lags, 1)})
    return pd.DataFrame(
        {
            "metric": [
                "intercept",
                "beta_dgs10",
                "beta_dgs10_tstat",
                "beta_dgs10_pvalue",
                "r_squared",
                "obs_count",
            ],
            "value": [
                float(model.params[0]),
                float(model.params[1]),
                float(model.tvalues[1]),
                float(model.pvalues[1]),
                float(model.rsquared),
                float(panel.shape[0]),
            ],
        },
    )


def estimate_macro_risk(panel: pd.DataFrame) -> pd.DataFrame:
    """Estimate macro risk on monthly panel or daily DGS10-aligned panel."""
    if _DAILY_REQUIRED.issubset(panel.columns):
        return _estimate_daily_dgs10_risk(panel)

    panel = _validate_macro_panel(panel)
    y = panel["nvda_return"].to_numpy(dtype=float)
    x = panel[["gdp_growth", "inflation_yoy", "policy_rate"]].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return pd.DataFrame(
        {
            "metric": ["intercept", "beta_gdp_growth", "beta_inflation", "beta_policy_rate"],
            "value": coef,
        },
    )

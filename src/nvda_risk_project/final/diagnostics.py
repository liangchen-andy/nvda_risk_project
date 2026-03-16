"""Utilities to build diagnostics artifacts for reproducibility checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nvda_risk_project.config import (
    ASSET_TICKER,
    BENCHMARK_TICKER,
    SAMPLE_END_DATE,
    SAMPLE_START_DATE,
)

_PANEL_CORE_COLUMNS: tuple[str, ...] = (
    "adj_close",
    "close",
    "volume",
    "ret",
    "logret",
    "market_ret",
)

_QUALITY_GATE_CHECKS: tuple[str, ...] = (
    "provenance_exists",
    "provenance_panel_sha_match",
    "provenance_nvda_raw_sha_match",
    "provenance_market_raw_sha_match",
    "panel_daily_required_cols_present",
    "panel_window_matches_target",
    "scorecard_status_consistent",
    "scorecard_status_reason_consistent",
    "var_backtest_historical_rows_consistent",
    "var_backtest_historical_fallback_events_consistent",
    "var_backtest_parametric_rows_consistent",
    "var_backtest_parametric_fallback_events_consistent",
    "var_backtest_other_rows_consistent",
    "var_backtest_other_fallback_events_consistent",
    "var_backtest_garch_rows_consistent",
    "var_backtest_garch_fallback_events_consistent",
    "var_backtest_garch_status_ok_count_consistent",
    "historical_kupiec_reject_rate_5pct_consistent",
    "historical_christoffersen_reject_rate_5pct_consistent",
    "parametric_kupiec_reject_rate_5pct_consistent",
    "parametric_christoffersen_reject_rate_5pct_consistent",
    "garch_kupiec_reject_rate_5pct_consistent",
    "garch_christoffersen_reject_rate_5pct_consistent",
    "systematic_beta_consistent",
    "systematic_alpha_consistent",
    "systematic_beta_rolling_60m_latest_consistent",
    "systematic_beta_rolling_60m_valid_points_consistent",
    "systematic_beta_rolling_60m_fallback_consistent",
    "systematic_beta_rolling_252m_latest_consistent",
    "systematic_beta_rolling_252m_valid_points_consistent",
    "systematic_beta_rolling_252m_fallback_consistent",
    "macro_frequency_ok",
)


def _safe_read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file safely and return an empty frame on failure."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_row_count(path: Path) -> int:
    """Return row count for a CSV path, or zero when unavailable."""
    if not path.exists():
        return 0
    return int(_safe_read_csv(path).shape[0])


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet file safely and return an empty frame on failure."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _safe_date_bounds(path: Path, date_column: str = "date") -> tuple[str, str]:
    """Return ISO date bounds from a CSV date column."""
    frame = _safe_read_csv(path)
    if frame.empty or date_column not in frame.columns:
        return "", ""
    dates = pd.to_datetime(frame[date_column], errors="coerce").dropna()
    if dates.empty:
        return "", ""
    return dates.min().date().isoformat(), dates.max().date().isoformat()


def _safe_parquet_date_bounds(path: Path, date_column: str = "date") -> tuple[str, str]:
    """Return ISO date bounds from a parquet date column."""
    frame = _safe_read_parquet(path)
    if frame.empty or date_column not in frame.columns:
        return "", ""
    dates = pd.to_datetime(frame[date_column], errors="coerce").dropna()
    if dates.empty:
        return "", ""
    return dates.min().date().isoformat(), dates.max().date().isoformat()


def _file_sha256(path: Path) -> str:
    """Return the SHA256 checksum for a file path."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_source_label(value: str) -> str:
    """Normalize source labels so order differences do not affect equality checks."""
    parts = [part.strip() for part in value.split("+") if part.strip()]
    return "+".join(sorted(parts)) if parts else "cache"


def _load_scorecard(path: Path) -> tuple[dict[str, object], bool]:
    """Load scorecard JSON and flag parse errors."""
    if not path.exists():
        return {}, False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload, False
    except json.JSONDecodeError:
        return {}, True
    return {}, True


def _provenance_stats(
    path: Path | None,
    *,
    nvda_raw_path: Path,
    market_raw_path: Path,
) -> dict[str, str]:
    """Summarize provenance file integrity and consistency checks."""
    defaults = {
        "provenance_exists": "False",
        "provenance_parse_error": "False",
        "provenance_ticker_match": "unknown",
        "provenance_sample_match": "unknown",
        "provenance_panel_sha_match": "unknown",
        "provenance_nvda_raw_sha_match": "unknown",
        "provenance_market_raw_sha_match": "unknown",
        "provenance_source_match": "unknown",
    }
    if path is None:
        return defaults

    defaults["provenance_exists"] = str(path.exists())
    if not path.exists():
        return defaults

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        defaults["provenance_parse_error"] = "True"
        return defaults

    if not isinstance(payload, dict):
        defaults["provenance_parse_error"] = "True"
        return defaults

    tickers = payload.get("tickers")
    if isinstance(tickers, list):
        observed = {str(ticker) for ticker in tickers}
        defaults["provenance_ticker_match"] = str(
            observed == {ASSET_TICKER, BENCHMARK_TICKER},
        )

    source = payload.get("source")
    if isinstance(source, dict):
        source_asset = source.get(ASSET_TICKER)
        source_market = source.get(BENCHMARK_TICKER)
        raw_asset = _source_from_raw(nvda_raw_path)
        raw_market = _source_from_raw(market_raw_path)
        if (
            isinstance(source_asset, str)
            and isinstance(source_market, str)
            and raw_asset != "unknown"
            and raw_market != "unknown"
        ):
            defaults["provenance_source_match"] = str(
                _normalize_source_label(source_asset) == _normalize_source_label(raw_asset)
                and _normalize_source_label(source_market)
                == _normalize_source_label(raw_market),
            )

    panel_artifact = payload.get("artifacts", {}).get("panel_daily", {})
    if not isinstance(panel_artifact, dict):
        return defaults

    raw_artifacts = payload.get("artifacts", {})
    if isinstance(raw_artifacts, dict):
        nvda_artifact = raw_artifacts.get("nvda_raw", {})
        market_artifact = raw_artifacts.get("sp500_raw", {})
        if isinstance(nvda_artifact, dict):
            nvda_sha = nvda_artifact.get("sha256")
            if isinstance(nvda_sha, str) and nvda_sha and nvda_raw_path.exists():
                defaults["provenance_nvda_raw_sha_match"] = str(
                    _file_sha256(nvda_raw_path) == nvda_sha,
                )
        if isinstance(market_artifact, dict):
            market_sha = market_artifact.get("sha256")
            if isinstance(market_sha, str) and market_sha and market_raw_path.exists():
                defaults["provenance_market_raw_sha_match"] = str(
                    _file_sha256(market_raw_path) == market_sha,
                )

    panel_path_raw = panel_artifact.get("path")
    panel_sha = panel_artifact.get("sha256")
    if not isinstance(panel_path_raw, str) or not panel_path_raw:
        return defaults

    panel_path = Path(panel_path_raw)
    if not panel_path.exists():
        return defaults

    if isinstance(panel_sha, str) and panel_sha:
        defaults["provenance_panel_sha_match"] = str(_file_sha256(panel_path) == panel_sha)

    sample_start = payload.get("sample_start")
    sample_end = payload.get("sample_end")
    panel_start, panel_end = _safe_parquet_date_bounds(panel_path)
    if (
        isinstance(sample_start, str)
        and isinstance(sample_end, str)
        and panel_start
        and panel_end
    ):
        defaults["provenance_sample_match"] = str(
            sample_start == panel_start and sample_end == panel_end,
        )

    return defaults


def _rolling_vol_stats(path: Path) -> tuple[int, int, str]:
    """Summarize rolling-volatility coverage and fallback status."""
    frame = _safe_read_csv(path)
    if frame.empty or "rolling_vol" not in frame.columns:
        return 0, 0, "unknown"

    total_rows = int(frame.shape[0])
    valid_points = int(pd.to_numeric(frame["rolling_vol"], errors="coerce").notna().sum())
    fallback_used = str(valid_points == 0)
    return total_rows, valid_points, fallback_used


def _source_from_raw(path: Path) -> str:
    """Infer a normalized source label from raw input data."""
    frame = _safe_read_csv(path)
    if frame.empty:
        return "unknown"
    if "data_source" not in frame.columns:
        return "cache"
    sources = sorted({str(value) for value in frame["data_source"].dropna().tolist()})
    return "+".join(sources) if sources else "cache"


def _combined_data_source(nvda_raw_path: Path, market_raw_path: Path) -> str:
    """Combine inferred raw-data sources across asset and benchmark."""
    combined = sorted(
        {
            _source_from_raw(nvda_raw_path),
            _source_from_raw(market_raw_path),
        },
    )
    known = [source for source in combined if source != "unknown"]
    return "+".join(known) if known else "unknown"


def _placeholder_source_from_raw(path: Path) -> str:
    """Return whether a raw input contains placeholder data_source rows."""
    frame = _safe_read_csv(path)
    if frame.empty:
        return "unknown"
    if "data_source" not in frame.columns:
        return "False"

    sources = {str(value).strip().lower() for value in frame["data_source"].dropna().tolist()}
    if not sources:
        return "False"
    return str("placeholder" in sources)


def _placeholder_source_detected(nvda_raw_path: Path, market_raw_path: Path) -> str:
    """Aggregate placeholder-source detection across asset and benchmark raws."""
    flags = [
        _placeholder_source_from_raw(nvda_raw_path),
        _placeholder_source_from_raw(market_raw_path),
    ]
    if "True" in flags:
        return "True"
    if all(flag == "False" for flag in flags):
        return "False"
    return "unknown"


def _panel_daily_qc_stats(path: Path | None) -> dict[str, str]:
    """Summarize panel_daily quality checks for diagnostics."""
    exists = str(path.exists()) if path is not None else "False"
    defaults = {
        "panel_daily_exists": exists,
        "panel_daily_rows": "0",
        "panel_daily_required_cols_present": "unknown",
        "panel_daily_date_monotonic": "unknown",
        "panel_daily_duplicate_dates": "0",
        "panel_daily_core_missing_rate": "nan",
    }
    if path is None:
        return defaults

    frame = _safe_read_parquet(path)
    if frame.empty:
        return defaults

    defaults["panel_daily_rows"] = str(int(frame.shape[0]))
    defaults["panel_daily_required_cols_present"] = str(
        set(_PANEL_CORE_COLUMNS).issubset(frame.columns),
    )

    available_core_columns = [column for column in _PANEL_CORE_COLUMNS if column in frame.columns]
    if available_core_columns:
        core = frame[available_core_columns]
        missing_cells = int(core.isna().sum().sum())
        total_cells = int(core.shape[0] * core.shape[1])
        if total_cells > 0:
            defaults["panel_daily_core_missing_rate"] = str(float(missing_cells / total_cells))

    if "date" in frame.columns:
        dates = pd.to_datetime(frame["date"], errors="coerce")
        non_missing_dates = dates.dropna()
        if not non_missing_dates.empty:
            defaults["panel_daily_date_monotonic"] = str(
                bool(dates.notna().all() and dates.is_monotonic_increasing),
            )
            defaults["panel_daily_duplicate_dates"] = str(int(non_missing_dates.duplicated().sum()))
        elif frame.shape[0] > 0:
            defaults["panel_daily_date_monotonic"] = "False"
    return defaults


def _window_match(sample_start: str, sample_end: str) -> str:
    """Return whether observed sample bounds match the configured target window."""
    if not sample_start or not sample_end:
        return "unknown"
    return str(sample_start == SAMPLE_START_DATE and sample_end == SAMPLE_END_DATE)


def _normalize_tristate(value: object) -> str:
    """Normalize bool-like diagnostics values to true/false/unknown."""
    normalized = str(value).strip().lower()
    if normalized in {"true", "false"}:
        return normalized
    return "unknown"


def _expected_scorecard_status(
    *,
    panel_window_matches_target: str,
    rolling_vol_fallback: str,
    backtest_fallback_events: object,
) -> str:
    """Derive expected scorecard status from diagnostics-observed gates."""
    if panel_window_matches_target not in {"True", "False"}:
        return "unknown"
    if rolling_vol_fallback not in {"True", "False"}:
        return "unknown"
    try:
        fallback_events = int(backtest_fallback_events)
    except (TypeError, ValueError):
        return "unknown"

    if panel_window_matches_target == "False" or rolling_vol_fallback == "True":
        return "fail"
    if fallback_events > 0:
        return "warn"
    return "ok"


def _expected_scorecard_status_reason(
    *,
    panel_window_matches_target: str,
    rolling_vol_fallback: str,
    backtest_fallback_events: object,
) -> str:
    """Derive expected scorecard status reason from diagnostics-observed gates."""
    if panel_window_matches_target not in {"True", "False"}:
        return "unknown"
    if rolling_vol_fallback not in {"True", "False"}:
        return "unknown"
    try:
        fallback_events = int(backtest_fallback_events)
    except (TypeError, ValueError):
        return "unknown"

    if panel_window_matches_target == "False":
        return "window_not_target"
    if rolling_vol_fallback == "True":
        return "rolling_vol_fallback"
    if fallback_events > 0:
        return "backtest_fallbacks"
    return "none"


def _garch_backtest_quality_stats(path: Path | None) -> dict[str, str]:
    """Summarize garch_t backtest counts from the raw backtest artifact."""
    defaults = {
        "var_backtest_garch_rows_observed": "unknown",
        "var_backtest_garch_fallback_events_observed": "unknown",
        "var_backtest_garch_status_ok_count_observed": "unknown",
    }
    if path is None:
        return defaults

    frame = _safe_read_csv(path)
    required_columns = {"method", "status"}
    if frame.empty or not required_columns.issubset(frame.columns):
        return defaults

    methods = frame["method"].astype(str).str.strip().str.lower()
    status = frame["status"].astype(str).str.lower()
    garch_mask = methods == "garch_t"
    defaults["var_backtest_garch_rows_observed"] = str(int(garch_mask.sum()))
    defaults["var_backtest_garch_fallback_events_observed"] = str(
        int((garch_mask & (status != "ok")).sum()),
    )
    defaults["var_backtest_garch_status_ok_count_observed"] = str(
        int((garch_mask & (status == "ok")).sum()),
    )
    return defaults


def _backtest_method_quality_stats(path: Path | None) -> dict[str, str]:
    """Summarize observed backtest row/fallback counts by method buckets."""
    defaults = {
        "var_backtest_historical_rows_observed": "unknown",
        "var_backtest_historical_fallback_events_observed": "unknown",
        "var_backtest_parametric_rows_observed": "unknown",
        "var_backtest_parametric_fallback_events_observed": "unknown",
        "var_backtest_other_rows_observed": "unknown",
        "var_backtest_other_fallback_events_observed": "unknown",
    }
    if path is None:
        return defaults

    frame = _safe_read_csv(path)
    if frame.empty or "status" not in frame.columns:
        return defaults

    status = frame["status"].astype(str).str.lower()
    fallback = status != "ok"
    methods = (
        frame["method"].astype(str).str.strip().str.lower()
        if "method" in frame.columns
        else pd.Series(["historical"] * len(frame), index=frame.index)
    )

    historical_mask = methods == "historical"
    parametric_mask = methods == "parametric_normal"
    other_mask = ~(historical_mask | parametric_mask)

    defaults["var_backtest_historical_rows_observed"] = str(int(historical_mask.sum()))
    defaults["var_backtest_historical_fallback_events_observed"] = str(
        int((historical_mask & fallback).sum()),
    )
    defaults["var_backtest_parametric_rows_observed"] = str(int(parametric_mask.sum()))
    defaults["var_backtest_parametric_fallback_events_observed"] = str(
        int((parametric_mask & fallback).sum()),
    )
    defaults["var_backtest_other_rows_observed"] = str(int(other_mask.sum()))
    defaults["var_backtest_other_fallback_events_observed"] = str(
        int((other_mask & fallback).sum()),
    )
    return defaults


def _int_value_consistent(recorded: object, observed: str) -> str:
    """Return true/false/unknown for integer consistency checks."""
    try:
        recorded_value = int(recorded)
        observed_value = int(observed)
    except (TypeError, ValueError):
        return "unknown"
    return str(recorded_value == observed_value)


def _float_value_consistent(recorded: object, observed: str) -> str:
    """Return true/false/unknown for float consistency checks (nan==nan)."""
    try:
        recorded_value = float(recorded)
        observed_value = float(observed)
    except (TypeError, ValueError):
        return "unknown"
    if np.isnan(recorded_value) and np.isnan(observed_value):
        return "True"
    if np.isnan(recorded_value) or np.isnan(observed_value):
        return "False"
    return str(abs(recorded_value - observed_value) <= 1e-12)


def _bool_value_consistent(recorded: object, observed: str) -> str:
    """Return true/false/unknown for bool consistency checks."""
    recorded_value = _normalize_tristate(recorded)
    observed_value = _normalize_tristate(observed)
    if recorded_value == "unknown" or observed_value == "unknown":
        return "unknown"
    return str(recorded_value == observed_value)


def _as_bool_like(value: object) -> bool | None:
    """Parse bool-like values from diagnostics artifacts."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    return None


def _backtest_method_reject_rates_observed(path: Path | None) -> dict[str, str]:
    """Compute observed method-level 5% reject rates from backtest artifacts."""
    defaults = {
        "historical_kupiec_reject_rate_5pct_observed": "unknown",
        "historical_christoffersen_reject_rate_5pct_observed": "unknown",
        "parametric_kupiec_reject_rate_5pct_observed": "unknown",
        "parametric_christoffersen_reject_rate_5pct_observed": "unknown",
        "garch_kupiec_reject_rate_5pct_observed": "unknown",
        "garch_christoffersen_reject_rate_5pct_observed": "unknown",
    }
    if path is None:
        return defaults

    frame = _safe_read_csv(path)
    if frame.empty or "status" not in frame.columns:
        return defaults

    methods = (
        frame["method"].astype(str).str.strip().str.lower()
        if "method" in frame.columns
        else pd.Series(["historical"] * len(frame), index=frame.index)
    )
    status_ok = frame["status"].astype(str).str.lower() == "ok"
    def _reject_series(
        reject_column: str,
        pvalue_column: str,
    ) -> pd.Series | None:
        if reject_column in frame.columns:
            return frame[reject_column].map(_as_bool_like)
        if pvalue_column in frame.columns:
            p_values = pd.to_numeric(frame[pvalue_column], errors="coerce")
            return p_values.map(lambda value: bool(value < 0.05) if pd.notna(value) else None)
        return None

    def _rate(
        method_label: str,
        reject_column: str,
        pvalue_column: str,
    ) -> str:
        parsed = _reject_series(reject_column, pvalue_column)
        if parsed is None:
            return "unknown"
        method_ok = (methods == method_label) & status_ok
        if not method_ok.any():
            return "nan"
        values = parsed.loc[method_ok].dropna()
        if values.empty:
            return "unknown"
        return str(float(values.mean()))

    defaults["historical_kupiec_reject_rate_5pct_observed"] = _rate(
        "historical",
        "kupiec_reject_5pct",
        "kupiec_p_value",
    )
    defaults["historical_christoffersen_reject_rate_5pct_observed"] = _rate(
        "historical",
        "christoffersen_reject_5pct",
        "christoffersen_p_value",
    )
    defaults["parametric_kupiec_reject_rate_5pct_observed"] = _rate(
        "parametric_normal",
        "kupiec_reject_5pct",
        "kupiec_p_value",
    )
    defaults["parametric_christoffersen_reject_rate_5pct_observed"] = _rate(
        "parametric_normal",
        "christoffersen_reject_5pct",
        "christoffersen_p_value",
    )
    defaults["garch_kupiec_reject_rate_5pct_observed"] = _rate(
        "garch_t",
        "kupiec_reject_5pct",
        "kupiec_p_value",
    )
    defaults["garch_christoffersen_reject_rate_5pct_observed"] = _rate(
        "garch_t",
        "christoffersen_reject_5pct",
        "christoffersen_p_value",
    )
    return defaults


def _historical_var_es_stats(path: Path) -> dict[str, str]:
    """Summarize historical VaR/ES artifact coverage for diagnostics."""
    defaults = {
        "hist_var_es_exists": str(path.exists()),
        "hist_var_es_method_count": "0",
        "parametric_var_es_present": "False",
        "garch_var_es_present": "False",
        "hist_var_es_alpha_count": "0",
        "hist_var_es_status_ok_count": "0",
        "hist_var_95": "nan",
        "hist_es_95": "nan",
        "hist_var_99": "nan",
        "hist_es_99": "nan",
        "param_var_95": "nan",
        "param_es_95": "nan",
        "param_var_99": "nan",
        "param_es_99": "nan",
    }
    frame = _safe_read_csv(path)
    required_columns = {"alpha", "var", "es", "status"}
    if frame.empty or not required_columns.issubset(frame.columns):
        return defaults

    historical_frame = frame
    parametric_frame = pd.DataFrame()
    if "method" in frame.columns:
        methods = frame["method"].astype(str).str.strip().str.lower()
        defaults["hist_var_es_method_count"] = str(int(methods.nunique()))
        defaults["parametric_var_es_present"] = str("parametric_normal" in set(methods))
        defaults["garch_var_es_present"] = str("garch_t" in set(methods))
        historical_frame = frame.loc[methods == "historical"]
        parametric_frame = frame.loc[methods == "parametric_normal"]
    else:
        defaults["hist_var_es_method_count"] = "1"

    if not parametric_frame.empty:
        param_alpha = pd.to_numeric(parametric_frame["alpha"], errors="coerce")
        param_var = pd.to_numeric(parametric_frame["var"], errors="coerce")
        param_es = pd.to_numeric(parametric_frame["es"], errors="coerce")
        for level in (0.95, 0.99):
            mask = (param_alpha - level).abs() < 1e-9
            if mask.any():
                defaults[f"param_var_{int(level * 100)}"] = str(float(param_var[mask].iloc[0]))
                defaults[f"param_es_{int(level * 100)}"] = str(float(param_es[mask].iloc[0]))

    if historical_frame.empty:
        return defaults

    alpha = pd.to_numeric(historical_frame["alpha"], errors="coerce")
    var = pd.to_numeric(historical_frame["var"], errors="coerce")
    es = pd.to_numeric(historical_frame["es"], errors="coerce")
    status = historical_frame["status"].astype(str)
    defaults["hist_var_es_alpha_count"] = str(int(alpha.dropna().nunique()))
    defaults["hist_var_es_status_ok_count"] = str(int((status == "ok").sum()))

    for level in (0.95, 0.99):
        mask = (alpha - level).abs() < 1e-9
        if mask.any():
            defaults[f"hist_var_{int(level * 100)}"] = str(float(var[mask].iloc[0]))
            defaults[f"hist_es_{int(level * 100)}"] = str(float(es[mask].iloc[0]))
    return defaults


def _garch_var_es_parameter_stats(path: Path) -> dict[str, str]:
    """Summarize whether garch_t parameter diagnostics are finite and valid."""
    defaults = {
        "garch_var_es_nu_finite": "unknown",
        "garch_var_es_nu_gt4": "unknown",
        "garch_var_es_sigma_next_positive": "unknown",
    }
    frame = _safe_read_csv(path)
    required_columns = {"method", "status", "nu", "sigma_next"}
    if frame.empty or not required_columns.issubset(frame.columns):
        return defaults

    methods = frame["method"].astype(str).str.strip().str.lower()
    garch = frame.loc[methods == "garch_t"].copy()
    if garch.empty:
        return defaults

    status_ok = garch["status"].astype(str).str.lower() == "ok"
    if not status_ok.any():
        return defaults

    nu = pd.to_numeric(garch.loc[status_ok, "nu"], errors="coerce")
    sigma_next = pd.to_numeric(garch.loc[status_ok, "sigma_next"], errors="coerce")
    defaults["garch_var_es_nu_finite"] = str(bool(np.isfinite(nu).all()))
    defaults["garch_var_es_nu_gt4"] = str(bool((nu > 4.0).all()))
    defaults["garch_var_es_sigma_next_positive"] = str(bool((sigma_next > 0).all()))
    return defaults


def _historical_exceedance_stats(path: Path) -> dict[str, str]:
    """Summarize historical VaR exceedance artifact coverage for diagnostics."""
    defaults = {
        "hist_exceed_exists": str(path.exists()),
        "hist_exceed_method_count": "0",
        "parametric_exceed_present": "False",
        "garch_exceed_present": "False",
        "hist_exceed_status_ok_count": "0",
        "hist_exceed_95_count": "0",
        "hist_exceed_95_rate": "nan",
        "hist_exceed_95_gap": "nan",
        "hist_exceed_99_count": "0",
        "hist_exceed_99_rate": "nan",
        "hist_exceed_99_gap": "nan",
    }
    frame = _safe_read_csv(path)
    required_columns = {"alpha", "exceedance_count", "exceedance_rate", "status"}
    if frame.empty or not required_columns.issubset(frame.columns):
        return defaults

    historical_frame = frame
    if "method" in frame.columns:
        methods = frame["method"].astype(str).str.strip().str.lower()
        defaults["hist_exceed_method_count"] = str(int(methods.nunique()))
        defaults["parametric_exceed_present"] = str("parametric_normal" in set(methods))
        defaults["garch_exceed_present"] = str("garch_t" in set(methods))
        historical_frame = frame.loc[methods == "historical"]
    else:
        defaults["hist_exceed_method_count"] = "1"

    if historical_frame.empty:
        return defaults

    alpha = pd.to_numeric(historical_frame["alpha"], errors="coerce")
    count = pd.to_numeric(historical_frame["exceedance_count"], errors="coerce")
    rate = pd.to_numeric(historical_frame["exceedance_rate"], errors="coerce")
    status = historical_frame["status"].astype(str)
    defaults["hist_exceed_status_ok_count"] = str(int((status == "ok").sum()))

    for level in (0.95, 0.99):
        mask = (alpha - level).abs() < 1e-9
        if mask.any():
            observed_rate = float(rate[mask].iloc[0])
            expected_rate = 1 - level
            gap = observed_rate - expected_rate
            gap = 0.0 if abs(gap) < 1e-12 else round(gap, 12)
            defaults[f"hist_exceed_{int(level * 100)}_count"] = str(int(count[mask].iloc[0]))
            defaults[f"hist_exceed_{int(level * 100)}_rate"] = str(observed_rate)
            defaults[f"hist_exceed_{int(level * 100)}_gap"] = str(float(gap))
    return defaults


def _historical_backtest_stats(path: Path | None) -> dict[str, str]:
    """Summarize historical Kupiec/Christoffersen backtest artifact coverage."""
    exists = str(path.exists()) if path is not None else "False"
    defaults = {
        "hist_backtest_exists": exists,
        "hist_backtest_method_count": "0",
        "parametric_backtest_present": "False",
        "garch_backtest_present": "False",
        "hist_backtest_alpha_count": "0",
        "hist_backtest_status_ok_count": "0",
        "hist_kupiec_lr_95": "nan",
        "hist_kupiec_lr_99": "nan",
        "hist_kupiec_pvalue_95": "nan",
        "hist_kupiec_pvalue_99": "nan",
        "hist_christoffersen_lr_95": "nan",
        "hist_christoffersen_lr_99": "nan",
        "hist_christoffersen_pvalue_95": "nan",
        "hist_christoffersen_pvalue_99": "nan",
        "param_kupiec_lr_95": "nan",
        "param_kupiec_lr_99": "nan",
        "param_kupiec_pvalue_95": "nan",
        "param_kupiec_pvalue_99": "nan",
        "param_christoffersen_lr_95": "nan",
        "param_christoffersen_lr_99": "nan",
        "param_christoffersen_pvalue_95": "nan",
        "param_christoffersen_pvalue_99": "nan",
        "garch_kupiec_lr_95": "nan",
        "garch_kupiec_lr_99": "nan",
        "garch_kupiec_pvalue_95": "nan",
        "garch_kupiec_pvalue_99": "nan",
        "garch_christoffersen_lr_95": "nan",
        "garch_christoffersen_lr_99": "nan",
        "garch_christoffersen_pvalue_95": "nan",
        "garch_christoffersen_pvalue_99": "nan",
    }
    if path is None:
        return defaults

    frame = _safe_read_csv(path)
    required_columns = {"alpha", "status"}
    if frame.empty or not required_columns.issubset(frame.columns):
        return defaults

    historical_frame = frame
    parametric_frame = pd.DataFrame()
    garch_frame = pd.DataFrame()
    if "method" in frame.columns:
        methods = frame["method"].astype(str).str.strip().str.lower()
        defaults["hist_backtest_method_count"] = str(int(methods.nunique()))
        defaults["parametric_backtest_present"] = str("parametric_normal" in set(methods))
        defaults["garch_backtest_present"] = str("garch_t" in set(methods))
        historical_frame = frame.loc[methods == "historical"]
        parametric_frame = frame.loc[methods == "parametric_normal"]
        garch_frame = frame.loc[methods == "garch_t"]
    else:
        defaults["hist_backtest_method_count"] = "1"

    def _fill_method_stats(method_frame: pd.DataFrame, prefix: str) -> None:
        if method_frame.empty:
            return
        alpha = pd.to_numeric(method_frame["alpha"], errors="coerce")

        def _numeric_column(column: str) -> pd.Series:
            if column not in method_frame.columns:
                return pd.Series(np.nan, index=method_frame.index, dtype=float)
            return pd.to_numeric(method_frame[column], errors="coerce")

        kupiec_lr = _numeric_column("kupiec_lr_stat")
        kupiec_p = _numeric_column("kupiec_p_value")
        christoffersen_lr = _numeric_column("christoffersen_lr_stat")
        christoffersen_p = _numeric_column("christoffersen_p_value")
        for level in (0.95, 0.99):
            mask = (alpha - level).abs() < 1e-9
            if mask.any():
                defaults[f"{prefix}_kupiec_lr_{int(level * 100)}"] = str(
                    float(kupiec_lr[mask].iloc[0]),
                )
                defaults[f"{prefix}_kupiec_pvalue_{int(level * 100)}"] = str(
                    float(kupiec_p[mask].iloc[0]),
                )
                defaults[f"{prefix}_christoffersen_lr_{int(level * 100)}"] = str(
                    float(christoffersen_lr[mask].iloc[0]),
                )
                defaults[f"{prefix}_christoffersen_pvalue_{int(level * 100)}"] = str(
                    float(christoffersen_p[mask].iloc[0]),
                )

    if historical_frame.empty:
        _fill_method_stats(parametric_frame, "param")
        _fill_method_stats(garch_frame, "garch")
        return defaults

    alpha = pd.to_numeric(historical_frame["alpha"], errors="coerce")
    status = historical_frame["status"].astype(str)
    defaults["hist_backtest_alpha_count"] = str(int(alpha.dropna().nunique()))
    defaults["hist_backtest_status_ok_count"] = str(int((status == "ok").sum()))

    _fill_method_stats(historical_frame, "hist")
    _fill_method_stats(parametric_frame, "param")
    _fill_method_stats(garch_frame, "garch")
    return defaults


def _garch_backtest_upstream_stats(path: Path | None) -> dict[str, str]:
    """Summarize upstream GARCH parameter traces in backtest artifacts."""
    exists = str(path.exists()) if path is not None else "False"
    defaults = {
        "garch_backtest_upstream_exists": exists,
        "garch_backtest_upstream_converged_present": "unknown",
        "garch_backtest_upstream_converged_true_count": "unknown",
        "garch_backtest_upstream_nu_finite": "unknown",
        "garch_backtest_upstream_sigma_positive": "unknown",
    }
    if path is None:
        return defaults

    frame = _safe_read_csv(path)
    required_columns = {
        "method",
        "status",
        "upstream_nu",
        "upstream_sigma_next",
        "upstream_garch_converged",
    }
    if frame.empty or not required_columns.issubset(frame.columns):
        return defaults

    methods = frame["method"].astype(str).str.strip().str.lower()
    status_ok = frame["status"].astype(str).str.lower() == "ok"
    garch_ok = frame.loc[(methods == "garch_t") & status_ok].copy()
    if garch_ok.empty:
        return defaults

    parsed_converged = garch_ok["upstream_garch_converged"].map(_as_bool_like)
    converged_present = bool(parsed_converged.notna().all())
    defaults["garch_backtest_upstream_converged_present"] = str(converged_present)
    defaults["garch_backtest_upstream_converged_true_count"] = (
        str(int(parsed_converged.sum())) if converged_present else "unknown"
    )

    nu = pd.to_numeric(garch_ok["upstream_nu"], errors="coerce")
    sigma_next = pd.to_numeric(garch_ok["upstream_sigma_next"], errors="coerce")
    defaults["garch_backtest_upstream_nu_finite"] = str(bool(np.isfinite(nu).all()))
    defaults["garch_backtest_upstream_sigma_positive"] = str(bool((sigma_next > 0).all()))
    return defaults


def _macro_frequency_stats(path: Path | None) -> dict[str, str]:
    """Summarize macro-data frequency checks for diagnostics."""
    exists = str(path.exists()) if path is not None else "False"
    defaults = {
        "macro_data_exists": exists,
        "macro_rows": "0",
        "macro_month_end_only": "unknown",
        "macro_duplicate_month": "unknown",
        "macro_frequency_ok": "unknown",
    }
    if path is None:
        return defaults

    frame = _safe_read_csv(path)
    if frame.empty or "date" not in frame.columns:
        return defaults

    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any():
        defaults["macro_month_end_only"] = "False"
        defaults["macro_duplicate_month"] = "unknown"
        defaults["macro_frequency_ok"] = "False"
        defaults["macro_rows"] = str(int(frame.shape[0]))
        return defaults

    month_end_only = bool(dates.dt.is_month_end.all())
    duplicate_month = bool(dates.dt.to_period("M").duplicated().any())
    defaults["macro_rows"] = str(int(frame.shape[0]))
    defaults["macro_month_end_only"] = str(month_end_only)
    defaults["macro_duplicate_month"] = str(duplicate_month)
    defaults["macro_frequency_ok"] = str(month_end_only and (not duplicate_month))
    return defaults


def _systematic_risk_summary_stats(path: Path) -> dict[str, str]:
    """Summarize systematic risk metrics observed in risk_summary."""
    defaults = {
        "systematic_metrics_present": "False",
        "systematic_beta_observed": "unknown",
        "systematic_alpha_observed": "unknown",
        "systematic_beta_rolling_60m_latest_observed": "unknown",
        "systematic_beta_rolling_60m_valid_points_observed": "unknown",
        "systematic_beta_rolling_60m_fallback_observed": "unknown",
        "systematic_beta_rolling_252m_latest_observed": "unknown",
        "systematic_beta_rolling_252m_valid_points_observed": "unknown",
        "systematic_beta_rolling_252m_fallback_observed": "unknown",
    }
    frame = _safe_read_csv(path)
    required_columns = {"risk_dimension", "metric", "value"}
    if frame.empty or not required_columns.issubset(frame.columns):
        return defaults

    systematic = frame.loc[
        frame["risk_dimension"].astype(str).str.strip().str.lower() == "systematic",
    ].copy()
    if systematic.empty:
        return defaults

    def _metric_float(metric: str) -> str:
        values = pd.to_numeric(
            systematic.loc[systematic["metric"] == metric, "value"],
            errors="coerce",
        ).dropna()
        if values.empty:
            return "unknown"
        return str(float(values.iloc[0]))

    def _metric_int(metric: str) -> str:
        values = pd.to_numeric(
            systematic.loc[systematic["metric"] == metric, "value"],
            errors="coerce",
        ).dropna()
        if values.empty:
            return "unknown"
        return str(int(values.iloc[0]))

    def _metric_bool(metric: str) -> str:
        values = pd.to_numeric(
            systematic.loc[systematic["metric"] == metric, "value"],
            errors="coerce",
        ).dropna()
        if values.empty:
            return "unknown"
        return str(bool(values.iloc[0]))

    defaults["systematic_beta_observed"] = _metric_float("beta")
    defaults["systematic_alpha_observed"] = _metric_float("alpha")
    defaults["systematic_beta_rolling_60m_latest_observed"] = _metric_float(
        "beta_rolling_60m_latest",
    )
    defaults["systematic_beta_rolling_60m_valid_points_observed"] = _metric_int(
        "beta_rolling_60m_valid_points",
    )
    defaults["systematic_beta_rolling_60m_fallback_observed"] = _metric_bool(
        "beta_rolling_60m_fallback",
    )
    defaults["systematic_beta_rolling_252m_latest_observed"] = _metric_float(
        "beta_rolling_252m_latest",
    )
    defaults["systematic_beta_rolling_252m_valid_points_observed"] = _metric_int(
        "beta_rolling_252m_valid_points",
    )
    defaults["systematic_beta_rolling_252m_fallback_observed"] = _metric_bool(
        "beta_rolling_252m_fallback",
    )

    required_metrics = {
        "systematic_beta_observed",
        "systematic_alpha_observed",
        "systematic_beta_rolling_60m_latest_observed",
        "systematic_beta_rolling_60m_valid_points_observed",
        "systematic_beta_rolling_60m_fallback_observed",
        "systematic_beta_rolling_252m_latest_observed",
        "systematic_beta_rolling_252m_valid_points_observed",
        "systematic_beta_rolling_252m_fallback_observed",
    }
    defaults["systematic_metrics_present"] = str(
        all(defaults[key] != "unknown" for key in required_metrics),
    )
    return defaults


def build_diagnostics_table(
    risk_summary_path: Path,
    scorecard_path: Path,
    nvda_raw_path: Path,
    market_raw_path: Path,
    rolling_vol_path: Path,
    var_es_hist_path: Path,
    var_exceedances_hist_path: Path,
    vol_window_days: int,
    var_backtest_hist_path: Path | None = None,
    provenance_path: Path | None = None,
    panel_daily_path: Path | None = None,
    macro_raw_path: Path | None = None,
) -> pd.DataFrame:
    """Create a key-value diagnostics table from current pipeline artifacts."""
    scorecard, scorecard_parse_error = _load_scorecard(scorecard_path)
    provenance = _provenance_stats(
        provenance_path,
        nvda_raw_path=nvda_raw_path,
        market_raw_path=market_raw_path,
    )
    nvda_start, nvda_end = _safe_date_bounds(nvda_raw_path)
    market_start, market_end = _safe_date_bounds(market_raw_path)
    rolling_rows, rolling_valid_points, rolling_fallback = _rolling_vol_stats(rolling_vol_path)
    data_source = _combined_data_source(nvda_raw_path, market_raw_path)
    historical_var_es = _historical_var_es_stats(var_es_hist_path)
    garch_var_es_parameters = _garch_var_es_parameter_stats(var_es_hist_path)
    historical_exceedance = _historical_exceedance_stats(var_exceedances_hist_path)
    historical_backtest = _historical_backtest_stats(var_backtest_hist_path)
    garch_backtest_upstream = _garch_backtest_upstream_stats(var_backtest_hist_path)
    garch_backtest_quality = _garch_backtest_quality_stats(var_backtest_hist_path)
    backtest_method_quality = _backtest_method_quality_stats(var_backtest_hist_path)
    garch_reject_rates_observed = _backtest_method_reject_rates_observed(var_backtest_hist_path)
    panel_daily_qc = _panel_daily_qc_stats(panel_daily_path)
    macro_frequency = _macro_frequency_stats(macro_raw_path)
    systematic_summary = _systematic_risk_summary_stats(risk_summary_path)
    placeholder_source_detected = _placeholder_source_detected(nvda_raw_path, market_raw_path)
    panel_start, panel_end = _safe_parquet_date_bounds(panel_daily_path) if panel_daily_path else ("", "")
    panel_window_matches_target = _window_match(panel_start, panel_end)
    scorecard_window_recorded = _normalize_tristate(scorecard.get("window_matches_target", "unknown"))
    scorecard_window_consistent = "unknown"
    if scorecard_window_recorded in {"true", "false"} and panel_window_matches_target.lower() in {"true", "false"}:
        scorecard_window_consistent = str(scorecard_window_recorded == panel_window_matches_target.lower())
    scorecard_status_recorded = str(scorecard.get("status", "unknown")).strip().lower()
    scorecard_status_expected = _expected_scorecard_status(
        panel_window_matches_target=panel_window_matches_target,
        rolling_vol_fallback=rolling_fallback,
        backtest_fallback_events=scorecard.get("var_backtest_fallback_events", 0),
    )
    scorecard_status_consistent = "unknown"
    if scorecard_status_recorded in {"ok", "warn", "fail"} and scorecard_status_expected in {
        "ok",
        "warn",
        "fail",
    }:
        scorecard_status_consistent = str(scorecard_status_recorded == scorecard_status_expected)
    scorecard_reason_recorded = str(scorecard.get("status_reason", "unknown")).strip().lower()
    scorecard_reason_expected = _expected_scorecard_status_reason(
        panel_window_matches_target=panel_window_matches_target,
        rolling_vol_fallback=rolling_fallback,
        backtest_fallback_events=scorecard.get("var_backtest_fallback_events", 0),
    )
    scorecard_reason_consistent = "unknown"
    if scorecard_reason_recorded in {
        "window_not_target",
        "rolling_vol_fallback",
        "backtest_fallbacks",
        "none",
    } and scorecard_reason_expected in {
        "window_not_target",
        "rolling_vol_fallback",
        "backtest_fallbacks",
        "none",
    }:
        scorecard_reason_consistent = str(scorecard_reason_recorded == scorecard_reason_expected)
    garch_rows_recorded = scorecard.get("var_backtest_garch_rows", 0)
    garch_fallback_events_recorded = scorecard.get("var_backtest_garch_fallback_events", 0)
    garch_status_ok_count_recorded = scorecard.get("var_backtest_garch_status_ok_count", 0)
    historical_rows_consistent = _int_value_consistent(
        scorecard.get("var_backtest_historical_rows", 0),
        backtest_method_quality["var_backtest_historical_rows_observed"],
    )
    historical_fallback_events_consistent = _int_value_consistent(
        scorecard.get("var_backtest_historical_fallback_events", 0),
        backtest_method_quality["var_backtest_historical_fallback_events_observed"],
    )
    parametric_rows_consistent = _int_value_consistent(
        scorecard.get("var_backtest_parametric_rows", 0),
        backtest_method_quality["var_backtest_parametric_rows_observed"],
    )
    parametric_fallback_events_consistent = _int_value_consistent(
        scorecard.get("var_backtest_parametric_fallback_events", 0),
        backtest_method_quality["var_backtest_parametric_fallback_events_observed"],
    )
    other_rows_consistent = _int_value_consistent(
        scorecard.get("var_backtest_other_rows", 0),
        backtest_method_quality["var_backtest_other_rows_observed"],
    )
    other_fallback_events_consistent = _int_value_consistent(
        scorecard.get("var_backtest_other_fallback_events", 0),
        backtest_method_quality["var_backtest_other_fallback_events_observed"],
    )
    garch_rows_consistent = _int_value_consistent(
        garch_rows_recorded,
        garch_backtest_quality["var_backtest_garch_rows_observed"],
    )
    garch_fallback_events_consistent = _int_value_consistent(
        garch_fallback_events_recorded,
        garch_backtest_quality["var_backtest_garch_fallback_events_observed"],
    )
    garch_status_ok_count_consistent = _int_value_consistent(
        garch_status_ok_count_recorded,
        garch_backtest_quality["var_backtest_garch_status_ok_count_observed"],
    )
    historical_kupiec_reject_rate_consistent = _float_value_consistent(
        scorecard.get("historical_kupiec_reject_rate_5pct", float("nan")),
        garch_reject_rates_observed["historical_kupiec_reject_rate_5pct_observed"],
    )
    historical_christoffersen_reject_rate_consistent = _float_value_consistent(
        scorecard.get("historical_christoffersen_reject_rate_5pct", float("nan")),
        garch_reject_rates_observed["historical_christoffersen_reject_rate_5pct_observed"],
    )
    parametric_kupiec_reject_rate_consistent = _float_value_consistent(
        scorecard.get("parametric_kupiec_reject_rate_5pct", float("nan")),
        garch_reject_rates_observed["parametric_kupiec_reject_rate_5pct_observed"],
    )
    parametric_christoffersen_reject_rate_consistent = _float_value_consistent(
        scorecard.get("parametric_christoffersen_reject_rate_5pct", float("nan")),
        garch_reject_rates_observed["parametric_christoffersen_reject_rate_5pct_observed"],
    )
    garch_kupiec_reject_rate_consistent = _float_value_consistent(
        scorecard.get("garch_kupiec_reject_rate_5pct", float("nan")),
        garch_reject_rates_observed["garch_kupiec_reject_rate_5pct_observed"],
    )
    garch_christoffersen_reject_rate_consistent = _float_value_consistent(
        scorecard.get("garch_christoffersen_reject_rate_5pct", float("nan")),
        garch_reject_rates_observed["garch_christoffersen_reject_rate_5pct_observed"],
    )
    systematic_beta_consistent = _float_value_consistent(
        scorecard.get("systematic_beta", float("nan")),
        systematic_summary["systematic_beta_observed"],
    )
    systematic_alpha_consistent = _float_value_consistent(
        scorecard.get("systematic_alpha", float("nan")),
        systematic_summary["systematic_alpha_observed"],
    )
    systematic_beta_rolling_60m_latest_consistent = _float_value_consistent(
        scorecard.get("systematic_beta_rolling_60m_latest", float("nan")),
        systematic_summary["systematic_beta_rolling_60m_latest_observed"],
    )
    systematic_beta_rolling_60m_valid_points_consistent = _int_value_consistent(
        scorecard.get("systematic_beta_rolling_60m_valid_points", 0),
        systematic_summary["systematic_beta_rolling_60m_valid_points_observed"],
    )
    systematic_beta_rolling_60m_fallback_consistent = _bool_value_consistent(
        scorecard.get("systematic_beta_rolling_60m_fallback", "unknown"),
        systematic_summary["systematic_beta_rolling_60m_fallback_observed"],
    )
    systematic_beta_rolling_252m_latest_consistent = _float_value_consistent(
        scorecard.get("systematic_beta_rolling_252m_latest", float("nan")),
        systematic_summary["systematic_beta_rolling_252m_latest_observed"],
    )
    systematic_beta_rolling_252m_valid_points_consistent = _int_value_consistent(
        scorecard.get("systematic_beta_rolling_252m_valid_points", 0),
        systematic_summary["systematic_beta_rolling_252m_valid_points_observed"],
    )
    systematic_beta_rolling_252m_fallback_consistent = _bool_value_consistent(
        scorecard.get("systematic_beta_rolling_252m_fallback", "unknown"),
        systematic_summary["systematic_beta_rolling_252m_fallback_observed"],
    )

    diagnostics = [
        {"check": "risk_summary_exists", "value": str(risk_summary_path.exists())},
        {"check": "scorecard_exists", "value": str(scorecard_path.exists())},
        {"check": "scorecard_parse_error", "value": str(scorecard_parse_error)},
        {"check": "provenance_exists", "value": provenance["provenance_exists"]},
        {"check": "provenance_parse_error", "value": provenance["provenance_parse_error"]},
        {"check": "provenance_ticker_match", "value": provenance["provenance_ticker_match"]},
        {"check": "provenance_sample_match", "value": provenance["provenance_sample_match"]},
        {
            "check": "provenance_panel_sha_match",
            "value": provenance["provenance_panel_sha_match"],
        },
        {
            "check": "provenance_nvda_raw_sha_match",
            "value": provenance["provenance_nvda_raw_sha_match"],
        },
        {
            "check": "provenance_market_raw_sha_match",
            "value": provenance["provenance_market_raw_sha_match"],
        },
        {"check": "provenance_source_match", "value": provenance["provenance_source_match"]},
        {"check": "panel_daily_exists", "value": panel_daily_qc["panel_daily_exists"]},
        {"check": "panel_daily_rows", "value": panel_daily_qc["panel_daily_rows"]},
        {
            "check": "panel_daily_required_cols_present",
            "value": panel_daily_qc["panel_daily_required_cols_present"],
        },
        {
            "check": "panel_daily_date_monotonic",
            "value": panel_daily_qc["panel_daily_date_monotonic"],
        },
        {
            "check": "panel_daily_duplicate_dates",
            "value": panel_daily_qc["panel_daily_duplicate_dates"],
        },
        {
            "check": "panel_daily_core_missing_rate",
            "value": panel_daily_qc["panel_daily_core_missing_rate"],
        },
        {"check": "nvda_raw_rows", "value": str(_safe_row_count(nvda_raw_path))},
        {"check": "market_raw_rows", "value": str(_safe_row_count(market_raw_path))},
        {"check": "nvda_sample_start", "value": nvda_start},
        {"check": "nvda_sample_end", "value": nvda_end},
        {"check": "market_sample_start", "value": market_start},
        {"check": "market_sample_end", "value": market_end},
        {"check": "sample_target_start", "value": SAMPLE_START_DATE},
        {"check": "sample_target_end", "value": SAMPLE_END_DATE},
        {
            "check": "nvda_window_matches_target",
            "value": _window_match(nvda_start, nvda_end),
        },
        {
            "check": "market_window_matches_target",
            "value": _window_match(market_start, market_end),
        },
        {
            "check": "panel_window_matches_target",
            "value": panel_window_matches_target,
        },
        {
            "check": "scorecard_sample_target_start_recorded",
            "value": str(scorecard.get("sample_target_start", SAMPLE_START_DATE)),
        },
        {
            "check": "scorecard_sample_target_end_recorded",
            "value": str(scorecard.get("sample_target_end", SAMPLE_END_DATE)),
        },
        {
            "check": "scorecard_window_matches_target_recorded",
            "value": scorecard_window_recorded,
        },
        {
            "check": "scorecard_window_matches_target_consistent",
            "value": scorecard_window_consistent,
        },
        {
            "check": "scorecard_status_recorded",
            "value": scorecard_status_recorded,
        },
        {
            "check": "scorecard_status_expected",
            "value": scorecard_status_expected,
        },
        {
            "check": "scorecard_status_consistent",
            "value": scorecard_status_consistent,
        },
        {
            "check": "scorecard_status_reason_recorded",
            "value": scorecard_reason_recorded,
        },
        {
            "check": "scorecard_status_reason_expected",
            "value": scorecard_reason_expected,
        },
        {
            "check": "scorecard_status_reason_consistent",
            "value": scorecard_reason_consistent,
        },
        {
            "check": "scorecard_metric_count",
            "value": str(scorecard.get("metric_count", 0)),
        },
        {
            "check": "systematic_metrics_present",
            "value": systematic_summary["systematic_metrics_present"],
        },
        {
            "check": "systematic_beta_recorded",
            "value": str(scorecard.get("systematic_beta", float("nan"))),
        },
        {
            "check": "systematic_beta_observed",
            "value": systematic_summary["systematic_beta_observed"],
        },
        {
            "check": "systematic_beta_consistent",
            "value": systematic_beta_consistent,
        },
        {
            "check": "systematic_alpha_recorded",
            "value": str(scorecard.get("systematic_alpha", float("nan"))),
        },
        {
            "check": "systematic_alpha_observed",
            "value": systematic_summary["systematic_alpha_observed"],
        },
        {
            "check": "systematic_alpha_consistent",
            "value": systematic_alpha_consistent,
        },
        {
            "check": "systematic_beta_rolling_60m_latest_recorded",
            "value": str(scorecard.get("systematic_beta_rolling_60m_latest", float("nan"))),
        },
        {
            "check": "systematic_beta_rolling_60m_latest_observed",
            "value": systematic_summary["systematic_beta_rolling_60m_latest_observed"],
        },
        {
            "check": "systematic_beta_rolling_60m_latest_consistent",
            "value": systematic_beta_rolling_60m_latest_consistent,
        },
        {
            "check": "systematic_beta_rolling_60m_valid_points_recorded",
            "value": str(scorecard.get("systematic_beta_rolling_60m_valid_points", 0)),
        },
        {
            "check": "systematic_beta_rolling_60m_valid_points_observed",
            "value": systematic_summary["systematic_beta_rolling_60m_valid_points_observed"],
        },
        {
            "check": "systematic_beta_rolling_60m_valid_points_consistent",
            "value": systematic_beta_rolling_60m_valid_points_consistent,
        },
        {
            "check": "systematic_beta_rolling_60m_fallback_recorded",
            "value": str(scorecard.get("systematic_beta_rolling_60m_fallback", "unknown")),
        },
        {
            "check": "systematic_beta_rolling_60m_fallback_observed",
            "value": systematic_summary["systematic_beta_rolling_60m_fallback_observed"],
        },
        {
            "check": "systematic_beta_rolling_60m_fallback_consistent",
            "value": systematic_beta_rolling_60m_fallback_consistent,
        },
        {
            "check": "systematic_beta_rolling_252m_latest_recorded",
            "value": str(scorecard.get("systematic_beta_rolling_252m_latest", float("nan"))),
        },
        {
            "check": "systematic_beta_rolling_252m_latest_observed",
            "value": systematic_summary["systematic_beta_rolling_252m_latest_observed"],
        },
        {
            "check": "systematic_beta_rolling_252m_latest_consistent",
            "value": systematic_beta_rolling_252m_latest_consistent,
        },
        {
            "check": "systematic_beta_rolling_252m_valid_points_recorded",
            "value": str(scorecard.get("systematic_beta_rolling_252m_valid_points", 0)),
        },
        {
            "check": "systematic_beta_rolling_252m_valid_points_observed",
            "value": systematic_summary["systematic_beta_rolling_252m_valid_points_observed"],
        },
        {
            "check": "systematic_beta_rolling_252m_valid_points_consistent",
            "value": systematic_beta_rolling_252m_valid_points_consistent,
        },
        {
            "check": "systematic_beta_rolling_252m_fallback_recorded",
            "value": str(scorecard.get("systematic_beta_rolling_252m_fallback", "unknown")),
        },
        {
            "check": "systematic_beta_rolling_252m_fallback_observed",
            "value": systematic_summary["systematic_beta_rolling_252m_fallback_observed"],
        },
        {
            "check": "systematic_beta_rolling_252m_fallback_consistent",
            "value": systematic_beta_rolling_252m_fallback_consistent,
        },
        {
            "check": "fallback_events_recorded",
            "value": str(scorecard.get("fallback_events", 0)),
        },
        {
            "check": "var_backtest_fallback_events_recorded",
            "value": str(scorecard.get("var_backtest_fallback_events", 0)),
        },
        {
            "check": "var_backtest_historical_rows_recorded",
            "value": str(scorecard.get("var_backtest_historical_rows", 0)),
        },
        {
            "check": "var_backtest_historical_fallback_events_recorded",
            "value": str(scorecard.get("var_backtest_historical_fallback_events", 0)),
        },
        {
            "check": "var_backtest_historical_rows_observed",
            "value": backtest_method_quality["var_backtest_historical_rows_observed"],
        },
        {
            "check": "var_backtest_historical_fallback_events_observed",
            "value": backtest_method_quality["var_backtest_historical_fallback_events_observed"],
        },
        {
            "check": "var_backtest_historical_rows_consistent",
            "value": historical_rows_consistent,
        },
        {
            "check": "var_backtest_historical_fallback_events_consistent",
            "value": historical_fallback_events_consistent,
        },
        {
            "check": "var_backtest_parametric_rows_recorded",
            "value": str(scorecard.get("var_backtest_parametric_rows", 0)),
        },
        {
            "check": "var_backtest_parametric_fallback_events_recorded",
            "value": str(scorecard.get("var_backtest_parametric_fallback_events", 0)),
        },
        {
            "check": "var_backtest_parametric_rows_observed",
            "value": backtest_method_quality["var_backtest_parametric_rows_observed"],
        },
        {
            "check": "var_backtest_parametric_fallback_events_observed",
            "value": backtest_method_quality["var_backtest_parametric_fallback_events_observed"],
        },
        {
            "check": "var_backtest_parametric_rows_consistent",
            "value": parametric_rows_consistent,
        },
        {
            "check": "var_backtest_parametric_fallback_events_consistent",
            "value": parametric_fallback_events_consistent,
        },
        {
            "check": "var_backtest_other_rows_recorded",
            "value": str(scorecard.get("var_backtest_other_rows", 0)),
        },
        {
            "check": "var_backtest_other_fallback_events_recorded",
            "value": str(scorecard.get("var_backtest_other_fallback_events", 0)),
        },
        {
            "check": "var_backtest_other_rows_observed",
            "value": backtest_method_quality["var_backtest_other_rows_observed"],
        },
        {
            "check": "var_backtest_other_fallback_events_observed",
            "value": backtest_method_quality["var_backtest_other_fallback_events_observed"],
        },
        {
            "check": "var_backtest_other_rows_consistent",
            "value": other_rows_consistent,
        },
        {
            "check": "var_backtest_other_fallback_events_consistent",
            "value": other_fallback_events_consistent,
        },
        {
            "check": "var_backtest_garch_rows_recorded",
            "value": str(scorecard.get("var_backtest_garch_rows", 0)),
        },
        {
            "check": "var_backtest_garch_fallback_events_recorded",
            "value": str(scorecard.get("var_backtest_garch_fallback_events", 0)),
        },
        {
            "check": "var_backtest_garch_status_ok_count_recorded",
            "value": str(scorecard.get("var_backtest_garch_status_ok_count", 0)),
        },
        {
            "check": "var_backtest_garch_rows_observed",
            "value": garch_backtest_quality["var_backtest_garch_rows_observed"],
        },
        {
            "check": "var_backtest_garch_fallback_events_observed",
            "value": garch_backtest_quality["var_backtest_garch_fallback_events_observed"],
        },
        {
            "check": "var_backtest_garch_status_ok_count_observed",
            "value": garch_backtest_quality["var_backtest_garch_status_ok_count_observed"],
        },
        {
            "check": "var_backtest_garch_rows_consistent",
            "value": garch_rows_consistent,
        },
        {
            "check": "var_backtest_garch_fallback_events_consistent",
            "value": garch_fallback_events_consistent,
        },
        {
            "check": "var_backtest_garch_status_ok_count_consistent",
            "value": garch_status_ok_count_consistent,
        },
        {
            "check": "historical_kupiec_reject_rate_5pct_recorded",
            "value": str(scorecard.get("historical_kupiec_reject_rate_5pct", float("nan"))),
        },
        {
            "check": "historical_christoffersen_reject_rate_5pct_recorded",
            "value": str(
                scorecard.get(
                    "historical_christoffersen_reject_rate_5pct",
                    float("nan"),
                ),
            ),
        },
        {
            "check": "parametric_kupiec_reject_rate_5pct_recorded",
            "value": str(scorecard.get("parametric_kupiec_reject_rate_5pct", float("nan"))),
        },
        {
            "check": "parametric_christoffersen_reject_rate_5pct_recorded",
            "value": str(
                scorecard.get(
                    "parametric_christoffersen_reject_rate_5pct",
                    float("nan"),
                ),
            ),
        },
        {
            "check": "garch_kupiec_reject_rate_5pct_recorded",
            "value": str(scorecard.get("garch_kupiec_reject_rate_5pct", float("nan"))),
        },
        {
            "check": "garch_christoffersen_reject_rate_5pct_recorded",
            "value": str(
                scorecard.get(
                    "garch_christoffersen_reject_rate_5pct",
                    float("nan"),
                ),
            ),
        },
        {
            "check": "historical_kupiec_reject_rate_5pct_observed",
            "value": garch_reject_rates_observed["historical_kupiec_reject_rate_5pct_observed"],
        },
        {
            "check": "historical_christoffersen_reject_rate_5pct_observed",
            "value": garch_reject_rates_observed[
                "historical_christoffersen_reject_rate_5pct_observed"
            ],
        },
        {
            "check": "parametric_kupiec_reject_rate_5pct_observed",
            "value": garch_reject_rates_observed["parametric_kupiec_reject_rate_5pct_observed"],
        },
        {
            "check": "parametric_christoffersen_reject_rate_5pct_observed",
            "value": garch_reject_rates_observed[
                "parametric_christoffersen_reject_rate_5pct_observed"
            ],
        },
        {
            "check": "garch_kupiec_reject_rate_5pct_observed",
            "value": garch_reject_rates_observed["garch_kupiec_reject_rate_5pct_observed"],
        },
        {
            "check": "garch_christoffersen_reject_rate_5pct_observed",
            "value": garch_reject_rates_observed[
                "garch_christoffersen_reject_rate_5pct_observed"
            ],
        },
        {
            "check": "historical_kupiec_reject_rate_5pct_consistent",
            "value": historical_kupiec_reject_rate_consistent,
        },
        {
            "check": "historical_christoffersen_reject_rate_5pct_consistent",
            "value": historical_christoffersen_reject_rate_consistent,
        },
        {
            "check": "parametric_kupiec_reject_rate_5pct_consistent",
            "value": parametric_kupiec_reject_rate_consistent,
        },
        {
            "check": "parametric_christoffersen_reject_rate_5pct_consistent",
            "value": parametric_christoffersen_reject_rate_consistent,
        },
        {
            "check": "garch_kupiec_reject_rate_5pct_consistent",
            "value": garch_kupiec_reject_rate_consistent,
        },
        {
            "check": "garch_christoffersen_reject_rate_5pct_consistent",
            "value": garch_christoffersen_reject_rate_consistent,
        },
        {
            "check": "fallback_reason_short_sample_count",
            "value": str(scorecard.get("var_backtest_fallback_reason_short_sample_count", 0)),
        },
        {
            "check": "fallback_reason_nan_var_count",
            "value": str(scorecard.get("var_backtest_fallback_reason_nan_var_count", 0)),
        },
        {
            "check": "fallback_reason_invalid_input_count",
            "value": str(scorecard.get("var_backtest_fallback_reason_invalid_input_count", 0)),
        },
        {
            "check": "fallback_reason_exception_count",
            "value": str(scorecard.get("var_backtest_fallback_reason_exception_count", 0)),
        },
        {
            "check": "fallback_reason_upstream_fallback_count",
            "value": str(
                scorecard.get("var_backtest_fallback_reason_upstream_fallback_count", 0),
            ),
        },
        {
            "check": "fallback_reason_unknown_count",
            "value": str(scorecard.get("var_backtest_fallback_reason_unknown_count", 0)),
        },
        {
            "check": "data_source",
            "value": data_source,
        },
        {
            "check": "placeholder_source_detected",
            "value": placeholder_source_detected,
        },
        {"check": "rolling_vol_window_days", "value": str(vol_window_days)},
        {"check": "rolling_vol_total_rows", "value": str(rolling_rows)},
        {"check": "rolling_vol_valid_points", "value": str(rolling_valid_points)},
        {"check": "rolling_vol_fallback_used", "value": rolling_fallback},
        {"check": "hist_var_es_exists", "value": historical_var_es["hist_var_es_exists"]},
        {"check": "hist_var_es_method_count", "value": historical_var_es["hist_var_es_method_count"]},
        {"check": "parametric_var_es_present", "value": historical_var_es["parametric_var_es_present"]},
        {"check": "garch_var_es_present", "value": historical_var_es["garch_var_es_present"]},
        {
            "check": "hist_var_es_alpha_count",
            "value": historical_var_es["hist_var_es_alpha_count"],
        },
        {
            "check": "hist_var_es_status_ok_count",
            "value": historical_var_es["hist_var_es_status_ok_count"],
        },
        {"check": "hist_var_95", "value": historical_var_es["hist_var_95"]},
        {"check": "hist_es_95", "value": historical_var_es["hist_es_95"]},
        {"check": "hist_var_99", "value": historical_var_es["hist_var_99"]},
        {"check": "hist_es_99", "value": historical_var_es["hist_es_99"]},
        {"check": "param_var_95", "value": historical_var_es["param_var_95"]},
        {"check": "param_es_95", "value": historical_var_es["param_es_95"]},
        {"check": "param_var_99", "value": historical_var_es["param_var_99"]},
        {"check": "param_es_99", "value": historical_var_es["param_es_99"]},
        {
            "check": "garch_var_es_nu_finite",
            "value": garch_var_es_parameters["garch_var_es_nu_finite"],
        },
        {
            "check": "garch_var_es_nu_gt4",
            "value": garch_var_es_parameters["garch_var_es_nu_gt4"],
        },
        {
            "check": "garch_var_es_sigma_next_positive",
            "value": garch_var_es_parameters["garch_var_es_sigma_next_positive"],
        },
        {"check": "macro_data_exists", "value": macro_frequency["macro_data_exists"]},
        {"check": "macro_rows", "value": macro_frequency["macro_rows"]},
        {"check": "macro_month_end_only", "value": macro_frequency["macro_month_end_only"]},
        {"check": "macro_duplicate_month", "value": macro_frequency["macro_duplicate_month"]},
        {"check": "macro_frequency_ok", "value": macro_frequency["macro_frequency_ok"]},
        {"check": "hist_exceed_exists", "value": historical_exceedance["hist_exceed_exists"]},
        {
            "check": "hist_exceed_method_count",
            "value": historical_exceedance["hist_exceed_method_count"],
        },
        {
            "check": "parametric_exceed_present",
            "value": historical_exceedance["parametric_exceed_present"],
        },
        {
            "check": "garch_exceed_present",
            "value": historical_exceedance["garch_exceed_present"],
        },
        {
            "check": "hist_exceed_status_ok_count",
            "value": historical_exceedance["hist_exceed_status_ok_count"],
        },
        {"check": "hist_exceed_95_count", "value": historical_exceedance["hist_exceed_95_count"]},
        {"check": "hist_exceed_95_rate", "value": historical_exceedance["hist_exceed_95_rate"]},
        {"check": "hist_exceed_95_gap", "value": historical_exceedance["hist_exceed_95_gap"]},
        {"check": "hist_exceed_99_count", "value": historical_exceedance["hist_exceed_99_count"]},
        {"check": "hist_exceed_99_rate", "value": historical_exceedance["hist_exceed_99_rate"]},
        {"check": "hist_exceed_99_gap", "value": historical_exceedance["hist_exceed_99_gap"]},
        {"check": "hist_backtest_exists", "value": historical_backtest["hist_backtest_exists"]},
        {
            "check": "hist_backtest_method_count",
            "value": historical_backtest["hist_backtest_method_count"],
        },
        {
            "check": "parametric_backtest_present",
            "value": historical_backtest["parametric_backtest_present"],
        },
        {
            "check": "garch_backtest_present",
            "value": historical_backtest["garch_backtest_present"],
        },
        {
            "check": "hist_backtest_alpha_count",
            "value": historical_backtest["hist_backtest_alpha_count"],
        },
        {
            "check": "hist_backtest_status_ok_count",
            "value": historical_backtest["hist_backtest_status_ok_count"],
        },
        {"check": "hist_kupiec_lr_95", "value": historical_backtest["hist_kupiec_lr_95"]},
        {"check": "hist_kupiec_lr_99", "value": historical_backtest["hist_kupiec_lr_99"]},
        {"check": "hist_kupiec_pvalue_95", "value": historical_backtest["hist_kupiec_pvalue_95"]},
        {"check": "hist_kupiec_pvalue_99", "value": historical_backtest["hist_kupiec_pvalue_99"]},
        {
            "check": "hist_christoffersen_lr_95",
            "value": historical_backtest["hist_christoffersen_lr_95"],
        },
        {
            "check": "hist_christoffersen_lr_99",
            "value": historical_backtest["hist_christoffersen_lr_99"],
        },
        {
            "check": "hist_christoffersen_pvalue_95",
            "value": historical_backtest["hist_christoffersen_pvalue_95"],
        },
        {
            "check": "hist_christoffersen_pvalue_99",
            "value": historical_backtest["hist_christoffersen_pvalue_99"],
        },
        {
            "check": "param_kupiec_lr_95",
            "value": historical_backtest["param_kupiec_lr_95"],
        },
        {
            "check": "param_kupiec_lr_99",
            "value": historical_backtest["param_kupiec_lr_99"],
        },
        {
            "check": "param_kupiec_pvalue_95",
            "value": historical_backtest["param_kupiec_pvalue_95"],
        },
        {
            "check": "param_kupiec_pvalue_99",
            "value": historical_backtest["param_kupiec_pvalue_99"],
        },
        {
            "check": "param_christoffersen_lr_95",
            "value": historical_backtest["param_christoffersen_lr_95"],
        },
        {
            "check": "param_christoffersen_lr_99",
            "value": historical_backtest["param_christoffersen_lr_99"],
        },
        {
            "check": "param_christoffersen_pvalue_95",
            "value": historical_backtest["param_christoffersen_pvalue_95"],
        },
        {
            "check": "param_christoffersen_pvalue_99",
            "value": historical_backtest["param_christoffersen_pvalue_99"],
        },
        {
            "check": "garch_kupiec_lr_95",
            "value": historical_backtest["garch_kupiec_lr_95"],
        },
        {
            "check": "garch_kupiec_lr_99",
            "value": historical_backtest["garch_kupiec_lr_99"],
        },
        {
            "check": "garch_kupiec_pvalue_95",
            "value": historical_backtest["garch_kupiec_pvalue_95"],
        },
        {
            "check": "garch_kupiec_pvalue_99",
            "value": historical_backtest["garch_kupiec_pvalue_99"],
        },
        {
            "check": "garch_christoffersen_lr_95",
            "value": historical_backtest["garch_christoffersen_lr_95"],
        },
        {
            "check": "garch_christoffersen_lr_99",
            "value": historical_backtest["garch_christoffersen_lr_99"],
        },
        {
            "check": "garch_christoffersen_pvalue_95",
            "value": historical_backtest["garch_christoffersen_pvalue_95"],
        },
        {
            "check": "garch_christoffersen_pvalue_99",
            "value": historical_backtest["garch_christoffersen_pvalue_99"],
        },
        {
            "check": "garch_backtest_upstream_exists",
            "value": garch_backtest_upstream["garch_backtest_upstream_exists"],
        },
        {
            "check": "garch_backtest_upstream_converged_present",
            "value": garch_backtest_upstream["garch_backtest_upstream_converged_present"],
        },
        {
            "check": "garch_backtest_upstream_converged_true_count",
            "value": garch_backtest_upstream["garch_backtest_upstream_converged_true_count"],
        },
        {
            "check": "garch_backtest_upstream_nu_finite",
            "value": garch_backtest_upstream["garch_backtest_upstream_nu_finite"],
        },
        {
            "check": "garch_backtest_upstream_sigma_positive",
            "value": garch_backtest_upstream["garch_backtest_upstream_sigma_positive"],
        },
    ]
    diagnostics_map = {
        str(item["check"]): str(item["value"])
        for item in diagnostics
    }
    gate_values = [diagnostics_map.get(check, "unknown") for check in _QUALITY_GATE_CHECKS]
    quality_pass_count = sum(value == "True" for value in gate_values)
    quality_fail_count = sum(value == "False" for value in gate_values)
    quality_unknown_count = len(gate_values) - quality_pass_count - quality_fail_count
    quality_failed_checks = [
        check
        for check, value in zip(_QUALITY_GATE_CHECKS, gate_values, strict=True)
        if value == "False"
    ]
    quality_failed_checks_value = ",".join(quality_failed_checks) if quality_failed_checks else "none"
    diagnostics.extend(
        [
            {"check": "quality_gates_pass_count", "value": str(quality_pass_count)},
            {"check": "quality_gates_fail_count", "value": str(quality_fail_count)},
            {"check": "quality_gates_unknown_count", "value": str(quality_unknown_count)},
            {"check": "quality_gates_failed_checks", "value": quality_failed_checks_value},
            {
                "check": "quality_gates_all_pass",
                "value": str(quality_fail_count == 0 and quality_unknown_count == 0),
            },
        ],
    )
    return pd.DataFrame(diagnostics)


def render_diagnostics_markdown(diagnostics: pd.DataFrame) -> str:
    """Render diagnostics markdown with a quality-gate summary and full checks."""
    full_checks_markdown = diagnostics.to_markdown(index=False)
    required_columns = {"check", "value"}
    if diagnostics.empty or not required_columns.issubset(diagnostics.columns):
        return "\n".join(
            [
                "# Diagnostics",
                "",
                "## Full Checks",
                "",
                full_checks_markdown,
                "",
            ],
        )

    diagnostics_map = dict(zip(diagnostics["check"], diagnostics["value"], strict=True))
    quality_summary_checks = (
        "quality_gates_all_pass",
        "quality_gates_pass_count",
        "quality_gates_fail_count",
        "quality_gates_unknown_count",
        "quality_gates_failed_checks",
    )
    quality_rows = [
        {"check": check, "value": str(diagnostics_map.get(check, "missing"))}
        for check in quality_summary_checks
    ]
    quality_rows.extend(
        [
        {"check": check, "value": str(diagnostics_map.get(check, "missing"))}
        for check in _QUALITY_GATE_CHECKS
        ],
    )
    quality_gates = pd.DataFrame(quality_rows)
    return "\n".join(
        [
            "# Diagnostics",
            "",
            "## Quality Gates",
            "",
            quality_gates.to_markdown(index=False),
            "",
            "## Full Checks",
            "",
            full_checks_markdown,
            "",
        ],
    )

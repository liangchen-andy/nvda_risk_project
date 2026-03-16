"""Tasks for creating a reproducibility scorecard."""

import json
from pathlib import Path

import pandas as pd

from nvda_risk_project.config import (
    ANALYSIS_OUTPUT,
    CHECKS_OUTPUT,
    PANEL_DAILY_DATA,
    SAMPLE_END_DATE,
    SAMPLE_START_DATE,
    TABLES_OUTPUT,
)


_REASON_LABELS: tuple[str, ...] = (
    "short_sample",
    "nan_var",
    "invalid_input",
    "exception",
    "upstream_fallback",
    "unknown",
)


def _summary_metric(
    summary: pd.DataFrame,
    *,
    risk_dimension: str,
    metric: str,
    default: float,
) -> float:
    """Read a numeric metric from risk_summary with a deterministic fallback."""
    if not {"risk_dimension", "metric", "value"}.issubset(summary.columns):
        return float(default)
    mask = (summary["risk_dimension"] == risk_dimension) & (summary["metric"] == metric)
    if not mask.any():
        return float(default)
    values = pd.to_numeric(summary.loc[mask, "value"], errors="coerce").dropna()
    if values.empty:
        return float(default)
    return float(values.iloc[0])


def _normalize_method(method: str) -> str:
    """Map raw method labels to historical/parametric/other buckets."""
    normalized = method.strip().lower()
    if normalized == "historical":
        return "historical"
    if normalized == "parametric_normal":
        return "parametric"
    return "other"


def _rolling_vol_quality(rolling_vol_path: Path) -> tuple[int, bool]:
    """Return valid rolling-vol points and whether fallback is needed."""
    if not rolling_vol_path.exists():
        return 0, True
    try:
        rolling_vol = pd.read_csv(rolling_vol_path)
    except Exception:
        return 0, True
    if "rolling_vol" not in rolling_vol.columns:
        return 0, True
    valid_points = int(pd.to_numeric(rolling_vol["rolling_vol"], errors="coerce").notna().sum())
    return valid_points, valid_points == 0


def _normalize_reason(reason: str) -> str:
    """Map raw fallback reasons to a compact canonical set."""
    normalized = reason.strip().lower()
    if normalized.startswith("upstream_"):
        return "upstream_fallback"
    if normalized in _REASON_LABELS:
        return normalized
    return "unknown"


def _backtest_fallback_reason_counts(var_backtest_path: Path) -> dict[str, int]:
    """Count fallback reasons in historical VaR backtest artifacts."""
    counts = {reason: 0 for reason in _REASON_LABELS}
    if not var_backtest_path.exists():
        return counts
    try:
        backtest = pd.read_csv(var_backtest_path)
    except Exception:
        return counts
    if "status" not in backtest.columns:
        return counts

    status = backtest["status"].astype(str).str.lower()
    fallback_mask = status != "ok"
    if not fallback_mask.any():
        return counts

    if "fallback_reason" not in backtest.columns:
        counts["unknown"] = int(fallback_mask.sum())
        return counts

    reasons = backtest.loc[fallback_mask, "fallback_reason"].astype(str)
    for reason in reasons.tolist():
        counts[_normalize_reason(reason)] += 1
    return counts


def _backtest_method_quality_counts(var_backtest_path: Path) -> dict[str, int]:
    """Count backtest rows and fallback rows by method bucket."""
    counts = {
        "historical_rows": 0,
        "historical_fallback_events": 0,
        "parametric_rows": 0,
        "parametric_fallback_events": 0,
        "other_rows": 0,
        "other_fallback_events": 0,
    }
    if not var_backtest_path.exists():
        return counts
    try:
        backtest = pd.read_csv(var_backtest_path)
    except Exception:
        return counts
    if "status" not in backtest.columns:
        return counts

    status = backtest["status"].astype(str).str.lower()
    fallback_mask = status != "ok"
    if "method" not in backtest.columns:
        total = int(backtest.shape[0])
        counts["historical_rows"] = total
        counts["historical_fallback_events"] = int(fallback_mask.sum())
        return counts

    methods = backtest["method"].astype(str).map(_normalize_method)
    for bucket in ("historical", "parametric", "other"):
        bucket_mask = methods == bucket
        counts[f"{bucket}_rows"] = int(bucket_mask.sum())
        counts[f"{bucket}_fallback_events"] = int((fallback_mask & bucket_mask).sum())
    return counts


def _garch_backtest_quality_counts(var_backtest_path: Path) -> dict[str, int]:
    """Count garch_t backtest rows, fallbacks, and status=ok rows."""
    counts = {
        "garch_rows": 0,
        "garch_fallback_events": 0,
        "garch_status_ok_count": 0,
    }
    if not var_backtest_path.exists():
        return counts
    try:
        backtest = pd.read_csv(var_backtest_path)
    except Exception:
        return counts
    required = {"method", "status"}
    if not required.issubset(backtest.columns):
        return counts

    methods = backtest["method"].astype(str).str.strip().str.lower()
    status = backtest["status"].astype(str).str.lower()
    garch_mask = methods == "garch_t"
    counts["garch_rows"] = int(garch_mask.sum())
    counts["garch_fallback_events"] = int((garch_mask & (status != "ok")).sum())
    counts["garch_status_ok_count"] = int((garch_mask & (status == "ok")).sum())
    return counts


def _as_bool(value: object) -> bool | None:
    """Parse bool-like values from CSV cells; return None if unknown."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    return None


def _backtest_method_reject_rates(var_backtest_path: Path) -> dict[str, float]:
    """Compute 5% reject rates by method for Kupiec and Christoffersen tests."""
    rates = {
        "historical_kupiec_reject_rate_5pct": float("nan"),
        "historical_christoffersen_reject_rate_5pct": float("nan"),
        "parametric_kupiec_reject_rate_5pct": float("nan"),
        "parametric_christoffersen_reject_rate_5pct": float("nan"),
        "garch_kupiec_reject_rate_5pct": float("nan"),
        "garch_christoffersen_reject_rate_5pct": float("nan"),
    }
    if not var_backtest_path.exists():
        return rates
    try:
        backtest = pd.read_csv(var_backtest_path)
    except Exception:
        return rates
    required = {"status", "kupiec_reject_5pct", "christoffersen_reject_5pct"}
    if not required.issubset(backtest.columns):
        return rates

    status_ok = backtest["status"].astype(str).str.lower() == "ok"
    if "method" in backtest.columns:
        raw_methods = backtest["method"].astype(str).str.strip().str.lower()
    else:
        raw_methods = pd.Series(["historical"] * len(backtest), index=backtest.index)

    def _rate_for(method_mask: pd.Series, reject_column: str) -> float:
        mask = method_mask & status_ok
        if not mask.any():
            return float("nan")
        parsed = backtest.loc[mask, reject_column].map(_as_bool).dropna()
        if parsed.empty:
            return float("nan")
        return float(parsed.mean())

    rates["historical_kupiec_reject_rate_5pct"] = _rate_for(
        raw_methods == "historical",
        "kupiec_reject_5pct",
    )
    rates["historical_christoffersen_reject_rate_5pct"] = _rate_for(
        raw_methods == "historical",
        "christoffersen_reject_5pct",
    )
    rates["parametric_kupiec_reject_rate_5pct"] = _rate_for(
        raw_methods == "parametric_normal",
        "kupiec_reject_5pct",
    )
    rates["parametric_christoffersen_reject_rate_5pct"] = _rate_for(
        raw_methods == "parametric_normal",
        "christoffersen_reject_5pct",
    )
    rates["garch_kupiec_reject_rate_5pct"] = _rate_for(
        raw_methods == "garch_t",
        "kupiec_reject_5pct",
    )
    rates["garch_christoffersen_reject_rate_5pct"] = _rate_for(
        raw_methods == "garch_t",
        "christoffersen_reject_5pct",
    )
    return rates


def _panel_window_matches_target(panel_daily_path: Path) -> str:
    """Return whether panel_daily spans the configured sample window."""
    if not panel_daily_path.exists():
        return "unknown"
    try:
        panel = pd.read_parquet(panel_daily_path)
    except Exception:
        return "unknown"
    if panel.empty or "date" not in panel.columns:
        return "unknown"
    dates = pd.to_datetime(panel["date"], errors="coerce").dropna()
    if dates.empty:
        return "unknown"
    sample_start = dates.min().date().isoformat()
    sample_end = dates.max().date().isoformat()
    return str(sample_start == SAMPLE_START_DATE and sample_end == SAMPLE_END_DATE)


def _scorecard_status(
    *,
    window_matches_target: str,
    rolling_vol_fallback: bool,
    backtest_fallback_events: int,
) -> str:
    """Derive scorecard status from quality gates."""
    if window_matches_target != "True" or rolling_vol_fallback:
        return "fail"
    if backtest_fallback_events > 0:
        return "warn"
    return "ok"


def _scorecard_status_reason(
    *,
    window_matches_target: str,
    rolling_vol_fallback: bool,
    backtest_fallback_events: int,
) -> str:
    """Explain why scorecard status is fail/warn/ok."""
    if window_matches_target != "True":
        return "window_not_target"
    if rolling_vol_fallback:
        return "rolling_vol_fallback"
    if backtest_fallback_events > 0:
        return "backtest_fallbacks"
    return "none"


def task_create_scorecard(
    risk_summary: Path = TABLES_OUTPUT / "risk_summary.csv",
    rolling_vol: Path = ANALYSIS_OUTPUT / "rolling_vol.csv",
    var_backtest_hist: Path = ANALYSIS_OUTPUT / "var_backtest_hist.csv",
    panel_daily: Path = PANEL_DAILY_DATA,
    produces: Path = CHECKS_OUTPUT / "scorecard.json",
) -> None:
    """Build a minimal scorecard artifact from the current pipeline outputs."""
    produces.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(risk_summary)
    rolling_vol_valid_points, rolling_vol_fallback = _rolling_vol_quality(rolling_vol)
    fallback_reason_counts = _backtest_fallback_reason_counts(var_backtest_hist)
    method_quality_counts = _backtest_method_quality_counts(var_backtest_hist)
    garch_quality_counts = _garch_backtest_quality_counts(var_backtest_hist)
    method_reject_rates = _backtest_method_reject_rates(var_backtest_hist)
    window_matches_target = _panel_window_matches_target(panel_daily)
    backtest_fallback_events = int(sum(fallback_reason_counts.values()))
    fallback_events = int(rolling_vol_fallback) + backtest_fallback_events
    status = _scorecard_status(
        window_matches_target=window_matches_target,
        rolling_vol_fallback=rolling_vol_fallback,
        backtest_fallback_events=backtest_fallback_events,
    )
    status_reason = _scorecard_status_reason(
        window_matches_target=window_matches_target,
        rolling_vol_fallback=rolling_vol_fallback,
        backtest_fallback_events=backtest_fallback_events,
    )
    systematic_beta = _summary_metric(
        summary,
        risk_dimension="systematic",
        metric="beta",
        default=float("nan"),
    )
    systematic_alpha = _summary_metric(
        summary,
        risk_dimension="systematic",
        metric="alpha",
        default=float("nan"),
    )
    systematic_beta_rolling_60m_latest = _summary_metric(
        summary,
        risk_dimension="systematic",
        metric="beta_rolling_60m_latest",
        default=0.0,
    )
    systematic_beta_rolling_60m_valid_points = int(
        _summary_metric(
            summary,
            risk_dimension="systematic",
            metric="beta_rolling_60m_valid_points",
            default=0.0,
        ),
    )
    systematic_beta_rolling_60m_fallback = bool(
        _summary_metric(
            summary,
            risk_dimension="systematic",
            metric="beta_rolling_60m_fallback",
            default=1.0,
        ),
    )
    systematic_beta_rolling_252m_latest = _summary_metric(
        summary,
        risk_dimension="systematic",
        metric="beta_rolling_252m_latest",
        default=0.0,
    )
    systematic_beta_rolling_252m_valid_points = int(
        _summary_metric(
            summary,
            risk_dimension="systematic",
            metric="beta_rolling_252m_valid_points",
            default=0.0,
        ),
    )
    systematic_beta_rolling_252m_fallback = bool(
        _summary_metric(
            summary,
            risk_dimension="systematic",
            metric="beta_rolling_252m_fallback",
            default=1.0,
        ),
    )
    payload = {
        "status": status,
        "status_reason": status_reason,
        "metric_count": int(summary.shape[0]),
        "risk_dimensions": sorted(summary["risk_dimension"].unique().tolist()),
        "systematic_beta": systematic_beta,
        "systematic_alpha": systematic_alpha,
        "systematic_beta_rolling_60m_latest": systematic_beta_rolling_60m_latest,
        "systematic_beta_rolling_60m_valid_points": systematic_beta_rolling_60m_valid_points,
        "systematic_beta_rolling_60m_fallback": systematic_beta_rolling_60m_fallback,
        "systematic_beta_rolling_252m_latest": systematic_beta_rolling_252m_latest,
        "systematic_beta_rolling_252m_valid_points": systematic_beta_rolling_252m_valid_points,
        "systematic_beta_rolling_252m_fallback": systematic_beta_rolling_252m_fallback,
        "rolling_vol_valid_points": rolling_vol_valid_points,
        "rolling_vol_fallback": rolling_vol_fallback,
        "var_backtest_fallback_events": backtest_fallback_events,
        "var_backtest_fallback_reason_short_sample_count": fallback_reason_counts[
            "short_sample"
        ],
        "var_backtest_fallback_reason_nan_var_count": fallback_reason_counts["nan_var"],
        "var_backtest_fallback_reason_invalid_input_count": fallback_reason_counts[
            "invalid_input"
        ],
        "var_backtest_fallback_reason_exception_count": fallback_reason_counts[
            "exception"
        ],
        "var_backtest_fallback_reason_upstream_fallback_count": fallback_reason_counts[
            "upstream_fallback"
        ],
        "var_backtest_fallback_reason_unknown_count": fallback_reason_counts["unknown"],
        "var_backtest_historical_rows": method_quality_counts["historical_rows"],
        "var_backtest_historical_fallback_events": method_quality_counts[
            "historical_fallback_events"
        ],
        "var_backtest_parametric_rows": method_quality_counts["parametric_rows"],
        "var_backtest_parametric_fallback_events": method_quality_counts[
            "parametric_fallback_events"
        ],
        "var_backtest_other_rows": method_quality_counts["other_rows"],
        "var_backtest_other_fallback_events": method_quality_counts[
            "other_fallback_events"
        ],
        "var_backtest_garch_rows": garch_quality_counts["garch_rows"],
        "var_backtest_garch_fallback_events": garch_quality_counts[
            "garch_fallback_events"
        ],
        "var_backtest_garch_status_ok_count": garch_quality_counts[
            "garch_status_ok_count"
        ],
        "historical_kupiec_reject_rate_5pct": method_reject_rates[
            "historical_kupiec_reject_rate_5pct"
        ],
        "historical_christoffersen_reject_rate_5pct": method_reject_rates[
            "historical_christoffersen_reject_rate_5pct"
        ],
        "parametric_kupiec_reject_rate_5pct": method_reject_rates[
            "parametric_kupiec_reject_rate_5pct"
        ],
        "parametric_christoffersen_reject_rate_5pct": method_reject_rates[
            "parametric_christoffersen_reject_rate_5pct"
        ],
        "garch_kupiec_reject_rate_5pct": method_reject_rates[
            "garch_kupiec_reject_rate_5pct"
        ],
        "garch_christoffersen_reject_rate_5pct": method_reject_rates[
            "garch_christoffersen_reject_rate_5pct"
        ],
        "sample_target_start": SAMPLE_START_DATE,
        "sample_target_end": SAMPLE_END_DATE,
        "window_matches_target": window_matches_target,
        "fallback_events": fallback_events,
        "notes": "Scorecard built from current pipeline outputs and diagnostics checks.",
    }
    produces.write_text(json.dumps(payload, indent=2), encoding="utf-8")


"""Observed-data portfolio workflow, independent of the legacy synthetic demo.

Run from the repository root: python scripts/verified_analysis.py
Refreshing vendor responses is explicit: append --refresh.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import exchange_calendars as xcals
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, norm

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/verified"
OUTPUT = ROOT / "documents/verified"
START, END = "2020-01-01", "2024-12-31"
URLS = {
    "nvda.json": "https://query2.finance.yahoo.com/v8/finance/chart/NVDA?period1=1577664000&period2=1735776000&interval=1d",
    "sp500.json": "https://query2.finance.yahoo.com/v8/finance/chart/%5EGSPC?period1=1577664000&period2=1735776000&interval=1d",
    "dgs10.csv": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10&cosd=2019-12-01&coed=2024-12-31",
}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def parse_equity(raw: bytes, symbol: str) -> pd.DataFrame:
    result = json.loads(raw)["chart"]["result"][0]
    if result["meta"]["symbol"] != symbol or result["meta"]["currency"] != "USD":
        raise ValueError("Unexpected instrument or currency in vendor response")
    quote = result["indicators"]["quote"][0]
    frame = pd.DataFrame({
        "date": pd.to_datetime(result["timestamp"], unit="s", utc=True)
        .tz_convert("America/New_York").tz_localize(None).normalize(),
        "close": quote["close"],
        "adj_close": result["indicators"]["adjclose"][0]["adjclose"],
        "volume": quote["volume"],
    }).sort_values("date")
    if frame.isna().any().any() or frame["date"].duplicated().any():
        raise ValueError("Missing or duplicate equity observations")
    if not np.isfinite(frame[["close", "adj_close", "volume"]]).all().all():
        raise ValueError("Non-finite equity observations")
    if (frame[["close", "adj_close"]] <= 0).any().any() or (frame["volume"] < 0).any():
        raise ValueError("Invalid prices or volumes")
    return frame


def parse_yields(raw: bytes) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(raw), na_values=["."])
    frame.columns = ["date", "dgs10"]
    frame["date"] = pd.to_datetime(frame["date"])
    frame["dgs10"] = pd.to_numeric(frame["dgs10"], errors="raise")
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError("Invalid FRED dates")
    return frame.sort_values("date")


def refresh() -> None:
    """Stage and validate all vendor responses before replacing the input bundle."""
    pending = {}
    for name, url in URLS.items():
        with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as response:
            pending[name] = response.read()
    nvda = parse_equity(pending["nvda.json"], "NVDA")
    market = parse_equity(pending["sp500.json"], "^GSPC")
    make_panel(nvda, market)
    macro = align_yields(make_panel(nvda, market), parse_yields(pending["dgs10.csv"]))
    if macro["dgs10"].isna().any():
        raise ValueError("Yield history does not cover the sample")
    manifest = {
        "data_kind": "observed",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample": [START, END],
        "return_basis": "Yahoo adjusted close; USD; S&P 500 price-index benchmark",
        "yield_units": "DGS10 percentage points, e.g. 4.0 means 4%",
        "files": {name: {"url": URLS[name], "sha256": sha256(raw)} for name, raw in pending.items()},
    }
    INPUT.mkdir(parents=True, exist_ok=True)
    for name, raw in pending.items():
        (INPUT / name).write_bytes(raw)
    (INPUT / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    manifest = json.loads((INPUT / "sources.json").read_text())
    if manifest["data_kind"] != "observed":
        raise ValueError("Observed input bundle required")
    raw = {}
    for name, url in URLS.items():
        raw[name] = (INPUT / name).read_bytes()
        if manifest["files"][name]["url"] != url or sha256(raw[name]) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Source ledger/hash mismatch: {name}")
    return parse_equity(raw["nvda.json"], "NVDA"), parse_equity(raw["sp500.json"], "^GSPC"), parse_yields(raw["dgs10.csv"]), manifest


def make_panel(nvda: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Compute returns before sample filtering to retain the genuine first return."""
    calendar = xcals.get_calendar("XNYS", start="2019-12-01", end="2025-01-31")
    expected = calendar.sessions_in_range(START, END).tz_localize(None)
    for frame in (nvda, market):
        dates = pd.DatetimeIndex(frame.loc[frame["date"].between(START, END), "date"])
        if not dates.equals(expected):
            raise ValueError("Missing or extra exchange sessions in input")
    asset = nvda.set_index("date").copy()
    benchmark = market.set_index("date").copy()
    asset["ret"] = asset["adj_close"].pct_change(fill_method=None)
    benchmark["market_ret"] = benchmark["adj_close"].pct_change(fill_method=None)
    panel = asset.join(benchmark[["market_ret"]], how="inner").loc[START:END].reset_index()
    if not np.isfinite(panel[["ret", "market_ret"]]).all().all():
        raise ValueError("Pre-sample adjusted price required for the first return")
    panel["dollar_volume"] = panel["close"] * panel["volume"]
    return panel


def align_yields(panel: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Descriptive contemporaneous alignment; never fill from a future date."""
    observations = macro.dropna(subset=["dgs10"]).rename(columns={"date": "yield_date"})
    return pd.merge_asof(panel.sort_values("date"), observations.sort_values("yield_date"),
                         left_on="date", right_on="yield_date", direction="backward",
                         tolerance=pd.Timedelta(days=7))


def monthly_panel(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.assign(month=panel["date"].dt.to_period("M").dt.to_timestamp("M"))
    return data.groupby("month").agg(
        ret=("ret", lambda values: (1 + values).prod() - 1),
        market_ret=("market_ret", lambda values: (1 + values).prod() - 1),
        dollar_volume=("dollar_volume", "mean"),
    ).reset_index()


def historical_tail(returns: np.ndarray, alpha: float) -> tuple[float, float]:
    threshold = float(np.quantile(returns, 1 - alpha))
    return threshold, float(returns[returns <= threshold].mean())


def rolling_forecasts(panel: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Forecast t using only the previous window returns; no refitting on t."""
    if window < 30 or len(panel) <= window:
        raise ValueError("At least 30 training observations and one forecast required")
    if panel["date"].duplicated().any() or not panel["date"].is_monotonic_increasing:
        raise ValueError("Sorted unique dates required")
    returns = panel["ret"].to_numpy(dtype=float)
    if not np.isfinite(returns).all():
        raise ValueError("Finite returns required")
    rows = []
    for index in range(window, len(panel)):
        history = returns[index - window:index]
        mean, sigma = history.mean(), history.std(ddof=1)
        for alpha in (0.95, 0.99):
            hvar, hes = historical_tail(history, alpha)
            z = norm.ppf(1 - alpha)
            thresholds = {"historical": (hvar, hes),
                          "normal": (mean + sigma * z, mean - sigma * norm.pdf(z) / (1 - alpha))}
            for method, (var, es) in thresholds.items():
                rows.append({"date": panel["date"].iloc[index],
                             "training_start": panel["date"].iloc[index - window],
                             "training_end": panel["date"].iloc[index - 1],
                             "training_size": window, "method": method, "alpha": alpha,
                             "var": var, "es": es, "realized_return": returns[index],
                             "exceedance": int(returns[index] < var)})
    return pd.DataFrame(rows)


def backtest_summary(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, alpha), group in forecasts.groupby(["method", "alpha"]):
        count, size = int(group["exceedance"].sum()), len(group)
        observed, expected = count / size, 1 - alpha
        def log_likelihood(probability):
            return (count * np.log(probability) if count else 0) + ((size - count) * np.log1p(-probability) if count < size else 0)
        lr = max(0.0, 2 * (log_likelihood(observed) - log_likelihood(expected)))
        loss = (expected - group["exceedance"]) * (group["realized_return"] - group["var"])
        rows.append({"method": method, "alpha": alpha, "forecasts": size, "exceedances": count,
                     "exceedance_rate": observed, "expected_rate": expected,
                     "kupiec_pvalue": float(chi2.sf(lr, 1)), "mean_quantile_loss": float(loss.mean())})
    return pd.DataFrame(rows)


def build(output: Path = OUTPUT) -> dict:
    nvda, market, yields, manifest = load_inputs()
    panel = make_panel(nvda, market)
    monthly = monthly_panel(panel)
    macro = align_yields(panel, yields)
    if macro["dgs10"].isna().any() or (macro["yield_date"] > macro["date"]).any():
        raise ValueError("Unavailable or future yields in aligned panel")
    forecasts = rolling_forecasts(panel)
    summary = backtest_summary(forecasts)
    if not (forecasts["training_end"] < forecasts["date"]).all():
        raise ValueError("Forecast uses future information")
    # Validate compounded returns against month-end adjusted-price ratios.
    monthly_prices = nvda.set_index("date")["adj_close"].resample("ME").last().pct_change().loc[START:END]
    if not np.allclose(monthly["ret"], monthly_prices.to_numpy()):
        raise ValueError("Monthly compounded returns disagree with adjusted prices")
    if not np.isfinite(forecasts[["var", "es", "realized_return"]]).all().all():
        raise ValueError("Non-finite forecasts")
    wealth = (1 + monthly["ret"]).cumprod()
    drawdown = wealth / wealth.cummax().clip(lower=1) - 1
    beta = monthly["ret"].cov(monthly["market_ret"]) / monthly["market_ret"].var()
    daily_beta = panel["ret"].rolling(252).cov(panel["market_ret"]) / panel["market_ret"].rolling(252).var()
    model = sm.OLS(macro["ret"], sm.add_constant(macro[["dgs10"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    var, es = historical_tail(panel["ret"].to_numpy(), .95)
    mvar, mes = historical_tail(monthly["ret"].to_numpy(), .95)
    metrics = {"sample_start": str(panel["date"].min().date()), "sample_end": str(panel["date"].max().date()),
               "daily_observations": len(panel), "monthly_observations": len(monthly),
               "annualized_monthly_volatility": float(monthly["ret"].std(ddof=1) * np.sqrt(12)),
               "daily_var_95": var, "daily_es_95": es, "monthly_var_95": mvar, "monthly_es_95": mes,
               "month_end_max_drawdown": float(drawdown.min()), "monthly_beta": float(beta),
               "latest_252_session_beta": float(daily_beta.iloc[-1]),
               "median_monthly_mean_dollar_volume": float(monthly["dollar_volume"].median()),
               "yield_level_beta": float(model.params["dgs10"]), "yield_level_pvalue": float(model.pvalues["dgs10"]),
               "yield_level_r_squared": float(model.rsquared), "oos_start": str(forecasts["date"].min().date()),
               "oos_end": str(forecasts["date"].max().date()), "oos_dates": int(forecasts["date"].nunique())}
    output.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output / "daily_panel.csv", index=False)
    monthly.to_csv(output / "monthly_panel.csv", index=False)
    forecasts.to_csv(output / "oos_forecasts.csv", index=False)
    summary.to_csv(output / "oos_summary.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    checks = {"vendor_hashes_match": True, "observed_instrument_usd": True, "exact_xnys_sessions": True,
              "adjusted_returns_finite": True, "monthly_compounding_matches_prices": True,
              "macro_alignment_no_future_fill": True, "oos_training_precedes_forecast": True,
              "forecasts_finite": True}
    (output / "validation.json").write_text(json.dumps({"checks": checks, "all_pass": all(checks.values()),
                                                      "input_manifest_sha256": sha256((INPUT / "sources.json").read_bytes())}, indent=2) + "\n")
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    axes[0].plot(panel["date"], panel["ret"].rolling(60).std() * np.sqrt(252) * 100)
    axes[0].set(title="NVDA: 60-session annualized daily volatility", ylabel="Volatility (%)")
    axes[1].fill_between(monthly["month"], drawdown * 100, 0, color="#ad4250", alpha=.6)
    axes[1].set(title="Month-end drawdown (including initial capital)", ylabel="Drawdown (%)")
    axes[2].plot(panel["date"], daily_beta, color="#2a766c")
    axes[2].set(title="252-session beta against S&P 500 price index", ylabel="Beta")
    for axis in axes:
        axis.grid(alpha=.2)
    fig.savefig(output / "risk_dashboard.png", dpi=140)
    plt.close(fig)
    h95 = forecasts[(forecasts["method"] == "historical") & (forecasts["alpha"] == .95)]
    fig, axis = plt.subplots(figsize=(11, 4), constrained_layout=True)
    axis.plot(h95["date"], h95["realized_return"] * 100, alpha=.4, linewidth=.7, label="Realized return")
    axis.plot(h95["date"], h95["var"] * 100, color="#ad4250", label="Prior-252-session historical VaR (95%)")
    breached = h95[h95["exceedance"] == 1]
    axis.scatter(breached["date"], breached["realized_return"] * 100, s=10, color="#ad4250", label="Breach")
    axis.set(title="NVDA: out-of-sample daily VaR", ylabel="Daily return (%)")
    axis.legend(fontsize=8)
    axis.grid(alpha=.2)
    fig.savefig(output / "oos_var.png", dpi=140)
    plt.close(fig)
    lines = ["# Verified observed-data results", "", "Generated by `scripts/verified_analysis.py` from the hash-checked vendor bundle.", "",
             f"Sample: {metrics['sample_start']}–{metrics['sample_end']}; {len(panel)} sessions, {len(monthly)} months.", "",
             "| Measure | Value |", "|---|---:|",
             f"| Annualized monthly volatility | {metrics['annualized_monthly_volatility']:.2%} |",
             f"| Daily historical VaR / ES (95%) | {var:.2%} / {es:.2%} |",
             f"| Monthly historical VaR / ES (95%) | {mvar:.2%} / {mes:.2%} |",
             f"| Month-end maximum drawdown | {metrics['month_end_max_drawdown']:.2%} |",
             f"| Monthly raw-return beta | {beta:.3f} |",
             f"| Latest 252-session beta | {daily_beta.iloc[-1]:.3f} |",
             f"| Median monthly mean dollar volume | ${metrics['median_monthly_mean_dollar_volume']/1e9:.2f} billion |", "",
             "## Out-of-sample VaR", "",
             f"{metrics['oos_start']}–{metrics['oos_end']}; each prediction uses exactly 252 prior sessions.", "",
             "| Method | Confidence | Breaches / forecasts | Rate | Expected | Kupiec p-value |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in summary.to_dict("records"):
        lines.append(f"| {row['method']} | {row['alpha']:.0%} | {row['exceedances']} / {row['forecasts']} | {row['exceedance_rate']:.2%} | {row['expected_rate']:.0%} | {row['kupiec_pvalue']:.4f} |")
    lines += ["", "## Interpretation and limits", "",
              "VaR and ES are negative return thresholds, not positive loss amounts. Drawdown uses month-end observations and can miss deeper intramonth losses. Beta uses raw returns rather than risk-free-adjusted CAPM returns.", "",
              f"Descriptive daily return regression on the DGS10 yield level (HAC, 5 lags): coefficient {metrics['yield_level_beta']:.6f} per percentage point of yield, p-value {metrics['yield_level_pvalue']:.4f}, R² {metrics['yield_level_r_squared']:.6f}. This is an association, not a causal or tradable forecast.", "",
              "Coverage tests do not establish independence, ES calibration, or future predictive performance. Normal and historical models use the same forecast dates. GARCH results from the legacy demo are excluded.", "",
              "USD results exclude EUR/USD risk, trading costs and portfolio diversification. S&P 500 is a price-index benchmark; adjusted NVDA prices and vendor histories can be revised.", "",
              "Liquidity is a turnover proxy; historical split-adjusted prices multiplied by vendor share volume are not an audited execution-capacity estimate. Inspect the vendor's split/volume conventions before economic use.", "",
              "Validation checks demonstrate input integrity and calculation consistency, not that the vendor data or the model is infallible."]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Explicitly replace archived vendor responses and source ledger")
    args = parser.parse_args()
    if args.refresh:
        refresh()
    print(json.dumps(build(), indent=2))

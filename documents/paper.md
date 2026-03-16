# NVDA Risk Analysis: Diagnostics-First Reproducible Pipeline

+++ {"part": "abstract"}

This project develops a reproducible and offline-first risk analysis system for NVDA
over the sample window `2020-01-01` to `2024-12-31`. The workflow combines market,
liquidity, drawdown, systematic, and macro risk modules inside a deterministic build
pipeline and treats diagnostics as a first-class product. Rather than stopping at model
outputs, the project also exports provenance metadata, cross-artifact consistency checks,
and a scorecard that records whether the full workflow remains internally coherent.

+++

```{raw} latex
\clearpage
```

## 1. Project Goal

The goal is to turn a single-name financial risk study into an engineering-grade
research artifact. The repository is organized as a deterministic workflow:

`data_management -> analysis -> final -> documents`

This structure lets the project regenerate figures, tables, and report materials from a
shared data backbone instead of relying on manual copy-and-paste steps.

## 2. Data and Reproducibility Design

The data pipeline follows a deterministic priority:

1. local cache in `bld/data/raw`,
2. repository snapshots in `src/nvda_risk_project/data/snapshots`,
3. online fetch as a last resort.

Each build emits `data_provenance.json`, which records sample window, source usage, and
hash metadata for critical files. Cleaned outputs are consolidated into
`panel_daily.parquet` and `panel_monthly.csv`, which are reused across all downstream
modules.

## 3. Analysis Coverage

The pipeline implements five risk dimensions:

1. Market risk: rolling volatility, VaR/ES under historical and parametric normal
   baselines, plus GARCH-t with fallback diagnostics.
2. Liquidity risk: dollar volume and Amihud illiquidity summaries.
3. Drawdown risk: drawdown magnitude and duration metrics.
4. Systematic risk: static CAPM beta/alpha and rolling beta diagnostics.
5. Macro risk: monthly specification plus a daily DGS10-aligned regression path.

Backtesting includes Kupiec and Christoffersen tests and is integrated into the
diagnostics layer.

## 4. Key Results

The consolidated summary table reports the following representative outputs:

- annualized volatility: `0.332914`,
- historical VaR at 95 percent: `-0.137596`,
- expected shortfall: `-0.172794`,
- maximum drawdown: `-0.237682`,
- median dollar volume: `7.46914e+08`.

```{figure} public/fig_market_overview.png
---
width: 88%
label: fig:market_overview
---
High-level overview of exported risk dimensions.
```

```{figure} public/fig_volatility.png
---
width: 88%
label: fig:volatility
---
Rolling volatility summary figure for NVDA.
```

```{figure} public/fig_var_exceedances.png
---
width: 88%
label: fig:var_exceed
---
VaR exceedance comparison across methods.
```

```{figure} public/fig_drawdown.png
---
width: 88%
label: fig:drawdown
---
Drawdown trajectory summary.
```

```{figure} public/fig_beta_rolling.png
---
width: 88%
label: fig:beta_roll
---
Rolling CAPM beta figure.
```

````{table} Consolidated risk metrics exported by the final pipeline.
---
label: tab:risk_summary
align: center
---
```{include} tables/estimation_results.md
```
````

## 5. Diagnostics and Quality Gates

In addition to model outputs, the workflow exports diagnostics and scorecard artifacts
that track:

- data source and sample-window consistency,
- provenance hash checks,
- model fallback reasons,
- VaR backtest consistency across recorded and observed metrics,
- macro-frequency validity,
- systematic risk metric consistency.

The latest diagnostics report records `32` passing quality gates and `0` failed checks.

````{table} Diagnostics and reproducibility quality-gate report.
---
label: tab:diagnostics
align: center
---
```{include} tables/diagnostics.md
```
````

## 6. Conclusion

The project delivers more than a collection of financial metrics. It provides a
test-backed, diagnostics-first workflow that keeps risk outputs traceable from raw data
through final presentation artifacts. That combination of analytical coverage and
engineering reliability is the main contribution of the repository.

# NVDA Risk Analysis Project

This site accompanies a reproducible risk analysis pipeline for NVIDIA (`NVDA`). The
project evaluates market, liquidity, drawdown, systematic, and macro risk while
producing diagnostics that make every build auditable.

## What This Project Delivers

- A deterministic workflow from raw data to paper and presentation artifacts.
- Shared daily and monthly panels used across all analysis modules.
- Public-facing diagnostics, provenance, and scorecards rather than hidden build logs.

## Current Snapshot

- Sample window: `2020-01-01` to `2024-12-31`
- Risk dimensions covered: `5`
- Latest diagnostics result: `32/32` quality gates passing
- Latest local verification baseline: `133` tests passed with `92.65%` coverage

## Representative Results

| Risk area | Headline result |
|:----------|:----------------|
| Market risk | Annualized volatility `33.3%`; historical VaR (95%) `-13.8%` |
| Liquidity risk | Median dollar volume about `$746.9M` |
| Drawdown risk | Maximum drawdown `-23.8%` |
| Macro risk | DGS10 coefficient `0.175` with very low `R^2` |

## Why The Workflow Matters

Many financial analysis repositories stop at figures or regression outputs. This one
also records:

- where the data came from,
- whether cached and snapshot data match the configured sample window,
- whether model fallbacks were triggered,
- whether summary tables and diagnostics remain internally consistent.

## Read Next

- `Getting Started` for the quickest way to run the pipeline.
- `Project Architecture` for the build flow and artifact layout.
- `Development Workflow` for testing, diagnostics, and extension points.

# Project Architecture

The repository is organized as a deterministic pipeline:

`data_management -> analysis -> final -> documents`

## Pipeline Stages

1. `data_management`
   Downloads, cleans, aligns, and documents the raw inputs.
2. `analysis`
   Computes risk metrics for market, liquidity, drawdown, systematic, and macro modules.
3. `final`
   Produces figures, summary tables, diagnostics, and scorecards.
4. `documents`
   Turns the exported artifacts into a paper and a presentation.

## Shared Data Products

- `bld/data/clean/panel_daily.parquet`
- `bld/data/processed/panel_monthly.csv`
- `bld/data/clean/data_provenance.json`

These shared products let the downstream modules use a consistent sample window and
avoid ad hoc joins inside individual notebooks or scripts.

## Analysis Outputs

The project exports:

- rolling volatility and VaR artifacts,
- drawdown metrics and drawdown figures,
- liquidity summaries,
- static and rolling CAPM outputs,
- macro regression summaries,
- diagnostics and scorecards.

## Reliability Features

- Source priority: cache, then repository snapshots, then online download
- Fallback-aware diagnostics for model execution
- Cross-artifact consistency checks
- Test coverage and end-to-end validation

The result is closer to a research system than a one-off analysis notebook.

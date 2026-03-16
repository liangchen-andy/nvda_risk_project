# NVDA Risk Analysis Project

Engineering-grade, reproducible NVDA risk analysis built on the EPP template.

The project is organized as a deterministic pipeline:
`data_management -> analysis -> final -> documents`.

## Scope

The workflow covers:

1. Market risk: rolling volatility, VaR/ES (historical, parametric normal, GARCH-t), backtests.
2. Liquidity risk: dollar volume and Amihud illiquidity.
3. Drawdown risk: drawdown magnitude and duration summaries.
4. Systematic risk: static and rolling CAPM beta.
5. Macro risk: monthly specification and daily DGS10-aligned regression path.

## Reproducibility Guarantees

Data acquisition follows a strict priority:

1. existing cache in `bld/data/raw`
2. repository snapshots in `src/nvda_risk_project/data/snapshots`
3. online download (`yfinance`) as last resort

Each build writes provenance metadata:

- `bld/data/clean/data_provenance.json`

Quality gates and consistency checks are produced in:

- `bld/metrics/diagnostics.csv`
- `documents/tables/diagnostics.md`
- `bld/checks/scorecard.json`

## Project Structure

- `src/nvda_risk_project/data_management`: download, clean, align, provenance.
- `src/nvda_risk_project/analysis`: risk model logic and analysis tasks.
- `src/nvda_risk_project/final`: summary tables, figures, diagnostics and scorecard.
- `documents`: paper/presentation sources and public assets.
- `tests`: unit, integration and e2e checks.

## Main Outputs

- Daily panel: `bld/data/clean/panel_daily.parquet`
- Monthly panel: `bld/data/processed/panel_monthly.csv`
- Risk metrics: `bld/metrics/*.csv`
- Figures: `documents/public/fig_*.png`
- Summary table: `documents/tables/estimation_results.md`
- Diagnostics: `documents/tables/diagnostics.md`

## Run Instructions

```bash
pixi install
pixi run npm install
pixi run pytest -q
pixi run pytask
```

Targeted runs:

```bash
pixi run pytask -k diagnostics --force
pixi run pytask -k figure --force
pixi run pytask -k risk --force
```

Presentation and paper preview:

```bash
pixi run view-paper
pixi run view-pres
```

Compile report artifacts:

```bash
pixi run pytask -k paper-pdf --force
pixi run pytask -k presentation --force
```

## Libraries Used

Core libraries used in this project:

- `numpy`
- `pandas`
- `scipy`
- `statsmodels`
- `plotly`
- `yfinance`
- `pytask`
- `pytest` (`pytest-cov`, `pytest-xdist`)
- `pyarrow`
- `python-kaleido`
- `jupyter-book`

## Known CI Limitation

Delivery baseline commit: `f9cf493`.

The project is reproducible and passes full local verification on this baseline:

```bash
pixi run pytest -q
pixi run pytask
pixi run pytask -k paper --force
pixi run pytask -k presentation --force
```

Most recent local result on this baseline:

- `133 passed` with coverage `92.65%`
- diagnostics pipeline and presentation/paper build succeed locally

GitHub Actions can still show intermittent failures on Linux/macOS jobs due to environment-specific execution instability
in pytask/plot rendering and cache state interactions. The core project outputs and quality-gate artifacts are verified
through the local reproducible workflow above.

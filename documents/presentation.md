---
theme: academic
class: text-center
highlighter: shiki
lineNumbers: false
drawings:
  persist: false
transition: fade
title: NVDA Risk Analysis
mdc: true
defaults:
  layout: center
---

# NVDA Risk Analysis

Reproducible engineering pipeline for market risk diagnostics

LIANGCHEN CHEN

---
layout: default
---

# Project Goal

- Build a fully reproducible and offline-first NVDA risk pipeline.
- Make verification and diagnostics first-class outputs.
- Deliver deterministic artifacts from data to report figures.

Pipeline:
`data_management -> analysis -> final -> documents`

---
layout: default
---

# 1.0 Engineering Constraints

- Data source priority: `cache -> snapshots -> online`.
- Provenance required for every build (`data_provenance.json`).
- Core functions must be test-covered with boundary cases.
- Coverage gate enforced in CI (`pytest --cov-fail-under=80`).
- Failures use fallback logic and must be recorded in diagnostics.

---
layout: default
---

# Risk Modules Implemented

- Market risk: rolling vol, VaR/ES, GARCH-t, VaR backtests.
- Liquidity risk: dollar volume and Amihud ILLIQ.
- Drawdown risk: drawdown path and MDD summary.
- Systematic risk: static and rolling CAPM beta.
- Macro risk: frequency-consistent regression path.

---
layout: default
---

# Diagnostics as Product

- `bld/metrics/diagnostics.csv`
- `documents/tables/diagnostics.md`
- `bld/checks/scorecard.json`

Diagnostics include:

- data source used, sample period, and missingness checks,
- model status and fallback events,
- VaR exceedance counts and backtest statistics,
- cross-file consistency checks.

---
layout: image-right
image: ./public/fig_volatility.png
---

# Market Volatility

Rolling volatility profile for NVDA over the configured sample window.

---
layout: image-right
image: ./public/fig_var_exceedances.png
---

# VaR Exceedances

Observed exceedances compared with expected exceedance rates by method.

---
layout: image-right
image: ./public/fig_drawdown.png
---

# Drawdown Risk

Drawdown trajectory and stress episodes in the sample period.

---
layout: image-right
image: ./public/fig_beta_rolling.png
---

# Systematic Risk

Rolling CAPM beta against market benchmark.

---
layout: default
---

# Reproducibility Commands

```bash
pixi install
pixi run npm install
pixi run pytest -q
pixi run pytask
```

Targeted reruns:

```bash
pixi run pytask -k diagnostics --force
pixi run pytask -k figure --force
```

---
layout: end
---

# Thank You

Questions?

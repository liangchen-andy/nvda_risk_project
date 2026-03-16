---
theme: academic
class: text-center
highlighter: shiki
lineNumbers: false
drawings:
  persist: false
transition: fade
title: NVDA Risk Analysis Project
mdc: true
defaults:
  layout: center
---

# NVDA Risk Analysis Project

Diagnostics-first, reproducible risk analytics for NVIDIA (`NVDA`)

Sample window: `2020-01-01` to `2024-12-31`

LIANGCHEN CHEN

---
layout: default
---

# Why This Project Exists

- NVDA is a high-volatility, event-sensitive stock with multiple risk channels.
- A one-off notebook can show point estimates, but it does not show data lineage,
  fallback behavior, or build reliability.
- This project treats reproducibility and diagnostics as part of the deliverable, not as
  afterthoughts.

---
layout: default
---

# What The Pipeline Produces

Pipeline:
`data_management -> analysis -> final -> documents`

- Shared daily and monthly panels
- Five risk modules in one workflow
- Presentation-ready figures and summary tables
- Diagnostics, provenance, and scorecard artifacts

---
layout: default
---

# Data Rules And Audit Trail

- Sample window fixed to `2020-01-01` through `2024-12-31`
- Source priority: `cache -> snapshots -> online`
- Every build emits `data_provenance.json`
- Diagnostics record consistency checks, fallback events, and status flags

---
layout: image-right
image: ./public/fig_market_overview.png
---

# Risk Coverage At A Glance

- Market risk
- Liquidity risk
- Drawdown risk
- Systematic risk
- Macro risk

One pipeline produces all five views from a shared data backbone.

---
layout: default
---

# Selected Findings

- Annualized volatility: `33.3%`
- Historical VaR (95%): `-13.8%`
- Expected shortfall: `-17.3%`
- Maximum drawdown: `-23.8%`
- Median dollar volume: about `$746.9M`
- Macro DGS10 coefficient: `0.175` with low explanatory power (`R^2 = 0.0007`)

---
layout: image-right
image: ./public/fig_volatility.png
---

# Volatility View

The rolling volatility figure gives regime context and is exported directly into the
paper and presentation workflow.

---
layout: image-right
image: ./public/fig_var_exceedances.png
---

# VaR Diagnostics

- Exceedance counts are compared with expected exceedance rates by method.
- Backtests are paired with diagnostics so model failures do not stay silent.

---
layout: image-right
image: ./public/fig_drawdown.png
---

# Stress Visualization

- Drawdown figures make both depth and persistence visible.
- Current summary outputs report a maximum drawdown of `-23.8%`.

---
layout: default
---

# Credibility Checks

- Latest diagnostics result: `32/32` quality gates passing
- Latest local verification baseline: `133` tests passed
- Coverage baseline: `92.65%`
- Public artifacts include diagnostics, scorecard, and provenance metadata

---
layout: default
---

# Reproduce The Project

```bash
pixi install
pixi run npm install
pixi run pytest -q
pixi run pytask
```

Focused reruns:

```bash
pixi run pytask -k diagnostics --force
pixi run pytask -k figure --force
```

---
layout: end
---

# Thank You

Questions?

# GEMINI.md

## Project Overview

This repository analyzes NVIDIA (`NVDA`) risk through a reproducible research pipeline.
It combines deterministic data preparation, multiple risk modules, diagnostics, and
document generation in one task graph.

## Key Tools

- Python `3.14`
- Pixi for environment and task management
- Pytask for workflow orchestration
- MyST / Jupyter Book for the paper and docs site
- Slidev for the presentation
- Pytest for verification

## Directory Guide

- `src/nvda_risk_project/`: core project package
- `documents/`: paper, presentation, figures, and tables
- `docs_template/source/`: short project documentation site
- `tests/`: unit, integration, and end-to-end checks

## Common Commands

```bash
pixi run pytest -q
pixi run pytask
pixi run pytask -k diagnostics --force
pixi run pytask -k paper-pdf --force
pixi run docs
pixi run view-paper
pixi run view-pres
```

## Development Conventions

- Prefer task-driven reproducibility over manual notebook steps.
- Keep public-facing artifacts honest: failed builds should error, not silently emit
  placeholders.
- When changing outputs, keep `README.md`, docs, paper, and slides consistent.

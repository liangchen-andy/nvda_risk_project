# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

This is a reproducible risk analysis project for NVIDIA (`NVDA`). The codebase builds a
deterministic pipeline from raw market and macro data to diagnostics, figures, tables,
and document artifacts.

Pipeline:
`data_management -> analysis -> final -> documents`

## Common Commands

```bash
pixi run pytask
pixi run pytest -q
pixi run pytask -k diagnostics --force
pixi run pytask -k figure --force
pixi run pytask -k paper-pdf --force
pixi run pytask -k presentation --force
pixi run docs
pixi run view-docs
pixi run view-paper
pixi run view-pres
```

## Important Paths

- `src/nvda_risk_project/data_management`: data download, cleaning, alignment, provenance
- `src/nvda_risk_project/analysis`: risk model implementations and task entry points
- `src/nvda_risk_project/final`: exported figures, tables, diagnostics, and scorecard tasks
- `documents`: paper and presentation sources
- `documents/public`: generated figures used for external presentation
- `documents/tables`: generated summary and diagnostics tables

## Working Notes

- The repository is offline-first: cached files and bundled snapshots are preferred over
  online downloads.
- `paper.pdf` depends on the LaTeX toolchain and `perl`, both of which are expected to
  be available through the project environment.
- Public-facing materials should stay aligned across `README.md`, `documents/paper.md`,
  `documents/presentation.md`, and the docs site in `docs_template/source`.

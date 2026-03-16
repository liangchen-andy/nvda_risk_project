# Getting Started

The fastest way to evaluate the project is to run the tests and rebuild the pipeline
from scratch.

## Requirements

- `pixi`
- `git`
- A working LaTeX installation if you want to export the paper as PDF

`nodejs` is provided through the Pixi environment, but the frontend dependencies still
need to be installed once via `npm install`.

## Quick Start

```bash
pixi install
pixi run npm install
pixi run pytest -q
pixi run pytask
```

## What You Should See

After a successful run, the repository will contain:

- cleaned data panels in `bld/data`
- metrics and diagnostics in `bld/metrics`
- figures in `documents/public`
- summary tables in `documents/tables`
- a presentation export in `presentation.pdf`

## Useful Follow-Up Commands

```bash
pixi run pytask -k diagnostics --force
pixi run pytask -k figure --force
pixi run view-paper
pixi run view-pres
pixi run docs
```

## Recommended Reading Order

1. Inspect `documents/tables/diagnostics.md` to confirm the latest quality gates.
2. Review `documents/tables/estimation_results.md` for the consolidated output metrics.
3. Open the exported figures in `documents/public` for presentation-ready visuals.

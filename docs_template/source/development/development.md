# Development Workflow

This project is maintained like a small research system rather than a single notebook.

## Core Commands

```bash
pixi run pytest -q
pixi run pytask
pixi run pytask -k diagnostics --force
pixi run docs
```

## Verification Strategy

- `pytest` covers the analysis and pipeline behavior.
- Coverage is enforced through the project test configuration.
- Diagnostics artifacts record quality-gate results and fallback status.
- Final figures and tables are generated through the same build graph as the reports.

## Extending The Project

When adding a new analysis branch, the usual sequence is:

1. add or adapt the upstream data task,
2. create the analysis logic and task entry point,
3. export a final artifact in `src/nvda_risk_project/final`,
4. surface the result in `documents`,
5. add tests and diagnostics coverage.

## Public-Facing Mindset

Because the repository is used for external presentation, a successful contribution
should improve both:

- analytical correctness,
- clarity of the generated artifacts for readers who did not build the project
  themselves.

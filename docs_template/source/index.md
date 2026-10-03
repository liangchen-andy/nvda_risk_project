# NVIDIA Risk Analytics Portfolio

The current portfolio uses observed Yahoo Finance prices and FRED DGS10 yields,
with archived source responses, adjusted returns and chronological VaR backtesting.

Read the [portfolio README](https://github.com/liangchen-andy/nvda_risk_project#readme),
[verified results](https://github.com/liangchen-andy/nvda_risk_project/blob/a7fb4a694f2191157d21a4005bd5c6a9291819de/documents/verified/RESULTS.md)
and [data audit](https://github.com/liangchen-andy/nvda_risk_project/blob/a7fb4a694f2191157d21a4005bd5c6a9291819de/documents/DATA_AUDIT.md).

The original documentation pages and root PDF exports describe a legacy synthetic
demonstration. Its figures, diagnostics and old numerical results are preserved for
traceability; they are not observed NVIDIA risk findings. Passing consistency checks
did not establish authenticity of those inputs.

The current observed-data workflow is `scripts/verified_analysis.py` and has its own
Python 3.12 environment and GitHub Actions verification job. The legacy Pytask/Pixi
workflow remains separate.

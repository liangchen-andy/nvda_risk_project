# Pipeline Map

The top-level workflow is intentionally simple:

```text
raw market and macro sources
  -> cached and snapshot inputs
  -> cleaned daily panel
  -> aligned monthly panel
  -> risk metrics and diagnostics
  -> figures and tables
  -> paper and presentation artifacts
```

## Why This Matters

`pytask` tracks dependencies between these stages, so a change in one upstream task only
rebuilds the downstream artifacts that depend on it. That gives the project two useful
properties:

- reruns stay fast when only a small part of the pipeline changes,
- output files can be traced back to the code and data that produced them.

## Public Artifacts Produced By The Graph

- `documents/public/fig_volatility.png`
- `documents/public/fig_var_exceedances.png`
- `documents/public/fig_drawdown.png`
- `documents/public/fig_beta_rolling.png`
- `documents/tables/estimation_results.md`
- `documents/tables/diagnostics.md`

This makes the build graph part of the presentation story, not just an internal
engineering detail.

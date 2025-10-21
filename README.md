
# sm_attribution

Reproducing and strengthening the analysis on human-induced changes in global soil-moisture droughts.
Python-first reimplementation of the prior MATLAB workflow.

## Layout
- `src/sm_attribution/` — library code (io, preprocess, metrics, analysis, viz)
- `notebooks/` — exploratory notebooks (lightweight, call into `src/`)
- `scripts/` — small CLIs / figure makers that import from `src/`
- `tests/` — unit tests
- `configs/` — YAML/JSON configs (paths, grids, depth, thresholds)
- `figures/` — generated figures (outputs)
- `matlab_code/` — original reference code (read-only)
- `data/` — local data only (ignored)
- `documentation/` — docs and drafts (ignored)

## Getting started
1. Create/activate your environment.
2. Put local file paths into `configs/data_registry.yml`.
3. Run notebooks and scripts that import from `src/sm_attribution`.

# sm_attribution

Reproducing and strengthening the analysis on human-induced changes in
global soil-moisture droughts.  Python-first reimplementation of the prior
MATLAB workflow.

## Quick start

```bash
# 1. Create the conda environment
conda env create -f environment.yml
conda activate sm-attr-311

# 2. Install the package in editable mode
pip install -e .

# 3. Configure local paths
#    Edit configs/data_registry.yml — set paths.root to your data directory.

# 4. Run the full pipeline
chmod +x run_all.sh
./run_all.sh                     # all 4 steps
./run_all.sh --start-from 3     # resume from step 3
./run_all.sh --dry-run           # print commands without executing
```

## Data preprocessing

Before the analysis pipeline can run, raw soil-moisture files must be
homogenized to a common reference format.  This is handled by the
`preprocess` subpackage and two preparation scripts.

**What the preprocessing does:**

- **Temporal harmonization** — monthly means on a `proleptic_gregorian`
  calendar (first-of-month timestamps).
- **Spatial harmonization** — regridding to a uniform 0.5° lat/lon grid
  (exact block means for 0.1°/0.25°/0.05°/1-arcmin sources; xESMF bilinear
  fallback for irregular grids).
- **Depth harmonization** — conversion to an approximate 0–1 m soil-moisture
  equivalent (model-specific recipes using ancillary depth/landcover maps).
- **Grid alignment** — snapping coordinates to the canonical ISIMIP 0.5°
  land-mask grid (bit-identical lat/lon values).

**Model preprocessing** (`build_models_1m.py`):

Each of the 7 ISIMIP models has a dedicated depth recipe in
`src/sm_attribution/preprocess/depth_1m.py`:

| Model | Method |
|-------|--------|
| H08 | Scale by ancillary soil depth map: `f = min(1, D) / D` |
| HydroPy | Root-zone mass pass-through |
| JULES-W2 | Sum layers 1–3 |
| MIROC-INTEG-LAND | Sum layers 1–3 |
| WaterGAP2-2e | Scale by rooting depth from landcover ancillary |
| WEB-DHM-SG | Scale by SiB2 landcover total depth ancillary |
| LPJmL5-7-10-fire | Exact 0–1 m integration using `depth_bnds` |

```bash
python scripts/build_models_1m.py                         # all 7 models × 4 scenarios
python scripts/build_models_1m.py --models h08 jules-w2   # subset
```

**Observation preprocessing** (`build_observed_1m.py`):

Each of the 10 observational products has a tailored pipeline in
`src/sm_attribution/preprocess/observations.py`:

| Dataset | Native grid | Depth / variable | Notes |
|---------|-------------|------------------|-------|
| ERA5-Land | 0.1° | swvl1–3 → 0–1 m mass | Block mean 5×5 |
| GLEAM v4.2a | 0.1° | SMrz (root-zone volumetric) | Block mean 5×5 |
| GLEAM v4.2b | 0.1° | SMrz (root-zone volumetric) | Block mean 5×5 |
| GLDAS v2.0 | 0.25° | Sum of 0–10/10–40/40–100 cm | Block mean 2×2 |
| GLDAS v2.1 | 0.25° | Sum of 0–10/10–40/40–100 cm | Block mean 2×2 |
| SoMo.ml | 0.25° | Layers 1–3 (0–50 cm), depth-weighted | Block mean 2×2 |
| GRACE-DA-DM | 0.25° | Root-zone percentile (weekly→monthly) | Block mean 2×2 |
| MERRA-2 LAND | 0.5°×0.625° | 5% SFMC + 95% RZMC → mass | Linear interp lon |
| GDO-ENSMIA | 0.1° | Standardized anomaly (3rd dekad/month) | Block mean 5×5 |
| GDO-SMIA | ~1 arcmin | Standardized anomaly (last dekad/month) | Block mean 30×30 |

```bash
python scripts/build_observed_1m.py --dataset era5-land
python scripts/build_observed_1m.py --dataset gldas-v21
```

Outputs land under `data/models_1m/` and `data/observed_1m/` as compressed
NetCDFs with standardized variable names (`soilmoist_1m` or
`soilmoist_anom_std`).

## Pipeline

The analysis runs in four sequential steps.  Steps 2–4 are independent of
each other and only depend on the outputs of step 1.

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `orchestrate_ssi_drought_features.py` | Compute SSI (Standardized Soil-moisture Index) for 10 obs products × 7 ISIMIP models × 4 scenarios, then extract 12 drought features via theory-of-runs |
| 2 | `batch_run_correlations.py` | Per-pixel Pearson temporal correlations (model SSI vs obs SSI/anomaly), plus multi-model mean |
| 3 | `orchestrate_drought_feature_spatial_correlations.py` | Cos-lat weighted Spearman spatial correlations (Global + AR6 regions) between obs and model drought features |
| 4 | `orchestrate_drought_feature_ar6_metrics.py` | AR6-aggregated regional metrics: `spearman_rank`, `pearson_z`, `rmse_iqr` |

Use `run_all.sh` to run the full pipeline with resume support.

## Parallelism

Two levels of parallelism are available (configured in `configs/settings.yml`):

| Setting | What it controls | Default |
|---------|-----------------|---------|
| `dask.max_workers` | Dask worker processes (inner loop: per-pixel SSI/features) | `os.cpu_count() - 2` |
| `dask.concurrent_models` | Outer-loop threads via `ThreadPoolExecutor` | `1` (serial) |
| `dask.use_distributed` | `true` = LocalCluster (singleton); `false` = `scheduler="processes"` | `false` |

### Runtime output policy

Pipeline scripts now favor concise progress output during long server runs:

- Third-party infrastructure chatter (`distributed`, `tornado`, `bokeh`) is
  reduced to warnings/errors.
- Non-actionable warning spam (chunk-splitting and empty-slice warnings) is
  suppressed.
- HDF5 C-level error-stack dumps (`HDF5-DIAG`) are silenced via
  `H5Eset_auto2` and `HDF5_LOG_LEVEL=none` (these bypass Python's stderr).
- Long loops report periodic counters (completed/total) instead of per-item
  flood lines.
- Real errors and tracebacks are still preserved.

Use `run_all.sh` logs (`logs/step*.log`) for full run history.

**Cluster recipe** (e.g. 64-core server):

```yaml
# configs/settings.yml
dask:
  max_workers: null          # or override with DASK_NUM_WORKERS env var
  use_distributed: false     # MUST be false when concurrent_models > 1
  concurrent_models: 4       # 4 outer threads × ~15 inner workers = 60 cores
```

```bash
export DASK_NUM_WORKERS=15
./run_all.sh
```

> **Important:** When `concurrent_models > 1`, set `use_distributed: false`.
> Otherwise all threads share one Dask cluster and effectively serialize.

**Laptop recipe** (e.g. 8-core MacBook):

```yaml
dask:
  max_workers: null
  use_distributed: false
  concurrent_models: 1
```

## Drought features reference

The 12 drought features extracted per pixel (theory-of-runs method):

| Variable | Description |
|----------|-------------|
| `n_events` | Number of drought events |
| `duration_mean` | Mean event duration (months) |
| `duration_max` | Max event duration (months) |
| `severity_mean` | Mean cumulative severity |
| `severity_max` | Max cumulative severity |
| `intensity_mean` | Mean peak intensity |
| `intensity_max` | Max peak intensity |
| `ttm10` | Time-to-Moderate: months until SSI ≤ −1.0 |
| `tts15` | Time-to-Severe: months until SSI ≤ −1.5 |
| `tte20` | Time-to-Extreme: months until SSI ≤ −2.0 |
| `inter_arrival_mean` | Mean inter-arrival time (months) |
| `inter_arrival_cv` | CV of inter-arrival time |

## Project layout

```
src/sm_attribution/       library code (io, preprocess, metrics, analysis, viz)
scripts/                  CLI scripts (pipeline entry points)
notebooks/                exploratory notebooks (call into src/)
tests/                    unit tests
configs/                  YAML configs (paths, settings, thresholds)
data/                     local data (gitignored)
figures/                  generated figures (gitignored)
matlab_code/              original MATLAB reference code (read-only)
documentation/            docs and drafts (gitignored)
```

## Configuration

- **`configs/settings.yml`** — SSI method parameters, Dask parallelism, depth
  and grid settings.
- **`configs/data_registry.yml`** — Path templates for all data products,
  model/obs metadata, period definitions. All file paths are resolved through
  this registry — no hardcoded absolute paths in the codebase.

## Tests

```bash
python -m pytest tests/ -x -q
```

## License

See [LICENSE](LICENSE).
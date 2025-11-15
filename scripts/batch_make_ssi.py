#!/usr/bin/env python3
"""
Batch-generation of SSI for all models and selected observational datasets.

This script orchestrates the computation of nonparametric SSI (AghaKouchak-style)
for:
  * All ISIMIP models listed in MODELS, across all scenarios in the registry.
  * A curated list of observational products listed in OBS.

By default, the SSI scale (in months) is read from `configs/settings.yml`
under `ssi.scale_months`, but can be overridden via the CLI.

Examples
--------
# Use defaults from settings.yml (e.g. scale_months: 3), standalone ECDF,
# reference period 2003-01 to 2019-12:
python scripts/batch_make_ssi.py

# Same, but pooled reference across scenarios for each model:
python scripts/batch_make_ssi.py --mode pooled

# Compute 12-month SSI over a different reference window:
python scripts/batch_make_ssi.py --scale 12 --ref-start 2004-01 --ref-end 2019-12
"""

from __future__ import annotations

import argparse

from sm_attribution.io.registry import default_registry
from sm_attribution.io.settings import get_settings
from sm_attribution.analysis.ensemble import ensure_all_models, ensure_all_obs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Read global defaults (e.g. SSI scale) from configs/settings.yml
SET = get_settings()
DEFAULT_SCALE = int(SET.ssi.get("scale_months", 3))

# ISIMIP models used in the analysis
MODELS = [
    "h08",
    "hydropy",
    "jules-w2",
    "miroc-integ-land",
    "watergap2-2e",
    "web-dhm-sg",
    "lpjml5-7-10-fire",
]

# Observational products for which SSI is computed
OBS = [
    "era5land_1950_2020",
    "gleam42a_1980_2020",
    "gleam42a_2003_2020",
    "gleam42b_2003_2020",
    "gldas_v21_2000_2020",
    "somo_ml_0p5m_2000_2019",
    "merra2_1980_2020",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute SSI for all models and selected observational datasets."
    )
    parser.add_argument(
        "--mode",
        choices=["standalone", "pooled"],
        default="standalone",
        help=(
            "Reference mode for ECDF: 'standalone' uses only the target series; "
            "'pooled' combines all scenarios of a model for the ECDF."
        ),
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=DEFAULT_SCALE,
        help=(
            "SSI temporal scale in months (rolling accumulation). "
            f"Default is ssi.scale_months from settings.yml (currently {DEFAULT_SCALE})."
        ),
    )
    parser.add_argument(
        "--ref-start",
        default="2003-01",
        help="Reference period start (YYYY-MM). Default: 2003-01.",
    )
    parser.add_argument(
        "--ref-end",
        default="2019-12",
        help="Reference period end (YYYY-MM). Default: 2019-12.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mode = args.mode
    scale = args.scale
    ref_start = args.ref_start
    ref_end = args.ref_end

    reg = default_registry()
    scenarios = reg.scenarios()

    # Compute-or-skip SSI for all models/scenarios
    model_paths = ensure_all_models(
        MODELS,
        scenarios,
        reg=reg,
        scale=scale,
        ref_start=ref_start,
        ref_end=ref_end,
        mode=mode,
    )

    # Compute-or-skip SSI for all observations
    obs_paths = ensure_all_obs(
        OBS,
        reg=reg,
        scale=scale,
        ref_start=ref_start,
        ref_end=ref_end,
    )

    print(
        f"Models SSI (mode={mode}, scale={scale} months, "
        f"ref window={ref_start} to {ref_end}):"
    )
    for (m, s), p in sorted(model_paths.items()):
        print(f"  {m:18s} {s:20s} -> {p}")

    print("\nObs SSI:")
    for k, p in sorted(obs_paths.items()):
        print(f"  {k:25s} -> {p}")


if __name__ == "__main__":
    main()
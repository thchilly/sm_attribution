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
from typing import List, Optional

from sm_attribution.io.registry import default_registry
from sm_attribution.io.settings import get_settings
from sm_attribution.analysis.ensemble import ensure_all_models, ensure_all_obs
from sm_attribution.analysis.ssi import ALLOWED_SSI_METHODS, DEFAULT_SSI_METHOD

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Read global defaults (e.g. SSI scale) from configs/settings.yml
SET = get_settings()
DEFAULT_SCALE = int(SET.ssi.get("scale_months", 3))
DEFAULT_TAIL_QUANTILE = float(SET.ssi.get("hybrid_tail_quantile", 0.10))
DEFAULT_MIN_TAIL_SIZE = int(SET.ssi.get("hybrid_min_tail_size", 20))
DEFAULT_LOC = str(SET.ssi.get("hybrid_loc", "median"))
DEFAULT_SCALE_METHOD = str(SET.ssi.get("hybrid_scale_method", "iqr"))

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
# (only SSI/SM-type products, no anomaly-only datasets like GDO)
OBS = [
    "era5-land",
    "gleam-42a",
    "gleam-42b",
    "gldas-v21",
    "somo-ml",
    "merra2-land",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute SSI for all models and selected observational datasets."
    )
    parser.add_argument(
        "--mode",
        choices=["standalone", "pooled", "fixed"],
        default="standalone",
        help=(
            "Reference mode for ECDF: 'standalone' uses only the target series; "
            "'pooled' combines all scenarios of a model for the ECDF; "
            "'fixed' uses a single designated scenario (see --fixed-ref-scenario)."
        ),
    )
    parser.add_argument(
        "--pool-scenarios",
        nargs="+",
        default=None,
        help=(
            "Only used when --mode pooled. If provided, build the ECDF reference "
            "from ONLY these scenarios (e.g. obsclim_histsoc counterclim_histsoc). "
            "If omitted, pools across ALL registry scenarios (legacy behaviour)."
        ),
    )
    parser.add_argument(
        "--fixed-ref-scenario",
        default=None,
        help=(
            "Only used when --mode fixed. The single scenario whose soil moisture "
            "is used as the ECDF reference for all scenarios of the same model "
            "(e.g. obsclim_histsoc). Required when --mode=fixed."
        ),
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Scenarios to compute SSI for. Default: all scenarios in the registry.",
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
        "--period-mode",
        choices=["common", "maxspan"],
        default=None,
        help=(
            "Derive ref-start/ref-end from the registry automatically. "
            "'common' uses common_period (2003-01..2019-12). "
            "'maxspan' uses the intersection of obs and model windows "
            "(per obs dataset).  Overrides --ref-start / --ref-end."
        ),
    )
    parser.add_argument(
        "--ref-start",
        default="2003-01",
        help="Reference period start (YYYY-MM). Default: 2003-01. Ignored when --period-mode is set.",
    )
    parser.add_argument(
        "--ref-end",
        default="2019-12",
        help="Reference period end (YYYY-MM). Default: 2019-12. Ignored when --period-mode is set.",
    )
    parser.add_argument(
        "--ssi-method",
        choices=list(ALLOWED_SSI_METHODS),
        default=DEFAULT_SSI_METHOD,
        help=(
            "SSI computation method. 'monthwise_ecdf' is the canonical month-wise ECDF; "
            "'deseasonal_ecdf_gpd' uses deseasonalized ECDF with GPD tail completion. "
            f"Default: {DEFAULT_SSI_METHOD}."
        ),
    )
    parser.add_argument(
        "--tail-quantile",
        type=float,
        default=DEFAULT_TAIL_QUANTILE,
        help=(
            "Tail quantile for GPD fitting (deseasonal_ecdf_gpd only). "
            f"Default: {DEFAULT_TAIL_QUANTILE}."
        ),
    )
    parser.add_argument(
        "--min-tail-size",
        type=int,
        default=DEFAULT_MIN_TAIL_SIZE,
        help=(
            "Min exceedances for GPD tail fit (deseasonal_ecdf_gpd only). "
            f"Default: {DEFAULT_MIN_TAIL_SIZE}."
        ),
    )
    parser.add_argument(
        "--loc",
        choices=["median", "mean"],
        default=DEFAULT_LOC,
        help=(
            "Location estimator for deseasonalization (deseasonal_ecdf_gpd only). "
            f"Default: {DEFAULT_LOC}."
        ),
    )
    parser.add_argument(
        "--scale-method",
        choices=["iqr", "std"],
        default=DEFAULT_SCALE_METHOD,
        help=(
            "Scale estimator for deseasonalization (deseasonal_ecdf_gpd only). "
            f"Default: {DEFAULT_SCALE_METHOD}."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mode = args.mode
    scale = args.scale
    ssi_method = args.ssi_method

    reg = default_registry()
    scenarios = args.scenarios if args.scenarios is not None else list(reg.scenarios())

    period_mode = args.period_mode  # None, "common", or "maxspan"

    # GPD-specific kwargs (ignored by monthwise_ecdf)
    gpd_kw = dict(
        tail_quantile=args.tail_quantile,
        min_tail_size=args.min_tail_size,
        loc=args.loc,
        scale_method=args.scale_method,
    )

    # -----------------------------------------------------------------
    # Resolve reference period(s)
    # -----------------------------------------------------------------
    if period_mode is not None:
        # Registry-driven: one ref window per obs dataset
        # (for "common" all obs share the same window; for "maxspan"
        # each obs gets its own window).

        # --- Models ---
        model_paths = {}
        for obs_key in OBS:
            ref_start, ref_end = reg.resolve_ref_period(obs_key, period_mode)
            mp = ensure_all_models(
                MODELS,
                scenarios,
                reg=reg,
                scale=scale,
                ref_start=ref_start,
                ref_end=ref_end,
                mode=mode,
                pool_scenarios=args.pool_scenarios,
                fixed_ref_scenario=args.fixed_ref_scenario,
                ssi_method=ssi_method,
                **gpd_kw,
            )
            model_paths.update(mp)

        # --- Obs ---
        obs_paths = {}
        for obs_key in OBS:
            ref_start, ref_end = reg.resolve_ref_period(obs_key, period_mode)
            op = ensure_all_obs(
                [obs_key],
                reg=reg,
                scale=scale,
                ref_start=ref_start,
                ref_end=ref_end,
                ssi_method=ssi_method,
                **gpd_kw,
            )
            obs_paths.update(op)
    else:
        # Legacy behaviour: single explicit window for everything
        ref_start = args.ref_start
        ref_end = args.ref_end

        model_paths = ensure_all_models(
            MODELS,
            scenarios,
            reg=reg,
            scale=scale,
            ref_start=ref_start,
            ref_end=ref_end,
            mode=mode,
            pool_scenarios=args.pool_scenarios,
            fixed_ref_scenario=args.fixed_ref_scenario,
            ssi_method=ssi_method,
            **gpd_kw,
        )

        obs_paths = ensure_all_obs(
            OBS,
            reg=reg,
            scale=scale,
            ref_start=ref_start,
            ref_end=ref_end,
            ssi_method=ssi_method,
            **gpd_kw,
        )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    pm_label = period_mode if period_mode else f"{args.ref_start}..{args.ref_end}"
    print(
        f"Models SSI (mode={mode}, method={ssi_method}, scale={scale} months, "
        f"period={pm_label}):"
    )
    for (m, s), p in sorted(model_paths.items()):
        print(f"  {m:18s} {s:20s} -> {p}")

    print("\nObs SSI:")
    for k, p in sorted(obs_paths.items()):
        print(f"  {k:25s} -> {p}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Batch compute drought features from SSI products.

Supports:
- model SSI (standalone or pooled)
- observed SSI

Key principle: SSI files are located via the registry templates,
not hard-coded filenames.

Examples
--------
# pooled model SSI (2-scenario pool), compute features on full 1901-2019:
python scripts/batch_make_drought_features.py \
  --models h08 hydropy jules-w2 miroc-integ-land watergap2-2e web-dhm-sg lpjml5-7-10-fire \
  --scenarios obsclim_histsoc counterclim_histsoc \
  --ssi-mode pooled \
  --pool-scenarios obsclim_histsoc counterclim_histsoc \
  --scale 3 \
  --ref-start 1901-01 --ref-end 2019-12 \
  --feat-start 1901-01 --feat-end 2019-12

# observed SSI:
python scripts/batch_make_drought_features.py \
  --obs era5-land gleam-42b gldas-v21 somo-ml merra2-land \
  --scale 3 --ref-start 2003-01 --ref-end 2019-12 \
  --feat-start 2003-01 --feat-end 2019-12
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from sm_attribution.io.registry import default_registry
from sm_attribution.analysis.drought_features import (
    ensure_drought_features_model,
    ensure_drought_features_obs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute drought feature maps from SSI files.")

    # Targets
    p.add_argument("--models", nargs="+", default=None, help="Model names to process.")
    p.add_argument("--scenarios", nargs="+", default=None, help="Scenarios to process for models.")
    p.add_argument("--obs", nargs="+", default=None, help="Observed dataset keys to process.")

    # SSI identification (must match how SSI was produced)
    p.add_argument("--ssi-mode", choices=["standalone", "pooled"], default="standalone")
    p.add_argument(
        "--pool-scenarios",
        nargs="+",
        default=None,
        help="When --ssi-mode pooled: scenarios included in the pool_id (order irrelevant).",
    )
    p.add_argument("--scale", type=int, default=3)
    p.add_argument("--ref-start", default="2003-01")
    p.add_argument("--ref-end", default="2019-12")

    # Feature window (what period to analyze for drought events)
    p.add_argument("--feat-start", default=None, help="Feature window start (YYYY-MM).")
    p.add_argument("--feat-end", default=None, help="Feature window end (YYYY-MM).")

    # Matlab-like thresholds
    p.add_argument("--bridge-low", type=float, default=0.0)
    p.add_argument("--bridge-high", type=float, default=1.0)
    p.add_argument("--bridge-len", type=int, default=3, help="Bridge runs shorter than this (months).")
    p.add_argument("--severity-threshold", type=float, default=-1.0, help="Event must reach <= this.")
    p.add_argument("--drought-threshold", type=float, default=0.0, help="Drought months are < this.")
    p.add_argument("--ttm10-threshold", type=float, default=-1.0, help="TTM10 SSI threshold.")
    p.add_argument("--tts15-threshold", type=float, default=-1.5, help="TTS15 SSI threshold.")
    p.add_argument("--tte20-threshold", type=float, default=-2.0, help="TTE20 SSI threshold.")

    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reg = default_registry()

    # Defaults for feature window
    feat_start = args.feat_start if args.feat_start is not None else args.ref_start
    feat_end = args.feat_end if args.feat_end is not None else args.ref_end

    # Determine pool_id string for model SSI paths
    if args.ssi_mode == "pooled":
        if args.pool_scenarios is None or len(args.pool_scenarios) == 0:
            pool_id = "ALL_SCENARIOS"
        else:
            pool_id = "__".join(sorted(args.pool_scenarios))
    else:
        pool_id = "standalone"

    outputs: List[str] = []

    # Models
    if args.models and args.scenarios:
        for m in args.models:
            for s in args.scenarios:
                out = ensure_drought_features_model(
                    m,
                    s,
                    reg=reg,
                    ssi_mode=args.ssi_mode,
                    pool_id=pool_id,
                    scale=args.scale,
                    ref_start=args.ref_start,
                    ref_end=args.ref_end,
                    feat_start=feat_start,
                    feat_end=feat_end,
                    bridge_low=args.bridge_low,
                    bridge_high=args.bridge_high,
                    bridge_len_months=args.bridge_len,
                    severity_threshold=args.severity_threshold,
                    drought_threshold=args.drought_threshold,
                    ttm10_threshold=args.ttm10_threshold,
                    tts15_threshold=args.tts15_threshold,
                    tte20_threshold=args.tte20_threshold,
                    overwrite=args.overwrite,
                )
                outputs.append(out)

    # Observations
    if args.obs:
        for k in args.obs:
            out = ensure_drought_features_obs(
                k,
                reg=reg,
                scale=args.scale,
                ref_start=args.ref_start,
                ref_end=args.ref_end,
                feat_start=feat_start,
                feat_end=feat_end,
                bridge_low=args.bridge_low,
                bridge_high=args.bridge_high,
                bridge_len_months=args.bridge_len,
                severity_threshold=args.severity_threshold,
                drought_threshold=args.drought_threshold,
                ttm10_threshold=args.ttm10_threshold,
                tts15_threshold=args.tts15_threshold,
                tte20_threshold=args.tte20_threshold,
                overwrite=args.overwrite,
            )
            outputs.append(out)

    if not outputs:
        raise SystemExit("Nothing to do. Provide --models/--scenarios and/or --obs.")

    print("Wrote drought feature files:")
    for pth in outputs:
        print("  ", pth)


if __name__ == "__main__":
    main()
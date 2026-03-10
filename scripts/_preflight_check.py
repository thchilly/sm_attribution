#!/usr/bin/env python3
"""Pre-flight check: verify all paths, imports, and inputs before long runs."""

import os
import sys

from sm_attribution.io.registry import default_registry
from sm_attribution.analysis.ensemble import (
    ssi_model_path,
    ssi_obs_path,
    correlation_map_path,
    correlation_multimodel_map_path,
)
from sm_attribution.analysis.ssi import DEFAULT_SSI_METHOD, ALLOWED_SSI_METHODS


def main():
    reg = default_registry()
    ok = True

    # --- 1. Registry ---
    scenarios = reg.scenarios()
    print(f"Scenarios: {scenarios}")

    cp = reg.get_common_period()
    print(f"Common period: {cp[0]} .. {cp[1]}")

    cs = reg.corr_start_from_ref(cp[0], 3)
    print(f"Corr start (scale=3): {cs}")
    print()

    # --- 2. SSI model paths (fixed mode) ---
    print("=== SSI MODEL PATHS (fixed, deseasonal_ecdf_gpd) ===")
    models = ["h08", "hydropy", "jules-w2", "miroc-integ-land",
              "watergap2-2e", "web-dhm-sg", "lpjml5-7-10-fire"]
    for model in models:
        for scen in scenarios:
            p = ssi_model_path(
                model, scen, reg=reg, scale=3,
                ref_start="2003-01", ref_end="2019-12",
                mode="fixed", ssi_method="deseasonal_ecdf_gpd",
                pool_id="obsclim_histsoc",
            )
            print(f"  {model:20s} {scen:20s} -> {p}")

    print()

    # --- 3. SSI obs paths ---
    print("=== SSI OBS PATHS (deseasonal_ecdf_gpd) ===")
    obs_ssi = ["era5-land", "gleam-42a", "gleam-42b", "gldas-v21",
               "somo-ml", "merra2-land", "grace-da-dm"]
    for o in obs_ssi:
        p = ssi_obs_path(
            o, reg=reg, scale=3,
            ref_start="2003-01", ref_end="2019-12",
            ssi_method="deseasonal_ecdf_gpd",
        )
        print(f"  {o:15s} -> {p}")

    print()

    # --- 4. Check raw model inputs exist ---
    print("=== RAW MODEL INPUTS ===")
    for model in models:
        for scen in scenarios:
            try:
                raw = reg.get_model_processed(model, scen)
                exists = os.path.exists(raw)
                tag = "OK" if exists else "MISSING"
                if not exists:
                    ok = False
                print(f"  {tag:7s} {model:20s} {scen:20s} -> {raw}")
            except Exception as e:
                print(f"  ERROR  {model:20s} {scen:20s} -> {e}")
                ok = False

    print()

    # --- 5. Check raw obs inputs exist ---
    print("=== RAW OBS INPUTS ===")
    all_obs = obs_ssi + ["gdo-ensmia", "gdo-smia"]
    for o in all_obs:
        try:
            raw = reg.get_obs_processed(o)
            exists = os.path.exists(raw)
            tag = "OK" if exists else "MISSING"
            if not exists:
                ok = False
            print(f"  {tag:7s} {o:15s} -> {raw}")
        except Exception as e:
            print(f"  ERROR  {o:15s} -> {e}")
            ok = False

    print()

    # --- 6. Correlation path templates ---
    print("=== CORRELATION PATH TEMPLATES ===")
    p_corr = correlation_map_path(
        "h08", "obsclim_histsoc", "era5-land",
        target="ssi", mode="fixed",
        corr_start="2003-04", corr_end="2019-12",
        ssi_method="deseasonal_ecdf_gpd", reg=reg,
    )
    print(f"  Corr map:    {p_corr}")

    p_mm = correlation_multimodel_map_path(
        "obsclim_histsoc", "era5-land",
        target="ssi", mode="fixed",
        corr_start="2003-04", corr_end="2019-12",
        ssi_method="deseasonal_ecdf_gpd", reg=reg,
    )
    print(f"  Corr MM map: {p_mm}")

    print()

    # --- 7. batch_run_correlations.py import test ---
    print("=== BATCH_RUN_CORRELATIONS IMPORT ===")
    try:
        # just verify it parses
        import ast
        with open("scripts/batch_run_correlations.py") as f:
            ast.parse(f.read())
        print("  Syntax OK")

        # verify key imports
        from sm_attribution.metrics.correlation import pearson_map
        from sm_attribution.io.load_mask import load_isimip_landmask
        print("  All imports OK")
    except Exception as e:
        print(f"  FAILED: {e}")
        ok = False

    print()
    if ok:
        print("ALL CHECKS PASSED. Ready to run.")
    else:
        print("SOME CHECKS FAILED. Fix issues before running.")
        sys.exit(1)


if __name__ == "__main__":
    main()

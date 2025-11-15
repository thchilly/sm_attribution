#!/usr/bin/env python3
"""
Build homogenized observational soil-moisture datasets (monthly, 0.5°, ~0-1 m).

This script processes a collection of observational and reanalysis
soil-moisture products and converts each into a common reference format
suitable for cross-dataset comparison, anomaly computation, and model
evaluation.  The homogenization includes:

    • Temporal harmonization:      conversion to monthly means
    • Spatial harmonization:       regridding to a uniform 0.5° lat/lon grid
    • Depth harmonization:         conversion to an approximate 0-1 m
                                   soil-moisture equivalent (where possible)
    • Grid alignment:              snapping to the canonical ISIMIP 0.5° grid
    • Calendar normalization:      enforcing a proleptic_gregorian monthly axis

Different source datasets provide soil moisture at different depths,
temporal resolutions, and native grids (0.05°, 0.1°, 0.25°, 0.5°).
Each dataset is processed using its dedicated pipeline defined in
`sm_attribution.preprocess.observations`, ensuring consistent units,
coordinate definitions, metadata, and land-mask boundaries.

The resulting datasets are written under `observed_1m/` as NetCDF files
with standardized variable names (`soilmoist_1m` or `soilmoist_anom_std`),
compression, fill values, and CF-compliant metadata.

---
USAGE
-----

Build a specific dataset:

    python scripts/build_observed_1m.py --dataset era5-land

Specify a custom data registry:

    python scripts/build_observed_1m.py --dataset gldas-v21 \
                                        --registry configs/data_registry.yml

---
SUPPORTED DATASET KEYS
----------------------

    era5-land
    gleam-42a
    gleam-42b
    gldas-v20
    gldas-v21
    somo-ml
    grace-da-dm
    merra2-land
    gdo-ensmia
    gdo-smia

Each dataset's processing pipeline is tailored to its native resolution,
vertical structure, temporal frequency, and data format (NetCDF, GeoTIFF).

---
OUTPUT
------

Each invocation produces a single NetCDF file with:

    • Dimensions:   time × lat × lon
    • Grid:         canonical ISIMIP 0.5° grid
    • Calendar:     proleptic_gregorian, monthly timestamps (MS)
    • Variables:    soilmoist_1m or soilmoist_anom_std
    • Encoding:     float32, zlib compression, fill value -9999
    • Chunking:     (12, 180, 360)

"""

from __future__ import annotations

import argparse
import numpy as np

from sm_attribution.io.registry import Registry
from sm_attribution.preprocess.observations import (
    era5land_to_1m_monthly_halfdeg_v1,
    gleam42a_1980_2020_v0,
    gleam42b_2003_2020_v0,
    gldas_v20_to_1m_monthly_halfdeg_v0,
    gldas_v21_to_1m_monthly_halfdeg_v0,
    somoml_to_0p5m_monthly_halfdeg_v0,
    gracedadm_rootzone_to_monthly_halfdeg_v0,
    merra2_land_to_1m_monthly_halfdeg_v1,
    gdo_ensmia_to_monthly_halfdeg_v0,
    gdo_smia_to_monthly_halfdeg_v0,
)

# Supported dataset keys for the CLI (must match data_registry.yml observed_1m keys)
DATASET_CHOICES = [
    "era5-land",
    "gleam-42a",
    "gleam-42b",
    "gldas-v20",
    "gldas-v21",
    "somo-ml",
    "grace-da-dm",
    "merra2-land",
    "gdo-ensmia",
    "gdo-smia",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build observed datasets (monthly 0.5°)"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASET_CHOICES,
        help="Dataset key to build (see DATASET_CHOICES).",
    )
    parser.add_argument(
        "--registry",
        default="configs/data_registry.yml",
        help="Path to data registry YAML file.",
    )
    args = parser.parse_args()

    reg = Registry(args.registry)

    ds_out = None

    if args.dataset == "era5-land":
        da = era5land_to_1m_monthly_halfdeg_v1(reg)
        out_path = reg.get_obs_processed("era5-land")
        ds_out = da.to_dataset(name="soilmoist_1m")

    elif args.dataset == "gleam-42a":
        da = gleam42a_1980_2020_v0(reg)
        out_path = reg.get_obs_processed("gleam-42a")
        ds_out = da.to_dataset(name="soilmoist_1m")

    elif args.dataset == "gleam-42b":
        da = gleam42b_2003_2020_v0(reg)
        out_path = reg.get_obs_processed("gleam-42b")
        ds_out = da.to_dataset(name="soilmoist_1m")

    elif args.dataset == "gldas-v20":
        da = gldas_v20_to_1m_monthly_halfdeg_v0(reg)
        out_path = reg.get_obs_processed("gldas-v20")
        ds_out = da.to_dataset(name="soilmoist_1m")

    elif args.dataset == "gldas-v21":
        da = gldas_v21_to_1m_monthly_halfdeg_v0(reg)
        out_path = reg.get_obs_processed("gldas-v21")
        ds_out = da.to_dataset(name="soilmoist_1m")

    elif args.dataset == "somo-ml":
        ds_out = somoml_to_0p5m_monthly_halfdeg_v0(reg)
        out_path = reg.get_obs_processed("somo-ml")

    elif args.dataset == "grace-da-dm":
        da = gracedadm_rootzone_to_monthly_halfdeg_v0(reg)
        out_path = reg.get_obs_processed("grace-da-dm")
        ds_out = da.to_dataset(name="rootzone_percentile")

    elif args.dataset == "merra2-land":
        da = merra2_land_to_1m_monthly_halfdeg_v1(reg)
        out_path = reg.get_obs_processed("merra2-land")
        ds_out = da.to_dataset(name="soilmoist_1m")

    elif args.dataset == "gdo-ensmia":
        da = gdo_ensmia_to_monthly_halfdeg_v0(reg)
        out_path = reg.get_obs_processed("gdo-ensmia")
        ds_out = da.to_dataset(name="soilmoist_anom_std")

    elif args.dataset == "gdo-smia":
        da = gdo_smia_to_monthly_halfdeg_v0(reg)
        out_path = reg.get_obs_processed("gdo-smia")
        ds_out = da.to_dataset(name="soilmoist_anom_std")

    else:
        # Defensive, should not be reached due to argparse choices
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if ds_out is not None:
        # NetCDF encoding: compression + fill values per primary variable
        if args.dataset == "grace-da-dm":
            var_name = "rootzone_percentile"
        elif args.dataset in ("gdo-ensmia", "gdo-smia"):
            var_name = "soilmoist_anom_std"
        else:
            var_name = "soilmoist_1m"

        encoding = {
            var_name: {
                "dtype": "float32",
                "zlib": True,
                "complevel": 4,
                "_FillValue": np.float32(-9999.0),
                "chunksizes": (12, 180, 360),
            }
        }

        ds_out.to_netcdf(out_path, encoding=encoding)
        print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
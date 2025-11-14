#!/usr/bin/env python3
"""
Build homogenized observational soil-moisture datasets (monthly, 0.5°, ~0–1 m).

This script processes a collection of observational and reanalysis
soil-moisture products and converts each into a common reference format
suitable for cross-dataset comparison, anomaly computation, and model
evaluation.  The homogenization includes:

    • Temporal harmonization:      conversion to monthly means
    • Spatial harmonization:       regridding to a uniform 0.5° lat/lon grid
    • Depth harmonization:         conversion to an approximate 0–1 m
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

    python scripts/build_observed_1m.py --dataset era5land_1950_2020

Specify a custom data registry:

    python scripts/build_observed_1m.py --dataset gldas_v21_2000_2020 \
                                        --registry configs/data_registry.yml

---
SUPPORTED DATASET KEYS
----------------------

    era5land_1950_2020
    gleam42a_1980_2020
    gleam42a_2003_2020
    gleam42b_2003_2020
    gldas_v20_1948_2014
    gldas_v21_2000_2020
    somo_ml_0p5m_2000_2019
    gracedadm_2003_2020
    merra2_1980_2020
    gdo_ensmia_2001_2020
    gdo_smia_1995_2020

Each dataset’s processing pipeline is tailored to its native resolution,
vertical structure, temporal frequency, and data format (NetCDF, GeoTIFF).

---
OUTPUT
------

Each invocation produces a single NetCDF file with:

    • Dimensions:   time × lat × lon
    • Grid:         canonical ISIMIP 0.5° grid
    • Calendar:     proleptic_gregorian, monthly timestamps (MS)
    • Variables:    soilmoist_1m or soilmoist_anom_std
    • Encoding:     float32, zlib compression, fill value −9999
    • Chunking:     (12, 180, 360)

"""

from __future__ import annotations

import argparse
import numpy as np

from sm_attribution.io.registry import Registry
from sm_attribution.preprocess.observations import (
    era5land_to_1m_monthly_halfdeg_v1,
    gleam42a_1980_2020_v0,
    gleam42a_2003_2020_v0,
    gleam42b_2003_2020_v0,
    gldas_v20_to_1m_monthly_halfdeg_v0,
    gldas_v21_to_1m_monthly_halfdeg_v0,
    somoml_to_0p5m_monthly_halfdeg_v0,
    gracedadm_rootzone_to_monthly_halfdeg_v0,
    merra2_land_to_1m_monthly_halfdeg_v1,
    gdo_ensmia_to_monthly_halfdeg_v0,
    gdo_smia_to_monthly_halfdeg_v0,
)

# Supported dataset keys for the CLI
DATASET_CHOICES = [
    "era5land_1950_2020",
    "gleam42a_1980_2020",
    "gleam42a_2003_2020",
    "gleam42b_2003_2020",
    "gldas_v20_1948_2014",
    "gldas_v21_2000_2020",
    "somo_ml_0p5m_2000_2019",
    "gracedadm_2003_2020",
    "merra2_1980_2020",
    "gdo_ensmia_2001_2020",
    "gdo_smia_1995_2020",
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

    if args.dataset == "era5land_1950_2020":
        da = era5land_to_1m_monthly_halfdeg_v1(reg)
        out_path = reg.get_obs_processed("era5land_1950_2020")
        ds_out = da.to_dataset(name="soilmoist_1m")

    elif args.dataset == "gleam42a_1980_2020":
        da = gleam42a_1980_2020_v0(reg)
        out_path = reg.get_obs_processed("gleam42a_1980_2020")
        ds_out = da.to_dataset(name="soilmoist_1m")

    elif args.dataset == "gleam42a_2003_2020":
        da = gleam42a_2003_2020_v0(reg)
        out_path = reg.get_obs_processed("gleam42a_2003_2020")
        ds_out = da.to_dataset(name="soilmoist_1m")

    elif args.dataset == "gleam42b_2003_2020":
        da = gleam42b_2003_2020_v0(reg)
        out_path = reg.get_obs_processed("gleam42b_2003_2020")
        ds_out = da.to_dataset(name="soilmoist_1m")

    elif args.dataset == "gldas_v20_1948_2014":
        da = gldas_v20_to_1m_monthly_halfdeg_v0(reg)
        out_path = reg.get_obs_processed("gldas_v20_1948_2014")
        ds_out = da.to_dataset(name="soilmoist_1m")

    elif args.dataset == "gldas_v21_2000_2020":
        da = gldas_v21_to_1m_monthly_halfdeg_v0(reg)
        out_path = reg.get_obs_processed("gldas_v21_2000_2020")
        ds_out = da.to_dataset(name="soilmoist_1m")

    elif args.dataset == "somo_ml_0p5m_2000_2019":
        ds_out = somoml_to_0p5m_monthly_halfdeg_v0(reg)
        out_path = reg.get_obs_processed("somo_ml_0p5m_2000_2019")

    elif args.dataset == "gracedadm_2003_2020":
        da = gracedadm_rootzone_to_monthly_halfdeg_v0(reg)
        out_path = reg.get_obs_processed("gracedadm_2003_2020")
        ds_out = da.to_dataset(name="rootzone_percentile")

    elif args.dataset == "merra2_1980_2020":
        da = merra2_land_to_1m_monthly_halfdeg_v1(reg)
        out_path = reg.get_obs_processed("merra2_1980_2020")
        ds_out = da.to_dataset(name="soilmoist_1m")

    elif args.dataset == "gdo_ensmia_2001_2020":
        da = gdo_ensmia_to_monthly_halfdeg_v0(reg)
        out_path = reg.get_obs_processed("gdo_ensmia_2001_2020")
        ds_out = da.to_dataset(name="soilmoist_anom_std")

    elif args.dataset == "gdo_smia_1995_2020":
        da = gdo_smia_to_monthly_halfdeg_v0(reg)
        out_path = reg.get_obs_processed("gdo_smia_1995_2020")
        ds_out = da.to_dataset(name="soilmoist_anom_std")

    else:
        # Defensive, should not be reached due to argparse choices
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if ds_out is not None:
        # NetCDF encoding: compression + fill values per primary variable
        if args.dataset == "gracedadm_2003_2020":
            var_name = "rootzone_percentile"
        elif args.dataset in ("gdo_ensmia_2001_2020", "gdo_smia_1995_2020"):
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
#!/usr/bin/env python3
from __future__ import annotations
import argparse
from sm_attribution.io.registry import Registry
import numpy as np
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
)

def main():
    ap = argparse.ArgumentParser(description="Build observed datasets (monthly 0.5°)")
    ap.add_argument(
        "--dataset",
        required=True,
        choices=[
            "era5land_1950_2020",
            "gleam42a_1980_2020",
            "gleam42a_2003_2020",
            "gleam42b_2003_2020",
            "gldas_v20_1948_2014",
            "gldas_v21_2000_2020",
            "somo_ml",
            "gracedadm_rootzone_2003_2020",
            "merra2_1980_2020",
            "gdo_ensmia_2001_2020",
        ],
    )
    ap.add_argument("--registry", default="configs/data_registry.yml")
    args = ap.parse_args()

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
    elif args.dataset == "somo_ml":
        ds_out = somoml_to_0p5m_monthly_halfdeg_v0(reg)
        out_path = reg.get_obs_processed("somo_ml_0p5m_2000_2019")
    elif args.dataset == "gracedadm_rootzone_2003_2020":
        da = gracedadm_rootzone_to_monthly_halfdeg_v0(reg)
        out_path = reg.get_obs_processed("gracedadm_2003_2020")
        ds_out = da.to_dataset(name="rootzone_percentile")
    elif args.dataset == "merra2_1980_2020":
        da = merra2_land_to_1m_monthly_halfdeg_v1(reg)
        out_path = reg.get_obs_processed("merra2_1980_2020")
        ds_out = da.to_dataset(name="soilmoist_1m")
    elif args.dataset == "gdo_ensmia_2001_2020":
        from sm_attribution.preprocess.observations import gdo_ensmia_to_monthly_halfdeg_v0
        da = gdo_ensmia_to_monthly_halfdeg_v0(reg)
        out_path = reg.get_obs_processed("gdo_ensmia_2001_2020")
        ds_out = da.to_dataset(name="soilmoist_anom_std")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if ds_out is not None:
        # netCDF encoding: compression + fill values
        if args.dataset == "gracedadm_rootzone_2003_2020":
            var_name = "rootzone_percentile"
        elif args.dataset == "gdo_ensmia_2001_2020":
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
#!/usr/bin/env python3
from __future__ import annotations
import argparse
from sm_attribution.io.registry import Registry
from sm_attribution.preprocess.observations import (
    era5land_to_1m_monthly_halfdeg_v1,
    gleam42a_1980_2020_v0,
    gleam42a_2003_2020_v0,
    gleam42b_2003_2020_v0,
)

def main():
    ap = argparse.ArgumentParser(description="Build observed datasets (monthly 0.5°)")
    ap.add_argument("--dataset", required=True,
                    choices=["era5land", "gleam42a_1980_2020", "gleam42a_2003_2020", "gleam42b_2003_2020"])
    ap.add_argument("--registry", default="configs/data_registry.yml")
    args = ap.parse_args()

    reg = Registry(args.registry)

    if args.dataset == "era5land":
        da = era5land_to_1m_monthly_halfdeg_v1(reg)
        out_path = reg.get_obs_processed("era5land")
    elif args.dataset == "gleam42a_1980_2020":
        da = gleam42a_1980_2020_v0(reg)
        out_path = reg.get_obs_processed("gleam42a_1980_2020")
    elif args.dataset == "gleam42a_2003_2020":
        da = gleam42a_2003_2020_v0(reg)
        out_path = reg.get_obs_processed("gleam42a_2003_2020")
    elif args.dataset == "gleam42b_2003_2020":
        da = gleam42b_2003_2020_v0(reg)
        out_path = reg.get_obs_processed("gleam42b_2003_2020")
    else:
        raise NotImplementedError(args.dataset)

    enc = {da.name: {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": -9999.0}}
    da.to_dataset().to_netcdf(out_path, encoding=enc)
    print(f"WROTE: {out_path}")

if __name__ == "__main__":
    main()
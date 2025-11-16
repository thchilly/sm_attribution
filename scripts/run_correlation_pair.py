#!/usr/bin/env python
"""
Compute a gridpoint correlation map between one model–scenario and one
observational product (either SSI/SSI-like or standardized anomaly).

The script:
  - Loads model SSI (or SSI-like) for a given (model, scenario).
  - Loads observational SSI/SSI-like OR standardized anomaly.
  - Applies the ISIMIP land mask (default: no Antarctica / Greenland).
  - Computes Pearson r, p-value, and sample size over a fixed correlation
    period (default 2004-01 to 2019-12).
  - Writes the result to the path given by `metrics.correlations_map`
    in `configs/data_registry.yml`.

Typical usage
-------------
# SSI vs SSI (3-month scale, standalone)
python scripts/run_correlation_pair.py \
    --model h08 \
    --scenario obsclim_histsoc \
    --obs era5-land \
    --target ssi

# SSI vs GDO anomaly (3-month SSI for models, standardized anomaly for obs)
python scripts/run_correlation_pair.py \
    --model h08 \
    --scenario obsclim_histsoc \
    --obs gdo-ensmia \
    --target anomaly
"""

from __future__ import annotations

import argparse
import os

import xarray as xr

from sm_attribution.io.registry import default_registry
from sm_attribution.io.settings import get_settings
from sm_attribution.analysis.ensemble import ssi_model_path, ssi_obs_path
from sm_attribution.io.load_mask import load_isimip_landmask
from sm_attribution.metrics.correlation import pearson_map


# ---------------------------------------------------------------------------
# Configuration defaults (SSI scale + correlation period) from settings
# ---------------------------------------------------------------------------

SET = get_settings()
SSI_SCALE_DEFAULT = int(SET.ssi.get("scale_months", 3))

# SSI reference period (used in filenames and SSI computation)
SSI_REF_START_DEFAULT = "2003-01"
SSI_REF_END_DEFAULT = "2019-12"

# Correlation period (after rolling scale=3, 2004-01..2019-12 makes sense)
CORR_START_DEFAULT = "2004-01"
CORR_END_DEFAULT = "2019-12"


def _format_from_template(tmpl: str, **kw) -> str:
    """
    Expand {paths.*} placeholders and format the remaining fields.
    """
    paths = kw.pop("paths", {})
    for k, v in paths.items():
        tmpl = tmpl.replace("{paths." + k + "}", v)
    return tmpl.format(**kw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute model–obs correlation maps on the canonical ISIMIP grid."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Hydrological model name (e.g. 'h08', 'jules-w2').",
    )
    parser.add_argument(
        "--scenario",
        required=True,
        choices=[
            "obsclim_histsoc",
            "counterclim_histsoc",
            "obsclim_1901soc",
            "counterclim_1901soc",
        ],
        help="Scenario name as used in the data registry.",
    )
    parser.add_argument(
        "--obs",
        required=True,
        help=(
            "Observed dataset key (e.g. 'era5-land', 'gleam-42a', "
            "'gldas-v21', 'somo-ml', 'merra2-land', 'gdo-ensmia', 'gdo-smia')."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["standalone", "pooled"],
        default="standalone",
        help="SSI mode for models (standalone vs pooled reference ECDF).",
    )
    parser.add_argument(
        "--target",
        choices=["ssi", "anomaly"],
        default="ssi",
        help=(
            "Correlation target for the observed dataset: "
            "'ssi' uses SSI/SSI-like; 'anomaly' uses standardized anomaly (e.g. GDO)."
        ),
    )
    parser.add_argument(
        "--n-min",
        type=int,
        default=60,
        help="Minimum number of valid time samples required to keep a correlation.",
    )
    parser.add_argument(
        "--landmask-key",
        default="isimip_no_ant_nogreenland",
        help="Landmask key in the registry (default: isimip_no_ant_nogreenland).",
    )
    args = parser.parse_args()

    reg = default_registry()

    # Load land mask
    land = load_isimip_landmask(args.landmask_key)

    # ------------------------------------------------------------------
    # Load model SSI (or SSI-like) on canonical 0.5° grid
    # ------------------------------------------------------------------
    m_path = ssi_model_path(
        args.model,
        args.scenario,
        reg=reg,
        scale=SSI_SCALE_DEFAULT,
        ref_start=SSI_REF_START_DEFAULT,
        ref_end=SSI_REF_END_DEFAULT,
        mode=args.mode,
    )
    da_m = xr.open_dataset(m_path)["ssi"]

    # ------------------------------------------------------------------
    # Load observations: SSI/SSI-like or standardized anomaly
    # ------------------------------------------------------------------
    if args.target == "ssi":
        # Observed SSI / SSI-like product
        o_path = ssi_obs_path(
            args.obs,
            reg=reg,
            scale=SSI_SCALE_DEFAULT,
            ref_start=SSI_REF_START_DEFAULT,
            ref_end=SSI_REF_END_DEFAULT,
        )
        da_o = xr.open_dataset(o_path)["ssi"]
    else:
        # Observed standardized anomaly (e.g. GDO SMIA / ENSMIA)
        o_path = reg.get_obs_processed(args.obs)
        ds_o = xr.open_dataset(o_path)

        var = "soilmoist_anom_std"
        if var not in ds_o:
            # Fallback: first time–lat–lon variable
            var = next(
                k
                for k in ds_o.data_vars
                if {"time", "lat", "lon"}.issubset(ds_o[k].dims)
            )
        da_o = ds_o[var]

        # Convert sentinel fill values to NaN if present
        fv = da_o.attrs.get("_FillValue", None)
        if fv is None and "missing_value" in da_o.attrs:
            fv = ds_o[var].attrs["missing_value"]
        if fv is not None:
            da_o = da_o.where(da_o != fv)

    # Optional light logging
    print("Model SSI:", da_m.dims, {k: da_m.sizes[k] for k in da_m.dims})
    print("Obs     :", da_o.dims, {k: da_o.sizes[k] for k in da_o.dims})
    print("Landmask:", land.dims, {k: land.sizes[k] for k in land.dims})

    # ------------------------------------------------------------------
    # Compute Pearson correlation over land, intersection only
    # ------------------------------------------------------------------
    ds_corr = pearson_map(
        da_m,
        da_o,
        land,
        period_start=CORR_START_DEFAULT,
        period_end=CORR_END_DEFAULT,
        n_min=args.n_min,
        time_name="time",
    )

    # ------------------------------------------------------------------
    # Build output path from registry template and write to disk
    # ------------------------------------------------------------------
    tmpl = reg.cfg_dict["metrics"]["correlations_map"]

    obs_short = args.obs  # keys are already "short" (no year suffix)
    corrstart_yr = CORR_START_DEFAULT[:4]
    corrend_yr = CORR_END_DEFAULT[:4]

    out_path = _format_from_template(
        tmpl,
        paths=reg.cfg_dict["paths"],
        mode=args.mode,
        target=args.target,
        obs_short=obs_short,
        model=args.model,
        scenario=args.scenario,
        corrstart_yr=corrstart_yr,
        corrend_yr=corrend_yr,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Annotate provenance
    ds_corr.attrs.update(
        {
            "model": args.model,
            "scenario": args.scenario,
            "obs": args.obs,
            "ssi_mode": args.mode,
            "target": args.target,
            "model_path": m_path,
            "obs_path": o_path,
            "corr_period": f"{CORR_START_DEFAULT}:{CORR_END_DEFAULT}",
            "ssi_scale_months": SSI_SCALE_DEFAULT,
            "ssi_ref_period": f"{SSI_REF_START_DEFAULT}:{SSI_REF_END_DEFAULT}",
        }
    )

    comp = dict(zlib=True, complevel=4, shuffle=True)
    enc = {k: comp for k in ds_corr.data_vars}
    ds_corr.to_netcdf(out_path, encoding=enc)
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
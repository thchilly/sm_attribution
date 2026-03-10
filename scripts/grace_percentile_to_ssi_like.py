#!/usr/bin/env python
#scripts/grace_percentile_to_ssi_like.py

"""
Convert GRACE-DA-DM monthly root-zone percentiles to an SSI-like product (z-scores),
optionally applying a multi-month window so it is comparable to model SSI.

The script reads a preprocessed GRACE percentile product from the `observed_1m`
registry entry (e.g. key 'grace-da-dm') and writes a NetCDF with variable `ssi`
to the standard SSI-observed location (via `ssi_obs_path`).

Example
-------
python scripts/grace_percentile_to_ssi_like.py \
    --obskey grace-da-dm \
    --scale 3 \
    --ref-start 2003-01 --ref-end 2019-12
"""

from __future__ import annotations

import argparse
import xarray as xr
from scipy.stats import norm

from sm_attribution.io.registry import default_registry
from sm_attribution.io.settings import get_settings
from sm_attribution.analysis.ensemble import ssi_obs_path  # registry-based output path


SETTINGS = get_settings()


def _default_scale_from_settings() -> int:
    """
    Return the default SSI scale (months) from settings.yml, falling back to 3
    if the key is missing or malformed.
    """
    ssi_cfg = getattr(SETTINGS, "ssi", None)
    if ssi_cfg is None:
        return 3
    if isinstance(ssi_cfg, dict):
        return int(ssi_cfg.get("scale_months", 3))
    return int(getattr(ssi_cfg, "scale_months", 3))


def _load_grace_monthly_percentile(
    path: str,
    var: str = "rootzone_percentile",
) -> xr.DataArray:
    """Load GRACE monthly root-zone percentile field as (time, lat, lon) DataArray."""
    ds = xr.open_dataset(path)

    # Select variable
    if var in ds:
        da = ds[var]
    else:
        # Fallback: first variable with (time, lat, lon) dimensions
        candidates = [
            k
            for k in ds.data_vars
            if {"time", "lat", "lon"}.issubset(ds[k].dims)
        ]
        if not candidates:
            raise KeyError(f"No (time, lat, lon) percentile variable found in {path}.")
        da = ds[candidates[0]]

    # Normalise dimension names
    ren: dict[str, str] = {}
    if "latitude" in da.dims:
        ren["latitude"] = "lat"
    if "longitude" in da.dims:
        ren["longitude"] = "lon"
    if ren:
        da = da.rename(ren)

    # Replace sentinel fill values with NaN if present
    fv = da.attrs.get("_FillValue", None)
    if fv is None and "missing_value" in da.attrs:
        fv = da.attrs["missing_value"]
    if fv is not None:
        da = da.where(da != fv)

    da = da.transpose("time", "lat", "lon", missing_dims="ignore").astype("float32")
    return da


def _percentile_to_z(perc_da: xr.DataArray) -> xr.DataArray:
    """
    Transform GRACE percentiles to z-scores using the standard normal CDF.

    Percentiles are clipped to [0.5, 99.5] to avoid infinities from norm.ppf.
    """
    p = perc_da.clip(0.5, 99.5) / 100.0
    z = xr.apply_ufunc(norm.ppf, p, dask="allowed").astype("float32")
    z.name = "ssi"
    z.attrs.update(
        {
            "long_name": "GRACE root-zone percentile transformed to z-score",
            "units": "-",
            "note": "percentile → z via norm.ppf(P/100); clipped to [0.5, 99.5] first",
        }
    )
    return z


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert GRACE root-zone percentiles to an SSI-like z-score product."
    )
    parser.add_argument(
        "--obskey",
        required=True,
        help="Registry key for the GRACE monthly percentile file "
             "(e.g. 'grace-da-dm').",
    )
    parser.add_argument(
        "--var",
        default="rootzone_percentile",
        help="Variable name in GRACE file (default: rootzone_percentile).",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=_default_scale_from_settings(),
        help=(
            "Temporal window (months). "
            "Default is taken from settings.yml (ssi.scale_months)."
        ),
    )
    parser.add_argument(
        "--ref-start",
        default="2003-01",
        help="Reference-period start (YYYY-MM). Used for slicing and metadata.",
    )
    parser.add_argument(
        "--ref-end",
        default="2019-12",
        help="Reference-period end (YYYY-MM). Used for slicing and metadata.",
    )
    parser.add_argument(
        "--engine",
        default=None,
        help="Optional netCDF engine passed to xarray.to_netcdf.",
    )
    args = parser.parse_args()

    reg = default_registry()
    in_path = reg.get_obs_processed(args.obskey)

    # Load monthly percentiles and convert to z-scores
    da_p = _load_grace_monthly_percentile(in_path, var=args.var)
    da_z = _percentile_to_z(da_p)

    # Optional rolling window in time (e.g. N-month mean from settings or CLI)
    if args.scale and args.scale > 1:
        da_z = da_z.rolling(time=args.scale, min_periods=args.scale).mean()

    # Restrict to requested reference period
    da_z = da_z.sel(time=slice(args.ref_start, args.ref_end))

    # Construct output path using standard SSI observed template
    out_path = ssi_obs_path(
        args.obskey,
        reg=reg,
        scale=args.scale,
        ref_start=args.ref_start,
        ref_end=args.ref_end,
    )

    # Update attributes with SSI metadata
    da_z.attrs.update(
        {
            "standard_name": "standardized_soil_moisture_index",
            "long_name": "Standardized Soil Moisture Index "
                         "(z from GRACE root-zone percentile)",
            "method": (
                "GRACE monthly root-zone percentile converted to z-score, "
                f"then {args.scale}-month rolling mean (if scale>1)."
            ),
            "ssi_scale": args.scale,
            "ssi_ref_period": f"{args.ref_start}:{args.ref_end}",
            "ssi_ref_description": (
                "No re-estimation of ECDF; native GRACE percentile climatology "
                "converted to z-scores, then a monthly window applied for SSI comparability."
            ),
            "source": str(in_path),
            "ssi_mode": "ssi-like-from-percentile",
        }
    )

    ds_out = da_z.to_dataset(name="ssi")
    comp = dict(zlib=True, complevel=4, shuffle=True)
    encoding = {k: comp for k in ds_out.data_vars}
    ds_out.to_netcdf(out_path, engine=args.engine, encoding=encoding)

    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
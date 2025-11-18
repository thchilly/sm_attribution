# src/sm_attribution/analysis/ar6_regions.py
"""
Utilities for aggregating gridded fields to AR6 reference regions.

Uses `regionmask`'s built-in AR6 land regions and returns both:
- regional means (1-D over region), and
- a 2-D field on the original grid with each region filled by its mean.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import xarray as xr
import regionmask


def get_ar6_land_regions() -> regionmask.Regions:
    """Return the AR6 land regions definition from regionmask."""
    return regionmask.defined_regions.ar6.land  # type: ignore[attr-defined]


def ar6_mean_and_field(
    da: xr.DataArray,
    *,
    min_valid: int = 1,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Aggregate a lat–lon field to AR6 land regions and expand back
    to a 2-D field.

    Parameters
    ----------
    da : xr.DataArray
        2-D field with dimensions ('lat', 'lon'). Typically correlation 'r'.
    min_valid : int, optional
        Minimum number of valid grid cells required for a region mean
        to be considered valid. Regions with fewer valid cells are set
        to NaN.

    Returns
    -------
    reg_mean : xr.DataArray
        1-D array of regional means with dimension 'region'.
        Has attributes: name, long_name.
    reg_field : xr.DataArray
        2-D field (lat, lon) where each AR6 region is filled with its
        regional mean value and non-land grid cells are NaN.
    """
    if not {"lat", "lon"}.issubset(da.dims):
        raise ValueError("Input DataArray must have 'lat' and 'lon' dimensions.")

    regions = get_ar6_land_regions()

    # mask_3D: dims ('region', 'lat', 'lon'), boolean
    mask_3d = regions.mask_3D(da["lon"], da["lat"])  # type: ignore[call-arg]

    # Broadcast to match da in case da has extra dims (we expect none but stay safe)
    mask_3d = mask_3d.broadcast_like(da)

    # Count valid cells per region
    valid = np.isfinite(da)
    n_valid = (mask_3d & valid).sum(dim=("lat", "lon"))

    # Regional mean (unweighted; grid is ~equal-area at 0.5°)
    reg_mean = (da.where(mask_3d)).mean(dim=("lat", "lon"))

    # Enforce minimum valid count
    reg_mean = reg_mean.where(n_valid >= min_valid)

    reg_mean.name = da.name or "value"
    reg_mean.attrs.setdefault("long_name", f"AR6 land-region mean of {reg_mean.name}")

    # Expand back to a 2-D field: sum over regions of mean * mask
    reg_field = (mask_3d.astype(float) * reg_mean).sum(dim="region")
    reg_field.name = f"{reg_mean.name}_ar6_mean"
    reg_field.attrs.setdefault(
        "long_name", f"{reg_mean.name} aggregated to AR6 regions (region-mean value)"
    )

    return reg_mean, reg_field
# src/sm_attribution/analysis/ar6_regions.py
"""
Utilities for aggregating gridded fields to AR6 reference regions.

Uses `regionmask`'s built-in AR6 land regions and returns both:
- regional means (1-D over region), and
- a 2-D field on the original grid with each region filled by its mean.
"""

from __future__ import annotations

from typing import Tuple

import xarray as xr
import regionmask


def get_ar6_land_regions() -> regionmask.Regions:
    """Return the AR6 land regions definition from regionmask."""
    return regionmask.defined_regions.ar6.land


def ar6_mean_and_field(
    da: xr.DataArray,
    min_valid: int = 10,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Aggregate a 2D (lat, lon) field onto AR6 land regions.

    Parameters
    ----------
    da : xr.DataArray
        Input field with dimensions including ('lat', 'lon').
    min_valid : int
        Minimum number of valid grid cells required for a region
        mean to be kept. Regions with fewer valid cells are set to NaN.

    Returns
    -------
    reg_mean : xr.DataArray (region)
        Mean value in each AR6 land region.
    reg_field : xr.DataArray (lat, lon)
        Map where each land grid cell takes the mean of its AR6 region.
        Ocean / non-region cells are NaN. Greenland and Antarctica are
        force-masked to NaN (for consistency with the ISIMIP landmask).
    """
    if not {"lat", "lon"}.issubset(da.dims):
        raise ValueError("Input DataArray must have 'lat' and 'lon' dimensions.")

    regions = get_ar6_land_regions()

    # mask_3D: dims ('region', 'lat', 'lon'), boolean
    # NOTE: mask_3D already returns True/False. Using .notnull()
    # would turn both True and False into True, so we MUST use it directly.
    mask_3d = regions.mask_3D(da["lon"], da["lat"])  # type: ignore[call-arg]
    mask_bool = mask_3d.astype(bool)

    # Count valid cells per region
    valid = mask_bool & xr.ufuncs.isfinite(da)
    n_valid = valid.sum(dim=("lat", "lon"))

    # Simple equal-area weighting within each region
    weights = mask_bool.astype(float)
    num = (da.where(mask_bool) * weights).sum(dim=("lat", "lon"))
    den = weights.sum(dim=("lat", "lon"))

    reg_mean = num / den
    if min_valid is not None and min_valid > 0:
        reg_mean = reg_mean.where(n_valid >= min_valid)

    # --------------------------------------------------------------
    # Identify AR6 regions corresponding to Greenland / Antarctica
    # --------------------------------------------------------------
    region_names = list(regions.names)
    region_abbrevs = list(getattr(regions, "abbrevs", []))

    to_mask_idx: list[int] = []

    for i, name in enumerate(region_names):
        lname = name.lower()
        if "greenland" in lname or "antarctica" in lname:
            to_mask_idx.append(i)

    # Also catch by AR6 abbreviations if present (e.g. ANT, GIC)
    for i, abbr in enumerate(region_abbrevs):
        if abbr in ("ANT", "GIC"):  # ANT = Antarctica; GIC = Greenland/Iceland
            to_mask_idx.append(i)

    to_mask_idx = sorted(set(to_mask_idx))
    mask_polar = None
    if to_mask_idx:
        # Mask these regions in reg_mean (their regional value becomes NaN)
        reg_mean[to_mask_idx] = float("nan")
        # 2-D mask of all cells belonging to any of these regions
        mask_polar = mask_bool.isel(region=to_mask_idx).any(dim="region")

    # Broadcast regional means back onto the grid
    reg_mean_broadcast = reg_mean.broadcast_like(mask_bool)
    reg_field = reg_mean_broadcast.where(mask_bool)

    # Collapse 'region' dimension: each grid cell belongs to at most one region.
    reg_field = reg_field.max(dim="region")

    # Mask out ocean / cells with no region at all
    mask_any = mask_bool.any(dim="region")
    reg_field = reg_field.where(mask_any)

    # Finally, force Greenland / Antarctica cells to NaN at the grid level
    if mask_polar is not None:
        reg_field = reg_field.where(~mask_polar)

    # Name outputs
    base_name = da.name or "value"
    reg_mean.name = f"{base_name}_ar6_mean"
    reg_field.name = f"{base_name}_ar6_field"

    return reg_mean, reg_field
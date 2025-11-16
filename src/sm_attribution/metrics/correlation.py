# src/sm_attribution/metrics/correlation.py
from __future__ import annotations
from sm_attribution.io.registry import Registry, default_registry

import numpy as np
import xarray as xr
import xskillscore as xs
from typing import Tuple


def _ensure_2d_landmask(land: xr.DataArray) -> xr.DataArray:
    """
    Ensure the land mask is a 2D (lat, lon) boolean array.

    Drops any time dimension if present and coerces to bool.
    """
    m = land

    # If mask has a time dimension, drop it (use first slice)
    if "time" in m.dims:
        m = m.isel(time=0, drop=True)
    if "time" in m.coords:
        m = m.drop_vars("time")

    # Coerce to boolean
    m = m.astype(bool)

    # Keep only lat/lon dimensions
    keep_dims = [d for d in m.dims if d in ("lat", "lon")]
    if set(m.dims) != set(keep_dims):
        m = m.transpose(*keep_dims)

    return m


def _subset_period(
    da: xr.DataArray,
    *,
    period_start: str,
    period_end: str,
    time_name: str = "time",
) -> xr.DataArray:
    """Subset DataArray to the given [period_start, period_end] window."""
    if time_name not in da.dims:
        raise ValueError(f"Expected time dimension '{time_name}' in DataArray.")
    return da.sel({time_name: slice(period_start, period_end)})


def _prepare_pair(
    a: xr.DataArray,
    b: xr.DataArray,
    land: xr.DataArray,
    *,
    period_start: str,
    period_end: str,
    time_name: str = "time",
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Align two fields on a common (time, lat, lon) grid and apply the land mask.

    - Subsets both series to [period_start, period_end].
    - Aligns on the intersection of (time, lat, lon).
    - Broadcasts the 2D land mask to all time steps and masks both series.
    """
    # Subset to correlation period
    a = _subset_period(a, period_start=period_start, period_end=period_end, time_name=time_name)
    b = _subset_period(b, period_start=period_start, period_end=period_end, time_name=time_name)

    # Align on intersection of time/space
    a, b = xr.align(a, b, join="inner")

    # Prepare land mask and reindex to data grid
    land2d = _ensure_2d_landmask(land)
    land2d = land2d.reindex(
        lat=a["lat"],
        lon=a["lon"],
        method=None,
        fill_value=False,
    )

    # Broadcast mask across time (no copy of data themselves)
    land3d = land2d.broadcast_like(a)

    # Mask both fields over land only
    a = a.where(land3d)
    b = b.where(land3d)

    return a, b, land2d


def pearson_map(
    a: xr.DataArray,
    b: xr.DataArray,
    land: xr.DataArray,
    *,
    period_start: str = "2004-01",
    period_end: str = "2019-12",
    n_min: int = 60,
    time_name: str = "time",
) -> xr.Dataset:
    """
    Compute gridpoint Pearson correlation between two time series over land.

    Parameters
    ----------
    a, b : xr.DataArray
        Input series with dimensions including (time, lat, lon). Units can differ
        (e.g. SSI vs standardized anomaly); correlation is dimensionless.
    land : xr.DataArray
        Land mask on the canonical ISIMIP grid. May have extra dimensions (e.g. time)
        which are dropped. Must contain lat/lon.
    period_start, period_end : str
        Correlation period (inclusive), e.g. "2004-01" to "2019-12".
    n_min : int
        Minimum number of valid time samples required to retain a correlation value.
    time_name : str
        Name of the time dimension (default: "time").

    Returns
    -------
    xr.Dataset
        Dataset with variables:
        - r : Pearson correlation coefficient
        - p : p-value of the correlation
        - n : number of valid paired samples
    """
    a, b, land2d = _prepare_pair(
        a,
        b,
        land,
        period_start=period_start,
        period_end=period_end,
        time_name=time_name,
    )

    # Pearson r and p-value over time
    r = xs.pearson_r(a, b, dim=time_name, skipna=True)
    p = xs.pearson_r_p_value(a, b, dim=time_name, skipna=True)

    # Sample size of valid pairs
    valid = xr.ufuncs.logical_and(np.isfinite(a), np.isfinite(b))
    n = valid.sum(time_name)

    # Enforce minimum sample size
    r = r.where(n >= n_min)
    p = p.where(n >= n_min)

    # Cast types
    r = r.astype("float32")
    p = p.astype("float32")
    n = n.astype("int16")

    # Attach variable-level metadata
    r.attrs.update(
        {
            "long_name": "Pearson correlation coefficient (model vs observation)",
            "units": "1",
            "valid_min": -1.0,
            "valid_max": 1.0,
        }
    )
    p.attrs.update(
        {
            "long_name": "Two-sided p-value for Pearson correlation",
            "units": "1",
            "valid_min": 0.0,
            "valid_max": 1.0,
        }
    )
    n.attrs.update(
        {
            "long_name": "Number of valid paired time samples",
            "units": "1",
            "valid_min": 0,
        }
    )

    ds = xr.Dataset({"r": r, "p": p, "n": n})

    ds.attrs.update(
        {
            "metric": "pearson_r",
            "period": f"{period_start} to {period_end}",
            "note": (
                "Computed over land only (ISIMIP land mask), using the intersection of "
                "time/space coordinates. Mask is 2D (lat, lon) broadcast to time. "
                f"Correlations masked where n < {n_min}."
            ),
        }
    )

    return ds
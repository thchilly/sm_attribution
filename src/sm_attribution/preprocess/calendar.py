from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr

def to_proleptic_monthly(ds: xr.Dataset) -> xr.Dataset:
    """
    Normalize monthly time coordinates to proleptic_gregorian month starts.
    - Works for standard/360_day/cftime by mapping each time to (year, month) then to
      pandas.Timestamp(year, month, 1) which is proleptic_gregorian.
    - Does NOT day-weight; it preserves monthly aggregates.
    """
    if "time" not in ds.coords:
        return ds

    # extract year, month robustly
    t = ds["time"].values
    # Support both numpy datetime64 and cftime datetime objects
    if np.issubdtype(getattr(t, "dtype", object), np.datetime64):
        # Vectorized via pandas for speed and robustness
        idx = pd.DatetimeIndex(t)
        years = idx.year
        months = idx.month
    else:
        # cftime or object array: fallback to attribute access
        years = np.array([getattr(tt, "year") for tt in t])
        months = np.array([getattr(tt, "month") for tt in t])

    # normalize to first-of-month (proleptic_gregorian)
    norm = pd.to_datetime(
        [f"{int(y):04d}-{int(m):02d}-01" for y, m in zip(years, months)], utc=False
    )
    ds = ds.copy()
    ds = ds.assign_coords(time=("time", norm))
    # Clean CF calendar attr to avoid xarray encoding conflicts when time is datetime64
    ds.time.attrs.pop("calendar", None)
    ds.time.attrs["standard_name"] = "time"
    return ds
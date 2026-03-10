# src/sm_attribution/io/load_mask.py
from __future__ import annotations

import os

import xarray as xr

from .registry import default_registry


def _expand_paths(tmpl: str, paths: dict) -> str:
    """Expand {paths.*} placeholders in a template string using a paths dict."""
    out = tmpl
    for k, v in paths.items():
        out = out.replace("{paths." + k + "}", v)
    return out


def load_isimip_landmask(key: str = "isimip_water_global") -> xr.DataArray:
    """
    Load an ISIMIP land–sea mask via the registry and return a boolean land mask
    on (lat, lon).
    """
    reg = default_registry()
    tmpl = reg.cfg_dict["ancils"]["landmask"][key]
    path = _expand_paths(tmpl, reg.cfg_dict["paths"])

    if not os.path.exists(path):
        raise FileNotFoundError(f"Land–sea mask not found at: {path}")

    ds = xr.open_dataset(path, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
    var = "mask" if "mask" in ds.data_vars else next(iter(ds.data_vars))
    m = ds[var]

    # Drop any time dimension/coordinate (mask should be purely spatial)
    if "time" in m.dims:
        m = m.isel(time=0, drop=True)
    if "time" in m.coords:
        m = m.drop_vars("time")

    # Normalize fill values to 0, then threshold to boolean land mask
    fv = m.attrs.get("_FillValue", None) or m.attrs.get("missing_value", None)
    if fv is not None:
        m = m.where(m != fv, 0)

    land = (m > 0.5).astype("bool").rename("landmask")
    return land
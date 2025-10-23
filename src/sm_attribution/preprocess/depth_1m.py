from __future__ import annotations
from ..io.settings import get_settings
import xarray as xr

SET = get_settings()
TARGET_DEPTH_M = SET.depth_target_m

def _add_common_attrs(da: xr.DataArray, model: str, scenario: str, note: str | None = None) -> xr.DataArray:
    da = da.copy()
    da.name = "soilmoist_1m"
    da.attrs["units"] = "kg m-2"
    da.attrs["target_depth_m"] = TARGET_DEPTH_M
    da.attrs["model"] = model
    da.attrs["scenario"] = scenario
    if note:
        da.attrs["native_depth_note"] = note
    return da

# ---- v0 recipes mirroring MATLAB layer selections ----

def h08_to_1m(ds: xr.Dataset, scenario: str) -> xr.DataArray:
    # pass-through single-layer total
    da = ds["soilmoist"]
    return _add_common_attrs(da, "h08", scenario, "single-layer total treated as 0–1 m (v0)")

def hydropy_to_1m(ds: xr.Dataset, scenario: str) -> xr.DataArray:
    # pass-through root-zone mass
    da = ds["rootmoist"]
    return _add_common_attrs(da, "hydropy", scenario, "root-zone mass; native depth may differ from 1 m (v0)")

def jules_w2_to_1m(ds: xr.Dataset, scenario: str) -> xr.DataArray:
    # sum layers 1–3 of soilmoist
    sm = ds["soilmoist"]
    # assume dim order (..., depth, time) or (depth, time, lat, lon) is unknown – index by name if present
    if "depth" in sm.dims:
        da = sm.isel(depth=slice(0, 3)).sum("depth", skipna=True)
    else:
        # Some JULES files use 'levsoi' or similar; fall back if present
        levdim = next((d for d in sm.dims if d.lower() in ("levsoi", "layer", "soil_layer")), None)
        if levdim is None:
            raise KeyError("JULES-W2: cannot find depth dimension")
        da = sm.isel({levdim: slice(0, 3)}).sum(levdim, skipna=True)

    # If last month is entirely missing, copy previous month (as in MATLAB)
    if da.isnull().isel(time=-1).all():
        da.loc[dict(time=da.time.isel(time=-1))] = da.isel(time=-2)
    return _add_common_attrs(da, "jules-w2", scenario, "sum of layers 1–3 (v0)")

def miroc_integ_land_to_1m(ds: xr.Dataset, scenario: str) -> xr.DataArray:
    # sum layers 1–3
    sm = ds["soilmoist"]
    depth_dim = next((d for d in sm.dims if d.lower() in ("depth", "layer", "lev", "levsoi")), None)
    if depth_dim is None:
        raise KeyError("MIROC-INTEG-LAND: cannot find depth dimension")
    da = sm.isel({depth_dim: slice(0, 3)}).sum(depth_dim, skipna=True)
    return _add_common_attrs(da, "miroc-integ-land", scenario, "sum of layers 1–3 (v0)")

def watergap22e_to_1m(ds: xr.Dataset, scenario: str) -> xr.DataArray:
    da = ds["soilmoist"]
    return _add_common_attrs(da, "watergap2-2e", scenario, "root-zone-like total treated as 0–1 m (v0)")

def web_dhm_sg_to_1m(ds: xr.Dataset, scenario: str) -> xr.DataArray:
    da = ds["soilmoist"]
    return _add_common_attrs(da, "web-dhm-sg", scenario, "single-layer total treated as 0–1 m (v0)")

# Dispatcher
MODEL_TO_FUNC = {
    "h08": h08_to_1m,
    "hydropy": hydropy_to_1m,
    "jules-w2": jules_w2_to_1m,
    "miroc-integ-land": miroc_integ_land_to_1m,
    "watergap2-2e": watergap22e_to_1m,
    "web-dhm-sg": web_dhm_sg_to_1m,
}
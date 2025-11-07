from __future__ import annotations
from ..io.settings import get_settings
import xarray as xr
import numpy as np
from ..io.registry import Registry  # type: ignore

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


# --- H08 v1: scale using ancillary soil depth map ---
def _h08_depth_map_from_ancil(reg: Registry, like: xr.Dataset) -> xr.DataArray:
    """
    Load H08 soil depth ancillary and return a DataArray D(lat, lon) in meters.
    The ancillary path is taken from data registry under model_ancils -> h08 -> soil_depth.
    We do not perform any regridding; the coordinates must match the model grid exactly.
    """
    ancil_path = reg.get_model_ancil("h08", "soil_depth")
    # Some ancils may have no/odd time definitions; avoid CF decoding pitfalls.
    ds_depth = xr.open_dataset(ancil_path, decode_times=False)

    # Normalize coordinate names to 'lat'/'lon' if needed
    rename = {}
    if "latitude" in ds_depth.coords and "lat" not in ds_depth.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds_depth.coords and "lon" not in ds_depth.coords:
        rename["longitude"] = "lon"
    if rename:
        ds_depth = ds_depth.rename(rename)

    # Heuristic: pick the first 2D variable over (lat, lon)
    cand_vars = [
        v for v in ds_depth.data_vars
        if set(ds_depth[v].dims) == {"lat", "lon"} and ds_depth[v].ndim == 2
    ]
    if not cand_vars:
        # If the variable has a singleton time dimension, drop it.
        cand_vars_time = [
            v for v in ds_depth.data_vars
            if "time" in ds_depth[v].dims and ds_depth[v].sizes.get("time", 1) == 1
            and set([d for d in ds_depth[v].dims if d != "time"]) == {"lat", "lon"}
        ]
        if not cand_vars_time:
            raise KeyError("H08 depth ancillary: could not find a (lat, lon) 2D depth variable.")
        varname = cand_vars_time[0]
        D = ds_depth[varname].isel(time=0, drop=True)
    else:
        varname = cand_vars[0]
        D = ds_depth[varname]

    # Ensure grids match exactly
    if ("lat" in like.coords) and ("lon" in like.coords):
        if not (np.allclose(like["lat"].values, D["lat"].values) and np.allclose(like["lon"].values, D["lon"].values)):
            raise ValueError("H08 soil depth ancillary grid does not match model grid (no regridding performed).")

    # Basic sanity: non-positive depths → treat as NaN to avoid division issues
    D = xr.where(D > 0, D, np.nan).astype("float32")
    D.name = "h08_soil_depth"
    D.attrs.update({
        "units": "m",
        "long_name": "H08 soil depth (ancillary)",
        "source": str(ancil_path),
        "mapping_note": "Applied directly; no regridding; invalid (≤0) masked."
    })
    return D

def h08_to_1m(ds: xr.Dataset, scenario: str, reg: Registry | None = None) -> xr.DataArray:
    # v1: scale using provided soil depth map D (m) with factor f = min(1, D)/D
    if reg is None:
        raise ValueError("Registry is required for H08 v1 homogenization (to get ancillaries).")
    sm = ds["soilmoist"]
    # Build depth map aligned to (lat, lon)
    D = _h08_depth_map_from_ancil(reg, like=sm.to_dataset())
    # Scale factor: no upscaling for shallow columns (D ≤ 1 m), downscale if D > 1 m
    f = xr.where(D > 1.0, 1.0 / D, 1.0)
    f_b = f.broadcast_like(sm)
    da = (sm * f_b).astype(sm.dtype)
    note = "v1: scaled by H08 soil depth ancillary using f=min(1,D)/D; no upscaling for D≤1 m"
    return _add_common_attrs(da, "h08", scenario, note)

def hydropy_to_1m(ds: xr.Dataset, scenario: str, reg: Registry | None = None) -> xr.DataArray:
    # pass-through root-zone mass
    da = ds["rootmoist"]
    return _add_common_attrs(da, "hydropy", scenario, "root-zone mass; native depth may differ from 1 m (v0)")

def jules_w2_to_1m(ds: xr.Dataset, scenario: str, reg: Registry | None = None) -> xr.DataArray:
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

def miroc_integ_land_to_1m(ds: xr.Dataset, scenario: str, reg: Registry | None = None) -> xr.DataArray:
    # sum layers 1–3
    sm = ds["soilmoist"]
    depth_dim = next((d for d in sm.dims if d.lower() in ("depth", "layer", "lev", "levsoi")), None)
    if depth_dim is None:
        raise KeyError("MIROC-INTEG-LAND: cannot find depth dimension")
    da = sm.isel({depth_dim: slice(0, 3)}).sum(depth_dim, skipna=True)
    return _add_common_attrs(da, "miroc-integ-land", scenario, "sum of layers 1–3 (v0)")


def _watergap_depth_map_from_landcover(reg: Registry, like: xr.Dataset) -> xr.DataArray:
    """Build per-pixel rooting depth D (m) for WaterGAP from the ancillary landcover file.
    Returns a DataArray (lat, lon) with depths in meters. Unknown/missing classes → NaN.
    Assumes the ancillary grid matches ISIMIP 0.5° (lat: -89.75..89.75, lon: -179.75..179.75).
    """
    if reg is None:
        raise ValueError("Registry is required to load WaterGAP landcover ancillary.")
    ancil_path = reg.get_model_ancil("watergap2-2e", "landcover")
    # WaterGAP landcover has time units in 'years since ...'; open without CF time decoding (dummy time).
    ds_lc = xr.open_dataset(ancil_path, decode_times=False)
    if "landcover" not in ds_lc:
        raise KeyError("Ancillary file missing 'landcover' variable: " + ancil_path)
    lc = ds_lc["landcover"]
    # Drop the dummy time dim (time=1) if present
    if "time" in lc.dims and lc.sizes.get("time", 1) == 1:
        lc = lc.isel(time=0, drop=True)
    # Ensure coord names match (lat/lon)
    if "latitude" in lc.coords:
        lc = lc.rename({"latitude": "lat"})
    if "longitude" in lc.coords:
        lc = lc.rename({"longitude": "lon"})

    # Align to target grid if necessary (expect same, otherwise will raise on mismatch)
    # We avoid any interpolation; require exact coords equality.
    if ("lat" in like.coords) and ("lon" in like.coords):
        if not (np.allclose(like.lat.values, lc.lat.values) and np.allclose(like.lon.values, lc.lon.values)):
            raise ValueError("WaterGAP landcover ancillary grid does not match model grid (no regridding performed).")
    
    # Class → rooting depth (m) table as provided by WaterGAP team (Appendix C, mail)
    # Missing codes 11 and 13 are set to 1.0 m (neutral) by default.
    # Index 0 is unused.
    depth_lookup = np.full(17, np.nan, dtype=np.float32)
    depth_lookup[1] = 2.0   # Evergreen needleleaf forest
    depth_lookup[2] = 4.0   # Evergreen broadleaf forest
    depth_lookup[3] = 2.0   # Deciduous needleleaf forest
    depth_lookup[4] = 2.0   # Deciduous broadleaf forest
    depth_lookup[5] = 2.0   # Mixed forest
    depth_lookup[6] = 1.0   # Closed shrubland
    depth_lookup[7] = 0.5   # Open shrubland
    depth_lookup[8] = 1.5   # Woody savanna
    depth_lookup[9] = 1.5   # Savanna
    depth_lookup[10] = 1.0  # Grassland
    depth_lookup[11] = 1.0  # (unused in legend) → neutral
    depth_lookup[12] = 1.0  # Cropland
    depth_lookup[13] = 1.0  # (unused in legend) → neutral
    depth_lookup[14] = 1.0  # Cropland/natural mosaic
    depth_lookup[15] = 1.0  # Snow and Ice (permanent)
    depth_lookup[16] = 0.1  # Barren or sparsely vegetated

    # Map classes to depths. Landcover is float; round then cast to int for lookup.
    lc_codes = xr.apply_ufunc(np.rint, lc).astype("int16")
    D = xr.apply_ufunc(np.take, depth_lookup, lc_codes, dask="allowed")
    D = xr.where(np.isfinite(lc), D, np.nan)
    D.name = "watergap_root_depth"
    D.attrs.update({
        "units": "m",
        "long_name": "Effective rooting depth derived from WaterGAP landcover",
        "source": str(ancil_path),
        "mapping_note": "Codes 1..16 mapped per Appendix C; codes 11 and 13 set to 1.0 m (neutral).",
    })
    return D


def watergap22e_to_1m(ds: xr.Dataset, scenario: str, reg: Registry | None = None) -> xr.DataArray:
    """
    v1: Scale WaterGAP total/root-zone soil moisture by per-pixel rooting depth D (m)
    using the WaterGAP landcover ancillary and the scaling factor f = min(1, D) / D.
    - For D > 1 m → f = 1/D (downscale deeper columns to 1 m equivalent)
    - For D ≤ 1 m → f = 1 (no upscaling of shallow columns)
    """
    if reg is None:
        raise ValueError("Registry is required for WaterGAP v1 homogenization (to get ancillaries).")
    sm = ds["soilmoist"]  # (time, depth=1, lat, lon) or (time, lat, lon)
    # Drop singleton depth if present
    if "depth" in sm.dims and sm.sizes.get("depth", 1) == 1:
        sm = sm.isel(depth=0, drop=True)

    # Build depth map aligned to (lat, lon)
    D = _watergap_depth_map_from_landcover(reg, like=sm.to_dataset())

    # Scale factor f = min(1, D) / D → equals 1/D if D>1 else 1
    f = xr.where(D > 1.0, 1.0 / D, 1.0)

    # Broadcast (lat, lon) → (time, lat, lon)
    f_b = f.broadcast_like(sm)
    da = (sm * f_b).astype(sm.dtype)

    note = (
        "v1: scaled by rooting depth from WaterGAP landcover using f=min(1,D)/D; "
        "no upscaling for D≤1 m; codes 11 & 13 treated as 1.0 m"
    )
    return _add_common_attrs(da, "watergap2-2e", scenario, note)


# Ancillary NetCDF for WEB-DHM-SG is created by scripts/make_webdhmsg_landcover_depths.py
# and contains variables 'webdhmsg_total_depth' and 'webdhmsg_landcover' (0.5° grid, lat: -89.75..89.75, lon: -179.75..179.75).

def web_dhm_sg_to_1m(ds: xr.Dataset, scenario: str, reg: Registry | None = None) -> xr.DataArray:
    """
    v1: Scale WEB-DHM-SG total soil moisture to a 0–1 m equivalent using ancillary
    total depth D (m) derived from the provided land-cover map (SiB2 classes).
    Scaling factor: f = min(1, D) / D → equals 1/D if D>1 else 1.
    Requires data_registry.yml entry model_ancils -> web-dhm-sg -> landcover_depths
    pointing to the NetCDF produced by make_webdhmsg_landcover_depths.py
    (variables: 'webdhmsg_total_depth', 'webdhmsg_landcover').
    If needed, generate the ancillary with `scripts/make_webdhmsg_landcover_depths.py --ref-nc <path_to_webdhm_file.nc>` so that its lat/lon centers and orientation match the model grid exactly.
    """
    if reg is None:
        raise ValueError("Registry is required for WEB-DHM-SG v1 homogenization (to get ancillaries).")

    if "soilmoist" not in ds:
        raise KeyError("WEB-DHM-SG: expected variable 'soilmoist' in dataset")
    sm = ds["soilmoist"]  # (time, lat, lon)

    # Load ancillary total depth map (lat, lon)
    ancil_path = reg.get_model_ancil("web-dhm-sg", "landcover_depths")
    ds_anc = xr.open_dataset(ancil_path, decode_times=False)

    # Identify the total-depth variable robustly
    depth_var_candidates = [
        v for v in ds_anc.data_vars
        if v.lower() in ("webdhmsg_total_depth", "total_depth", "depth_total")
    ]
    if depth_var_candidates:
        D = ds_anc[depth_var_candidates[0]]
    else:
        # Fallback: first 2D float var with dims (lat, lon) whose units look like meters
        cand2d = [
            v for v in ds_anc.data_vars
            if ds_anc[v].ndim == 2 and set(ds_anc[v].dims) == {"lat", "lon"}
        ]
        if not cand2d:
            raise KeyError("WEB-DHM-SG ancillary does not contain a (lat, lon) depth variable.")
        D = ds_anc[cand2d[0]]

    # Normalize coordinates to lat/lon if necessary
    ren = {}
    if "latitude" in D.coords and "lat" not in D.coords:
        ren["latitude"] = "lat"
    if "longitude" in D.coords and "lon" not in D.coords:
        ren["longitude"] = "lon"
    if ren:
        D = D.rename(ren)

    # Ensure grids match exactly (no regridding in this routine)
    if ("lat" in sm.coords) and ("lon" in sm.coords):
        if not (np.allclose(sm["lat"].values, D["lat"].values) and np.allclose(sm["lon"].values, D["lon"].values)):
            raise ValueError("WEB-DHM-SG depth ancillary grid does not match model grid (no regridding performed).")

    # Clean and cast depth
    D = xr.where(D > 0, D, np.nan).astype("float32")

    # Scale factor f = min(1, D) / D
    f = xr.where(D > 1.0, 1.0 / D, 1.0)

    # Broadcast and scale
    da = (sm * f.broadcast_like(sm)).astype(sm.dtype)

    note = (
        "v1: scaled by WEB-DHM-SG ancillary total depth using f=min(1,D)/D; "
        "no upscaling for D≤1 m; depth from SiB2 land-cover mapping"
    )
    return _add_common_attrs(da, "web-dhm-sg", scenario, note)


def lpjml5_7_10_fire_to_1m(ds: xr.Dataset, scenario: str, reg: Registry | None = None) -> xr.DataArray:
    """
    Integrate LPJmL5-7-10-fire layered soil moisture to 0–1 m using depth bounds.
    Expects variables:
      - soilmoist(time, depth, lat, lon) [kg m-2 per layer]
      - depth_bnds(depth, bnds) [m], with bnds=(top, bottom)
    """
    sm = ds["soilmoist"]
    if "depth_bnds" not in ds:
        raise KeyError("LPJmL5-7-10-fire: missing 'depth_bnds' for vertical integration")
    bnds = ds["depth_bnds"]  # (depth, bnds)
    top = bnds.isel(bnds=0)   # (depth)
    bot = bnds.isel(bnds=1)   # (depth)

    # Intersection length of each layer with [0, 1.0] meters
    top_clip = top.clip(min=0.0, max=TARGET_DEPTH_M)
    bot_clip = bot.clip(min=0.0, max=TARGET_DEPTH_M)
    overlap = (bot_clip - top_clip).clip(min=0.0)  # (depth)

    # Fraction of each layer included in 0–1 m
    thickness = (bot - top)
    frac = (overlap / thickness).fillna(0.0)

    # Broadcast frac over (time, lat, lon) and sum over depth
    da = (sm * frac).sum("depth", skipna=True)
    note = "depth-weighted integration 0–1 m using depth_bnds (v0 exact)"
    return _add_common_attrs(da, "lpjml5-7-10-fire", scenario, note)


# Dispatcher
MODEL_TO_FUNC = {
    "h08": h08_to_1m,
    "hydropy": hydropy_to_1m,
    "jules-w2": jules_w2_to_1m,
    "miroc-integ-land": miroc_integ_land_to_1m,
    "watergap2-2e": watergap22e_to_1m,
    "web-dhm-sg": web_dhm_sg_to_1m,
    "lpjml5-7-10-fire": lpjml5_7_10_fire_to_1m,
}
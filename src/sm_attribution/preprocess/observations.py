from __future__ import annotations
import glob
import numpy as np
import xarray as xr

from .calendar import to_proleptic_monthly
from ..io.settings import get_settings

SET = get_settings()

# ERA5-Land layer thicknesses (m) for layers 1..3
_ERA5L_THICK_M = np.array([0.07, 0.21, 0.72])  # 0-7cm, 7-28cm, 28-100cm

def _detect_era5l_vars(ds: xr.Dataset) -> list[str]:
    """
    Return the names of ERA5-Land soil water volumetric vars in order [L1, L2, L3].
    Tries swvl1-3 first; falls back to common alternatives.
    """
    candidates = [
        ("swvl1", "swvl2", "swvl3"),
        ("swvl01", "swvl02", "swvl03"),
        ("volsm_l1", "volsm_l2", "volsm_l3"),
    ]
    for trio in candidates:
        if all(v in ds.data_vars for v in trio):
            return list(trio)
    # last resort: scan keys that look like swvl\d
    swv = sorted([v for v in ds.data_vars if v.lower().startswith("swvl")])[:3]
    if len(swv) == 3:
        return swv
    raise KeyError("Could not find ERA5-Land volumetric soil water vars (swvl1..3) in dataset.")

def _to_lon_m180_180(ds: xr.Dataset) -> xr.Dataset:
    """Ensure longitudes in [-180,180) and sorted."""
    if "lon" not in ds.coords and "longitude" in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    if "lat" not in ds.coords and "latitude" in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    lon = ds["lon"]
    if lon.max() > 180.0 + 1e-6:  # likely 0..360
        lon_new = (((lon + 180.0) % 360.0) - 180.0)
        ds = ds.assign_coords(lon=lon_new).sortby("lon")
    return ds

def _maybe_block_coarsen_to_half_degree(da: xr.DataArray) -> xr.DataArray:
    """
    If grid is ~0.1° with constant spacing, block-mean to 0.5° (factor 5).
    Otherwise, try xESMF if available; else raise with instructions.
    """
    if ("lat" not in da.coords) or ("lon" not in da.coords):
        raise ValueError("DataArray must have lat/lon coordinates.")

    def _step(arr):
        vals = np.diff(arr.values)
        return float(np.round(np.median(vals), 6))

    dlat = _step(da["lat"])
    dlon = _step(da["lon"])
    # Accept a narrow tolerance around 0.1°
    if np.isclose(abs(dlat), 0.1, atol=5e-3) and np.isclose(abs(dlon), 0.1, atol=5e-3):
        # Make sizes divisible by 5 by trimming edges if needed
        def _trim_to_multiple(coord, factor=5):
            n = coord.size
            r = n % factor
            if r == 0:
                return slice(None)
            # Trim equally from both ends if possible; else from end.
            trim = r
            return slice(None, n - trim)

        slat = _trim_to_multiple(da["lat"])
        slon = _trim_to_multiple(da["lon"])
        da_t = da.isel(lat=slat, lon=slon)

        # Ensure ascending lat for coarsen with boundary="trim"
        if da_t["lat"][1] < da_t["lat"][0]:
            da_t = da_t.sortby("lat")

        coarsened = da_t.coarsen(lat=5, lon=5, boundary="trim").mean()
        # Rebuild regular 0.5° coords approximately
        new_dlat = abs(float(np.round(5 * dlat, 6)))
        new_dlon = abs(float(np.round(5 * dlon, 6)))
        # Keep existing coords from coarsen (they're midpoints), that’s fine.
        return coarsened
    else:
        # Try xESMF if installed
        try:
            import xesmf as xe  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                f"Grid spacing is not ~0.1°. Install xESMF to regrid to 0.5° (pip install xesmf). "
                f"Detected dlat={dlat}, dlon={dlon}."
            ) from exc

        # Build a 0.5° target grid covering same extent
        lat = da["lat"].values
        lon = da["lon"].values
        lat_min, lat_max = float(lat.min()), float(lat.max())
        lon_min, lon_max = float(lon.min()), float(lon.max())
        lat_out = np.arange(np.ceil(lat_min*2)/2, np.floor(lat_max*2)/2 + 0.5, 0.5)
        lon_out = np.arange(np.ceil(lon_min*2)/2, np.floor(lon_max*2)/2 + 0.5, 0.5)
        tgt = xr.Dataset(coords={"lat": lat_out, "lon": lon_out})

        regridder = xe.Regridder(da.to_dataset(name="var"), tgt, "bilinear", periodic=False, reuse_weights=False)
        out = regridder(da)
        regridder.clean_weight_file()
        return out

def era5land_to_1m_monthly_halfdeg_v1(registry) -> xr.DataArray:
    """
    Reads ERA5-Land monthly files (volumetric soil water) from path in data_registry.yml,
    converts to 0–1 m soil moisture (kg m-2), resampled to 0.5° and proleptic_gregorian monthly calendar.
    """
    import glob
    path_glob = registry.get_obs_raw("era5land")
    files = sorted(glob.glob(path_glob))
    if not files:
        raise FileNotFoundError(f"No ERA5-Land files found at {path_glob}")

    ds = xr.open_mfdataset(files, combine="by_coords", parallel=False)
    ds = to_proleptic_monthly(ds)

    # Detect variables
    for v in ["longitude", "latitude"]:
        if v in ds:
            ds = ds.rename({v: v[:3]})  # lon/lat
    ds = _to_lon_m180_180(ds)

    # Thickness-weighted 0–1 m integration
    thick = np.array([0.07, 0.21, 0.72])  # m
    sm = (ds["swvl1"]*thick[0] + ds["swvl2"]*thick[1] + ds["swvl3"]*thick[2]) * 1000.0
    sm.name = "soilmoist_1m"

    # Regrid from 0.1° to 0.5° via block mean
    sm = _maybe_block_coarsen_to_half_degree(sm)

    sm.attrs.update({
        "units": "kg m-2",
        "long_name": "ERA5-Land 0–1 m soil moisture (converted from swvl1–3 volumetric)",
        "method": "sum(theta_i * thickness_i * 1000) for layers 1–3, block-mean to 0.5°",
        "calendar": "proleptic_gregorian",
        "target_depth_m": 1.0,
        "source_files": path_glob,
    })
    return sm
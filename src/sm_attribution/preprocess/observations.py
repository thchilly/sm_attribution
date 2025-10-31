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
    if float(lon.max().item()) > 180.0 + 1e-6:  # likely 0..360
        lon_new = (((lon + 180.0) % 360.0) - 180.0)
        ds = ds.assign_coords(lon=lon_new).sortby("lon")
    return ds

def _maybe_block_coarsen_to_half_degree(
    da: xr.DataArray,
    *,
    allow_xesmf: bool = False,
    atol_deg: float = 5e-3
) -> xr.DataArray:
    """
    Coarsen a regular lat/lon grid to 0.5° by exact block means when possible.
    - 0.1° → factor 5  (backwards compatible path)
    - 0.25° → factor 2 (GLDAS)
    For any other spacing, raise with a helpful message unless allow_xesmf=True.

    Assumes the variable is *intensive* (kg m-2, m3 m-3), so block *mean* is correct.
    """
    if ("lat" not in da.coords) or ("lon" not in da.coords):
        raise ValueError("DataArray must have 'lat' and 'lon' coordinates.")

    # Ensure ascending latitude for coarsen
    if da["lat"].size > 1 and da["lat"][1] < da["lat"][0]:
        da = da.sortby("lat")

    # Detect nominal spacing
    def _step(coord):
        vals = np.diff(coord.values)
        return float(np.round(np.median(vals), 6))

    dlat = abs(_step(da["lat"]))
    dlon = abs(_step(da["lon"]))

    # Map spacing to coarsen factors
    def _detect_factor(step):
        if np.isclose(step, 0.1, atol=atol_deg):   # ERA5/GLEAM legacy
            return 5
        if np.isclose(step, 0.25, atol=atol_deg):  # GLDAS
            return 2
        return None

    f_lat = _detect_factor(dlat)
    f_lon = _detect_factor(dlon)

    if f_lat is not None and f_lon is not None:
        # Trim edges so dimension is divisible by factor (same behavior as before)
        def _trim_slice(n, f):
            r = n % f
            return slice(None) if r == 0 else slice(None, n - r)

        da_t = da.isel(lat=_trim_slice(da.sizes["lat"], f_lat),
                       lon=_trim_slice(da.sizes["lon"], f_lon))

        # Coarsen by exact block means (no interpolation)
        out = da_t.coarsen(lat=f_lat, lon=f_lon, boundary="trim").mean()
        out.attrs.update(da.attrs)
        out.attrs["regrid_note"] = f"block-mean from ~{dlat:.3f}°×{dlon:.3f}° using factors ({f_lat},{f_lon}) to 0.5°"
        return out

    # Fallback: only if explicitly allowed
    if not allow_xesmf:
        raise RuntimeError(
            "Grid spacing is not ~0.1° or ~0.25°. "
            f"Detected dlat={dlat}, dlon={dlon}. "
            "To regrid, call with allow_xesmf=True (requires xESMF), "
            "or add an explicit block factor for this spacing."
        )

    # Optional xESMF path (off by default)
    import xesmf as xe  # type: ignore
    lat = da["lat"].values
    lon = da["lon"].values
    lat_out = np.arange(-89.75, 90.0, 0.5)
    lon_out = np.arange(-179.75, 180.0, 0.5)
    tgt = xr.Dataset(coords={"lat": lat_out, "lon": lon_out})
    regridder = xe.Regridder(da.to_dataset(name="var"), tgt, "bilinear", periodic=False, reuse_weights=False)
    out = regridder(da)
    regridder.clean_weight_file()
    out.attrs.update(da.attrs)
    out.attrs["regrid_note"] = "xesmf bilinear to 0.5° (allow_xesmf=True)"
    return out


def era5land_to_1m_monthly_halfdeg_v1(registry) -> xr.DataArray:
    """
    Reads ERA5-Land monthly files (volumetric soil water) from path in data_registry.yml,
    converts to 0–1 m soil moisture (kg m-2), resampled to 0.5° and proleptic_gregorian monthly calendar.
    """
    path_glob = registry.get_obs_raw("era5land")
    files = sorted(glob.glob(path_glob))
    if not files:
        raise FileNotFoundError(f"No ERA5-Land files found at {path_glob}")

    ds = xr.open_mfdataset(files, combine="by_coords", parallel=False)
    ds = to_proleptic_monthly(ds)

    # Normalize coords and detect layer variables
    if "longitude" in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    if "latitude" in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    ds = _to_lon_m180_180(ds)
    v1, v2, v3 = _detect_era5l_vars(ds)

    # Thickness-weighted 0–1 m integration
    thick = np.array([0.07, 0.21, 0.72])  # m
    sm = (ds[v1]*thick[0] + ds[v2]*thick[1] + ds[v3]*thick[2]) * 1000.0
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

# -----------------------------------------------------------------------------
# GLEAM v4.2 (SMrz) — v0 processing (keep volumetric root-zone moisture)
# Outputs:
#  - v4.2a 1980–2020
#  - v4.2a 2003–2020 (aligned with v4.2b period)
#  - v4.2b 2003–2020
# -----------------------------------------------------------------------------

def _to_month_start(ds: xr.Dataset) -> xr.Dataset:
    """Ensure monthly timestamps are at month start (00:00 on first day)."""
    if "time" not in ds.coords:
        return ds
    t = xr.cftime_range(start=str(ds.time.dt.year.min().item()) + "-01-01",
                        end=str(ds.time.dt.year.max().item()) + "-12-01",
                        freq="MS",
                        calendar="proleptic_gregorian")
    # Only re-stamp if monthly and lengths match
    if ds.sizes.get("time", -1) == t.size:
        ds = ds.assign_coords(time=t)
    return ds

def _open_gleam_stack(path_glob: str) -> xr.Dataset:
    files = sorted(glob.glob(path_glob))
    if not files:
        raise FileNotFoundError(f"No GLEAM files matched: {path_glob}")
    ds = xr.open_mfdataset(files, combine="by_coords", parallel=False)
    # Normalize coords
    if "longitude" in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    if "latitude" in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    ds = _to_lon_m180_180(ds)
    # Time handling: already proleptic_gregorian monthly with end-of-month stamps → shift to month-start
    ds = _to_month_start(ds)
    return ds

def _gleam_smrz_to_halfdeg_v0(ds: xr.Dataset) -> xr.DataArray:
    """Block-average SMrz from 0.1° to 0.5°. Keep volumetric units (m3 m-3)."""
    if "SMrz" not in ds.data_vars:
        raise KeyError("Expected variable 'SMrz' in GLEAM dataset.")
    da = ds["SMrz"]
    da = _maybe_block_coarsen_to_half_degree(da)
    da.name = "soilmoist_1m"
    da.attrs.update({
        "units": "m3 m-3",
        "long_name": "GLEAM root-zone soil moisture (v0, volumetric)",
        "method": "block-mean 0.1°→0.5°; timestamps set to month-start",
        "note": "v0 retains root-zone volumetric moisture without depth remapping to 0–1 m.",
        "calendar": "proleptic_gregorian",
    })
    return da

def gleam42a_1980_2020_v0(registry) -> xr.DataArray:
    """GLEAM v4.2a SMrz, 1980–2020, monthly 0.5°, volumetric (v0)."""
    path_glob = registry.get_obs_raw("gleam42a")
    ds = _open_gleam_stack(path_glob)
    da = _gleam_smrz_to_halfdeg_v0(ds.sel(time=slice("1980-01-01", "2020-12-01")))
    return da

def gleam42a_2003_2020_v0(registry) -> xr.DataArray:
    """GLEAM v4.2a SMrz subset 2003–2020, to compare with v4.2b."""
    path_glob = registry.get_obs_raw("gleam42a")
    ds = _open_gleam_stack(path_glob)
    da = _gleam_smrz_to_halfdeg_v0(ds.sel(time=slice("2003-01-01", "2020-12-01")))
    return da

def gleam42b_2003_2020_v0(registry) -> xr.DataArray:
    """GLEAM v4.2b SMrz, 2003–2020, monthly 0.5°, volumetric (v0)."""
    path_glob = registry.get_obs_raw("gleam42b")
    ds = _open_gleam_stack(path_glob)
    da = _gleam_smrz_to_halfdeg_v0(ds.sel(time=slice("2003-01-01", "2020-12-01")))
    return da

# Backward-compatible aliases (old naming)
gleam42a_full_v0 = gleam42a_1980_2020_v0
gleam42b_full_v0 = gleam42b_2003_2020_v0


def _drop_vars_if_present(ds: xr.Dataset, names: list[str]) -> xr.Dataset:
    present = [n for n in names if n in ds.variables]
    return ds.drop_vars(present) if present else ds

def _open_gldas_stack(path_glob: str) -> xr.Dataset:
    files = sorted(glob.glob(path_glob))
    if not files:
        raise FileNotFoundError(f"No GLDAS files matched: {path_glob}")
    ds = xr.open_mfdataset(files, combine="by_coords", parallel=False)
    # Normalize coords
    if "longitude" in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    if "latitude" in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    ds = _to_lon_m180_180(ds)
    # Normalize monthly calendar to proleptic_gregorian
    ds = to_proleptic_monthly(ds)
    # Drop time bounds to avoid xarray encoding clashes
    ds = _drop_vars_if_present(ds, ["time_bnds"])
    return ds

def _gldas_0_1m_sum(ds: xr.Dataset) -> xr.DataArray:
    needed = ["SoilMoi0_10cm_inst", "SoilMoi10_40cm_inst", "SoilMoi40_100cm_inst"]
    missing = [v for v in needed if v not in ds.data_vars]
    if missing:
        raise KeyError(f"GLDAS: missing required soil moisture vars: {missing}")
    da = ds[needed[0]] + ds[needed[1]] + ds[needed[2]]
    da.name = "soilmoist_1m"
    return da

def gldas_v20_to_1m_monthly_halfdeg_v0(registry) -> xr.DataArray:
    """GLDAS v2.0 (0.25° monthly): sum 0–100 cm soil moisture layers, block-mean to 0.5°. Units: kg m-2.""" 
    path_glob = registry.get_obs_raw("gldas_v20")
    ds = _open_gldas_stack(path_glob)
    da = _gldas_0_1m_sum(ds)
    da = _maybe_block_coarsen_to_half_degree(da)
    da.attrs.update({
        "units": "kg m-2",
        "long_name": "GLDAS NOAH v2.0 soil moisture 0–1 m (sum of 0–10,10–40,40–100 cm)",
        "method": "sum layer masses; block-mean 0.25°→0.5°; monthly proleptic_gregorian",
        "calendar": "proleptic_gregorian",
        "source": "GLDAS2.0 NOAH monthly",
        "target_depth_m": 1.0,
        "note": "v0 uses provided layer-integrated masses; no ancillary remapping.",
        "source_files": path_glob,
    })
    return da

def gldas_v21_to_1m_monthly_halfdeg_v0(registry) -> xr.DataArray:
    """GLDAS v2.1 (0.25° monthly): sum 0–100 cm soil moisture layers, block-mean to 0.5°. Units: kg m-2.""" 
    path_glob = registry.get_obs_raw("gldas_v21")
    ds = _open_gldas_stack(path_glob)
    da = _gldas_0_1m_sum(ds)
    da = _maybe_block_coarsen_to_half_degree(da)
    da.attrs.update({
        "units": "kg m-2",
        "long_name": "GLDAS NOAH v2.1 soil moisture 0–1 m (sum of 0–10,10–40,40–100 cm)",
        "method": "sum layer masses; block-mean 0.25°→0.5°; monthly proleptic_gregorian",
        "calendar": "proleptic_gregorian",
        "source": "GLDAS2.1 NOAH monthly",
        "target_depth_m": 1.0,
        "note": "v0 uses provided layer-integrated masses; no ancillary remapping.",
        "source_files": path_glob,
    })
    return da
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

# Replace fill values with NaN; otherwise keep only finite values.
def _nan_fill(da: xr.DataArray) -> xr.DataArray:
    """Replace encoded/attr fill values with NaN; otherwise keep only finite values."""
    fill = da.encoding.get("_FillValue", None)
    if fill is None:
        fill = da.attrs.get("_FillValue", None)
    if fill is None:
        fill = da.attrs.get("missing_value", None)
    if fill is not None:
        return da.where(da != np.float32(fill))
    return da.where(np.isfinite(da))

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
# GRACE-DA-DM (Global V3.0) — root-zone percentile → monthly 0.5° (v0)
# -----------------------------------------------------------------------------

def _open_gracedadm_weekly_stack(registry) -> xr.Dataset:
    """Open all GRACE-DA-DM weekly files and normalize coordinates.
    Expects a glob from registry key 'gracedadm_weekly'.
    """
    path_glob = registry.get_obs_raw("gracedadm_weekly")
    files = sorted(glob.glob(path_glob))
    if not files:
        raise FileNotFoundError(f"No GRACE-DA-DM files matched: {path_glob}")
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=False,
        decode_times=True,
        engine="netcdf4",
        mask_and_scale=True,
    )
    # Normalize coord names and longitudes
    if "longitude" in ds.coords and "lon" not in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    if "latitude" in ds.coords and "lat" not in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    ds = _to_lon_m180_180(ds)

    # Ensure lat ascending
    if "lat" in ds.coords and (ds.lat.size > 1) and (ds.lat[1] < ds.lat[0]):
        ds = ds.sortby("lat")

    return ds


def gracedadm_rootzone_to_monthly_halfdeg_v0(registry) -> xr.DataArray:
    """
    Build GRACE-DA-DM root-zone soil moisture percentile product as monthly means
    on a 0.5° grid (v0). We simply average percentiles in time (weekly→monthly)
    and in space (0.25°→0.5°) via block means.

    Output variable is a percentile (0–100), dimensioned (time, lat, lon).
    Period clipped to 2003-02 through 2020-12 to match the study window.
    """
    ds = _open_gracedadm_weekly_stack(registry)

    if "rtzsm_inst" not in ds.data_vars:
        # Some distributions might use alternative short names; be strict for now
        raise KeyError("Expected variable 'rtzsm_inst' (root zone soil moisture percentile) in GRACE-DA-DM files.")

    da = ds["rtzsm_inst"].astype("float32")

    # Replace fill values (-999) with NaN if present
    fill = da.encoding.get("_FillValue", None)
    if fill is None:
        fill = da.attrs.get("_FillValue", None)
    if fill is None:
        fill = da.attrs.get("missing_value", None)
    if fill is not None:
        da = da.where(da != np.float32(fill))
    else:
        da = da.where(np.isfinite(da))

    # Clip the time span explicitly
    da = da.sel(time=slice("2003-02-01", "2020-12-31"))

    # Weekly → monthly mean, month-start timestamps
    da_m = da.resample(time="MS").mean("time", skipna=True)

    # 0.25° → 0.5° block mean (intensive variable: percentile)
    da_05 = _maybe_block_coarsen_to_half_degree(da_m)

    # Name & attributes
    da_05.name = "rootzone_percentile"
    da_05.attrs.update({
        "units": "%",
        "long_name": "GRACE-DA-DM root-zone soil moisture percentile (monthly mean)",
        "standard_name": "root_zone_soil_moisture_percentile",
        "method": "weekly percentiles averaged to monthly means; 0.25°→0.5° block mean",
        "note": "Percentiles (0–100) are averaged in time/space as a pragmatic summary statistic.",
        "calendar": "proleptic_gregorian",
        "source": "GRACEDADM_CLSM025GL_7D v3.0",
    })

    return da_05

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

def _open_somoml_layer(registry, key: str, var_name: str) -> xr.DataArray:
    """Open all yearly files for one SoMo.ml layer and return a daily DataArray."""
    pat = registry.get_obs_raw(key)  # glob pattern like .../SoMo.ml_v1_layer1/*.nc
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(f"SoMo.ml: no files found for pattern: {pat}")
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=False,
        decode_times=True,
        engine="netcdf4",
    )
    # Some files have the variable named exactly as layer1/layer2/layer3
    if var_name not in ds:
        # try to find the only data var if necessary
        data_vars = [v for v in ds.data_vars]
        if len(data_vars) == 1:
            ds = ds.rename({data_vars[0]: var_name})
        else:
            raise KeyError(f"Expected variable {var_name} not found in SoMo.ml files.")
    da = ds[var_name]
    # Ensure lat ascending & lon in [-180,180) if needed
    if "lat" in da.coords and (da.lat[0] > da.lat[-1]):
        da = da.sortby("lat")
    if "lon" in da.coords and (da.lon.max() > 180.0):
        # Convert [0,360) to [-180,180)
        da = da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon")
    return da

def _somoml_daily_to_monthly(da: xr.DataArray) -> xr.DataArray:
    """Daily to monthly mean, month-start timestamps (MS)."""
    # Using resample to handle leap years, etc.
    return da.resample(time="MS").mean("time", skipna=True)

def _somoml_depth_weighted_volumetric(l1: xr.DataArray, l2: xr.DataArray, l3: xr.DataArray) -> xr.DataArray:
    """
    Depth-weighted volumetric average (m3/m3) across 0–50 cm:
      layer1: 0–10 cm (0.1 m)
      layer2: 10–30 cm (0.2 m)
      layer3: 30–50 cm (0.2 m)
    """
    w1, w2, w3 = 0.1, 0.2, 0.2  # meters
    depth = w1 + w2 + w3        # 0.5 m
    return (w1*l1 + w2*l2 + w3*l3) / depth

def _somoml_volumetric_to_mass(theta_0p5m: xr.DataArray) -> xr.DataArray:
    """Convert volumetric (m3/m3) to water mass per area (kg/m2) over 0.5 m."""
    rho_w = 1000.0  # kg/m3
    depth_m = 0.5   # meters
    return theta_0p5m * rho_w * depth_m


def somoml_to_0p5m_monthly_halfdeg_v0(registry) -> xr.Dataset:
    """
    Build SoMo.ml 0–50 cm product as monthly means on 0.5° grid. Output Dataset contains a standardized variable 'soilmoist_1m' (depth documented as 0–0.5 m).
    """
    # 1) Load daily for each layer
    l1 = _open_somoml_layer(registry, "somo_ml_layer1", "layer1")
    l2 = _open_somoml_layer(registry, "somo_ml_layer2", "layer2")
    l3 = _open_somoml_layer(registry, "somo_ml_layer3", "layer3")

    # 2) Daily → monthly means
    l1m = _somoml_daily_to_monthly(l1)
    l2m = _somoml_daily_to_monthly(l2)
    l3m = _somoml_daily_to_monthly(l3)

    # 3) Align months
    l1m, l2m = xr.align(l1m, l2m, join="inner")
    l1m, l3m = xr.align(l1m, l3m, join="inner")

    # 4) Depth-weighted volumetric average over 0–50 cm
    theta_0p5m = _somoml_depth_weighted_volumetric(l1m, l2m, l3m)

    # 5) Convert to kg m-2 (water column over 0.5 m)
    sm_kgm2 = _somoml_volumetric_to_mass(theta_0p5m)

    # 6) Regrid to 0.5° by block-mean if needed
    sm_kgm2_05 = _maybe_block_coarsen_to_half_degree(sm_kgm2)

    # 7) Pack to Dataset with standardized variable name for compatibility
    da = sm_kgm2_05.astype("float32").drop_vars([v for v in sm_kgm2_05.coords if v not in ("time","lat","lon")], errors="ignore")
    da = da.where(np.isfinite(da), other=np.float32(-9999.0))
    da.name = "soilmoist_1m"  # keep the same name used elsewhere in your pipeline

    da.attrs.update({
        "standard_name": "soil_moisture_content",
        "long_name": "SoMo.ml depth-integrated soil moisture (0–0.5 m) converted to water mass",
        "units": "kg m-2",
        "source_product": "SoMo.ml v1 (layer1 0–10 cm, layer2 10–30 cm, layer3 30–50 cm)",
        "native_units": "m3 m-3 (volumetric)",
        "conversion": "theta * 1000 kg/m3 * 0.5 m",
        "depth_m": 0.5,
        "processing_version": "v0",
        "regridding": "0.25°→0.5° by 2x2 block mean",
    })

    ds_out = da.to_dataset()

    # Coordinates tidy-up
    for c in ("lat", "lon"):
        if c in ds_out.coords:
            ds_out[c].attrs.pop("_FillValue", None)

    # Global attrs
    ds_out.attrs.update({
        "title": "SoMo.ml 0–50 cm soil moisture (monthly, 0.5°), converted to kg m-2",
        "comment": "Depth is 0–0.5 m; variable kept as 'soilmoist_1m' for pipeline compatibility.",
        "institution": "Your lab / project",
        "history": "SoMo.ml layer1/2/3 daily → monthly mean; depth-weighted; mass conversion; block coarsen to 0.5°.",
    })

    return ds_out

def _interp_lon_to_half_degree(da: xr.DataArray) -> xr.DataArray:
    """
    For inputs on 0.625° lon spacing (e.g., MERRA-2), remap longitude to 0.5° by 1D linear interpolation.
    Leaves latitude unchanged (already 0.5° in MERRA-2).
    """
    if "lon" not in da.coords:
        raise ValueError("DataArray must have 'lon' coordinate.")
    # Ensure lon in [-180,180)
    da = _to_lon_m180_180(da.to_dataset(name="var"))["var"]
    lon_out = np.arange(-179.75, 180.0, 0.5)  # 0.5° centers
    # Interpolate only along lon
    out = da.interp(lon=lon_out, method="linear")
    out.attrs.update(da.attrs)
    out.attrs["regrid_note"] = "1D linear lon interpolation 0.625°→0.5° for MERRA-2"
    return out
def merra2_land_to_1m_monthly_halfdeg_v1(registry) -> xr.DataArray:
    """
    Build MERRA-2 LAND 0–1 m soil moisture (kg m-2) using a depth-weighted blend:
      - SFMC (surface layer volumetric, 0–0.05 m) with weight 0.05
      - RZMC (root-zone volumetric, 0.05–1.0 m) with weight 0.95
    We treat SFMC and RZMC as volumetric water content (m3 m-3) over their respective layers,
    compute a depth-weighted average theta over 0–1 m, convert to kg m-2, regrid to 0.5° monthly
    on a proleptic_gregorian calendar.

    Notes:
    - Native grid: 0.5° (lat) × 0.625° (lon).
    - Time is monthly means with non-standard units; converted via to_proleptic_monthly.
    """
    path_glob = registry.get_obs_raw("merra2_land")
    files = sorted(glob.glob(path_glob))
    if not files:
        raise FileNotFoundError(f"No MERRA2-LAND files found at {path_glob}")

    # Open and normalize coordinates
    ds = xr.open_mfdataset(files, combine="by_coords", parallel=False)
    ds = to_proleptic_monthly(ds)

    if "longitude" in ds.coords and "lon" not in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    if "latitude" in ds.coords and "lat" not in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    ds = _to_lon_m180_180(ds)

    # Required variables
    for v in ("SFMC", "RZMC"):
        if v not in ds.data_vars:
            raise KeyError(f"MERRA2-LAND: required variable '{v}' not found.")

    sfmc = ds["SFMC"].astype("float32")
    rzmc = ds["RZMC"].astype("float32")
    sfmc = _nan_fill(sfmc)
    rzmc = _nan_fill(rzmc)

    # Volumetric blend: 5% surface (0–5 cm) + 95% root-zone proxy
    theta = 0.05 * sfmc + 0.95 * rzmc  # m3/m3

    # Convert volumetric to kg m-2 over 1 m
    rho_w = 1000.0  # kg/m3
    depth_m = 1.0
    sm = theta * rho_w * depth_m
    sm.name = "soilmoist_1m"
    sm.attrs.update({
        "units": "kg m-2",
        "long_name": "MERRA-2 0–1 m soil moisture (5% SFMC + 95% RZMC → mass)",
        "method": "theta = 0.05*SFMC + 0.95*RZMC; soilmoist = theta*1000*1.0",
        "calendar": "proleptic_gregorian",
        "target_depth_m": 1.0,
        "source_files": path_glob,
        "note": "Depth weights follow project decision; constants not read from ancillary files.",
    })

    # Regrid lon: 0.625° → 0.5° without xESMF (linear lon interp)
    sm_05 = _interp_lon_to_half_degree(sm)
    sm_05 = sm_05.astype("float32")

    # Ensure ascending lat for consistency
    if sm_05["lat"].size > 1 and sm_05["lat"][1] < sm_05["lat"][0]:
        sm_05 = sm_05.sortby("lat")


    # Attributes
    sm_05.attrs.update({
        "units": "kg m-2",
        "long_name": "MERRA-2 0–1 m soil moisture (SFMC 0–0.05 m and RZMC 0.05–1.0 m depth-weighted)",
        "standard_name": "soil_moisture_content",
        "method": "theta = (0.05*SFMC + 0.95*RZMC)/1.0; soilmoist = theta * 1000 kg/m3 * 1.0 m; linear interpolation to 0.5°",
        "calendar": "proleptic_gregorian",
        "target_depth_m": 1.0,
        "source_files": path_glob,
        "native_grid": "0.5° lat × 0.625° lon",
        "weights_note": "dzsf=0.05 m and dzrz=0.95 m as global constants.",
    })
    return sm_05

# -----------------------------------------------------------------------------
# GDO-ENSMIA (Ensemble Soil Moisture Anomaly, JRC) — v0
# -----------------------------------------------------------------------------

def _open_gdo_ensmia_stack(registry) -> xr.Dataset:
    """Open all GDO-ENSMIA anomaly files and normalize coordinates."""
    path_glob = registry.get_obs_raw("gdo_ensmia")
    files = sorted(glob.glob(path_glob))
    if not files:
        raise FileNotFoundError(f"No GDO-ENSMIA files matched: {path_glob}")
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=False,
        decode_times=True,
        engine="netcdf4",
        mask_and_scale=True,
    )

    # Normalize coordinate names
    if "longitude" in ds.coords and "lon" not in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    if "latitude" in ds.coords and "lat" not in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    ds = _to_lon_m180_180(ds)

    # Ensure ascending latitude
    if "lat" in ds.coords and (ds.lat.size > 1) and (ds.lat[1] < ds.lat[0]):
        ds = ds.sortby("lat")

    return ds


def gdo_ensmia_to_monthly_halfdeg_v0(registry) -> xr.DataArray:
    """
    Build GDO Ensemble Soil Moisture Index product (v0):
      - Select 3rd dekad (last timestep) per month → monthly
      - Coarsen spatially from 0.1° → 0.5° via block mean
      - Output standardized anomaly (-3 to 3, dimensionless)
    """
    ds = _open_gdo_ensmia_stack(registry)

    if "smant" not in ds.data_vars:
        raise KeyError("Expected variable 'smant' in GDO-ENSMIA files.")

    da = ds["smant"].astype("float32")
    da = _nan_fill(da)

    # Keep only last timestep (3rd dekad) per month using resample bins
    # Resample to month-start bins and select the final element from each bin.
    # This avoids reliance on unsupported virtual coords like time.to_period('M').
    da_monthly = da.resample(time="MS").map(lambda x: x.isel(time=-1))

    # Coarsen spatially 0.1° → 0.5° (mean of standardized index)
    da_05 = _maybe_block_coarsen_to_half_degree(da_monthly)

    da_05.name = "soilmoist_anom_std"
    da_05.attrs.update({
        "units": "dimensionless",
        "long_name": "GDO Ensemble Soil Moisture Anomaly (standardized index)",
        "method": "selected 3rd dekad per month; block-mean 0.1°→0.5°",
        "calendar": "proleptic_gregorian",
        "note": "Index represents standardized anomalies (-3 to +3) derived from LISFLOOD, MODIS LST, and ESA CCI.",
        "source": "Global Drought Observatory, JRC (2001–2020)",
    })

    return da_05
from __future__ import annotations

import glob
import os

import numpy as np
import xarray as xr

from .calendar import to_proleptic_monthly
from ..io.settings import get_settings
from sm_attribution.io.load_mask import load_isimip_landmask

SET = get_settings()


# -----------------------------------------------------------------------------
# Canonical grid utilities (ISIMIP land mask)
# -----------------------------------------------------------------------------

def get_canonical_grid(mask_key: str = "isimip_no_ant_nogreenland"):
    """
    Return the canonical ISIMIP land-mask grid as (lat, lon) 1D numpy arrays.
    """
    land = load_isimip_landmask(mask_key)  # (lat, lon)
    return land["lat"].values, land["lon"].values


# -----------------------------------------------------------------------------
# Generic helpers (coords, fill values, coarsening)
# -----------------------------------------------------------------------------

# ERA5-Land layer thicknesses (m) for layers 1..3: 0–7 cm, 7–28 cm, 28–100 cm
_ERA5L_THICK_M = np.array([0.07, 0.21, 0.72])


def _detect_era5l_vars(ds: xr.Dataset) -> list[str]:
    """
    Return ERA5-Land volumetric soil water variable names in order [L1, L2, L3].

    Tries common naming patterns first (swvl1–3, swvl01–03, volsm_l1–3),
    then falls back to scanning for keys starting with "swvl".
    """
    candidates = [
        ("swvl1", "swvl2", "swvl3"),
        ("swvl01", "swvl02", "swvl03"),
        ("volsm_l1", "volsm_l2", "volsm_l3"),
    ]
    for trio in candidates:
        if all(v in ds.data_vars for v in trio):
            return list(trio)

    # Fallback: scan keys that look like swvl*
    swv = sorted([v for v in ds.data_vars if v.lower().startswith("swvl")])[:3]
    if len(swv) == 3:
        return swv

    raise KeyError(
        "Could not find ERA5-Land volumetric soil water vars (swvl1..3) in dataset."
    )


def _to_lon_m180_180(ds: xr.Dataset) -> xr.Dataset:
    """
    Normalize longitude coordinate to [-180, 180) and ensure sorting by lon.
    """
    if "lon" not in ds.coords and "longitude" in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    if "lat" not in ds.coords and "latitude" in ds.coords:
        ds = ds.rename({"latitude": "lat"})

    lon = ds["lon"]
    if float(lon.max().item()) > 180.0 + 1e-6:  # likely in 0..360
        lon_new = (((lon + 180.0) % 360.0) - 180.0)
        ds = ds.assign_coords(lon=lon_new).sortby("lon")
    return ds


def _nan_fill(da: xr.DataArray) -> xr.DataArray:
    """
    Replace encoded / attribute fill values with NaN; otherwise keep only finite
    values.
    """
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
    atol_deg: float = 5e-3,
) -> xr.DataArray:
    """
    Coarsen a regular lat/lon grid to 0.5° by exact block means when possible.

    Supported native spacings:
      - 0.10° → factor 5  (e.g. ERA5, GLEAM)
      - 0.25° → factor 2  (e.g. GLDAS)
      - 0.05° → factor 10 (some GDO tiles)
      - ~0.0167° (1 arcmin) → factor 30 (GDO SMIA)

    If spacing is unsupported and allow_xesmf=False, a RuntimeError is raised.
    When allow_xesmf=True, fall back to bilinear regridding with xESMF.

    Assumes the variable is intensive (e.g. kg m-2, m3 m-3), so block means are
    appropriate.
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
        if np.isclose(step, 0.1, atol=atol_deg):
            return 5
        if np.isclose(step, 0.25, atol=atol_deg):
            return 2
        if np.isclose(step, 0.05, atol=atol_deg):
            return 10
        if np.isclose(step, 1.0 / 60.0, atol=atol_deg):
            return 30
        return None

    f_lat = _detect_factor(dlat)
    f_lon = _detect_factor(dlon)

    if f_lat is not None and f_lon is not None:
        # Trim edges so dimensions are divisible by the coarsen factor
        def _trim_slice(n, f):
            r = n % f
            return slice(None) if r == 0 else slice(None, n - r)

        da_t = da.isel(
            lat=_trim_slice(da.sizes["lat"], f_lat),
            lon=_trim_slice(da.sizes["lon"], f_lon),
        )

        # Coarsen by exact block means (no interpolation)
        out = da_t.coarsen(lat=f_lat, lon=f_lon, boundary="trim").mean()
        out.attrs.update(da.attrs)
        out.attrs["regrid_note"] = (
            f"block-mean from ~{dlat:.3f}°×{dlon:.3f}° "
            f"using factors ({f_lat},{f_lon}) to 0.5°"
        )

        # Snap to canonical ISIMIP grid if compatible
        out = _snap_to_canonical(out)
        return out

    # Fallback: xESMF path (opt-in)
    if not allow_xesmf:
        raise RuntimeError(
            "Grid spacing is not supported for block coarsening. "
            f"Detected dlat={dlat}, dlon={dlon}. "
            "To regrid, call with allow_xesmf=True (requires xESMF), "
            "or add an explicit block factor for this spacing."
        )

    import xesmf as xe  # type: ignore

    lat = da["lat"].values
    lon = da["lon"].values
    lat_out, lon_out = get_canonical_grid()
    tgt = xr.Dataset(coords={"lat": lat_out, "lon": lon_out})
    regridder = xe.Regridder(
        da.to_dataset(name="var"),
        tgt,
        "bilinear",
        periodic=False,
        reuse_weights=False,
    )
    out = regridder(da)
    regridder.clean_weight_file()
    out.attrs.update(da.attrs)
    out.attrs["regrid_note"] = "xesmf bilinear to 0.5° (allow_xesmf=True)"
    return out


def _snap_to_canonical(da: xr.DataArray) -> xr.DataArray:
    """
    Force (lat, lon) to match the canonical ISIMIP mask grid exactly
    (same length, orientation, and bitwise-identical coordinates),
    whenever the input grid is clearly the same 0.5° global grid up
    to small shifts / reversals.

    If the shape or spacing look incompatible, the input is returned unchanged.
    """
    if ("lat" not in da.coords) or ("lon" not in da.coords):
        return da

    lat_can, lon_can = get_canonical_grid()

    # Require same shape
    if da.sizes.get("lat", 0) != lat_can.size or da.sizes.get("lon", 0) != lon_can.size:
        return da

    lat = da["lat"].values
    lon = da["lon"].values

    if lat.size < 2 or lon.size < 2:
        return da

    def _step(arr: np.ndarray) -> float:
        return float(np.abs(np.median(np.diff(arr))))

    def _compatible_axis(axis: np.ndarray, canon: np.ndarray) -> bool:
        """Check ~0.5° spacing and endpoints close to canonical (allow small shift)."""
        step = _step(axis)
        step_can = _step(canon)
        if not np.isclose(step, step_can, atol=1e-3):
            return False

        # compare axis endpoints to canonical endpoints (allow up to quarter-grid shift)
        tol = 0.25  # degrees
        diffs = [
            np.abs(axis[0] - canon[0]),
            np.abs(axis[0] - canon[-1]),
            np.abs(axis[-1] - canon[0]),
            np.abs(axis[-1] - canon[-1]),
        ]
        return np.nanmin(diffs) <= tol

    if not _compatible_axis(lat, lat_can) or not _compatible_axis(lon, lon_can):
        # Doesn't look like the same grid → leave untouched
        return da

    # Match orientation of canonical grid
    lat_can_asc = lat_can[1] > lat_can[0]
    lon_can_asc = lon_can[1] > lon_can[0]

    lat_asc = lat[1] > lat[0]
    lon_asc = lon[1] > lon[0]

    if lat_asc != lat_can_asc:
        da = da.sortby("lat", ascending=lat_can_asc)
    if lon_asc != lon_can_asc:
        da = da.sortby("lon", ascending=lon_can_asc)

    # Overwrite with canonical coordinates (bit-identical)
    da = da.assign_coords(
        lat=("lat", lat_can.astype(da["lat"].dtype)),
        lon=("lon", lon_can.astype(da["lon"].dtype)),
    )
    return da

# -----------------------------------------------------------------------------
# ERA5-Land — 0–1 m soil moisture, monthly 0.5°
# -----------------------------------------------------------------------------

def era5land_to_1m_monthly_halfdeg_v1(registry) -> xr.DataArray:
    """
    Build ERA5-Land 0–1 m soil moisture (kg m-2) as monthly means on a 0.5°
    grid using swvl1–3 volumetric layers.

    Steps
    -----
    - Read monthly ERA5-Land files from the registry (single file or glob).
    - Normalize coordinates and restrict time to 1950–2020.
    - Convert volumetric layers (0–7, 7–28, 28–100 cm) to 0–1 m mass.
    - Coarsen from 0.1° to 0.5° by block mean (or xESMF if needed).
    - Snap to canonical ISIMIP 0.5° grid when shape-compatible.
    """
    path = registry.get_obs_raw("era5land")

    # Support both a single file path and a glob pattern
    files = sorted(glob.glob(path))
    if len(files) == 1:
        open_target = files[0]
    elif len(files) > 1:
        open_target = files
    elif os.path.exists(path):
        open_target = path
    else:
        raise FileNotFoundError(f"No ERA5-Land file(s) found at {path}")

    if isinstance(open_target, list):
        ds = xr.open_mfdataset(
            open_target,
            combine="by_coords",
            parallel=False,
            engine="netcdf4",
            chunks={"valid_time": 12, "latitude": 300, "longitude": 600},
        )
    else:
        ds = xr.open_dataset(
            open_target,
            engine="netcdf4",
            chunks={"valid_time": 12, "latitude": 300, "longitude": 600},
        )

    # Normalize coordinates and time
    if "longitude" in ds.coords and "lon" not in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    if "latitude" in ds.coords and "lat" not in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    if "time" not in ds.coords and "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    # Drop helper / auxiliary variables if present
    ds = _drop_vars_if_present(ds, ["number", "expver"])

    # Ensure longitudes in [-180, 180), lat ascending
    ds = _to_lon_m180_180(ds)
    if "lat" in ds.coords and (ds.lat.size > 1) and (ds.lat[1] < ds.lat[0]):
        ds = ds.sortby("lat")

    # Restrict to requested period before normalizing calendar
    ds = ds.sel(time=slice("1950-01-01", "2020-12-31"))

    # Normalize monthly calendar to proleptic_gregorian
    ds = to_proleptic_monthly(ds)

    # Detect layer variables and compute 0–1 m water mass
    v1, v2, v3 = _detect_era5l_vars(ds)
    thick = _ERA5L_THICK_M  # [0.07, 0.21, 0.72] m
    l1 = ds[v1].astype("float32")
    l2 = ds[v2].astype("float32")
    l3 = ds[v3].astype("float32")
    sm = (l1 * thick[0] + l2 * thick[1] + l3 * thick[2]) * np.float32(1000.0)  # kg m-2
    sm.name = "soilmoist_1m"

    # Regrid from 0.1° to 0.5° via block mean (or xESMF if needed).
    # This already includes a safe `_snap_to_canonical` call that only
    # overwrites coords when they truly match the canonical grid.
    sm = _maybe_block_coarsen_to_half_degree(sm, allow_xesmf=True)

    # Harmonize chunking for writing
    if {"time", "lat", "lon"}.issubset(sm.dims):
        sm = sm.chunk({"time": 12, "lat": 300, "lon": 600})

    sm.attrs.update({
        "units": "kg m-2",
        "long_name": "ERA5-Land 0–1 m soil moisture (converted from swvl1–3 volumetric)",
        "method": "sum(theta_i * thickness_i * 1000) for layers 1–3, block-mean to 0.5°",
        "calendar": "proleptic_gregorian",
        "target_depth_m": 1.0,
        "source_files": str(open_target),
        "note": (
            "Input is a single CDS NetCDF or a set of files; "
            "time normalized to proleptic monthly and clipped to 1950–2020."
        ),
    })

    return sm

    # NOTE: The chunking and attribute assignment for ERA5-Land appear below
    # due to historical edits and are left unchanged to preserve behavior.


# -----------------------------------------------------------------------------
# GRACE-DA-DM (Global V3.0) — root-zone percentile → monthly 0.5° (v0)
# -----------------------------------------------------------------------------

def _open_gracedadm_weekly_stack(registry) -> xr.Dataset:
    """
    Open all GRACE-DA-DM weekly files and normalize coordinates.

    Expects a glob from the registry key 'gracedadm_weekly'.
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

    # Normalize coordinate names and longitudes
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
    Build GRACE-DA-DM root-zone soil moisture percentile product as monthly
    means on a 0.5° grid (v0).

    The output is a percentile (0–100) with dims (time, lat, lon), clipped to
    2003-02 through 2020-12.
    """
    ds = _open_gracedadm_weekly_stack(registry)

    if "rtzsm_inst" not in ds.data_vars:
        raise KeyError(
            "Expected variable 'rtzsm_inst' (root zone soil moisture percentile) "
            "in GRACE-DA-DM files."
        )

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

    # Embed into canonical 0.5° ISIMIP mask grid (pad Antarctica with NaNs)
    lat_can, lon_can = get_canonical_grid()
    da_05 = da_05.reindex(
        lat=xr.DataArray(lat_can, dims="lat"),
        lon=xr.DataArray(lon_can, dims="lon"),
    )

    da_05.name = "rootzone_percentile"
    da_05.attrs.update({
        "units": "%",
        "long_name": "GRACE-DA-DM root-zone soil moisture percentile (monthly mean)",
        "standard_name": "root_zone_soil_moisture_percentile",
        "method": (
            "weekly percentiles averaged to monthly means; 0.25°→0.5° block mean; "
            "reindexed to canonical 0.5° ISIMIP landmask grid"
        ),
        "note": (
            "Percentiles (0–100) are averaged in time/space as a pragmatic summary "
            "statistic. Native GRACE domain excludes Antarctica; those cells are NaN "
            "on the canonical grid."
        ),
        "calendar": "proleptic_gregorian",
        "source": "GRACEDADM_CLSM025GL_7D v3.0",
    })

    return da_05


# -----------------------------------------------------------------------------
# GLEAM v4.2 (SMrz) — root-zone volumetric soil moisture, monthly 0.5° (v0)
#   - v4.2a 1980–2020
#   - v4.2a 2003–2020 (subset)
#   - v4.2b 2003–2020
# -----------------------------------------------------------------------------

def _to_month_start(ds: xr.Dataset) -> xr.Dataset:
    """
    Ensure monthly timestamps are at month start (00:00 on the first day).
    """
    if "time" not in ds.coords:
        return ds

    t = xr.cftime_range(
        start=str(ds.time.dt.year.min().item()) + "-01-01",
        end=str(ds.time.dt.year.max().item()) + "-12-01",
        freq="MS",
        calendar="proleptic_gregorian",
    )

    # Only re-stamp if monthly and lengths match
    if ds.sizes.get("time", -1) == t.size:
        ds = ds.assign_coords(time=t)
    return ds


def _open_gleam_stack(path_glob: str) -> xr.Dataset:
    """
    Open GLEAM files from a glob pattern and normalize coordinates and time.
    """
    files = sorted(glob.glob(path_glob))
    if not files:
        raise FileNotFoundError(f"No GLEAM files matched: {path_glob}")

    ds = xr.open_mfdataset(files, combine="by_coords", parallel=False)

    # Normalize coordinates
    if "longitude" in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    if "latitude" in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    ds = _to_lon_m180_180(ds)

    # Time handling: GLEAM is monthly proleptic_gregorian with end-of-month stamps
    ds = _to_month_start(ds)
    return ds


def _gleam_smrz_to_halfdeg_v0(ds: xr.Dataset) -> xr.DataArray:
    """
    Coarsen GLEAM SMrz from 0.1° to 0.5° using block means and keep volumetric
    units (m3 m-3).
    """
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
    """
    GLEAM v4.2a SMrz, 1980–2020, monthly 0.5°, volumetric (v0).
    """
    path_glob = registry.get_obs_raw("gleam42a")
    ds = _open_gleam_stack(path_glob)
    da = _gleam_smrz_to_halfdeg_v0(ds.sel(time=slice("1980-01-01", "2020-12-01")))
    return da


def gleam42a_2003_2020_v0(registry) -> xr.DataArray:
    """
    GLEAM v4.2a SMrz subset 2003–2020 (aligned with v4.2b period).
    """
    path_glob = registry.get_obs_raw("gleam42a")
    ds = _open_gleam_stack(path_glob)
    da = _gleam_smrz_to_halfdeg_v0(ds.sel(time=slice("2003-01-01", "2020-12-01")))
    return da


def gleam42b_2003_2020_v0(registry) -> xr.DataArray:
    """
    GLEAM v4.2b SMrz, 2003–2020, monthly 0.5°, volumetric (v0).
    """
    path_glob = registry.get_obs_raw("gleam42b")
    ds = _open_gleam_stack(path_glob)
    da = _gleam_smrz_to_halfdeg_v0(ds.sel(time=slice("2003-01-01", "2020-12-01")))
    return da


# Backward-compatible aliases (old naming)
gleam42a_full_v0 = gleam42a_1980_2020_v0
gleam42b_full_v0 = gleam42b_2003_2020_v0
# -----------------------------------------------------------------------------
# GLDAS — 0–1 m soil moisture, monthly 0.5°
# -----------------------------------------------------------------------------

def _drop_vars_if_present(ds: xr.Dataset, names: list[str]) -> xr.Dataset:
    """
    Drop variables from a Dataset if they exist.
    """
    present = [n for n in names if n in ds.variables]
    return ds.drop_vars(present) if present else ds


def _open_gldas_stack(path_glob: str) -> xr.Dataset:
    """
    Open GLDAS files from a glob pattern and normalize coordinates and calendar.
    """
    files = sorted(glob.glob(path_glob))
    if not files:
        raise FileNotFoundError(f"No GLDAS files matched: {path_glob}")

    ds = xr.open_mfdataset(files, combine="by_coords", parallel=False)

    # Normalize coordinates
    if "longitude" in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    if "latitude" in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    ds = _to_lon_m180_180(ds)

    # Normalize monthly calendar to proleptic_gregorian
    ds = to_proleptic_monthly(ds)
    ds = _drop_vars_if_present(ds, ["time_bnds"])

    # Enforce a regular monthly time axis, filling missing months with NaN
    if "time" in ds.coords:
        t0 = ds["time"].min()
        t1 = ds["time"].max()
        t_full = xr.cftime_range(
            start=f"{t0.dt.year.item():04d}-{t0.dt.month.item():02d}-01",
            end=f"{t1.dt.year.item():04d}-{t1.dt.month.item():02d}-01",
            freq="MS",
            calendar="proleptic_gregorian",
        )
        ds = ds.reindex(time=t_full)

    return ds


def _gldas_0_1m_sum(ds: xr.Dataset) -> xr.DataArray:
    """
    Sum GLDAS soil moisture layers over 0–100 cm.

    Uses:
      - SoilMoi0_10cm_inst
      - SoilMoi10_40cm_inst
      - SoilMoi40_100cm_inst
    """
    needed = ["SoilMoi0_10cm_inst", "SoilMoi10_40cm_inst", "SoilMoi40_100cm_inst"]
    missing = [v for v in needed if v not in ds.data_vars]
    if missing:
        raise KeyError(f"GLDAS: missing required soil moisture vars: {missing}")
    da = ds[needed[0]] + ds[needed[1]] + ds[needed[2]]
    da.name = "soilmoist_1m"
    return da


def gldas_v20_to_1m_monthly_halfdeg_v0(registry) -> xr.DataArray:
    """
    GLDAS v2.0 monthly: sum 0–100 cm soil moisture layers, block-mean to 0.5°,
    and embed into canonical ISIMIP 0.5° grid. Units: kg m-2.
    """
    path_glob = registry.get_obs_raw("gldas_v20")
    ds = _open_gldas_stack(path_glob)
    da = _gldas_0_1m_sum(ds)
    da = _maybe_block_coarsen_to_half_degree(da)

    # Embed into canonical ISIMIP 0.5° grid (Antarctica padded with NaNs)
    lat_can, lon_can = get_canonical_grid()
    da = da.reindex(
        lat=xr.DataArray(lat_can, dims="lat"),
        lon=xr.DataArray(lon_can, dims="lon"),
    )

    da.attrs.update({
        "units": "kg m-2",
        "long_name": (
            "GLDAS NOAH v2.0 soil moisture 0–1 m "
            "(sum of 0–10, 10–40, 40–100 cm)"
        ),
        "method": (
            "sum layer masses; block-mean 0.25°→0.5°; "
            "monthly proleptic_gregorian; reindexed to canonical 0.5° grid"
        ),
        "calendar": "proleptic_gregorian",
        "source": "GLDAS2.0 NOAH monthly",
        "target_depth_m": 1.0,
        "note": (
            "GLDAS domain excludes Antarctica; those cells are NaN on the "
            "canonical grid."
        ),
        "source_files": path_glob,
    })
    return da


def gldas_v21_to_1m_monthly_halfdeg_v0(registry) -> xr.DataArray:
    """
    GLDAS v2.1 monthly: sum 0–100 cm soil moisture layers, block-mean to 0.5°,
    then reindex to canonical 0.5° grid. Units: kg m-2.
    """
    path_glob = registry.get_obs_raw("gldas_v21")
    ds = _open_gldas_stack(path_glob)
    da = _gldas_0_1m_sum(ds)
    da = _maybe_block_coarsen_to_half_degree(da)

    # Embed into canonical ISIMIP 0.5° grid (Antarctica padded with NaNs)
    lat_can, lon_can = get_canonical_grid()
    da = da.reindex(
        lat=xr.DataArray(lat_can, dims="lat"),
        lon=xr.DataArray(lon_can, dims="lon"),
    )

    da.attrs.update({
        "units": "kg m-2",
        "long_name": (
            "GLDAS NOAH v2.1 soil moisture 0–1 m "
            "(sum of 0–10, 10–40, 40–100 cm)"
        ),
        "method": (
            "sum layer masses; block-mean 0.25°→0.5°; "
            "monthly proleptic_gregorian; reindexed to canonical 0.5° grid"
        ),
        "calendar": "proleptic_gregorian",
        "source": "GLDAS2.1 NOAH monthly",
        "target_depth_m": 1.0,
        "note": (
            "GLDAS domain excludes Antarctica; those cells are NaN on the "
            "canonical grid."
        ),
        "source_files": path_glob,
    })
    return da


# -----------------------------------------------------------------------------
# SoMo.ml — 0–50 cm soil moisture, monthly 0.5°
# -----------------------------------------------------------------------------

def _open_somoml_layer(registry, key: str, var_name: str) -> xr.DataArray:
    """
    Open all yearly files for one SoMo.ml layer and return a daily DataArray.
    """
    pat = registry.get_obs_raw(key)  # e.g. .../SoMo.ml_v1_layer1/*.nc
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
        data_vars = [v for v in ds.data_vars]
        if len(data_vars) == 1:
            ds = ds.rename({data_vars[0]: var_name})
        else:
            raise KeyError(
                f"Expected variable {var_name} not found in SoMo.ml files."
            )

    da = ds[var_name]

    # Ensure lat ascending & lon in [-180,180) if needed
    if "lat" in da.coords and (da.lat[0] > da.lat[-1]):
        da = da.sortby("lat")
    if "lon" in da.coords and (da.lon.max() > 180.0):
        da = da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon")

    return da


def _somoml_daily_to_monthly(da: xr.DataArray) -> xr.DataArray:
    """
    Convert daily SoMo.ml data to monthly means, month-start timestamps (MS).
    """
    return da.resample(time="MS").mean("time", skipna=True)


def _somoml_depth_weighted_volumetric(
    l1: xr.DataArray,
    l2: xr.DataArray,
    l3: xr.DataArray,
) -> xr.DataArray:
    """
    Depth-weighted volumetric average (m3/m3) across 0–50 cm:

      - layer1: 0–10 cm (0.1 m)
      - layer2: 10–30 cm (0.2 m)
      - layer3: 30–50 cm (0.2 m)
    """
    w1, w2, w3 = 0.1, 0.2, 0.2  # meters
    depth = w1 + w2 + w3        # 0.5 m
    return (w1 * l1 + w2 * l2 + w3 * l3) / depth


def _somoml_volumetric_to_mass(theta_0p5m: xr.DataArray) -> xr.DataArray:
    """
    Convert volumetric water content (m3/m3) to water mass per area (kg m-2)
    over 0.5 m.
    """
    rho_w = 1000.0  # kg/m3
    depth_m = 0.5   # meters
    return theta_0p5m * rho_w * depth_m


def somoml_to_0p5m_monthly_halfdeg_v0(registry) -> xr.Dataset:
    """
    Build SoMo.ml 0–50 cm product as monthly means on a 0.5° grid (v0).

    Output Dataset contains a standardized variable 'soilmoist_1m'
    (depth documented as 0–0.5 m).
    """
    # Load daily for each layer
    l1 = _open_somoml_layer(registry, "somo_ml_layer1", "layer1")
    l2 = _open_somoml_layer(registry, "somo_ml_layer2", "layer2")
    l3 = _open_somoml_layer(registry, "somo_ml_layer3", "layer3")

    # Daily → monthly means
    l1m = _somoml_daily_to_monthly(l1)
    l2m = _somoml_daily_to_monthly(l2)
    l3m = _somoml_daily_to_monthly(l3)

    # Align monthly time axis across layers
    l1m, l2m = xr.align(l1m, l2m, join="inner")
    l1m, l3m = xr.align(l1m, l3m, join="inner")

    # Depth-weighted volumetric average over 0–50 cm
    theta_0p5m = _somoml_depth_weighted_volumetric(l1m, l2m, l3m)

    # Convert to kg m-2 (water column over 0.5 m)
    sm_kgm2 = _somoml_volumetric_to_mass(theta_0p5m)

    # Regrid to 0.5° by block mean if needed
    sm_kgm2_05 = _maybe_block_coarsen_to_half_degree(sm_kgm2)

    # Snap to canonical ISIMIP 0.5° grid (lat/lon bit-identical)
    sm_kgm2_05 = _snap_to_canonical(sm_kgm2_05)

    # Pack to Dataset with standardized variable name for compatibility
    da = sm_kgm2_05.astype("float32").drop_vars(
        [v for v in sm_kgm2_05.coords if v not in ("time", "lat", "lon")],
        errors="ignore",
    )
    da = da.where(np.isfinite(da), other=np.float32(-9999.0))
    da.name = "soilmoist_1m"

    da.attrs.update({
        "standard_name": "soil_moisture_content",
        "long_name": (
            "SoMo.ml depth-integrated soil moisture (0–0.5 m) converted "
            "to water mass"
        ),
        "units": "kg m-2",
        "source_product": (
            "SoMo.ml v1 (layer1 0–10 cm, layer2 10–30 cm, layer3 30–50 cm)"
        ),
        "native_units": "m3 m-3 (volumetric)",
        "conversion": "theta * 1000 kg/m3 * 0.5 m",
        "depth_m": 0.5,
        "processing_version": "v0",
        "regridding": "0.25°→0.5° by 2×2 block mean; snapped to canonical 0.5° grid",
    })

    ds_out = da.to_dataset()

    # Coordinate metadata tidy-up
    for c in ("lat", "lon"):
        if c in ds_out.coords:
            ds_out[c].attrs.pop("_FillValue", None)

    ds_out.attrs.update({
        "title": (
            "SoMo.ml 0–50 cm soil moisture (monthly, 0.5°), converted to kg m-2"
        ),
        "comment": (
            "Depth is 0–0.5 m; variable kept as 'soilmoist_1m' for pipeline "
            "compatibility."
        ),
        "institution": "Your lab / project",
        "history": (
            "SoMo.ml layer1/2/3 daily → monthly mean; depth-weighted; mass "
            "conversion; block coarsen to 0.5°; snapped to canonical ISIMIP "
            "0.5° grid."
        ),
    })

    return ds_out


# -----------------------------------------------------------------------------
# MERRA-2 LAND — 0–1 m soil moisture, monthly 0.5°
# -----------------------------------------------------------------------------

def _interp_lon_to_half_degree(da: xr.DataArray) -> xr.DataArray:
    """
    Remap longitude from 0.625° spacing (e.g. MERRA-2) to 0.5° by 1D linear
    interpolation. Latitude is left unchanged (already 0.5°).
    """
    if "lon" not in da.coords:
        raise ValueError("DataArray must have 'lon' coordinate.")

    # Ensure lon in [-180,180)
    da = _to_lon_m180_180(da.to_dataset(name="var"))["var"]

    lon_out = np.arange(-179.75, 180.0, 0.5)  # 0.5° centers
    out = da.interp(lon=lon_out, method="linear")
    out.attrs.update(da.attrs)
    out.attrs["regrid_note"] = (
        "1D linear lon interpolation 0.625°→0.5° for MERRA-2"
    )
    return out


def merra2_land_to_1m_monthly_halfdeg_v1(registry) -> xr.DataArray:
    """
    Build MERRA-2 LAND 0–1 m soil moisture (kg m-2) using a depth-weighted
    blend:

      - SFMC (surface layer volumetric, 0–0.05 m) with weight 0.05
      - RZMC (root-zone volumetric, 0.05–1.0 m) with weight 0.95

    SFMC and RZMC are treated as volumetric water content (m3 m-3) over their
    respective layers. A depth-weighted average over 0–1 m is computed,
    converted to kg m-2, and regridded to 0.5° on a proleptic_gregorian
    monthly calendar.

    Native grid: 0.5° (lat) × 0.625° (lon).
    """
    path_glob = registry.get_obs_raw("merra2_land")
    files = sorted(glob.glob(path_glob))
    if not files:
        raise FileNotFoundError(f"No MERRA2-LAND files found at {path_glob}")

    # Open and normalize coordinates and time
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
        "note": (
            "Depth weights follow project decision; constants not read from "
            "ancillary files."
        ),
    })

    # Regrid lon: 0.625° → 0.5° (this already matches canonical lon centers)
    sm_05 = _interp_lon_to_half_degree(sm).astype("float32")

    # Regrid lat to the canonical ISIMIP 0.5° grid (centers -89.75..89.75)
    lat_can, lon_can = get_canonical_grid()

    # xarray.interp expects increasing coordinate; use ascending copy of canonical lats
    lat_can_asc = np.sort(lat_can)

    # Ensure ascending lat for interpolation
    if sm_05["lat"].size > 1 and sm_05["lat"][1] < sm_05["lat"][0]:
        sm_05 = sm_05.sortby("lat")

    sm_rg = sm_05.interp(lat=("lat", lat_can_asc), method="linear")

    # Reorder to canonical descending latitude and overwrite coords bit-wise
    sm_rg = sm_rg.sortby("lat", ascending=False)
    sm_rg = sm_rg.assign_coords(
        lat=("lat", lat_can.astype(sm_rg["lat"].dtype)),
        lon=("lon", lon_can.astype(sm_rg["lon"].dtype)),
    )

    note_prev = sm_rg.attrs.get("regrid_note", "")
    extra_note = (
        " + 1D linear lat interpolation to canonical 0.5° ISIMIP grid"
    )
    sm_rg.attrs["regrid_note"] = (note_prev + extra_note).strip(" +")

    sm_rg.attrs.update({
        "units": "kg m-2",
        "long_name": (
            "MERRA-2 0–1 m soil moisture (SFMC 0–0.05 m and RZMC 0.05–1.0 m "
            "depth-weighted)"
        ),
        "standard_name": "soil_moisture_content",
        "method": (
            "theta = (0.05*SFMC + 0.95*RZMC)/1.0; soilmoist = theta * 1000 kg/m3 "
            "* 1.0 m; linear interpolation 0.625°→0.5° in lon and "
            "native→canonical 0.5° in lat"
        ),
        "calendar": "proleptic_gregorian",
        "target_depth_m": 1.0,
        "source_files": path_glob,
        "native_grid": "0.5° lat × 0.625° lon",
        "weights_note": "dzsf=0.05 m and dzrz=0.95 m as global constants.",
    })

    return sm_rg


# -----------------------------------------------------------------------------
# GDO SMIA / ENSMIA — standardized soil moisture anomalies, monthly 0.5°
# -----------------------------------------------------------------------------

def _require_rioxarray():
    """
    Ensure rioxarray (and rasterio) are available for reading GDO GeoTIFFs.
    """
    try:
        import rioxarray as _rxr  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "Reading GDO-SMIA GeoTIFFs requires 'rioxarray' and 'rasterio' in "
            "your environment (conda-forge: rioxarray, rasterio)."
        ) from exc


def _open_gdo_smia_geotiffs(registry) -> xr.DataArray:
    """
    Open all GDO-SMIA GeoTIFFs (ten-daily anomaly) and stack into a single
    DataArray with dims (time, lat, lon).

    Filenames are expected to contain a YYYYMMDD date stamp.
    """
    _require_rioxarray()
    import re
    import rioxarray as rxr

    # Collect all .tif recursively from the registry glob
    path_glob = registry.get_obs_raw("gdo_smia")
    file_list = sorted(glob.glob(path_glob, recursive=True))
    if not file_list:
        raise FileNotFoundError(
            f"No GDO-SMIA GeoTIFFs matched pattern: {path_glob}"
        )

    da_list = []
    time_list = []

    # Regex to capture the first YYYYMMDD in filename
    date_re = re.compile(
        r"(?P<yyyymmdd>(?P<yyyy>\d{4})(?P<mm>\d{2})(?P<dd>\d{2}))"
    )

    for fp in file_list:
        m = date_re.search(os.path.basename(fp))
        if not m:
            # Skip files without a parsable date
            continue

        y, mth, d = int(m.group("yyyy")), int(m.group("mm")), int(m.group("dd"))
        tstamp = np.datetime64(f"{y:04d}-{mth:02d}-{d:02d}")

        da = rxr.open_rasterio(fp, masked=True, chunks={"x": 1024, "y": 1024})

        # Drop band dimension (single-band rasters)
        if "band" in da.dims and da.sizes["band"] == 1:
            da = da.squeeze("band", drop=True)

        # Rename axes to lon/lat and ensure proper orientation
        rename = {}
        if "x" in da.dims:
            rename["x"] = "lon"
        if "y" in da.dims:
            rename["y"] = "lat"
        da = da.rename(rename)

        # Ensure lon in [-180,180), sort lon/lat ascending for coarsen
        if "lon" in da.coords:
            lon = da["lon"]
            if float(lon.max().item()) > 180.0 + 1e-6:
                da = da.assign_coords(
                    lon=((lon + 180.0) % 360.0) - 180.0
                ).sortby("lon")
        if "lat" in da.coords and da.lat.size > 1 and da.lat[1] < da.lat[0]:
            da = da.sortby("lat")

        da = da.astype("float32")

        # Clean up any existing 'time' metadata from GeoTIFFs
        if "time" in da.dims:
            if da.sizes.get("time", 0) == 1:
                da = da.isel(time=0, drop=True)
            else:
                da = da.rename({"time": "_time_tmp"}).reset_coords(
                    "_time_tmp", drop=True
                )

        if ("time" in da.coords) and ("time" not in da.dims):
            da = da.reset_coords("time", drop=True)

        # Attach dekad timestamp as a proper time dimension
        da = da.expand_dims(time=[tstamp]).assign_coords(time=("time", [tstamp]))
        da.name = "smia"

        # Replace nodata with NaN if not already masked
        nodata = da.rio.nodata if hasattr(da, "rio") else None
        if nodata is not None and np.isfinite(nodata):
            da = da.where(da != np.float32(nodata))

        da_list.append(da)
        time_list.append(tstamp)

    if not da_list:
        raise RuntimeError(
            "No valid GDO-SMIA GeoTIFFs with parsable YYYYMMDD dates were found."
        )

    da = xr.concat(da_list, dim="time").sortby("time")

    # Keep only the requested era (1995–2020) for v0
    da = da.sel(time=slice("1995-01-01", "2020-12-31"))
    da.name = "soilmoist_anom_std"
    da.attrs.update({
        "units": "dimensionless",
        "long_name": "GDO Soil Moisture Index Anomaly (SMA)",
        "note": (
            "Values are standardized anomalies (baseline 1995–2024) from "
            "LISFLOOD (top two layers)."
        ),
        "source": "GDO/EDO (EFAS5 LISFLOOD), GeoTIFF inputs",
    })
    return da


def gdo_smia_to_monthly_halfdeg_v0(registry) -> xr.DataArray:
    """
    Build GDO-SMIA (LISFLOOD-only standardized anomaly) as monthly 0.5° (v0).

    Steps:
      - Read GeoTIFFs via rioxarray, stack to (time, lat, lon).
      - For each month, select the last available dekad (nearest to month end).
      - Coarsen ~1 arcmin (~0.0167°) to 0.5° using exact 30×30 block means.
      - Snap native 0.5° coords to nearest canonical ISIMIP 0.5° values.
      - Reindex to the full canonical grid, padding outside the native domain
        with NaNs.
    """
    da = _open_gdo_smia_geotiffs(registry)

    # Last dekad per month → monthly series
    da_monthly = da.resample(time="MS").map(lambda x: x.isel(time=-1))

    # 1 arcmin → 0.5° block mean (still on LISFLOOD domain, ~60S–90N)
    da_05 = _maybe_block_coarsen_to_half_degree(da_monthly)

    # Snap native 0.5° coords to nearest canonical coords
    lat_can, lon_can = get_canonical_grid()
    lat_src = da_05["lat"].values
    lon_src = da_05["lon"].values

    def _snap_axis(src, can, tol=5e-3):
        """
        Map each coordinate in src to the nearest value in can, preserving
        the original order of src and ensuring a one-to-one mapping.
        """
        idxs = []
        for v in src:
            j = int(np.argmin(np.abs(can - v)))
            if np.abs(can[j] - v) > tol:
                raise RuntimeError(
                    f"GDO-SMIA: cannot snap coord value {v} to canonical grid "
                    f"(nearest={can[j]}, diff={np.abs(can[j]-v)})"
                )
            idxs.append(j)

        if len(set(idxs)) != len(src):
            raise RuntimeError(
                "GDO-SMIA: snapping produced duplicate canonical indices; "
                "check that both grids are regular 0.5°."
            )

        idxs = np.asarray(idxs, dtype=int)
        return can[idxs]

    lat_sub = _snap_axis(lat_src, lat_can)
    lon_sub = _snap_axis(lon_src, lon_can)

    # Overwrite existing coords on the overlapping domain with canonical values
    da_05 = da_05.assign_coords(
        lat=("lat", lat_sub.astype(da_05["lat"].dtype)),
        lon=("lon", lon_sub.astype(da_05["lon"].dtype)),
    )

    # Embed into the full canonical grid (outside domain → NaN)
    da_05 = da_05.reindex(
        lat=xr.DataArray(lat_can, dims="lat"),
        lon=xr.DataArray(lon_can, dims="lon"),
    )

    da_05.name = "soilmoist_anom_std"
    da_05.attrs.update({
        "units": "dimensionless",
        "long_name": "GDO SMIA (standardized anomaly), monthly, 0.5°",
        "method": (
            "GeoTIFF stack → monthly last-dekad selection; "
            "1 arcmin → 0.5° block mean; "
            "native 0.5° coords snapped to nearest ISIMIP 0.5°; "
            "reindexed to canonical ISIMIP 0.5° grid"
        ),
        "calendar": "proleptic_gregorian",
        "target_grid": "0.5° regular lat/lon (ISIMIP mask, no Antarctica / Greenland)",
        "spatial_coverage_note": (
            "Native LISFLOOD domain covers approx. 60°S–90°N; "
            "cells outside this are NaN on the canonical grid."
        ),
    })
    return da_05


def _open_gdo_ensmia_stack(registry) -> xr.Dataset:
    """
    Open all GDO-ENSMIA anomaly files and normalize coordinates.
    """
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
    Build GDO Ensemble Soil Moisture Index product (v0) on a monthly 0.5° grid.

    Steps:
      - Select 3rd dekad (last timestep) per month → monthly series.
      - Coarsen spatially from 0.1° → 0.5° via block mean.
      - Snap native 0.5° coords to canonical ISIMIP 0.5° grid.
      - Embed into full canonical grid (outside native domain → NaN).
      - Output standardized anomaly (-3 to 3, dimensionless).
    """
    ds = _open_gdo_ensmia_stack(registry)

    if "smant" not in ds.data_vars:
        raise KeyError("Expected variable 'smant' in GDO-ENSMIA files.")

    da = ds["smant"].astype("float32")
    da = _nan_fill(da)

    # Last timestep (3rd dekad) per month
    da_monthly = da.resample(time="MS").map(lambda x: x.isel(time=-1))

    # Coarsen spatially 0.1° → 0.5° (mean of standardized index)
    da_05 = _maybe_block_coarsen_to_half_degree(da_monthly)

    # Snap to canonical ISIMIP 0.5° grid on the overlapping domain
    lat_can, lon_can = get_canonical_grid()
    lat_src = da_05["lat"].values
    lon_src = da_05["lon"].values

    def _snap_axis(src, can, tol=5e-3):
        """
        Map each coordinate in src to the nearest value in can, preserving
        the original order of src and ensuring a one-to-one mapping.
        """
        idxs = []
        for v in src:
            j = int(np.argmin(np.abs(can - v)))
            if np.abs(can[j] - v) > tol:
                raise RuntimeError(
                    f"GDO-ENSMIA: cannot snap coord value {v} to canonical grid "
                    f"(nearest={can[j]}, diff={np.abs(can[j]-v)})"
                )
            idxs.append(j)

        if len(set(idxs)) != len(src):
            raise RuntimeError(
                "GDO-ENSMIA: snapping produced duplicate canonical indices; "
                "check that both grids are regular 0.5°."
            )

        idxs = np.asarray(idxs, dtype=int)
        return can[idxs]

    lat_sub = _snap_axis(lat_src, lat_can)
    lon_sub = _snap_axis(lon_src, lon_can)

    # Overwrite coords on the overlapping domain with the canonical values
    da_05 = da_05.assign_coords(
        lat=("lat", lat_sub.astype(da_05["lat"].dtype)),
        lon=("lon", lon_sub.astype(da_05["lon"].dtype)),
    )

    # Embed into the full canonical grid (outside domain → NaN)
    da_05 = da_05.reindex(
        lat=xr.DataArray(lat_can, dims="lat"),
        lon=xr.DataArray(lon_can, dims="lon"),
    )

    da_05.name = "soilmoist_anom_std"
    da_05.attrs.update({
        "units": "dimensionless",
        "long_name": (
            "GDO Ensemble Soil Moisture Anomaly (standardized index), "
            "monthly, 0.5°"
        ),
        "method": (
            "selected 3rd dekad per month; block-mean 0.1°→0.5°; "
            "native 0.5° coords snapped to nearest ISIMIP 0.5°; "
            "reindexed to canonical ISIMIP 0.5° grid"
        ),
        "calendar": "proleptic_gregorian",
        "note": (
            "Index represents standardized anomalies (-3 to +3) derived from "
            "LISFLOOD, MODIS LST, and ESA CCI."
        ),
        "source": "Global Drought Observatory, JRC (2001–2020)",
        "target_grid": "0.5° regular lat/lon (ISIMIP mask, no Antarctica / Greenland)",
    })

    return da_05
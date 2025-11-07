import os
import numpy as np
import xarray as xr
from datetime import datetime, timezone
import argparse

# ---- INPUTS ----
parser = argparse.ArgumentParser(description="Create WEB-DHM-SG land-cover derived total/root depths as NetCDF.")
parser.add_argument("--asc", dest="asc_path", default="/Users/thchilly/projects/sm_attribution/data/ancils/web-dhm-sg/land use.asc", help="Path to ESRI ASCII land-cover file")
parser.add_argument("--legend", dest="leg_path", default="/Users/thchilly/projects/sm_attribution/data/ancils/web-dhm-sg/gbsbm2geo20.leg", help="Path to legend file")
parser.add_argument("--out", dest="out_nc", default="/Users/thchilly/projects/sm_attribution/data/ancils/web-dhm-sg/webdhmsg_landcover_depths_0p5deg.nc", help="Output NetCDF path")
parser.add_argument("--ref-nc", dest="ref_nc", default=None, help="Optional reference NetCDF with target lat/lon centers to adopt (e.g., a WEB-DHM-SG model file)")
args = parser.parse_args()

asc_path = args.asc_path
leg_path = args.leg_path
out_nc   = args.out_nc
ref_nc   = args.ref_nc

# ---- Parse ESRI ASCII header + grid ----
header = {}
data_lines = []
with open(asc_path, "r") as f:
    # first 6 lines are header in ESRI ASCII
    for _ in range(6):
        k, v = f.readline().strip().split(None, 1)
        header[k.lower()] = float(v)
    # remaining lines are rows of data (top -> bottom)
    for line in f:
        if line.strip():
            data_lines.append(line)

ncols     = int(header["ncols"])
nrows     = int(header["nrows"])
xllcorner = header["xllcorner"]
yllcorner = header["yllcorner"]
cellsize  = header["cellsize"]
nodata    = header.get("nodata_value", -9999.0)

# load values into 2D array (rows: north->south)
vals = np.loadtxt(data_lines, dtype=float)
assert vals.shape == (nrows, ncols)

# ---- Build lon/lat 1D coords for cell centers ----
lon = xllcorner + cellsize*(np.arange(ncols) + 0.5)
# Wrap longitude to [-180, 180)
lon = ((lon + 180) % 360) - 180
lon = np.sort(lon)

# ESRI ASCII rows are north->south; first row is the northernmost band
lat_north_to_south = yllcorner + cellsize*(nrows - np.arange(nrows) - 0.5)
# Ascending latitude for consistency:
lat = lat_north_to_south[::-1]
# Flip data so that row 0 corresponds to southernmost latitude
vals_asc_lat = vals[::-1, :]
# Ensure ascending latitudes (south to north)
if lat[0] > lat[-1]:
    lat = lat[::-1]
    vals_asc_lat = vals_asc_lat[::-1, :]

# Mask NODATA
lc = np.where(vals_asc_lat == nodata, np.nan, vals_asc_lat).astype(float)

# ---- Build class→depth lookups ----
# (A) Total depth of three layers (m)
total_depth_map = {
    1: 3.5,  # Broadleaf Evergreen Trees
    2: 2.0,  # Broadleaf Deciduous Trees
    3: 2.0,  # Broadleaf + Needleleaf Trees
    4: 2.0,  # Needleleaf Evergreen Trees
    5: 2.0,  # Needleleaf Deciduous Trees
    6: 2.0,  # Short Veg / C4 Grassland
    7: 2.0,  # Shrubs w/ bare soil
    8: 2.0,  # Dwarf trees & shrubs
    9: 2.0,  # Agriculture / C3 Grassland
    10: 1.0, # Water/Wetlands -> default 1 m
    11: 1.0, # Ice/Snow -> default 1 m
    100: np.nan # NO DATA -> np.nan 
}
# (B) Rooting depth (m) — from Wei’s confirmation
root_depth_map = {
    1: 2.0,  # Broadleaf Evergreen Trees
    2: 1.5,  # Broadleaf Deciduous Trees
    3: 1.5,  # Broadleaf + Needleleaf Trees
    4: 1.5,  # Needleleaf Evergreen Trees
    5: 1.5,  # Needleleaf Deciduous Trees
    6: 1.5,  # Short Veg / C4 Grassland
    7: 1.5,  # Shrubs w/ bare soil
    8: 1.5,  # Dwarf trees & shrubs
    9: 1.5,  # Agriculture / C3 Grassland
    10: 1.0, # Water/Wetlands -> default 1 m
    11: 1.0, # Ice/Snow -> default 1 m
    100: np.nan # NO DATA -> np.nan
}

# helper to map class codes to depth
def map_depth(arr, lut, default=np.nan):
    out = np.full(arr.shape, default, dtype=np.float32)
    # Only map where class is finite
    finite = np.isfinite(arr)
    codes = arr[finite].astype(int)
    # vectorized mapping
    mapped = np.array([lut.get(c, default) for c in codes], dtype=np.float32)
    out[finite] = mapped
    return out

total_depth = map_depth(lc, total_depth_map, default=np.nan)
root_depth  = map_depth(lc, root_depth_map,  default=np.nan)

# ---- Build xarray Dataset + attrs ----
ds = xr.Dataset(
    data_vars=dict(
        webdhmsg_landcover=(("lat","lon"), lc.astype(np.float32)),
        webdhmsg_total_depth=(("lat","lon"), total_depth),
        webdhmsg_root_depth=(("lat","lon"),  root_depth),
    ),
    coords=dict(
        lat=("lat", lat, {"units":"degrees_north", "standard_name":"latitude", "long_name":"latitude"}),
        lon=("lon", lon, {"units":"degrees_east",  "standard_name":"longitude","long_name":"longitude"}),
    ),
    attrs=dict(
        title="WEB-DHM-SG land-cover derived depths",
        institution="Provided by Wei Qi (WEB-DHM-SG); processed by TUC HydroMech Group",
        source_ascii=os.path.basename(asc_path),
        source_legend=os.path.basename(leg_path),
        description="Land-cover classes from WEB-DHM-SG ancillary (SiB2/IGBP-like) raster at 0.5°; "
                    "mapped to total soil depth (three-layer) and rooting depth per class.",
        contact="qiwei_waterresources@hotmail.com; atsilimigkras1@tuc.gr",
        conventions="CF-1.8",
        license="As provided by WEB-DHM-SG team; derived ancillary for research use",
        history=f"Created {datetime.now(timezone.utc).isoformat(timespec='seconds')} by sm_attribution pipeline helper",
    ),
)

# variable attrs
ds["webdhmsg_landcover"].attrs.update({
    "long_name": "WEB-DHM-SG land-cover class (SiB2 legend)",
    "units": "-",
    "grid_mapping": "latitude_longitude",
    "legend": (
        "0: Interrupted areas; "
        "1: Broadleaf Evergreen Trees; 2: Broadleaf Deciduous Trees; 3: Broadleaf+Needleleaf Trees; "
        "4: Needleleaf Evergreen Trees; 5: Needleleaf Deciduous Trees; 6: Short Vegetation/C4 Grassland; "
        "7: Shrubs with Bare Soil; 8: Dwarf Trees and Shrubs; 9: Agriculture or C3 Grassland; "
        "10: Water/Wetlands; 11: Ice/Snow; 100: No data"
    ),
})

ds["webdhmsg_total_depth"].attrs.update({
    "long_name": "Total depth of three soil-moisture layers (per class)",
    "units": "m",
    "notes": "Mapped from land-cover using WEB-DHM-SG ISIMIP3a configuration; 10/11/100 set to 1.0 m by convention.",
})

ds["webdhmsg_root_depth"].attrs.update({
    "long_name": "Rooting depth (per class)",
    "units": "m",
    "notes": "Mapped from land-cover using WEB-DHM-SG ISIMIP3a configuration; 10/11/100 set to 1.0 m by convention.",
})

# match WEB-DHM-SG grid: lon ascending, lat DESCENDING
ds = ds.sortby("lon")
ds = ds.isel(lat=slice(None, None, -1))  # flip latitude order

# ---- Write NetCDF ----
comp = dict(zlib=True, complevel=4, shuffle=True)
encoding = {v: comp | {"_FillValue": np.float32(np.nan)} for v in ["webdhmsg_total_depth","webdhmsg_root_depth","webdhmsg_landcover"]}
for v in ["webdhmsg_total_depth", "webdhmsg_root_depth", "webdhmsg_landcover"]:
    if "_FillValue" in ds[v].attrs:
        del ds[v].attrs["_FillValue"]
ds.to_netcdf(out_nc, format="NETCDF4", encoding=encoding)
print(f"Wrote: {out_nc}")
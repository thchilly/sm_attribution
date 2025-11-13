#!/usr/bin/env python
"""
Build an ISIMIP-style land–sea mask without Antarctica and without Greenland.

- Starts from the "isimip_no_ant" mask defined in data_registry.yml
- Uses a Greenland shapefile ("greenland_gl") to remove Greenland (+ buffer)
- Writes out "isimip_no_ant_nogreenland" to the ancils/landmask location
"""

from __future__ import annotations

import os
from typing import Dict

import geopandas as gpd
import numpy as np
import xarray as xr

from sm_attribution.io.registry import default_registry


def _expand_paths(template: str, paths: Dict[str, str]) -> str:
    """Expand {paths.*} tokens in a registry template string."""
    out = template
    for key, value in paths.items():
        out = out.replace("{paths." + key + "}", value)
    return out


def main() -> None:
    reg = default_registry()
    cfg = reg.cfg_dict
    paths = cfg["paths"]

    in_tmpl = cfg["ancils"]["landmask"]["isimip_no_ant"]
    out_tmpl = cfg["ancils"]["landmask"]["isimip_no_ant_nogreenland"]
    shp_tmpl = cfg["ancils"]["shapes"]["greenland_gl"]

    in_path = _expand_paths(in_tmpl, paths)
    out_path = _expand_paths(out_tmpl, paths)
    shp_path = _expand_paths(shp_tmpl, paths)

    print("Input mask :", in_path)
    print("Shapefile  :", shp_path)
    print("Output mask:", out_path)

    # ------------------------------------------------------------------ #
    # Load base mask (no Antarctica)
    # ------------------------------------------------------------------ #
    ds = xr.open_dataset(in_path)
    var = "mask" if "mask" in ds.data_vars else list(ds.data_vars)[0]
    m = ds[var]

    # Drop dummy time dimension if present
    if "time" in m.dims:
        time_coord = ds["time"]
        m2d = m.isel(time=0, drop=True)
    else:
        time_coord = xr.DataArray([0], dims=("time",), name="time")
        m2d = m

    lat = ds["lat"]
    lon = ds["lon"]

    # Convert to boolean land mask from numeric mask
    fill_value = m.attrs.get("_FillValue") or m.attrs.get("missing_value")
    if fill_value is not None:
        m2d = m2d.where(m2d != fill_value)

    land = m2d > 0.5

    # ------------------------------------------------------------------ #
    # Load Greenland polygon and add buffer
    # ------------------------------------------------------------------ #
    gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
    green_poly = gdf.geometry.union_all()

    # Each grid cell is 0.5 degrees; buffer_deg controls halo around Greenland
    buffer_deg = 1.0
    green_poly = green_poly.buffer(buffer_deg)

    # ------------------------------------------------------------------ #
    # Build grid of cell centers and test membership in Greenland polygon
    # ------------------------------------------------------------------ #
    lat2d, lon2d = xr.broadcast(lat, lon)  # (lat, lon)
    pts = gpd.GeoSeries.from_xy(
        lon2d.values.ravel(),
        lat2d.values.ravel(),
        crs="EPSG:4326",
    )
    inside = pts.within(green_poly).values.reshape(lat2d.shape)

    inside_da = xr.DataArray(
        inside,
        coords={"lat": lat2d["lat"], "lon": lon2d["lon"]},
        dims=("lat", "lon"),
        name="inside_greenland",
    )

    # Land & inside Greenland → water
    land_nogreen = land.where(~inside_da, False)

    # ------------------------------------------------------------------ #
    # Back to numeric mask: 1 over land, NaN over sea
    # ------------------------------------------------------------------ #
    new_mask2d = xr.where(land_nogreen, 1.0, np.nan).astype("float32")

    # Preserve original attributes except fill/missing
    new_mask2d.attrs.update(
        {k: v for k, v in m2d.attrs.items() if k not in ("_FillValue", "missing_value")}
    )
    new_mask2d.attrs["_FillValue"] = np.float32(np.nan)

    # Restore dummy time dimension
    new_mask = new_mask2d.expand_dims(time=time_coord)

    ds_out = xr.Dataset(
        {"mask": new_mask, "lat": lat, "lon": lon, "time": time_coord},
        attrs=ds.attrs,
    )

    comp = dict(zlib=True, complevel=4, shuffle=True)
    enc = {"mask": comp}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ds_out.to_netcdf(out_path, encoding=enc)

    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
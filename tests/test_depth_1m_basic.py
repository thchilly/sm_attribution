import numpy as np
import pandas as pd
import xarray as xr

from sm_attribution.preprocess.depth_1m import jules_w2_to_1m, miroc_integ_land_to_1m

def _mk_time(n=3):
    return xr.cftime_range("2001-01-01", periods=n, freq="MS")

def test_jules_sum_first_three_layers():
    # build (depth=4, time=3, lat=1, lon=1)
    time = _mk_time()
    da = xr.DataArray(
        np.stack([
            np.ones((3,1,1))*1,  # L1
            np.ones((3,1,1))*2,  # L2
            np.ones((3,1,1))*3,  # L3
            np.ones((3,1,1))*100,# L4 (ignored)
        ], axis=0),
        dims=("depth", "time", "lat", "lon"),
        coords={"depth": [0,1,2,3], "time": time, "lat":[10.0], "lon":[20.0]},
        name="soilmoist",
    )
    ds = xr.Dataset({"soilmoist": da})
    out = jules_w2_to_1m(ds, "obsclim_histsoc")
    # Expect sum=1+2+3=6 for all times
    assert float(out.isel(time=0, lat=0, lon=0)) == 6.0

def test_miroc_sum_first_three_layers_any_depth_dim_name():
    time = _mk_time()
    da = xr.DataArray(
        np.stack([np.full((3,),1.0), np.full((3,),2.0), np.full((3,),3.0), np.full((3,),50.0)], axis=0),
        dims=("layer", "time"),
        coords={"layer": [0,1,2,3], "time": time},
        name="soilmoist",
    )
    ds = xr.Dataset({"soilmoist": da})
    out = miroc_integ_land_to_1m(ds, "counterclim_histsoc")
    assert np.allclose(out.values, np.full((3,), 6.0))
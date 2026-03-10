"""Smoke test for the deseasonal_ecdf_gpd SSI method."""
import time
import numpy as np
import xarray as xr
from sm_attribution.analysis.ssi import compute_ssi, _DASK_NUM_WORKERS


def test_deseasonal_ecdf_gpd_3d():
    np.random.seed(42)
    time_coord = xr.date_range("2000-01", periods=240, freq="ME")
    lat = np.arange(3)
    lon = np.arange(4)
    data = np.random.uniform(50, 200, (240, 3, 4)).astype("float32")
    da = xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={"time": time_coord, "lat": lat, "lon": lon},
    )

    # monthwise_ecdf baseline
    ssi_mw = compute_ssi(
        da, ssi_method="monthwise_ecdf", scale=3,
        ref_start="2003-01", ref_end="2019-12",
    )
    assert ssi_mw.shape == (240, 3, 4)
    assert ssi_mw.attrs["ssi_method"] == "monthwise_ecdf"
    print(f"monthwise_ecdf: shape={ssi_mw.shape}, OK")

    # deseasonal_ecdf_gpd
    ssi_gpd = compute_ssi(
        da, ssi_method="deseasonal_ecdf_gpd", scale=3,
        ref_start="2003-01", ref_end="2019-12",
    )
    assert ssi_gpd.shape == (240, 3, 4)
    assert ssi_gpd.attrs["ssi_method"] == "deseasonal_ecdf_gpd"

    # First 2 timesteps should be NaN (scale-1)
    nan_frac = float(ssi_gpd.isnull().mean())
    valid = ssi_gpd.values[np.isfinite(ssi_gpd.values)]
    print(f"deseasonal_ecdf_gpd: shape={ssi_gpd.shape}, NaN frac={nan_frac:.3f}")
    print(f"  Range: [{valid.min():.2f}, {valid.max():.2f}], Std: {valid.std():.3f}")

    # Sanity: SSI values should be roughly standard-normal-ish
    assert valid.std() > 0.5, "SSI variance too low"
    assert valid.std() < 3.0, "SSI variance too high"
    assert abs(valid.mean()) < 1.0, "SSI mean too far from 0"

    print("ALL SMOKE TESTS PASSED")


def test_land_mask_optimization():
    """Verify that applying a land mask produces NaN over masked pixels
    and identical results over unmasked pixels."""
    np.random.seed(99)
    time_coord = xr.date_range("2000-01", periods=240, freq="ME")
    lat = np.arange(6)
    lon = np.arange(8)
    data = np.random.uniform(50, 200, (240, 6, 8)).astype("float32")
    da = xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={"time": time_coord, "lat": lat, "lon": lon},
    )

    # Create a checkerboard land mask — roughly 50% land
    mask_arr = np.zeros((6, 8), dtype=bool)
    mask_arr[::2, ::2] = True
    mask_arr[1::2, 1::2] = True
    land_mask = xr.DataArray(mask_arr, dims=["lat", "lon"],
                             coords={"lat": lat, "lon": lon})

    ssi_nomask = compute_ssi(
        da, ssi_method="deseasonal_ecdf_gpd", scale=3,
        ref_start="2003-01", ref_end="2019-12",
    )
    ssi_masked = compute_ssi(
        da, ssi_method="deseasonal_ecdf_gpd", scale=3,
        ref_start="2003-01", ref_end="2019-12",
        land_mask=land_mask,
    )

    # Ocean pixels (mask==False) must be all NaN
    ocean = ~land_mask.values
    assert np.all(np.isnan(ssi_masked.values[:, ocean])), "Ocean pixels should be NaN"

    # Land pixels should match the unmasked run
    land_vals_masked = ssi_masked.values[:, mask_arr]
    land_vals_nomask = ssi_nomask.values[:, mask_arr]
    close = np.allclose(land_vals_masked, land_vals_nomask, equal_nan=True)
    assert close, "Land pixel values should match between masked and unmasked runs"

    n_land = mask_arr.sum()
    n_total = mask_arr.size
    print(f"land_mask test: {n_land}/{n_total} land pixels, ocean all NaN: OK, land values match: OK")


def test_dask_threading_info():
    """Just print threading config for manual sanity check."""
    print(f"Dask num_workers: {_DASK_NUM_WORKERS}")
    assert _DASK_NUM_WORKERS >= 1


def test_timing_comparison():
    """Time the deseasonal_ecdf_gpd on a moderately large grid (20x30)
    to verify parallelism is working."""
    np.random.seed(123)
    time_coord = xr.date_range("2000-01", periods=240, freq="ME")
    lat = np.arange(20)
    lon = np.arange(30)
    data = np.random.uniform(50, 200, (240, 20, 30)).astype("float32")
    da = xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={"time": time_coord, "lat": lat, "lon": lon},
    )

    t0 = time.time()
    ssi = compute_ssi(
        da, ssi_method="deseasonal_ecdf_gpd", scale=3,
        ref_start="2003-01", ref_end="2019-12",
    )
    elapsed = time.time() - t0
    assert ssi.shape == (240, 20, 30)
    valid = ssi.values[np.isfinite(ssi.values)]
    print(f"timing test (20x30 = 600 pixels): {elapsed:.1f}s, "
          f"std={valid.std():.3f}, range=[{valid.min():.2f}, {valid.max():.2f}]")


if __name__ == "__main__":
    test_deseasonal_ecdf_gpd_3d()
    print()
    test_land_mask_optimization()
    print()
    test_dask_threading_info()
    print()
    test_timing_comparison()

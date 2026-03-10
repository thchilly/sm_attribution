from __future__ import annotations
import numpy as np
import xarray as xr
from sm_attribution.metrics.spatial_correlation import _coslat_weights, _ensure_2d_landmask

FEATURES_12 = (
    "duration", "magnitude", "intensity", "peak_intensity",
    "ddd", "ttm10", "tts15", "tte20", "drd",
    "n_events", "interarrival", "return_period",
)

# Backward-compatible alias (deprecated)
FEATURES_7 = ("duration", "magnitude", "intensity", "ddd", "tts15", "drd", "n_events")

def aggregate_ds_to_ar6(
    ds: xr.Dataset,
    *,
    land2d: xr.DataArray,
    ar6_mask3d: xr.DataArray,     # (region, lat, lon) bool
    region_abbrevs: list[str],
    region_names: list[str],
    features: tuple[str, ...] = FEATURES_12,
) -> xr.Dataset:
    land2d = _ensure_2d_landmask(land2d)
    w_lat = _coslat_weights(ds["lat"])
    w2d = w_lat.broadcast_like(ds[features[0]].transpose("lat","lon"))

    R = ar6_mask3d.sizes["region"]
    out = xr.Dataset(
        coords=dict(
            region=("region", np.array(region_abbrevs, dtype="U")),
            region_name=("region", np.array(region_names, dtype="U")),
            feature=("feature", np.array(list(features), dtype="U")),
        )
    )

    vals = np.full((len(features), R), np.nan, dtype="float32")
    npx  = np.zeros((len(features), R), dtype="int32")

    for fi, feat in enumerate(features):
        da = ds[feat].transpose("lat","lon")
        for r in range(R):
            rm = ar6_mask3d.isel(region=r).transpose("lat","lon")
            valid = land2d & rm & xr.ufuncs.isfinite(da)
            n = int(valid.sum().item())
            npx[fi, r] = n
            if n == 0:
                continue
            ww = w2d.where(valid)
            sww = float(ww.sum().item())
            if sww <= 0:
                continue
            vals[fi, r] = float((da.where(valid) * ww).sum().item() / sww)

    out["value"] = (("feature","region"), vals)
    out["n_pixels"] = (("feature","region"), npx)
    return out
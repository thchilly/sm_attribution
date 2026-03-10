# src/sm_attribution/metrics/spatial_correlation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import xarray as xr
import regionmask
from scipy.stats import rankdata, t as student_t


FEATURES_12 = (
    "duration", "magnitude", "intensity", "peak_intensity",
    "ddd", "ttm10", "tts15", "tte20", "drd",
    "n_events", "interarrival", "return_period",
)

# Backward-compatible alias (deprecated)
FEATURES_7 = ("duration", "magnitude", "intensity", "ddd", "tts15", "drd", "n_events")


# ----------------------------
# Core utilities
# ----------------------------

def _ensure_2d_landmask(land: xr.DataArray) -> xr.DataArray:
    """
    Ensure land mask is 2D (lat, lon) boolean; drop time if present.

    CRITICAL:
    Many landmask files encode ocean as NaN. If you do astype(bool) directly,
    bool(np.nan) is True -> oceans become land. So we must fill NaNs with 0 first.
    """
    m = land

    # Drop any time dimension/coord if present
    if "time" in m.dims:
        m = m.isel(time=0, drop=True)
    if "time" in m.coords:
        m = m.drop_vars("time")

    # Keep only lat/lon dims if extra singleton dims exist
    keep = [d for d in m.dims if d in ("lat", "lon")]
    if set(m.dims) != set(keep):
        m = m.transpose(*keep)

    # Fix NaN->False behavior
    m = m.fillna(0)

    # Robust boolean conversion:
    # - if already bool, keep
    # - if numeric, interpret nonzero as True
    if m.dtype != bool:
        m = (m != 0)

    m = m.astype(bool)

    # Ensure canonical ordering
    return m.transpose("lat", "lon")


def _load_first_latlon_var(ds: xr.Dataset) -> xr.DataArray:
    """
    Pick first variable that has lat/lon dims.
    Useful for landmask datasets with unknown variable name.
    """
    for k in ds.data_vars:
        da = ds[k]
        if {"lat", "lon"}.issubset(set(da.dims)):
            return da
    raise KeyError("No variable with (lat, lon) dims found in dataset.")


def _align_2d(
    a: xr.DataArray,
    b: xr.DataArray,
    land2d: xr.DataArray,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Align a and b with join='inner' on (lat, lon), and reindex landmask to aligned grid.
    """
    a2, b2 = xr.align(a, b, join="inner")

    land2 = _ensure_2d_landmask(land2d)
    land2 = land2.reindex(lat=a2["lat"], lon=a2["lon"], method=None, fill_value=False)

    # Ensure canonical ordering
    a2 = a2.transpose("lat", "lon")
    b2 = b2.transpose("lat", "lon")
    land2 = land2.transpose("lat", "lon")

    return a2, b2, land2


def _coslat_weights(lat: xr.DataArray) -> xr.DataArray:
    """
    cos(lat) weights for regular lat/lon grid (lat in degrees).
    Returns 1D weights over lat.
    """
    w = np.cos(np.deg2rad(lat.astype("float64")))
    w = xr.where(w < 0, 0.0, w)  # numerical safety
    return w


def _zero_fill_where_no_events(
    ds: xr.Dataset,
    land2d: xr.DataArray,
    features: Tuple[str, ...] = FEATURES_7,
    n_events_name: str = "n_events",
) -> xr.Dataset:
    """
    Zero-fill ONLY where n_events == 0 (inside land2d). Keep ocean as NaN.

    - n_events: land NaN -> 0
    - other features: set to 0 where (n_events == 0) on land
    """
    if n_events_name not in ds:
        raise KeyError(f"Dataset missing '{n_events_name}'.")

    out = ds.copy()

    land2d = _ensure_2d_landmask(land2d)

    # enforce lat/lon ordering where possible
    ne = out[n_events_name]
    if {"lat", "lon"}.issubset(ne.dims):
        ne = ne.transpose("lat", "lon")

    # Fill NaNs -> 0 everywhere, then restore ocean to NaN
    ne = ne.fillna(0.0)
    ne = ne.where(land2d)  # ocean becomes NaN
    out[n_events_name] = ne

    # True where land AND n_events==0
    is_zero_event_land = (ne == 0)

    for f in features:
        if f not in out:
            continue
        da = out[f]
        if {"lat", "lon"}.issubset(da.dims):
            da = da.transpose("lat", "lon")

        # Set feature=0 where no events on land
        da = da.where(~is_zero_event_land, other=0.0)

        # Ensure ocean remains NaN
        da = da.where(land2d)

        out[f] = da

    return out


# ----------------------------
# Weighted Spearman (global / region)
# ----------------------------

@dataclass(frozen=True)
class CorrResult:
    rho: float
    pval: float
    n_cells: int
    sum_weights: float


def _weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """
    Weighted Pearson correlation for 1D arrays (already filtered).
    """
    if x.size == 0:
        return np.nan
    sw = np.sum(w)
    if not np.isfinite(sw) or sw <= 0:
        return np.nan

    mx = np.sum(w * x) / sw
    my = np.sum(w * y) / sw

    xc = x - mx
    yc = y - my

    cov = np.sum(w * xc * yc) / sw
    vx = np.sum(w * xc * xc) / sw
    vy = np.sum(w * yc * yc) / sw

    if vx <= 0 or vy <= 0:
        return np.nan

    r = cov / np.sqrt(vx * vy)
    if np.isfinite(r):
        r = float(np.clip(r, -1.0, 1.0))
    return r


def _kish_neff(w: np.ndarray) -> float:
    """
    Kish effective sample size: (sum w)^2 / sum(w^2)
    """
    sw = np.sum(w)
    sw2 = np.sum(w * w)
    if sw2 <= 0:
        return np.nan
    return float((sw * sw) / sw2)


def weighted_spearman_from_maps(
    a: xr.DataArray,
    b: xr.DataArray,
    land2d: xr.DataArray,
    *,
    region_mask: Optional[xr.DataArray] = None,  # 2D bool or numeric-with-NaNs
    n_min_cells: int = 1000,
) -> CorrResult:
    """
    Weighted Spearman correlation between two 2D maps:
    - ranks computed with average ties
    - weighted Pearson on ranks using weights=cos(lat)
    - p-value via Kish n_eff + t approximation
    """
    a2, b2, land2 = _align_2d(a, b, land2d)

    # base valid mask: land AND finite in both
    valid = land2 & xr.ufuncs.isfinite(a2) & xr.ufuncs.isfinite(b2)

    # region restriction
    if region_mask is not None:
        rm = region_mask
        if "time" in rm.dims:
            rm = rm.isel(time=0, drop=True)

        # IMPORTANT: regionmask outputs NaN outside region -> must fillna(0) before bool conversion
        rm = rm.fillna(0)
        if rm.dtype != bool:
            rm = (rm != 0)
        rm = rm.astype(bool)

        rm = rm.reindex(lat=a2["lat"], lon=a2["lon"], method=None, fill_value=False)
        rm = rm.transpose("lat", "lon")

        valid = valid & rm

    n_cells0 = int(valid.sum().item())

    # weights (coslat) as 2D
    w_lat = _coslat_weights(a2["lat"])
    w2d = w_lat.broadcast_like(a2)

    sum_w0 = float(w2d.where(valid).sum().item()) if n_cells0 > 0 else 0.0

    if n_cells0 < n_min_cells:
        return CorrResult(rho=np.nan, pval=np.nan, n_cells=n_cells0, sum_weights=sum_w0)

    # Extract vectors
    x = a2.where(valid).values.astype("float64").ravel()
    y = b2.where(valid).values.astype("float64").ravel()
    w = w2d.where(valid).values.astype("float64").ravel()

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x = x[ok]
    y = y[ok]
    w = w[ok]

    n_cells = int(x.size)
    sum_w = float(np.sum(w))

    if n_cells < n_min_cells or sum_w <= 0:
        return CorrResult(rho=np.nan, pval=np.nan, n_cells=n_cells, sum_weights=sum_w)

    # Spearman ranks (average ties)
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")

    rho = _weighted_pearson(rx, ry, w)

    # p-value via Kish n_eff
    neff = _kish_neff(w)
    if (not np.isfinite(rho)) or (not np.isfinite(neff)) or neff <= 2:
        pval = np.nan
    else:
        denom = max(1e-15, 1.0 - rho * rho)
        tstat = rho * np.sqrt((neff - 2.0) / denom)
        df = neff - 2.0
        pval = float(2.0 * student_t.sf(np.abs(tstat), df))

    return CorrResult(rho=float(rho), pval=pval, n_cells=n_cells, sum_weights=float(sum_w))


# ----------------------------
# AR6 support
# ----------------------------

def get_ar6_land_regions() -> regionmask.Regions:
    return regionmask.defined_regions.ar6.land


def build_ar6_masks_on_grid(
    lon: xr.DataArray, lat: xr.DataArray
) -> Tuple[xr.DataArray, List[str], List[str]]:
    """
    Build AR6 land region 3D mask on (region, lat, lon) and return:
    - mask_bool: DataArray bool (region, lat, lon)
    - region_abbrevs
    - region_names

    IMPORTANT:
    regionmask produces NaN outside regions; NaN->bool becomes True if cast directly.
    We must fillna(0) before boolean conversion.
    """
    regions = get_ar6_land_regions()
    mask_3d = regions.mask_3D(lon, lat)  # dims: region, lat, lon

    # Fix NaNs before boolean conversion
    mask_3d = mask_3d.fillna(0)
    if mask_3d.dtype != bool:
        mask_bool = (mask_3d != 0).astype(bool)
    else:
        mask_bool = mask_3d.astype(bool)

    abbrevs = list(getattr(regions, "abbrevs", []))
    names = list(getattr(regions, "names", []))

    if not abbrevs or len(abbrevs) != mask_bool.sizes["region"]:
        abbrevs = [f"R{i:02d}" for i in range(mask_bool.sizes["region"])]
    if not names or len(names) != mask_bool.sizes["region"]:
        names = [f"Region {i:02d}" for i in range(mask_bool.sizes["region"])]

    return mask_bool, abbrevs, names
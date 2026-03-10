# src/sm_attribution/analysis/ssi.py

"""
Standardized Soil Moisture Index (SSI).

This module implements a nonparametric SSI following the empirical CDF (ECDF)
framework described by Farahmand & AghaKouchak (2015), where probabilities are
mapped to standard normal z-scores via the inverse normal transform.

The core implementation provided here is a month-wise ECDF approach:
  1) Accumulate soil moisture over an `n`-month window (rolling sum),
  2) Build an ECDF for each calendar month using a reference period,
  3) Transform ECDF probabilities to z-scores.

The module also provides an IO wrapper (`save_ssi`) that writes SSI products
using filename templates defined in the project data registry.
"""

from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import os
import sys
import threading
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr
from scipy.stats import expon, genpareto, norm

from sm_attribution.io.registry import Registry, default_registry
from sm_attribution.io.settings import get_settings

logger = logging.getLogger(__name__)

# On Linux, use 'forkserver' to avoid OpenBLAS / MKL deadlocks that can
# occur when the default 'fork' method is used with scipy/numpy threads.
# macOS already defaults to 'spawn' (Python 3.8+).  This MUST be called
# before any multiprocessing.Pool is created.
if sys.platform == "linux":
    try:
        mp.set_start_method("forkserver")
    except RuntimeError:
        pass  # already set by user or another library

_SETTINGS = get_settings()
_DEFAULT_SSI_SCALE = int(_SETTINGS.ssi.get("scale_months", 3))

# Defaults for deseasonal_ecdf_gpd method (overridable via kwargs)
_DEFAULT_TAIL_QUANTILE = float(_SETTINGS.ssi.get("hybrid_tail_quantile", 0.10))
_DEFAULT_MIN_TAIL_SIZE = int(_SETTINGS.ssi.get("hybrid_min_tail_size", 20))
_DEFAULT_LOC = str(_SETTINGS.ssi.get("hybrid_loc", "median"))
_DEFAULT_SCALE_METHOD = str(_SETTINGS.ssi.get("hybrid_scale_method", "iqr"))

# Supported SSI computation methods. The default preserves the canonical
# month-wise ECDF definition used throughout the project.
ALLOWED_SSI_METHODS = ("monthwise_ecdf", "deseasonal_ecdf_gpd")
DEFAULT_SSI_METHOD = "monthwise_ecdf"


def _format_from_template(tmpl: str, **kw) -> str:
    """
    Expand {paths.*} placeholders and then apply standard `.format()` keys.
    """
    paths = kw.get("paths")
    if isinstance(paths, dict):
        for k, v in paths.items():
            tmpl = tmpl.replace("{paths." + k + "}", v)
    return tmpl.format(**kw)


def compute_ssi_np(
    da: xr.DataArray,
    *,
    scale: int = _DEFAULT_SSI_SCALE,
    ref_start: str = "2003-01",
    ref_end: str = "2019-12",
    ref_data: xr.DataArray | None = None,
    time_name: str = "time",
) -> xr.DataArray:
    """
    Compute nonparametric SSI using a month-wise ECDF.

    Parameters
    ----------
    da
        Monthly soil moisture (dims include time, lat, lon).
    scale
        Accumulation window in months (rolling sum). Default taken from
        project settings (configs/settings.yml).
    ref_start, ref_end
        Reference period (inclusive) used to construct the ECDF.
    ref_data
        Optional reference DataArray used to construct the ECDF. If None,
        the ECDF is built from `da` itself over the reference period.
        If provided, `ref_data` is accumulated with the same `scale` and
        sliced to the same reference window.
    time_name
        Name of the time dimension.

    Returns
    -------
    xr.DataArray
        SSI z-scores with the same dimensions as the accumulated series.
        The first (scale-1) time steps are NaN by design due to rolling sum.
    """
    # Rolling accumulation
    y = da.rolling({time_name: scale}, min_periods=scale).sum()

    # Reference accumulation and slicing
    src_for_ref = ref_data if ref_data is not None else da
    y_ref_src = src_for_ref.rolling({time_name: scale}, min_periods=scale).sum()
    ref = y_ref_src.sel({time_name: slice(ref_start, ref_end)})

    # Output container
    ssi = xr.full_like(y, np.nan, dtype="float32")

    months_all = y[time_name].dt.month
    months_ref = ref[time_name].dt.month

    # Compute month-wise ECDF and transform to z-scores
    for m in range(1, 13):
        ref_m = ref.where(months_ref == m, drop=True)
        tgt_m = y.where(months_all == m, drop=True)

        if ref_m.sizes.get(time_name, 0) == 0 or tgt_m.sizes.get(time_name, 0) == 0:
            continue

        # Flatten spatial dims into columns: (time, ngrid)
        ref_vals = ref_m.data
        tgt_vals = tgt_m.data
        ref_flat = ref_vals.reshape(ref_vals.shape[0], -1)
        tgt_flat = tgt_vals.reshape(tgt_vals.shape[0], -1)

        valid_ref = np.isfinite(ref_flat)
        valid_tgt = np.isfinite(tgt_flat)

        # Sort reference values per grid. NaNs are carried through and ignored via nansum.
        ref_sorted = np.sort(np.where(valid_ref, ref_flat, np.nan), axis=0)

        # Valid sample count per grid
        n = valid_ref.sum(axis=0).astype(float)
        n[n == 0] = np.nan

        # Count of reference values <= each target value (per grid)
        leq = ref_sorted[None, ...] <= tgt_flat[:, None, :]
        idx = np.nansum(leq, axis=1)

        # Plotting-position probability (Farahmand & AghaKouchak-style)
        p = (idx - 0.44) / (n + 0.12)
        p[~valid_tgt] = np.nan
        p = np.clip(p, 1e-6, 1.0 - 1e-6)

        z = norm.ppf(p).astype("float32")
        z_da = xr.DataArray(
            z.reshape(tgt_m.shape),
            coords=tgt_m.coords,
            dims=tgt_m.dims,
            name="ssi",
        )
        ssi.loc[{time_name: tgt_m[time_name]}] = z_da

    ssi.name = "ssi"
    ssi.attrs.update(
        {
            "long_name": "Standardized Soil Moisture Index",
            "units": "-",
            "ssi_scale": scale,
            "ssi_ref_period": f"{ref_start}:{ref_end}",
            "ssi_method": "monthwise_ecdf",
            "method": (
                "Rolling sum over `scale` months; month-wise ECDF from reference period; "
                "inverse normal transform (norm.ppf)."
            ),
        }
    )
    return ssi


# =====================================================================
# deseasonal_ecdf_gpd: hybrid ECDF + GPD tails
# =====================================================================


def _ssi_deseasonal_ecdf_gpd_1d(
    tgt_accum: np.ndarray,
    ref_accum: np.ndarray,
    months_tgt: np.ndarray,
    months_ref: np.ndarray,
    tail_quantile: float = _DEFAULT_TAIL_QUANTILE,
    min_tail_size: int = _DEFAULT_MIN_TAIL_SIZE,
    loc: str = _DEFAULT_LOC,
    scale_method: str = _DEFAULT_SCALE_METHOD,
) -> np.ndarray:
    """
    Pure-NumPy 1D kernel for the hybrid deseasonalized ECDF + GPD SSI.

    Works on a single spatial pixel (1-D time vectors).

    Steps
    -----
    1. Compute monthly location (median/mean) and scale (IQR/std) from *ref_accum*.
    2. Standardize both ref and target: ``z = (x - loc_m) / scale_m``.
    3. Pool all calendar months into a single sorted reference.
    4. Empirical CDF for the core (between ``tail_quantile`` and ``1 - tail_quantile``).
    5. Fit GPD to lower/upper tail exceedances; fall back to Exponential if ξ < 0.
    6. Assemble hybrid CDF → ``norm.ppf(p)`` → SSI z-scores.

    Parameters
    ----------
    tgt_accum, ref_accum : 1-D float arrays
        Accumulated (rolling-sum) soil moisture for target and reference.
    months_tgt, months_ref : 1-D int arrays
        Calendar month (1-12) for each time step.
    tail_quantile : float
        Fraction of the distribution allocated to each tail (default 0.10).
    min_tail_size : int
        Minimum number of exceedances required to fit GPD; otherwise
        the tail reverts to the empirical CDF.
    loc : {"median", "mean"}
        Location estimator for deseasonalization.
    scale_method : {"iqr", "std"}
        Scale estimator for deseasonalization.

    Returns
    -------
    np.ndarray  (same length as *tgt_accum*)
        SSI z-scores (float32). All-NaN pixels propagate as all-NaN.
    """
    n_tgt = len(tgt_accum)
    out = np.full(n_tgt, np.nan, dtype=np.float32)

    # Quick bail-out for all-NaN pixels
    valid_ref = np.isfinite(ref_accum)
    valid_tgt = np.isfinite(tgt_accum)
    if valid_ref.sum() < 12 or valid_tgt.sum() == 0:
        return out

    # ------------------------------------------------------------------
    # 1. Monthly location & scale from reference
    # ------------------------------------------------------------------
    month_loc = np.zeros(13, dtype=np.float64)
    month_sc = np.ones(13, dtype=np.float64)

    for m in range(1, 13):
        mask = (months_ref == m) & valid_ref
        vals = ref_accum[mask]
        if len(vals) == 0:
            continue
        if loc == "mean":
            month_loc[m] = np.mean(vals)
        else:
            month_loc[m] = np.median(vals)

        if scale_method == "std":
            sc = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        else:
            q75, q25 = np.percentile(vals, [75, 25])
            sc = q75 - q25
        month_sc[m] = sc if sc > 1e-6 else 1.0

    # ------------------------------------------------------------------
    # 2. Standardize
    # ------------------------------------------------------------------
    z_ref = np.where(
        valid_ref,
        (ref_accum - month_loc[months_ref]) / month_sc[months_ref],
        np.nan,
    )
    z_tgt = np.where(
        valid_tgt,
        (tgt_accum - month_loc[months_tgt]) / month_sc[months_tgt],
        np.nan,
    )

    # Keep only finite reference values for the pooled distribution
    z_ref_clean = z_ref[np.isfinite(z_ref)]
    n_ref = len(z_ref_clean)
    if n_ref < 12:
        return out

    sorted_ref = np.sort(z_ref_clean)

    # ------------------------------------------------------------------
    # 3. Thresholds & exceedances
    # ------------------------------------------------------------------
    u_low = np.quantile(z_ref_clean, tail_quantile)
    u_high = np.quantile(z_ref_clean, 1.0 - tail_quantile)

    lower_exceed = u_low - z_ref_clean[z_ref_clean < u_low]
    upper_exceed = z_ref_clean[z_ref_clean > u_high] - u_high

    # ------------------------------------------------------------------
    # 4. Fit GPD / Exponential tails
    # ------------------------------------------------------------------
    use_lower = len(lower_exceed) >= min_tail_size
    use_upper = len(upper_exceed) >= min_tail_size

    xi_low = beta_low = 0.0
    xi_high = beta_high = 0.0

    if use_lower:
        try:
            xi_low, _, beta_low = genpareto.fit(lower_exceed, floc=0)
        except Exception:
            use_lower = False
        if use_lower and xi_low < 0:
            # Exponential fallback (infinite tail support)
            try:
                _, beta_low = expon.fit(lower_exceed, floc=0)
            except Exception:
                use_lower = False
            xi_low = 0.0

    if use_upper:
        try:
            xi_high, _, beta_high = genpareto.fit(upper_exceed, floc=0)
        except Exception:
            use_upper = False
        if use_upper and xi_high < 0:
            try:
                _, beta_high = expon.fit(upper_exceed, floc=0)
            except Exception:
                use_upper = False
            xi_high = 0.0

    # ------------------------------------------------------------------
    # 5. Empirical CDF (vectorized over all target timesteps)
    #    Farahmand & AghaKouchak plotting position: p = (r - 0.44)/(n + 0.12)
    # ------------------------------------------------------------------
    F_u_low = (np.searchsorted(sorted_ref, u_low, side="right") - 0.44) / (n_ref + 0.12)
    F_u_high = (np.searchsorted(sorted_ref, u_high, side="right") - 0.44) / (n_ref + 0.12)

    # ------------------------------------------------------------------
    # 6. Hybrid CDF evaluation → z-scores  (VECTORIZED)
    # ------------------------------------------------------------------
    finite_mask = np.isfinite(z_tgt)
    x_all = np.where(finite_mask, z_tgt, 0.0)  # safe placeholder for NaN slots

    # Core: empirical CDF for all points at once
    ranks = np.searchsorted(sorted_ref, x_all, side="right")
    p_all = (ranks - 0.44) / (n_ref + 0.12)

    # Lower tail override (batch genpareto.cdf call)
    if use_lower:
        lower_mask = finite_mask & (x_all < u_low)
        if lower_mask.any():
            y_exc = u_low - x_all[lower_mask]
            G = genpareto.cdf(y_exc, xi_low, loc=0, scale=beta_low)
            p_all[lower_mask] = F_u_low * (1.0 - G)

    # Upper tail override (batch genpareto.cdf call)
    if use_upper:
        upper_mask = finite_mask & (x_all > u_high)
        if upper_mask.any():
            y_exc = x_all[upper_mask] - u_high
            G = genpareto.cdf(y_exc, xi_high, loc=0, scale=beta_high)
            p_all[upper_mask] = F_u_high + (1.0 - F_u_high) * G

    np.clip(p_all, 1e-6, 1.0 - 1e-6, out=p_all)
    out = np.where(finite_mask, norm.ppf(p_all).astype(np.float32), np.nan).astype(np.float32)

    return out


# Default number of Dask workers for deseasonal_ecdf_gpd parallelism.
# Resolution order:
#   1. DASK_NUM_WORKERS environment variable  (explicit override)
#   2. settings.yml  dask.max_workers         (project-level cap)
#   3. os.cpu_count() - 2                     (safe auto-detect)
# The "-2" headroom keeps the OS/SSH responsive on heavy VMs.

def _default_workers() -> int:
    """Safe default: leave 2 cores for OS overhead, respect settings cap."""
    n_cpus = os.cpu_count() or 4
    safe = max(1, n_cpus - 2)
    cfg_max = _SETTINGS.dask.get("max_workers")
    if cfg_max is not None:
        safe = min(safe, int(cfg_max))
    return safe

_DASK_NUM_WORKERS = int(os.environ.get("DASK_NUM_WORKERS", _default_workers()))

# Spatial chunk sizes for dask parallelisation (configurable via settings.yml)
_CHUNK_LAT = int(_SETTINGS.dask.get("chunk_lat", 15))
_CHUNK_LON = int(_SETTINGS.dask.get("chunk_lon", 15))

# Whether to use dask.distributed LocalCluster (recommended for servers)
_USE_DISTRIBUTED = bool(_SETTINGS.dask.get("use_distributed", True))

# ---------------------------------------------------------------------------
# Dask distributed client management
# ---------------------------------------------------------------------------

_DASK_CLIENT = None  # module-level singleton
_DASK_CLIENT_LOCK = threading.Lock()  # guards lazy initialisation


def _get_dask_client():
    """Lazily create a dask.distributed Client with a LocalCluster.

    Returns None if dask.distributed is not installed or if the user
    has disabled it via ``settings.yml  dask.use_distributed: false``.

    Thread-safe: a ``threading.Lock`` ensures only one ``LocalCluster``
    is created even when multiple orchestration threads call this
    concurrently.
    """
    global _DASK_CLIENT
    # Fast path — no lock needed once the client already exists.
    if _DASK_CLIENT is not None:
        return _DASK_CLIENT

    if not _USE_DISTRIBUTED:
        return None

    with _DASK_CLIENT_LOCK:
        # Double-checked locking: another thread may have initialised
        # the client between the fast-path check and acquiring the lock.
        if _DASK_CLIENT is not None:
            return _DASK_CLIENT

        try:
            from dask.distributed import Client, LocalCluster
        except ImportError:
            logger.info("dask.distributed not installed — falling back to scheduler='processes'")
            return None

        n = _DASK_NUM_WORKERS
        mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3) if hasattr(os, "sysconf") else 0
        mem_limit = f"{max(1, int(mem_gb // n))}GiB" if mem_gb > 0 else "auto"

        cluster = LocalCluster(
            n_workers=n,
            threads_per_worker=1,   # 1 thread b/c genpareto.fit holds the GIL
            memory_limit=mem_limit,
        )
        _DASK_CLIENT = Client(cluster)
        logger.info(
            "Dask distributed cluster started: %d workers, memory_limit=%s, dashboard=%s",
            n, mem_limit, _DASK_CLIENT.dashboard_link,
        )
        atexit.register(_shutdown_dask_client)
        return _DASK_CLIENT


def _shutdown_dask_client():
    """Cleanly shut down the cluster on process exit."""
    global _DASK_CLIENT
    if _DASK_CLIENT is not None:
        try:
            _DASK_CLIENT.close()
        except Exception:
            pass
        _DASK_CLIENT = None


def compute_ssi_deseasonal_ecdf_gpd(
    da: xr.DataArray,
    *,
    scale: int = _DEFAULT_SSI_SCALE,
    ref_start: str = "2003-01",
    ref_end: str = "2019-12",
    ref_data: xr.DataArray | None = None,
    time_name: str = "time",
    tail_quantile: float = _DEFAULT_TAIL_QUANTILE,
    min_tail_size: int = _DEFAULT_MIN_TAIL_SIZE,
    loc: str = _DEFAULT_LOC,
    scale_method: str = _DEFAULT_SCALE_METHOD,
    land_mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """
    Compute SSI via deseasonalized ECDF with GPD tail completion.

    Rolling accumulation is performed at the xarray level (dask-friendly),
    then the 1-D kernel is applied per pixel via ``xr.apply_ufunc``.

    Performance
    -----------
    When the inputs are not already chunked, spatial dimensions are
    automatically chunked so that ``dask="parallelized"`` distributes
    work across ``_DASK_NUM_WORKERS`` threads.  ``land_mask`` (optional)
    sets ocean/ice pixels to NaN so the 1-D kernel early-exits instantly,
    avoiding unnecessary GPD fitting over water.

    Parameters
    ----------
    da : xr.DataArray
        Monthly soil moisture (time, lat, lon).
    scale : int
        Accumulation window in months.
    ref_start, ref_end : str
        Reference period (inclusive) for deseasonalization and ECDF.
    ref_data : xr.DataArray or None
        Optional external reference (e.g. pooled scenarios). If None, da
        itself is used as the reference.
    time_name : str
        Name of the time dimension.
    tail_quantile, min_tail_size, loc, scale_method
        Forwarded to ``_ssi_deseasonal_ecdf_gpd_1d``; see that function
        for documentation.
    land_mask : xr.DataArray or None
        Boolean (lat, lon) mask — True over land.  When provided, ocean
        pixels are set to NaN before processing, which significantly
        reduces computation time.

    Returns
    -------
    xr.DataArray
        SSI z-scores (float32), same shape as accumulated da.
    """
    # Ensure inputs are chunked so rolling sums stay lazy (dask graphs).
    _chunks = {"lat": _CHUNK_LAT, "lon": _CHUNK_LON}
    if not da.chunks:
        da = da.chunk(_chunks)
    if ref_data is not None and not ref_data.chunks:
        ref_data = ref_data.chunk(_chunks)
    src_for_ref = ref_data if ref_data is not None else da
    if land_mask is not None and not getattr(land_mask, "chunks", None):
        land_mask = land_mask.chunk(_chunks)

    # Rolling accumulation (xarray-native, stays lazy on chunked data)
    y = da.rolling({time_name: scale}, min_periods=scale).sum()
    y_ref = src_for_ref.rolling({time_name: scale}, min_periods=scale).sum()
    y_ref = y_ref.sel({time_name: slice(ref_start, ref_end)})

    # Apply land mask — sets ocean/ice pixels to NaN so 1-D kernel
    # hits the early bail-out path and skips GPD fitting.
    if land_mask is not None:
        y = y.where(land_mask)
        y_ref = y_ref.where(land_mask)

    # Rechunk so each spatial tile has a single chunk along time.
    # apply_ufunc with dask='parallelized' requires core dimensions
    # (time) to be unchunked — one chunk spanning the full axis.
    _full_chunks = {time_name: -1, "lat": _CHUNK_LAT, "lon": _CHUNK_LON}
    y = y.chunk(_full_chunks)
    y_ref = y_ref.chunk({time_name: -1, "lat": _CHUNK_LAT, "lon": _CHUNK_LON})

    # Persist the rechunked rolling sums into distributed worker memory.
    # This computes rolling+mask+rechunk in parallel across workers and
    # keeps results in their RAM.  The subsequent apply_ufunc graph then
    # references tiny chunk-keys (~100 bytes each) instead of embedding
    # the full arrays (~1.5 GiB) as graph literals.
    client = _get_dask_client()
    if client is not None:
        y, y_ref = client.persist([y, y_ref])

    # Rename the reference time dimension so apply_ufunc does not try to
    # align it with the (longer) target time dimension.
    ref_time_dim = f"{time_name}_ref"
    y_ref = y_ref.rename({time_name: ref_time_dim})

    # Calendar months as integer arrays (needed by the 1-D kernel)
    months_tgt = y[time_name].dt.month
    months_ref = y_ref[ref_time_dim].dt.month

    ssi = xr.apply_ufunc(
        _ssi_deseasonal_ecdf_gpd_1d,
        y,
        y_ref,
        months_tgt,
        months_ref,
        input_core_dims=[[time_name], [ref_time_dim], [time_name], [ref_time_dim]],
        output_core_dims=[[time_name]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
        kwargs=dict(
            tail_quantile=tail_quantile,
            min_tail_size=min_tail_size,
            loc=loc,
            scale_method=scale_method,
        ),
    )

    # Execute — use dask.distributed if available (persistent workers,
    # work-stealing across uneven land/ocean chunks, dashboard); fall
    # back to the basic multiprocessing scheduler otherwise.
    if client is not None:
        ssi = ssi.compute()  # routes through distributed client
    else:
        ssi = ssi.compute(scheduler="processes", num_workers=_DASK_NUM_WORKERS)

    # apply_ufunc places output core dims last; restore original dim order
    ssi = ssi.transpose(*da.dims)

    ssi.name = "ssi"
    ssi.attrs.update(
        {
            "long_name": "Standardized Soil Moisture Index",
            "units": "-",
            "ssi_scale": scale,
            "ssi_ref_period": f"{ref_start}:{ref_end}",
            "ssi_method": "deseasonal_ecdf_gpd",
            "method": (
                f"Rolling sum over {scale} months; deseasonalized (loc={loc}, "
                f"scale={scale_method}) from reference; pooled ECDF core with "
                f"GPD tails (tail_q={tail_quantile}, min_tail={min_tail_size}); "
                "exponential fallback if xi<0; inverse normal transform."
            ),
        }
    )
    return ssi


def compute_ssi(
    da: xr.DataArray,
    *,
    ssi_method: str = DEFAULT_SSI_METHOD,
    scale: int = _DEFAULT_SSI_SCALE,
    ref_start: str = "2003-01",
    ref_end: str = "2019-12",
    ref_data: xr.DataArray | None = None,
    time_name: str = "time",
    # --- deseasonal_ecdf_gpd kwargs (ignored by monthwise_ecdf) ---
    tail_quantile: float = _DEFAULT_TAIL_QUANTILE,
    min_tail_size: int = _DEFAULT_MIN_TAIL_SIZE,
    loc: str = _DEFAULT_LOC,
    scale_method: str = _DEFAULT_SCALE_METHOD,
    land_mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """
    Compute SSI using a selectable method.

    Parameters
    ----------
    ssi_method
        SSI computation method. Currently supported:
        - "monthwise_ecdf": month-wise ECDF (canonical SSI definition)
        - "deseasonal_ecdf_gpd": deseasonalized ECDF with GPD tail completion

    tail_quantile, min_tail_size, loc, scale_method
        Only used when ``ssi_method="deseasonal_ecdf_gpd"``.
        See ``compute_ssi_deseasonal_ecdf_gpd`` for details.

    Other parameters match ``compute_ssi_np``.

    Returns
    -------
    xr.DataArray
        SSI z-scores.
    """
    if ssi_method not in ALLOWED_SSI_METHODS:
        raise ValueError(f"ssi_method must be one of {ALLOWED_SSI_METHODS}, got '{ssi_method}'")

    if ssi_method == "monthwise_ecdf":
        return compute_ssi_np(
            da,
            scale=scale,
            ref_start=ref_start,
            ref_end=ref_end,
            ref_data=ref_data,
            time_name=time_name,
        )

    # deseasonal_ecdf_gpd
    return compute_ssi_deseasonal_ecdf_gpd(
        da,
        scale=scale,
        ref_start=ref_start,
        ref_end=ref_end,
        ref_data=ref_data,
        time_name=time_name,
        tail_quantile=tail_quantile,
        min_tail_size=min_tail_size,
        loc=loc,
        scale_method=scale_method,
        land_mask=land_mask,
    )


def save_ssi(
    da_1m: xr.DataArray,
    *,
    key: str,
    is_model: bool,
    reg: Registry | None = None,
    scale: int = _DEFAULT_SSI_SCALE,
    ref_start: str = "2003-01",
    ref_end: str = "2019-12",
    mode: str = "standalone",
    ref_data: xr.DataArray | None = None,
    pool_id: str | None = None,
    pool_scenarios: Sequence[str] | None = None,
    ssi_method: str = DEFAULT_SSI_METHOD,
    # --- deseasonal_ecdf_gpd kwargs ---
    tail_quantile: float = _DEFAULT_TAIL_QUANTILE,
    min_tail_size: int = _DEFAULT_MIN_TAIL_SIZE,
    loc: str = _DEFAULT_LOC,
    scale_method: str = _DEFAULT_SCALE_METHOD,
    land_mask: xr.DataArray | None = None,
) -> str:
    """
    Compute SSI from a preprocessed 0–1 m soil-moisture DataArray and write it to disk
    using filename templates from the data registry.

    Parameters
    ----------
    da_1m
        Monthly depth-integrated soil moisture (typically 0–1 m) on the common grid.
    key
        Registry identifier. For models, use "model_scenario" (e.g. "h08_obsclim_histsoc").
        For observations, use the observation key (e.g. "era5-land").
    is_model
        If True, writes using the model SSI template; otherwise uses the observed SSI template.
    reg
        Data registry instance. If None, uses `default_registry()`.
    scale, ref_start, ref_end
        Passed to the SSI computation and used in output naming.
    mode
        SSI output subfolder mode used in templates (e.g. "standalone", "pooled", "fixed").
        This parameter does not change the SSI math; it only informs output naming and metadata.
    ref_data
        Optional DataArray used as the reference sample for the ECDF (e.g. pooled across scenarios,
        or a fixed reference scenario). If None, reference is taken from `da_1m` itself.
    pool_id, pool_scenarios
        Optional metadata used for naming and provenance (primarily for model outputs).
    ssi_method
        SSI computation method (see `compute_ssi`).

    Returns
    -------
    str
        Path to the SSI NetCDF file on disk.
    """
    if reg is None:
        reg = default_registry()

    cfg = reg.cfg_dict
    paths = cfg.get("paths", {})
    tmpls = cfg.get("ssi_templates", {})

    refstart_yr = ref_start[:4]
    refend_yr = ref_end[:4]

    effective_pool_id = (pool_id or "standalone") if is_model else ""

    if is_model:
        try:
            model, scenario = key.split("_", 1)
        except ValueError:
            raise ValueError("For models, key must be 'model_scenario', e.g. 'h08_obsclim_histsoc'.")

        tmpl = tmpls.get("model")
        out_path = _format_from_template(
            tmpl,
            paths=paths,
            model=model,
            scenario=scenario,
            scale=scale,
            refstart=ref_start.replace("-", ""),
            refend=ref_end.replace("-", ""),
            mode=mode,
            refstart_yr=refstart_yr,
            refend_yr=refend_yr,
            pool_id=effective_pool_id,
            ssi_method=ssi_method,
        )
    else:
        tmpl = tmpls.get("observed")
        out_path = _format_from_template(
            tmpl,
            paths=paths,
            obskey=key,
            scale=scale,
            refstart=ref_start.replace("-", ""),
            refend=ref_end.replace("-", ""),
            ssi_method=ssi_method,
        )

    ssi = compute_ssi(
        da_1m,
        ssi_method=ssi_method,
        scale=scale,
        ref_start=ref_start,
        ref_end=ref_end,
        ref_data=ref_data,
        tail_quantile=tail_quantile,
        min_tail_size=min_tail_size,
        loc=loc,
        scale_method=scale_method,
        land_mask=land_mask,
    )

    # Provenance annotations
    ref_mode = mode if is_model else "standalone"
    if ref_data is None:
        ref_desc = "ECDF reference built from the target series (within the reference window)."
    else:
        pool_desc = ",".join(pool_scenarios) if pool_scenarios else "ALL_SCENARIOS"
        ref_desc = f"ECDF reference built from provided reference data (pool={pool_desc}) within the reference window."

    ssi.attrs.update(
        {
            "ssi_method": ssi_method,
            "ssi_mode": ref_mode,
            "ssi_ref_start": ref_start,
            "ssi_ref_end": ref_end,
            "ssi_ref_description": ref_desc,
            "ssi_pool_id": effective_pool_id if is_model else "",
            "ssi_pool_scenarios": ",".join(pool_scenarios) if (is_model and pool_scenarios) else "",
        }
    )

    enc = {
        "ssi": {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        }
    }
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    ds_out = ssi.to_dataset()
    ds_out.attrs.update(
        {
            "ssi_method": ssi_method,
            "ssi_mode": ref_mode,
            "ssi_scale": scale,
            "ssi_ref_start": ref_start,
            "ssi_ref_end": ref_end,
            "ssi_ref_description": ref_desc,
            "ssi_pool_id": effective_pool_id if is_model else "",
            "ssi_pool_scenarios": ",".join(pool_scenarios) if (is_model and pool_scenarios) else "",
            "source_key": key,
            "source_type": "model" if is_model else "observed",
        }
    )
    ds_out.to_netcdf(str(outp), encoding=enc)
    return str(outp)






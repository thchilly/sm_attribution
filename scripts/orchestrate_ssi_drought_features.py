#!/usr/bin/env python3
"""
Orchestrate (1) SSI generation and (2) drought-feature generation per obs dataset.

Runs two variants:
  - COMMON  : registry.common_period
  - MAXSPAN : registry.obs_periods[obs] clipped to registry.model_period
  - BOTH

Notes
-----
- Model SSI can be computed in "standalone" or "pooled" ECDF mode.
- Observations are handled via registry.obs_ssi_method:
    * "standard"               -> nonparametric SSI (AghaKouchak-style)
    * "grace_percentile_custom"-> percentile->z (SSI-like), optional rolling mean
    * "ssi_ready"              -> standardized anomaly (SSI-like), optional rolling mean


Correlations are computed elsewhere.
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence, Tuple, Optional

os.environ.setdefault("HDF5_LOG_LEVEL", "none")  # before xr/netCDF4 import

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore", message="Unable to decode time axis", category=xr.SerializationWarning)
warnings.filterwarnings(
    "ignore",
    message="The specified chunks separate the stored chunks",
    category=UserWarning,
)
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)
for _name in ("distributed", "distributed.worker", "distributed.nanny", "tornado", "bokeh"):
    logging.getLogger(_name).setLevel(logging.WARNING)

from sm_attribution.io._hdf5 import suppress_hdf5_diagnostics
suppress_hdf5_diagnostics()

from scipy.stats import norm

from sm_attribution.io.registry import default_registry, Registry
from sm_attribution.io.settings import get_settings
from sm_attribution.analysis.ensemble import (
    ensure_ssi_model,
    ensure_ssi_obs,
    ssi_model_path,
    ssi_obs_path,
)
from sm_attribution.analysis.ssi import ALLOWED_SSI_METHODS, DEFAULT_SSI_METHOD
from sm_attribution.analysis.drought_features import (
    ensure_drought_features_model,
    ensure_drought_features_obs,
)

SET = get_settings()
DEFAULT_SCALE = int(SET.ssi.get("scale_months", 3))
_CONCURRENT_MODELS = int(SET.dask.get("concurrent_models", 1))
DEFAULT_TAIL_QUANTILE = float(SET.ssi.get("hybrid_tail_quantile", 0.10))
DEFAULT_MIN_TAIL_SIZE = int(SET.ssi.get("hybrid_min_tail_size", 20))
DEFAULT_LOC = str(SET.ssi.get("hybrid_loc", "median"))
DEFAULT_SCALE_METHOD = str(SET.ssi.get("hybrid_scale_method", "iqr"))

DEFAULT_MODELS = [
    "h08",
    "hydropy",
    "jules-w2",
    "miroc-integ-land",
    "watergap2-2e",
    "web-dhm-sg",
    "lpjml5-7-10-fire",
]

DEFAULT_OBS = [
    "era5-land",
    "gleam-42a",
    "gleam-42b",
    "gldas-v20",
    "gldas-v21",
    "somo-ml",
    "grace-da-dm",
    "merra2-land",
    "gdo-ensmia",
    "gdo-smia",
]


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _remove_if_overwrite(path: str, overwrite: bool) -> None:
    if overwrite and os.path.exists(path):
        os.remove(path)


def _clip_period_to_model(obs_start: str, obs_end: str, reg: Registry) -> Tuple[str, str]:
    m0, m1 = reg.get_model_period()
    start = max(obs_start, m0)
    end = min(obs_end, m1)
    if start > end:
        raise ValueError(f"Empty clipped window: obs {obs_start}..{obs_end} vs model {m0}..{m1}")
    return start, end


def _pool_id_for_models(mode: str, pool_scenarios: Optional[Sequence[str]], fixed_ref_scenario: Optional[str] = None) -> str:
    if mode == "standalone":
        return "standalone"
    if mode == "fixed":
        return fixed_ref_scenario or "UNKNOWN_REF"
    # pooled
    if pool_scenarios is None:
        return "ALL_SCENARIOS"
    return "__".join(sorted(pool_scenarios))


# -----------------------
# SSI-like helpers (obs)
# -----------------------

def _load_time_lat_lon_var(ds: xr.Dataset, preferred: str) -> xr.DataArray:
    if preferred in ds:
        da = ds[preferred]
    else:
        candidates = [k for k in ds.data_vars if {"time", "lat", "lon"}.issubset(ds[k].dims)]
        if not candidates:
            raise KeyError("No (time, lat, lon) variable found in dataset.")
        da = ds[candidates[0]]

    # replace sentinel fill values
    fv = da.attrs.get("_FillValue", None)
    if fv is None and "missing_value" in da.attrs:
        fv = da.attrs["missing_value"]
    if fv is not None:
        da = da.where(da != fv)

    return da.transpose("time", "lat", "lon", missing_dims="ignore").astype("float32")


def ensure_ssi_obs_grace_like(
    obs_key: str,
    *,
    reg: Registry,
    scale: int,
    ref_start: str,
    ref_end: str,
    var: str = "rootzone_percentile",
    overwrite: bool = False,
    ssi_method: str = DEFAULT_SSI_METHOD,
) -> str:
    out_path = ssi_obs_path(obs_key, reg=reg, scale=scale, ref_start=ref_start, ref_end=ref_end, ssi_method=ssi_method)
    _remove_if_overwrite(out_path, overwrite)
    if os.path.exists(out_path):
        return out_path

    in_path = reg.get_obs_processed(obs_key)
    ds = xr.open_dataset(in_path)
    da_p = _load_time_lat_lon_var(ds, var)

    # percentile -> z-score
    p = da_p.clip(0.5, 99.5) / 100.0
    da_z = xr.apply_ufunc(norm.ppf, p, dask="allowed").astype("float32")
    da_z.name = "ssi"

    # "Option A": rolling mean to approximate multi-month standardized index
    if scale and scale > 1:
        da_z = da_z.rolling(time=scale, min_periods=scale).mean()

    da_z = da_z.sel(time=slice(ref_start, ref_end))
    da_z.attrs.update(
        {
            "ssi_mode": "ssi-like-from-percentile",
            "ssi_scale": scale,
            "ssi_ref_period": f"{ref_start}:{ref_end}",
            "source": str(in_path),
            "method": "percentile -> z via norm.ppf (clipped), then rolling mean (Option A)",
        }
    )

    _ensure_parent(out_path)
    ds_out = da_z.to_dataset(name="ssi")
    comp = dict(zlib=True, complevel=4, shuffle=True)
    ds_out.to_netcdf(out_path, encoding={k: comp for k in ds_out.data_vars})
    return out_path


def ensure_ssi_obs_from_anom_std(
    obs_key: str,
    *,
    reg: Registry,
    scale: int,
    ref_start: str,
    ref_end: str,
    var: str = "soilmoist_anom_std",
    overwrite: bool = False,
    ssi_method: str = DEFAULT_SSI_METHOD,
) -> str:
    out_path = ssi_obs_path(obs_key, reg=reg, scale=scale, ref_start=ref_start, ref_end=ref_end, ssi_method=ssi_method)
    _remove_if_overwrite(out_path, overwrite)
    if os.path.exists(out_path):
        return out_path

    in_path = reg.get_obs_processed(obs_key)
    ds = xr.open_dataset(in_path)
    da = _load_time_lat_lon_var(ds, var)

    # "Option A": rolling mean on standardized anomalies (keeps z-units; approximates persistence)
    if scale and scale > 1:
        da = da.rolling(time=scale, min_periods=scale).mean()

    da = da.sel(time=slice(ref_start, ref_end))
    da.name = "ssi"
    da.attrs.update(
        {
            "ssi_mode": "ssi-like-from-standardized-anomaly",
            "ssi_scale": scale,
            "ssi_ref_period": f"{ref_start}:{ref_end}",
            "source": str(in_path),
            "method": "standardized anomaly, then rolling mean (Option A)",
        }
    )

    _ensure_parent(out_path)
    ds_out = da.to_dataset(name="ssi")
    comp = dict(zlib=True, complevel=4, shuffle=True)
    ds_out.to_netcdf(out_path, encoding={k: comp for k in ds_out.data_vars})
    return out_path


def ensure_obs_ssi(
    obs_key: str,
    *,
    reg: Registry,
    scale: int,
    ref_start: str,
    ref_end: str,
    overwrite: bool,
    ssi_method: str = DEFAULT_SSI_METHOD,
    tail_quantile: float = DEFAULT_TAIL_QUANTILE,
    min_tail_size: int = DEFAULT_MIN_TAIL_SIZE,
    loc: str = DEFAULT_LOC,
    scale_method: str = DEFAULT_SCALE_METHOD,
) -> str:
    method = reg.get_obs_ssi_method(obs_key)

    if method == "standard":
        out_path = ssi_obs_path(
            obs_key, reg=reg, scale=scale, ref_start=ref_start, ref_end=ref_end,
            ssi_method=ssi_method,
        )
        _remove_if_overwrite(out_path, overwrite)
        return ensure_ssi_obs(
            obs_key, reg=reg, scale=scale, ref_start=ref_start, ref_end=ref_end,
            ssi_method=ssi_method,
            tail_quantile=tail_quantile,
            min_tail_size=min_tail_size,
            loc=loc,
            scale_method=scale_method,
        )

    if method == "grace_percentile_custom":
        return ensure_ssi_obs_grace_like(
            obs_key, reg=reg, scale=scale, ref_start=ref_start, ref_end=ref_end,
            overwrite=overwrite, ssi_method=ssi_method,
        )

    if method == "ssi_ready":
        return ensure_ssi_obs_from_anom_std(
            obs_key, reg=reg, scale=scale, ref_start=ref_start, ref_end=ref_end,
            overwrite=overwrite, ssi_method=ssi_method,
        )

    raise ValueError(f"Unknown obs_ssi_method '{method}' for {obs_key}")


# -----------------------
# Main orchestration
# -----------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Orchestrate SSI + drought features for common/maxspan periods."
    )
    p.add_argument("--period-mode", choices=["common", "maxspan", "both"], default="both")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--obs", nargs="+", default=DEFAULT_OBS)
    p.add_argument("--scenarios", nargs="+", default=None, help="Default: registry.scenarios()")

    p.add_argument("--scale", type=int, default=DEFAULT_SCALE)
    p.add_argument("--model-ssi-mode", choices=["standalone", "pooled", "fixed"], default="standalone")
    p.add_argument(
        "--pool-scenarios",
        nargs="+",
        default=None,
        help=(
            "Only used when --model-ssi-mode pooled. "
            "If set, pool ECDF reference from ONLY these scenarios."
        ),
    )
    p.add_argument(
        "--fixed-ref-scenario",
        default=None,
        help=(
            "Only used when --model-ssi-mode fixed. The single scenario used as "
            "ECDF reference for all model scenarios (e.g. obsclim_histsoc)."
        ),
    )

    p.add_argument("--overwrite", action="store_true")

    # SSI method
    p.add_argument(
        "--ssi-method",
        choices=list(ALLOWED_SSI_METHODS),
        default=DEFAULT_SSI_METHOD,
        help=f"SSI computation method. Default: {DEFAULT_SSI_METHOD}.",
    )
    p.add_argument("--tail-quantile", type=float, default=DEFAULT_TAIL_QUANTILE,
                    help="Tail quantile for GPD (deseasonal_ecdf_gpd only).")
    p.add_argument("--min-tail-size", type=int, default=DEFAULT_MIN_TAIL_SIZE,
                    help="Min exceedances for GPD (deseasonal_ecdf_gpd only).")
    p.add_argument("--loc", choices=["median", "mean"], default=DEFAULT_LOC,
                    help="Location estimator (deseasonal_ecdf_gpd only).")
    p.add_argument("--scale-method", choices=["iqr", "std"], default=DEFAULT_SCALE_METHOD,
                    help="Scale estimator (deseasonal_ecdf_gpd only).")

    # drought-feature params (defaults match drought_features.py)
    p.add_argument("--bridge-len", type=int, default=3)
    p.add_argument("--severity-threshold", type=float, default=-1.0)
    p.add_argument("--drought-threshold", type=float, default=0.0)
    p.add_argument("--tts15-threshold", type=float, default=-1.5)
    p.add_argument("--ttm10-threshold", type=float, default=-1.0)
    p.add_argument("--tte20-threshold", type=float, default=-2.0)

    return p.parse_args()


def run_window(
    *,
    reg: Registry,
    label: str,
    obs_key: str,
    models: Sequence[str],
    scenarios: Sequence[str],
    scale: int,
    ref_start: str,
    ref_end: str,
    model_ssi_mode: str,
    pool_scenarios: Optional[Sequence[str]],
    fixed_ref_scenario: Optional[str] = None,
    overwrite: bool,
    bridge_len: int,
    severity_threshold: float,
    drought_threshold: float,
    tts15_threshold: float,
    ttm10_threshold: float,
    tte20_threshold: float,
    ssi_method: str = DEFAULT_SSI_METHOD,
    tail_quantile: float = DEFAULT_TAIL_QUANTILE,
    min_tail_size: int = DEFAULT_MIN_TAIL_SIZE,
    loc: str = DEFAULT_LOC,
    scale_method: str = DEFAULT_SCALE_METHOD,
) -> None:
    pool_id = _pool_id_for_models(model_ssi_mode, pool_scenarios, fixed_ref_scenario)

    logger.info(
        f"\n=== {label} | {obs_key} | "
        f"ref {ref_start}..{ref_end} | scale={scale} | model_mode={model_ssi_mode} | pool_id={pool_id} ==="
    )

    # 1) obs SSI (or SSI-like)
    obs_ssi_path = ensure_obs_ssi(
        obs_key, reg=reg, scale=scale, ref_start=ref_start, ref_end=ref_end,
        overwrite=overwrite,
        ssi_method=ssi_method,
        tail_quantile=tail_quantile,
        min_tail_size=min_tail_size,
        loc=loc,
        scale_method=scale_method,
    )
    logger.info("Obs SSI ready: %s", obs_ssi_path)

    # -- helpers for per-model work (used by ThreadPoolExecutor) ----------
    def _ssi_for_one_model(m: str) -> list:
        """Compute SSI for all scenarios of *one* model.  Returns results."""
        results = []
        for s in scenarios:
            m_out_expected = ssi_model_path(
                m, s, reg=reg, scale=scale,
                ref_start=ref_start, ref_end=ref_end,
                mode=model_ssi_mode, pool_id=pool_id,
                ssi_method=ssi_method,
            )
            _remove_if_overwrite(m_out_expected, overwrite)
            out = ensure_ssi_model(
                m, s, reg=reg, scale=scale,
                ref_start=ref_start, ref_end=ref_end,
                mode=model_ssi_mode,
                pool_scenarios=pool_scenarios,
                fixed_ref_scenario=fixed_ref_scenario,
                ssi_method=ssi_method,
                tail_quantile=tail_quantile,
                min_tail_size=min_tail_size,
                loc=loc, scale_method=scale_method,
            )
            results.append((m, s, out))
        return results

    def _features_for_one_model(m: str) -> list:
        """Compute drought features for all scenarios of *one* model."""
        results = []
        for s in scenarios:
            feat_path = ensure_drought_features_model(
                m, s, reg=reg, ssi_mode=model_ssi_mode, pool_id=pool_id,
                scale=scale, ref_start=ref_start, ref_end=ref_end,
                feat_start=ref_start, feat_end=ref_end,
                bridge_len_months=bridge_len,
                severity_threshold=severity_threshold,
                drought_threshold=drought_threshold,
                tts15_threshold=tts15_threshold,
                ttm10_threshold=ttm10_threshold,
                tte20_threshold=tte20_threshold,
                overwrite=overwrite, ssi_method=ssi_method,
            )
            results.append((m, s, feat_path))
        return results

    n_concurrent = _CONCURRENT_MODELS
    n_models_total = len(models)

    # 2) model SSI (standalone/pooled) — up to n_concurrent models at once
    done_models = 0
    with ThreadPoolExecutor(max_workers=n_concurrent) as pool:
        futures = {pool.submit(_ssi_for_one_model, m): m for m in models}
        for fut in as_completed(futures):
            m = futures[fut]
            _ = fut.result()
            done_models += 1
            logger.info("Model SSI progress: %d/%d models done (%s)", done_models, n_models_total, m)

    # 3) drought features for models — up to n_concurrent models at once
    done_models = 0
    with ThreadPoolExecutor(max_workers=n_concurrent) as pool:
        futures = {pool.submit(_features_for_one_model, m): m for m in models}
        for fut in as_completed(futures):
            m = futures[fut]
            _ = fut.result()
            done_models += 1
            logger.info(
                "Model drought-features progress: %d/%d models done (%s)",
                done_models,
                n_models_total,
                m,
            )

    # 4) drought features for obs (direct)
    obs_feat_path = ensure_drought_features_obs(
        obs_key,
        reg=reg,
        scale=scale,
        ref_start=ref_start,
        ref_end=ref_end,
        feat_start=ref_start,
        feat_end=ref_end,
        bridge_len_months=bridge_len,
        severity_threshold=severity_threshold,
        drought_threshold=drought_threshold,
        tts15_threshold=tts15_threshold,
        ttm10_threshold=ttm10_threshold,
        tte20_threshold=tte20_threshold,
        overwrite=overwrite,
        ssi_method=ssi_method,
    )
    logger.info("Obs drought-features ready: %s", obs_feat_path)


def main() -> None:
    args = parse_args()
    reg = default_registry()
    scenarios = args.scenarios if args.scenarios is not None else list(reg.scenarios())

    for obs_key in args.obs:
        if args.period_mode in ("common", "both"):
            c0, c1 = reg.get_common_period()
            run_window(
                reg=reg,
                label="COMMON",
                obs_key=obs_key,
                models=args.models,
                scenarios=scenarios,
                scale=args.scale,
                ref_start=c0,
                ref_end=c1,
                model_ssi_mode=args.model_ssi_mode,
                pool_scenarios=args.pool_scenarios,
                fixed_ref_scenario=args.fixed_ref_scenario,
                overwrite=args.overwrite,
                bridge_len=args.bridge_len,
                severity_threshold=args.severity_threshold,
                drought_threshold=args.drought_threshold,
                tts15_threshold=args.tts15_threshold,
                ttm10_threshold=args.ttm10_threshold,
                tte20_threshold=args.tte20_threshold,
                ssi_method=args.ssi_method,
                tail_quantile=args.tail_quantile,
                min_tail_size=args.min_tail_size,
                loc=args.loc,
                scale_method=args.scale_method,
            )

        if args.period_mode in ("maxspan", "both"):
            o0, o1 = reg.get_obs_period(obs_key)
            r0, r1 = _clip_period_to_model(o0, o1, reg)
            run_window(
                reg=reg,
                label="MAXSPAN",
                obs_key=obs_key,
                models=args.models,
                scenarios=scenarios,
                scale=args.scale,
                ref_start=r0,
                ref_end=r1,
                model_ssi_mode=args.model_ssi_mode,
                pool_scenarios=args.pool_scenarios,
                fixed_ref_scenario=args.fixed_ref_scenario,
                overwrite=args.overwrite,
                bridge_len=args.bridge_len,
                severity_threshold=args.severity_threshold,
                drought_threshold=args.drought_threshold,
                tts15_threshold=args.tts15_threshold,
                ttm10_threshold=args.ttm10_threshold,
                tte20_threshold=args.tte20_threshold,
                ssi_method=args.ssi_method,
                tail_quantile=args.tail_quantile,
                min_tail_size=args.min_tail_size,
                loc=args.loc,
                scale_method=args.scale_method,
            )


if __name__ == "__main__":
    main()
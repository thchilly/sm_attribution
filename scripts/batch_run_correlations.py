#!/usr/bin/env python3
"""
Batch computation of model–obs correlation maps and multi-model means.

This script:

  * Loops over all ISIMIP models, all four scenarios, and a list of
    observational products.
  * For each (model, scenario, obs) triple, computes a gridpoint Pearson
    correlation map (SSI vs SSI or model SSI vs obs anomaly).
  * For each (scenario, obs) pair, computes a simple multi-model mean of
    the correlation fields across all models and writes it to a separate
    "multi-model" file.

Now supports:

  * ``--ssi-method`` (monthwise_ecdf | deseasonal_ecdf_gpd)
  * ``--mode`` includes ``fixed`` alongside ``standalone`` / ``pooled``
  * ``--period-mode`` derives SSI-reference and correlation windows from
    the registry (``common`` or ``maxspan``) instead of hard-coding dates
  * ``--fixed-ref-scenario`` for fixed-mode SSI

The per-pair outputs use the ``metrics.correlations_map`` template in
``configs/data_registry.yml``.

The multi-model outputs use ``metrics.correlations_multimodel_map``.
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from typing import Dict, List, Tuple

os.environ.setdefault("HDF5_LOG_LEVEL", "none")  # before xr/netCDF4 import

import xarray as xr

from sm_attribution.io.registry import default_registry
from sm_attribution.io.settings import get_settings
from sm_attribution.analysis.ensemble import ssi_model_path, ssi_obs_path
from sm_attribution.analysis.ssi import ALLOWED_SSI_METHODS, DEFAULT_SSI_METHOD
from sm_attribution.io.load_mask import load_isimip_landmask
from sm_attribution.metrics.correlation import pearson_map

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

# ---------------------------------------------------------------------------
# Configuration defaults (SSI scale) from settings
# ---------------------------------------------------------------------------

SET = get_settings()
SSI_SCALE_DEFAULT = int(SET.ssi.get("scale_months", 3))

MODELS = [
    "h08",
    "hydropy",
    "jules-w2",
    "miroc-integ-land",
    "watergap2-2e",
    "web-dhm-sg",
    "lpjml5-7-10-fire",
]

# Obs that participate in SSI–SSI correlations
OBS_SSI = [
    "era5-land",
    "gleam-42a",
    "gleam-42b",
    "gldas-v21",
    "somo-ml",
    "merra2-land",
    "grace-da-dm",
]

# Obs that are standardized anomalies (no SSI; use target='anomaly')
OBS_ANOM = [
    "gdo-ensmia",
    "gdo-smia",
]


def _format_from_template(tmpl: str, **kw) -> str:
    paths = kw.pop("paths", {})
    for k, v in paths.items():
        tmpl = tmpl.replace("{paths." + k + "}", v)
    return tmpl.format(**kw)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch model–obs correlations + multi-model means."
    )
    p.add_argument(
        "--mode",
        choices=["standalone", "pooled", "fixed"],
        default="standalone",
        help=(
            "SSI mode for models: 'standalone' uses per-scenario ECDF; "
            "'pooled' combines all scenarios; "
            "'fixed' uses a single reference scenario (see --fixed-ref-scenario)."
        ),
    )
    p.add_argument(
        "--fixed-ref-scenario",
        default=None,
        help=(
            "Only used when --mode fixed. The single scenario whose SM "
            "is used as the ECDF reference for all scenarios "
            "(e.g. obsclim_histsoc). Required when --mode=fixed."
        ),
    )
    p.add_argument(
        "--ssi-method",
        choices=list(ALLOWED_SSI_METHODS),
        default=DEFAULT_SSI_METHOD,
        help=(
            "SSI computation method. 'monthwise_ecdf' is the canonical month-wise "
            "ECDF; 'deseasonal_ecdf_gpd' uses deseasonalized ECDF with GPD tail "
            f"completion. Default: {DEFAULT_SSI_METHOD}."
        ),
    )
    p.add_argument(
        "--period-mode",
        choices=["common", "maxspan"],
        default=None,
        help=(
            "Derive SSI-reference and correlation windows from the registry. "
            "'common' uses common_period (2003-01..2019-12). "
            "'maxspan' uses the intersection of obs and model windows. "
            "Overrides --ref-start / --ref-end / --corr-start / --corr-end."
        ),
    )
    p.add_argument(
        "--ref-start",
        default="2003-01",
        help="SSI reference period start (YYYY-MM). Ignored when --period-mode is set.",
    )
    p.add_argument(
        "--ref-end",
        default="2019-12",
        help="SSI reference period end (YYYY-MM). Ignored when --period-mode is set.",
    )
    p.add_argument(
        "--corr-start",
        default=None,
        help=(
            "Correlation period start (YYYY-MM). "
            "Default: ref-start + SSI scale months. Ignored when --period-mode is set."
        ),
    )
    p.add_argument(
        "--corr-end",
        default=None,
        help=(
            "Correlation period end (YYYY-MM). "
            "Default: same as --ref-end. Ignored when --period-mode is set."
        ),
    )
    p.add_argument(
        "--target",
        choices=["ssi", "anomaly", "both"],
        default="ssi",
        help=(
            "What to compute: 'ssi' (SSI vs SSI), 'anomaly' (SSI vs standardized "
            "anomaly, GDO), or 'both'."
        ),
    )
    p.add_argument(
        "--n-min",
        type=int,
        default=60,
        help="Minimum number of valid time samples required to keep a correlation.",
    )
    p.add_argument(
        "--landmask-key",
        default="isimip_no_ant_nogreenland",
        help="Landmask key in the registry.",
    )
    p.add_argument(
        "--obs",
        nargs="+",
        default=None,
        help=(
            "Optional list of observation keys to process. "
            "If omitted, use the full built-in OBS_SSI / OBS_ANOM lists."
        ),
    )
    return p.parse_args()


def compute_pair(
    model: str,
    scenario: str,
    obs_key: str,
    *,
    target: str,
    mode: str,
    ssi_method: str,
    ssi_ref_start: str,
    ssi_ref_end: str,
    corr_start: str,
    corr_end: str,
    pool_id: str | None,
    n_min: int,
    land,
    reg,
) -> str:
    """Compute a single (model, scenario, obs) correlation map and write to disk.

    Returns
    -------
    str
        Path to the written NetCDF file.
    """
    # Model SSI
    m_kw: dict = dict(
        reg=reg,
        scale=SSI_SCALE_DEFAULT,
        ref_start=ssi_ref_start,
        ref_end=ssi_ref_end,
        mode=mode,
        ssi_method=ssi_method,
    )
    if pool_id is not None:
        m_kw["pool_id"] = pool_id

    m_path = ssi_model_path(model, scenario, **m_kw)
    da_m = xr.open_dataset(m_path)["ssi"]

    # Obs side
    if target == "ssi":
        o_path = ssi_obs_path(
            obs_key,
            reg=reg,
            scale=SSI_SCALE_DEFAULT,
            ref_start=ssi_ref_start,
            ref_end=ssi_ref_end,
            ssi_method=ssi_method,
        )
        da_o = xr.open_dataset(o_path)["ssi"]
    else:
        o_path = reg.get_obs_processed(obs_key)
        ds_o = xr.open_dataset(o_path)

        var = "soilmoist_anom_std"
        if var not in ds_o:
            var = next(
                k
                for k in ds_o.data_vars
                if {"time", "lat", "lon"}.issubset(ds_o[k].dims)
            )
        da_o = ds_o[var]

        fv = da_o.attrs.get("_FillValue", None)
        if fv is None and "missing_value" in da_o.attrs:
            fv = ds_o[var].attrs["missing_value"]
        if fv is not None:
            da_o = da_o.where(da_o != fv)

    # Correlation
    ds_corr = pearson_map(
        da_m,
        da_o,
        land,
        period_start=corr_start,
        period_end=corr_end,
        n_min=n_min,
        time_name="time",
    )

    # Output path
    tmpl = reg.cfg_dict["metrics"]["correlations_map"]
    corrstart_yr = corr_start[:4]
    corrend_yr = corr_end[:4]

    out_path = _format_from_template(
        tmpl,
        paths=reg.cfg_dict["paths"],
        mode=mode,
        target=target,
        obs_short=obs_key,
        model=model,
        scenario=scenario,
        corrstart_yr=corrstart_yr,
        corrend_yr=corrend_yr,
        ssi_method=ssi_method,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Provenance
    ds_corr.attrs.update(
        {
            "model": model,
            "scenario": scenario,
            "obs": obs_key,
            "ssi_mode": mode,
            "ssi_method": ssi_method,
            "target": target,
            "model_path": m_path,
            "obs_path": o_path,
            "corr_period": f"{corr_start}:{corr_end}",
            "ssi_scale_months": SSI_SCALE_DEFAULT,
            "ssi_ref_period": f"{ssi_ref_start}:{ssi_ref_end}",
        }
    )

    comp = dict(zlib=True, complevel=4, shuffle=True)
    enc = {k: comp for k in ds_corr.data_vars}
    ds_corr.to_netcdf(out_path, encoding=enc)
    return out_path


def compute_multimodel_mean(
    paths: List[str],
    *,
    scenario: str,
    obs_key: str,
    target: str,
    mode: str,
    ssi_method: str,
    corr_start: str,
    corr_end: str,
    ssi_ref_start: str,
    ssi_ref_end: str,
    reg,
) -> str:
    """Compute a simple multi-model mean correlation map from per-model files."""
    if not paths:
        raise ValueError("No correlation paths provided for multi-model mean.")

    ds_list = [xr.open_dataset(p) for p in paths]

    # Stack models along a new dimension and average
    r_stack = xr.concat([ds["r"] for ds in ds_list], dim="model")
    p_stack = xr.concat([ds["p"] for ds in ds_list], dim="model")
    n_stack = xr.concat([ds["n"] for ds in ds_list], dim="model")

    ds_mm = xr.Dataset(
        {
            "r": r_stack.mean("model", skipna=True),
            "p": p_stack.mean("model", skipna=True),
            "n": n_stack.mean("model", skipna=True).round().astype("int16"),
        },
        coords=ds_list[0].coords,
    )

    ds_mm["r"].attrs.update(
        {
            "long_name": "Multi-model mean Pearson correlation",
            "units": "1",
        }
    )
    ds_mm["p"].attrs.update(
        {
            "long_name": "Multi-model mean p-value",
            "units": "1",
        }
    )
    ds_mm["n"].attrs.update(
        {
            "long_name": "Mean number of valid paired samples across models",
            "units": "1",
        }
    )

    corrstart_yr = corr_start[:4]
    corrend_yr = corr_end[:4]

    tmpl_mm = reg.cfg_dict["metrics"]["correlations_multimodel_map"]
    out_path = _format_from_template(
        tmpl_mm,
        paths=reg.cfg_dict["paths"],
        mode=mode,
        target=target,
        obs_short=obs_key,
        scenario=scenario,
        corrstart_yr=corrstart_yr,
        corrend_yr=corrend_yr,
        ssi_method=ssi_method,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ds_mm.attrs.update(
        {
            "metric": "pearson_r_multi-model_mean",
            "scenario": scenario,
            "obs": obs_key,
            "target": target,
            "ssi_mode": mode,
            "ssi_method": ssi_method,
            "models": ", ".join(MODELS),
            "corr_period": f"{corr_start}:{corr_end}",
            "ssi_ref_period": f"{ssi_ref_start}:{ssi_ref_end}",
            "note": (
                "Multi-model mean of per-model correlation maps. "
                "Simple arithmetic mean across models for r, p, and n."
            ),
        }
    )

    comp = dict(zlib=True, complevel=4, shuffle=True)
    enc = {k: comp for k in ds_mm.data_vars}
    ds_mm.to_netcdf(out_path, encoding=enc)
    return out_path


def _run_for_obs_list(
    obs_list: List[str],
    *,
    target: str,
    scenarios: tuple[str, ...],
    mode: str,
    ssi_method: str,
    ssi_ref_start: str,
    ssi_ref_end: str,
    corr_start: str,
    corr_end: str,
    pool_id: str | None,
    n_min: int,
    land,
    reg,
) -> None:
    """Process all (model, scenario, obs) pairs for a list of obs datasets."""
    report_every = 2
    for obs_key in obs_list:
        for scen in scenarios:
            paths: List[str] = []
            total_models = len(MODELS)
            for idx, m in enumerate(MODELS, start=1):
                paths.append(
                    compute_pair(
                        model=m,
                        scenario=scen,
                        obs_key=obs_key,
                        target=target,
                        mode=mode,
                        ssi_method=ssi_method,
                        ssi_ref_start=ssi_ref_start,
                        ssi_ref_end=ssi_ref_end,
                        corr_start=corr_start,
                        corr_end=corr_end,
                        pool_id=pool_id,
                        n_min=n_min,
                        land=land,
                        reg=reg,
                    )
                )
                if (idx % report_every == 0) or (idx == total_models):
                    logger.info(
                        "Pair progress: obs=%s scen=%s target=%s %d/%d models",
                        obs_key,
                        scen,
                        target,
                        idx,
                        total_models,
                    )

            mm_path = compute_multimodel_mean(
                paths,
                scenario=scen,
                obs_key=obs_key,
                target=target,
                mode=mode,
                ssi_method=ssi_method,
                corr_start=corr_start,
                corr_end=corr_end,
                ssi_ref_start=ssi_ref_start,
                ssi_ref_end=ssi_ref_end,
                reg=reg,
            )
            logger.info(
                "Multi-model written: obs=%s scen=%s target=%s -> %s",
                obs_key,
                scen,
                target,
                mm_path,
            )


def main() -> None:
    args = parse_args()
    reg = default_registry()
    land = load_isimip_landmask(args.landmask_key)

    scenarios = reg.scenarios()
    mode = args.mode
    ssi_method = args.ssi_method

    # Resolve pool_id for fixed mode
    pool_id: str | None = None
    if mode == "fixed":
        if not args.fixed_ref_scenario:
            raise SystemExit("ERROR: --fixed-ref-scenario is required when --mode=fixed")
        pool_id = args.fixed_ref_scenario
    elif mode == "standalone":
        pool_id = "standalone"
    # pooled: leave pool_id=None → ssi_model_path uses its default

    do_ssi = args.target in ("ssi", "both")
    do_anom = args.target in ("anomaly", "both")

    # Resolve which obs to use for each category
    if args.obs is None:
        obs_ssi_list = OBS_SSI
        obs_anom_list = OBS_ANOM
    else:
        obs_ssi_list = [k for k in args.obs if k in OBS_SSI]
        obs_anom_list = [k for k in args.obs if k in OBS_ANOM]

    # -----------------------------------------------------------------
    # Resolve reference / correlation periods
    # -----------------------------------------------------------------
    period_mode = args.period_mode

    if period_mode is not None:
        # Registry-driven period: same window for all obs when "common",
        # per-obs window when "maxspan".
        all_obs: list[tuple[str, str]] = []
        if do_ssi:
            all_obs += [(k, "ssi") for k in obs_ssi_list]
        if do_anom:
            all_obs += [(k, "anomaly") for k in obs_anom_list]

        for obs_key, target in all_obs:
            ssi_ref_start, ssi_ref_end = reg.resolve_ref_period(obs_key, period_mode)
            corr_start = reg.corr_start_from_ref(ssi_ref_start, SSI_SCALE_DEFAULT)
            corr_end = ssi_ref_end

            logger.info(
                f"\n--- {obs_key} ({target}): "
                f"ref={ssi_ref_start}..{ssi_ref_end}, "
                f"corr={corr_start}..{corr_end} ---"
            )

            _run_for_obs_list(
                [obs_key],
                target=target,
                scenarios=scenarios,
                mode=mode,
                ssi_method=ssi_method,
                ssi_ref_start=ssi_ref_start,
                ssi_ref_end=ssi_ref_end,
                corr_start=corr_start,
                corr_end=corr_end,
                pool_id=pool_id,
                n_min=args.n_min,
                land=land,
                reg=reg,
            )
    else:
        # Legacy: explicit windows from CLI
        ssi_ref_start = args.ref_start
        ssi_ref_end = args.ref_end

        if args.corr_start is not None:
            corr_start = args.corr_start
        else:
            corr_start = reg.corr_start_from_ref(ssi_ref_start, SSI_SCALE_DEFAULT)

        corr_end = args.corr_end if args.corr_end is not None else ssi_ref_end

        logger.info(
            f"\nref={ssi_ref_start}..{ssi_ref_end}, "
            f"corr={corr_start}..{corr_end}, "
            f"mode={mode}, ssi_method={ssi_method}"
        )

        if do_ssi:
            _run_for_obs_list(
                obs_ssi_list,
                target="ssi",
                scenarios=scenarios,
                mode=mode,
                ssi_method=ssi_method,
                ssi_ref_start=ssi_ref_start,
                ssi_ref_end=ssi_ref_end,
                corr_start=corr_start,
                corr_end=corr_end,
                pool_id=pool_id,
                n_min=args.n_min,
                land=land,
                reg=reg,
            )

        if do_anom:
            _run_for_obs_list(
                obs_anom_list,
                target="anomaly",
                scenarios=scenarios,
                mode=mode,
                ssi_method=ssi_method,
                ssi_ref_start=ssi_ref_start,
                ssi_ref_end=ssi_ref_end,
                corr_start=corr_start,
                corr_end=corr_end,
                pool_id=pool_id,
                n_min=args.n_min,
                land=land,
                reg=reg,
            )


if __name__ == "__main__":
    main()

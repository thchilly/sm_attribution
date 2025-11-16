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

The per-pair outputs use the `metrics.correlations_map` template in
`configs/data_registry.yml`.

The multi-model outputs use `metrics.correlations_multimodel_map`.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Dict, Tuple, List

import xarray as xr

from sm_attribution.io.registry import default_registry
from sm_attribution.io.settings import get_settings
from sm_attribution.analysis.ensemble import ssi_model_path, ssi_obs_path
from sm_attribution.io.load_mask import load_isimip_landmask
from sm_attribution.metrics.correlation import pearson_map

# ---------------------------------------------------------------------------
# Configuration defaults (SSI scale + correlation period) from settings
# ---------------------------------------------------------------------------

SET = get_settings()
SSI_SCALE_DEFAULT = int(SET.ssi.get("scale_months", 3))

SSI_REF_START_DEFAULT = "2003-01"
SSI_REF_END_DEFAULT = "2019-12"

CORR_START_DEFAULT = "2004-01"
CORR_END_DEFAULT = "2019-12"

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
        choices=["standalone", "pooled"],
        default="standalone",
        help="SSI mode for models (standalone vs pooled ECDF).",
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
    return p.parse_args()


def compute_pair(
    model: str,
    scenario: str,
    obs_key: str,
    *,
    target: str,
    mode: str,
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
    m_path = ssi_model_path(
        model,
        scenario,
        reg=reg,
        scale=SSI_SCALE_DEFAULT,
        ref_start=SSI_REF_START_DEFAULT,
        ref_end=SSI_REF_END_DEFAULT,
        mode=mode,
    )
    da_m = xr.open_dataset(m_path)["ssi"]

    # Obs side
    if target == "ssi":
        o_path = ssi_obs_path(
            obs_key,
            reg=reg,
            scale=SSI_SCALE_DEFAULT,
            ref_start=SSI_REF_START_DEFAULT,
            ref_end=SSI_REF_END_DEFAULT,
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
        period_start=CORR_START_DEFAULT,
        period_end=CORR_END_DEFAULT,
        n_min=n_min,
        time_name="time",
    )

    # Output path
    tmpl = reg.cfg_dict["metrics"]["correlations_map"]
    corrstart_yr = CORR_START_DEFAULT[:4]
    corrend_yr = CORR_END_DEFAULT[:4]

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
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Provenance
    ds_corr.attrs.update(
        {
            "model": model,
            "scenario": scenario,
            "obs": obs_key,
            "ssi_mode": mode,
            "target": target,
            "model_path": m_path,
            "obs_path": o_path,
            "corr_period": f"{CORR_START_DEFAULT}:{CORR_END_DEFAULT}",
            "ssi_scale_months": SSI_SCALE_DEFAULT,
            "ssi_ref_period": f"{SSI_REF_START_DEFAULT}:{SSI_REF_END_DEFAULT}",
        }
    )

    comp = dict(zlib=True, complevel=4, shuffle=True)
    enc = {k: comp for k in ds_corr.data_vars}
    ds_corr.to_netcdf(out_path, encoding=enc)
    print(f"Wrote pair: {model:15s} {scenario:18s} {obs_key:12s} ({target}) -> {out_path}")
    return out_path


def compute_multimodel_mean(
    paths: List[str],
    *,
    scenario: str,
    obs_key: str,
    target: str,
    mode: str,
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

    corrstart_yr = CORR_START_DEFAULT[:4]
    corrend_yr = CORR_END_DEFAULT[:4]

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
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ds_mm.attrs.update(
        {
            "metric": "pearson_r_multi-model_mean",
            "scenario": scenario,
            "obs": obs_key,
            "target": target,
            "ssi_mode": mode,
            "models": ", ".join(MODELS),
            "corr_period": f"{CORR_START_DEFAULT}:{CORR_END_DEFAULT}",
            "note": (
                "Multi-model mean of per-model correlation maps. "
                "Simple arithmetic mean across models for r, p, and n."
            ),
        }
    )

    comp = dict(zlib=True, complevel=4, shuffle=True)
    enc = {k: comp for k in ds_mm.data_vars}
    ds_mm.to_netcdf(out_path, encoding=enc)
    print(f"Wrote multi-model: {scenario:18s} {obs_key:12s} ({target}) -> {out_path}")
    return out_path


def main() -> None:
    args = parse_args()
    reg = default_registry()
    land = load_isimip_landmask(args.landmask_key)

    scenarios = reg.scenarios()
    corrstart_yr = CORR_START_DEFAULT[:4]
    corrend_yr = CORR_END_DEFAULT[:4]

    do_ssi = args.target in ("ssi", "both")
    do_anom = args.target in ("anomaly", "both")

    if do_ssi:
        for obs_key in OBS_SSI:
            for scen in scenarios:
                paths = [
                    compute_pair(
                        model=m,
                        scenario=scen,
                        obs_key=obs_key,
                        target="ssi",
                        mode=args.mode,
                        n_min=args.n_min,
                        land=land,
                        reg=reg,
                    )
                    for m in MODELS
                ]
                compute_multimodel_mean(
                    paths,
                    scenario=scen,
                    obs_key=obs_key,
                    target="ssi",
                    mode=args.mode,
                    reg=reg,
                )

    if do_anom:
        for obs_key in OBS_ANOM:
            for scen in scenarios:
                paths = [
                    compute_pair(
                        model=m,
                        scenario=scen,
                        obs_key=obs_key,
                        target="anomaly",
                        mode=args.mode,
                        n_min=args.n_min,
                        land=land,
                        reg=reg,
                    )
                    for m in MODELS
                ]
                compute_multimodel_mean(
                    paths,
                    scenario=scen,
                    obs_key=obs_key,
                    target="anomaly",
                    mode=args.mode,
                    reg=reg,
                )


if __name__ == "__main__":
    main()
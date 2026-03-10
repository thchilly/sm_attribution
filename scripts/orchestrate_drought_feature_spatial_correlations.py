#!/usr/bin/env python3
"""
Compute weighted Spearman spatial correlations between observed and model drought-feature maps.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Sequence, Tuple, Optional, List

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

from sm_attribution.io.registry import default_registry, Registry
from sm_attribution.io.settings import get_settings
from sm_attribution.analysis.drought_features import expected_drought_features_path
from sm_attribution.analysis.ssi import ALLOWED_SSI_METHODS, DEFAULT_SSI_METHOD
from sm_attribution.metrics.spatial_correlation import (
    FEATURES_12,
    _ensure_2d_landmask,
    _load_first_latlon_var,
    _zero_fill_where_no_events,
    weighted_spearman_from_maps,
    build_ar6_masks_on_grid,
)

SET = get_settings()
DEFAULT_SCALE = int(SET.ssi.get("scale_months", 3))

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
    if pool_scenarios is None:
        return "ALL_SCENARIOS"
    return "__".join(sorted(pool_scenarios))


def _format_from_template(tmpl: str, reg: Registry, **kw) -> str:
    paths = reg.cfg_dict.get("paths", {}) or {}
    for k, v in paths.items():
        tmpl = tmpl.replace("{paths." + k + "}", v)
    return tmpl.format(**kw)


def expected_master_corr_path(
    *,
    reg: Registry,
    obskey: str,
    period_mode: str,
    model_mode: str,
    pool_id: str,
    scale: int,
    ref_start: str,
    ref_end: str,
    ssi_method: str = DEFAULT_SSI_METHOD,
) -> str:
    metrics = reg.cfg_dict.get("metrics", {}) or {}
    if "spatial_correlation_master" not in metrics:
        raise KeyError(
            "Missing metrics.spatial_correlation_master in data_registry.yml. "
            "Add a template like:\n"
            '  spatial_correlation_master: "{paths.root}/metrics/spatial_correlation/{obskey}/{period_mode}_{model_mode}_ref{refstart}-{refend}_s{scale}.nc"'
        )
    tmpl = metrics["spatial_correlation_master"]
    out = _format_from_template(
        tmpl,
        reg,
        obskey=obskey,
        period_mode=period_mode,
        model_mode=model_mode,
        pool_id=pool_id,
        scale=scale,
        refstart=ref_start.replace("-", ""),
        refend=ref_end.replace("-", ""),
        ssi_method=ssi_method,
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    return out


def _load_isimip_landmask(reg: Registry) -> xr.DataArray:
    anc = reg.cfg_dict.get("ancils", {}) or {}
    landmask_cfg = (anc.get("landmask", {}) or {}).get("isimip_no_ant_nogreenland", None)
    if landmask_cfg is None:
        raise KeyError("Registry missing ancils.landmask.isimip_no_ant_nogreenland")

    if not isinstance(landmask_cfg, str):
        raise TypeError("ancils.landmask.isimip_no_ant_nogreenland must be a string template/path")

    p = _format_from_template(landmask_cfg, reg)
    if not os.path.exists(p):
        raise FileNotFoundError(f"ISIMIP landmask not found: {p}")

    ds = xr.open_dataset(p)
    da = _load_first_latlon_var(ds)

    # IMPORTANT: this now does fillna(0) before bool conversion
    da = _ensure_2d_landmask(da)

    da.name = "landmask"
    return da


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weighted Spearman spatial correlations of drought features (Global + AR6).")

    p.add_argument("--period-mode", choices=["common", "maxspan"], required=True)
    p.add_argument("--obs", nargs="+", default=DEFAULT_OBS)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--scenarios", nargs="+", default=None, help="Default: registry.scenarios()")

    p.add_argument("--scale", type=int, default=DEFAULT_SCALE)

    p.add_argument("--model-ssi-mode", choices=["standalone", "pooled", "fixed"], default="standalone")
    p.add_argument("--pool-scenarios", nargs="+", default=None)
    p.add_argument(
        "--fixed-ref-scenario",
        default=None,
        help="Only used when --model-ssi-mode fixed. The single scenario used as ECDF reference.",
    )

    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--n-min-global", type=int, default=1000)
    p.add_argument("--n-min-ar6", type=int, default=100)

    p.add_argument(
        "--ssi-method",
        choices=list(ALLOWED_SSI_METHODS),
        default=DEFAULT_SSI_METHOD,
        help=f"SSI computation method. Default: {DEFAULT_SSI_METHOD}.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    reg = default_registry()
    scenarios = args.scenarios if args.scenarios is not None else list(reg.scenarios())
    pool_id = _pool_id_for_models(args.model_ssi_mode, args.pool_scenarios, getattr(args, 'fixed_ref_scenario', None))

    # fixed landmask used everywhere
    land2d = _load_isimip_landmask(reg)

    for obskey in args.obs:
        # deterministic period selection (mirrors generation logic)
        if args.period_mode == "common":
            ref_start, ref_end = reg.get_common_period()
        else:
            o0, o1 = reg.get_obs_period(obskey)
            ref_start, ref_end = _clip_period_to_model(o0, o1, reg)

        feat_start, feat_end = ref_start, ref_end

        out_path = expected_master_corr_path(
            reg=reg,
            obskey=obskey,
            period_mode=args.period_mode,
            model_mode=args.model_ssi_mode,
            pool_id=pool_id,
            scale=args.scale,
            ref_start=ref_start,
            ref_end=ref_end,
            ssi_method=args.ssi_method,
        )

        if args.overwrite and os.path.exists(out_path):
            os.remove(out_path)

        if os.path.exists(out_path):
            logger.info("Exists, skipping: %s", out_path)
            continue

        # obs drought-features file (deterministic)
        obs_feat_path = expected_drought_features_path(
            yaml_path=reg.yaml_path,
            is_model=False,
            key=obskey,
            mode="standalone",
            pool_id="standalone",
            scale=args.scale,
            ref_start=ref_start,
            ref_end=ref_end,
            feat_start=feat_start,
            feat_end=feat_end,
            ssi_method=args.ssi_method,
        )
        if not os.path.exists(obs_feat_path):
            raise FileNotFoundError(f"Observed drought-features file missing: {obs_feat_path}")

        ds_obs_raw = xr.open_dataset(obs_feat_path)

        # align landmask to obs grid for correct no-event filling
        land_on_obs = land2d.reindex(lat=ds_obs_raw["lat"], lon=ds_obs_raw["lon"], method=None, fill_value=False)
        ds_obs = _zero_fill_where_no_events(ds_obs_raw, land_on_obs)

        # AR6 masks on OBS grid
        ar6_mask3d, ar6_abbrevs, ar6_names = build_ar6_masks_on_grid(ds_obs["lon"], ds_obs["lat"])

        keep_idx: List[int] = []
        keep_abbr: List[str] = []
        keep_name: List[str] = []
        for i, ab in enumerate(ar6_abbrevs):
            if ab in ("ANT", "GIC"):
                continue
            keep_idx.append(i)
            keep_abbr.append(ab)
            keep_name.append(ar6_names[i])

        region_labels = ["Global"] + keep_abbr
        region_names = ["Global"] + keep_name

        n_models = len(args.models)
        n_scen = len(scenarios)
        n_feat = len(FEATURES_12)
        n_reg = len(region_labels)

        rho = np.full((n_models, n_scen, n_feat, n_reg), np.nan, dtype="float32")
        pval = np.full((n_models, n_scen, n_feat, n_reg), np.nan, dtype="float32")
        n_cells = np.zeros((n_models, n_scen, n_feat, n_reg), dtype="int32")
        sum_w = np.zeros((n_models, n_scen, n_feat, n_reg), dtype="float64")

        logger.info(
            f"\n=== {obskey} | {args.period_mode} | ref {ref_start}..{ref_end} | "
            f"model_mode={args.model_ssi_mode} pool_id={pool_id} ==="
        )
        logger.info("Obs features: %s", obs_feat_path)

        total_pairs = n_models * n_scen
        done_pairs = 0
        report_every = 4
        for mi, model in enumerate(args.models):
            for si, scen in enumerate(scenarios):
                key = f"{model}_{scen}"
                mod_feat_path = expected_drought_features_path(
                    yaml_path=reg.yaml_path,
                    is_model=True,
                    key=key,
                    mode=args.model_ssi_mode,
                    pool_id=pool_id,
                    scale=args.scale,
                    ref_start=ref_start,
                    ref_end=ref_end,
                    feat_start=feat_start,
                    feat_end=feat_end,
                    ssi_method=args.ssi_method,
                )
                if not os.path.exists(mod_feat_path):
                    raise FileNotFoundError(f"Model drought-features file missing: {mod_feat_path}")

                ds_mod_raw = xr.open_dataset(mod_feat_path)
                land_on_mod = land2d.reindex(lat=ds_mod_raw["lat"], lon=ds_mod_raw["lon"], method=None, fill_value=False)
                ds_mod = _zero_fill_where_no_events(ds_mod_raw, land_on_mod)

                for fi, feat in enumerate(FEATURES_12):
                    obs_map = ds_obs[feat].transpose("lat", "lon")
                    mod_map = ds_mod[feat].transpose("lat", "lon")

                    # Global
                    res_g = weighted_spearman_from_maps(
                        obs_map,
                        mod_map,
                        land2d,
                        region_mask=None,
                        n_min_cells=args.n_min_global,
                    )
                    rho[mi, si, fi, 0] = np.float32(res_g.rho)
                    pval[mi, si, fi, 0] = np.float32(res_g.pval)
                    n_cells[mi, si, fi, 0] = int(res_g.n_cells)
                    sum_w[mi, si, fi, 0] = float(res_g.sum_weights)

                    # AR6 regions
                    for rj, ridx in enumerate(keep_idx, start=1):
                        rm = ar6_mask3d.isel(region=ridx).transpose("lat", "lon")
                        # rm is already bool and NaN-safe from build_ar6_masks_on_grid
                        res_r = weighted_spearman_from_maps(
                            obs_map,
                            mod_map,
                            land2d,
                            region_mask=rm,
                            n_min_cells=args.n_min_ar6,
                        )
                        rho[mi, si, fi, rj] = np.float32(res_r.rho)
                        pval[mi, si, fi, rj] = np.float32(res_r.pval)
                        n_cells[mi, si, fi, rj] = int(res_r.n_cells)
                        sum_w[mi, si, fi, rj] = float(res_r.sum_weights)

                done_pairs += 1
                if (done_pairs % report_every == 0) or (done_pairs == total_pairs):
                    logger.info(
                        "Spatial corr progress: %d/%d pairs (latest=%s/%s)",
                        done_pairs,
                        total_pairs,
                        model,
                        scen,
                    )

        ds_out = xr.Dataset(
            data_vars=dict(
                rho=(("model", "scenario", "feature", "region"), rho),
                pval=(("model", "scenario", "feature", "region"), pval),
                n_cells=(("model", "scenario", "feature", "region"), n_cells),
                sum_weights=(("model", "scenario", "feature", "region"), sum_w),
            ),
            coords=dict(
                model=np.array(list(args.models), dtype="U"),
                scenario=np.array(list(scenarios), dtype="U"),
                feature=np.array(list(FEATURES_12), dtype="U"),
                region=np.array(region_labels, dtype="U"),
                region_name=("region", np.array(region_names, dtype="U")),
            ),
            attrs=dict(
                obs_key=obskey,
                period_mode=args.period_mode,
                ref_start=ref_start,
                ref_end=ref_end,
                feat_start=feat_start,
                feat_end=feat_end,
                scale_months=int(args.scale),
                model_ssi_mode=args.model_ssi_mode,
                pool_id=str(pool_id),
                pool_scenarios=json.dumps(args.pool_scenarios) if args.pool_scenarios is not None else "null",
                ssi_method=args.ssi_method,
                n_min_global=int(args.n_min_global),
                n_min_ar6=int(args.n_min_ar6),
                landmask="isimip_no_ant_nogreenland",
                weighting="cos(lat)",
                correlation="weighted_spearman (Pearson on average ranks)",
                p_value="two-sided via t-approx using Kish n_eff",
                zero_fill_rule="zero-fill ONLY where n_events==0 (inside landmask) prior to finiteness checks",
                notes="AR6 regions from regionmask. ANT and GIC excluded.",
                inputs=json.dumps(
                    dict(
                        obs_features=obs_feat_path,
                        model_features_template="drought_features_templates.model",
                        models=list(args.models),
                        scenarios=list(scenarios),
                        features=list(FEATURES_12),
                    )
                ),
            ),
        )

        ds_out["rho"].attrs.update({"long_name": "Weighted Spearman rho (coslat weighted Pearson on ranks)", "units": "1"})
        ds_out["pval"].attrs.update({"long_name": "Two-sided p-value using Kish n_eff", "units": "1"})
        ds_out["n_cells"].attrs.update({"long_name": "Number of valid grid cells used", "units": "1"})
        ds_out["sum_weights"].attrs.update({"long_name": "Sum of weights over valid sample (effective area proxy)", "units": "1"})

        comp_f = {"zlib": True, "complevel": 4, "shuffle": True, "dtype": "float32"}
        comp_i = {"zlib": True, "complevel": 4, "shuffle": True, "dtype": "int32"}
        comp_w = {"zlib": True, "complevel": 4, "shuffle": True, "dtype": "float64"}

        enc = {"rho": comp_f, "pval": comp_f, "n_cells": comp_i, "sum_weights": comp_w}

        ds_out.to_netcdf(out_path, encoding=enc)
        logger.info("Wrote: %s", out_path)


if __name__ == "__main__":
    main()
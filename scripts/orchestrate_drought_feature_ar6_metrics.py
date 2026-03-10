#!/usr/bin/env python3
"""
Aggregate drought-feature maps to AR6 regions (no Global) and compute
regional-vector metrics vs observations for each model/scenario.

Metrics computed per (model, scenario, feature):
  - spearman_rank: Spearman correlation across AR6 regions
  - pearson_z: Pearson correlation after z-scoring across regions within each product
  - rmse_iqr: RMSE after robust normalization (median/IQR) across regions within each product

Writes a master NetCDF per obs dataset.
"""

from __future__ import annotations

import argparse
import logging
import json
import os
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

os.environ.setdefault("HDF5_LOG_LEVEL", "none")  # before xr/netCDF4 import

import numpy as np
import xarray as xr
from scipy.stats import spearmanr

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


# -----------------------------
# Period / path helpers
# -----------------------------
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


def expected_ar6_metrics_master_path(
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
    if "droughtfeat_ar6_metrics_master" not in metrics:
        raise KeyError(
            "Missing metrics.droughtfeat_ar6_metrics_master in data_registry.yml."
        )
    tmpl = metrics["droughtfeat_ar6_metrics_master"]
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
    da = _ensure_2d_landmask(da)
    da.name = "landmask"
    return da


# -----------------------------
# Regional aggregation
# -----------------------------
def _coslat_weights(lat: xr.DataArray) -> xr.DataArray:
    w = np.cos(np.deg2rad(lat.astype("float64")))
    w = xr.where(w < 0, 0.0, w)
    return w


def _area_weighted_mean_over_mask(
    da: xr.DataArray, land2d: xr.DataArray, region_mask2d: xr.DataArray
) -> Tuple[float, int]:
    """
    Weighted mean over valid pixels:
      valid = land2d & region_mask2d & finite(da)
      weights = cos(lat)
    Returns (mean, n_pixels).
    """
    da2 = da.transpose("lat", "lon")
    land2d = _ensure_2d_landmask(land2d).reindex(lat=da2["lat"], lon=da2["lon"], fill_value=False)
    rm = region_mask2d.transpose("lat", "lon").reindex(lat=da2["lat"], lon=da2["lon"], fill_value=False)

    valid = land2d & rm & xr.ufuncs.isfinite(da2)
    n = int(valid.sum().item())
    if n == 0:
        return np.nan, 0

    w_lat = _coslat_weights(da2["lat"])
    w2d = w_lat.broadcast_like(da2)
    ww = w2d.where(valid)
    sw = float(ww.sum().item())
    if sw <= 0 or not np.isfinite(sw):
        return np.nan, n

    val = float((da2.where(valid) * ww).sum().item() / sw)
    return val, n


def aggregate_features_to_ar6(
    ds: xr.Dataset,
    *,
    land2d: xr.DataArray,
    ar6_mask3d: xr.DataArray,     # (region, lat, lon) bool
    ar6_abbrevs: List[str],
    ar6_names: List[str],
    keep_idx: List[int],
    features: Sequence[str] = FEATURES_12,
) -> xr.Dataset:
    """
    Returns ds_reg with:
      value(feature, region)
      n_pixels(feature, region)
    region labels are abbrevs for kept regions (no ANT/GIC, no Global).
    """
    reg_labels = [ar6_abbrevs[i] for i in keep_idx]
    reg_names = [ar6_names[i] for i in keep_idx]

    vals = np.full((len(features), len(keep_idx)), np.nan, dtype="float32")
    npx = np.zeros((len(features), len(keep_idx)), dtype="int32")

    for fi, feat in enumerate(features):
        if feat not in ds:
            continue
        da = ds[feat]
        for rj, ridx in enumerate(keep_idx):
            rm = ar6_mask3d.isel(region=ridx).transpose("lat", "lon")
            v, n = _area_weighted_mean_over_mask(da, land2d, rm)
            vals[fi, rj] = np.float32(v) if np.isfinite(v) else np.float32(np.nan)
            npx[fi, rj] = int(n)

    out = xr.Dataset(
        data_vars=dict(
            value=(("feature", "region"), vals),
            n_pixels=(("feature", "region"), npx),
        ),
        coords=dict(
            feature=np.array(list(features), dtype="U"),
            region=np.array(reg_labels, dtype="U"),
            region_name=("region", np.array(reg_names, dtype="U")),
        ),
    )
    return out


# -----------------------------
# Metrics on regional vectors
# -----------------------------
def _finite_pair(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok], int(ok.sum())


def metric_spearman(x: np.ndarray, y: np.ndarray, *, n_min: int) -> float:
    x2, y2, n = _finite_pair(x, y)
    if n < n_min:
        return np.nan
    return float(spearmanr(x2, y2).correlation)


def metric_pearson_z(x: np.ndarray, y: np.ndarray, *, n_min: int) -> float:
    x2, y2, n = _finite_pair(x, y)
    if n < n_min:
        return np.nan
    sx = np.nanstd(x2)
    sy = np.nanstd(y2)
    if sx == 0 or sy == 0:
        return np.nan
    zx = (x2 - np.nanmean(x2)) / sx
    zy = (y2 - np.nanmean(y2)) / sy
    return float(np.corrcoef(zx, zy)[0, 1])


def metric_rmse_iqr(x: np.ndarray, y: np.ndarray, *, n_min: int) -> float:
    x2, y2, n = _finite_pair(x, y)
    if n < n_min:
        return np.nan

    def rnorm(v: np.ndarray) -> np.ndarray:
        med = np.nanmedian(v)
        q75, q25 = np.nanpercentile(v, [75, 25])
        iqr = q75 - q25
        if not np.isfinite(iqr) or iqr == 0:
            return np.full_like(v, np.nan, dtype=float)
        return (v - med) / iqr

    xn = rnorm(x2)
    yn = rnorm(y2)
    ok = np.isfinite(xn) & np.isfinite(yn)
    if int(ok.sum()) < n_min:
        return np.nan
    return float(np.sqrt(np.nanmean((xn[ok] - yn[ok]) ** 2)))


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AR6-aggregated drought-feature regional metrics vs obs.")
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
    p.add_argument("--n-min-regions", type=int, default=20, help="Min AR6 regions needed for metric.")

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

    land2d = _load_isimip_landmask(reg)

    for obskey in args.obs:
        ref_start, ref_end = reg.resolve_ref_period(obskey, args.period_mode)
        feat_start, feat_end = ref_start, ref_end

        out_path = expected_ar6_metrics_master_path(
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

        # --- Load OBS drought features (deterministic)
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

        # AR6 masks on OBS grid (so region defs are consistent with obs)
        ar6_mask3d, ar6_abbrevs, ar6_names = build_ar6_masks_on_grid(ds_obs["lon"], ds_obs["lat"])

        keep_idx: List[int] = []
        for i, ab in enumerate(ar6_abbrevs):
            if ab in ("ANT", "GIC"):
                continue
            keep_idx.append(i)

        # obs -> regional vectors
        obs_reg = aggregate_features_to_ar6(
            ds_obs,
            land2d=land_on_obs,
            ar6_mask3d=ar6_mask3d,
            ar6_abbrevs=ar6_abbrevs,
            ar6_names=ar6_names,
            keep_idx=keep_idx,
            features=FEATURES_12,
        )

        n_models = len(args.models)
        n_scen = len(scenarios)
        n_feat = len(FEATURES_12)

        spearman = np.full((n_models, n_scen, n_feat), np.nan, dtype="float32")
        pearson_z = np.full((n_models, n_scen, n_feat), np.nan, dtype="float32")
        rmse_iqr = np.full((n_models, n_scen, n_feat), np.nan, dtype="float32")
        n_regions = np.zeros((n_models, n_scen, n_feat), dtype="int32")

        logger.info(
            f"\n=== {obskey} | {args.period_mode} | ref {ref_start}..{ref_end} | "
            f"model_mode={args.model_ssi_mode} pool_id={pool_id} ==="
        )
        logger.info("Obs features: %s", obs_feat_path)
        logger.info("Output: %s", out_path)

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

                # aggregate model on *obs* AR6 masks is not possible if grids differ
                # so: rebuild AR6 masks on MODEL grid too, using same region definitions
                ar6_mod_mask3d, ar6_mod_ab, ar6_mod_names = build_ar6_masks_on_grid(ds_mod["lon"], ds_mod["lat"])

                mod_reg = aggregate_features_to_ar6(
                    ds_mod,
                    land2d=land_on_mod,
                    ar6_mask3d=ar6_mod_mask3d,
                    ar6_abbrevs=ar6_mod_ab,
                    ar6_names=ar6_mod_names,
                    keep_idx=keep_idx,   # same indices correspond to same AR6 regions
                    features=FEATURES_12,
                )

                for fi, feat in enumerate(FEATURES_12):
                    x = mod_reg["value"].sel(feature=feat).values.astype("float64")
                    y = obs_reg["value"].sel(feature=feat).values.astype("float64")
                    _, _, n = _finite_pair(x, y)
                    n_regions[mi, si, fi] = int(n)

                    spearman[mi, si, fi] = np.float32(metric_spearman(x, y, n_min=args.n_min_regions))
                    pearson_z[mi, si, fi] = np.float32(metric_pearson_z(x, y, n_min=args.n_min_regions))
                    rmse_iqr[mi, si, fi] = np.float32(metric_rmse_iqr(x, y, n_min=args.n_min_regions))

                done_pairs += 1
                if (done_pairs % report_every == 0) or (done_pairs == total_pairs):
                    logger.info(
                        "AR6 metrics progress: %d/%d pairs (latest=%s/%s)",
                        done_pairs,
                        total_pairs,
                        model,
                        scen,
                    )

        ds_out = xr.Dataset(
            data_vars=dict(
                spearman_rank=(("model", "scenario", "feature"), spearman),
                pearson_z=(("model", "scenario", "feature"), pearson_z),
                rmse_iqr=(("model", "scenario", "feature"), rmse_iqr),
                n_regions=(("model", "scenario", "feature"), n_regions),
            ),
            coords=dict(
                model=np.array(list(args.models), dtype="U"),
                scenario=np.array(list(scenarios), dtype="U"),
                feature=np.array(list(FEATURES_12), dtype="U"),
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
                aggregation="AR6 land regions (excluding ANT,GIC); area-weighted mean using cos(lat)",
                metrics=(
                    "spearman_rank: Spearman correlation across regions; "
                    "pearson_z: Pearson correlation after z-scoring across regions within product; "
                    "rmse_iqr: RMSE after robust normalization (median/IQR) across regions within product"
                ),
                n_min_regions=int(args.n_min_regions),
                landmask="isimip_no_ant_nogreenland",
                zero_fill_rule="zero-fill ONLY where n_events==0 (inside landmask) prior to aggregation",
            ),
        )

        ds_out["spearman_rank"].attrs.update({"long_name": "Spearman rank correlation across AR6 regions", "units": "1"})
        ds_out["pearson_z"].attrs.update({"long_name": "Pearson correlation of z-scored regional vectors", "units": "1"})
        ds_out["rmse_iqr"].attrs.update({"long_name": "RMSE of robust-normalized (median/IQR) regional vectors", "units": "1"})
        ds_out["n_regions"].attrs.update({"long_name": "Number of valid AR6 regions used", "units": "1"})

        comp_f = {"zlib": True, "complevel": 4, "shuffle": True, "dtype": "float32"}
        comp_i = {"zlib": True, "complevel": 4, "shuffle": True, "dtype": "int32"}
        enc = {"spearman_rank": comp_f, "pearson_z": comp_f, "rmse_iqr": comp_f, "n_regions": comp_i}

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        ds_out.to_netcdf(out_path, encoding=enc)
        logger.info("Wrote: %s", out_path)


if __name__ == "__main__":
    main()
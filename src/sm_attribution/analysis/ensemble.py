# src/sm_attribution/analysis/ensemble.py

"""
Utilities for standardized SSI paths and on-demand SSI generation
for model and observational datasets, plus helpers for correlation
map paths (single-model and multi-model).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple, Sequence

import xarray as xr
import yaml

from ..io.registry import Registry, default_registry
from ..io.settings import get_settings
from ..io.load_mask import load_isimip_landmask
from .ssi import (
    save_ssi,
    DEFAULT_SSI_METHOD,
    _DEFAULT_TAIL_QUANTILE,
    _DEFAULT_MIN_TAIL_SIZE,
    _DEFAULT_LOC,
    _DEFAULT_SCALE_METHOD,
    _CHUNK_LAT,
    _CHUNK_LON,
)  # uses ssi_templates in data_registry.yml

_chunks = {"lat": _CHUNK_LAT, "lon": _CHUNK_LON}
logger = logging.getLogger(__name__)

# Land mask key used to skip ocean/ice pixels in deseasonal_ecdf_gpd.
_GPD_LAND_MASK_KEY = "isimip_no_ant_nogreenland"


# ---------------------------------------------------------------------------
# Global settings (SSI scale)
# ---------------------------------------------------------------------------

_SETTINGS = get_settings()
_DEFAULT_SSI_SCALE = int(_SETTINGS.ssi.get("scale_months", 3))


# ---------------------------------------------------------------------------
# Internal helpers for SSI filename templates
# ---------------------------------------------------------------------------


def _load_registry_yaml(yaml_path: str) -> dict:
    """Load the raw YAML dict from a registry file path."""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def _format_from_template(tmpl: str, **kw) -> str:
    """
    Expand {paths.*} placeholders and then regular `.format()` keys.
    Shared between SSI templates and correlation templates.
    """
    paths = kw.get("paths")
    if isinstance(paths, dict):
        for k, v in paths.items():
            tmpl = tmpl.replace("{paths." + k + "}", v)
    return tmpl.format(**kw)


def expected_ssi_path(
    *,
    yaml_path: str,
    is_model: bool,
    key: str,
    scale: int,
    ref_start: str,
    ref_end: str,
    mode: str,
    pool_id: str | None = None,
    ssi_method: str = DEFAULT_SSI_METHOD,
) -> str:
    """
    Resolve the expected SSI output path from data_registry.yml for either
    a model or observational dataset, without computing SSI.

    Parameters
    ----------
    yaml_path : str
        Path to configs/data_registry.yml.
    is_model : bool
        True for model SSI (model+scenario), False for observations.
    key : str
        If is_model=True, this is "model_scenario".
        If is_model=False, this is the observation key (e.g. "era5-land").
    scale : int
        SSI temporal scale in months.
    ref_start, ref_end : str
        Reference period for the ECDF (YYYY-MM strings).
    mode : {"standalone", "pooled", "fixed"}
        Reference mode for models. Ignored for observations.

    Returns
    -------
    str
        Fully-resolved SSI file path.
    """
    cfg = _load_registry_yaml(yaml_path)
    paths = cfg.get("paths", {})
    tmpls = cfg.get("ssi_templates", {})

    refstart_yr = ref_start[:4]
    refend_yr = ref_end[:4]

    if is_model:
        model, scenario = key.split("_", 1)
        tmpl = tmpls["model"]
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
            pool_id=(pool_id or "standalone"),
            ssi_method=ssi_method,
        )
    else:
        tmpl = tmpls["observed"]
        out_path = _format_from_template(
            tmpl,
            paths=paths,
            obskey=key,
            scale=scale,
            refstart=ref_start.replace("-", ""),
            refend=ref_end.replace("-", ""),
            ssi_method=ssi_method,
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    return out_path


# ---------------------------------------------------------------------------
# Public SSI path helpers
# ---------------------------------------------------------------------------


def ssi_model_path(
    model: str,
    scenario: str,
    *,
    reg: Registry | None = None,
    scale: int = _DEFAULT_SSI_SCALE,
    ref_start: str = "2003-01",
    ref_end: str = "2019-12",
    mode: str = "standalone",
    pool_id: str | None = None,
    ssi_method: str = DEFAULT_SSI_METHOD,
) -> str:
    """
    Return the expected SSI file path for a given (model, scenario).
    """
    reg = reg or default_registry()
    key = f"{model}_{scenario}"

    if pool_id is None:
        pool_id = "standalone" if mode == "standalone" else "ALL_SCENARIOS"

    return expected_ssi_path(
        yaml_path=reg.yaml_path,
        is_model=True,
        key=key,
        scale=scale,
        ref_start=ref_start,
        ref_end=ref_end,
        mode=mode,
        pool_id=pool_id,
        ssi_method=ssi_method,
    )


def ssi_obs_path(
    obs_key: str,
    *,
    reg: Registry | None = None,
    scale: int = _DEFAULT_SSI_SCALE,
    ref_start: str = "2003-01",
    ref_end: str = "2019-12",
    ssi_method: str = DEFAULT_SSI_METHOD,
) -> str:
    """
    Return the expected SSI file path for an observed dataset.
    """
    reg = reg or default_registry()
    return expected_ssi_path(
        yaml_path=reg.yaml_path,
        is_model=False,
        key=obs_key,
        scale=scale,
        ref_start=ref_start,
        ref_end=ref_end,
        mode="standalone",
        ssi_method=ssi_method,
    )


# ---------------------------------------------------------------------------
# Compute-or-skip SSI for models and observations
# ---------------------------------------------------------------------------


def ensure_ssi_model(
    model: str,
    scenario: str,
    *,
    reg: Registry | None = None,
    scale: int = _DEFAULT_SSI_SCALE,
    ref_start: str = "2003-01",
    ref_end: str = "2019-12",
    mode: str = "standalone",
    pool_scenarios: Sequence[str] | None = None,
    fixed_ref_scenario: str | None = None,
    ssi_method: str = DEFAULT_SSI_METHOD,
    tail_quantile: float = _DEFAULT_TAIL_QUANTILE,
    min_tail_size: int = _DEFAULT_MIN_TAIL_SIZE,
    loc: str = _DEFAULT_LOC,
    scale_method: str = _DEFAULT_SCALE_METHOD,
) -> str:
    """
    Compute-or-skip SSI for a single (model, scenario).

    Parameters
    ----------
    mode : {"standalone", "pooled", "fixed"}
        - standalone: ECDF reference is the target series itself.
        - pooled: ECDF reference is the concatenation of multiple scenarios.
        - fixed: ECDF reference is a single designated scenario
          (specified by *fixed_ref_scenario*, e.g. "obsclim_histsoc").
    pool_scenarios : sequence of str, optional
        Scenarios to pool (only used when mode="pooled").
    fixed_ref_scenario : str, optional
        The single scenario used as ECDF reference when mode="fixed"
        (e.g. "obsclim_histsoc"). Required when mode="fixed".

    Returns
    -------
    str
        Path to the SSI file on disk.
    """
    if mode == "fixed" and not fixed_ref_scenario:
        raise ValueError("mode='fixed' requires fixed_ref_scenario (e.g. 'obsclim_histsoc').")

    reg = reg or default_registry()
    key = f"{model}_{scenario}"

    scens: list[str] | None = None
    if mode == "pooled":
        scens = list(pool_scenarios) if pool_scenarios is not None else list(reg.scenarios())
        pool_id = "__".join(sorted(scens)) if pool_scenarios is not None else "ALL_SCENARIOS"
    elif mode == "fixed":
        pool_id = fixed_ref_scenario  # type: ignore[assignment]
    else:
        pool_id = "standalone"

    out_path = expected_ssi_path(
        yaml_path=reg.yaml_path,
        is_model=True,
        key=key,
        scale=scale,
        ref_start=ref_start,
        ref_end=ref_end,
        mode=mode,
        pool_id=pool_id,
        ssi_method=ssi_method,
    )
    if os.path.exists(out_path):
        return out_path

    # 0–1 m processed file — open lazily with spatial chunks so data
    # flows through the dask graph without materialising in the main process.
    src_path = reg.get_model_processed(model, scenario)
    if not os.path.exists(src_path):
        raise FileNotFoundError(
            f"Missing processed model input for SSI: model={model} scenario={scenario} path={src_path}"
        )
    _time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds = xr.open_dataset(src_path, decode_times=_time_coder, chunks=_chunks)

    # Load land mask for GPD method (skips ocean/ice pixels)
    land_mask = None
    if ssi_method == "deseasonal_ecdf_gpd":
        try:
            land_mask = load_isimip_landmask(_GPD_LAND_MASK_KEY)
        except FileNotFoundError:
            pass  # degrade gracefully — run without mask

    # Build reference data depending on mode
    if mode == "pooled":
        assert scens is not None
        ref_list = []
        for sc in scens:
            p = reg.get_model_processed(model, sc)
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Missing pooled reference input for SSI: model={model} scenario={sc} path={p}"
                )
            ref_list.append(xr.open_dataset(p, decode_times=_time_coder, chunks=_chunks)["soilmoist_1m"])
        ref_da = xr.concat(ref_list, dim="time").sortby("time")
    elif mode == "fixed":
        ref_path = reg.get_model_processed(model, fixed_ref_scenario)  # type: ignore[arg-type]
        if not os.path.exists(ref_path):
            raise FileNotFoundError(
                f"Missing fixed reference input for SSI: model={model} ref_scenario={fixed_ref_scenario} path={ref_path}"
            )
        ref_da = xr.open_dataset(ref_path, decode_times=_time_coder, chunks=_chunks)["soilmoist_1m"]
    else:
        ref_da = None

    out = save_ssi(
        ds["soilmoist_1m"],
        key=key,
        is_model=True,
        scale=scale,
        ref_start=ref_start,
        ref_end=ref_end,
        mode=mode,
        ref_data=ref_da,
        pool_id=pool_id,
        pool_scenarios=scens,
        ssi_method=ssi_method,
        tail_quantile=tail_quantile,
        min_tail_size=min_tail_size,
        loc=loc,
        scale_method=scale_method,
        land_mask=land_mask,
    )
    return out


def ensure_ssi_obs(
    obs_key: str,
    *,
    reg: Registry | None = None,
    scale: int = _DEFAULT_SSI_SCALE,
    ref_start: str = "2003-01",
    ref_end: str = "2019-12",
    ssi_method: str = DEFAULT_SSI_METHOD,
    tail_quantile: float = _DEFAULT_TAIL_QUANTILE,
    min_tail_size: int = _DEFAULT_MIN_TAIL_SIZE,
    loc: str = _DEFAULT_LOC,
    scale_method: str = _DEFAULT_SCALE_METHOD,
) -> str:
    """
    Compute-or-skip SSI for a single observed dataset key.

    Returns
    -------
    str
        Path to the SSI file on disk.
    """
    reg = reg or default_registry()
    out_path = expected_ssi_path(
        yaml_path=reg.yaml_path,
        is_model=False,
        key=obs_key,
        scale=scale,
        ref_start=ref_start,
        ref_end=ref_end,
        mode="standalone",
        ssi_method=ssi_method,
    )
    if os.path.exists(out_path):
        return out_path

    src_path = reg.get_obs_processed(obs_key)
    if not os.path.exists(src_path):
        raise FileNotFoundError(
            f"Missing processed observation input for SSI: obs={obs_key} path={src_path}"
        )
    _time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds = xr.open_dataset(src_path, decode_times=_time_coder, chunks=_chunks)

    if "soilmoist_1m" in ds:
        da = ds["soilmoist_1m"]
    else:
        first = next(iter(ds.data_vars))
        da = ds[first]

    # Load land mask for GPD method (skips ocean/ice pixels)
    land_mask = None
    if ssi_method == "deseasonal_ecdf_gpd":
        try:
            land_mask = load_isimip_landmask(_GPD_LAND_MASK_KEY)
        except FileNotFoundError:
            pass  # degrade gracefully — run without mask

    out = save_ssi(
        da,
        key=obs_key,
        is_model=False,
        scale=scale,
        ref_start=ref_start,
        ref_end=ref_end,
        ssi_method=ssi_method,
        tail_quantile=tail_quantile,
        min_tail_size=min_tail_size,
        loc=loc,
        scale_method=scale_method,
        land_mask=land_mask,
    )
    return out


def ensure_all_models(
    models: Iterable[str],
    scenarios: Iterable[str],
    *,
    reg: Registry | None = None,
    scale: int = _DEFAULT_SSI_SCALE,
    ref_start: str = "2003-01",
    ref_end: str = "2019-12",
    mode: str = "standalone",
    pool_scenarios: Sequence[str] | None = None,
    fixed_ref_scenario: str | None = None,
    ssi_method: str = DEFAULT_SSI_METHOD,
    tail_quantile: float = _DEFAULT_TAIL_QUANTILE,
    min_tail_size: int = _DEFAULT_MIN_TAIL_SIZE,
    loc: str = _DEFAULT_LOC,
    scale_method: str = _DEFAULT_SCALE_METHOD,
) -> Dict[Tuple[str, str], str]:
    """
    Compute-or-skip SSI for all (model, scenario) combinations.

    Returns
    -------
    dict
        Mapping {(model, scenario): ssi_path}.
    """
    reg = reg or default_registry()
    out: Dict[Tuple[str, str], str] = {}
    for m in models:
        for s in scenarios:
            out[(m, s)] = ensure_ssi_model(
                m,
                s,
                reg=reg,
                scale=scale,
                ref_start=ref_start,
                ref_end=ref_end,
                mode=mode,
                pool_scenarios=pool_scenarios,
                fixed_ref_scenario=fixed_ref_scenario,
                ssi_method=ssi_method,
                tail_quantile=tail_quantile,
                min_tail_size=min_tail_size,
                loc=loc,
                scale_method=scale_method,
            )
    return out


def ensure_all_obs(
    obs_keys: Iterable[str],
    *,
    reg: Registry | None = None,
    scale: int = _DEFAULT_SSI_SCALE,
    ref_start: str = "2003-01",
    ref_end: str = "2019-12",
    ssi_method: str = DEFAULT_SSI_METHOD,
    tail_quantile: float = _DEFAULT_TAIL_QUANTILE,
    min_tail_size: int = _DEFAULT_MIN_TAIL_SIZE,
    loc: str = _DEFAULT_LOC,
    scale_method: str = _DEFAULT_SCALE_METHOD,
) -> Dict[str, str]:
    """
    Compute-or-skip SSI for all observational datasets.

    Returns
    -------
    dict
        Mapping {obs_key: ssi_path}.
    """
    reg = reg or default_registry()
    return {
        k: ensure_ssi_obs(
            k,
            reg=reg,
            scale=scale,
            ref_start=ref_start,
            ref_end=ref_end,
            ssi_method=ssi_method,
            tail_quantile=tail_quantile,
            min_tail_size=min_tail_size,
            loc=loc,
            scale_method=scale_method,
        )
        for k in obs_keys
    }


def correlation_map_path(
    model: str,
    scenario: str,
    obs_key: str,
    *,
    target: str = "ssi",
    mode: str = "standalone",
    corr_start: str = "2004-01",
    corr_end: str = "2019-12",
    ssi_method: str = DEFAULT_SSI_METHOD,
    reg: Registry | None = None,
) -> str:
    """
    Return the expected path for a *single-model* correlation map
    according to data_registry.yml.

    This mirrors ssi_model_path/ssi_obs_path and is useful for
    analysis/plotting code that needs to locate an existing
    correlation file without hard-coding absolute paths.
    """
    reg = reg or default_registry()
    cfg = reg.cfg_dict
    tmpl = cfg["metrics"]["correlations_map"]

    corrstart_yr = corr_start[:4]
    corrend_yr = corr_end[:4]

    out_path = _format_from_template(
        tmpl,
        paths=cfg["paths"],
        mode=mode,
        target=target,
        obs_short=obs_key,
        model=model,
        scenario=scenario,
        corrstart_yr=corrstart_yr,
        corrend_yr=corrend_yr,
        ssi_method=ssi_method,
    )
    return out_path


def correlation_multimodel_map_path(
    scenario: str,
    obs_key: str,
    *,
    target: str = "ssi",
    mode: str = "standalone",
    corr_start: str = "2004-01",
    corr_end: str = "2019-12",
    ssi_method: str = DEFAULT_SSI_METHOD,
    reg: Registry | None = None,
) -> str:
    """
    Return the expected path for a *multi-model* mean correlation map
    for a given scenario and observational dataset.

    This uses the `metrics.correlations_multimodel_map` template in
    data_registry.yml and matches the filenames written by the
    batch correlation script.
    """
    reg = reg or default_registry()
    cfg = reg.cfg_dict
    tmpl = cfg["metrics"]["correlations_multimodel_map"]

    corrstart_yr = corr_start[:4]
    corrend_yr = corr_end[:4]

    out_path = _format_from_template(
        tmpl,
        paths=cfg["paths"],
        mode=mode,
        target=target,
        obs_short=obs_key,
        scenario=scenario,
        corrstart_yr=corrstart_yr,
        corrend_yr=corrend_yr,
        ssi_method=ssi_method,
    )
    return out_path
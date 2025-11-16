"""
Utilities for standardized SSI paths and on-demand SSI generation
for model and observational datasets, plus helpers for correlation
map paths (single-model and multi-model).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import xarray as xr
import yaml

from ..io.registry import Registry, default_registry
from ..io.settings import get_settings
from .ssi import save_ssi  # uses ssi_templates in data_registry.yml


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

    This is shared between SSI templates and correlation templates,
    so keep behaviour very generic.
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
    key: str,  # "model_scenario" for models, obs key for observations
    scale: int,
    ref_start: str,
    ref_end: str,
    mode: str,
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
    mode : {"standalone", "pooled"}
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
) -> str:
    """
    Return the expected SSI file path for a given (model, scenario)
    according to data_registry.yml, without computing it.

    This is useful for analysis steps (e.g., correlation) that only
    need to locate already-produced SSI files.
    """
    reg = reg or default_registry()
    key = f"{model}_{scenario}"
    return expected_ssi_path(
        yaml_path=reg.yaml_path,
        is_model=True,
        key=key,
        scale=scale,
        ref_start=ref_start,
        ref_end=ref_end,
        mode=mode,
    )


def ssi_obs_path(
    obs_key: str,
    *,
    reg: Registry | None = None,
    scale: int = _DEFAULT_SSI_SCALE,
    ref_start: str = "2003-01",
    ref_end: str = "2019-12",
) -> str:
    """
    Return the expected SSI file path for an observed dataset
    according to data_registry.yml, without computing it.
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
) -> str:
    """
    Compute-or-skip SSI for a single (model, scenario).

    Returns
    -------
    str
        Path to the SSI file on disk.
    """
    reg = reg or default_registry()
    key = f"{model}_{scenario}"
    out_path = expected_ssi_path(
        yaml_path=reg.yaml_path,
        is_model=True,
        key=key,
        scale=scale,
        ref_start=ref_start,
        ref_end=ref_end,
        mode=mode,
    )
    if os.path.exists(out_path):
        return out_path

    # 0–1 m processed file
    src_path = reg.get_model_processed(model, scenario)
    ds = xr.open_dataset(src_path)

    # Reference data for pooled ECDF (across scenarios)
    if mode == "pooled":
        scens = list(reg.scenarios())
        ref_list = []
        for sc in scens:
            p = reg.get_model_processed(model, sc)
            ref_list.append(xr.open_dataset(p)["soilmoist_1m"])
        ref_da = xr.concat(ref_list, dim="time").sortby("time")
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
    )
    return out


def ensure_ssi_obs(
    obs_key: str,
    *,
    reg: Registry | None = None,
    scale: int = _DEFAULT_SSI_SCALE,
    ref_start: str = "2003-01",
    ref_end: str = "2019-12",
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
    )
    if os.path.exists(out_path):
        return out_path

    src_path = reg.get_obs_processed(obs_key)  # 0–1 m (or equivalent) file
    ds = xr.open_dataset(src_path)

    # Observed 1m files may not always use "soilmoist_1m" as a name; normalize.
    if "soilmoist_1m" in ds:
        da = ds["soilmoist_1m"]
    else:
        # Fallback: first data variable
        first = next(iter(ds.data_vars))
        da = ds[first]

    out = save_ssi(
        da,
        key=obs_key,
        is_model=False,
        scale=scale,
        ref_start=ref_start,
        ref_end=ref_end,
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
            )
    return out


def ensure_all_obs(
    obs_keys: Iterable[str],
    *,
    reg: Registry | None = None,
    scale: int = _DEFAULT_SSI_SCALE,
    ref_start: str = "2003-01",
    ref_end: str = "2019-12",
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
        )
        for k in obs_keys
    }


# ---------------------------------------------------------------------------
# Correlation map path helpers
# ---------------------------------------------------------------------------


def correlation_map_path(
    model: str,
    scenario: str,
    obs_key: str,
    *,
    target: str = "ssi",
    mode: str = "standalone",
    corr_start: str = "2004-01",
    corr_end: str = "2019-12",
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
    )
    return out_path
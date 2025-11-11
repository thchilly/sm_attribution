# src/sm_attribution/analysis/ensemble.py
from __future__ import annotations
import os
import xarray as xr
from typing import Dict, Iterable, Tuple
from pathlib import Path
import yaml

from ..io.registry import Registry, default_registry
from .ssi import save_ssi  # uses ssi_templates in data_registry.yml

# --- small helpers to resolve SSI output path without computing ---
def _load_registry_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

def _format_from_template(tmpl: str, **kw) -> str:
    if "paths" in kw and isinstance(kw["paths"], dict):
        for k, v in kw["paths"].items():
            tmpl = tmpl.replace("{paths." + k + "}", v)
    return tmpl.format(**kw)

def expected_ssi_path(
    *,
    yaml_path: str,
    is_model: bool,
    key: str,                 # "model_scenario" for models, obs key for observations
    scale: int,
    ref_start: str,
    ref_end: str,
    mode: str,
) -> str:
    cfg = _load_registry_yaml(yaml_path)
    paths = cfg.get("paths", {})
    tmpls = cfg.get("ssi_templates", {})
    refstart_yr = ref_start[:4]
    refend_yr = ref_end[:4]
    if is_model:
        model, scenario = key.split("_", 1)
        tmpl = tmpls["model"]
        out_path = _format_from_template(
            tmpl, paths=paths, model=model, scenario=scenario,
            scale=scale, refstart=ref_start.replace("-", ""), refend=ref_end.replace("-", ""),
            mode=mode, refstart_yr=refstart_yr, refend_yr=refend_yr
        )
    else:
        tmpl = tmpls["observed"]
        out_path = _format_from_template(
            tmpl, paths=paths, obskey=key,
            scale=scale, refstart=ref_start.replace("-", ""), refend=ref_end.replace("-", "")
        )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    return out_path

# --- public API ---
def ensure_ssi_model(
    model: str,
    scenario: str,
    *,
    reg: Registry | None = None,
    scale: int = 3,
    ref_start: str = "2003-01",
    ref_end: str = "2019-12",
    mode: str = "standalone",
) -> str:
    """Compute-or-skip SSI for a single (model, scenario). Returns SSI path."""
    reg = reg or default_registry()
    key = f"{model}_{scenario}"
    out_path = expected_ssi_path(
        yaml_path=reg.yaml_path, is_model=True, key=key,
        scale=scale, ref_start=ref_start, ref_end=ref_end, mode=mode
    )
    if os.path.exists(out_path):
        return out_path

    src_path = reg.get_model_processed(model, scenario)  # 0–1 m file
    ds = xr.open_dataset(src_path)

    if mode == "pooled":
        scens = list(reg.scenarios())
        ref_list = []
        for sc in scens:
            p = reg.get_model_processed(model, sc)
            ref_list.append(xr.open_dataset(p)["soilmoist_1m"])
        ref_da = xr.concat(ref_list, dim="time")
        ref_da = ref_da.sortby("time")
    else:
        ref_da = None

    out = save_ssi(
        ds["soilmoist_1m"], key=key, is_model=True,
        scale=scale, ref_start=ref_start, ref_end=ref_end,
        mode=mode, ref_data=ref_da
    )
    return out

def ensure_ssi_obs(
    obs_key: str,
    *,
    reg: Registry | None = None,
    scale: int = 3,
    ref_start: str = "2003-01",
    ref_end: str = "2019-12",
) -> str:
    """Compute-or-skip SSI for a single observed dataset key. Returns SSI path."""
    reg = reg or default_registry()
    out_path = expected_ssi_path(
        yaml_path=reg.yaml_path, is_model=False, key=obs_key,
        scale=scale, ref_start=ref_start, ref_end=ref_end,
        mode="standalone"
    )
    if os.path.exists(out_path):
        return out_path

    src_path = reg.get_obs_processed(obs_key)  # 0–1 m file
    ds = xr.open_dataset(src_path)
    # observed 1m files may not always use "soilmoist_1m" as a name; normalize here if needed
    if "soilmoist_1m" in ds:
        da = ds["soilmoist_1m"]
    else:
        # fallback: first data var
        first = next(iter(ds.data_vars))
        da = ds[first]
    out = save_ssi(
        da, key=obs_key, is_model=False,
        scale=scale, ref_start=ref_start, ref_end=ref_end
    )
    return out

def ensure_all_models(
    models: Iterable[str],
    scenarios: Iterable[str],
    *,
    reg: Registry | None = None,
    scale: int = 3,
    ref_start: str = "2003-01",
    ref_end: str = "2019-12",
    mode: str = "standalone",
) -> Dict[Tuple[str, str], str]:
    """Compute-or-skip SSI for all (model, scenario). Returns {(model,scenario): path}."""
    reg = reg or default_registry()
    out: Dict[Tuple[str, str], str] = {}
    for m in models:
        for s in scenarios:
            out[(m, s)] = ensure_ssi_model(
                m, s, reg=reg, scale=scale, ref_start=ref_start, ref_end=ref_end, mode=mode
            )
    return out

def ensure_all_obs(
    obs_keys: Iterable[str],
    *,
    reg: Registry | None = None,
    scale: int = 3,
    ref_start: str = "2003-01",
    ref_end: str = "2019-12",
) -> Dict[str, str]:
    """Compute-or-skip SSI for all obs datasets. Returns {obs_key: path}."""
    reg = reg or default_registry()
    return {
        k: ensure_ssi_obs(k, reg=reg, scale=scale, ref_start=ref_start, ref_end=ref_end)
        for k in obs_keys
    }
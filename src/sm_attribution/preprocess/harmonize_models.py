from __future__ import annotations
import xarray as xr
from .calendar import to_proleptic_monthly
from ..io.registry import Registry
from .depth_1m import MODEL_TO_FUNC

def harmonize_one(model: str, scenario: str, reg: Registry) -> str:
    """
    Read raw, normalize calendar, apply v0 depth recipe for 0–1 m, and write _1m.nc.
    Returns path to the written file.
    """
    raw_path = reg.get_model_raw(model, scenario)
    out_path = reg.get_model_processed(model, scenario)

    # Open lazily; we only massage coordinates & one reduction
    ds = xr.open_dataset(raw_path, decode_times=True)

    # Calendar normalization
    ds = to_proleptic_monthly(ds)

    # Per-model 0–1 m transform
    try:
        fn = MODEL_TO_FUNC[model]
    except KeyError as e:
        raise KeyError(f"No v0 depth recipe registered for model={model}") from e
    da_1m = fn(ds, scenario)

    # write as netCDF (NetCDF4 classic compatible)
    enc = {da_1m.name: {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": -9999.0}}
    da_1m.to_dataset().to_netcdf(out_path, encoding=enc)

    return out_path

def harmonize_all(selected_models: list[str], reg: Registry) -> list[str]:
    written = []
    for model in selected_models:
        for scenario in reg.scenarios():
            written.append(harmonize_one(model, scenario, reg))
    return written
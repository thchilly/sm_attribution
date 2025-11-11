"""
Nonparametric SSI (AghaKouchak-style [2015]) with month-wise ECDF and Gaussian transform.

- compute_ssi_np(): core SSI for a single DataArray (monthly, already 0–1 m).
- save_ssi(): writes SSI to a standardized filename using data_registry templates.
- Supports pooled or standalone reference via optional ref_data.
"""

from __future__ import annotations
import numpy as np
import xarray as xr
from scipy.stats import norm
from typing import Tuple, Optional, Dict
from sm_attribution.io.registry import Registry, default_registry
from pathlib import Path  # keep this for writing files


def _format_from_template(tmpl: str, **kw) -> str:
    # Allow nested "{paths.ssi_models}" resolution
    # 1) expand {paths.*}
    if "paths" in kw and isinstance(kw["paths"], dict):
        for k, v in kw["paths"].items():
            tmpl = tmpl.replace("{paths." + k + "}", v)
    # 2) expand the rest
    return tmpl.format(**kw)

# --- Core SSI ---
def compute_ssi_np(
    da: xr.DataArray,
    *,
    scale: int = 3,
    ref_start: str = "2003-01",
    ref_end: str = "2019-12",
    ref_data: xr.DataArray | None = None,
    time_name: str = "time",
) -> xr.DataArray:
    """
    Nonparametric SSI with month-wise ECDF:
      1) rolling sum over `scale` months,
      2) per calendar month, compute ECDF from reference period only,
      3) map probabilities to z via inverse normal.

    Parameters
    ----------
    da : xr.DataArray
        Monthly soil moisture mass [kg m-2], dims include (time, lat, lon).
    scale : int
        Accumulation window in months.
    ref_start, ref_end : str
        Reference period (inclusive).
    ref_data : xr.DataArray or None
        Optional reference DataArray to build the ECDF (e.g., pooled across scenarios).
        If None, the reference is taken from the target series itself.
    time_name : str
        Name of time dimension (default 'time').

    Returns
    -------
    xr.DataArray
        SSI with same dims as input; first (scale-1) per-month entries are NaN by design.
    """
    # 1) rolling accumulation
    y = da.rolling({time_name: scale}, min_periods=scale).sum()

    # Reference subset
    src_for_ref = ref_data if ref_data is not None else da
    y_ref_src = src_for_ref.rolling({time_name: scale}, min_periods=scale).sum()
    ref = y_ref_src.sel({time_name: slice(ref_start, ref_end)})

    # Output container
    ssi = xr.full_like(y, np.nan, dtype="float32")

    months_all = y[time_name].dt.month
    months_ref = ref[time_name].dt.month

    # Work per calendar month
    for m in range(1, 13):
        ref_m = ref.where(months_ref == m, drop=True)
        tgt_m = y.where(months_all == m, drop=True)

        if ref_m.sizes.get(time_name, 0) == 0 or tgt_m.sizes.get(time_name, 0) == 0:
            continue

        # Flatten space dims into columns: (time, ngrid)
        ref_vals = ref_m.data  # (t_ref, lat, lon)
        tgt_vals = tgt_m.data  # (t_tgt, lat, lon)
        ref_flat = ref_vals.reshape(ref_vals.shape[0], -1)
        tgt_flat = tgt_vals.reshape(tgt_vals.shape[0], -1)

        # Valid mask per (time, grid)
        valid_ref = np.isfinite(ref_flat)
        # Sort reference values per grid (NaNs end up last if present)
        ref_sorted = np.sort(np.where(valid_ref, ref_flat, np.nan), axis=0)

        # Number of valid samples per grid
        n = valid_ref.sum(axis=0).astype(float)
        n[n == 0] = np.nan  # avoid divide-by-zero

        # Count <= for each target against ref distribution
        # Broadcasting: (t_tgt, 1, ngrid) <= (1, t_ref, ngrid) -> (t_tgt, t_ref, ngrid)
        # For memory safety, do a small loop over blocks of time if needed; here it’s fine.
        leq = (ref_sorted[None, ...] <= tgt_flat[:, None, :])
        idx = np.nansum(leq, axis=1)  # (t_tgt, ngrid)

        # Plotting-position probability
        p = (idx - 0.44) / (n + 0.12)
        p = np.clip(p, 1e-6, 1 - 1e-6)

        z = norm.ppf(p).astype("float32")
        z_da = xr.DataArray(
            z.reshape(tgt_m.shape),
            coords=tgt_m.coords,
            dims=tgt_m.dims,
            name="ssi",
        )
        ssi.loc[{time_name: tgt_m[time_name]}] = z_da

    ssi.name = "ssi"
    ssi.attrs.update({
        "long_name": "Standardized Soil Moisture Index (nonparametric, month-wise ECDF)",
        "units": "-",
        "ssi_scale": scale,
        "ssi_ref_period": f"{ref_start}:{ref_end}",
        "method": "Rolling sum over scale months; month-wise ECDF from reference period; norm.ppf transform.",
    })
    return ssi


def save_ssi(
    da_1m: xr.DataArray,
    *,
    key: str,                 # e.g., "h08_obsclim_histsoc" or "era5land_1950_2020"
    is_model: bool,           # True for models, False for observed
    reg: Registry | None = None,
    scale: int = 3,
    ref_start: str = "2003-01",
    ref_end: str = "2019-12",
    mode: str = "standalone",
    ref_data: xr.DataArray | None = None,
) -> str:
    if reg is None:
        reg = default_registry()

    cfg = reg.cfg_dict
    paths = cfg.get("paths", {})
    tmpls = cfg.get("ssi_templates", {})

    refstart_yr = ref_start[:4]
    refend_yr = ref_end[:4]

    if is_model:
        try:
            model, scenario = key.split("_", 1)
        except ValueError:
            raise ValueError("For models, key must be 'model_scenario', e.g. 'h08_obsclim_histsoc'.")
        tmpl = tmpls.get("model")
        out_path = _format_from_template(
            tmpl, paths=paths, model=model, scenario=scenario,
            scale=scale, refstart=ref_start.replace("-", ""), refend=ref_end.replace("-", ""),
            mode=mode, refstart_yr=refstart_yr, refend_yr=refend_yr
        )
    else:
        tmpl = tmpls.get("observed")
        out_path = _format_from_template(
            tmpl, paths=paths, obskey=key,
            scale=scale, refstart=ref_start.replace("-", ""), refend=ref_end.replace("-", "")
        )

    ssi = compute_ssi_np(da_1m, scale=scale, ref_start=ref_start, ref_end=ref_end, ref_data=ref_data)

    # Annotate mode/reference details in attributes
    ref_mode = mode if is_model else "standalone"
    if ref_data is None:
        ref_desc = "standalone: ECDF from the target series only (within reference window)."
    else:
        ref_desc = "pooled: ECDF built from pooled scenarios of the same model (within reference window)."
    ssi.attrs["ssi_mode"] = ref_mode
    ssi.attrs["ssi_ref_start"] = ref_start
    ssi.attrs["ssi_ref_end"] = ref_end
    ssi.attrs["ssi_ref_description"] = ref_desc

    enc = {"ssi": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)}}
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    ds_out = ssi.to_dataset()
    # also record key-level provenance in global attrs
    ds_out.attrs.update({
        "ssi_mode": ref_mode,
        "ssi_scale": scale,
        "ssi_ref_start": ref_start,
        "ssi_ref_end": ref_end,
        "ssi_ref_description": ref_desc,
        "source_key": key,
        "source_type": "model" if is_model else "observed",
    })
    ds_out.to_netcdf(str(outp), encoding=enc)
    return str(outp)
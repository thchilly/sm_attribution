"""
Drought feature extraction from SSI products.

Implements the theory-of-runs method for drought characterisation:

1) "Bridge" short mildly-wet gaps: contiguous runs where 0 <= SSI <= 1
   with length < bridge_len_months (default 3) are converted to slightly
   negative (-eps).  This prevents short near-normal interruptions from
   splitting drought events.

2) Remove weak negative spells: contiguous runs where SSI < 0 whose minimum
   is > severity_threshold (default -1) are converted to slightly positive (+eps).
   This enforces that an event must reach at least severity_threshold at least once.

Then drought events are contiguous runs where SSI < 0 in the modified series.

Per-event features (nanmean across events):
- D   : duration (months)
- M   : magnitude = sum(abs(SSI)) over event months
- PI  : intensity = M / D   (mean deficit per month)
- PeakI : peak intensity = |min(SSI)| in the event
- DDD : development duration = months from start to first minimum SSI
- TTM10 : months from start to first SSI ≤ −1.0 (NaN if never crossed)
- TTS15 : months from start to first SSI ≤ −1.5
- TTE20 : months from start to first SSI ≤ −2.0
- DRD : recovery duration = months from last minimum SSI to end of event
- N   : number of events

Inter-event features:
- Ld  : inter-arrival time = mean(start_i+1 − start_i) across consecutive pairs
- Rp  : return period       = mean(start_i+1 − end_i)   across consecutive pairs
        (NaN when fewer than 2 events)

Note: DDD + DRD ≤ D.  They may not sum to D when the minimum SSI value
occurs at multiple time steps (DDD uses the first, DRD uses the last).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Tuple, Optional

import numpy as np
import xarray as xr
import yaml

from ..io.registry import Registry, default_registry
from .ensemble import ssi_model_path, ssi_obs_path
from .ssi import DEFAULT_SSI_METHOD, _CHUNK_LAT, _CHUNK_LON, _get_dask_client, _DASK_NUM_WORKERS


# ---------------------------------------------------------------------
# Internal template helpers (same style as ensemble.py)
# ---------------------------------------------------------------------


def _load_registry_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def _format_from_template(tmpl: str, **kw) -> str:
    paths = kw.get("paths")
    if isinstance(paths, dict):
        for k, v in paths.items():
            tmpl = tmpl.replace("{paths." + k + "}", v)
    return tmpl.format(**kw)


def expected_drought_features_path(
    *,
    yaml_path: str,
    is_model: bool,
    key: str,  # model_scenario if model else obskey
    mode: str,
    pool_id: str,
    scale: int,
    ref_start: str,
    ref_end: str,
    feat_start: str,
    feat_end: str,
    ssi_method: str = DEFAULT_SSI_METHOD,
) -> str:
    cfg = _load_registry_yaml(yaml_path)
    paths = cfg.get("paths", {})
    tmpls = cfg.get("drought_features_templates", {})
    if not tmpls:
        raise KeyError(
            "Missing `drought_features_templates` in data_registry.yml. "
            "Add templates for model and observed outputs."
        )

    refstart_yr = ref_start[:4]
    refend_yr = ref_end[:4]
    featstart_yr = feat_start[:4]
    featend_yr = feat_end[:4]

    if is_model:
        model, scenario = key.split("_", 1)
        tmpl = tmpls["model"]
        out_path = _format_from_template(
            tmpl,
            paths=paths,
            mode=mode,
            pool_id=pool_id,
            model=model,
            scenario=scenario,
            scale=scale,
            refstart=ref_start.replace("-", ""),
            refend=ref_end.replace("-", ""),
            refstart_yr=refstart_yr,
            refend_yr=refend_yr,
            featstart=feat_start.replace("-", ""),
            featend=feat_end.replace("-", ""),
            featstart_yr=featstart_yr,
            featend_yr=featend_yr,
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
            featstart=feat_start.replace("-", ""),
            featend=feat_end.replace("-", ""),
            featstart_yr=featstart_yr,
            featend_yr=featend_yr,
            ssi_method=ssi_method,
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    return out_path


# ---------------------------------------------------------------------
# Matlab-faithful 1D event logic
# ---------------------------------------------------------------------


def _label_runs_1d(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Label contiguous True runs in a 1D boolean array.
    Returns (labels, n_labels). labels are 0 where mask is False.
    """
    mask = mask.astype(bool)
    n = mask.size
    labels = np.zeros(n, dtype=np.int32)
    if n == 0:
        return labels, 0

    # run starts where mask becomes True
    starts = np.flatnonzero(mask & np.concatenate(([True], ~mask[:-1])))
    if starts.size == 0:
        return labels, 0

    # run ends where mask becomes False (exclusive end index)
    ends = np.flatnonzero(mask & np.concatenate((~mask[1:], [True]))) + 1

    for i, (s, e) in enumerate(zip(starts, ends), start=1):
        labels[s:e] = i
    return labels, int(len(starts))


def _runtheo_features_1d(
    ssi: np.ndarray,
    *,
    bridge_low: float = 0.0,
    bridge_high: float = 1.0,
    bridge_len_months: int = 3,
    severity_threshold: float = -1.0,
    drought_threshold: float = 0.0,
    ttm10_threshold: float = -1.0,
    tts15_threshold: float = -1.5,
    tte20_threshold: float = -2.0,
    eps: float = 1e-6,
) -> Tuple[
    np.float32, np.float32, np.float32, np.float32,
    np.float32, np.float32, np.float32, np.float32,
    np.float32, np.float32, np.float32, np.float32,
]:
    """
    Compute drought features for a single 1-D SSI time series
    using the theory-of-runs method.

    Algorithm
    ---------
    1. **Bridge** short mildly-wet gaps: contiguous runs where
       ``bridge_low <= SSI <= bridge_high`` with length
       ``< bridge_len_months`` are set to ``-eps`` (merged into the
       surrounding drought).  Default bridge length is **3 months**.
    2. **Remove weak spells**: contiguous negative runs whose minimum
       never reaches ``<= severity_threshold`` (default −1) are set
       to ``+eps`` (discarded as non-drought).
    3. **Drought events** are the remaining contiguous runs where
       ``SSI < drought_threshold`` (default 0).

    Per-event features
    ------------------
    D   – duration (months)
    M   – magnitude = Σ|SSI| over event months
    PI  – intensity = M / D
    PeakI – peak intensity = |min(SSI)| in the event
    DDD – development duration = months from start to **first**
          occurrence of the minimum SSI (1-based)
    TTM10 – months from start to first SSI ≤ −1.0 (NaN if never)
    TTS15 – months from start to first SSI ≤ −1.5 (NaN if never)
    TTE20 – months from start to first SSI ≤ −2.0 (NaN if never)
    DRD – recovery duration = months from **last** occurrence of
          the minimum SSI to the end of the event (``D − last_min_pos``)

    Inter-event features (computed across consecutive events)
    --------------------------------------------------------
    Ld  – inter-arrival time = mean of ``start(i+1) − start(i)``
    Rp  – return period       = mean of ``start(i+1) − end(i)``

    Returns
    -------
    12-tuple of np.float32 :
        ``(D, M, PI, PeakI, DDD, TTM10, TTS15, TTE20, DRD,
          N_events, Ld, Rp)``
        All per-event features are **nanmean** across events.
        ``N_events`` is the count.  ``Ld`` and ``Rp`` are NaN when
        there are fewer than 2 events.
    """
    # All-NaN sentinel for ocean / all-missing input
    nan12 = tuple([np.float32(np.nan)] * 12)
    # Valid land with zero drought events: n_events = 0, rest NaN
    noevents12 = tuple([np.float32(np.nan)] * 9) + (np.float32(0.0),) + (np.float32(np.nan),) * 2

    x = np.asarray(ssi, dtype=np.float32)

    if np.all(~np.isfinite(x)):
        return nan12

    ssi_ct = x.copy()

    # Rule A: bridge short mildly-wet gaps (<bridge_len_months) → −eps
    xp = np.isfinite(x) & (x >= bridge_low) & (x <= bridge_high)
    lab_p, n_p = _label_runs_1d(xp)
    for i in range(1, n_p + 1):
        idx = np.flatnonzero(lab_p == i)
        if idx.size < bridge_len_months:
            ssi_ct[idx] = -eps

    # Rule B: discard negative runs that never reach ≤ severity_threshold
    xn = np.isfinite(ssi_ct) & (ssi_ct < drought_threshold)
    lab_n, n_n = _label_runs_1d(xn)
    for i in range(1, n_n + 1):
        idx = np.flatnonzero(lab_n == i)
        if idx.size == 0:
            continue
        if np.nanmin(ssi_ct[idx]) > severity_threshold:
            ssi_ct[idx] = +eps

    # Final drought runs: ssi_ct < drought_threshold
    xd = np.isfinite(ssi_ct) & (ssi_ct < drought_threshold)
    lab_d, n_d = _label_runs_1d(xd)

    if n_d == 0:
        return noevents12

    # Per-event accumulators
    Ds = []
    Ms = []
    PIs = []
    PeakIs = []
    DDDs = []
    TTM10s = []
    TTS15s = []
    TTE20s = []
    DRDs = []

    def _safe_nanmean(vals) -> np.float32:
        if not vals:
            return np.float32(np.nan)
        arr = np.asarray(vals, dtype=np.float32)
        if not np.isfinite(arr).any():
            return np.float32(np.nan)
        return np.float32(np.nanmean(arr))

    # Track event start/end indices (0-based in the original series)
    # for inter-event features (Ld, Rp).
    event_starts = []
    event_ends = []  # inclusive end index

    for i in range(1, n_d + 1):
        idx = np.flatnonzero(lab_d == i)
        if idx.size == 0:
            continue
        ev = ssi_ct[idx]

        event_starts.append(int(idx[0]))
        event_ends.append(int(idx[-1]))

        D = float(idx.size)
        M = float(np.nansum(np.abs(ev)))
        PI = float(M / D) if D > 0 else np.nan

        # Minimum SSI in the event
        ev_min = np.nanmin(ev)
        kmin = np.flatnonzero(ev == ev_min)

        # Peak intensity = |min SSI|
        peak_i = float(np.abs(ev_min))

        # DDD: first occurrence of minimum (1-based)
        ddd = float(kmin[0] + 1) if kmin.size else np.nan

        # DRD: months from LAST occurrence of minimum to end
        # E.g. if min at positions [1,9,14,17] in a 20-month event →
        #   last_min_pos = 18 (1-based), DRD = 20 − 18 = 2
        drd = float(D - (kmin[-1] + 1)) if kmin.size else np.nan

        # Threshold-crossing features (all 1-based, NaN if never)
        k10 = np.flatnonzero(ev <= ttm10_threshold)
        ttm10 = float(k10[0] + 1) if k10.size else np.nan

        k15 = np.flatnonzero(ev <= tts15_threshold)
        tts15 = float(k15[0] + 1) if k15.size else np.nan

        k20 = np.flatnonzero(ev <= tte20_threshold)
        tte20 = float(k20[0] + 1) if k20.size else np.nan

        Ds.append(D)
        Ms.append(M)
        PIs.append(PI)
        PeakIs.append(peak_i)
        DDDs.append(ddd)
        TTM10s.append(ttm10)
        TTS15s.append(tts15)
        TTE20s.append(tte20)
        DRDs.append(drd)

    # Means across events
    Dm       = _safe_nanmean(Ds)
    Mm       = _safe_nanmean(Ms)
    PIm      = _safe_nanmean(PIs)
    PeakIm   = _safe_nanmean(PeakIs)
    DDDm     = _safe_nanmean(DDDs)
    TTM10m   = _safe_nanmean(TTM10s)
    TTS15m   = _safe_nanmean(TTS15s)
    TTE20m   = _safe_nanmean(TTE20s)
    DRDm     = _safe_nanmean(DRDs)
    Nm       = np.float32(len(Ds))

    # Inter-event features — require ≥ 2 events
    if len(event_starts) >= 2:
        starts = np.array(event_starts)
        ends   = np.array(event_ends)
        # Ld: inter-arrival = start(i+1) − start(i) for consecutive pairs
        ld_vals = np.diff(starts).astype(np.float32)
        Ldm = np.float32(np.nanmean(ld_vals))
        # Rp: return period = start(i+1) − end(i) (gap length)
        rp_vals = (starts[1:] - ends[:-1]).astype(np.float32)
        Rpm = np.float32(np.nanmean(rp_vals))
    else:
        Ldm = np.float32(np.nan)
        Rpm = np.float32(np.nan)

    return Dm, Mm, PIm, PeakIm, DDDm, TTM10m, TTS15m, TTE20m, DRDm, Nm, Ldm, Rpm


def drought_features_from_ssi(
    ssi: xr.DataArray,
    *,
    feat_start: str,
    feat_end: str,
    bridge_low: float = 0.0,
    bridge_high: float = 1.0,
    bridge_len_months: int = 3,
    severity_threshold: float = -1.0,
    drought_threshold: float = 0.0,
    ttm10_threshold: float = -1.0,
    tts15_threshold: float = -1.5,
    tte20_threshold: float = -2.0,
) -> xr.Dataset:
    """
    Compute drought feature maps from an SSI DataArray (time, lat, lon).

    Returns an xr.Dataset with 12 variables:
        duration, magnitude, intensity, peak_intensity,
        ddd, ttm10, tts15, tte20, drd,
        n_events, interarrival, return_period
    """
    ssi_win = ssi.sel(time=slice(feat_start, feat_end))

    # Ensure spatial chunking so dask="parallelized" distributes the
    # per-pixel kernel across all available workers.
    if not hasattr(ssi_win, "chunks") or ssi_win.chunks is None:
        ssi_win = ssi_win.chunk({"lat": _CHUNK_LAT, "lon": _CHUNK_LON})

    # Rechunk so time is a single chunk — apply_ufunc with
    # dask='parallelized' requires core dimensions to be unchunked.
    ssi_win = ssi_win.chunk({"time": -1, "lat": _CHUNK_LAT, "lon": _CHUNK_LON})

    _n_out = 12
    (D, M, PI, PeakI,
     DDD, TTM10, TTS15, TTE20, DRD,
     N, Ld, Rp) = xr.apply_ufunc(
        _runtheo_features_1d,
        ssi_win,
        input_core_dims=[["time"]],
        output_core_dims=[[]] * _n_out,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32] * _n_out,
        kwargs=dict(
            bridge_low=bridge_low,
            bridge_high=bridge_high,
            bridge_len_months=bridge_len_months,
            severity_threshold=severity_threshold,
            drought_threshold=drought_threshold,
            ttm10_threshold=ttm10_threshold,
            tts15_threshold=tts15_threshold,
            tte20_threshold=tte20_threshold,
        ),
    )

    # Compute — uses distributed client if available, else processes.
    all_vars = [D, M, PI, PeakI, DDD, TTM10, TTS15, TTE20, DRD, N, Ld, Rp]
    client = _get_dask_client()
    if client is not None:
        all_vars = [v.compute() if hasattr(v, "compute") else v for v in all_vars]
    else:
        all_vars = [
            v.compute(scheduler="processes", num_workers=_DASK_NUM_WORKERS)
            if hasattr(v, "compute") else v
            for v in all_vars
        ]
    (D, M, PI, PeakI, DDD, TTM10, TTS15, TTE20, DRD, N, Ld, Rp) = all_vars

    ds = xr.Dataset(
        dict(
            duration=D,
            magnitude=M,
            intensity=PI,
            peak_intensity=PeakI,
            ddd=DDD,
            ttm10=TTM10,
            tts15=TTS15,
            tte20=TTE20,
            drd=DRD,
            n_events=N,
            interarrival=Ld,
            return_period=Rp,
        )
    )

    ds["duration"].attrs.update(
        {"long_name": "Mean drought duration", "units": "months"})
    ds["magnitude"].attrs.update(
        {"long_name": "Mean drought magnitude", "units": "SSI-months (sum |SSI|)"})
    ds["intensity"].attrs.update(
        {"long_name": "Mean drought intensity (M/D)", "units": "-"})
    ds["peak_intensity"].attrs.update(
        {"long_name": "Mean peak intensity |min(SSI)| per event", "units": "-"})
    ds["ddd"].attrs.update(
        {"long_name": "Mean drought development duration (to first min SSI)",
         "units": "months"})
    ds["ttm10"].attrs.update(
        {"long_name": f"Mean months to first SSI <= {ttm10_threshold}",
         "units": "months"})
    ds["tts15"].attrs.update(
        {"long_name": f"Mean months to first SSI <= {tts15_threshold}",
         "units": "months"})
    ds["tte20"].attrs.update(
        {"long_name": f"Mean months to first SSI <= {tte20_threshold}",
         "units": "months"})
    ds["drd"].attrs.update(
        {"long_name": "Mean drought recovery duration (last min SSI to end)",
         "units": "months"})
    ds["n_events"].attrs.update(
        {"long_name": "Number of drought events", "units": "count"})
    ds["interarrival"].attrs.update(
        {"long_name": "Mean inter-arrival time (start-to-start)",
         "units": "months"})
    ds["return_period"].attrs.update(
        {"long_name": "Mean return period (end-to-next-start)",
         "units": "months"})

    ds.attrs.update(
        dict(
            feat_start=feat_start,
            feat_end=feat_end,
            drought_threshold=drought_threshold,
            severity_threshold=severity_threshold,
            ttm10_threshold=ttm10_threshold,
            tts15_threshold=tts15_threshold,
            tte20_threshold=tte20_threshold,
            bridge_low=bridge_low,
            bridge_high=bridge_high,
            bridge_len_months=bridge_len_months,
            method=(
                "Theory-of-runs drought feature extraction.  "
                "Bridge short [0,1] gaps (<bridge_len months).  "
                "Discard negative spells whose min > severity_threshold.  "
                "Events = contiguous SSI < drought_threshold.  "
                "DDD = first min, DRD = last min to end.  "
                "TTM10/TTS15/TTE20 = first crossing of −1.0/−1.5/−2.0.  "
                "Peak intensity = |min(SSI)|.  "
                "Ld = inter-arrival (start→start), Rp = return period (end→start)."
            ),
        )
    )
    return ds


# ---------------------------------------------------------------------
# IO wrappers: compute-or-skip for models and obs
# ---------------------------------------------------------------------


def ensure_drought_features_model(
    model: str,
    scenario: str,
    *,
    reg: Optional[Registry] = None,
    ssi_mode: str = "standalone",      # "standalone", "pooled", or "fixed"
    pool_id: Optional[str] = None,     # must match SSI folder naming
    scale: int = 3,
    ttm10_threshold: float = -1.0,
    tts15_threshold: float = -1.5,
    tte20_threshold: float = -2.0,
    ref_start: str = "2003-01",
    ref_end: str = "2019-12",
    feat_start: str = "1901-01",
    feat_end: str = "2019-12",
    bridge_low: float = 0.0,
    bridge_high: float = 1.0,
    bridge_len_months: int = 3,
    severity_threshold: float = -1.0,
    drought_threshold: float = 0.0,
    overwrite: bool = False,
    ssi_method: str = DEFAULT_SSI_METHOD,
) -> str:
    reg = reg or default_registry()
    key = f"{model}_{scenario}"

    if pool_id is None:
        pool_id = "standalone" if ssi_mode == "standalone" else "ALL_SCENARIOS"

    out_path = expected_drought_features_path(
        yaml_path=reg.yaml_path,
        is_model=True,
        key=key,
        mode=ssi_mode,
        pool_id=pool_id,
        scale=scale,
        ref_start=ref_start,
        ref_end=ref_end,
        feat_start=feat_start,
        feat_end=feat_end,
        ssi_method=ssi_method,
    )
    if (not overwrite) and os.path.exists(out_path):
        return out_path

    in_path = ssi_model_path(
        model,
        scenario,
        reg=reg,
        scale=scale,
        ref_start=ref_start,
        ref_end=ref_end,
        mode=ssi_mode,
        pool_id=pool_id,
        ssi_method=ssi_method,
    )
    if not os.path.exists(in_path):
        raise FileNotFoundError(
            f"Missing model SSI input for drought features: model={model} scenario={scenario} path={in_path}"
        )
    _time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds_in = xr.open_dataset(in_path, decode_times=_time_coder,
                            chunks={"lat": _CHUNK_LAT, "lon": _CHUNK_LON})
    if "ssi" not in ds_in:
        raise KeyError(f"'ssi' not found in {in_path}")
    ssi = ds_in["ssi"]

    ds_out = drought_features_from_ssi(
        ssi,
        feat_start=feat_start,
        feat_end=feat_end,
        bridge_low=bridge_low,
        bridge_high=bridge_high,
        bridge_len_months=bridge_len_months,
        severity_threshold=severity_threshold,
        drought_threshold=drought_threshold,
        ttm10_threshold=ttm10_threshold,
        tts15_threshold=tts15_threshold,
        tte20_threshold=tte20_threshold,
    )

    ds_out.attrs.update(
        dict(
            source_ssi=str(in_path),
            source_type="model",
            source_key=key,
            ssi_mode=ssi_mode,
            ssi_method=ssi_method,
            pool_id=pool_id,
            ssi_scale=scale,
            ssi_ref_start=ref_start,
            ssi_ref_end=ref_end,
        )
    )

    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds_out.data_vars}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_path, encoding=enc)
    return out_path


def ensure_drought_features_obs(
    obs_key: str,
    *,
    reg: Optional[Registry] = None,
    scale: int = 3,
    ttm10_threshold: float = -1.0,
    tts15_threshold: float = -1.5,
    tte20_threshold: float = -2.0,
    ref_start: str = "2003-01",
    ref_end: str = "2019-12",
    feat_start: str = "2003-01",
    feat_end: str = "2019-12",
    bridge_low: float = 0.0,
    bridge_high: float = 1.0,
    bridge_len_months: int = 3,
    severity_threshold: float = -1.0,
    drought_threshold: float = 0.0,
    overwrite: bool = False,
    ssi_method: str = DEFAULT_SSI_METHOD,
) -> str:
    reg = reg or default_registry()

    out_path = expected_drought_features_path(
        yaml_path=reg.yaml_path,
        is_model=False,
        key=obs_key,
        mode="standalone",
        pool_id="standalone",
        scale=scale,
        ref_start=ref_start,
        ref_end=ref_end,
        feat_start=feat_start,
        feat_end=feat_end,
        ssi_method=ssi_method,
    )
    if (not overwrite) and os.path.exists(out_path):
        return out_path

    in_path = ssi_obs_path(
        obs_key,
        reg=reg,
        scale=scale,
        ref_start=ref_start,
        ref_end=ref_end,
        ssi_method=ssi_method,
    )
    if not os.path.exists(in_path):
        raise FileNotFoundError(
            f"Missing observed SSI input for drought features: obs={obs_key} path={in_path}"
        )
    _time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds_in = xr.open_dataset(in_path, decode_times=_time_coder,
                            chunks={"lat": _CHUNK_LAT, "lon": _CHUNK_LON})
    if "ssi" not in ds_in:
        raise KeyError(f"'ssi' not found in {in_path}")
    ssi = ds_in["ssi"]

    ds_out = drought_features_from_ssi(
        ssi,
        feat_start=feat_start,
        feat_end=feat_end,
        bridge_low=bridge_low,
        bridge_high=bridge_high,
        bridge_len_months=bridge_len_months,
        severity_threshold=severity_threshold,
        drought_threshold=drought_threshold,
        ttm10_threshold=ttm10_threshold,
        tts15_threshold=tts15_threshold,
        tte20_threshold=tte20_threshold,
    )
    ds_out.attrs.update(
        dict(
            source_ssi=str(in_path),
            source_type="observed",
            source_key=obs_key,
            ssi_method=ssi_method,
            ssi_scale=scale,
            ssi_ref_start=ref_start,
            ssi_ref_end=ref_end,
        )
    )

    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds_out.data_vars}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_path, encoding=enc)
    return out_path
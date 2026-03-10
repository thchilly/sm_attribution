# src/sm_attribution/viz/drought_features_delta_maps.py
"""
Multi-panel delta maps for drought features across ISIMIP models.

For a single drought feature (e.g. ``duration``, ``magnitude``, …) produces a
figure with:

  Rows  = individual models + a multi-model mean row at the bottom
  Col 1 = absolute value of the feature under *scenario_base*
  Col 2 = Δ climate change   (scenario_base − scenario_counter)
  Col 3 = Δ direct human forcing (scenario_base − scenario_1901soc)
  Col 4 = Δ combined effect  (scenario_base − scenario_combined_counter)

Colorbars:
  Left  – sequential palette for absolute values  (col 1)
  Right – diverging palette centred on 0 for Δ columns (cols 2–4, shared)

Sign convention: **positive Δ = climate change / human forcing increased the
feature** (e.g. longer droughts, higher magnitude).

All colour bins and palette choices are predefined per feature in
``FEATURE_STYLE`` and can be adjusted there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import cartopy.crs as ccrs
import xarray as xr

from sm_attribution.analysis.drought_features import expected_drought_features_path
from sm_attribution.analysis.ssi import DEFAULT_SSI_METHOD
from sm_attribution.analysis.ar6_regions import get_ar6_land_regions, ar6_mean_and_field
from sm_attribution.io.registry import default_registry, project_root


# =====================================================================
# Default model list
# =====================================================================

DEFAULT_MODELS: List[str] = [
    "h08",
    "hydropy",
    "jules-w2",
    "miroc-integ-land",
    "watergap2-2e",
    "web-dhm-sg",
    "lpjml5-7-10-fire",
]

# Pretty labels for models (fallback to raw name)
MODEL_LABELS: Dict[str, str] = {
    "h08":               "H08",
    "hydropy":           "HydroPy",
    "jules-w2":          "JULES-W2",
    "miroc-integ-land":  "MIROC-INTEG-LAND",
    "watergap2-2e":      "WaterGAP2-2e",
    "web-dhm-sg":        "WEB-DHM-S/G",
    "lpjml5-7-10-fire":  "LPJmL5.7-10-fire",
}

# =====================================================================
# Per-feature style configuration
# =====================================================================


@dataclass
class FeatureStyle:
    """Colour bins and palette for one drought feature."""
    full_name: str
    units: str
    abs_bins: List[float]
    abs_colors: List[str]
    delta_bins: List[float]
    delta_colors: List[str]
    # When True, a *positive* delta means conditions got worse (drier,
    # longer droughts, etc.) and is coloured brown.  When False, a
    # positive delta means conditions got better (longer time between
    # droughts, etc.) and is coloured green/teal.
    delta_positive_is_worse: bool = True
    # AR6 regional-mean bins – same #classes but tighter range
    # (area averaging removes extremes).  Falls back to cell-level
    # bins when None.
    ar6_abs_bins: Optional[List[float]] = None
    ar6_delta_bins: Optional[List[float]] = None


# --- Absolute palettes (sequential, 6 colours each) -----------------
# Yellow → Orange → Red
_ABS_YlOrRd = [
    "#ffffb2",  # lightest
    "#fed976",
    "#feb24c",
    "#fd8d3c",
    "#f03b20",
    "#bd0026",  # darkest
]

# Light blue → Dark blue
_ABS_Blues = [
    "#eff3ff",
    "#c6dbef",
    "#9ecae1",
    "#6baed6",
    "#3182bd",
    "#08519c",
]

# Light teal → Dark teal
_ABS_Teal = [
    "#e5f5f9",
    "#ccece6",
    "#99d8c9",
    "#66c2a4",
    "#2ca25f",
    "#006d2c",
]

# Light purple → Dark purple
_ABS_Purples = [
    "#f2f0f7",
    "#dadaeb",
    "#bcbddc",
    "#9e9ac8",
    "#756bb1",
    "#54278f",
]

# --- Diverging palette for deltas (8 colours, centred on 0) ----------
# Brown (negative) → White → Teal/Green (positive)
# Use _DELTA_BrGn when positive = "better" (wetter);  positive values
# map to green/teal.  Use _DELTA_GnBr (reversed) when positive = "worse"
# (drier); positive values map to brown.
# _DELTA_BrGn = [
#     "#543005",  # strong brown  (most negative)
#     "#8c510a",
#     "#bf812d",
#     "#dfc27d",  # pale brown      ← near zero
#     "#80cdc1",  # pale teal       ← near zero
#     "#35978f",
#     "#01665e",
#     "#003c30",  # strong teal   (most positive)
# ]
_DELTA_BrGn = [
    "#8c510a",  # strong brown  (most negative)
    "#bf812d",
    "#dfc27d",
    "#f6e8c3",  # pale brown      ← near zero
    "#c7eae5",  # pale teal       ← near zero
    "#80cdc1",
    "#35978f",
    "#01665e",  # strong teal   (most positive)
]
# Reversed: green/teal (negative) → brown (positive)
_DELTA_GnBr = list(reversed(_DELTA_BrGn))

# --- Custom palettes for n_events (integer feature) -----------------
# Absolute: dark red for zero events, then blues
_ABS_NEVENTS = [
    "#bd0026",  # dark red: zero events  [0, 0.5)
    "#eff3ff",  # lightest blue          [0.5, 5)
    "#c6dbef",  #                        [5, 7)
    "#9ecae1",  #                        [7, 9)
    "#6baed6",  #                        [9, 11)
    "#3182bd",  #                        [11, 15)
    "#08519c",  # darkest blue           [15, 55)  → set_over
]
# Delta: GnBr with explicit white for exactly-zero change
_DELTA_NEVENTS = [
    "#01665e",  # strong teal   (most negative, getting better)
    "#35978f",
    "#80cdc1",
    "#c7eae5",  # pale teal
    "#ffffff",  # white: exactly zero change
    "#f6e8c3",  # pale brown
    "#dfc27d",
    "#bf812d",
    "#8c510a",  # strong brown  (most positive, getting worse)
]

# # --- LEGACY palettes (kept for reference) ---------------------------
# # Purple (negative) → White → Green (positive)
# _DELTA_PuGn = [
#     "#54278f", "#756bb1", "#9e9ac8", "#bcbddc",
#     "#d9f0d3", "#7fbf7b", "#1b7837", "#00441b",
# ]
# # Red (negative) → White → Blue (positive)
# _DELTA_RdBu = [
#     "#b2182b", "#d6604d", "#f4a582", "#f7f7f7",
#     "#92c5de", "#4393c3", "#2166ac", "#053061",
# ]

# Bins calibrated from actual data (7 models, obsclim_histsoc,
# ref 1950-2019, feat 1950-2019, monthwise_ecdf).  See notebook for
# percentile summaries used to set these.
#
# Absolute values (obsclim_histsoc):
#   duration   med≈24  p75≈34   p95≈108  max≈495
#   magnitude  med≈17  p75≈27   p95≈101  max≈332
#   intensity  med≈0.7 p75≈0.78 p95≈0.94 max≈2.4
#   ddd        med≈12  p75≈17   p95≈52   max≈439
#   tts15      med≈10  p75≈14   p95≈40   max≈396
#   drd        med≈12  p75≈17   p95≈47   max≈453
#   n_events   med≈0   p75≈1    p95≈25   max≈51
#
# Deltas (obsclim_histsoc − counterclim_histsoc):
#   duration   p5≈-18  med≈0   p95≈18
#   magnitude  p5≈-16  med≈0   p95≈15
#   intensity  p5≈-0.13 med≈0  p95≈0.11
#   ddd        p5≈-11  med≈0   p95≈10
#   tts15      p5≈-10  med≈0   p95≈9
#   drd        p5≈-11  med≈0   p95≈12
#   n_events   p5≈-2   med≈0   p95≈2

FEATURE_STYLE: Dict[str, FeatureStyle] = {
    "duration": FeatureStyle(
        full_name="Mean Drought Duration",
        units="months",
        #abs_bins=[0, 10, 20, 35, 60, 110, 500],
        abs_bins=[0, 12, 14, 18, 22, 30, 500],
        abs_colors=_ABS_YlOrRd,
        delta_bins=[-100, -10, -3, -1, 0, 1, 3, 10, 100],
        delta_colors=_DELTA_GnBr,
        delta_positive_is_worse=True,
        ar6_abs_bins=[0, 12, 14, 16, 18, 20, 120],
        ar6_delta_bins=[-30, -4, -2, -1, 0, 1, 2, 4, 30],
    ),
    "magnitude": FeatureStyle(
        full_name="Mean Drought Magnitude",
        units="SSI-months (Σ|SSI|)",
        abs_bins=[0, 8, 12, 16, 22, 34, 350],
        abs_colors=_ABS_YlOrRd,
        delta_bins=[-100, -6, -3, -1, 0, 1, 3, 6, 100],
        delta_colors=_DELTA_GnBr,
        delta_positive_is_worse=True,
        ar6_abs_bins=[0, 8, 10, 12, 14, 18, 100],
        ar6_delta_bins=[-30, -5, -2, -1, 0, 1, 2, 5, 30],
    ),
    "intensity": FeatureStyle(
        full_name="Mean Drought Intensity",
        units="–",
        abs_bins=[0, 0.78, 0.82, 0.86, 0.9, 0.94, 2.5],
        abs_colors=_ABS_Purples,
        delta_bins=[-0.5, -0.15, -0.07, -0.05, 0, 0.05, 0.07, 0.15, 0.5],
        delta_colors=_DELTA_GnBr,
        delta_positive_is_worse=True,
        ar6_abs_bins=[0, 0.80, 0.82, 0.84, 0.86, 0.88, 1.2],
        ar6_delta_bins=[-0.15, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1, 0.15],
    ),
    "ddd": FeatureStyle(
        full_name="Mean Drought Development Duration",
        units="months",
        #abs_bins=[0, 5, 10, 18, 30, 55, 450],
        abs_bins=[0, 3, 6, 8, 10, 12, 450],  # aris
        abs_colors=_ABS_YlOrRd,
        #delta_bins=[-50, -12, -3, -1, 0, 1, 3, 12, 50],
        delta_bins=[-50, -4, -2, -1, 0, 1, 2, 4, 50],
        delta_colors=_DELTA_GnBr,
        delta_positive_is_worse=True,
        ar6_abs_bins=[0, 6, 7, 8, 9, 10, 80],
        ar6_delta_bins=[-15, -2, -1, -0.5, 0, 0.5, 1, 2, 15],

    ),
    "ttm10": FeatureStyle(
        full_name="Time-to-Moderate TTM10 [SSI ≤ −1.0]",
        units="months",
        abs_bins=[0, 4, 6, 8, 9, 10, 450],
        abs_colors=_ABS_YlOrRd,
        delta_bins=[-50, -3, -2, -1, 0, 1, 2, 3, 50],
        delta_colors=_DELTA_GnBr,
        delta_positive_is_worse=True,
        ar6_abs_bins=[0, 5, 6, 7, 8, 10, 80],
        ar6_delta_bins=[-15, -2, -1, -0.5, 0, 0.5, 1, 2, 15],
    ),
    "tts15": FeatureStyle(
        full_name="Time-to-Severe TTS15 [SSI ≤ −1.5]",
        units="months",
        abs_bins=[0, 4, 6, 8, 9, 10, 450],
        abs_colors=_ABS_YlOrRd,
        delta_bins=[-50, -3, -2, -1, 0, 1, 2, 3, 50],
        delta_colors=_DELTA_GnBr,
        delta_positive_is_worse=True,
        ar6_abs_bins=[0, 5, 6, 7, 8, 10, 80],
        ar6_delta_bins=[-15, -2, -1, -0.5, 0, 0.5, 1, 2, 15],
    ),
    "tte20": FeatureStyle(
        full_name="Time-to-Extreme TTE20 [SSI ≤ −2.0]",
        units="months",
        abs_bins=[0, 4, 6, 8, 9, 10, 450],
        abs_colors=_ABS_YlOrRd,
        delta_bins=[-50, -3, -2, -1, 0, 1, 2, 3, 50],
        delta_colors=_DELTA_GnBr,
        delta_positive_is_worse=True,
        ar6_abs_bins=[0, 5, 6, 7, 8, 10, 80],
        ar6_delta_bins=[-15, -2, -1, -0.5, 0, 0.5, 1, 2, 15],
    ),
    "drd": FeatureStyle(
        full_name="Mean Drought Recovery Duration",
        units="months",
        abs_bins=[0, 3, 6, 8, 10, 12, 450],  # aris
        abs_colors=_ABS_YlOrRd,
        #delta_bins=[-50, -12, -3, -1, 0, 1, 3, 12, 50],
        delta_bins=[-50, -4, -2, -1, 0, 1, 2, 4, 50],
        delta_colors=_DELTA_GnBr,
        delta_positive_is_worse=True,
        ar6_abs_bins=[0, 6, 7, 8, 9, 10, 80],
        ar6_delta_bins=[-15, -2, -1, -0.5, 0, 0.5, 1, 2, 15],
    ),
    "n_events": FeatureStyle(
        full_name="Number of Drought Events",
        units="count",
        abs_bins=[0, 0.5, 5, 7, 9, 11, 15, 55],
        abs_colors=_ABS_NEVENTS,
        delta_bins=[-10, -3, -2, -1, -0.0001, 0.0001, 1, 2, 3, 10],
        delta_colors=_DELTA_NEVENTS,
        delta_positive_is_worse=True,
        ar6_abs_bins=[0, 0.5, 5, 6, 7, 8, 10, 30],
        ar6_delta_bins=[-10, -3, -2, -1, -0.01, 0.01, 1, 2, 3, 10],
    ),
    "peak_intensity": FeatureStyle(
        full_name="Mean Peak Intensity (|min SSI|)",
        units="–",
        abs_bins=[1.5, 1.65, 1.7, 1.75, 1.8, 1.85, 1000],
        abs_colors=_ABS_Purples,
        delta_bins=[-0.5, -0.15, -0.07, -0.05, 0, 0.05, 0.07, 0.15, 0.5],
        delta_colors=_DELTA_GnBr,
        delta_positive_is_worse=True,
        ar6_abs_bins=[1.5, 1.65, 1.75, 1.775, 1.8, 1.85, 1000],
        ar6_delta_bins=[-0.15, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1, 0.15],
    ),
    "interarrival": FeatureStyle(
        full_name="Mean Inter-Arrival Time",
        units="months",
        abs_bins=[0, 20, 40, 60, 100, 160, 500],
        abs_colors=_ABS_Blues,
        delta_bins=[-80, -20, -5, -1, 0, 1, 5, 20, 80],
        delta_colors=_DELTA_BrGn,
        delta_positive_is_worse=False,
        ar6_abs_bins=[0, 20, 35, 50, 70, 100, 200],
        ar6_delta_bins=[-30, -10, -3, -0.5, 0, 0.5, 3, 10, 30],
    ),
    "return_period": FeatureStyle(
        full_name="Mean Return Period",
        units="months",
        abs_bins=[0, 5, 10, 12, 15, 25, 500],
        abs_colors=_ABS_Blues,
        delta_bins=[-80, -10, -5, -1, 0, 1, 5, 10, 80],
        delta_colors=_DELTA_BrGn,
        delta_positive_is_worse=False,
        ar6_abs_bins=[0, 5, 10, 12, 15, 20, 500],
        ar6_delta_bins=[-30, -6, -3, -1, 0, 1, 3, 6, 30],
    ),
}

# The set of all recognised feature variable names
ALL_FEATURES = list(FEATURE_STYLE.keys())

# Default projection for global maps
DEFAULT_PROJ = ccrs.EckertIV()


# =====================================================================
# Internal helpers
# =====================================================================


def _classified_norm_and_cmap(bins, colors, extend="neither"):
    """Return (ListedColormap, BoundaryNorm) from class bins and colours.

    Parameters
    ----------
    extend : ``"neither"`` | ``"both"`` | ``"max"`` | ``"min"``
        Which sentinel edges to strip from the norm.  Stripped edges
        become ``set_under`` / ``set_over`` colours, rendered as
        extend-arrows by the colorbar.

        * ``"both"`` – strip first *and* last bin; arrows on both ends.
        * ``"max"``  – strip only the last bin; right arrow only.
          (Keeps 0 as the left boundary for absolute maps.)
        * ``"min"``  – strip only the first bin; left arrow only.
        * ``"neither"`` – keep all bins, ``clip=True``.
    """
    bins = list(bins)
    colors = list(colors)
    n_intervals = len(bins) - 1
    if len(colors) < n_intervals:
        colors = colors + [colors[-1]] * (n_intervals - len(colors))
    colors = colors[:n_intervals]

    if extend == "both" and len(bins) >= 4:
        inner_bins = bins[1:-1]
        cmap = mcolors.ListedColormap(colors[1:-1])
        cmap.set_under(colors[0])
        cmap.set_over(colors[-1])
        norm = mcolors.BoundaryNorm(inner_bins, ncolors=cmap.N)
    elif extend == "max" and len(bins) >= 3:
        inner_bins = bins[:-1]
        cmap = mcolors.ListedColormap(colors[:-1])
        cmap.set_over(colors[-1])
        norm = mcolors.BoundaryNorm(inner_bins, ncolors=cmap.N)
    elif extend == "min" and len(bins) >= 3:
        inner_bins = bins[1:]
        cmap = mcolors.ListedColormap(colors[1:])
        cmap.set_under(colors[0])
        norm = mcolors.BoundaryNorm(inner_bins, ncolors=cmap.N)
    else:
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(bins, ncolors=cmap.N, clip=True)

    return cmap, norm


def _plot_global_map(
    ax,
    da: xr.DataArray,
    *,
    bins: Iterable[float],
    colors: Iterable[str],
    title: str,
    draw_ar6_outlines: bool = False,
    extend: str = "neither",
):
    """Classified pcolormesh on a global GeoAxes."""
    cmap, norm = _classified_norm_and_cmap(bins, colors, extend=extend)

    ax.set_global()
    ax.coastlines(linewidth=0.4)
    ax.gridlines(draw_labels=False, linewidth=0.2, color="lightgray", alpha=0.5)

    mesh = ax.pcolormesh(
        da["lon"],
        da["lat"],
        da,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading="auto",
    )

    if draw_ar6_outlines:
        regions = get_ar6_land_regions()
        regions.plot(
            ax=ax,
            add_label=False,
            add_ocean=False,
            add_land=False,
            line_kws={"linewidth": 0.4, "color": "black"},
            coastline_kws={"linewidth": 0.0},
        )

    ax.set_title(title, fontsize=8)
    return mesh


def _load_drought_feature(
    model: str,
    scenario: str,
    feature: str,
    *,
    mode: str,
    pool_id: str,
    scale: int,
    ref_start: str,
    ref_end: str,
    feat_start: str,
    feat_end: str,
    ssi_method: str = DEFAULT_SSI_METHOD,
) -> xr.DataArray:
    """
    Load a single drought-feature variable from the precomputed NetCDF.

    Returns the 2-D (lat, lon) DataArray for *feature*.
    """
    reg = default_registry()
    key = f"{model}_{scenario}"
    path = expected_drought_features_path(
        yaml_path=reg.yaml_path,
        is_model=True,
        key=key,
        mode=mode,
        pool_id=pool_id,
        scale=scale,
        ref_start=ref_start,
        ref_end=ref_end,
        feat_start=feat_start,
        feat_end=feat_end,
        ssi_method=ssi_method,
    )
    # Resolve relative paths (e.g. "data/...") against the project root
    # so the lookup works regardless of the working directory (e.g. from
    # notebooks/).
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    if not p.exists():
        raise FileNotFoundError(
            f"Drought-feature file not found:\n  {p}\n"
            f"  model={model}, scenario={scenario}, ssi_method={ssi_method}, "
            f"mode={mode}, pool_id={pool_id}, ref={ref_start}–{ref_end}, "
            f"feat={feat_start}–{feat_end}"
        )
    ds = xr.open_dataset(p)
    if feature not in ds:
        raise KeyError(
            f"Variable '{feature}' not in {p}. "
            f"Available: {list(ds.data_vars)}"
        )
    da = ds[feature]
    return da


def _load_all_models(
    models: List[str],
    scenario: str,
    feature: str,
    **load_kw,
) -> xr.DataArray:
    """Load *feature* for every model and stack along a ``model`` dim."""
    arrays = []
    for m in models:
        da = _load_drought_feature(m, scenario, feature, **load_kw)
        da = da.expand_dims(model=[m])
        arrays.append(da)
    return xr.concat(arrays, dim="model")


# Minimum number of non-NaN models required per pixel for the
# multi-model median to be valid.  4 out of 7 ⇒ majority required.
_MIN_VALID_MODELS = 4


def _load_multimodel_median(
    models: List[str],
    scenario: str,
    feature: str,
    **load_kw,
) -> xr.DataArray:
    """Multi-model **median** with a 4/7 min-valid threshold.

    Pixels where fewer than ``_MIN_VALID_MODELS`` models have data
    are set to NaN.
    """
    stacked = _load_all_models(models, scenario, feature, **load_kw)
    n_valid = stacked.count(dim="model")
    med = stacked.median(dim="model")
    return med.where(n_valid >= _MIN_VALID_MODELS)


def _load_model_agreement(
    models: List[str],
    scenario_base: str,
    scenario_counter: str,
    feature: str,
    **load_kw,
) -> xr.DataArray:
    """Compute per-pixel sign agreement across models for a delta.

    Returns a 2-D DataArray with integer classes:
      2  = all valid models agree positive  ("strongly agree +")
      1  = 5–6 out of n_valid agree positive ("agree +")
      0  = no consensus (≤ 4 agree either way)
     -1  = 5–6 out of n_valid agree negative ("agree −")
     -2  = all valid models agree negative  ("strongly agree −")
     NaN = fewer than ``_MIN_VALID_MODELS`` valid models
    """
    stacked_base = _load_all_models(models, scenario_base, feature, **load_kw)
    stacked_counter = _load_all_models(models, scenario_counter, feature, **load_kw)
    delta = stacked_base - stacked_counter  # (model, lat, lon)

    n_valid = delta.count(dim="model")
    n_pos = (delta > 0).sum(dim="model")
    n_neg = (delta < 0).sum(dim="model")

    # Start with 0 = no consensus
    agree = xr.zeros_like(n_valid, dtype=float)

    # Positive agreement
    agree = agree.where(~(n_pos == n_valid), 2.0)      # all agree +
    agree = agree.where(~((n_pos >= 5) & (n_pos < n_valid)), 1.0)  # 5-6 agree +

    # Negative agreement
    agree = agree.where(~(n_neg == n_valid), -2.0)      # all agree −
    agree = agree.where(~((n_neg >= 5) & (n_neg < n_valid)), -1.0) # 5-6 agree −

    # Mask pixels with too few valid models
    agree = agree.where(n_valid >= _MIN_VALID_MODELS)

    agree.name = f"agreement_{feature}"
    return agree


# Fixed style for agreement maps
_AGREE_BINS = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]  # 5 intervals

# Impact-aware agreement colours (brown = drier/worse, teal = wetter/better).
# _POS_WORSE: positive agreement → brown (worse); negative agreement → teal
_AGREE_COLORS_POS_WORSE = [
    "#01665e",  # −strongly agree → strong teal  (getting better)
    "#80cdc1",  # −agree          → light teal
    "#ffffff",  # no consensus    → white
    "#bf812d",  # +agree          → light brown
    "#8c510a",  # +strongly agree → strong brown (getting worse)
]
# _POS_BETTER: positive agreement → teal (better); negative agreement → brown
_AGREE_COLORS_POS_BETTER = [
    "#8c510a",  # −strongly agree → strong brown (getting worse)
    "#bf812d",  # −agree          → light brown
    "#ffffff",  # no consensus    → white
    "#80cdc1",  # +agree          → light teal
    "#01665e",  # +strongly agree → strong teal  (getting better)
]

_AGREE_LABELS = [
    "100% −agree (7/7)",
    "≥70% −agree (≥5/7)",
    "no strong agreement",
    "≥70% +agree (≥5/7)",
    "100% +agree (7/7)",
]


def _agree_colors_for(style: FeatureStyle) -> list:
    """Return the agreement palette matching a feature's impact semantics."""
    if style.delta_positive_is_worse:
        return _AGREE_COLORS_POS_WORSE
    return _AGREE_COLORS_POS_BETTER


def _ar6_field_landonly(
    da: xr.DataArray,
    min_valid: int = 10,
    agg: str = "mean",
) -> xr.DataArray:
    """AR6 regional aggregate, painted only on **land** pixels.

    Uses :func:`ar6_mean_and_field` for the regional aggregation,
    then masks the returned field to only those pixels where the
    original data *da* was finite (≈ land mask).

    Parameters
    ----------
    agg : {"mean", "median"}
        Passed through to :func:`ar6_mean_and_field`.
    """
    land_mask = np.isfinite(da)
    _, reg_field = ar6_mean_and_field(da, min_valid=min_valid, agg=agg)
    return reg_field.where(land_mask)


# =====================================================================
# Public API
# =====================================================================


def plot_drought_feature_delta_map(
    feature: str,
    *,
    models: Optional[List[str]] = None,
    scenario_base: str = "obsclim_histsoc",
    scenario_counter: str = "counterclim_histsoc",
    scenario_1901soc: str = "obsclim_1901soc",
    scenario_combined_counter: str = "counterclim_1901soc",
    mode: str = "standalone",
    pool_id: str = "standalone",
    scale: int = 3,
    ref_start: str = "1950-01",
    ref_end: str = "2019-12",
    feat_start: str = "1950-01",
    feat_end: str = "2019-12",
    ssi_method: str = DEFAULT_SSI_METHOD,
    proj=DEFAULT_PROJ,
    figsize: Optional[tuple] = None,
    hspace: float = 0.30,
    wspace: float = 0.02,
) -> plt.Figure:
    """
    Multi-panel delta map for a single drought feature.

    Parameters
    ----------
    feature : str
        Variable name in the drought-feature NetCDF.
        One of: ``duration``, ``magnitude``, ``intensity``,
        ``ddd``, ``tts15``, ``drd``, ``n_events``.
    models : list of str, optional
        Models to plot (one row each).  Defaults to :data:`DEFAULT_MODELS`.
    scenario_base : str
        Factual scenario (default ``obsclim_histsoc`` = FC·HS).
    scenario_counter : str
        Counter-factual climate (default ``counterclim_histsoc`` = CfC·HS).
    scenario_1901soc : str
        Pre-industrial land-use (default ``obsclim_1901soc`` = FC·PS).
    scenario_combined_counter : str
        Both counter-factual (default ``counterclim_1901soc`` = CfC·PS).
    mode, pool_id, scale, ref_start, ref_end, feat_start, feat_end, ssi_method
        Forwarded to :func:`expected_drought_features_path` for file look-up.
    proj : cartopy.crs.Projection
        Map projection (default EckertIV).
    figsize : tuple, optional
        Figure size ``(width, height)``.  Auto-computed from row count
        if *None*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if models is None:
        models = list(DEFAULT_MODELS)

    if feature not in FEATURE_STYLE:
        raise ValueError(
            f"Unknown feature '{feature}'. "
            f"Must be one of {ALL_FEATURES}"
        )
    style = FEATURE_STYLE[feature]

    nrows = len(models) + 1  # +1 for multi-model mean
    ncols = 4

    if figsize is None:
        figsize = (20, 2.8 * nrows + 1.5)

    load_kw = dict(
        mode=mode,
        pool_id=pool_id,
        scale=scale,
        ref_start=ref_start,
        ref_end=ref_end,
        feat_start=feat_start,
        feat_end=feat_end,
        ssi_method=ssi_method,
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        subplot_kw={"projection": proj},
        constrained_layout=False,
    )

    if nrows == 1:
        axes = axes.reshape(1, -1)

    # For colorbar reference
    mesh_abs = None
    mesh_delta = None

    ref_yr = f"{ref_start[:4]}–{ref_end[:4]}"
    feat_yr = f"{feat_start[:4]}–{feat_end[:4]}"

    # ------------------------------------------------------------------
    # Plot individual model rows
    # ------------------------------------------------------------------
    for irow, model in enumerate(models):
        label = MODEL_LABELS.get(model, model)

        base = _load_drought_feature(
            model, scenario_base, feature, **load_kw)
        counter = _load_drought_feature(
            model, scenario_counter, feature, **load_kw)
        soc1901 = _load_drought_feature(
            model, scenario_1901soc, feature, **load_kw)
        combined = _load_drought_feature(
            model, scenario_combined_counter, feature, **load_kw)

        # Col 1: absolute value (base scenario)
        m1 = _plot_global_map(
            axes[irow, 0], base,
            bins=style.abs_bins,
            colors=style.abs_colors,
            title=f"{label}\n{style.full_name} [FC·HS]",
            extend="max",
        )
        if mesh_abs is None:
            mesh_abs = m1

        # Col 2: Δ climate change  (FC·HS − CfC·HS)
        delta_cc = base - counter
        m2 = _plot_global_map(
            axes[irow, 1], delta_cc,
            bins=style.delta_bins,
            colors=style.delta_colors,
            title=f"{label}\nΔ Climate change [FC·HS − CfC·HS]",
            extend="both",
        )
        if mesh_delta is None:
            mesh_delta = m2

        # Col 3: Δ direct human forcing  (FC·HS − FC·PS)
        delta_hf = base - soc1901
        _plot_global_map(
            axes[irow, 2], delta_hf,
            bins=style.delta_bins,
            colors=style.delta_colors,
            title=f"{label}\nΔ Direct human forcing [FC·HS − FC·PS]",
            extend="both",
        )

        # Col 4: Δ combined  (FC·HS − CfC·PS)
        delta_cb = base - combined
        _plot_global_map(
            axes[irow, 3], delta_cb,
            bins=style.delta_bins,
            colors=style.delta_colors,
            title=f"{label}\nΔ Combined effect [FC·HS − CfC·PS]",
            extend="both",
        )

    # ------------------------------------------------------------------
    # Multi-model median row (last row)
    # ------------------------------------------------------------------
    irow_mm = len(models)

    mm_base = _load_multimodel_median(
        models, scenario_base, feature, **load_kw)
    mm_counter = _load_multimodel_median(
        models, scenario_counter, feature, **load_kw)
    mm_1901 = _load_multimodel_median(
        models, scenario_1901soc, feature, **load_kw)
    mm_combined = _load_multimodel_median(
        models, scenario_combined_counter, feature, **load_kw)

    n_mod = len(models)
    mm_label = f"Multi-model median (n={n_mod})"

    m_abs_mm = _plot_global_map(
        axes[irow_mm, 0], mm_base,
        bins=style.abs_bins,
        colors=style.abs_colors,
        title=f"{mm_label}\n{style.full_name} [FC·HS]",
        extend="max",
    )
    if mesh_abs is None:
        mesh_abs = m_abs_mm

    delta_cc_mm = mm_base - mm_counter
    m_d_mm = _plot_global_map(
        axes[irow_mm, 1], delta_cc_mm,
        bins=style.delta_bins,
        colors=style.delta_colors,
        title=f"{mm_label}\nΔ Climate change [FC·HS − CfC·HS]",
        extend="both",
    )
    if mesh_delta is None:
        mesh_delta = m_d_mm

    delta_hf_mm = mm_base - mm_1901
    _plot_global_map(
        axes[irow_mm, 2], delta_hf_mm,
        bins=style.delta_bins,
        colors=style.delta_colors,
        title=f"{mm_label}\nΔ Direct human forcing [FC·HS − FC·PS]",
        extend="both",
    )

    delta_cb_mm = mm_base - mm_combined
    _plot_global_map(
        axes[irow_mm, 3], delta_cb_mm,
        bins=style.delta_bins,
        colors=style.delta_colors,
        title=f"{mm_label}\nΔ Combined effect [FC·HS − CfC·PS]",
        extend="both",
    )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    fig.subplots_adjust(
        hspace=hspace,
        wspace=wspace,
        top=0.93,
        bottom=0.08,
        left=0.03,
        right=0.97,
    )

    # ------------------------------------------------------------------
    # Colorbars  (extend arrows replace sentinel edge labels)
    # ------------------------------------------------------------------
    # Left: absolute values (col 1) — extend="max" keeps 0, arrow on right
    cax_abs = fig.add_axes([0.04, 0.05, 0.18, 0.012])
    cbar_abs = fig.colorbar(mesh_abs, cax=cax_abs, orientation="horizontal",
                            extend="max", extendfrac=0.15, extendrect=True)
    cbar_abs.set_label(f"{style.full_name} ({style.units})")

    # Right: deltas (cols 2–4, shared) — extend="both"
    cax_delta = fig.add_axes([0.30, 0.05, 0.60, 0.012])
    cbar_delta = fig.colorbar(mesh_delta, cax=cax_delta, orientation="horizontal",
                              extend="both", extendfrac=0.15, extendrect=True)
    cbar_delta.set_label(
        f"Δ {style.full_name} ({style.units})  "
        f"[ref {ref_yr} · feat {feat_yr}]"
    )

    return fig


# =====================================================================
# Multi-feature summary (multi-model mean per row)
# =====================================================================


def plot_drought_feature_mm_panel(
    feature: str,
    *,
    models: Optional[List[str]] = None,
    scenario_base: str = "obsclim_histsoc",
    scenario_counter: str = "counterclim_histsoc",
    scenario_1901soc: str = "obsclim_1901soc",
    scenario_combined_counter: str = "counterclim_1901soc",
    mode: str = "standalone",
    pool_id: str = "standalone",
    scale: int = 3,
    ref_start: str = "1950-01",
    ref_end: str = "2019-12",
    feat_start: str = "1950-01",
    feat_end: str = "2019-12",
    ssi_method: str = DEFAULT_SSI_METHOD,
    proj=DEFAULT_PROJ,
    figsize: Optional[tuple] = None,
    hspace: float = 0.55,
    wspace: float = 0.005,
    ar6_agg: str = "mean",
) -> plt.Figure:
    """
    Single-feature MM-median panel with AR6 aggregation and model agreement.

    Produces a **3-row × 4-column** figure for *one* drought feature:

    1. **Cell-level** multi-model median (absolute + 3 deltas).
       Uses the standard ``abs_bins`` / ``delta_bins`` colour scale.
    2. **AR6-aggregated** — regional mean or median, land-only
       colouring, outlines.  Uses ``ar6_abs_bins`` / ``ar6_delta_bins``
       (tighter range because area averaging removes extremes).
    3. **Model sign-agreement** — 5-class map for cols 2–4; col 1
       repeats the MM median absolute.

    Each row has its own pair of colorbars (absolute + delta / agreement).

    Parameters
    ----------
    feature : str
        One of :data:`ALL_FEATURES`.
    models, scenario_*, mode, pool_id, scale, ref_start, ref_end,
    feat_start, feat_end, ssi_method, proj :
        Same as :func:`plot_drought_feature_delta_map`.
    figsize : tuple, optional
        Figure size.  Defaults to ``(20, 9.5)``.
    hspace, wspace : float
        Subplot spacing.
    ar6_agg : {"mean", "median"}
        Aggregation method for AR6 regional values (Row 1).
        Default ``"mean"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if models is None:
        models = list(DEFAULT_MODELS)
    if feature not in FEATURE_STYLE:
        raise ValueError(
            f"Unknown feature '{feature}'. Must be one of {ALL_FEATURES}"
        )

    style = FEATURE_STYLE[feature]
    n_mod = len(models)
    nrows, ncols = 3, 4

    if figsize is None:
        figsize = (20, 9.5)

    load_kw = dict(
        mode=mode, pool_id=pool_id, scale=scale,
        ref_start=ref_start, ref_end=ref_end,
        feat_start=feat_start, feat_end=feat_end,
        ssi_method=ssi_method,
    )

    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize,
        subplot_kw={"projection": proj},
        constrained_layout=False,
    )

    mm_label = f"MM median (n={n_mod})"

    # Resolve AR6 bins (fall back to cell-level when not defined)
    ar6_abs_bins   = style.ar6_abs_bins   or style.abs_bins
    ar6_delta_bins = style.ar6_delta_bins or style.delta_bins

    # ---------- Load data -------------------------------------------------
    mm_base = _load_multimodel_median(
        models, scenario_base, feature, **load_kw)
    mm_counter = _load_multimodel_median(
        models, scenario_counter, feature, **load_kw)
    mm_1901 = _load_multimodel_median(
        models, scenario_1901soc, feature, **load_kw)
    mm_combined = _load_multimodel_median(
        models, scenario_combined_counter, feature, **load_kw)

    delta_cc = mm_base - mm_counter
    delta_hf = mm_base - mm_1901
    delta_cb = mm_base - mm_combined

    # ==================================================================
    # Row 0  –  Cell-level MM median
    # ==================================================================
    m_abs0 = _plot_global_map(
        axes[0, 0], mm_base,
        bins=style.abs_bins, colors=style.abs_colors,
        title=f"Cell-level  {mm_label}\n{style.full_name} [FC·HS]",
        extend="max",
    )
    m_delta0 = _plot_global_map(
        axes[0, 1], delta_cc,
        bins=style.delta_bins, colors=style.delta_colors,
        title=f"Cell-level  Δ Climate\n[FC·HS − CfC·HS]",
        extend="both",
    )
    _plot_global_map(
        axes[0, 2], delta_hf,
        bins=style.delta_bins, colors=style.delta_colors,
        title=f"Cell-level  Δ Human\n[FC·HS − FC·PS]",
        extend="both",
    )
    _plot_global_map(
        axes[0, 3], delta_cb,
        bins=style.delta_bins, colors=style.delta_colors,
        title=f"Cell-level  Δ Combined\n[FC·HS − CfC·PS]",
        extend="both",
    )

    # ==================================================================
    # Row 1  –  AR6-aggregated (mean/median within regions, land-only)
    # ==================================================================
    ar6_abs = _ar6_field_landonly(mm_base, agg=ar6_agg)
    # For n_events, aggregate each scenario first then subtract
    # to avoid cancellation of small integer deltas.
    if feature == "n_events":
        ar6_base_f    = _ar6_field_landonly(mm_base, agg=ar6_agg)
        ar6_counter_f = _ar6_field_landonly(mm_counter, agg=ar6_agg)
        ar6_1901_f    = _ar6_field_landonly(mm_1901, agg=ar6_agg)
        ar6_combined_f = _ar6_field_landonly(mm_combined, agg=ar6_agg)
        ar6_cc = ar6_base_f - ar6_counter_f
        ar6_hf = ar6_base_f - ar6_1901_f
        ar6_cb = ar6_base_f - ar6_combined_f
    else:
        ar6_cc  = _ar6_field_landonly(delta_cc, agg=ar6_agg)
        ar6_hf  = _ar6_field_landonly(delta_hf, agg=ar6_agg)
        ar6_cb  = _ar6_field_landonly(delta_cb, agg=ar6_agg)

    ar6_lbl = f"AR6 regional {ar6_agg}"

    m_abs1 = _plot_global_map(
        axes[1, 0], ar6_abs,
        bins=ar6_abs_bins, colors=style.abs_colors,
        title=f"{ar6_lbl}\n{style.full_name} [FC·HS]",
        draw_ar6_outlines=True,
        extend="max",
    )
    m_delta1 = _plot_global_map(
        axes[1, 1], ar6_cc,
        bins=ar6_delta_bins, colors=style.delta_colors,
        title=f"{ar6_lbl}  Δ Climate\n[FC·HS − CfC·HS]",
        draw_ar6_outlines=True,
        extend="both",
    )
    _plot_global_map(
        axes[1, 2], ar6_hf,
        bins=ar6_delta_bins, colors=style.delta_colors,
        title=f"{ar6_lbl}  Δ Human\n[FC·HS − FC·PS]",
        draw_ar6_outlines=True,
        extend="both",
    )
    _plot_global_map(
        axes[1, 3], ar6_cb,
        bins=ar6_delta_bins, colors=style.delta_colors,
        title=f"{ar6_lbl}  Δ Combined\n[FC·HS − CfC·PS]",
        draw_ar6_outlines=True,
        extend="both",
    )

    # ==================================================================
    # Row 2  –  Model sign-agreement
    # ==================================================================
    # Col 1: repeat MM median absolute for visual reference
    _plot_global_map(
        axes[2, 0], mm_base,
        bins=style.abs_bins, colors=style.abs_colors,
        title=f"{mm_label}\n{style.full_name} [FC·HS]",
        extend="max",
    )

    agree_cc = _load_model_agreement(
        models, scenario_base, scenario_counter, feature, **load_kw)
    m_agree = _plot_global_map(
        axes[2, 1], agree_cc,
        bins=_AGREE_BINS, colors=_agree_colors_for(style),
        title=f"Agreement: Δ Climate\n{style.full_name}",
    )

    agree_hf = _load_model_agreement(
        models, scenario_base, scenario_1901soc, feature, **load_kw)
    _plot_global_map(
        axes[2, 2], agree_hf,
        bins=_AGREE_BINS, colors=_agree_colors_for(style),
        title=f"Agreement: Δ Human\n{style.full_name}",
    )

    agree_cb = _load_model_agreement(
        models, scenario_base, scenario_combined_counter, feature, **load_kw)
    _plot_global_map(
        axes[2, 3], agree_cb,
        bins=_AGREE_BINS, colors=_agree_colors_for(style),
        title=f"Agreement: Δ Combined\n{style.full_name}",
    )

    # ==================================================================
    # Layout
    # ==================================================================
    fig.subplots_adjust(
        hspace=hspace, wspace=wspace,
        top=0.92, bottom=0.08, left=0.02, right=0.98,
    )

    # ==================================================================
    # Per-row colorbars
    # ==================================================================
    def _add_row_colorbars(row_idx, mesh_abs, mesh_rhs, label_abs, label_rhs,
                            rhs_ticks=None, rhs_ticklabels=None):
        """Place *abs* cbar under col 1 and *rhs* cbar under cols 2–4.

        The mesh objects already carry the correct norm (with sentinel
        edges stripped via ``extend``), so ticks are placed automatically
        at the norm boundaries.

        For the agreement row, *rhs_ticks* / *rhs_ticklabels* override
        the auto-ticks and ``extend="neither"`` is used.
        """
        row_axes = axes[row_idx, :]
        row_bot = min(ax.get_position().y0 for ax in row_axes)
        c1l = row_axes[0].get_position().x0
        c1r = row_axes[0].get_position().x1
        c2l = row_axes[1].get_position().x0
        c4r = row_axes[3].get_position().x1

        cb_y = row_bot - 0.022
        cb_h = 0.008

        # Absolute (below col 1) — extend="max" (arrow on right only)
        cax_a = fig.add_axes([c1l, cb_y, c1r - c1l, cb_h])
        cb_a = fig.colorbar(mesh_abs, cax=cax_a, orientation="horizontal",
                            extend="max", extendfrac=0.15, extendrect=True)
        cb_a.set_label(label_abs, fontsize=7)
        cb_a.ax.tick_params(labelsize=6)

        # RHS (below cols 2-4, 55 % of span, centred)
        rhs_extend = "neither" if rhs_ticks is not None else "both"
        span = c4r - c2l
        w = span * 0.55
        x = c2l + (span - w) / 2
        cax_r = fig.add_axes([x, cb_y, w, cb_h])
        cb_r = fig.colorbar(mesh_rhs, cax=cax_r, orientation="horizontal",
                            extend=rhs_extend, extendfrac=0.15, extendrect=True)
        cb_r.set_label(label_rhs, fontsize=7)
        cb_r.ax.tick_params(labelsize=6)
        if rhs_ticks is not None:
            cb_r.set_ticks(rhs_ticks)
            if rhs_ticklabels is not None:
                cb_r.set_ticklabels(rhs_ticklabels)

    # Row 0: cell-level
    _add_row_colorbars(
        0, m_abs0, m_delta0,
        f"{style.full_name} ({style.units})",
        f"Δ {style.full_name} ({style.units})",
    )

    # Row 1: AR6
    _add_row_colorbars(
        1, m_abs1, m_delta1,
        f"AR6 {ar6_agg} {style.full_name} ({style.units})",
        f"AR6 {ar6_agg} Δ {style.full_name} ({style.units})",
    )

    # Row 2: agreement – explicit ticks, no extend arrows
    _add_row_colorbars(
        2, m_abs0, m_agree,  # reuse cell-level abs mesh for col 1
        f"{style.full_name} ({style.units})",
        "Model sign agreement",
        rhs_ticks=[-2, -1, 0, 1, 2],
        rhs_ticklabels=_AGREE_LABELS,
    )

    return fig

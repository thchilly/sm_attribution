# src/sm_attribution/viz/corr_multifig.py
"""
Visualisation utilities for correlation maps.

Provides a helper to build a multi-panel figure:
  rows   = observational products
  col 1  = global multi-model mean correlation (obsclim_histsoc)
  col 2  = same, aggregated to AR6 regions
  col 3  = AR6-aggregated difference:
           obsclim_histsoc (multi-model) minus counterclim_histsoc (multi-model)

All thresholds, colours and projections are configurable via module-level
variables or function keyword arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import xarray as xr

from sm_attribution.analysis.ensemble import (
    correlation_multimodel_map_path,
)
from sm_attribution.analysis.ar6_regions import ar6_mean_and_field
from sm_attribution.io.registry import default_registry

# ---------------------------------------------------------------------
# Configuration (easy to tweak)
# ---------------------------------------------------------------------


# Default observational rows: key in registry + which "target" to use
@dataclass
class ObsRow:
    key: str          # e.g. "era5-land"
    target: str       # "ssi" or "anomaly"
    label: str        # row label for plotting


DEFAULT_OBS_ROWS: List[ObsRow] = [
    ObsRow("era5-land",   "ssi",     "ERA5-Land 0–1 m"),
    ObsRow("gleam-42a",   "ssi",     "GLEAM v4.2a SMrz"),
    ObsRow("gleam-42b",   "ssi",     "GLEAM v4.2b SMrz"),
    ObsRow("gldas-v21",   "ssi",     "GLDAS-NOAH v2.1"),
    ObsRow("somo-ml",     "ssi",     "SoMo.ml 0–0.5 m"),
    ObsRow("merra2-land", "ssi",     "MERRA-2 Land 0–1 m"),
    ObsRow("gdo-ensmia",  "anomaly", "GDO ENSMIA (anom)"),
    ObsRow("gdo-smia",    "anomaly", "GDO SMIA (anom)"),
]


# Equal-area style projection for global maps
DEFAULT_PROJ = ccrs.Mollweide()

# Classes for the absolute correlation (r)
DEFAULT_R_BINS = [-1.0, 0.0, 0.4, 0.6, 0.8, 1.0]
DEFAULT_R_COLORS = [
    "#b2182b",  # r < 0 : reddish
    "#deebf7",
    "#9ecae1",
    "#4292c6",
    "#08519c",  # r >= 0.8 : deep blue
]

# Classes for the difference in r between scenarios
DEFAULT_DIFF_BINS = [-1.0, -0.1, -0.05, -0.01, 0.0, 0.01, 0.05, 0.1, 1.0]
DEFAULT_DIFF_COLORS = [
    "#762a83",
    "#af8dc3",
    "#e7d4e8",
    "#f7f7f7",
    "#d9f0d3",
    "#7fbf7b",
    "#1b7837",
    "#00441b",
]


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _load_mm_corr(
    scenario: str,
    obs_key: str,
    *,
    target: str,
    mode: str,
    corr_start: str,
    corr_end: str,
) -> xr.Dataset:
    """Load a multi-model mean correlation dataset for a scenario/obs."""
    path = correlation_multimodel_map_path(
        scenario=scenario,
        obs_key=obs_key,
        target=target,
        mode=mode,
        corr_start=corr_start,
        corr_end=corr_end,
    )
    return xr.open_dataset(path)


import matplotlib.colors as mcolors

def _classified_norm_and_cmap(bins, colors):
    """
    Return a BoundaryNorm + ListedColormap from bins/colours.

    If there are more bins (including 'extend' regions) than colours,
    pad the colour list by repeating the last colour so that
    BoundaryNorm does not raise a ValueError.
    """
    bins = list(bins)
    colors = list(colors)

    while True:
        cmap = mcolors.ListedColormap(colors)
        try:
            # extend="both" adds extra regions for <min and >max
            norm = mcolors.BoundaryNorm(bins, ncolors=cmap.N, extend="both")
            break
        except ValueError:
            # Not enough colours for the number of regions: pad by
            # repeating the last colour and try again.
            colors.append(colors[-1])

    return cmap, norm


def _plot_global_map(
    ax,
    da: xr.DataArray,
    *,
    bins: Iterable[float],
    colors: Iterable[str],
    title: str,
):
    """Classified pcolormesh on a global map axis."""
    cmap, norm = _classified_norm_and_cmap(bins, colors)

    ax.set_global()
    ax.coastlines(linewidth=0.4)
    ax.gridlines(draw_labels=False, linewidth=0.2, color="lightgray", alpha=0.5)

    # Use pcolormesh with PlateCarree (lat/lon)
    mesh = ax.pcolormesh(
        da["lon"],
        da["lat"],
        da,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
    )
    ax.set_title(title, fontsize=9)
    return mesh


# ---------------------------------------------------------------------
# Public plotting API
# ---------------------------------------------------------------------


def plot_corr_multifig(
    *,
    obs_rows: Optional[Iterable[ObsRow]] = None,
    scenario_base: str = "obsclim_histsoc",
    scenario_counter: str = "counterclim_histsoc",
    mode: str = "standalone",
    corr_start: str = "2004-01",
    corr_end: str = "2019-12",
    proj=DEFAULT_PROJ,
    r_bins: Iterable[float] = DEFAULT_R_BINS,
    r_colors: Iterable[str] = DEFAULT_R_COLORS,
    diff_bins: Iterable[float] = DEFAULT_DIFF_BINS,
    diff_colors: Iterable[str] = DEFAULT_DIFF_COLORS,
    figsize=(12, 16),
) -> plt.Figure:
    """
    Build a multi-panel figure comparing scenarios across observational products.

    For each observational dataset (row):

      Col 1: Global multi-model mean correlation for `scenario_base`.
      Col 2: Same as Col 1 but aggregated to AR6 regions.
      Col 3: AR6-aggregated difference in multi-model mean correlation:
             `scenario_base` minus `scenario_counter`.

    Parameters
    ----------
    obs_rows : iterable of ObsRow, optional
        Rows to plot. Defaults to DEFAULT_OBS_ROWS.
    scenario_base, scenario_counter : str
        Scenario names, e.g. 'obsclim_histsoc' and 'counterclim_histsoc'.
    mode : {'standalone', 'pooled'}
        SSI mode used when computing correlations.
    corr_start, corr_end : str
        Correlation period (YYYY-MM strings).
    proj : cartopy.crs, optional
        Map projection for all panels.
    r_bins, r_colors : iterable
        Class boundaries and colours for absolute correlation r.
    diff_bins, diff_colors : iterable
        Class boundaries and colours for scenario differences in r.
    figsize : tuple
        Figure size passed to plt.subplots.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure. Axes can be accessed via fig.axes.
    """
    if obs_rows is None:
        obs_rows = DEFAULT_OBS_ROWS

    obs_rows = list(obs_rows)
    nrows = len(obs_rows)
    ncols = 3

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )

    if nrows == 1:
        axes = axes.reshape(1, -1)

    # For column-wise colourbars
    meshes_col1 = []
    meshes_col3 = []

    corrstart_yr = corr_start[:4]
    corrend_yr = corr_end[:4]

    for irow, row in enumerate(obs_rows):
        obs_key = row.key
        target = row.target
        label = row.label

        # --------------------------------------------------------------
        # Load multi-model mean correlations for base & counter scenarios
        # --------------------------------------------------------------
        ds_base = _load_mm_corr(
            scenario=scenario_base,
            obs_key=obs_key,
            target=target,
            mode=mode,
            corr_start=corr_start,
            corr_end=corr_end,
        )
        ds_counter = _load_mm_corr(
            scenario=scenario_counter,
            obs_key=obs_key,
            target=target,
            mode=mode,
            corr_start=corr_start,
            corr_end=corr_end,
        )

        r_base = ds_base["r"]
        r_counter = ds_counter["r"]

        # Col 1: global r (multi-model mean)
        ax1 = axes[irow, 0]
        mesh1 = _plot_global_map(
            ax1,
            r_base,
            bins=r_bins,
            colors=r_colors,
            title=f"{label}\n{scenario_base} (multi-model r)",
        )
        meshes_col1.append(mesh1)

        # Col 2: AR6-aggregated r (base scenario)
        reg_mean, reg_field = ar6_mean_and_field(r_base)
        ax2 = axes[irow, 1]
        _plot_global_map(
            ax2,
            reg_field,
            bins=r_bins,
            colors=r_colors,
            title=f"{label}\nAR6 mean r ({scenario_base})",
        )

        # Col 3: AR6-aggregated difference (base - counter)
        _, reg_field_counter = ar6_mean_and_field(r_counter)
        diff_field = reg_field - reg_field_counter
        diff_field.name = "r_diff"

        ax3 = axes[irow, 2]
        mesh3 = _plot_global_map(
            ax3,
            diff_field,
            bins=diff_bins,
            colors=diff_colors,
            title=f"{label}\nΔr (base−counter)",
        )
        meshes_col3.append(mesh3)

        # Row label on the left side (y-axis)
        ax1.text(
            -0.05,
            0.5,
            label,
            transform=ax1.transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=9,
        )

    # Column titles
    axes[0, 0].set_title(f"{scenario_base} (multi-model r)", fontsize=10)
    axes[0, 1].set_title(f"{scenario_base} – AR6 region means", fontsize=10)
    axes[0, 2].set_title(
        f"{scenario_base} − {scenario_counter} (AR6 Δr)", fontsize=10
    )

    # Shared colourbars for col 1 and col 3
    cbar1 = fig.colorbar(
        meshes_col1[0],
        ax=[axes[i, 0] for i in range(nrows)],
        orientation="horizontal",
        fraction=0.03,
        pad=0.05,
    )
    cbar1.set_label("Pearson r (multi-model mean)")

    cbar3 = fig.colorbar(
        meshes_col3[0],
        ax=[axes[i, 2] for i in range(nrows)],
        orientation="horizontal",
        fraction=0.03,
        pad=0.05,
    )
    cbar3.set_label(
        f"Δr = r({scenario_base}) − r({scenario_counter}) "
        f" [{corrstart_yr}–{corrend_yr}]"
    )

    return fig
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

from sm_attribution.analysis.ensemble import correlation_multimodel_map_path
from sm_attribution.analysis.ar6_regions import (
    ar6_mean_and_field,
    get_ar6_land_regions,
)

# ---------------------------------------------------------------------
# Configuration (easy to tweak)
# ---------------------------------------------------------------------


@dataclass
class ObsRow:
    """One observational product row in the multifigure."""
    key: str          # e.g. "era5-land"
    target: str       # "ssi" or "anomaly"
    label: str        # row label for plotting


DEFAULT_OBS_ROWS: List[ObsRow] = [
    ObsRow("era5-land",   "ssi",     "ERA5-Land 0–1 m"),
    ObsRow("gleam-42a",   "ssi",     "GLEAM v4.2a SMrz"),
    ObsRow("gleam-42b",   "ssi",     "GLEAM v4.2b SMrz"),
    ObsRow("gldas-v21",   "ssi",     "GLDAS-NOAH v2.1"),
    ObsRow("somo-ml",     "ssi",     "SoMo.ml 0-0.5 m"),
    ObsRow("merra2-land", "ssi",     "MERRA-2 Land 0–1 m"),
    ObsRow("gdo-ensmia",  "anomaly", "GDO ENSMIA (anom)"),
    ObsRow("gdo-smia",    "anomaly", "GDO SMIA (anom)"),
]

# Equal-area style projection for global maps
DEFAULT_PROJ = ccrs.Mollweide()

# Classes for the absolute correlation (r);
# first bin [-1, 0) is red; remaining bins are increasing blues.
DEFAULT_R_BINS = [-1.0, 0.0, 0.2, 0.4, 0.7, 0.9, 1.0]
DEFAULT_R_COLORS = [
    "#b2182b",  # r < 0 : reddish
    "#deebf7",  # 0–0.2 (lightest blue)
    "#9ecae1",  # 0.2–0.4
    "#4292c6",  # 0.4–0.7
    "#08519c",  # 0.7–0.9
    "#08306b",  # 0.9–1.0 (deepest blue)
]

# Classes for the difference in r between scenarios (Δr)
DEFAULT_DIFF_BINS = [-1.0, -0.1, -0.05, -0.01, 0.0, 0.01, 0.05, 0.1, 1.0]
DEFAULT_DIFF_COLORS = [
    "#762a83",
    "#af8dc3",
    "#e7d4e8",
    "#faf0fa",
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


def _classified_norm_and_cmap(bins, colors):
    """
    Return a BoundaryNorm + ListedColormap from bins/colours.

    We assume len(colors) == len(bins) - 1 (one colour per interval).
    If there are fewer colours than intervals, pad by repeating
    the last colour. No 'extend' regions are used so the first
    colour corresponds exactly to bins[0]–bins[1], etc.
    """
    bins = list(bins)
    colors = list(colors)

    n_intervals = len(bins) - 1
    if len(colors) < n_intervals:
        colors = colors + [colors[-1]] * (n_intervals - len(colors))

    cmap = mcolors.ListedColormap(colors[:n_intervals])
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
):
    """Classified pcolormesh on a global map axis."""
    cmap, norm = _classified_norm_and_cmap(bins, colors)

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
    figsize=(12, 21),
) -> plt.Figure:
    """
    Build a multi-panel figure comparing scenarios across observational products.

    For each observational dataset (row):

      Col 1: Global multi-model mean correlation for `scenario_base`.
      Col 2: Same as Col 1 but aggregated to AR6 regions.
      Col 3: AR6-aggregated difference in multi-model mean correlation:
             `scenario_base` minus `scenario_counter`.
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
        constrained_layout=False,
    )

    if nrows == 1:
        axes = axes.reshape(1, -1)

    # For column-wise colourbars (we'll place them manually, so this
    # list is only to grab the mappable)
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
            title=f"{label}\nMulti-model FC correlation (multi-model mean)",
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
            title=f"{label}\nMulti-model FC correlation (AR6 region average)",
            draw_ar6_outlines=True,
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
            title=f"{label}\nGain in correlation [FC−CfC] (AR6 region average)",
            draw_ar6_outlines=True,
        )
        meshes_col3.append(mesh3)

    # Adjust spacing: more vertical whitespace between rows,
    # but keep all map axes the same size.
    fig.subplots_adjust(
        hspace=0.2,
        wspace=0.02,
        top=0.96,
        bottom=0.08,
        left=0.03,
        right=0.97,
    )

    # ------------------------------------------------------------------
    # Colourbars: place them in fixed figure coordinates so they do not
    # change the size of any particular column.
    # ------------------------------------------------------------------
    # Left colourbar (for r)
    cax1 = fig.add_axes([0.10, 0.06, 0.35, 0.01])  # [left, bottom, width, height]
    cbar1 = fig.colorbar(
        meshes_col1[0],
        cax=cax1,
        orientation="horizontal",
    )
    cbar1.set_label("Pearson r (multi-model mean)")

    # Right colourbar (for Δr)
    cax3 = fig.add_axes([0.55, 0.06, 0.35, 0.01])
    cbar3 = fig.colorbar(
        meshes_col3[0],
        cax=cax3,
        orientation="horizontal",
    )
    cbar3.set_label(
        f"Δr = r({scenario_base}) − r({scenario_counter}) "
        f"[{corrstart_yr}–{corrend_yr}]"
    )

    return fig
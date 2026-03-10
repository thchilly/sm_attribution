# src/sm_attribution/viz/corr_multifig.py
"""
Visualisation utilities for correlation maps.

Provides a helper to build a multi-panel figure:
  rows   = observational products
  col 1  = global multi-model mean correlation (obsclim_histsoc)
  col 2  = same, aggregated to AR6 regions
  col 3  = AR6-aggregated Δr: climate change effect
           obsclim_histsoc minus counterclim_histsoc
  col 4  = AR6-aggregated Δr: direct human forcing effect
           obsclim_histsoc minus obsclim_1901soc
  col 5  = AR6-aggregated Δr: combined (climate + human) effect
           obsclim_histsoc minus counterclim_1901soc

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
    correlation_map_path,
    correlation_multimodel_map_path,
)
from sm_attribution.analysis.ssi import DEFAULT_SSI_METHOD
from sm_attribution.analysis.ar6_regions import (
    ar6_mean_and_field,
    get_ar6_land_regions,
)
from sm_attribution.metadata.obs_groups import OBS_GROUPS

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
    ObsRow("grace-da-dm", "ssi", "GRACE-DA-DM root-zone"),
    ObsRow("gdo-ensmia",  "anomaly", "GDO ENSMIA (anom)"),
    ObsRow("gdo-smia",    "anomaly", "GDO SMIA (anom)"),
 ]

DEFAULT_MODELS: List[str] = [
    "h08",
    "hydropy",
    "jules-w2",
    "miroc-integ-land",
    "watergap2-2e",
    "web-dhm-sg",
    "lpjml5-7-10-fire",
]

FORCING_GROUPS = {k: v["group"] for k, v in OBS_GROUPS.items()}
GROUP_LABELS = {
    1: "Reanalysis-based datasets",
    2: "Satellite-based datasets",
    3: "Hybrid (Reanalysis + Satellite) datasets",
    4: "In-situ / ML-based datasets",
}

# Equal-area style projection for global maps (keep all - use one at a time)
# DEFAULT_PROJ = ccrs.Mollweide()
DEFAULT_PROJ = ccrs.EckertIV()
# DEFAULT_PROJ = ccrs.EckertVI()
# DEFAULT_PROJ = ccrs.LambertCylindrical()
# DEFAULT_PROJ = ccrs.Sinusoidal()


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
    ssi_method: str = DEFAULT_SSI_METHOD,
) -> xr.Dataset:
    """Load a multi-model mean correlation dataset for a scenario/obs."""
    path = correlation_multimodel_map_path(
        scenario=scenario,
        obs_key=obs_key,
        target=target,
        mode=mode,
        corr_start=corr_start,
        corr_end=corr_end,
        ssi_method=ssi_method,
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
    scenario_1901soc: str = "obsclim_1901soc",
    scenario_combined_counter: str = "counterclim_1901soc",
    mode: str = "standalone",
    corr_start: str = "2004-01",
    corr_end: str = "2019-12",
    ssi_method: str = DEFAULT_SSI_METHOD,
    proj=DEFAULT_PROJ,
    r_bins: Iterable[float] = DEFAULT_R_BINS,
    r_colors: Iterable[str] = DEFAULT_R_COLORS,
    diff_bins: Iterable[float] = DEFAULT_DIFF_BINS,
    diff_colors: Iterable[str] = DEFAULT_DIFF_COLORS,
    figsize=(20, 24.5),
) -> plt.Figure:
    """
    Build a multi-panel figure comparing scenarios across observational products.

    For each observational dataset (row):

      Col 1: Global multi-model mean correlation for `scenario_base`.
      Col 2: Same as Col 1 but aggregated to AR6 regions.
      Col 3: AR6 Δr – climate change effect:
             `scenario_base` minus `scenario_counter`.
      Col 4: AR6 Δr – direct human forcing effect:
             `scenario_base` minus `scenario_1901soc`.
      Col 5: AR6 Δr – combined (climate + human) effect:
             `scenario_base` minus `scenario_combined_counter`.
    """
    if obs_rows is None:
        obs_rows = DEFAULT_OBS_ROWS

    # ------------------------------------------------------------------
    # Attach forcing-group IDs and sort rows by group, then by label
    # ------------------------------------------------------------------
    obs_rows = list(obs_rows)
    rows_with_group: List[tuple[int, ObsRow]] = []
    for row in obs_rows:
        g = FORCING_GROUPS.get(row.key, 999)  # 999 = unknown/other
        rows_with_group.append((g, row))

    rows_with_group.sort(key=lambda gr: (gr[0], gr[1].label))

    groups_for_rows = [g for g, _ in rows_with_group]
    obs_rows_sorted = [r for _, r in rows_with_group]
    obs_rows = obs_rows_sorted

    nrows = len(obs_rows)
    ncols = 5

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
            ssi_method=ssi_method,
        )
        ds_counter = _load_mm_corr(
            scenario=scenario_counter,
            obs_key=obs_key,
            target=target,
            mode=mode,
            corr_start=corr_start,
            corr_end=corr_end,
            ssi_method=ssi_method,
        )
        ds_1901soc = _load_mm_corr(
            scenario=scenario_1901soc,
            obs_key=obs_key,
            target=target,
            mode=mode,
            corr_start=corr_start,
            corr_end=corr_end,
            ssi_method=ssi_method,
        )
        ds_combined = _load_mm_corr(
            scenario=scenario_combined_counter,
            obs_key=obs_key,
            target=target,
            mode=mode,
            corr_start=corr_start,
            corr_end=corr_end,
            ssi_method=ssi_method,
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

        # Calculate difference on the original grid
        r_base = ds_base["r"]
        r_counter = ds_counter["r"]
        r_diff_grid = r_base - r_counter
        r_diff_grid.name = "r_diff"

        # Col 3: AR6-aggregated difference (base - counter)
        # Now, aggregate the difference map (r_diff_grid) to AR6 regions
        _, diff_field = ar6_mean_and_field(r_diff_grid)
        
        ax3 = axes[irow, 2]
        mesh3 = _plot_global_map(
            ax3,
            diff_field,
            bins=diff_bins,
            colors=diff_colors,
            title=f"{label}\nClimate change effect [FC\u00b7HS \u2212 CfC\u00b7HS]",
            draw_ar6_outlines=True,
        )
        meshes_col3.append(mesh3)

        # Col 4: AR6-aggregated Δr – direct human forcing
        # obsclim_histsoc minus obsclim_1901soc
        r_1901soc = ds_1901soc["r"]
        r_diff_human = r_base - r_1901soc
        r_diff_human.name = "r_diff"
        _, diff_human_field = ar6_mean_and_field(r_diff_human)

        ax4 = axes[irow, 3]
        _plot_global_map(
            ax4,
            diff_human_field,
            bins=diff_bins,
            colors=diff_colors,
            title=f"{label}\nDirect human forcing [FC\u00b7HS \u2212 FC\u00b7PS]",
            draw_ar6_outlines=True,
        )

        # Col 5: AR6-aggregated Δr – combined effect
        # obsclim_histsoc minus counterclim_1901soc
        r_combined = ds_combined["r"]
        r_diff_combined = r_base - r_combined
        r_diff_combined.name = "r_diff"
        _, diff_combined_field = ar6_mean_and_field(r_diff_combined)

        ax5 = axes[irow, 4]
        _plot_global_map(
            ax5,
            diff_combined_field,
            bins=diff_bins,
            colors=diff_colors,
            title=f"{label}\nCombined effect [FC\u00b7HS \u2212 CfC\u00b7PS]",
            draw_ar6_outlines=True,
        )

    # Adjust spacing: more vertical whitespace between rows,
    # but keep all map axes the same size.
    fig.subplots_adjust(
        hspace=0.29,
        wspace=0.02,
        top=0.95,
        bottom=0.08,
        left=0.03,
        right=0.97,
    )

    # ------------------------------------------------------------------
    # Add group subtitles above the first row of each forcing group
    # ------------------------------------------------------------------
    seen_groups: set[int] = set()
    for irow, row in enumerate(obs_rows):
        g = groups_for_rows[irow]
        if g in seen_groups or g == 999:
            continue
        seen_groups.add(g)

        row_axes = axes[irow, :]
        left = row_axes[0].get_position().x0
        right = row_axes[-1].get_position().x1
        top = max(ax.get_position().y1 for ax in row_axes)

        x = 0.5 * (left + right)
        y = top + 0.015   # was 0.01; a bit closer to the row

        fig.text(
            x,
            y,
            GROUP_LABELS.get(g, f"Group {g}"),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # ------------------------------------------------------------------
    # Colourbars: place them in fixed figure coordinates so they do not
    # change the size of any particular column.
    # ------------------------------------------------------------------
    # Left colourbar (for r — cols 1–2)
    cax1 = fig.add_axes([0.04, 0.06, 0.22, 0.01])  # [left, bottom, width, height]
    cbar1 = fig.colorbar(
        meshes_col1[0],
        cax=cax1,
        orientation="horizontal",
    )
    cbar1.set_label("Pearson r (multi-model mean)")

    # Right colourbar (for Δr — cols 3–5)
    cax3 = fig.add_axes([0.35, 0.06, 0.55, 0.01])
    cbar3 = fig.colorbar(
        meshes_col3[0],
        cax=cax3,
        orientation="horizontal",
    )
    cbar3.set_label(
        f"Δr (AR6 region average) [{corrstart_yr}–{corrend_yr}]"
    )

    return fig


# ---------------------------------------------------------------------
# Multi-obs mean summary figure (single row, 5 columns)
# ---------------------------------------------------------------------


def _load_model_mean_corr(
    models: List[str],
    scenario: str,
    obs_key: str,
    *,
    target: str,
    mode: str,
    corr_start: str,
    corr_end: str,
    ssi_method: str = DEFAULT_SSI_METHOD,
) -> xr.DataArray:
    """Load per-model correlation maps and return their mean ``r``."""
    acc: Optional[xr.DataArray] = None
    for model in models:
        path = correlation_map_path(
            model=model,
            scenario=scenario,
            obs_key=obs_key,
            target=target,
            mode=mode,
            corr_start=corr_start,
            corr_end=corr_end,
            ssi_method=ssi_method,
        )
        r = xr.open_dataset(path)["r"]
        acc = r.copy() if acc is None else acc + r
    return acc / len(models)


def plot_corr_obs_mean(
    *,
    obs_rows: Optional[Iterable[ObsRow]] = None,
    models: Optional[List[str]] = None,
    scenario_base: str = "obsclim_histsoc",
    scenario_counter: str = "counterclim_histsoc",
    scenario_1901soc: str = "obsclim_1901soc",
    scenario_combined_counter: str = "counterclim_1901soc",
    mode: str = "standalone",
    corr_start: str = "2004-01",
    corr_end: str = "2019-12",
    ssi_method: str = DEFAULT_SSI_METHOD,
    proj=DEFAULT_PROJ,
    r_bins: Iterable[float] = DEFAULT_R_BINS,
    r_colors: Iterable[str] = DEFAULT_R_COLORS,
    diff_bins: Iterable[float] = DEFAULT_DIFF_BINS,
    diff_colors: Iterable[str] = DEFAULT_DIFF_COLORS,
    figsize=(20, 4.0),
) -> plt.Figure:
    """
    Single-row, 5-column summary figure showing the **multi-obs mean**.

    Parameters
    ----------
    models : list of str, optional
        Subset of models to average over.  When *None* (default) the
        pre-computed multi-model mean files are used (all 7 models).
        When a list is given, the per-model correlation files are
        loaded individually and averaged over the selected subset.

    For every scenario the (multi-)model mean correlation map is first
    loaded per observational dataset, then averaged across all obs rows.
    The resulting fields are plotted identically to :func:`plot_corr_multifig`:

      Col 1  Global multi-obs mean r (obsclim_histsoc).
      Col 2  Same, aggregated to AR6 regions.
      Col 3  AR6 Δr – climate change effect   (FC·HS − CfC·HS).
      Col 4  AR6 Δr – direct human forcing     (FC·HS − FC·PS).
      Col 5  AR6 Δr – combined effect           (FC·HS − CfC·PS).
    """
    if obs_rows is None:
        obs_rows = DEFAULT_OBS_ROWS
    obs_rows = list(obs_rows)

    n_obs = len(obs_rows)
    n_models = len(models) if models is not None else len(DEFAULT_MODELS)
    corrstart_yr = corr_start[:4]
    corrend_yr = corr_end[:4]

    # ------------------------------------------------------------------
    # 1. Accumulate r fields across all obs datasets
    # ------------------------------------------------------------------
    r_base_sum: Optional[xr.DataArray] = None
    r_counter_sum: Optional[xr.DataArray] = None
    r_1901soc_sum: Optional[xr.DataArray] = None
    r_combined_sum: Optional[xr.DataArray] = None

    _load_kw = dict(mode=mode, corr_start=corr_start, corr_end=corr_end,
                    ssi_method=ssi_method)

    for row in obs_rows:
        if models is not None:
            # Custom model subset → load per-model files & average
            r_b = _load_model_mean_corr(
                models, scenario_base, row.key,
                target=row.target, **_load_kw)
            r_c = _load_model_mean_corr(
                models, scenario_counter, row.key,
                target=row.target, **_load_kw)
            r_s = _load_model_mean_corr(
                models, scenario_1901soc, row.key,
                target=row.target, **_load_kw)
            r_x = _load_model_mean_corr(
                models, scenario_combined_counter, row.key,
                target=row.target, **_load_kw)
        else:
            # Use pre-computed multi-model mean files (all models)
            r_b = _load_mm_corr(scenario=scenario_base, obs_key=row.key,
                                target=row.target, **_load_kw)["r"]
            r_c = _load_mm_corr(scenario=scenario_counter, obs_key=row.key,
                                target=row.target, **_load_kw)["r"]
            r_s = _load_mm_corr(scenario=scenario_1901soc, obs_key=row.key,
                                target=row.target, **_load_kw)["r"]
            r_x = _load_mm_corr(scenario=scenario_combined_counter, obs_key=row.key,
                                target=row.target, **_load_kw)["r"]

        if r_base_sum is None:
            r_base_sum = r_b.copy()
            r_counter_sum = r_c.copy()
            r_1901soc_sum = r_s.copy()
            r_combined_sum = r_x.copy()
        else:
            r_base_sum = r_base_sum + r_b
            r_counter_sum = r_counter_sum + r_c
            r_1901soc_sum = r_1901soc_sum + r_s
            r_combined_sum = r_combined_sum + r_x

    # Compute the mean across obs datasets
    r_base_mean = r_base_sum / n_obs
    r_counter_mean = r_counter_sum / n_obs
    r_1901soc_mean = r_1901soc_sum / n_obs
    r_combined_mean = r_combined_sum / n_obs

    # ------------------------------------------------------------------
    # 2. Compute Δr fields on the gridded level, then aggregate to AR6
    # ------------------------------------------------------------------
    dr_climate = r_base_mean - r_counter_mean
    dr_climate.name = "r_diff"
    dr_human = r_base_mean - r_1901soc_mean
    dr_human.name = "r_diff"
    dr_combined = r_base_mean - r_combined_mean
    dr_combined.name = "r_diff"

    # ------------------------------------------------------------------
    # 3. Build the 1-row × 5-column figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        1, 5,
        figsize=figsize,
        subplot_kw={"projection": proj},
        constrained_layout=False,
    )

    model_desc = ", ".join(models) if models is not None else f"all {n_models}"
    label = f"Multi-obs mean (n_obs={n_obs}, n_mod={n_models})"

    # Col 1: gridded r (base scenario, multi-obs mean)
    mesh1 = _plot_global_map(
        axes[0], r_base_mean,
        bins=r_bins, colors=r_colors,
        title=f"{label}\nMulti-model correlation (gridded)",
    )

    # Col 2: AR6-aggregated r (base scenario)
    _, reg_field = ar6_mean_and_field(r_base_mean)
    _plot_global_map(
        axes[1], reg_field,
        bins=r_bins, colors=r_colors,
        title=f"{label}\nMulti-model correlation (AR6)",
        draw_ar6_outlines=True,
    )

    # Col 3: AR6 Δr – climate change
    _, dr_climate_ar6 = ar6_mean_and_field(dr_climate)
    mesh3 = _plot_global_map(
        axes[2], dr_climate_ar6,
        bins=diff_bins, colors=diff_colors,
        title=f"{label}\nClimate change [FC·HS − CfC·HS]",
        draw_ar6_outlines=True,
    )

    # Col 4: AR6 Δr – direct human forcing
    _, dr_human_ar6 = ar6_mean_and_field(dr_human)
    _plot_global_map(
        axes[3], dr_human_ar6,
        bins=diff_bins, colors=diff_colors,
        title=f"{label}\nDirect human forcing [FC·HS − FC·PS]",
        draw_ar6_outlines=True,
    )

    # Col 5: AR6 Δr – combined effect
    _, dr_combined_ar6 = ar6_mean_and_field(dr_combined)
    _plot_global_map(
        axes[4], dr_combined_ar6,
        bins=diff_bins, colors=diff_colors,
        title=f"{label}\nCombined effect [FC·HS − CfC·PS]",
        draw_ar6_outlines=True,
    )

    # ------------------------------------------------------------------
    # Layout & colourbars
    # ------------------------------------------------------------------
    fig.subplots_adjust(
        wspace=0.02,
        top=0.85,
        bottom=0.18,
        left=0.03,
        right=0.97,
    )

    # Left colourbar (Pearson r — cols 1–2)
    cax1 = fig.add_axes([0.04, 0.10, 0.22, 0.025])
    cbar1 = fig.colorbar(mesh1, cax=cax1, orientation="horizontal")
    cbar1.set_label("Pearson r (multi-model × multi-obs mean)")

    # Right colourbar (Δr — cols 3–5)
    cax3 = fig.add_axes([0.35, 0.10, 0.55, 0.025])
    cbar3 = fig.colorbar(mesh3, cax=cax3, orientation="horizontal")
    cbar3.set_label(f"Δr (AR6 region average) [{corrstart_yr}–{corrend_yr}]")

    return fig
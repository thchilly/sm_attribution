# src/sm_attribution/viz/ar6_delta_corr_1d.py
"""
AR6 regional differences in correlation (FC − CfC) across observational products.

For each AR6 land region (excluding Greenland & Antarctica), we:
  * take the multi-model mean correlation map r for the factual scenario (FC)
  * take the multi-model mean correlation map r for the counterfactual (CfC)
  * form Δr = r(FC) − r(CfC) at each grid point
  * aggregate Δr to AR6 land regions using `ar6_mean_and_field`
  * plot, for each region:
      - a boxplot summarising Δr across all observational products, and
      - individual points for each product, colour-coded by dataset and with
        marker shape indicating the forcing-based group.

This is the 1-D analogue of the 3rd column of `corr_multifig`, but with
"subtract first, then aggregate".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import xarray as xr
import pandas as pd

from sm_attribution.analysis.ensemble import correlation_multimodel_map_path
from sm_attribution.analysis.ssi import DEFAULT_SSI_METHOD
from sm_attribution.analysis.ar6_regions import ar6_mean_and_field, get_ar6_land_regions
from sm_attribution.metadata.obs_groups import (
    OBS_GROUPS,
    GROUP_MARKERS,
    OBS_COLORS,
    GROUP_LABELS,
)


# ---------------------------------------------------------------------
# Obs row definition
# ---------------------------------------------------------------------

@dataclass
class ObsRow:
    key: str          # registry key, e.g. "era5-land"
    target: str       # "ssi" or "anomaly"
    label: str        # human-readable label


DEFAULT_OBS_ROWS: List[ObsRow] = [
    ObsRow("era5-land",   "ssi",     "ERA5-Land 0–1 m"),
    ObsRow("gleam-42a",   "ssi",     "GLEAM v4.2a SMrz"),
    ObsRow("gleam-42b",   "ssi",     "GLEAM v4.2b SMrz"),
    ObsRow("gldas-v21",   "ssi",     "GLDAS-NOAH v2.1"),
    ObsRow("somo-ml",     "ssi",     "SoMo.ml 0–0.5 m"),
    ObsRow("merra2-land", "ssi",     "MERRA-2 Land 0–1 m"),
    ObsRow("grace-da-dm", "ssi",     "GRACE-DA-DM root-zone"),
    ObsRow("gdo-ensmia",  "anomaly", "GDO ENSMIA (anom)"),
    ObsRow("gdo-smia",    "anomaly", "GDO SMIA (anom)"),
]


# Map obs key → group id for convenience
FORCING_GROUPS: Dict[str, int] = {k: v["group"] for k, v in OBS_GROUPS.items()}


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
) -> xr.DataArray:
    """Load the multi-model mean correlation field r(lat, lon) for a given (scenario, obs) pair."""
    path = correlation_multimodel_map_path(
        scenario=scenario,
        obs_key=obs_key,
        target=target,
        mode=mode,
        corr_start=corr_start,
        corr_end=corr_end,
        ssi_method=ssi_method,
    )
    ds = xr.open_dataset(path)
    return ds["r"]


def _calculate_regional_delta_matrix(
    obs_rows: List[ObsRow],
    scenario_base: str,
    scenario_counter: str,
    mode: str,
    corr_start: str,
    corr_end: str,
    ssi_method: str = DEFAULT_SSI_METHOD,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Computes the AR6 regional mean difference (Δr) for all datasets.
    
    Returns:
        - delta_mat: NumPy array (rows=datasets, cols=regions)
        - region_labels: List of valid AR6 region abbreviations
        - obs_keys: List of dataset keys used
    """
    regions = get_ar6_land_regions()
    region_abbrevs_all = list(getattr(regions, "abbrevs", []))
    n_regions_total = len(region_abbrevs_all)
    obs_keys = [row.key for row in obs_rows]
    n_obs = len(obs_keys)
    
    delta_mat = np.full((n_obs, n_regions_total), np.nan, dtype=float)

    for i, row in enumerate(obs_rows):
        # Load correlations and compute difference map
        r_base = _load_mm_corr(scenario=scenario_base, obs_key=row.key, target=row.target, mode=mode, corr_start=corr_start, corr_end=corr_end, ssi_method=ssi_method)
        r_counter = _load_mm_corr(scenario=scenario_counter, obs_key=row.key, target=row.target, mode=mode, corr_start=corr_start, corr_end=corr_end, ssi_method=ssi_method)
        diff_map = r_base - r_counter
        
        # Aggregate difference map to AR6 regions
        reg_mean_diff, _ = ar6_mean_and_field(diff_map)

        # Store 1D array of regional means
        delta_mat[i, :] = reg_mean_diff.values

    # Drop regions that are NaN for all obs (polar regions, etc.)
    valid_region_mask = ~np.all(np.isnan(delta_mat), axis=0)
    keep_idx = np.where(valid_region_mask)[0]

    # Filter matrix and labels
    delta_mat = delta_mat[:, keep_idx]
    region_labels = [region_abbrevs_all[i] for i in keep_idx]
    
    return delta_mat, region_labels, obs_keys


# ---------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------

def plot_ar6_delta_corr_1d(
    *,
    obs_rows: Optional[Iterable[ObsRow]] = None,
    scenario_base: str = "obsclim_histsoc",
    scenario_counter: str = "counterclim_histsoc",
    mode: str = "standalone",
    corr_start: str = "2004-01",
    corr_end: str = "2019-12",
    ssi_method: str = DEFAULT_SSI_METHOD,
    figsize=(14, 6),
) -> plt.Figure:
    """
    Plots AR6 regional differences in Pearson r between factual and counterfactual scenarios (Δr).
    """
    if obs_rows is None:
        obs_rows = DEFAULT_OBS_ROWS
    obs_rows = list(obs_rows)

    # Calculate data matrix using the common helper function
    delta_mat, region_labels, obs_keys = _calculate_regional_delta_matrix(
        obs_rows=obs_rows,
        scenario_base=scenario_base,
        scenario_counter=scenario_counter,
        mode=mode,
        corr_start=corr_start,
        corr_end=corr_end,
        ssi_method=ssi_method,
    )
    n_regions = len(region_labels)
    n_obs = len(obs_keys)

    # ------------------------------------------------------------------
    # Plot Setup
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_regions)

    # Prepare data for boxplots (list of 1D arrays, one per region)
    data_for_box = [delta_mat[:, j][~np.isnan(delta_mat[:, j])]
                    for j in range(n_regions)]

    # --- Boxplot Rendering (Three Passes for Full Control) ---

    # 1. Fill only (zorder=1, behind markers)
    bp_fill = ax.boxplot(
        data_for_box, positions=x, widths=0.5, manage_ticks=False, showfliers=False, patch_artist=True, zorder=1
    )
    for box in bp_fill["boxes"]: box.set(facecolor="#e0e0e0", edgecolor="none")
    for median in bp_fill["medians"]: median.set(color="none", linewidth=0)
    for whisker in bp_fill["whiskers"]: whisker.set(color="none", linewidth=0)
    for cap in bp_fill["caps"]: cap.set(color="none", linewidth=0)


    # Scatter points per dataset

    # Calculate jitter for horizontal spread (0.0 means no jitter, markers are vertically aligned)
    jit_offset = 0.0 # Set to 0.0 for perfect vertical alignment
    jitter_offset = np.linspace(-jit_offset, jit_offset, n_obs)

    for i_obs, key in enumerate(obs_keys):
        y_vals = delta_mat[i_obs, :]
        
        # Apply offset to the x-position array
        x_scatter = x + jitter_offset[i_obs] 
        meta = OBS_GROUPS.get(key, {})
        g = meta.get("group", 0)
        marker = meta.get("marker", GROUP_MARKERS.get(g, "o"))
        color = meta.get("color", OBS_COLORS.get(key, "C0"))

        ax.scatter(
            x_scatter,
            y_vals,
            marker=marker,
            s=30,
            facecolor=color,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.9,
            zorder=3,
        )

    # 2. Outlines only (zorder=4, around markers)
    bp_outline = ax.boxplot(
        data_for_box, positions=x, widths=0.5, manage_ticks=False, showfliers=False, patch_artist=False, zorder=4
    )
    for box in bp_outline["boxes"]: box.set(linewidth=0.8, color="black")
    for median in bp_outline["medians"]: median.set(color="black", linewidth=0) # Hide median again
    for whisker in bp_outline["whiskers"]: whisker.set(color="black", linewidth=0.8)
    for cap in bp_outline["caps"]: cap.set(color="black", linewidth=0.8)


    # 3. Median only (zorder=5, on top)
    bp_median = ax.boxplot(
        data_for_box, positions=x, widths=0.5, manage_ticks=False, showfliers=False, patch_artist=False, zorder=5
    )
    for box in bp_median["boxes"]: box.set(linewidth=0)
    for median in bp_median["medians"]: median.set(color="red", linewidth=1.2) # Use red median
    for whisker in bp_median["whiskers"]: whisker.set(color="black", linewidth=0)
    for cap in bp_median["caps"]: cap.set(color="black", linewidth=0)

    # Horizontal zero line
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--", zorder=1)

    # X-axis formatting: minimal whitespace at edges
    ax.set_xticks(x)
    ax.set_xticklabels(region_labels, rotation=90, fontsize=8)
    ax.set_xlim(-0.5, n_regions - 0.5)
    ax.margins(x=0)

    ax.set_ylabel(
        f"Δr = r({scenario_base}) − r({scenario_counter})",
        fontsize=10,
    )

    # ------------------------------------------------------------------
    # Legends
    # ------------------------------------------------------------------
    # Dataset legend (colour + marker per pseudo-observation)
    dataset_handles = []
    for key in obs_keys:
        meta = OBS_GROUPS.get(key, {})
        color = meta.get("color", OBS_COLORS.get(key, "C0"))
        g = meta.get("group", 0)
        marker = meta.get("marker", GROUP_MARKERS.get(g, "o"))
        lbl = next(row.label for row in obs_rows if row.key == key)

        h = Line2D(
            [0], [0], marker=marker, linestyle="None", markerfacecolor=color, markeredgecolor="black",
            markersize=6, label=lbl,
        )
        dataset_handles.append(h)

    # Group legend (shared markers)
    group_handles = []
    for g, label in GROUP_LABELS.items():
        marker = GROUP_MARKERS.get(g, "o")
        h = Line2D(
            [0], [0], marker=marker, linestyle="None", markerfacecolor="white", markeredgecolor="black",
            markersize=6, label=label,
        )
        group_handles.append(h)

    # Place legends BELOW the plot:
    dataset_legend = ax.legend(
        handles=dataset_handles, title="Pseudo-observational datasets\n ", loc="upper left",
        bbox_to_anchor=(0.0, -0.30), borderaxespad=0.0, fontsize=8, title_fontsize=9, ncol=5,
    )
    ax.add_artist(dataset_legend)

    group_legend = ax.legend(
        handles=group_handles, title="Forcing-based group\n ", loc="upper left",
        bbox_to_anchor=(0.70, -0.30), borderaxespad=0.0, fontsize=8, title_fontsize=9, ncol=2,
    )

    # Adjust layout to accommodate legends
    fig.tight_layout(rect=[0.03, 0.25, 0.98, 0.95])

    return fig


# ---------------------------------------------------------------------
# Helper: render a single boxplot panel onto an existing Axes
# ---------------------------------------------------------------------


def _render_boxplot_panel(
    ax: plt.Axes,
    delta_mat: np.ndarray,
    region_labels: List[str],
    obs_rows: List[ObsRow],
    obs_keys: List[str],
    *,
    ylabel: str,
    title: str,
    show_xticklabels: bool = True,
    show_legend: bool = False,
) -> None:
    """Draw one boxplot panel (regions on x, obs on scatter) onto *ax*."""
    n_regions = len(region_labels)
    n_obs = len(obs_keys)
    x = np.arange(n_regions)

    data_for_box = [
        delta_mat[:, j][~np.isnan(delta_mat[:, j])] for j in range(n_regions)
    ]

    # 1. Fill
    bp_fill = ax.boxplot(
        data_for_box, positions=x, widths=0.5, manage_ticks=False,
        showfliers=False, patch_artist=True, zorder=1,
    )
    for box in bp_fill["boxes"]:
        box.set(facecolor="#e0e0e0", edgecolor="none")
    for el in (*bp_fill["medians"], *bp_fill["whiskers"], *bp_fill["caps"]):
        el.set(color="none", linewidth=0)

    # Scatter
    jitter_offset = np.linspace(0.0, 0.0, n_obs)
    for i_obs, key in enumerate(obs_keys):
        y_vals = delta_mat[i_obs, :]
        x_scatter = x + jitter_offset[i_obs]
        meta = OBS_GROUPS.get(key, {})
        g = meta.get("group", 0)
        marker = meta.get("marker", GROUP_MARKERS.get(g, "o"))
        color = meta.get("color", OBS_COLORS.get(key, "C0"))
        ax.scatter(
            x_scatter, y_vals, marker=marker, s=30,
            facecolor=color, edgecolor="black", linewidth=0.4,
            alpha=0.9, zorder=3,
        )

    # 2. Outlines
    bp_outline = ax.boxplot(
        data_for_box, positions=x, widths=0.5, manage_ticks=False,
        showfliers=False, patch_artist=False, zorder=4,
    )
    for box in bp_outline["boxes"]:
        box.set(linewidth=0.8, color="black")
    for median in bp_outline["medians"]:
        median.set(color="black", linewidth=0)
    for el in (*bp_outline["whiskers"], *bp_outline["caps"]):
        el.set(color="black", linewidth=0.8)

    # 3. Median
    bp_median = ax.boxplot(
        data_for_box, positions=x, widths=0.5, manage_ticks=False,
        showfliers=False, patch_artist=False, zorder=5,
    )
    for box in bp_median["boxes"]:
        box.set(linewidth=0)
    for median in bp_median["medians"]:
        median.set(color="red", linewidth=1.2)
    for el in (*bp_median["whiskers"], *bp_median["caps"]):
        el.set(color="black", linewidth=0)

    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xticks(x)
    if show_xticklabels:
        ax.set_xticklabels(region_labels, rotation=90, fontsize=8)
    else:
        ax.set_xticklabels([])
    ax.set_xlim(-0.5, n_regions - 0.5)
    ax.margins(x=0)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, pad=6)

    if show_legend:
        dataset_handles = []
        for key in obs_keys:
            meta = OBS_GROUPS.get(key, {})
            color = meta.get("color", OBS_COLORS.get(key, "C0"))
            g = meta.get("group", 0)
            marker = meta.get("marker", GROUP_MARKERS.get(g, "o"))
            lbl = next(row.label for row in obs_rows if row.key == key)
            h = Line2D(
                [0], [0], marker=marker, linestyle="None",
                markerfacecolor=color, markeredgecolor="black",
                markersize=6, label=lbl,
            )
            dataset_handles.append(h)

        group_handles = []
        for g, label in GROUP_LABELS.items():
            marker = GROUP_MARKERS.get(g, "o")
            h = Line2D(
                [0], [0], marker=marker, linestyle="None",
                markerfacecolor="white", markeredgecolor="black",
                markersize=6, label=label,
            )
            group_handles.append(h)

        dataset_legend = ax.legend(
            handles=dataset_handles,
            title="Pseudo-observational datasets\n ",
            loc="upper left", bbox_to_anchor=(0.0, -0.30),
            borderaxespad=0.0, fontsize=8, title_fontsize=9, ncol=5,
        )
        ax.add_artist(dataset_legend)
        ax.legend(
            handles=group_handles,
            title="Forcing-based group\n ",
            loc="upper left", bbox_to_anchor=(0.70, -0.30),
            borderaxespad=0.0, fontsize=8, title_fontsize=9, ncol=2,
        )


# ---------------------------------------------------------------------
# 3-row figure: climate change · human forcing · combined
# ---------------------------------------------------------------------


def plot_ar6_delta_corr_3effects(
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
    shared_ylim: bool = True,
    figsize=(14, 16),
) -> plt.Figure:
    """
    Three-row boxplot figure across AR6 regions.

    Row 1: Climate change effect      (FC·HS − CfC·HS)
    Row 2: Direct human forcing        (FC·HS − FC·PS)
    Row 3: Combined effect             (FC·HS − CfC·PS)

    Parameters
    ----------
    obs_rows : iterable of ObsRow, optional
        Which observational datasets to include. Defaults to all 9.
    scenario_base / scenario_counter / scenario_1901soc / scenario_combined_counter
        The four ISIMIP scenario identifiers.
    shared_ylim : bool
        If True (default), enforce the same y-axis range across all three
        rows so the panels are directly comparable.  If False, each row
        uses its own auto-scaled limits.
    """
    if obs_rows is None:
        obs_rows = DEFAULT_OBS_ROWS
    obs_rows = list(obs_rows)

    _kw = dict(mode=mode, corr_start=corr_start, corr_end=corr_end,
               ssi_method=ssi_method)

    # Compute the three delta matrices
    dm_cc, regions, obs_keys = _calculate_regional_delta_matrix(
        obs_rows=obs_rows, scenario_base=scenario_base,
        scenario_counter=scenario_counter, **_kw,
    )
    dm_hf, _, _ = _calculate_regional_delta_matrix(
        obs_rows=obs_rows, scenario_base=scenario_base,
        scenario_counter=scenario_1901soc, **_kw,
    )
    dm_cb, _, _ = _calculate_regional_delta_matrix(
        obs_rows=obs_rows, scenario_base=scenario_base,
        scenario_counter=scenario_combined_counter, **_kw,
    )

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    _render_boxplot_panel(
        axes[0], dm_cc, regions, obs_rows, obs_keys,
        ylabel="Δr  [FC·HS − CfC·HS]",
        title="Climate change effect",
        show_xticklabels=False,
    )
    _render_boxplot_panel(
        axes[1], dm_hf, regions, obs_rows, obs_keys,
        ylabel="Δr  [FC·HS − FC·PS]",
        title="Direct human forcing effect",
        show_xticklabels=False,
    )
    _render_boxplot_panel(
        axes[2], dm_cb, regions, obs_rows, obs_keys,
        ylabel="Δr  [FC·HS − CfC·PS]",
        title="Combined (climate + human) effect",
        show_xticklabels=True,
        show_legend=True,
    )

    # ------------------------------------------------------------------
    # Optionally enforce a shared y-axis range across all three rows
    # ------------------------------------------------------------------
    if shared_ylim:
        all_data = np.concatenate([
            dm_cc.ravel(), dm_hf.ravel(), dm_cb.ravel(),
        ])
        finite = all_data[np.isfinite(all_data)]
        if finite.size > 0:
            margin = 0.05
            ymin, ymax = float(finite.min()), float(finite.max())
            span = ymax - ymin if ymax > ymin else 0.01
            ylim = (ymin - margin * span, ymax + margin * span)
            for ax in axes:
                ax.set_ylim(ylim)

    fig.tight_layout(rect=[0.03, 0.12, 0.98, 0.95])

    return fig


# ---------------------------------------------------------------------
# Function for table creation (NEW)
# ---------------------------------------------------------------------

def create_ar6_delta_corr_table(
    *,
    obs_rows: Optional[Iterable[ObsRow]] = None,
    scenario_base: str = "obsclim_histsoc",
    scenario_counter: str = "counterclim_histsoc",
    mode: str = "standalone",
    corr_start: str = "2004-01",
    corr_end: str = "2019-12",
    ssi_method: str = DEFAULT_SSI_METHOD,
    decimals: int = 4,
) -> pd.DataFrame:
    """
    Creates a table (Pandas DataFrame) summarizing the AR6 regional difference 
    in correlation (Δr) for all datasets, plus the ensemble mean.
    
    The table columns are the human-readable dataset labels and a 'Mean' column.
    """
    if obs_rows is None:
        obs_rows = DEFAULT_OBS_ROWS
    obs_rows_list = list(obs_rows)

    # Calculate data matrix using the common helper function
    delta_mat, region_labels, obs_keys = _calculate_regional_delta_matrix(
        obs_rows=obs_rows_list,
        scenario_base=scenario_base,
        scenario_counter=scenario_counter,
        mode=mode,
        corr_start=corr_start,
        corr_end=corr_end,
        ssi_method=ssi_method,
    )

    # Convert the (n_obs, n_regions) NumPy array to a DataFrame (index=region, columns=dataset)
    df_delta = pd.DataFrame(
        data=delta_mat.T,
        index=region_labels,
        columns=obs_keys
    )

    # Calculate the mean across all datasets for each region (row mean)
    df_delta['Mean'] = df_delta.mean(axis=1)

    # Rename columns using human-readable labels for the final table
    label_map = {row.key: row.label.split('(')[0].strip() for row in obs_rows_list}
    label_map['Mean'] = 'Mean'
    df_delta.rename(columns=label_map, inplace=True)

    # Apply formatting (rounding)
    df_table = df_delta.round(decimals)
    
    return df_table

# ---------------------------------------------------------------------
# Function for table creation (NEW Figure Function)
# ---------------------------------------------------------------------

def plot_ar6_delta_corr_table(
    *,
    obs_rows: Optional[Iterable[ObsRow]] = None,
    scenario_base: str = "obsclim_histsoc",
    scenario_counter: str = "counterclim_histsoc",
    mode: str = "standalone",
    corr_start: str = "2004-01",
    corr_end: str = "2019-12",
    ssi_method: str = DEFAULT_SSI_METHOD,
    decimals: int = 4,
) -> plt.Figure:
    """
    Creates a Matplotlib figure containing a table of the AR6 regional difference 
    in correlation (Δr) for all datasets and the ensemble mean.
    """
    if obs_rows is None:
        obs_rows = DEFAULT_OBS_ROWS
    obs_rows_list = list(obs_rows)

    # 1. Calculate the base data using the common helper function
    delta_mat, region_labels, obs_keys = _calculate_regional_delta_matrix(
        obs_rows=obs_rows_list,
        scenario_base=scenario_base,
        scenario_counter=scenario_counter,
        mode=mode,
        corr_start=corr_start,
        corr_end=corr_end,
        ssi_method=ssi_method,
    )

    # Reconstruct the DataFrame (index=region, columns=dataset)
    df_delta = pd.DataFrame(
        data=delta_mat.T,
        index=region_labels,
        columns=obs_keys
    )

    # 2. Process DataFrame for table content
    df_delta['Mean'] = df_delta.mean(axis=1)

    # Rename columns using human-readable labels
    label_map = {row.key: row.label.split('(')[0].strip() for row in obs_rows_list}
    label_map['Mean'] = 'Mean'
    df_delta.rename(columns=label_map, inplace=True)

    # Format the data for the table (rounding)
    df_table = df_delta.round(decimals)
    
    # 3. Create the Matplotlib Figure and Table
    
    # Setup Figure and Axes
    fig, ax = plt.subplots(figsize=(16, 12)) 
    ax.axis('off')
    ax.axis('tight')

    # Define the table properties
    table = ax.table(
        cellText=df_table.values,
        rowLabels=df_table.index,
        colLabels=df_table.columns,
        loc='center',
        cellLoc='center',
        colLoc='center',
    )

    # General styling: Scale to fit the window and set font size
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # --- Colormap setup for data cells (RWG: Red-White-Green) ---
    
    # 1. Determine the absolute maximum for symmetric normalization based on 95th percentile
    numerical_data = df_delta.values.flatten()
    
    # Filter out NaN values before calculating the percentile
    finite_data = numerical_data[np.isfinite(numerical_data)]
    
    # Calculate the 95th percentile of the absolute data values
    if finite_data.size > 0:
        percentile_95 = np.percentile(np.abs(finite_data), 95)
        # Use the 95th percentile for symmetric bounding
        abs_max = percentile_95
    else:
        abs_max = 0.01 # Default small value if no data is present

    # 2. Define Symmetric Normalization: [ -abs_max, 0, +abs_max ]
    norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)
    
    # 3. Create Custom RWG Colormap: Forces center to pure white.
    colors = [(1, 0, 0), (1, 1, 1), (0, 0.5, 0)]  # Red, White, Darker Green
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list("RWG", colors, N=256)
    
    # Index of the 'Mean' column
    mean_col_idx = len(df_table.columns) - 1

    # Highlight the header row and apply colormapping to data cells
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_facecolor('#f0f0f0') 
            cell.set_text_props(fontweight='bold')
        
        else: # Data rows
            # Apply colormap to numerical data cells *except* the Mean column
            if j < mean_col_idx: 
                try:
                    # Use df_delta (unrounded) for accurate color mapping
                    value = df_delta.iloc[i-1, j] 
                    if not pd.isna(value):
                        # Use cmap(norm(value)) to get the color based on symmetric magnitude
                        cell_color = cmap(norm(value))
                        cell.set_facecolor(cell_color)
                        
                except IndexError:
                    pass 

            # Apply bold text styling and black color to the 'Mean' column
            if j == mean_col_idx:
                # FIX: Set color to black, but keep bold font weight
                cell.set_text_props(fontweight='bold', color='#000000')

    # Set the title
    title_str = (
        f"AR6 Regional Correlation Difference (Δr)\n"
        f"Δr = r({scenario_base}) − r({scenario_counter}) [{corr_start[:4]}–{corr_end[:4]}]"
    )
    ax.set_title(title_str, fontsize=14, pad=20, y=1.08) 

    fig.tight_layout()
    
    return fig
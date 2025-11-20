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
from typing import Iterable, List, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import xarray as xr

from sm_attribution.analysis.ensemble import correlation_multimodel_map_path
from sm_attribution.analysis.ar6_regions import ar6_mean_and_field, get_ar6_land_regions
from sm_attribution.metadata.obs_groups import (
    OBS_GROUPS,
    GROUP_MARKERS,
    OBS_COLORS,
    GROUP_LABELS,
)


# ---------------------------------------------------------------------
# Obs row definition (mirrors corr_multifig)
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
) -> xr.DataArray:
    """
    Load the multi-model mean correlation field r(lat, lon) for a given
    (scenario, obs) pair.
    """
    path = correlation_multimodel_map_path(
        scenario=scenario,
        obs_key=obs_key,
        target=target,
        mode=mode,
        corr_start=corr_start,
        corr_end=corr_end,
    )
    ds = xr.open_dataset(path)
    return ds["r"]



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
    figsize=(14, 6),
) -> plt.Figure:
    """
    Plot AR6 regional differences in Pearson r between factual and
    counterfactual scenarios (Δr = r(FC) − r(CfC)).

    [docstring unchanged...]
    """
    if obs_rows is None:
        obs_rows = DEFAULT_OBS_ROWS
    obs_rows = list(obs_rows)

    regions = get_ar6_land_regions()
    region_abbrevs = list(getattr(regions, "abbrevs", []))
    n_regions_total = len(region_abbrevs)

    # ------------------------------------------------------------------
    # Compute AR6 regional Δr for each observational product
    # ------------------------------------------------------------------
    delta_by_obs: Dict[str, xr.DataArray] = {}

    for row in obs_rows:
        obs_key = row.key
        target = row.target

        r_base = _load_mm_corr(
            scenario=scenario_base,
            obs_key=obs_key,
            target=target,
            mode=mode,
            corr_start=corr_start,
            corr_end=corr_end,
        )
        r_counter = _load_mm_corr(
            scenario=scenario_counter,
            obs_key=obs_key,
            target=target,
            mode=mode,
            corr_start=corr_start,
            corr_end=corr_end,
        )

        diff_map = r_base - r_counter
        reg_mean_diff, _ = ar6_mean_and_field(diff_map)

        delta_by_obs[obs_key] = reg_mean_diff  # (region)

    # Stack into (n_obs, n_region)
    obs_keys = [row.key for row in obs_rows]
    n_obs = len(obs_keys)
    delta_mat = np.full((n_obs, n_regions_total), np.nan, dtype=float)

    for i, key in enumerate(obs_keys):
        da = delta_by_obs[key]
        delta_mat[i, :] = da.values

    # Drop regions that are NaN for all obs (ANT, GIC, etc.)
    valid_region_mask = ~np.all(np.isnan(delta_mat), axis=0)
    keep_idx = np.where(valid_region_mask)[0]

    delta_mat = delta_mat[:, keep_idx]
    region_labels = [region_abbrevs[i] for i in keep_idx]
    n_regions = len(keep_idx)


    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_regions)

    # Boxplots (background distribution across pseudo-observations)
    data_for_box = [delta_mat[:, j][~np.isnan(delta_mat[:, j])]
                    for j in range(n_regions)]

    # --- First pass: fill only, behind everything ---
    bp_fill = ax.boxplot(
        data_for_box,
        positions=x,
        widths=0.5,
        manage_ticks=False,
        showfliers=False,
        patch_artist=True,
        zorder=1,
    )
    for box in bp_fill["boxes"]:
        box.set(facecolor="#e0e0e0", edgecolor="none")
    # Hide all line elements in this pass
    for median in bp_fill["medians"]:
        median.set(color="none", linewidth=0)
    for whisker in bp_fill["whiskers"]:
        whisker.set(color="none", linewidth=0)
    for cap in bp_fill["caps"]:
        cap.set(color="none", linewidth=0)


    # Scatter points per dataset

    # Define the maximum desired offset from the center line
    jit_offset = 0.0
    n_obs = len(obs_keys)
    jitter_offset = np.linspace(-jit_offset, jit_offset, n_obs)

    # Scatter points per dataset
    for i_obs, key in enumerate(obs_keys):
        y_vals = delta_mat[i_obs, :]
        # --- Apply a small jitter/offset to the x-position array ---
        x_scatter = x + jitter_offset[i_obs] # Shift the entire array by a small amount
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

    # --- Second pass: outlines only, on top of markers ---
    bp_outline = ax.boxplot(
        data_for_box,
        positions=x,
        widths=0.5,
        manage_ticks=False,
        showfliers=False,
        patch_artist=False,  # no fill from this pass
        zorder=1,
    )

    for box in bp_outline["boxes"]:
        box.set_linewidth(0.8)
        box.set_color("black")
    for median in bp_outline["medians"]:
        median.set(color="black", linewidth=0)
    for whisker in bp_outline["whiskers"]:
        whisker.set(color="black", linewidth=0.8)
    for cap in bp_outline["caps"]:
        cap.set(color="black", linewidth=0.8)


    # --- Third pass: median only, on top of markers ---
    bp_outline = ax.boxplot(
        data_for_box,
        positions=x,
        widths=0.5,
        manage_ticks=False,
        showfliers=False,
        patch_artist=False,  # no fill from this pass
        zorder=5,
    )

    for box in bp_outline["boxes"]:
        box.set_linewidth(0)
        box.set_color("black")
    for median in bp_outline["medians"]:
        median.set(color="black", linewidth=1.2)
    for whisker in bp_outline["whiskers"]:
        whisker.set(color="black", linewidth=0)
    for cap in bp_outline["caps"]:
        cap.set(color="black", linewidth=0)

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
            [0], [0],
            marker=marker,
            linestyle="None",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=6,
            label=lbl,
        )
        dataset_handles.append(h)

    # Group legend (shared markers)
    group_handles = []
    for g, label in GROUP_LABELS.items():
        marker = GROUP_MARKERS.get(g, "o")
        h = Line2D(
            [0], [0],
            marker=marker,
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6,
            label=label,
        )
        group_handles.append(h)

    # Place legends BELOW the plot:
    #   - datasets: left ~70% of width, 2 rows × up to 5 columns
    #   - groups:   right ~30% of width, 2 rows × 2 columns
    dataset_legend = ax.legend(
        handles=dataset_handles,
        title="Pseudo-observational datasets\n ",
        loc="upper left",
        bbox_to_anchor=(0.0, -0.30),
        borderaxespad=0.0,
        fontsize=8,
        title_fontsize=9,
        ncol=5,
    )
    ax.add_artist(dataset_legend)

    group_legend = ax.legend(
        handles=group_handles,
        title="Forcing-based group\n ",
        loc="upper left",
        bbox_to_anchor=(0.70, -0.30),
        borderaxespad=0.0,
        fontsize=8,
        title_fontsize=9,
        ncol=2,
    )

    # Leave enough space at the bottom for the legends; no extra space on right.
    fig.tight_layout(rect=[0.03, 0.25, 0.98, 0.95])

    return fig
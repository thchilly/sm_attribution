# src/sm_attribution/viz/spatial_correlation_plots.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sm_attribution.io.registry import Registry
from sm_attribution.analysis.ssi import DEFAULT_SSI_METHOD


DEFAULT_SCENARIO_ORDER = (
    "obsclim_histsoc",       # 1st
    "obsclim_1901soc",       # 2nd
    "counterclim_histsoc",   # 3rd
    "counterclim_1901soc",   # 4th
)

DEFAULT_FEATURE_ORDER = (
    "duration",
    "magnitude",
    "intensity",
    "peak_intensity",
    "ddd",
    "ttm10",
    "tts15",
    "tte20",
    "drd",
    "n_events",
    "interarrival",
    "return_period",
)

DEFAULT_SCENARIO_COLORS = {
    "obsclim_histsoc": "#2E7D32",      # green
    "obsclim_1901soc": "#C8B53B",  # yellow-ish
    "counterclim_histsoc": "#2F5D8A",      # blue
    "counterclim_1901soc": "#D1495B",  # red
}


def _format_from_template(tmpl: str, reg: Registry, **kw) -> str:
    paths = reg.cfg_dict.get("paths", {}) or {}
    for k, v in paths.items():
        tmpl = tmpl.replace("{paths." + k + "}", v)
    return tmpl.format(**kw)


def _master_path_for_obs(
    *,
    reg: Registry,
    obskey: str,
    period_mode: str,
    model_mode: str,
    pool_id: str = "standalone",
    scale: int,
    ref_start: str,
    ref_end: str,
    ssi_method: str = DEFAULT_SSI_METHOD,
) -> str:
    metrics = reg.cfg_dict.get("metrics", {}) or {}
    if "spatial_correlation_master" not in metrics:
        raise KeyError(
            "Missing metrics.spatial_correlation_master in registry config."
        )

    tmpl = metrics["spatial_correlation_master"]
    return _format_from_template(
        tmpl,
        reg,
        obskey=obskey,
        period_mode=period_mode,
        model_mode=model_mode,
        pool_id=pool_id,
        scale=scale,
        refstart=ref_start.replace("-", ""),
        refend=ref_end.replace("-", ""),
        ssi_method=ssi_method,
    )


def _pretty_feature(name: str) -> str:
    # keep your naming but nicer for plots
    mapping = {
        "peak_intensity": "peak_int",
        "ttm10": "TTM10",
        "tts15": "TTS15",
        "tte20": "TTE20",
        "n_events": "n_events",
        "interarrival": "Ld",
        "return_period": "Rp",
    }
    return mapping.get(name, name)


def plot_spatial_correlation_global_boxgrid(
    *,
    reg: Registry,
    obs_keys: Sequence[str],
    period_mode: str,
    model_mode: str,
    scale: int,
    pool_id: str = "standalone",
    ssi_method: str = DEFAULT_SSI_METHOD,
    scenarios: Optional[Sequence[str]] = None,
    features: Optional[Sequence[str]] = None,
    region: str = "Global",
    show_mean: bool = True,
    scenario_colors: Optional[Dict[str, str]] = None,
    group_gap: float = 0.45,
    scenario_step: float = 0.40,
    box_width: float = 0.35,
    row_height: float = 2.5,
    figsize: Optional[Tuple[float, float]] = None,
    ylim: Tuple[float, float] = (-0.2, 0.7),
    title: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    One row per obs dataset. For each row:
      - x grouped by feature
      - within each feature: boxplot across models for each scenario
      - uses ds['rho'] at region='Global' by default
    """

    scenarios = tuple(scenarios) if scenarios is not None else DEFAULT_SCENARIO_ORDER
    features = tuple(features) if features is not None else DEFAULT_FEATURE_ORDER
    colors = dict(DEFAULT_SCENARIO_COLORS)
    if scenario_colors:
        colors.update(scenario_colors)

    nrows = len(obs_keys)
    if figsize is None:
        total_units = len(features) * (len(scenarios) * scenario_step + group_gap)
        figsize = (max(8.0, 1.0 * total_units), row_height * nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=figsize, sharex=True, sharey=True)
    if nrows == 1:
        axes = np.array([axes])

    # Precompute x positions
    n_s = len(scenarios)
    positions: List[float] = []
    pos_meta: List[Tuple[str, str]] = []  # (feature, scenario)
    xticks: List[float] = []
    xticklabels: List[str] = []

    x = 1.0
    for feat in features:
        feat_positions = []
        for s in scenarios:
            positions.append(x)
            pos_meta.append((feat, s))
            feat_positions.append(x)
            x += scenario_step     # <--- was 1.0
        xticks.append(np.mean(feat_positions))
        xticklabels.append(_pretty_feature(feat))
        x += group_gap

    # Legend handles
    handles = []
    for s in scenarios:
        handles.append(mpatches.Patch(color=colors.get(s, "gray"), label=s))

    for r, obskey in enumerate(obs_keys):
        ax = axes[r]

        ref_start, ref_end = reg.resolve_ref_period(obskey, period_mode)
        p = _master_path_for_obs(
            reg=reg,
            obskey=obskey,
            period_mode=period_mode,
            model_mode=model_mode,
            pool_id=pool_id,
            scale=scale,
            ref_start=ref_start,
            ref_end=ref_end,
            ssi_method=ssi_method,
        )
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing master correlation file for {obskey}: {p}")

        ds = xr.open_dataset(p)

        # select global region
        if "region" not in ds.coords:
            raise KeyError(f"'region' coordinate not found in {p}")
        if region not in ds["region"].values:
            raise KeyError(f"Region '{region}' not found in {p}. Available: {list(ds['region'].values)}")

        rho = ds["rho"].sel(region=region)

        # boxplot data: list of arrays (one per position)
        data = []
        means = []
        for feat, scen in pos_meta:
            v = rho.sel(feature=feat, scenario=scen).values  # dim: model
            v = v[np.isfinite(v)]
            data.append(v if v.size else np.array([np.nan], dtype=float))
            means.append(np.nanmean(v) if v.size else np.nan)

        # Build boxplot
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),  # robust whiskers
        )

        # Color by scenario (each box corresponds to one scenario within a feature group)
        for i, box in enumerate(bp["boxes"]):
            _, scen = pos_meta[i]
            box.set_facecolor(colors.get(scen, "gray"))
            box.set_alpha(0.8)

        # Style medians
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(1.5)

        # Mean markers
        if show_mean:
            ax.scatter(
                positions,
                means,
                marker="D",
                s=18,
                zorder=3,
                edgecolors="black",
                facecolors="black",
                linewidths=0.4,
            )

        ax.axhline(0.0, linewidth=1.0)
        ax.set_ylim(*ylim)
        ax.set_ylabel(obskey)

        # small subtitle per row
        ax.text(
            0.01,
            0.92,
            f"{period_mode} | {model_mode} | ref {ref_start}..{ref_end} | {region}",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
        )

        if r == 0:
            ax.legend(handles=handles, loc="upper right", frameon=False, ncol=4)

    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels(xticklabels, rotation=0)
    axes[-1].set_xlabel("Drought feature")
    fig.supylabel("Weighted Spearman rho (Global)", x=0.01)

    if title:
        fig.suptitle(title, y=0.995)

    fig.tight_layout()
    return fig, axes
# src/sm_attribution/viz/droughtfeat_ar6_metrics_plots.py
from __future__ import annotations

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
    "obsclim_histsoc": "#2E7D32",   # green
    "obsclim_1901soc": "#C8B53B",   # yellow-ish
    "counterclim_histsoc": "#2F5D8A",  # blue
    "counterclim_1901soc": "#D1495B",  # red
}

# Master file variables (as written by orchestrator)
ALLOWED_METRICS = ("spearman_rank", "pearson_z", "rmse_iqr")


def _format_from_template(tmpl: str, reg: Registry, **kw) -> str:
    paths = reg.cfg_dict.get("paths", {}) or {}
    for k, v in paths.items():
        tmpl = tmpl.replace("{paths." + k + "}", v)
    return tmpl.format(**kw)


def _pool_id_for_models(model_mode: str, pool_scenarios: Optional[Sequence[str]], fixed_ref_scenario: Optional[str] = None) -> str:
    if model_mode == "standalone":
        return "standalone"
    if model_mode == "fixed":
        return fixed_ref_scenario or "UNKNOWN_REF"
    if pool_scenarios is None:
        return "ALL_SCENARIOS"
    return "__".join(sorted(pool_scenarios))


def _master_path_for_obs_ar6_metrics(
    *,
    reg: Registry,
    obskey: str,
    period_mode: str,
    model_mode: str,
    pool_id: str,
    scale: int,
    ref_start: str,
    ref_end: str,
    ssi_method: str = DEFAULT_SSI_METHOD,
) -> str:
    metrics = reg.cfg_dict.get("metrics", {}) or {}
    if "droughtfeat_ar6_metrics_master" not in metrics:
        raise KeyError("Missing metrics.droughtfeat_ar6_metrics_master in registry config.")

    tmpl = metrics["droughtfeat_ar6_metrics_master"]
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


def _metric_label(metric: str) -> str:
    if metric == "spearman_rank":
        return "Spearman (rank agreement across AR6 regions)"
    if metric == "pearson_z":
        return "Pearson on z-scored regional vectors"
    if metric == "rmse_iqr":
        return "RMSE on robust-normalized (median/IQR) regional vectors"
    return metric


def plot_droughtfeat_ar6_metrics_boxgrid(
    *,
    reg: Registry,
    obs_keys: Sequence[str],
    period_mode: str,
    model_mode: str,   # "pooled", "standalone", or "fixed"
    scale: int,
    metric: str,       # one of: spearman_rank, pearson_z, rmse_iqr
    pool_scenarios: Optional[Sequence[str]] = None,
    fixed_ref_scenario: Optional[str] = None,
    ssi_method: str = DEFAULT_SSI_METHOD,
    scenarios: Optional[Sequence[str]] = None,
    features: Optional[Sequence[str]] = None,
    show_mean: bool = True,
    scenario_colors: Optional[Dict[str, str]] = None,
    group_gap: float = 0.45,
    scenario_step: float = 0.40,
    box_width: float = 0.26,
    row_height: float = 2.6,
    figsize: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    ncols: int = 2,
) -> Tuple[plt.Figure, np.ndarray]:

    if metric not in ALLOWED_METRICS:
        raise ValueError(f"metric must be one of {ALLOWED_METRICS}, got '{metric}'")

    scenarios = tuple(scenarios) if scenarios is not None else DEFAULT_SCENARIO_ORDER
    features = tuple(features) if features is not None else DEFAULT_FEATURE_ORDER

    colors = dict(DEFAULT_SCENARIO_COLORS)
    if scenario_colors:
        colors.update(scenario_colors)

    pool_id = _pool_id_for_models(model_mode, pool_scenarios, fixed_ref_scenario)

    # Default y-lims depending on metric
    if ylim is None:
        if metric in ("spearman_rank", "pearson_z"):
            ylim = (-0.2, 0.8)
        else:
            ylim = (0.0, 2.0)

    n_panels = len(obs_keys)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n_panels / ncols))

    # Figure sizing
    if figsize is None:
        total_units = len(features) * (len(scenarios) * scenario_step + group_gap)
        base_w = max(10.0, 1.0 * total_units)
        figsize = (base_w * (ncols / 1.6), row_height * nrows)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)

    # Precompute x positions
    positions: List[float] = []
    pos_meta: List[Tuple[str, str]] = []
    xticks: List[float] = []
    xticklabels: List[str] = []

    x = 1.0
    for feat in features:
        feat_positions = []
        for s in scenarios:
            positions.append(x)
            pos_meta.append((feat, s))
            feat_positions.append(x)
            x += scenario_step
        xticks.append(float(np.mean(feat_positions)))
        xticklabels.append(_pretty_feature(feat))
        x += group_gap

    # Figure-level legend handles
    handles = [mpatches.Patch(color=colors.get(s, "gray"), label=s) for s in scenarios]

    # Plot each obs dataset
    for i, obskey in enumerate(obs_keys):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        ref_start, ref_end = reg.resolve_ref_period(obskey, period_mode)
        p = _master_path_for_obs_ar6_metrics(
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
            raise FileNotFoundError(f"Missing AR6 metrics master for {obskey}: {p}")

        ds = xr.open_dataset(p)
        if metric not in ds.data_vars:
            raise KeyError(f"'{metric}' not found in {p}. Found vars: {list(ds.data_vars)}")

        arr = ds[metric]  # dims: model, scenario, feature

        data = []
        means = []
        for feat, scen in pos_meta:
            v = arr.sel(feature=feat, scenario=scen).values  # dim: model
            v = v[np.isfinite(v)]
            data.append(v if v.size else np.array([np.nan], dtype=float))
            means.append(np.nanmean(v) if v.size else np.nan)

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
        )

        # Color boxes by scenario
        for j, box in enumerate(bp["boxes"]):
            _, scen = pos_meta[j]
            box.set_facecolor(colors.get(scen, "gray"))
            box.set_alpha(0.8)

        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(1.5)

        if show_mean:
            ax.scatter(
                positions,
                means,
                marker="D",
                s=16,
                zorder=3,
                edgecolors="black",
                facecolors="black",
                linewidths=0.4,
            )

        if metric in ("spearman_rank", "pearson_z"):
            ax.axhline(0.0, linewidth=1.0)

        ax.set_ylim(*ylim)
        ax.set_ylabel(obskey)

        ax.text(
            0.01,
            0.97,
            f"{period_mode} | {model_mode} | ref {ref_start}..{ref_end}",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
        )

        # Show the drought-feature group labels under each subplot
        # --- x ticks/labels on EVERY subplot (sharex normally hides them) ---
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=0)

        # force tick labels to show even with sharex=True
        ax.tick_params(axis="x", which="both", labelbottom=True)

        # remove the x-axis title completely
        ax.set_xlabel("")

    # Hide unused axes
    for j in range(n_panels, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].set_visible(False)

    fig.supylabel(_metric_label(metric), x=0.02)

    # Title + legend placement (legend below title, outside axes)
    if title:
        fig.suptitle(title, y=0.985)

    # Legend below title; adjust ncol to your taste
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.2,
    )

    # Reserve top margin for title+legend
    fig.tight_layout(rect=(0.02, 0.02, 1.00, 0.97))
    return fig, axes
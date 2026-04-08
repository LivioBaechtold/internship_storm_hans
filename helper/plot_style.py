# plot_style.py
"""
Matplotlib style defaults and the two-panel Storm Hans figure
All purely visual/plotting code lives here; no statistical logic
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import xarray as xr

from return_period import (
    get_annual_maxima,
    get_event_annual_max,
    fit_gev,
    gev_return_level,
    estimate_return_period,
    weibull_plotting_positions,   # add this line
)

# Matplotlib style defaults
plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi":    600,})


# ──────────────────────────────────────
# Create 2-Panel figure for the main Storm Hans analysis (timeseries + return-period plot)
# ────────────────────────────────────── 

def make_figure(
    da: xr.DataArray,
    catchment_title: str,
    dataset: str,
    resolution: str,
    window_days: int,
    event_year: int,
    out_paths: list,
    exclude_event_year_from_fit: bool = True,
) -> None:

    # ── Compute everything
    annual_max_all = get_annual_maxima(da)
    event_val, event_date_actual = get_event_annual_max(da, event_year)

    fit_sample = (
        annual_max_all.drop(index=event_year, errors="ignore")
        if exclude_event_year_from_fit
        else annual_max_all)

    if len(fit_sample) < 10:
        raise ValueError("Too few annual maxima left for a stable GEV fit.")

    c, loc, scale = fit_gev(fit_sample)
    event_T = estimate_return_period(event_val, c, loc, scale)

    if np.isfinite(event_T):
        event_T_rounded = int(round(event_T))
        event_T_str     = f"T ≈ {event_T_rounded} yr"
        event_T_title   = f"{event_T_rounded} year"
    else:
        # Should not occur with inclusive fit, but guard defensively
        event_T_rounded = None
        event_T_str     = "T > record (beyond GEV range)"
        event_T_title   = "beyond record (∞)"

    # Empirical plotting positions based on the fit sample
    vals_desc, T_all = weibull_plotting_positions(fit_sample)
    annual_desc = pd.Series(vals_desc, index=fit_sample.sort_values(ascending=False).index)
    emp_T_all   = pd.Series(T_all,     index=annual_desc.index)

    annual_scatter = annual_desc.drop(index=event_year, errors="ignore")
    emp_T_scatter  = emp_T_all.drop(index=event_year, errors="ignore")

    T_min   = max(1.01, float(emp_T_scatter.min()))
    T_curve = np.logspace(np.log10(T_min), np.log10(2000.0), 500)
    x_curve = gev_return_level(c, loc, scale, T_curve)

    finite  = np.isfinite(x_curve)
    T_curve = T_curve[finite]
    x_curve = x_curve[finite]

    # ── Layout
    acc_label = f"{window_days}-day"

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    fig.suptitle(
        f"Analysis Storm Hans: {dataset.upper()}/{resolution}/{acc_label} —  {catchment_title}",
        fontsize=16, fontweight="normal", y=0.98,)

    # ── Panel A: Full time series
    ax = axes[0]
    ts = da.to_series()

    ax.fill_between(ts.index, ts.values, color="steelblue", alpha=0.65, linewidth=0)
    ax.plot(ts.index, ts.values, color="steelblue", linewidth=0.4)

    ax.plot(
        event_date_actual, event_val, "o", color="red", markersize=7, zorder=5,
        label=f"Storm Hans ({event_date_actual.date()}): {event_val:.1f} mm",)
    ax.legend(fontsize=12, loc="upper left", frameon=False)

    ax.set_title(
        f"A)  Weighted Catchment {acc_label.capitalize()} Accumulated Precipitation Time Series",
        loc="left", x=-0.10, pad=12, fontsize=14, fontweight="normal",)

    ax.set_ylabel(f"{acc_label} Accumulation (mm)")
    ax.set_xlim(ts.index[0], ts.index[-1])
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=11, labelbottom=True)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel B: Return-period plot
    ax = axes[1]

    ax.scatter(
        emp_T_scatter.values, annual_scatter.values,
        color="steelblue", s=18, zorder=3, alpha=0.85,
        label="Empirical (Weibull PP)",)
    ax.plot(T_curve, x_curve, color="black", linewidth=1.5, zorder=4, label="GEV fit")

    ax.axhline(event_val, color="black", linestyle="--", linewidth=0.9, zorder=2)

    # Only draw the vertical return-period line if T is finite
    if np.isfinite(event_T):
        ax.axvline(event_T, color="black", linestyle="--", linewidth=0.9, zorder=2)
        ax.plot(
            event_T, event_val, "o", color="red", markersize=8, zorder=5,
            label=f"Storm Hans: {event_T_str}",)
    else:
        # Event is beyond the plotted x-range; use legend-only marker + annotation
        ax.plot(
            [], [], "o", color="red", markersize=8,
            label=f"Storm Hans: {event_T_str}",)
        ax.annotate(
            f"{event_val:.1f} mm  ({event_T_str})",
            xy=(2000, event_val),
            xytext=(-10, 6), textcoords="offset points",
            ha="right", fontsize=10, color="red",)

    ax.set_xscale("log")
    ax.set_title(
        f"B)  Weighted Catchment {acc_label.capitalize()} Accumulated Precipitation, "
        f"Return Period Storm Hans: {event_T_title}",
        loc="left", x=-0.10, pad=12, fontsize=14, fontweight="normal",)

    ax.set_xlabel("Return Period (years)")
    ax.set_ylabel(f"{acc_label} Accumulation (mm)")
    ax.legend(fontsize=12, loc="upper left", bbox_to_anchor=(0.0, 0.8), frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _b_ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    ax.set_xticks(_b_ticks)
    ax.set_xticklabels(
        [str(t) for t in _b_ticks],
        fontsize=11, fontstyle="normal", rotation=0, ha="center",)

    # Save to all figure roots
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), format="pdf", bbox_inches="tight")
        print(f"    [fig]   Saved → {out_path}")
    plt.close(fig)


# ──────────────────────────────────────
# Create Panel and Plotstyle for SMILE ensemble figure (1 panel!)
# ──────────────────────────────────────

def make_smile_return_period_figure(
    annual_max_yearly: pd.Series,
    catchment_title: str,
    dataset: str,
    window_days: int,
    reference_mode: str,
    reference_value: float,
    reference_label: str,
    out_paths: list,) -> None:
    """
    Single-panel SMILE return-period figure.

    reference_mode
    --------------
    "precip_value"   : reference_value is Storm Hans precipitation from ERA5/0.5°;
                       plot its return period in the climate-model GEV fit

    "return_period"  : reference_value is Storm Hans return period from ERA5/0.5°;
                       plot the climate-model precipitation belonging to that T
    """
    acc_label = f"{window_days}-day"

    label_map = {
        "cesm2_le": "CESM2-LE/1°x1°",
        "gfdl_spear_med_le": "GFDL SPEAR-LE/1°x1°",}
    ds_label = label_map.get(dataset, dataset.upper())

    if len(annual_max_yearly) < 10:
        raise ValueError("Too few yearly maxima left for a stable GEV fit.")

    c, loc, scale = fit_gev(annual_max_yearly)

    vals_desc, T_all = weibull_plotting_positions(annual_max_yearly)
    annual_desc = pd.Series(vals_desc, index=annual_max_yearly.sort_values(ascending=False).index)
    emp_T       = pd.Series(T_all,     index=annual_desc.index)

    T_min = max(1.01, float(emp_T.min()))
    T_curve = np.logspace(np.log10(T_min), np.log10(2000.0), 500)
    x_curve = gev_return_level(c, loc, scale, T_curve)
    finite = np.isfinite(x_curve)
    T_curve = T_curve[finite]
    x_curve = x_curve[finite]

    if reference_mode == "precip_value":
        ref_precip = float(reference_value)
        ref_T = estimate_return_period(ref_precip, c, loc, scale)

        if np.isfinite(ref_T):
            title_tail = f"{int(round(ref_T))} year"
        else:
            title_tail = "beyond record (∞)"
        # Always show precipitation in mm in the point label;
        # the return period is already in the panel title.
        point_label = f"Storm Hans ({reference_label}): {ref_precip:.1f} mm"

    elif reference_mode == "return_period":
        ref_T = float(reference_value)
        ref_precip = float(gev_return_level(c, loc, scale, np.asarray([ref_T]))[0])

        if np.isfinite(ref_T):
            title_tail = f"{int(round(ref_T))} year"
            point_label = (
                f"Storm Hans ({reference_label} T): {ref_precip:.1f} mm")
        else:
            title_tail = "beyond record (∞)"
            point_label = f"Storm Hans ({reference_label} T): beyond record"
    else:
        raise ValueError("reference_mode must be 'precip_value' or 'return_period'.")

    fig, ax = plt.subplots(1, 1, figsize=(12, 5.2))
    fig.suptitle(
        f"Analysis Storm Hans: {ds_label}/{acc_label} —  {catchment_title}",
        fontsize=16, fontweight="normal", y=0.98,)

    ax.scatter(
        emp_T.values, annual_desc.values,
        color="steelblue", s=18, zorder=3, alpha=0.85,
        label="Empirical (Weibull PP)",)
    ax.plot(
        T_curve, x_curve,
        color="black", linewidth=1.5, zorder=4, label="GEV fit")

    ax.axhline(ref_precip, color="black", linestyle="--", linewidth=0.9, zorder=2)

    if np.isfinite(ref_T):
        ax.axvline(ref_T, color="black", linestyle="--", linewidth=0.9, zorder=2)
        ax.plot(
            ref_T, ref_precip, "o",
            color="red", markersize=8, zorder=5,
            label=point_label,)
    else:
        ax.plot([], [], "o", color="red", markersize=8, label=point_label)
        ax.annotate(
            f"{ref_precip:.1f} mm",
            xy=(2000, ref_precip),
            xytext=(-10, 6), textcoords="offset points",
            ha="right", fontsize=10, color="red",)

    ax.set_xscale("log")
    ax.set_title(
        f"Weighted Catchment {acc_label.capitalize()} Accumulated Precipitation, "
        f"Return Period Storm Hans: {title_tail}",
        loc="left", x=-0.02, pad=12, fontsize=14, fontweight="normal",)

    ax.set_xlabel("Return Period (years)")
    ax.set_ylabel(f"{acc_label} Accumulation (mm)")
    ax.legend(fontsize=12, loc="upper left", bbox_to_anchor=(0.0, 0.82), frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _b_ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    ax.set_xticks(_b_ticks)
    ax.set_xticklabels(
        [str(t) for t in _b_ticks],
        fontsize=11, fontstyle="normal", rotation=0, ha="center",)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), format="pdf", bbox_inches="tight")
        print(f"    [fig]   Saved → {out_path}")
    plt.close(fig)

# ── Map helper ────────────────────────────────────────────────────────────────

def draw_catchments(
    ax,
    catchments: dict,
    data_crs,
    catchment_labels: dict,
    label_offsets: dict,
) -> None:
    """
    Draw catchment outlines and annotated labels on a Cartopy axes.

    Parameters
    ----------
    ax             : Cartopy GeoAxes
    catchments     : {slug: GeoDataFrame}  — pre-loaded polygons in EPSG:4326
    data_crs       : cartopy CRS of the GeoDataFrames (usually PlateCarree())
    catchment_labels : {slug: short_label_string}
    label_offsets  : {slug: (dx, dy)} offset in display-points from centroid
    """
    for slug, gdf in catchments.items():
        # Outline
        ax.add_geometries(
            gdf.geometry,
            crs=data_crs,
            facecolor="none",
            edgecolor="red",
            linewidth=1.2,
            zorder=4,
        )
        # Representative point for label placement
        try:
            geom = gdf.geometry.union_all()
        except AttributeError:
            geom = gdf.geometry.unary_union

        pt = geom.representative_point()
        dx, dy = label_offsets.get(slug, (70, 40))

        ax.annotate(
            catchment_labels.get(slug, slug),
            xy=(pt.x, pt.y),
            xycoords=data_crs._as_mpl_transform(ax),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor="white",
                edgecolor="none",
                alpha=0.82,
            ),
            arrowprops=dict(
                arrowstyle="->",
                color="black",
                lw=0.9,
            ),
            zorder=5,)


# ═══════════════════════════════════════════════════════════════════════════
# Climate-model evaluation figures
# ═══════════════════════════════════════════════════════════════════════════

# Consistent colors and labels across all evaluation figures
MODEL_COLORS = {
    "senorge":           "#E8564A",   # red/salmon  — matches Image 3 pink
    "era5_0.5":          "#4682B4",   # steel blue
    "era5_0.25":         "#008B8B",   # teal
    "cesm2_le":          "#2CA02C",   # green
    "gfdl_spear_med_le": "#FF7F0E",   # orange
}
MODEL_LABELS = {
    "senorge":           "seNorge / 1 km",
    "era5_0.5":          "ERA5 / 0.5°",
    "era5_0.25":         "ERA5 / 0.25°",
    "cesm2_le":          "CESM2-LE / 1°",
    "gfdl_spear_med_le": "GFDL-SPEAR / 1°",
}
# Canonical display order (left to right in box plot, top to bottom in legend)
MODEL_ORDER = ["senorge", "era5_0.5", "era5_0.25", "cesm2_le", "gfdl_spear_med_le"]


def make_distribution_figure(
    annual_maxima: dict,
    window_days: int,
    out_paths: list,
) -> None:
    """
    Two-panel distribution figure for all models

    Panel A : Filled density curves 
    Panel B : Horizontal box plots

    Parameters
    ----------
    annual_maxima : dict
  
    window_days : int

    out_paths : list of Path

    """
    from scipy.stats import gaussian_kde

    acc_label = f"{window_days}-day"
    models = [k for k in MODEL_ORDER if k in annual_maxima]

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    fig.suptitle(
        f"{acc_label.capitalize()} Accumulated Precipitation Distribution\n"
        "All Catchments Pooled",
        fontsize=15, fontweight="normal", y=0.99,
    )

    # ── Panel A: KDE density curves ──────────────────────────────────────────
    ax = axes[0]
    all_vals = np.concatenate([annual_maxima[k] for k in models])
    x_max = float(np.nanpercentile(all_vals, 99.5)) * 1.05
    x_grid = np.linspace(0.0, x_max, 600)

    for key in models:
        data = annual_maxima[key][np.isfinite(annual_maxima[key])]
        kde = gaussian_kde(data, bw_method="scott")
        density = kde(x_grid)
        col = MODEL_COLORS[key]
        lbl = MODEL_LABELS[key]
        ax.fill_between(x_grid, density * 100, alpha=0.20, color=col)
        ax.plot(x_grid, density * 100, color=col, linewidth=2.0, label=lbl)

    ax.set_title("A)  Frequency Distribution (KDE)",
                 loc="left", pad=8, fontsize=13, fontweight="normal")
    ax.set_xlabel(f"{acc_label.capitalize()} Accumulated Precipitation (mm)", fontsize=11)
    ax.set_ylabel("Frequency (%)", fontsize=11)
    ax.set_xlim(left=0, right=x_max)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel B: Horizontal box plots ────────────────────────────────────────
    ax = axes[1]
    box_data   = [annual_maxima[k][np.isfinite(annual_maxima[k])] for k in models]
    box_labels = [MODEL_LABELS[k] for k in models]
    box_colors = [MODEL_COLORS[k] for k in models]

    bp = ax.boxplot(
        box_data,
        vert=False,
        patch_artist=True,
        labels=box_labels,
        widths=0.5,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3.5, linestyle="none", alpha=0.55),
    )
    for patch, col in zip(bp["boxes"], box_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.45)
    for flier, col in zip(bp["fliers"], box_colors):
        flier.set(markerfacecolor=col, markeredgecolor=col)

    ax.set_title("B)  Box Plots (ordered by display)",
                 loc="left", pad=8, fontsize=13, fontweight="normal")
    ax.set_xlabel(f"{acc_label.capitalize()} Accumulated Precipitation (mm)", fontsize=11)
    ax.set_xlim(left=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), format="pdf", bbox_inches="tight")
        print(f"    [fig]   Saved → {out_path}")
    plt.close(fig)


def make_qq_figure(
    climate_key: str,
    climate_data: np.ndarray,
    reanalysis_data: dict,
    window_days: int,
    out_paths: list,
) -> None:
    """
    Quantile–Quantile mapping: climate model (x-axis) vs each reanalysis model (y-axis)

    Parameters
    ----------
    climate_key : str

    climate_data : np.ndarray

    reanalysis_data : dict

    window_days : int

    out_paths : list of Path
    """
    acc_label      = f"{window_days}-day"
    climate_label  = MODEL_LABELS.get(climate_key, climate_key)

    probs   = np.linspace(0.01, 0.99, 300)
    c_clean = climate_data[np.isfinite(climate_data)]
    q_clim  = np.quantile(c_clean, probs)

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    fig.suptitle(
        f"{acc_label.capitalize()} Accumulated Precipitation Q-Q Mapping\n"
        f"All Catchments Pooled",
        fontsize=15, fontweight="normal", y=0.99,
    )

    all_vals = list(c_clean)
    reanalysis_order = [k for k in MODEL_ORDER if k in reanalysis_data]

    for key in reanalysis_order:
        data  = reanalysis_data[key][np.isfinite(reanalysis_data[key])]
        q_ref = np.quantile(data, probs)
        all_vals.extend(data.tolist())
        ax.plot(q_clim, q_ref,
                color=MODEL_COLORS.get(key, "grey"),
                linewidth=2.0,
                label=MODEL_LABELS.get(key, key),
                alpha=0.90)

    # 1:1 perfect-match reference line
    ax_max = float(np.nanpercentile(all_vals, 99.5)) * 1.05
    ref    = np.array([0.0, ax_max])
    ax.plot(ref, ref,
            color="black", linewidth=2.8, linestyle="--",
            label="1 : 1 line (perfect match)", zorder=5)

    ax.set_xlabel(
        f"{climate_label}\n{acc_label.capitalize()} Precipitation (mm)",
        fontsize=12)
    ax.set_ylabel(
        f"Reanalysis / Other Models\n{acc_label.capitalize()} Precipitation (mm)",
        fontsize=12)
    ax.set_title(
        f"Quantile–Quantile Comparison:  {climate_label}  vs.  Reanalysis",
        loc="left", x=0.0, pad=10, fontsize=13, fontweight="normal")
    ax.legend(fontsize=10, frameon=False, loc="upper left")
    ax.set_xlim(0, ax_max)
    ax.set_ylim(0, ax_max)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), format="pdf", bbox_inches="tight")
        print(f"    [fig]   Saved → {out_path}")
    plt.close(fig)
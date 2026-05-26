# plot_style.py
"""
Matplotlib style defaults and the two-panel Storm Hans figure
All purely visual/plotting code lives here; no statistical logic
"""

import matplotlib
matplotlib.use('Agg')  # Must be BEFORE pyplot import — forces non-interactive backend
                       # on headless HPC (NIRD). Prevents X11/Qt hang on first import.

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
    exclude_event_year_from_fit: bool = False,
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

    if dataset == "senorge":
        dataset_display = "SeNorge/1kmx1km"
    elif dataset == "era5" and resolution == "0.25x0.25":
        dataset_display = "ERA5/0.25°x0.25°"
    elif dataset == "era5" and resolution == "0.5x0.5":
        dataset_display = "ERA5/0.5°x0.5°"
    else:
        dataset_display = f"{dataset.upper()}/{resolution}" if resolution else dataset.upper()

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    fig.suptitle(
        f"Analysis Storm Hans: {dataset_display}/{acc_label} —  {catchment_title}",
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
    annual_max_pooled: pd.Series,
    catchment_title: str,
    dataset: str,
    window_days: int,
    reference_value: float,
    reference_label: str,
    out_paths: list,
) -> None:
    """
    Single-panel SMILE return-period figure.

    reference_value:
        Storm Hans precipitation value from the selected reference dataset.
    """
    acc_label = f"{window_days}-day"

    label_map = {
        "cesm2_le": "CESM2-LE / 0.94° x 1.25°",
        "gfdl_spear_med_le": "GFDL-SPEAR / 0.5° x 0.625°",
    }
    ds_label = label_map.get(dataset, dataset.upper())

    if len(annual_max_pooled) < 10:
        raise ValueError("Too few yearly maxima left for a stable GEV fit.")

    c, loc, scale = fit_gev(annual_max_pooled)

    vals_desc, T_all = weibull_plotting_positions(annual_max_pooled)
    annual_desc = pd.Series(vals_desc)
    emp_T = pd.Series(T_all)

    T_min = max(1.01, float(emp_T.min()))
    T_curve = np.logspace(np.log10(T_min), np.log10(2000.0), 500)
    x_curve = gev_return_level(c, loc, scale, T_curve)
    finite = np.isfinite(x_curve)
    T_curve = T_curve[finite]
    x_curve = x_curve[finite]

    ref_precip = float(reference_value)
    ref_T = estimate_return_period(ref_precip, c, loc, scale)

    if np.isfinite(ref_T):
        title_tail = f"{int(round(ref_T))} year"
    else:
        title_tail = "beyond record (∞)"

    point_label = f"Storm Hans ({reference_label}): {ref_precip:.1f} mm"

    fig, ax = plt.subplots(1, 1, figsize=(12, 5.2))
    fig.suptitle(
        f"Analysis Storm Hans: {ds_label}/{acc_label} —  {catchment_title}",
        fontsize=16, fontweight="normal", y=0.98,
    )

    ax.scatter(
        emp_T.values, annual_desc.values,
        color="steelblue", s=18, zorder=3, alpha=0.85,
        label="Empirical (Weibull PP)",
    )
    ax.plot(
        T_curve, x_curve,
        color="black", linewidth=1.5, zorder=4, label="GEV fit",
    )

    ax.axhline(ref_precip, color="black", linestyle="--", linewidth=0.9, zorder=2)

    if np.isfinite(ref_T):
        ax.axvline(ref_T, color="black", linestyle="--", linewidth=0.9, zorder=2)
        ax.plot(
            ref_T, ref_precip, "o",
            color="red", markersize=8, zorder=5,
            label=point_label,
        )
    else:
        ax.plot([], [], "o", color="red", markersize=8, label=point_label)
        ax.annotate(
            f"{ref_precip:.1f} mm",
            xy=(2000, ref_precip),
            xytext=(-10, 6), textcoords="offset points",
            ha="right", fontsize=10, color="red",
        )

    ax.set_xscale("log")
    ax.set_title(
        f"Weighted Catchment {acc_label.capitalize()} Accumulated Precipitation, "
        f"Return Period Storm Hans: {title_tail}",
        loc="left", x=-0.02, pad=12, fontsize=14, fontweight="normal",
    )

    ax.set_xlabel("Return Period (years)")
    ax.set_ylabel(f"{acc_label} Accumulation (mm)")
    ax.legend(fontsize=12, loc="upper left", bbox_to_anchor=(0.0, 0.82), frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _b_ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    ax.set_xticks(_b_ticks)
    ax.set_xticklabels(
        [str(t) for t in _b_ticks],
        fontsize=11, fontstyle="normal", rotation=0, ha="center",
    )

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
            zorder=4,)
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
                alpha=0.82,),
            arrowprops=dict(
                arrowstyle="->",
                color="black",
                lw=0.9,),
            zorder=5,)


# ═══════════════════════════════════════════════════════════════════════════
# Climate-model evaluation figures
# ═══════════════════════════════════════════════════════════════════════════

# Consistent colors and labels across all evaluation figures
# Ordered from highest / finest reference resolution to coarser / model data:
# seNorge -> ERA5 0.25 -> ERA5 0.5 -> GFDL-SPEAR -> CESM2-LE
MODEL_COLORS = {
    "senorge":           "#D73027",   # deep red
    "era5_0.25":         "#F28E2B",   # orange
    "era5_0.5":          "#E3B505",   # golden yellow
    "gfdl_spear_med_le": "#4DAF4A",   # green
    "cesm2_le":          "#2C7BB6",   # blue
}
MODEL_LABELS = {
    "senorge":           "SeNorge / 1 km",
    "era5_0.25":         "ERA5 / 0.25°",
    "era5_0.5":          "ERA5 / 0.5°",
    "gfdl_spear_med_le": "GFDL-SPEAR / 0.5° x 0.625°",
    "cesm2_le":          "CESM2-LE / 0.94° x 1.25°",
}
# Canonical display order (left to right in box plot, top to bottom in legend)
MODEL_ORDER = ["senorge", "era5_0.25", "era5_0.5", "gfdl_spear_med_le", "cesm2_le"]

# Define the two-panel distribution figure for all models (density + boxplot)
def make_distribution_figure(
    annual_maxima: dict,
    window_days: int,
    out_paths: list,
    data_type: str = "annual_max",   # "annual_max" or "daily"
    catchment_title: str = "",
) -> None:
    """
    Two-panel distribution figure for all models

    Panel A : Filled density curves
    Panel B : Horizontal box plots

    """
    from scipy.stats import gaussian_kde

    acc_label = f"{window_days}-day"
    models = [k for k in MODEL_ORDER if k in annual_maxima]

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    if data_type == "daily":
        data_label     = f"Daily {acc_label}"
        x_axis_label   = f"Daily {acc_label} Accumulated Precipitation (mm)"
    else:
        data_label     = f"Annual Maximal {acc_label}"
        x_axis_label   = f"Annual Maximal {acc_label} Accumulated Precipitation (mm)"

    title_suffix = f"  —  {catchment_title}" if catchment_title else ""
    fig.suptitle(
        f"Distribution of {data_label} Accumulated Precipitation{title_suffix}",
        fontsize=16,
        fontweight="normal",
        y=0.98,)

    # Panel titles in figure coordinates (not axes coordinates)
    # Panel titles in figure coordinates
    fig.text(
        -0.005, 0.875,
        "A)  Frequency Distribution",
        ha="left", va="center",
        fontsize=14, fontweight="normal",)
    fig.text(
        -0.005, 0.425,
        "B)  Boxplot Distribution",
        ha="left", va="center",
        fontsize=14, fontweight="normal",)

    # ── Panel A: density curves ───────────────────────────────────────────────
    ax = axes[0]
    all_vals = np.concatenate([annual_maxima[k] for k in models])
    all_vals = all_vals[np.isfinite(all_vals)]

    # Show full visible range up to the true maximum
    x_max = float(np.nanmax(all_vals)) * 1.03
    x_grid = np.linspace(0.0, x_max, 800)

    for key in models:
        data = annual_maxima[key][np.isfinite(annual_maxima[key])]
        if data.size < 2:
            continue

        kde = gaussian_kde(data, bw_method="scott")
        density = kde(x_grid)

        ax.fill_between(x_grid, density * 100,
            alpha=0.20,
            color=MODEL_COLORS[key],)
        ax.plot(
            x_grid, density * 100,
            color=MODEL_COLORS[key],
            linewidth=2.0,
            label=MODEL_LABELS[key],)
    
    ax.set_xlabel(x_axis_label, fontsize=11)
    ax.set_ylabel("Frequency (%)", fontsize=11, labelpad=2)
    ax.set_xlim(left=0, right=x_max)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(fontsize=12, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel B: horizontal box plots ────────────────────────────────────────
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
        flierprops=dict(marker="o", markersize=3.5, linestyle="none", alpha=0.55),)

    for patch, col in zip(bp["boxes"], box_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.45)

    for flier, col in zip(bp["fliers"], box_colors):
        flier.set(markerfacecolor=col, markeredgecolor=col)

    ax.set_xlabel(x_axis_label, fontsize=11)
    ax.set_xlim(left=0, right=x_max)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel A: move the whole subplot further left
    axes[0].set_position([0.03, 0.54, 0.82, 0.30])
    axes[1].set_position([0.19, 0.09, 0.73, 0.30])
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            str(out_path),
            format="pdf",
            dpi=plt.rcParams["figure.dpi"],
            bbox_inches="tight",)
        print(f"    [fig]   Saved → {out_path}")
    plt.close(fig)


# Define the quantile-quantile comparison figure for all models (scatter of percentiles)
def make_qq_figure(
    climate_key: str,
    climate_data: np.ndarray,
    reanalysis_data: dict,
    window_days: int,
    out_paths: list,
    data_type: str = "annual_max",
    catchment_title: str = "",
) -> None:
    """
    Percentile-mapping comparison.

    x-axis:
        percentile in the climate model

    y-axis:
        percentile rank in each reanalysis model of the climate-model
        precipitation value at that percentile

    Example:
        x = 10 means the 10th percentile precipitation value in the climate model.
        y = 14 for a reanalysis line means that same precipitation value lies at
        the 14th percentile in that reanalysis distribution.
    """
    acc_label = f"{window_days}-day"
    climate_label = MODEL_LABELS.get(climate_key, climate_key)

    probs = np.linspace(0.01, 0.99, 199)
    x_pct = probs * 100.0

    c_clean = np.asarray(climate_data, dtype=float)
    c_clean = c_clean[np.isfinite(c_clean)]
    q_clim = np.quantile(c_clean, probs)

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    data_label = "Annual Maximal" if data_type == "annual_max" else "Daily"
    title_suffix = f"  —  {catchment_title}" if catchment_title else ""
    fig.suptitle(
        f"Percentile Mapping of {data_label} {acc_label} Accumulated Precipitation{title_suffix}",
        fontsize=16,
        fontweight="normal",
        y=0.98,)

    reanalysis_order = [k for k in MODEL_ORDER if k in reanalysis_data]

    for key in reanalysis_order:
        data = np.asarray(reanalysis_data[key], dtype=float)
        data = np.sort(data[np.isfinite(data)])
        if data.size == 0:
            continue

        # Empirical percentile position of the climate-model quantile values
        # inside the reanalysis distribution
        y_pct = 100.0 * np.searchsorted(data, q_clim, side="right") / data.size

        ax.plot(
            x_pct,
            y_pct,
            color=MODEL_COLORS.get(key, "grey"),
            linewidth=2.2,
            label=MODEL_LABELS.get(key, key),
            alpha=0.95,)

    # 1:1 line = perfect percentile agreement
    ref = np.array([0.0, 100.0])
    ax.plot(
        ref,
        ref,
        color="black",
        linewidth=2.8,
        linestyle="--",
        label="1 : 1 line (perfect match)",
        zorder=5,)

    ax.set_xlabel(
        f"Percentile in {climate_label} (%)",
        fontsize=12,)
    ax.set_ylabel(
        "Corresponding Percentile in other Models (%)",
        fontsize=12,)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(0, 101, 10))
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(fontsize=12, frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            str(out_path),
            format="pdf",
            dpi=plt.rcParams["figure.dpi"],
            bbox_inches="tight",
        )
        print(f"    [fig]   Saved → {out_path}")
    plt.close(fig)



# Distribution Difference figure

def make_distribution_difference_figure(
    bin_centers: np.ndarray,
    avg_diff: np.ndarray,
    n_pairs: int,
    window_before: int,
    window_after: int,
    dataset: str,
    window_days: int,
    catchment_title: str,
    start_year: int,
    end_year: int,
    out_paths: list,
    bin_width_mm: float = 5.0,
    data_type: str = "annual_max",
) -> None:
    """
    Single-panel bar chart of the average normalised distribution difference
    (after − before) across all non-overlapping consecutive window pairs.

    y > 0  (blue) → the "after" window has proportionally MORE events in that bin
    y < 0  (red)  → the "before" window has proportionally MORE events in that bin
    """
    acc_label = f"{window_days}-day"

    label_map = {
    "cesm2_le":          "CESM2-LE / 0.94° x 1.25°",
    "gfdl_spear_med_le": "GFDL-SPEAR / 0.5° x 0.625°",
    "senorge":           "SeNorge / 1 km",
    "era5_0.25":         "ERA5 / 0.25°",
    "era5_0.5":          "ERA5 / 0.5°",}
    ds_label = label_map.get(dataset, dataset.upper())

    before_str = f"{window_before} yr"
    after_str  = f"{window_after} yr"
    period_str = f"{start_year}–{end_year}"

    pos_mask  = avg_diff >= 0
    neg_mask  = ~pos_mask
    bar_width = bin_width_mm * 0.85

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    fig.suptitle(
        f"Distribution Difference  ({before_str} → {after_str})  ·  "
        f"{ds_label} / {acc_label}  ·  {catchment_title}",
        fontsize=13, fontweight="normal", y=1.02,)

    # Positive bars — blue: after window has more probability mass here
    if pos_mask.any():
        ax.bar(
            bin_centers[pos_mask], avg_diff[pos_mask],
            width=bar_width,
            color="#2C7BB6", alpha=0.82, zorder=3,
            label=f"After-period rel. frequency higher",)

    # Negative bars — red: before window has more probability mass here
    if neg_mask.any():
        ax.bar(
            bin_centers[neg_mask], avg_diff[neg_mask],
            width=bar_width,
            color="#D73027", alpha=0.82, zorder=3,
            label=f"Before-period rel. frequency higher",)

    # Zero reference line
    ax.axhline(0, color="black", linewidth=0.9, zorder=5)

    type_label = "Annual Maximal" if data_type == "annual_max" else "Daily"
    ax.set_xlabel(f"{type_label} {acc_label} Accumulated Precipitation (mm)", fontsize=11)
    ax.set_ylabel("Mean Δ Rel. Frequency\n(after − before)", fontsize=11)
    max_abs = float(np.nanmax(np.abs(avg_diff))) if avg_diff.size > 0 else 0.05
    y_margin = max(max_abs * 1.25, 0.02)   # at least ±0.02 so zero-signal plots aren't flat lines
    ax.set_ylim(-y_margin, y_margin)
    ax.set_xlim(
        bin_centers[0]  - bin_width_mm * 1.5,
        bin_centers[-1] + bin_width_mm * 1.5,)

    # Metadata annotation top-right
    ax.text(
        0.985, 0.97,
        f"n pairs = {n_pairs}  ·  period: {period_str}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8, color="dimgrey",)

    ax.legend(fontsize=10, frameon=False, loc="upper right",
              bbox_to_anchor=(0.985, 0.80))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), format="pdf", bbox_inches="tight")
        print(f"    [fig]   Saved → {out_path}")
    plt.close(fig)


# ── Numbered-catchment map helper ─────────────────────────────────────────────
def draw_catchments_numbered(
    ax,
    catchments: dict,
    data_crs,
    catchment_numbers: dict,
    edge_color: str = "red",
    linewidth: float = 1.2,
    fontsize: int = 9,
    zorder: int = 4,
    position_overrides: dict = None,
) -> None:
    """
    Draw catchment outlines with a circled number at each representative point.

    Parameters
    ----------
    catchment_numbers   : dict  {slug: int}  — e.g. {"nevina_bergheim": 1, ...}
    position_overrides  : dict  {slug: (lon, lat)}  — override label position in
                          geographic coordinates (EPSG:4326) for specific catchments.
                          If None, the representative point is used for all catchments.
    """
    overrides = position_overrides or {}

    for slug, gdf in catchments.items():
        num = catchment_numbers.get(slug, "?")

        ax.add_geometries(
            gdf.geometry,
            crs=data_crs,
            facecolor="none",
            edgecolor=edge_color,
            linewidth=linewidth,
            zorder=zorder,)

        if slug in overrides:
            lon, lat = overrides[slug]
        else:
            try:
                geom = gdf.geometry.union_all()
            except AttributeError:
                geom = gdf.geometry.unary_union
            pt = geom.representative_point()
            lon, lat = pt.x, pt.y

        ax.annotate(
            str(num),
            xy=(lon, lat),
            xycoords=data_crs._as_mpl_transform(ax),
            xytext=(0, -2.5),
            textcoords="offset points",
            ha="center", va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=edge_color,
            bbox=dict(
                boxstyle="circle,pad=0.25",
                facecolor="white",
                edgecolor=edge_color,
                linewidth=0.9,
                alpha=0.88,
            ),
            zorder=zorder + 1,)
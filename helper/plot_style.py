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
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


from return_period import (
    get_annual_maxima,
    get_event_annual_max,
    fit_gev,
    gev_return_level,
    estimate_return_period,
    weibull_plotting_positions,   # add this line
)

#Define general Figure dpi
fig_dpi = 150


# Matplotlib style defaults
plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi":    fig_dpi,})

# ── Map projection and CRS constants shared across all map figures ─────────────
MAP_PROJ         = ccrs.LambertConformal(
    central_longitude=8.75, central_latitude=61.25,
    standard_parallels=(58.0, 64.0))
DATA_CRS_LATLON  = ccrs.PlateCarree()
DATA_CRS_SENORGE = ccrs.UTM(zone=33)
OCEAN_COLOR      = "#a8d8ea"

def _load_ipcc_prec_seq(txt_path: Path) -> mcolors.LinearSegmentedColormap:
    """Load IPCC sequential precipitation colormap from a 256-row RGB text file."""
    rgb = np.loadtxt(str(txt_path), dtype=float) / 255.0
    return mcolors.LinearSegmentedColormap.from_list("prec_seq_ipcc", rgb, N=256)


def _load_ipcc_prec_div(txt_path: Path) -> mcolors.LinearSegmentedColormap:
    """Load IPCC diverging precipitation colormap from a 256-row RGB text file."""
    rgb = np.loadtxt(str(txt_path), dtype=float) / 255.0
    return mcolors.LinearSegmentedColormap.from_list("prec_div_ipcc", rgb, N=256)


_HELPER_DIR = Path(__file__).parent
PRECIP_CMAP = _load_ipcc_prec_seq(_HELPER_DIR / "prec_seq.txt")
PRECIP_DIV_CMAP = _load_ipcc_prec_div(_HELPER_DIR / "prec_div.txt")
WEIGHT_CMAP = plt.get_cmap("viridis")

# MODEL_* moved to config_paths.py to break the circular import with catchment_tools
from config_paths import MODEL_COLORS, MODEL_LABELS, MODEL_ORDER  # noqa: E402

# ── Diverging-norm helper ──────────────────────────────────────────────────────
def _finite_max_abs(values: np.ndarray, fallback: float = 1.0) -> float:
    """
    Return max(|finite values|) of an array, or `fallback` if there are none.

    Guards against ``np.nanmax`` raising
    "zero-size array to reduction operation fmax which has no identity",
    which happens when a difference field is entirely non-finite (e.g. two
    grids do not share coordinates, or the reference field is all-zero so the
    relative-difference denominator is all-NaN). In that degenerate case the
    diverging colorbar simply falls back to a symmetric ±`fallback` range.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return fallback
    m = float(np.max(np.abs(finite)))
    return m if m > 0 else fallback


# ── Diverging-norm helper ──────────────────────────────────────────────────────
def _finite_max_abs(values: np.ndarray, fallback: float = 1.0) -> float:
    """
    Return max(|finite values|) of an array, or `fallback` if there are none.

    Guards against ``np.nanmax`` raising
    "zero-size array to reduction operation fmax which has no identity",
    which happens when a difference field is entirely non-finite (e.g. two
    grids do not share coordinates, or the reference field is all-zero so the
    relative-difference denominator is all-NaN). In that degenerate case the
    diverging colorbar simply falls back to a symmetric ±`fallback` range.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return fallback
    m = float(np.max(np.abs(finite)))
    return m if m > 0 else fallback


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


# ═══════════════════════════════════════════════════════════════════════════
# Storm Hans event precipitation map figures
# ═══════════════════════════════════════════════════════════════════════════

def round_up_nice(value: float) -> float:
    """Round a positive value up to a visually clean colorbar upper bound."""
    if not np.isfinite(value) or value <= 0:
        return 1.0
    for threshold, step in [(10, 1), (25, 2), (50, 5), (100, 10), (200, 20)]:
        if value <= threshold:
            return float(np.ceil(value / step) * step)
    return float(np.ceil(value / 25) * 25)


def colorbar_label(window_days: int) -> str:
    return f"Max {window_days}-Day Accumulated Precipitation during Event (mm)"


def title_text(combo: dict) -> str:
    return (
        f"Storm Hans – Max {combo['window_days']}-Day Precip. Envelope "
        f"(7–9 Aug 2023): {combo['title_dataset']}/{combo['title_res']}"
    )


def make_colorbar_ticks(vmax: float) -> np.ndarray:
    """Return evenly-spaced colorbar ticks from 0 to vmax."""
    n = 5 if vmax <= 20 else (6 if vmax <= 100 else 7)
    return np.round(np.linspace(0, vmax, n), 0)


def compute_vmax_by_window(event_fields: dict) -> dict:
    """Fixed colorbar maxima for Storm Hans 1-day and 2-day maps."""
    return {1: 120.0, 2: 175.0}


def plot_precip_map(
    combo: dict,
    da_evt,
    catchments: dict,
    vmax: float,
    out_paths: list,
    map_extent: tuple = (5.0, 12.5, 57.5, 65.0),
    catchment_numbers: dict = None,
    catchment_legend_text: str = "",
    label_overrides: dict = None,
) -> None:
    """Single-panel Storm Hans event precipitation map saved to out_paths."""
    west, east, south, north = map_extent
    catchment_numbers = catchment_numbers or {}

    fig = plt.figure(figsize=(8.6, 9.0))
    ax  = plt.axes(projection=MAP_PROJ)
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.15, top=0.92)
    ax.set_extent([west, east, south, north], crs=DATA_CRS_LATLON)
    ax.set_rasterization_zorder(4)  # rasterize map background; catchment labels(5) stay vector

    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    if combo["dataset"] == "era5":
        mesh = ax.pcolormesh(
            da_evt["longitude"].values, da_evt["latitude"].values, da_evt.values,
            transform=DATA_CRS_LATLON, cmap=PRECIP_CMAP, norm=norm,
            shading="nearest", zorder=1)
    else:
        mesh = ax.pcolormesh(
            da_evt["X"].values, da_evt["Y"].values, da_evt.values,
            transform=DATA_CRS_SENORGE, cmap=PRECIP_CMAP, norm=norm,
            shading="nearest", zorder=1)

    try:
        ax.coastlines(resolution="10m", color="black", linewidth=0.8, zorder=3)
    except Exception as exc:
        print(f"Warning: coastlines unavailable ({exc}).")

    if catchment_numbers:
        draw_catchments_numbered(ax, catchments, DATA_CRS_LATLON, catchment_numbers,
                                 position_overrides=label_overrides)
    if catchment_legend_text:
        ax.text(0.02, 0.03, catchment_legend_text, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="0.6", alpha=0.88), zorder=6)

    gl = ax.gridlines(crs=DATA_CRS_LATLON, draw_labels=True, linewidth=0.5,
                      color="0.35", alpha=0.7, linestyle="--",
                      x_inline=False, y_inline=False, zorder=2)
    gl.top_labels = gl.right_labels = False
    gl.rotate_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(5.0, 12.5 + 0.001, 2.5))
    gl.ylocator = mticker.FixedLocator(np.arange(57.5, 65.0 + 0.001, 2.5))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = gl.ylabel_style = {"size": 9}
    ax.set_title(title_text(combo), pad=10)

    cax  = fig.add_axes([0.22, 0.07, 0.56, 0.028])
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal",
                        ticks=make_colorbar_ticks(vmax), extend="max")
    cbar.set_label(colorbar_label(combo["window_days"]))
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=fig_dpi)
        print(f"Saved → {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Catchment weight map figures
# ═══════════════════════════════════════════════════════════════════════════

def plot_single_catchment_weight_map(
    combo: dict,
    catchment_slug: str,
    catchment_title: str,
    da_w,
    catchment_gdf: gpd.GeoDataFrame,
    out_paths: list,
) -> None:
    """One weight-fraction map for a single catchment and dataset/resolution."""
    from catchment_tools import crop_weight_field_to_nonzero_bbox, get_plot_extent_and_crs

    da_plot = crop_weight_field_to_nonzero_bbox(
        da_w, pad_cells=combo.get("pad_cells", 2)).load()
    extent, extent_crs = get_plot_extent_and_crs(da_plot)

    fig = plt.figure(figsize=combo.get("figsize", (7.2, 7.8)))
    ax  = plt.axes(projection=MAP_PROJ)
    fig.subplots_adjust(left=0.08, right=0.94, bottom=0.14, top=0.91)
    ax.set_extent(extent, crs=extent_crs)
    ax.set_rasterization_zorder(4)  # rasterize map background; catchment outline(4) stays vector

    da_masked = da_plot.where(da_plot > 0)
    cmap = WEIGHT_CMAP.copy()
    cmap.set_bad(color="white", alpha=0.0)
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    pmesh_kw = dict(cmap=cmap, norm=norm, shading="nearest", zorder=1,
                    rasterized=combo.get("rasterized", False))

    if {"longitude", "latitude"}.issubset(da_plot.dims):
        mesh = ax.pcolormesh(da_plot["longitude"].values, da_plot["latitude"].values,
                             da_masked.values, transform=DATA_CRS_LATLON, **pmesh_kw)
    elif {"lon", "lat"}.issubset(da_plot.dims):
        mesh = ax.pcolormesh(da_plot["lon"].values, da_plot["lat"].values,
                             da_masked.values, transform=DATA_CRS_LATLON, **pmesh_kw)
    elif {"X", "Y"}.issubset(da_plot.dims):
        mesh = ax.pcolormesh(da_plot["X"].values, da_plot["Y"].values,
                             da_masked.values, transform=DATA_CRS_SENORGE, **pmesh_kw)
    else:
        raise ValueError(f"Unsupported weight-grid dimensions: {da_plot.dims}")

    try:
        ax.coastlines(resolution="10m", color="black", linewidth=0.7, zorder=3)
    except Exception as exc:
        print(f"Warning: coastlines unavailable ({exc}).")

    ax.add_geometries(catchment_gdf.geometry, crs=DATA_CRS_LATLON,
                      facecolor="none", edgecolor="red",
                      linewidth=combo.get("border_lw", 0.8), zorder=4)

    gl = ax.gridlines(crs=DATA_CRS_LATLON, draw_labels=True, linewidth=0.45,
                      color="0.35", alpha=0.65, linestyle="--",
                      x_inline=False, y_inline=False, zorder=2)
    gl.top_labels = gl.right_labels = False
    gl.rotate_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = gl.ylabel_style = {"size": 8}
    ax.set_title(f'Catchment Weights: {combo["label"]} — {catchment_title}', pad=10)

    cax  = fig.add_axes([0.22, 0.07, 0.56, 0.028])
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal",
                        ticks=np.linspace(0.0, 1.0, 6))
    cbar.set_label("Weight")
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        print(f"Saved → {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Joint-distribution scatter figures (compound flood-risk analysis)
# ═══════════════════════════════════════════════════════════════════════════

# Cyclic day-of-year colormap — 'hsv' reproduces the Blöschl et al. (2017)
# colour wheel: Jan≈red → Mar≈yellow → May≈green → Jul≈cyan → Oct≈blue → Dec≈magenta.
DOY_CMAP = plt.cm.hsv
DOY_MAX  = 366   # day-of-year normalisation upper bound (leap-safe)

JOINT_VAR_TITLES: dict[str, str] = {
    "precipitation": "Precipitation Sum",
    "soil_moisture": "Soil-Moisture Mean",
    "snowmelt":      "Snowmelt (SWE Difference)",
}
JOINT_VAR_AXIS_LABELS: dict[str, str] = {
    "precipitation": "Precipitation Sum (mm)",
    "soil_moisture": "Soil Moisture Mean (kg/m²)",
    "snowmelt":      "Snowmelt (SWE Difference) (kg/m²)",
}


def add_month_color_wheel(fig, rect: tuple = (0.815, 0.60, 0.13, 0.13)) -> None:
    """Circular Jan–Dec day-of-year colour legend (paper-style wheel).
    January sits at the top; months run clockwise (Jan → Apr → Jul → Oct)."""
    WHEEL_INNER_RADIUS = 0.60   # inner edge of the colour ring (fraction of radius)
    MONTH_LABEL_FS     = 6      # font size of the Jan/Apr/Jul/Oct labels

    ax = fig.add_axes(rect, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    theta_edges = np.linspace(0.0, 2.0 * np.pi, DOY_MAX + 1)
    ring = (0.5 * (theta_edges[:-1] + theta_edges[1:]) / (2.0 * np.pi))
    ax.pcolormesh(theta_edges, np.array([WHEEL_INNER_RADIUS, 1.0]),
                  ring[np.newaxis, :], cmap=DOY_CMAP, vmin=0.0, vmax=1.0)
    month_doy = {"Jan": 1, "Apr": 91, "Jul": 182, "Oct": 274}
    ax.set_xticks([2.0 * np.pi * (d - 1) / DOY_MAX for d in month_doy.values()])
    ax.set_xticklabels(list(month_doy.keys()), fontsize=MONTH_LABEL_FS)
    ax.set_yticks([])
    ax.set_ylim(0.0, 1.0)
    ax.grid(False)
    ax.spines["polar"].set_visible(False)


def make_joint_distribution_figure(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    doy_vals: np.ndarray,
    x_variable: str,
    y_variable: str,
    window_days: int,
    start_year: int,
    end_year: int,
    catchment_title: str,
    n_members: int,
    out_paths: list,
) -> None:
    """
    Joint-distribution scatter of two catchment-averaged window quantities:
    one point per (member, date), coloured by day of year, with a circular
    Jan–Dec legend (Blöschl et al. 2017 style).
    """
    POINT_SIZE  = 3.0    # marker area — small because up to ~10^6 points are drawn
    POINT_ALPHA = 0.35   # transparency so point-cloud density stays visible
    SCATTER_DPI = 300    # raster resolution of the rasterized point layer in the PDF

    TITLE_PAD     = 18     # points of whitespace between title and axes
    AXIS_HEADROOM = 0.02   # fraction of the data range kept free above the maxima

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    fig.subplots_adjust(right=0.76)   # leave room for the month wheel
    ax.scatter(x_vals, y_vals, c=doy_vals, cmap=DOY_CMAP, vmin=1, vmax=DOY_MAX,
               s=POINT_SIZE, alpha=POINT_ALPHA, linewidths=0, rasterized=True)

    # Open x-y axes instead of a full box; the point cloud starts exactly in
    # the bottom-left corner (no margin below the minima), headroom only above.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    x_min, x_max = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
    y_min, y_max = float(np.nanmin(y_vals)), float(np.nanmax(y_vals))
    ax.set_xlim(x_min, x_max + AXIS_HEADROOM * (x_max - x_min))
    ax.set_ylim(y_min, y_max + AXIS_HEADROOM * (y_max - y_min))

    ax.set_xlabel(JOINT_VAR_AXIS_LABELS[x_variable])
    ax.set_ylabel(JOINT_VAR_AXIS_LABELS[y_variable])
    ax.set_title(
        f"Joint Distribution of {window_days}-Day {JOINT_VAR_TITLES[x_variable]} "
        f"and {window_days}-Day {JOINT_VAR_TITLES[y_variable]}, "
        f"{start_year}-{end_year}", fontsize=11, pad=TITLE_PAD)
    ax.text(0.02, 0.98, f"{catchment_title} · CESM2-LE · {n_members} members",
            transform=ax.transAxes, va="top", ha="left", fontsize=8, color="0.3")
    ax.grid(alpha=0.25, linewidth=0.4)
    add_month_color_wheel(fig)

    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=SCATTER_DPI)
        print(f"Saved → {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Annual median / 2-day precipitation map figures
# ═══════════════════════════════════════════════════════════════════════════

def _plot_annmedian_panel(
    ax,
    da,
    dataset_type: str,
    panel_title: str,
    catchments: dict,
    norm,
    catchment_numbers: dict = None,
    annmedian_extent: tuple = (5.0, 14.0, 57.5, 64.0),
    label_overrides: dict = None,
    show_left_labels: bool = True,
    show_bottom_labels: bool = True,
):
    """Draw one panel in an annual median map figure. Returns the mesh for the shared colorbar."""
    west, east, south, north = annmedian_extent
    catchment_numbers = catchment_numbers or {}
    ax.set_extent([west, east, south, north], crs=DATA_CRS_LATLON)
    ax.set_rasterization_zorder(4)  # rasterize background (coastlines/OCEAN/LAND at zorder<4); catchment borders(4) and labels(5) stay vector
    ax.add_feature(cfeature.OCEAN, color=OCEAN_COLOR, zorder=0)
    ax.add_feature(cfeature.LAND,  color="#f0ece6",   zorder=0)

    if dataset_type == "senorge":
        mesh = ax.pcolormesh(da["X"].values, da["Y"].values, da.values,
                             transform=DATA_CRS_SENORGE, cmap=PRECIP_CMAP, norm=norm,
                             shading="nearest", zorder=1, rasterized=True)
    elif dataset_type in ("cesm2", "cesm2_2day", "era5_interp"):
        mesh = ax.pcolormesh(da["lon"].values, da["lat"].values, da.values,
                             transform=DATA_CRS_LATLON, cmap=PRECIP_CMAP, norm=norm,
                             shading="nearest", zorder=1)
    else:
        mesh = ax.pcolormesh(da["longitude"].values, da["latitude"].values, da.values,
                             transform=DATA_CRS_LATLON, cmap=PRECIP_CMAP, norm=norm,
                             shading="nearest", zorder=1)

    ax.add_feature(cfeature.OCEAN, color=OCEAN_COLOR, zorder=2, alpha=1.0)
    ax.coastlines(resolution="10m", color="black", linewidth=0.7, zorder=3)

    if catchment_numbers:
        draw_catchments_numbered(ax, catchments, DATA_CRS_LATLON, catchment_numbers,
                                 linewidth=1.0, fontsize=9.5, zorder=4,
                                 position_overrides=label_overrides)

    gl = ax.gridlines(crs=DATA_CRS_LATLON, draw_labels=True, linewidth=0.4,
                      color="0.4", alpha=0.6, linestyle="--",
                      x_inline=False, y_inline=False, zorder=2)
    gl.top_labels = gl.right_labels = False
    gl.left_labels   = show_left_labels
    gl.bottom_labels = show_bottom_labels
    gl.rotate_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(5.0, 14.5, 2.5))
    gl.ylocator = mticker.FixedLocator(np.arange(58.0, 64.5, 2.0))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = gl.ylabel_style = {"size": 9}
    if panel_title:
        ax.set_title(panel_title, fontsize=11.5, pad=7)
    return mesh


def plot_annual_median_4panel(
    da_cesm2,
    da_senorge,
    da_era5_05,
    da_era5_025,
    catchments: dict,
    start_year: int,
    end_year: int,
    out_paths: list,
    catchment_numbers: dict = None,
    catchment_legend_text: str = "",
    label_overrides: dict = None,
    annmedian_vmax: float = 3500.0,
    annmedian_extent: tuple = (5.0, 14.0, 57.5, 64.0),
) -> None:
    """2×2 panel figure: annual median precipitation from four sources."""
    norm = mcolors.Normalize(vmin=0.0, vmax=annmedian_vmax)
    catchment_numbers = catchment_numbers or {}

    # Slightly shorter/wider figure to reduce empty space created by fixed map aspect.
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 10.4),
                        subplot_kw={"projection": MAP_PROJ})

    # Anchor each GeoAxes toward the centre of the 2×2 layout.
    # This keeps the map aspect but pulls panels closer together.
    axes[0, 0].set_anchor("SE")
    axes[0, 1].set_anchor("SW")
    axes[1, 0].set_anchor("NE")
    axes[1, 1].set_anchor("NW")


    fig.suptitle(f"Annual Median Precipitation ({start_year}–{end_year})",
                fontsize=18, fontweight="normal", x=0.5, y=0.98, ha="center")

    panels = [
        (axes[0, 0], da_cesm2,    "cesm2",
         "CESM2-LE / 0.942° × 1.25°\n100-Member Median"),
        (axes[0, 1], da_senorge,  "senorge", "SeNorge / 1 km × 1 km"),
        (axes[1, 0], da_era5_05,  "era5",    "ERA5 / 0.5° × 0.5°"),
        (axes[1, 1], da_era5_025, "era5",    "ERA5 / 0.25° × 0.25°"),
    ]
    mesh_ref = None
    for ax, da, dtype, title in panels:
        mesh = _plot_annmedian_panel(ax, da, dtype, title, catchments, norm,
                                   catchment_numbers=catchment_numbers,
                                   annmedian_extent=annmedian_extent,
                                   label_overrides=label_overrides)
        if mesh_ref is None:
            mesh_ref = mesh

    cbar_ax = fig.add_axes([0.20, 0.055, 0.42, 0.014])
    cbar = fig.colorbar(mesh_ref, cax=cbar_ax, orientation="horizontal",
                        ticks=np.arange(0, annmedian_vmax + 1, 500), extend="max")
    cbar.set_label("Annual Median Precipitation (mm/year)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    if catchment_legend_text:
        fig.text(0.65, 0.016, catchment_legend_text, ha="left", va="bottom",
                 fontsize=7.5, transform=fig.transFigure,
                 bbox=dict(boxstyle="round,pad=0.40", facecolor="white",
                           edgecolor="0.55", alpha=0.92))

    fig.subplots_adjust(left=0.07, right=0.92, bottom=0.15, top=0.90,
                        wspace=0.12, hspace=0.16)

    
    # bbox_inches="tight" trims the left and right empty spaces in the output pdf plot
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), format="pdf", dpi=fig_dpi, bbox_inches="tight", pad_inches=0.15)
        print(f"Saved → {out_path}")
    plt.close(fig)


def plot_window_median_2panel(
    da_cesm2,
    da_era5,
    catchments: dict,
    start_year: int,
    end_year: int,
    out_paths: list,
    catchment_numbers: dict = None,
    catchment_legend_text: str = "",
    label_overrides: dict = None,
    window_vmax: float = 14.0,
    window_days: int = 2,
    annmedian_extent: tuple = (5.0, 14.0, 57.5, 64.0),
) -> None:
    """2-panel figure: annual median of N-day rolling precipitation, CESM2-LE and ERA5 0.5°."""
    norm = mcolors.Normalize(vmin=0.0, vmax=window_vmax)

    catchment_numbers = catchment_numbers or {}

    fig, axes = plt.subplots(1, 2, figsize=(13, 8.5),
                             subplot_kw={"projection": MAP_PROJ})
    fig.suptitle(f"{window_days}-Day Precipitation Median ({start_year}–{end_year})",
                 fontsize=18, fontweight="normal", x=0.5, y=1.01, ha="center")

    panels = [
        (axes[0], da_cesm2, "cesm2_2day",
         "CESM2-LE / 0.942° × 1.25°\n100-Member Median"),
        (axes[1], da_era5,  "era5_2day", "ERA5 / 0.5° × 0.5°"),
    ]
    mesh_ref = None
    for ax, da, dtype, title in panels:
        mesh = _plot_annmedian_panel(ax, da, dtype, title, catchments, norm,
                                     catchment_numbers=catchment_numbers,
                                     annmedian_extent=annmedian_extent,
                                     label_overrides=label_overrides)
        if mesh_ref is None:
            mesh_ref = mesh

    cbar_ax = fig.add_axes([0.19, -0.04, 0.50, 0.025])
    cbar = fig.colorbar(mesh_ref, cax=cbar_ax, orientation="horizontal",
                        ticks=np.linspace(0, window_vmax, 8), extend="max")
    cbar.set_label(f"{window_days}-Day Precipitation Median (mm)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    if catchment_legend_text:
        fig.text(0.71, -0.0275, catchment_legend_text, ha="left", va="center",
                    fontsize=9, transform=fig.transFigure,
                    bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                            edgecolor="0.55", alpha=0.92))

    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.08, top=0.90, wspace=0.10)
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), format="pdf", bbox_inches="tight", dpi=fig_dpi)
        print(f"Saved → {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 2-day precipitation difference figures (CESM2-LE vs ERA5 interpolated)
# ═══════════════════════════════════════════════════════════════════════════

def _add_pixel_hatch_overlay(
    ax,
    lons: np.ndarray,
    lats: np.ndarray,
    mask: np.ndarray,
    hatch: str,
    transform,
    hatch_color: str = "black",
    linewidth: float = 0.4,
    zorder: int = 5,
) -> None:
    """
    Draw a hatch pattern over every True cell in mask as individual rectangles.

    Uses PatchCollection so each grid cell is fully covered — avoids the
    half-pixel artifacts that contourf produces near significance boundaries.
    Rectangle borders are suppressed (PatchCollection linewidth=0); only the
    hatch lines are rendered, using `linewidth` via hatch.linewidth rc context.

    Parameters
    ----------
    lons, lats  : 1-D coordinate arrays (cell centres)
    mask        : 2-D bool array [len(lats), len(lons)]
    hatch       : matplotlib hatch string, e.g. "//" or "\\\\"
    transform   : Cartopy CRS of the coordinate arrays
    linewidth   : hatch line width (not the rectangle border)
    """
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection

    if not np.any(mask):
        return

    dlon = float(abs(lons[1] - lons[0])) if len(lons) > 1 else 1.25
    dlat = float(abs(lats[1] - lats[0])) if len(lats) > 1 else 0.942

    patches = [
        Rectangle(
            (float(lons[i]) - dlon / 2.0, float(lats[j]) - dlat / 2.0),
            dlon, dlat,
        )
        for j in range(mask.shape[0])
        for i in range(mask.shape[1])
        if mask[j, i]
    ]

    if not patches:
        return

    with matplotlib.rc_context({'hatch.linewidth': linewidth}):
        pc = PatchCollection(
            patches,
            facecolor="none",
            edgecolor=hatch_color,
            hatch=hatch,
            linewidth=0,        # suppress rectangle borders; hatch lines use rc_context above
            transform=transform,
            zorder=zorder,
        )
        ax.add_collection(pc)


def _plot_diff_panel(
    ax,
    da,
    panel_title: str,
    catchments: dict,
    norm,
    catchment_numbers: dict = None,
    annmedian_extent: tuple = (5.0, 14.0, 57.5, 64.0),
    label_overrides: dict = None,
    show_left_labels: bool = True,
    show_bottom_labels: bool = True,
):
    """Draw one difference panel using the diverging colormap. Returns mesh."""
    west, east, south, north = annmedian_extent
    catchment_numbers = catchment_numbers or {}
    ax.set_extent([west, east, south, north], crs=DATA_CRS_LATLON)
    ax.set_rasterization_zorder(4)  # rasterize background; catchment borders(4) and labels(5) stay vector
    ax.add_feature(cfeature.OCEAN, color=OCEAN_COLOR, zorder=0)
    ax.add_feature(cfeature.LAND,  color="#f0ece6",   zorder=0)

    mesh = ax.pcolormesh(da["lon"].values, da["lat"].values, da.values,
                         transform=DATA_CRS_LATLON, cmap=PRECIP_DIV_CMAP, norm=norm,
                         shading="nearest", zorder=1)

    ax.add_feature(cfeature.OCEAN, color=OCEAN_COLOR, zorder=2, alpha=1.0)
    ax.coastlines(resolution="10m", color="black", linewidth=0.7, zorder=3)

    if catchment_numbers:
        draw_catchments_numbered(ax, catchments, DATA_CRS_LATLON, catchment_numbers,
                                 linewidth=1.0, fontsize=9.5, zorder=4,
                                 position_overrides=label_overrides)

    gl = ax.gridlines(crs=DATA_CRS_LATLON, draw_labels=True, linewidth=0.4,
                      color="0.4", alpha=0.6, linestyle="--",
                      x_inline=False, y_inline=False, zorder=2)
    gl.top_labels = gl.right_labels = False
    gl.left_labels   = show_left_labels
    gl.bottom_labels = show_bottom_labels
    gl.rotate_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(5.0, 14.5, 2.5))
    gl.ylocator = mticker.FixedLocator(np.arange(58.0, 64.5, 2.0))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = gl.ylabel_style = {"size": 9}
    if panel_title:
        ax.set_title(panel_title, fontsize=11.5, pad=7)
    return mesh


def plot_window_interp_3panel(
    da_cesm2,
    da_era5_interp,
    da_diff,
    catchments: dict,
    start_year: int,
    end_year: int,
    out_paths: list,
    fig_title: str,
    seq_cbar_label: str,
    div_cbar_label: str,
    catchment_numbers: dict = None,
    catchment_legend_text: str = "",
    label_overrides: dict = None,
    window_vmax: float = 14.0,
    annmedian_extent: tuple = (5.0, 14.0, 57.5, 64.0),
    sig_cesm_higher: "np.ndarray | None" = None,
    sig_era5_higher: "np.ndarray | None" = None,
    sig_legend_text: str = "",
) -> None:
    """
    3-panel figure: CESM2-LE median | ERA5 Interpolated median | difference.

    Sequential colorbar centred under panels 0+1.
    Diverging colorbar centred under panel 2.
    Optional significance hatching overlaid on the diff panel (panel 2, axes[2]):
        sig_cesm_higher : 2-D bool array (lat × lon) — // hatch where CESM2-LE > ERA5
        sig_era5_higher : 2-D bool array (lat × lon) — \\\\ hatch where ERA5 > CESM2-LE
        sig_legend_text : appended to catchment_legend_text box
    Hatches use zorder=3 so catchment borders (4) and labels (5) render above them.
    """
    norm_seq = mcolors.Normalize(vmin=0.0, vmax=window_vmax)
    diff_abs = _finite_max_abs(da_diff.values)

    norm_div = mcolors.TwoSlopeNorm(vmin=-diff_abs, vcenter=0.0, vmax=diff_abs)
    catchment_numbers = catchment_numbers or {}

    fig, axes = plt.subplots(1, 3, figsize=(19, 8),
                             subplot_kw={"projection": MAP_PROJ})
    fig.suptitle(fig_title, fontsize=16, fontweight="normal",
                 x=0.5, y=1.02, ha="center")

    panels = [
        (axes[0], da_cesm2,       "cesm2_2day",
         "CESM2-LE / 0.942° × 1.25°\n100-Member Median"),
        (axes[1], da_era5_interp, "era5_interp", "ERA5 Interpolated"),
    ]
    mesh_seq = None
    for ax, da, dtype, title in panels:
        mesh = _plot_annmedian_panel(ax, da, dtype, title, catchments, norm_seq,
                                     catchment_numbers=catchment_numbers,
                                     annmedian_extent=annmedian_extent,
                                     label_overrides=label_overrides,
                                     show_left_labels=True,
                                     show_bottom_labels=True)
        if mesh_seq is None:
            mesh_seq = mesh

    mesh_div = _plot_diff_panel(axes[2], da_diff,
                                "Difference CESM2-LE and ERA5 Interpolated",
                                catchments, norm_div,
                                catchment_numbers=catchment_numbers,
                                annmedian_extent=annmedian_extent,
                                label_overrides=label_overrides,
                                show_left_labels=True,
                                show_bottom_labels=True)

    # Significance hatching on diff panel (axes[2]) — zorder=3 keeps hatches
    # below catchment borders (4) and labels (5)
    if sig_cesm_higher is not None or sig_era5_higher is not None:
        lons = da_era5_interp["lon"].values
        lats = da_era5_interp["lat"].values
        if sig_cesm_higher is not None:
            _add_pixel_hatch_overlay(axes[2], lons, lats, sig_cesm_higher,
                                     hatch="//", transform=DATA_CRS_LATLON, zorder=3)
        if sig_era5_higher is not None:
            _add_pixel_hatch_overlay(axes[2], lons, lats, sig_era5_higher,
                                     hatch="\\\\", transform=DATA_CRS_LATLON, zorder=3)

    # Sequential colorbar centred under panels 0+1
    # For regular / 90-pctl diff plots: shift first colorbar + catchment legend
    # slightly right by +0.01.
    # For 2/98 and 5/95 pctl plots: keep original positions.
    layout_key = (fig_title + " " + " ".join(str(p) for p in out_paths)).lower()
    layout_key_compact = (
        layout_key
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )

    keep_original_seq_layout = any(
        token in layout_key_compact
        for token in (
            "2/98", "5/95",
            "2pctl", "98pctl", "5pctl", "95pctl",
            "2pct", "98pct", "5pct", "95pct",
            "2ndpercentile", "98thpercentile",
            "5thpercentile", "95thpercentile",
        )
    )

    seq_layout_dx = 0.0 if keep_original_seq_layout else 0.01

    cbar_seq_ax = fig.add_axes([0.10 + seq_layout_dx, -0.04, 0.39, 0.025])
    cbar_seq = fig.colorbar(mesh_seq, cax=cbar_seq_ax, orientation="horizontal",
                            ticks=np.linspace(0, window_vmax, 8), extend="max")
    cbar_seq.set_label(seq_cbar_label, fontsize=10)
    cbar_seq.ax.tick_params(labelsize=9)
    cbar_seq.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # Diverging colorbar under panel 2, shorter
    cbar_div_ax = fig.add_axes([0.73, -0.04, 0.23, 0.025])
    div_ticks = np.linspace(-diff_abs, diff_abs, 9)
    cbar_div = fig.colorbar(mesh_div, cax=cbar_div_ax, orientation="horizontal",
                            ticks=div_ticks)
    cbar_div.set_label(div_cbar_label, fontsize=9)
    cbar_div.ax.tick_params(labelsize=8)
    cbar_div.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # Combined legend (catchments + optional significance text)
    legend_body = catchment_legend_text
    if sig_legend_text:
        legend_body = legend_body + "\n\n" + sig_legend_text if legend_body else sig_legend_text
    if legend_body:
        fig.text(0.50 + seq_layout_dx, -0.04, legend_body, ha="left", va="center",
                 fontsize=8.0, transform=fig.transFigure,
                 bbox=dict(boxstyle="round,pad=0.40", facecolor="white",
                           edgecolor="0.55", alpha=0.92))

    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.10, top=0.93, wspace=0.08)
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), format="pdf", bbox_inches="tight", dpi=fig_dpi)
        print(f"Saved → {out_path}")
    plt.close(fig)


def plot_window_interp_diffonly(
    da_diff,
    catchments: dict,
    start_year: int,
    end_year: int,
    out_paths: list,
    fig_title: str,
    div_cbar_label: str,
    catchment_numbers: dict = None,
    catchment_legend_text: str = "",
    label_overrides: dict = None,
    annmedian_extent: tuple = (5.0, 14.0, 57.5, 64.0),
) -> None:
    """Single-panel difference figure (CESM2-LE – ERA5 Interpolated)."""
    diff_abs = _finite_max_abs(da_diff.values)

    norm_div = mcolors.TwoSlopeNorm(vmin=-diff_abs, vcenter=0.0, vmax=diff_abs)
    catchment_numbers = catchment_numbers or {}

    fig = plt.figure(figsize=(9, 8))
    ax  = plt.axes(projection=MAP_PROJ)
    fig.suptitle(fig_title, fontsize=16, fontweight="normal", x=0.5, y=1.01, ha="center")

    mesh = _plot_diff_panel(ax, da_diff,
                            "Difference CESM2-LE and ERA5 Interpolated",
                            catchments, norm_div,
                            catchment_numbers=catchment_numbers,
                            annmedian_extent=annmedian_extent,
                            label_overrides=label_overrides)

    diff_abs_ticks = np.linspace(-diff_abs, diff_abs, 9)

    # Colorbar: [left, bottom, width, height]
    # left kleiner  -> weiter nach links
    # width kleiner -> kürzer
    # height kleiner -> dünner
    cbar_ax = fig.add_axes([0.32, 0.05, 0.35, 0.015])
    cbar = fig.colorbar(mesh, cax=cbar_ax, orientation="horizontal",
                        ticks=diff_abs_ticks)
    cbar.set_label(div_cbar_label, fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    if catchment_legend_text:
        ax.text(0.77, 0.05, catchment_legend_text,
                ha="left", va="center",
                fontsize=6.8,
                linespacing=0.95,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.28",
                          facecolor="white",
                          edgecolor="0.55",
                          alpha=0.92),
                zorder=10)

    fig.subplots_adjust(left=0.04, right=0.97, bottom=0.10, top=0.92)
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), format="pdf", bbox_inches="tight", dpi=fig_dpi)
        print(f"Saved → {out_path}")
    plt.close(fig)


def plot_window_interp_diffonly_sig(
    da_diff,
    catchments: dict,
    start_year: int,
    end_year: int,
    out_paths: list,
    fig_title: str,
    div_cbar_label: str,
    catchment_numbers: dict = None,
    catchment_legend_text: str = "",
    label_overrides: dict = None,
    annmedian_extent: tuple = (5.0, 14.0, 57.5, 64.0),
    sig_cesm_higher: "np.ndarray | None" = None,
    sig_era5_higher: "np.ndarray | None" = None,
    sig_legend_text: str = "",
) -> None:
    """
    Single-panel difference figure with significance hatching.

    Like plot_2day_interp_diffonly but overlays per-pixel hatching
    (sig_cesm_higher → //, sig_era5_higher → \\\\) and merges sig_legend_text
    into the catchment legend box. Legend is shifted left and up compared to
    the plain diffonly to accommodate the extra sig entries.
    Hatches use zorder=3 (below catchment borders at 4, labels at 5).
    """
    diff_abs = _finite_max_abs(da_diff.values)

    norm_div = mcolors.TwoSlopeNorm(vmin=-diff_abs, vcenter=0.0, vmax=diff_abs)
    catchment_numbers = catchment_numbers or {}

    fig = plt.figure(figsize=(9, 8))
    ax  = plt.axes(projection=MAP_PROJ)
    fig.suptitle(fig_title, fontsize=16, fontweight="normal", x=0.5, y=1.01, ha="center")

    mesh = _plot_diff_panel(ax, da_diff,
                            "Difference CESM2-LE and ERA5 Interpolated",
                            catchments, norm_div,
                            catchment_numbers=catchment_numbers,
                            annmedian_extent=annmedian_extent,
                            label_overrides=label_overrides)

    # Significance hatching — zorder=3 keeps it below catchment borders (4) / labels (5)
    lons = da_diff["lon"].values
    lats = da_diff["lat"].values
    if sig_cesm_higher is not None:
        _add_pixel_hatch_overlay(ax, lons, lats, sig_cesm_higher,
                                 hatch="//", transform=DATA_CRS_LATLON, zorder=3)
    if sig_era5_higher is not None:
        _add_pixel_hatch_overlay(ax, lons, lats, sig_era5_higher,
                                 hatch="\\\\", transform=DATA_CRS_LATLON, zorder=3)

    diff_abs_ticks = np.linspace(-diff_abs, diff_abs, 9)
    cbar_ax = fig.add_axes([0.32, 0.05, 0.35, 0.015])
    cbar = fig.colorbar(mesh, cax=cbar_ax, orientation="horizontal",
                        ticks=diff_abs_ticks)
    cbar.set_label(div_cbar_label, fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # Combined catchment + significance legend — shifted left and up relative to
    # the plain diffonly (0.77, 0.05) to accommodate the wider/taller content
    combined_legend = catchment_legend_text
    if sig_legend_text:
        combined_legend = (combined_legend + "\n\n" + sig_legend_text
                           if combined_legend else sig_legend_text)
    if combined_legend:
        ax.text(0.685, 0.08, combined_legend,
                ha="left", va="center",
                fontsize=6.8,
                linespacing=0.95,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.28",
                          facecolor="white",
                          edgecolor="0.55",
                          alpha=0.92),
                zorder=10)

    fig.subplots_adjust(left=0.04, right=0.97, bottom=0.10, top=0.92)
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), format="pdf", bbox_inches="tight", dpi=fig_dpi)
        print(f"Saved → {out_path}")
    plt.close(fig)


def plot_window_interp_seasonal_4row_3col(
    seasonal_data: list,
    catchments: dict,
    start_year: int,
    end_year: int,
    out_paths: list,
    fig_title: str,
    seq_cbar_label: str,
    div_cbar_label: str,
    catchment_numbers: dict = None,
    catchment_legend_text: str = "",
    label_overrides: dict = None,
    window_vmax: float = 14.0,
    annmedian_extent: tuple = (5.0, 14.0, 57.5, 64.0),
    sig_legend_text: str = "",
) -> None:
    """
    4-row × 3-column seasonal significance plot.

    seasonal_data : list of 4 dicts, one per season ordered [DJF, MAM, JJA, SON].
    Each dict must contain:
      "season_label"   : str  — e.g. "Winter (DJF)"
      "da_cesm2"       : xr.DataArray [lat, lon]  — seasonal global median/p90 CESM2-LE
      "da_era5_interp" : xr.DataArray [lat, lon]  — ERA5-interp seasonal median/p90
      "da_diff"        : xr.DataArray [lat, lon]  — percentage difference
      "sig_cesm_higher": np.ndarray or None        — bool [lat, lon], // hatch
      "sig_era5_higher": np.ndarray or None        — bool [lat, lon], \\\\ hatch

    Layout rules:
    - Column labels appear ONLY above row 0, bold large font — they count for all rows.
    - Season labels are drawn vertically on the far left, one per row, bold large font.
    - Colorbars and combined legend appear ONLY below row 3.
    - Significance hatching on the diff panel (col 2) in every row.
    - Lat/lon tick labels: left labels on col 0 only, bottom labels on row 3 only.
    """
    # ── Shared norms ──────────────────────────────────────────────────────
    PANEL_LABEL_FS  = 17     # shared size for season labels and column labels
    PANEL_LABEL_GAP = 0.035  # x-fraction distance of season labels left of column 0
    # Figure is 20x28 (not square) and we adjust top-label with *0.7 to visually match the season labels
    COL_LABEL_GAP   = PANEL_LABEL_GAP * (20.0 / 28.0) * 0.7
    SEASON_LABEL_FS = PANEL_LABEL_FS
    COL_LABEL_FS    = PANEL_LABEL_FS


    catchment_numbers = catchment_numbers or {}
    norm_seq = mcolors.Normalize(vmin=0.0, vmax=window_vmax)

    # Diverging norm: shared max-abs across all four seasons
    all_diff_vals = np.concatenate(
        [d["da_diff"].values[np.isfinite(d["da_diff"].values)].ravel()
         for d in seasonal_data]
    )
    diff_abs = float(np.nanmax(np.abs(all_diff_vals))) if len(all_diff_vals) > 0 else 1.0
    if diff_abs == 0:
        diff_abs = 1.0
    norm_div = mcolors.TwoSlopeNorm(vmin=-diff_abs, vcenter=0.0, vmax=diff_abs)

    # Slightly narrower figure + negative wspace reduces horizontal gaps
    # while Cartopy keeps the map aspect.
    fig = plt.figure(figsize=(20, 28))
    fig.suptitle(fig_title, fontsize=22, fontweight="normal",
                 x=0.57, y=0.97, ha="center")

    gs = gridspec.GridSpec(
        4, 3,
        figure=fig,
        left=0.105, right=0.985,
        top=0.925, bottom=0.065,
        hspace=0.10, wspace=0.02,)

    all_axes = [
        [fig.add_subplot(gs[r, c], projection=MAP_PROJ) for c in range(3)]
        for r in range(4)]
    
    # Pull columns toward each other inside their GridSpec cells.
    # This reduces visible gaps caused by fixed-aspect map axes.
    for row_axes in all_axes:
        row_axes[0].set_anchor("E")
        row_axes[1].set_anchor("C")
        row_axes[2].set_anchor("W")

    # ── Column labels are added later with fig.text so their distance
    # from the panels matches the season labels on the left.
    col_titles = [
        "CESM2-LE / 0.942° × 1.25°\n100-Member Median",
        "ERA5 Interpolated",
        "Difference CESM2-LE – ERA5 Interp.",]

    # ── Draw all 12 panels ────────────────────────────────────────────────
    mesh_seq_ref = None
    mesh_div_ref = None

    for r, sdict in enumerate(seasonal_data):
        m = _plot_annmedian_panel(
            all_axes[r][0], sdict["da_cesm2"], "cesm2_2day",
            "", catchments, norm_seq,
            catchment_numbers=catchment_numbers,
            annmedian_extent=annmedian_extent,
            label_overrides=label_overrides,
            show_left_labels=True,
            show_bottom_labels=True,
        )
        if mesh_seq_ref is None:
            mesh_seq_ref = m

        _plot_annmedian_panel(
            all_axes[r][1], sdict["da_era5_interp"], "era5_interp",
            "", catchments, norm_seq,
            catchment_numbers=catchment_numbers,
            annmedian_extent=annmedian_extent,
            label_overrides=label_overrides,
            show_left_labels=True,
            show_bottom_labels=True,
        )

        md = _plot_diff_panel(
            all_axes[r][2], sdict["da_diff"],
            "", catchments, norm_div,
            catchment_numbers=catchment_numbers,
            annmedian_extent=annmedian_extent,
            label_overrides=label_overrides,
            show_left_labels=True,
            show_bottom_labels=True,
        )

        if mesh_div_ref is None:
            mesh_div_ref = md

        # Significance hatching on diff panel
        sig_cesm = sdict.get("sig_cesm_higher")
        sig_era5 = sdict.get("sig_era5_higher")
        if sig_cesm is not None or sig_era5 is not None:
            lons = sdict["da_era5_interp"]["lon"].values
            lats = sdict["da_era5_interp"]["lat"].values
            if sig_cesm is not None:
                _add_pixel_hatch_overlay(all_axes[r][2], lons, lats, sig_cesm,
                                         hatch="//", transform=DATA_CRS_LATLON, zorder=3)
            if sig_era5 is not None:
                _add_pixel_hatch_overlay(all_axes[r][2], lons, lats, sig_era5,
                                         hatch="\\\\", transform=DATA_CRS_LATLON, zorder=3)

    # ── Column + season labels
    # Added after canvas.draw() to get stable final GeoAxes positions.
    # The same font size and same panel-distance are used for both.
    fig.canvas.draw()

    row0_top = max(all_axes[0][c].get_position().y1 for c in range(3))
    col_label_y = row0_top + COL_LABEL_GAP

    for c, title in enumerate(col_titles):
        pos = all_axes[0][c].get_position()
        col_center_x = (pos.x0 + pos.x1) / 2.0
        fig.text(
            col_center_x, col_label_y,
            title,
            ha="center", va="center",
            fontsize=COL_LABEL_FS, fontweight="bold",
            linespacing=0.95,
            transform=fig.transFigure,
        )

    season_label_x = all_axes[0][0].get_position().x0 - PANEL_LABEL_GAP

    for r, sdict in enumerate(seasonal_data):
        pos = all_axes[r][0].get_position()
        row_center_y = (pos.y0 + pos.y1) / 2.0
        fig.text(
            season_label_x, row_center_y,
            sdict["season_label"],
            ha="center", va="center",
            fontsize=SEASON_LABEL_FS, fontweight="bold",
            rotation=90,
            transform=fig.transFigure,
        )


    # ── Colorbars below the bottom row ─────────────────────────────────────
    pos_l = all_axes[3][0].get_position()
    pos_m = all_axes[3][1].get_position()
    pos_r = all_axes[3][2].get_position()

    cbar_y = pos_l.y0 - 0.045    # just below bottom row
    cbar_h = 0.013

    # Sequential colorbar: 50 % of col-0+col-1 total width, inset 4 % from
    # the left edge of col 0.  The saved space lets the catchment legend sit
    # to the right of the bar — both elements stay within the cols 0+1 region
    # and no longer overflow into the diverging-colorbar area of col 2.
    seq_total_w = pos_m.x1 - pos_l.x0
    seq_cbar_w  = seq_total_w * 0.72
    seq_cbar_x  = pos_l.x0 + seq_total_w * 0.04

    cbar_seq_ax = fig.add_axes([seq_cbar_x, cbar_y, seq_cbar_w, cbar_h])
    cbar_seq = fig.colorbar(mesh_seq_ref, cax=cbar_seq_ax, orientation="horizontal",
                            ticks=np.linspace(0, window_vmax, 8), extend="max")
    cbar_seq.set_label(seq_cbar_label, fontsize=10)
    cbar_seq.ax.tick_params(labelsize=9)
    cbar_seq.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # Diverging: spans column 2 of the bottom row (unchanged)
    div_left  = pos_r.x0 + 0.01
    div_width = pos_r.width - 0.02

    cbar_div_ax = fig.add_axes([div_left, cbar_y, div_width, cbar_h])
    div_ticks = np.linspace(-diff_abs, diff_abs, 9)
    cbar_div = fig.colorbar(mesh_div_ref, cax=cbar_div_ax, orientation="horizontal",
                            ticks=div_ticks)
    cbar_div.set_label(div_cbar_label, fontsize=9)
    cbar_div.ax.tick_params(labelsize=8)
    cbar_div.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # ── Combined legend: anchored right of the seq colorbar, within cols 0+1 ─
    legend_body = catchment_legend_text
    if sig_legend_text:
        legend_body = (legend_body + "\n\n" + sig_legend_text
                       if legend_body else sig_legend_text)
    if legend_body:
        leg_x = seq_cbar_x + seq_cbar_w + 0.003
        leg_y = cbar_y + cbar_h / 2.0
        fig.text(
            leg_x, leg_y, legend_body,
            ha="left", va="center",
            fontsize=8.0, transform=fig.transFigure,
            bbox=dict(boxstyle="round,pad=0.40", facecolor="white",
                      edgecolor="0.55", alpha=0.92),
        )

    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), format="pdf", bbox_inches="tight", dpi=fig_dpi)
        print(f"Saved → {out_path}")
    plt.close(fig)

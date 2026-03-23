# plot_style.py
"""
Matplotlib style defaults and the two-panel Storm Hans figure.
All purely visual/plotting code lives here; no statistical logic.
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
    estimate_return_period,)


# Matplotlib style defaults
plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi":    600,})


# Two-panel figure 

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
    annual_desc = fit_sample.sort_values(ascending=False)
    emp_T_all = pd.Series(
        (len(annual_desc) + 1) / np.arange(1, len(annual_desc) + 1),
        index=annual_desc.index,)

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
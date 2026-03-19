# catchment_tools.py
# Weight file discovery, grid alignment, weighted catchment-mean computation, NetCDF caching, and the main loop that orchestrates all catchments.

# Import important libraries
import xarray as xr
import numpy as np
from pathlib import Path

import config_paths as cfg
from data_era5 import find_era5_files, load_era5_precipitation, get_year_range
from plot_style import make_figure


# ── Weight file helpers ────────────────────────────────────────────────────────

# Function to find the respective weight file for a given catchment, dataset, and resolution
def find_weight_file(dataset: str, resolution: str, catchment_slug: str) -> Path:
    """
    Locate the weight file for a given catchment, dataset, and resolution.
    Filename pattern:  weights_catchment_<slug>_<dataset>_<resolution>.nc
    Also tries the ø-variant for the hønnefoss catchment.
    """
    slugs = [catchment_slug]
    if "honnefoss" in catchment_slug:
        slugs.append(catchment_slug.replace("honnefoss", "hønnefoss"))

    res_part = f"_{resolution}" if resolution else ""

    candidates = [
        cfg.CATCHMENT_RAW_DIR / f"weights_catchment_{slug}_{dataset}{res_part}.nc"
        for slug in slugs
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Weight file not found.\n"
        f"Catchment: '{catchment_slug}', dataset: '{dataset}', resolution: '{resolution}'\n"
        f"Tried:\n" + "\n".join(f"  {c}" for c in candidates) +
        f"\n\nActual .nc files in {cfg.CATCHMENT_RAW_DIR}:\n" +
        "\n".join(f"  {f.name}" for f in sorted(cfg.CATCHMENT_RAW_DIR.iterdir())
                  if f.suffix == ".nc")
    )


def load_weights(weight_path: Path) -> xr.DataArray:
    """Load catchment_weight from a weight NetCDF file."""
    ds = xr.open_dataset(str(weight_path))
    return ds["catchment_weight"]


# ── Grid alignment ─────────────────────────────────────────────────────────────

def align_weights_to_precip(precip_da: xr.DataArray,
                             weights_da: xr.DataArray,
                             tol: float = 0.01) -> xr.DataArray:
    """
    Align the weight grid to the precipitation grid.
    - Checks resolution compatibility.
    - If coordinates match within tol, reassigns them exactly (avoids float issues).
    - Otherwise reindexes via nearest-neighbor (handles minor grid offsets).
    Raises ValueError if grids are spatially incompatible.
    """
    w_lat = weights_da["latitude"].values
    w_lon = weights_da["longitude"].values
    p_lat = precip_da["latitude"].values
    p_lon = precip_da["longitude"].values

    # Resolution check
    if len(w_lat) > 1 and len(p_lat) > 1:
        w_dlat = float(np.abs(np.diff(w_lat)).mean())
        p_dlat = float(np.abs(np.diff(p_lat)).mean())
        if abs(w_dlat - p_dlat) > tol:
            raise ValueError(
                f"Weight grid resolution ({w_dlat:.4f}°) does not match "
                f"ERA5 grid resolution ({p_dlat:.4f}°).\n"
                f"Make sure you are using the correct weight file for resolution '{p_dlat:.2f}'."
            )

    # Spatial overlap check
    if not ((w_lat.max() >= p_lat.min()) and (w_lat.min() <= p_lat.max()) and
            (w_lon.max() >= p_lon.min()) and (w_lon.min() <= p_lon.max())):
        raise ValueError(
            "Weight grid and ERA5 grid do not overlap spatially.\n"
            f"  Weight lat [{w_lat.min():.2f}, {w_lat.max():.2f}], "
            f"lon [{w_lon.min():.2f}, {w_lon.max():.2f}]\n"
            f"  ERA5   lat [{p_lat.min():.2f}, {p_lat.max():.2f}], "
            f"lon [{p_lon.min():.2f}, p_lon.max():.2f)]"
        )

    lats_ok = (len(w_lat) == len(p_lat)) and np.allclose(w_lat, p_lat, atol=tol)
    lons_ok = (len(w_lon) == len(p_lon)) and np.allclose(w_lon, p_lon, atol=tol)

    if lats_ok and lons_ok:
        return weights_da.assign_coords(
            latitude=precip_da["latitude"],
            longitude=precip_da["longitude"]
        )
    print("    [align] Reindexing weights → precip grid (nearest neighbor) ...")
    return weights_da.reindex(
        latitude=precip_da["latitude"],
        longitude=precip_da["longitude"],
        method="nearest",
        tolerance=tol * 2,
    )


# ── Weighted mean ──────────────────────────────────────────────────────────────

def compute_catchment_mean(precip_da: xr.DataArray,
                           weights_da: xr.DataArray) -> xr.DataArray:
    """
    Compute weighted catchment-mean daily precipitation.

    Formula:  P_t = Σ_i (w_i × p_i,t) / Σ_i w_i
    Only cells with finite weight > 0 are included.
    NaN precipitation values are excluded from both numerator and denominator.

    Returns
    -------
    xr.DataArray
        1-D, dim (time,), units mm.
    """
    valid_mask = np.isfinite(weights_da.values) & (weights_da.values > 0)
    if valid_mask.sum() == 0:
        raise ValueError(
            "No valid (finite, > 0) weight cells found. "
            "Check that the weight file covers the correct catchment."
        )

    w_vals      = weights_da.values[valid_mask]               # (N_cells,)
    precip_vals = precip_da.values                            # (N_time, N_lat, N_lon)
    p_masked    = precip_vals[:, valid_mask]                  # (N_time, N_cells)

    weighted_sum   = np.nansum(p_masked * w_vals[np.newaxis, :], axis=1)
    eff_weight_sum = np.nansum(
        np.where(np.isfinite(p_masked), w_vals[np.newaxis, :], 0.0), axis=1
    )
    catchment_mean = np.where(eff_weight_sum > 0,
                               weighted_sum / eff_weight_sum, np.nan)

    return xr.DataArray(
        catchment_mean,
        coords={"time": precip_da["time"]},
        dims=["time"],
        attrs={"units": "mm",
               "long_name": "Weighted catchment-mean daily precipitation"},
        name="tp_catchment",
    )


# ── Cache helpers ──────────────────────────────────────────────────────────────

def save_catchment_timeseries(da: xr.DataArray, out_path: Path) -> None:
    """Save a catchment time-series DataArray to NetCDF (postprocessed cache)."""
    da.to_dataset(name="tp_catchment").to_netcdf(str(out_path))
    print(f"    [cache] Saved → {out_path.name}")


def load_catchment_timeseries(nc_path: Path) -> xr.DataArray:
    """Load a cached catchment time-series NetCDF."""
    ds = xr.open_dataset(str(nc_path))
    return ds["tp_catchment"]

# ── 2day-rolling accumulation helper ────────────────────────────────
def rolling_accumulation(da: xr.DataArray, window_days: int = 2) -> xr.DataArray:
    """
    Rolling accumulated precipitation from a daily catchment time series.

    For window_days=2:
        value(t) = precip(t-1) + precip(t)
    """
    out = da.rolling(time=window_days, min_periods=window_days).sum()
    out.name = f"tp_{window_days}day_catchment_acc"
    out.attrs["units"] = "mm"
    out.attrs["long_name"] = f"{window_days}-day accumulated weighted catchment precipitation"
    return out

# ── Main orchestration loop ────────────────────────────────────────────────────

def run_all(dataset: str, resolution: str,
            window_days: int = 2,
            force_recompute: bool = False,
            fig_subdir: str = "timeseries_return_hans") -> None:
    """
    Run the full analysis for all catchments defined in config_paths.CATCHMENTS.

    Parameters
    ----------
    dataset : str
        e.g. "era5" or "senorge"
    resolution : str
        e.g. "0.5x0.5" or "0.25x0.25" (use "" for Senorge which has no resolution suffix)
    window_days : int
        Accumulation window in days (1 = daily, 2 = 2-day rolling sum).
    force_recompute : bool
        False (default) → load cached catchment NetCDF if it exists (fast rerun).
        True            → always recompute from raw ERA5 files (slow, ~minutes).
    fig_subdir : str
        Name of the subfolder inside FIGURES_DIR where PDFs are saved.
    """

    # ── Discover ERA5 files and year range ────────────────────────────────────
    era5_files = find_era5_files(cfg.ERA5_RAW_DIR, resolution)
    start_year, end_year = get_year_range(era5_files, resolution)
    print(f"\n[run_all] Dataset: {dataset} | Resolution: {resolution}")
    print(f"[run_all] ERA5 files found: {len(era5_files)}  ({start_year}–{end_year})")

    # Load ERA5 grid lazily (Dask — no full RAM load yet)
    era5_da = load_era5_precipitation(era5_files)

    # ── Loop over all catchments ───────────────────────────────────────────────
    for slug, title in cfg.CATCHMENTS.items():
        print(f"\n── Catchment: {title} ({slug}) ──")

        nc_path = cfg.catchment_nc_path(dataset, resolution, slug)

        # Step 1: Load or compute weighted catchment time series
        if not force_recompute and nc_path.exists():
            print(f"  [cache] Found cached time series → {nc_path.name}")
            da_catchment = load_catchment_timeseries(nc_path)
        else:
            w_path   = find_weight_file(dataset, resolution, slug)
            weights  = load_weights(w_path)
            w_aligned = align_weights_to_precip(era5_da, weights)

            print(f"  Computing weighted mean (this triggers Dask computation) ...")
            da_catchment = compute_catchment_mean(era5_da, w_aligned)
            da_catchment = da_catchment.compute()   # ← executes the Dask graph here

            save_catchment_timeseries(da_catchment, nc_path)

        da_acc = rolling_accumulation(da_catchment, window_days=window_days)

        fig_path = cfg.figure_path(
            dataset, resolution, slug, start_year, end_year, window_days, fig_subdir
        )
        make_figure(
            da                          = da_acc,
            catchment_title             = title,
            dataset                     = dataset,
            resolution                  = resolution,
            window_days                 = window_days,
            event_year                  = cfg.HANS_SEARCH_YEAR,
            out_path                    = fig_path,
            exclude_event_year_from_fit = True,
        )

    print(f"\n[run_all] ✓ All PDFs saved to:\n  {cfg.FIGURES_DIR / fig_subdir}")
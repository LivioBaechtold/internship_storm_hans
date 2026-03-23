# catchment_tools.py
# Weight file discovery, grid alignment, weighted catchment-mean computation, NetCDF caching, and the main loop that orchestrates all catchments.

# Import important libraries
import xarray as xr
import numpy as np
from pathlib import Path

import config_paths as cfg
from data_era5 import find_era5_files, load_era5_precipitation, get_year_range
from data_senorge import find_senorge_files, load_senorge_precipitation, get_year_range_senorge
from plot_style import make_figure


# ─── Weight file helpers ───

def find_weight_file(dataset: str, resolution: str, catchment_slug: str,
                     weight_dir: Path = None) -> Path:
    """
    Locate the weight file for a given catchment, dataset, and resolution.
    Filename pattern: weights_catchment_<slug>_<dataset>_<resolution>.nc
    Also tries the ø-variant for the hønnefoss catchment.
    weight_dir: override the search directory (default: cfg.CATCHMENT_RAW_DIR).
    """
    search_dir = weight_dir if weight_dir is not None else cfg.CATCHMENT_RAW_DIR

    slugs = [catchment_slug]
    if "honnefoss" in catchment_slug:
        slugs.append(catchment_slug.replace("honnefoss", "hønnefoss"))

    res_part = f"_{resolution}" if resolution else ""

    candidates = [
        search_dir / f"weights_catchment_{slug}_{dataset}{res_part}.nc"
        for slug in slugs]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "No catchment weight file found. Searched:\n  "
        f"{searched}")

def load_weights(weight_path: Path) -> xr.DataArray:
    """Load catchment_weight from a weight NetCDF file."""
    if weight_path is None:
        raise FileNotFoundError("Weight path is None.")

    ds = xr.open_dataset(str(weight_path))
    if "catchment_weight" not in ds:
        raise KeyError(f"'catchment_weight' not found in {weight_path}")
    return ds["catchment_weight"]

def _spatial_dims(da: xr.DataArray) -> tuple[str, str]:
    """
    Detect the two spatial dimension names of a DataArray.
    Handles ERA5 (latitude, longitude) and seNorge (Y, X).
    Returns (dim1, dim2) where dim1 is the y-axis and dim2 is the x-axis.
    """
    for lat_name, lon_name in [("latitude", "longitude"), ("Y", "X"), ("y", "x")]:
        if lat_name in da.dims and lon_name in da.dims:
            return lat_name, lon_name
    raise ValueError(
        f"Cannot identify spatial dimensions in: {list(da.dims)}\n"
        f"Expected one of: (latitude, longitude), (Y, X)")

# ─── Grid alignment ───

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
    lat_dim, lon_dim = _spatial_dims(precip_da)
    w_lat = weights_da[lat_dim].values
    w_lon = weights_da[lon_dim].values
    p_lat = precip_da[lat_dim].values
    p_lon = precip_da[lon_dim].values

    # Resolution check
    if len(w_lat) > 1 and len(p_lat) > 1:
        w_dlat = float(np.abs(np.diff(w_lat)).mean())
        p_dlat = float(np.abs(np.diff(p_lat)).mean())
        if abs(w_dlat - p_dlat) > tol:
            raise ValueError(
                f"Weight grid resolution ({w_dlat:.4f}) does not match "
                f"precipitation grid resolution ({p_dlat:.4f}).\n"
                f"Make sure you are using the correct weight file.")

    # Spatial overlap check
    if not ((w_lat.max() >= p_lat.min()) and (w_lat.min() <= p_lat.max()) and
            (w_lon.max() >= p_lon.min()) and (w_lon.min() <= p_lon.max())):
        raise ValueError(
            f"Weight grid and precipitation grid do not overlap spatially.\n"
            f"  Weight {lat_dim} [{w_lat.min():.2f}, {w_lat.max():.2f}], "
            f"{lon_dim} [{w_lon.min():.2f}, {w_lon.max():.2f}]\n"
            f"  Precip {lat_dim} [{p_lat.min():.2f}, {p_lat.max():.2f}], "
            f"{lon_dim} [{p_lon.min():.2f}, {p_lon.max():.2f}]")

    lats_ok = (len(w_lat) == len(p_lat)) and np.allclose(w_lat, p_lat, atol=tol)
    lons_ok = (len(w_lon) == len(p_lon)) and np.allclose(w_lon, p_lon, atol=tol)

    if lats_ok and lons_ok:
        return weights_da.assign_coords(
            {lat_dim: precip_da[lat_dim], lon_dim: precip_da[lon_dim]})
    print("    [align] Reindexing weights → precip grid (nearest neighbor) ...")
    return weights_da.reindex(
        {lat_dim: precip_da[lat_dim], lon_dim: precip_da[lon_dim]},
        method="nearest",
        tolerance=tol * 2,)

# ── Weighted mean ───

def compute_catchment_mean(precip_da: xr.DataArray,
                           weights_da: xr.DataArray) -> xr.DataArray:
    """
    Compute weighted catchment-mean precipitation lazily with xarray/Dask.

    Formula:  P_t = Σ_i (w_i × p_i,t) / Σ_i w_i
    Only cells with finite weight > 0 are included.
    NaN precipitation values are excluded from both numerator and denominator.

    Returns
    -------
    xr.DataArray
        1-D, dim (time,), units mm.
    """
    valid_weights = weights_da.where(np.isfinite(weights_da) & (weights_da > 0))

    n_valid = int(valid_weights.notnull().sum().item())
    if n_valid == 0:
        raise ValueError(
            "No valid (finite, > 0) weight cells found. "
            "Check that the weight file covers the correct catchment.")

    lat_dim, lon_dim = _spatial_dims(precip_da)

    # numerator: sum(w * p) over space
    weighted_sum = (precip_da * valid_weights).sum(
        dim=(lat_dim, lon_dim),
        skipna=True,)

    # denominator: only count weights where precip is finite at that time step
    eff_weight_sum = valid_weights.where(precip_da.notnull()).sum(
        dim=(lat_dim, lon_dim),
        skipna=True,)

    catchment_mean = weighted_sum / eff_weight_sum.where(eff_weight_sum > 0)
    catchment_mean = catchment_mean.where(eff_weight_sum > 0)
    catchment_mean.name = "tp_catchment"
    catchment_mean.attrs["units"] = "mm"
    catchment_mean.attrs["long_name"] = "Weighted catchment-mean daily precipitation"

    return catchment_mean


# ── Cache helpers ───

def crop_to_weight_bbox(
    precip_da: xr.DataArray,
    weights_da: xr.DataArray,) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Crop precip and weights to the bounding box of finite, positive weights.
    This avoids processing the full seNorge grid when only a small catchment is needed.
    """
    lat_dim, lon_dim = _spatial_dims(weights_da)

    valid = np.isfinite(weights_da.values) & (weights_da.values > 0)
    if not valid.any():
        raise ValueError(
            "No valid (finite, > 0) weight cells found. "
            "Check that the weight file covers the correct catchment.")

    iy, ix = np.where(valid)
    y0, y1 = int(iy.min()), int(iy.max())
    x0, x1 = int(ix.min()), int(ix.max())

    indexer = {
        lat_dim: slice(y0, y1 + 1),
        lon_dim: slice(x0, x1 + 1),}

    return precip_da.isel(indexer), weights_da.isel(indexer)


def save_postproc_dataset(ds: xr.Dataset, out_path: Path) -> None:
    """Save the cached catchment dataset to NetCDF.
    Must contain tp_catchment; may also contain catchment_weight."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    encoding = {}
    if "tp_catchment" in ds:
        encoding["tp_catchment"] = {
            "zlib": True,
            "complevel": 4,}
    if "catchment_weight" in ds:
        encoding["catchment_weight"] = {
            "zlib": True,
            "complevel": 4,}

    ds.to_netcdf(
        str(out_path),
        mode="w",
        format="NETCDF4",
        encoding=encoding,)
    print(f"    [cache] Saved → {out_path.name}")


def load_postproc_dataset(nc_path: Path) -> xr.Dataset:
    """Load a cached postprocessed grid-level Dataset."""
    return xr.open_dataset(str(nc_path))

# ── 2day-rolling accumulation helper ───

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

# ── Main orchestration loop ───
"""
Run the full analysis for all catchments defined in config_paths.CATCHMENTS.

Parameters
----------
dataset : str
    e.g. "era5" or "senorge"
resolution : str
    e.g. "0.5x0.5" or "0.25x0.25" (use "" for Senorge — no resolution suffix)
window_days : int
    Accumulation window in days (1 = daily, 2 = 2-day rolling sum).
force_recompute : bool
    False → use all cached postprocessed NetCDFs if the complete set exists.
    True  → always recompute the full loop from raw ERA5 (slow, ~minutes).
fig_subdir : str
    Subfolder inside both FIGURES_DIR roots where PDFs are saved.
"""

def run_all(dataset: str, resolution: str,
            window_days: int = 2,
            force_recompute: bool = False,
            fig_subdir: str = "timeseries_return_hans",
            weight_dir: Path = None) -> None:

    # ── Step 1: Discover raw files and infer year range from filenames only
    # No raw data is loaded here — just path inspection.
    if dataset == "senorge":
        raw_files = find_senorge_files(cfg.SENORGE_RAW_DIR)
        start_year, end_year = get_year_range_senorge(raw_files)
    else:
        raw_files = find_era5_files(cfg.ERA5_RAW_DIR, resolution)
        start_year, end_year = get_year_range(raw_files, resolution)
    print(f"\n[run_all] Dataset: {dataset} | Resolution: {resolution or 'n/a'}")
    print(f"[run_all] Files found: {len(raw_files)}  ({start_year}–{end_year})")

    # ── Step 2: Build expected postprocessed .nc paths for every catchment
    expected_nc = {
        slug: cfg.catchment_postproc_path(
            dataset, resolution, window_days, slug, start_year, end_year
        )
        for slug in cfg.CATCHMENTS}

    # ── Step 3: Only load raw data if at least one catchment needs recomputation
    raw_da = None

    # ── Step 4: Loop over all catchments
    for slug, title in cfg.CATCHMENTS.items():
        print(f"\n── Catchment: {title} ({slug}) ──")

        nc_path = expected_nc[slug]
        use_cache = (not force_recompute) and nc_path.exists()

        if use_cache:
            # ── Fast path: reuse this catchment's cached time series
            print(f"  [cache] Found postprocessed file → {nc_path.name}")
            ds_catchment = load_postproc_dataset(nc_path)
            da_catchment = ds_catchment["tp_catchment"]

        else:
            # ── Slow path: compute only this catchment from raw data
            if raw_da is None:
                print("  [raw] Loading raw data because at least one catchment needs recomputation ...")
                if dataset == "senorge":
                    raw_da = load_senorge_precipitation(raw_files)
                else:
                    raw_da = load_era5_precipitation(raw_files)

            w_path = find_weight_file(dataset, resolution, slug, weight_dir=weight_dir)
            weights = load_weights(w_path)
            w_aligned = align_weights_to_precip(raw_da, weights)

            # Crop to the catchment bounding box before any rolling operation
            precip_roi, w_roi = crop_to_weight_bbox(raw_da, w_aligned)

            # Mask outside catchment and apply rolling accumulation lazily
            precip_masked = precip_roi.where(w_roi > 0)
            if window_days > 1:
                precip_for_mean = precip_masked.rolling(
                    time=window_days, min_periods=window_days
                ).sum()
            else:
                precip_for_mean = precip_masked

            print("  Computing weighted mean (lazy over time chunks) ...")
            da_catchment = compute_catchment_mean(precip_for_mean, w_roi).load()

            # Save a compact cache: 1-D catchment time series (+ optional 2-D weights)
            ds_out = xr.Dataset({
                "tp_catchment": da_catchment,
                "catchment_weight": w_roi.astype("float32"),})
            ds_out.attrs.update({
                "dataset": dataset,
                "resolution": resolution,
                "window_days": window_days,
                "catchment_slug": slug,
                "start_year": start_year,
                "end_year": end_year,
                "units": "mm",
                "source": f"{dataset} postprocessed",})
            save_postproc_dataset(ds_out, nc_path)
  
        # ── Figure: save to both roots
        out_paths = cfg.figure_paths(
            dataset, resolution, window_days, slug, start_year, end_year, fig_subdir)
        make_figure(
            da                          = da_catchment,
            catchment_title             = title,
            dataset                     = dataset,
            resolution                  = resolution,
            window_days                 = window_days,
            event_year                  = cfg.HANS_SEARCH_YEAR,
            out_paths                   = out_paths,
            exclude_event_year_from_fit = False,)

    print(f"\n[run_all] ✓ All PDFs saved to:")


    # Derive the two figure directories from one representative path set
    for p in cfg.figure_paths(dataset, resolution, window_days,
                               next(iter(cfg.CATCHMENTS)),
                               start_year, end_year, fig_subdir):
        print(f"  {p.parent}")
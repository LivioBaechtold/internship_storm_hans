# catchment_tools.py
# Weight file discovery, grid alignment, weighted catchment-mean computation, NetCDF caching, and the main loop that orchestrates all catchments.

# Import important libraries
import re
import xarray as xr
import numpy as np
from pathlib import Path
import pandas as pd

import config_paths as cfg
from data_era5 import find_era5_files, load_era5_precipitation, get_year_range
from data_senorge import find_senorge_files, load_senorge_precipitation, get_year_range_senorge
from plot_style import make_figure


# Define catchment processing helpers ───────────────────────────────────────────────
def _infer_year_range_from_cache(
    dataset: str, resolution: str, window_days: int) -> tuple:
    cache_dir = cfg.postproc_dir(dataset)
    tag = cfg.res_tag(dataset, resolution)
    pat = re.compile(
        rf"^post_processed_{re.escape(tag)}_{window_days}day_"
        r"[\w]+_(\d{4})-(\d{4})\.nc$"
    )
    try:
        for f in cache_dir.iterdir():
            m = pat.match(f.name)
            if m:
                return int(m.group(1)), int(m.group(2))
    except (FileNotFoundError, PermissionError):
        pass
    return None, None

def get_cached_year_range(dataset: str, resolution: str, window_days: int) -> tuple:
    """
    Public wrapper — return (start_year, end_year) inferred from postprocessed
    cache files.  Returns (None, None) if no matching cache is found.
    """
    return _infer_year_range_from_cache(dataset, resolution, window_days)


# ─── Weight file helpers ───

def find_weight_file(dataset: str, resolution: str, catchment_slug: str,
                     weight_dir: Path = None) -> Path:
    """
    Locate the weight file for a given catchment, dataset, and resolution
    Filename pattern: weights_catchment_<slug>_<dataset>_<resolution>.nc

    Also tries the ø-variant for the hønnefoss catchment
    weight_dir: override the search directory (default: cfg.WEIGHTS_DIR)
    """
    search_dir = weight_dir if weight_dir is not None else cfg.WEIGHTS_DIR

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
    Detect the two spatial dimension names of a DataArray
    Handles ERA5 (latitude, longitude) and seNorge (Y, X)
    Returns (dim1, dim2) where dim1 is the y-axis and dim2 is the x-axis
    """
    for lat_name, lon_name in [("latitude", "longitude"), ("lat", "lon"), ("Y", "X"), ("y", "x")]:
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
    Align the weight grid to the precipitation grid
    - Checks resolution compatibility
    - If coordinates match, reassigns them exactly
    - Otherwise reindexes via nearest-neighbor (handles minor grid offsets)
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
    Compute weighted catchment-mean precipitation

    Formula:  P_t = Σ_i (w_i × p_i,t) / Σ_i w_i
    Only cells with finite weight > 0 are included
    NaN precipitation values are excluded from both numerator and denominator

    Returns
    -------
    xr.DataArray
        1-D, dim (time,), units mm
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
    Crop precip. and weights to the bounding box of finite, positive weights
    This avoids processing the full seNorge grid when only a small catchment is needed
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

#Define path builders ──────────────────────────────────────
def save_postproc_dataset(ds: xr.Dataset, out_path: Path) -> None:
    """Save the cached catchment dataset to NetCDF
    Must contain tp_catchment; may also contain catchment_weight"""
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
    Rolling accumulated precipitation from a daily catchment time series

    For window_days=2:
        value(t) = precip(t-1) + precip(t)
    """
    out = da.rolling(time=window_days, min_periods=window_days).sum()
    out.name = f"tp_{window_days}day_catchment_acc"
    out.attrs["units"] = "mm"
    out.attrs["long_name"] = f"{window_days}-day accumulated weighted catchment precipitation"
    return out

def _check_series_reasonableness(
    da: xr.DataArray,
    label: str,
    window_days: int,) -> None:
    finite = da.where(np.isfinite(da), drop=True)

    if finite.size == 0:
        raise ValueError(f"{label}: no finite values in catchment series.")

    max_val = float(finite.max().item())
    min_val = float(finite.min().item())

    if max_val <= 0:
        raise ValueError(
            f"{label}: all values are <= 0. "
            "Check units, weights, and caches.")

    if max_val > 500.0:
        print(
            f"  [warning] {label}: max {window_days}-day catchment mean = "
            f"{max_val:.1f} mm (min = {min_val:.1f} mm). "
            "This is unusually large; verify SMILE tp24 units.")

# ── Main orchestration loop ───
"""
Run the full analysis for all catchments defined in config_paths.CATCHMENTS

Parameters
----------
dataset : str
    e.g. "era5" or "senorge"
resolution : str
    e.g. "0.5x0.5" or "0.25x0.25" (use "" for Senorge — no resolution suffix)
window_days : int
    Accumulation window in days (1 = daily, 2 = 2-day rolling sum)
force_recompute : bool
    False → use all cached postprocessed NetCDFs if the complete set exists
    True  → always recompute the full loop from raw ERA5 (slow, ~minutes)
fig_subdir : str
    Subfolder inside both FIGURES_DIR roots where PDFs are saved
"""

def run_all(dataset: str, resolution: str,
            window_days: int = 2,
            force_recompute: bool = False,
            fig_subdir: str = "timeseries_return_hans",
            weight_dir: Path = None) -> None:

    # ── Step 1: Discover raw files and infer year range
    # Falls back to scanning the postprocessed cache when raw data is
    # unavailable (e.g. datapeak filesystem not mounted).
    raw_files: list = []
    raw_available = True
    try:
        if dataset == "senorge":
            raw_files = find_senorge_files(cfg.SENORGE_RAW_DIR)
            start_year, end_year = get_year_range_senorge(raw_files)
        else:
            raw_files = find_era5_files(cfg.ERA5_RAW_DIR, resolution)
            start_year, end_year = get_year_range(raw_files, resolution)
    except FileNotFoundError:
        raw_available = False
        start_year, end_year = _infer_year_range_from_cache(
            dataset, resolution, window_days
        )
        if start_year is None:
            raise FileNotFoundError(
                f"Raw data directory not accessible for {dataset}/"
                f"{resolution or 'n/a'} and no postprocessed cache files found.\n"
                f"Either mount the raw data filesystem or run the full "
                f"analysis at least once first."
            )

    print(f"\n[run_all] Dataset: {dataset} | Resolution: {resolution or 'n/a'}")
    if raw_available:
        print(f"[run_all] Files found: {len(raw_files)}  ({start_year}–{end_year})")
    else:
        print(
            f"[run_all] Raw data not accessible; "
            f"year range inferred from cache: {start_year}–{end_year}"
        )

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
            if not raw_available:
                raise FileNotFoundError(
                    f"Catchment '{slug}' has no postprocessed cache and the "
                    f"raw data directory is not accessible.\n"
                    f"Mount the raw data filesystem and rerun, or set "
                    f"FORCE_RECOMPUTE=False once caches exist."
                )
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

            # Build exactly the quantity we conceptually want:
            # 1) one daily catchment value
            # 2) rolling 1-day / 2-day accumulation on that 1-D series
            precip_masked = precip_roi.where(w_roi > 0)

            print("  Computing daily weighted catchment mean ...")
            da_daily = compute_catchment_mean(precip_masked, w_roi).load()

            if window_days > 1:
                da_catchment = rolling_accumulation(da_daily, window_days).load()
            else:
                da_catchment = da_daily

            _check_series_reasonableness(
                da_catchment,
                label=f"{dataset}/{slug}",
                window_days=window_days,)

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


# Climate Models Definitions
# ── SMILE reference loader ─────────────────────────────────────────────────────

def _load_smile_hans_reference(
    dataset: str,
    window_days: int,
    event_year: int,) -> dict[str, dict[str, float]]:
    """
    Load Storm Hans reference statistics from the configured ERA5 reference
    dataset/resolution for one SMILE model.

    Returns per catchment:
        {"event_value_mm": ...,
        "event_return_period_years": ...}
    """
    from return_period import (
        get_event_annual_max,
        get_annual_maxima,
        fit_gev,
        estimate_return_period,)

    smile_cfg = cfg.SMILE_CONFIG[dataset]
    ref_dataset = smile_cfg["ref_dataset"]
    ref_resolution = smile_cfg["ref_resolution"]

    start_year, end_year = _infer_year_range_from_cache(
        ref_dataset, ref_resolution, window_days)
    if start_year is None:
        raise FileNotFoundError(
            f"Cannot determine {ref_dataset}/{ref_resolution} year range: "
            f"no {window_days}-day postprocessed cache files found in\n"
            f"  {cfg.postproc_dir(ref_dataset)}")

    ref: dict[str, dict[str, float]] = {}

    for slug in cfg.CATCHMENTS:
        nc_path = cfg.catchment_postproc_path(
            ref_dataset, ref_resolution, window_days, slug, start_year, end_year)
        if not nc_path.exists():
            raise FileNotFoundError(
                f"Reference cache missing for '{slug}'. Expected:\n  {nc_path}")

        ds = load_postproc_dataset(nc_path)
        da = ds["tp_catchment"]

        event_val, _ = get_event_annual_max(da, event_year)
        annual_max = get_annual_maxima(da)
        c, loc, scale = fit_gev(annual_max)
        event_T = estimate_return_period(event_val, c, loc, scale)

        ref[slug] = {
            "event_value_mm": float(event_val),
            "event_return_period_years": float(event_T),}
        ds.close()

    return ref


# ── SMILE ensemble orchestration loop ─────────────────────────────────────────
def run_all_smile(
    dataset:         str,
    model_dir:       Path,
    start_year:      int,
    end_year:        int,
    window_days:     int  = 2,
    force_recompute: bool = False,
    fig_subdir:      str  = "timeseries_return_hans",
    weight_dir:      Path = None,
) -> None:
    """
    Run the SMILE catchment analysis for one climate-model ensemble.

    New methodology
    ---------------
    1. For each ensemble member, compute/store the weighted catchment-mean
       1-day or 2-day precipitation time series
    2. For each member, compute annual maxima
    3. For each calendar year, take the maximum across all members
       -> exactly one value per year for the GEV fit
    4. Create two single-panel return-period figures per catchment:
       a) ERA5/0.5° Hans precipitation -> return period in climate-model GEV
       b) ERA5/0.5° Hans return period -> precipitation in climate-model GEV
    """
    from data_smile import (
        find_smile_members,
        find_smile_files_for_member,
        load_smile_precipitation,
    )
    from return_period import get_annual_maxima, combine_member_annual_maxima
    from plot_style import make_smile_return_period_figure

    w_dir = weight_dir if weight_dir is not None else cfg.WEIGHTS_DIR
    smile_cfg = cfg.SMILE_CONFIG[dataset]
    unit_mode = smile_cfg.get("tp24_unit_mode", "auto")
    members = find_smile_members(model_dir, dataset)
    n_members = len(members)

    print(f"\n[run_all_smile] Dataset   : {dataset}")
    print(f"[run_all_smile] Members   : {n_members}  ({members[0]}–{members[-1]})")
    print(f"[run_all_smile] Period    : {start_year}–{end_year}  ({end_year - start_year + 1} yr)")
    print(f"[run_all_smile] Window    : {window_days}-day")

    print("\n  Loading Storm Hans reference values from ERA5/0.5° cache ...")
    event_ref = _load_smile_hans_reference(dataset, window_days, cfg.HANS_SEARCH_YEAR)

    for slug, title in cfg.CATCHMENTS.items():
        print(f"\n── Catchment: {title} ({slug}) ──")

        stats_path = cfg.smile_yearmax_stats_path(
            dataset, window_days, slug, start_year, end_year
        )

        # ── Fast path: yearly-max stats already cached ────────────────────────
        if (not force_recompute) and stats_path.exists():
            print(f"  [cache] Year-max stats found → {stats_path.name}")
            ds_stats = load_postproc_dataset(stats_path)

            annual_max_yearly = ds_stats["annual_max_yearly"].to_series()
            ref_event_value_mm = float(ds_stats.attrs["ref_event_value_mm"])
            ref_event_return_period_years = float(ds_stats.attrs["ref_event_return_period_years"])
            ds_stats.close()

        # ── Slow path: build from member caches or raw files ──────────────────
        else:
            weight_path = find_weight_file(dataset, "", slug, weight_dir=w_dir)
            weights = load_weights(weight_path)
            member_annual_maxima: list[pd.Series] = []

            for member_id in members:
                member_cache = cfg.smile_member_postproc_path(
                    dataset, window_days, member_id, slug, start_year, end_year
                )

                if (not force_recompute) and member_cache.exists():
                    ds_m = load_postproc_dataset(member_cache)
                    da_m = ds_m["tp_catchment"].load()
                    ds_m.close()
                else:
                    print(f"  [raw] Computing member {member_id} ...")
                    # For window_days > 1 load one extra year so that the
                    # rolling window is seeded correctly on 1 Jan of start_year.
                    load_from = start_year - 1 if window_days > 1 else start_year
                    files = find_smile_files_for_member(
                        model_dir, dataset, member_id, load_from, end_year
                    )
                    raw_da = load_smile_precipitation(
                        files,
                        load_from,
                        end_year,
                        unit_mode=unit_mode,)

                    w_aligned = align_weights_to_precip(raw_da, weights)
                    precip_roi, w_roi = crop_to_weight_bbox(raw_da, w_aligned)

                    # Exact conceptual order:
                    # daily catchment mean first, then rolling accumulation
                    precip_masked = precip_roi.where(w_roi > 0)
                    da_daily = compute_catchment_mean(precip_masked, w_roi).load()

                    if window_days > 1:
                        da_m = rolling_accumulation(da_daily, window_days).load()
                        # Clip off the seed year now that rolling is done.
                        da_m = da_m.isel(
                            time=(da_m.time.dt.year >= start_year).values
                        )
                    else:
                        da_m = da_daily

                    _check_series_reasonableness(
                        da_m,
                        label=f"{dataset}/{slug}/member{member_id}",
                        window_days=window_days,)


                    ds_out = xr.Dataset({"tp_catchment": da_m})
                    ds_out.attrs.update({
                        "dataset": dataset,
                        "member": member_id,
                        "window_days": window_days,
                        "catchment_slug": slug,
                        "start_year": start_year,
                        "end_year": end_year,})
                    save_postproc_dataset(ds_out, member_cache)

                am_member = get_annual_maxima(da_m).rename(member_id)
                member_annual_maxima.append(am_member)

            print(f"  Computing yearly maxima across all {n_members} members ...")
            annual_max_yearly = combine_member_annual_maxima(member_annual_maxima)

            ref_event_value_mm = event_ref[slug]["event_value_mm"]
            ref_event_return_period_years = event_ref[slug]["event_return_period_years"]

            ds_stats = xr.Dataset(
                {
                    "annual_max_yearly": xr.DataArray(
                        annual_max_yearly.values,
                        dims=["year"],
                        coords={"year": annual_max_yearly.index.values},
                        attrs={"units": "mm"},
                    ),
                }
            )
            ds_stats.attrs.update({
                "dataset": dataset,
                "n_members": n_members,
                "window_days": window_days,
                "start_year": start_year,
                "end_year": end_year,
                "ref_event_value_mm": ref_event_value_mm,
                "ref_event_return_period_years": ref_event_return_period_years,
            })

            stats_path.parent.mkdir(parents=True, exist_ok=True)
            ds_stats.to_netcdf(
                str(stats_path),
                encoding={"annual_max_yearly": {"zlib": True, "complevel": 4}},
            )
            print(f"    [cache] Saved → {stats_path.name}")

        # ── Figure 1: ERA5 Hans precipitation -> SMILE return period ──────────
        out_paths_precip = cfg.smile_figure_paths(
            dataset,
            window_days,
            slug,
            start_year,
            end_year,
            fig_subdir,
            reference_mode="ref_precip",
        )
        make_smile_return_period_figure(
            annual_max_yearly = annual_max_yearly,
            catchment_title   = title,
            dataset           = dataset,
            window_days       = window_days,
            reference_mode    = "precip_value",
            reference_value   = ref_event_value_mm,
            reference_label   = "ERA5/0.5°",
            out_paths         = out_paths_precip,
        )

        # ── Figure 2: ERA5 Hans return period -> SMILE precipitation ──────────
        out_paths_rp = cfg.smile_figure_paths(
            dataset,
            window_days,
            slug,
            start_year,
            end_year,
            fig_subdir,
            reference_mode="ref_returnperiod",
        )
        make_smile_return_period_figure(
            annual_max_yearly = annual_max_yearly,
            catchment_title   = title,
            dataset           = dataset,
            window_days       = window_days,
            reference_mode    = "return_period",
            reference_value   = ref_event_return_period_years,
            reference_label   = "ERA5/0.5°",
            out_paths         = out_paths_rp,
        )

    print(f"\n[run_all_smile] ✓ All PDFs saved to:")
    for p in cfg.smile_figure_paths(
        dataset,
        window_days,
        next(iter(cfg.CATCHMENTS)),
        start_year,
        end_year,
        fig_subdir,
        reference_mode="ref_precip",
    ):
        print(f"  {p.parent}")
# catchment_tools.py
# Weight file discovery, grid alignment, weighted catchment-mean computation, NetCDF caching, and the main loop that orchestrates all catchments.

# Import important libraries
import re
import xarray as xr
import numpy as np
from pathlib import Path
import pandas as pd

import config_paths as cfg
from return_period import get_annual_maxima


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

def _validate_requested_years(
    requested_start: int | None,
    requested_end: int | None,
    available_start: int,
    available_end: int,
    label: str = "data",
) -> tuple[int, int]:
    start_year = available_start if requested_start is None else int(requested_start)
    end_year   = available_end   if requested_end   is None else int(requested_end)

    if start_year > end_year:
        raise ValueError(
            f"Invalid year selection for {label}: "
            f"start_year ({start_year}) > end_year ({end_year})."
        )

    if start_year < available_start or end_year > available_end:
        raise ValueError(
            f"Requested year selection for {label} ({start_year}–{end_year}) "
            f"lies outside the available range {available_start}–{available_end}."
        )

    return start_year, end_year


def subset_time_series_by_year(
    da: xr.DataArray,
    start_year: int | None = None,
    end_year: int | None = None,
) -> xr.DataArray:
    if start_year is None and end_year is None:
        return da

    years = da["time"].dt.year
    mask = xr.ones_like(years, dtype=bool)

    if start_year is not None:
        mask = mask & (years >= start_year)
    if end_year is not None:
        mask = mask & (years <= end_year)

    out = da.isel(time=mask.values)
    if out.sizes.get("time", 0) == 0:
        raise ValueError(
            f"No time steps remain after selecting {start_year}–{end_year}."
        )
    return out


# ─── Weight file helpers ───

#Define weigth file finding
def find_weight_file(dataset: str, resolution: str, catchment_slug: str,
                     weight_dir: Path = None) -> Path:
    """
    Locate the weight file for a given catchment, dataset, and resolution.

    Filename pattern:
        weights_catchment_<slug>_<dataset>_<resolution>.nc
        weights_catchment_<slug>_<dataset>.nc   (if resolution == "")

    Search behaviour
    ----------------
    - If weight_dir is given: search only there.
    - If weight_dir is None: search first in cfg.WEIGHTS_DIR, then in
      cfg.CATCHMENT_RAW_DIR.

    Also tries the ø-variant for the hønnefoss catchment.
    """
    if weight_dir is not None:
        search_dirs = [weight_dir]
    else:
        search_dirs = [cfg.WEIGHTS_DIR, cfg.CATCHMENT_RAW_DIR]

    slugs = [catchment_slug]
    if "honnefoss" in catchment_slug:
        slugs.append(catchment_slug.replace("honnefoss", "hønnefoss"))

    res_part = f"_{resolution}" if resolution else ""

    searched = []

    for search_dir in search_dirs:
        candidates = [
            search_dir / f"weights_catchment_{slug}_{dataset}{res_part}.nc"
            for slug in slugs
        ]

        for candidate in candidates:
            searched.append(candidate)
            if candidate.exists():
                return candidate

    searched_txt = "\n  ".join(str(p) for p in searched)
    raise FileNotFoundError(
        "No catchment weight file found. Searched:\n  "
        f"{searched_txt}"
    )

#Define weights loading
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

# ── SMILE ensemble aggregation ─────────────────────────────────────────────
def pool_member_annual_maxima(member_annual_maxima: list[pd.Series]) -> pd.Series:
    """
    Pool all (member × year) annual maxima into a single flat Series for GEV fitting.

    This is the correct approach for SMILE ensembles: each member-year is treated
    as an independent realisation of the climate system.

    CESM2-LE (100 members × 40 yr) → 4 000 values
    GFDL-SPEAR (30 members × 40 yr) → 1 200 values

    Compare with the ERA5/seNorge approach (one value per calendar year, 69–84 values).
    Pooling gives a stable, unbiased estimate of the marginal distribution of annual maxima.
    """
    if not member_annual_maxima:
        raise ValueError("member_annual_maxima is empty.")
    pooled = pd.concat(member_annual_maxima, ignore_index=True).dropna()
    pooled.name = "annual_max_pooled"
    return pooled

# Define Annual Max. Builder for Smile Models
def _load_or_build_smile_annual_maxima_for_period(
    dataset: str,
    window_days: int,
    catchment_slug: str,
    start_year: int,
    end_year: int,
    force_recompute: bool = False,
    weight_dir: Path | None = None,
) -> pd.Series:
    # yearmax_stats still keyed to analysis window, but now flat (no subfolder)
    stats_path = cfg.smile_yearmax_stats_path(
        dataset, window_days, catchment_slug, start_year, end_year)

    if (not force_recompute) and stats_path.exists():
        with xr.open_dataset(str(stats_path)) as ds_stats:
            vals = ds_stats["annual_max_pooled"].values
        return pd.Series(vals[np.isfinite(vals)], name="annual_max_pooled")

    from data_smile import (
        find_smile_members,
        find_smile_files_for_member,
        load_smile_precipitation,
        get_year_range_smile,)

    smile_cfg = cfg.SMILE_CONFIG[dataset]
    model_dir = smile_cfg["model_dir"]
    # ── NEW: get the FULL available range for member cache paths ──
    avail_start, avail_end = get_year_range_smile(model_dir, dataset)

    # Validate requested analysis years against available range
    start_year, end_year = _validate_requested_years(
        start_year, end_year, avail_start, avail_end, label=dataset)

    w_dir = weight_dir if weight_dir is not None else cfg.WEIGHTS_DIR
    weight_path = find_weight_file(dataset, "", catchment_slug, weight_dir=w_dir)
    weights = load_weights(weight_path)

    unit_mode = smile_cfg.get("tp24_unit_mode", "auto")
    member_annual_maxima = []
    members = find_smile_members(model_dir, dataset)

    for member_id in members:
        # ── NEW: member cache path uses full available range, not analysis range ──
        member_cache = cfg.smile_member_postproc_path(
            dataset, window_days, member_id, catchment_slug, avail_start, avail_end)

        if (not force_recompute) and member_cache.exists():
            with xr.open_dataset(str(member_cache), use_cftime=True) as ds_m:
                da_m = ds_m["tp_catchment"].load()
        else:
            # Build from raw over the FULL available range
            load_from = avail_start - 1 if window_days > 1 else avail_start

            files = find_smile_files_for_member(
                model_dir, dataset, member_id, load_from, avail_end)
            raw_da = load_smile_precipitation(
                files, load_from, avail_end, unit_mode=unit_mode)

            w_aligned = align_weights_to_precip(raw_da, weights)
            precip_roi, w_roi = crop_to_weight_bbox(raw_da, w_aligned)
            precip_masked = precip_roi.where(w_roi > 0)

            da_daily = compute_catchment_mean(precip_masked, w_roi).load()

            if window_days > 1:
                da_m = rolling_accumulation(da_daily, window_days).load()
            else:
                da_m = da_daily

            ds_out = xr.Dataset({"tp_catchment": da_m})
            ds_out.attrs.update({
                "dataset": dataset, "member": member_id,
                "window_days": window_days, "catchment_slug": catchment_slug,
                "start_year": avail_start, "end_year": avail_end,})
            save_postproc_dataset(ds_out, member_cache)

        # ── NEW: subset to analysis window AFTER loading the full-range cache ──
        da_m_subset = subset_time_series_by_year(da_m, start_year, end_year)
        member_annual_maxima.append(get_annual_maxima(da_m_subset).rename(member_id))

    annual_max_pooled = pool_member_annual_maxima(member_annual_maxima)

    ds_stats = xr.Dataset({"annual_max_pooled": xr.DataArray(
        annual_max_pooled.values, dims=["sample"],
        attrs={"units": "mm", "description": "Pooled (member × year) annual maxima"},
    )})
    ds_stats.attrs.update({
        "dataset": dataset, "window_days": window_days,
        "catchment_slug": catchment_slug,
        "start_year": start_year, "end_year": end_year,
        "n_members": len(members),
    })
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    ds_stats.to_netcdf(
        str(stats_path),
        encoding={"annual_max_pooled": {"zlib": True, "complevel": 4}},
    )
    print(f"    [cache] Saved → {stats_path.name}")

    return annual_max_pooled

#Define Annual Max. Loader for Smile Models (for evaluation notebook)
def load_smile_annual_maxima_for_evaluation(
    window_days: int,
    start_year: int | None = None,
    end_year: int | None = None,
    force_recompute: bool = False,
) -> dict:
    """
    Load annual maxima for all SMILE and reanalysis models, pooled across all 5 catchments.
    Used by the climate model evaluation notebook.
    Returns {model_key: np.ndarray}.
    """
    from data_smile import get_year_range_smile

    result = {}

    # Reanalysis models: load full cached series, then subset in time
    for label, ds, res in [
        ("era5_0.5",  "era5",    "0.5x0.5"),
        ("era5_0.25", "era5",    "0.25x0.25"),
        ("senorge",   "senorge", ""),
    ]:
        available_start, available_end = get_cached_year_range(ds, res, window_days)
        if available_start is None:
            print(f"[skip] No {window_days}-day cache found for {label}")
            continue

        use_start, use_end = _validate_requested_years(
            start_year, end_year, available_start, available_end, label=label
        )

        vals = []
        for slug in cfg.CATCHMENTS:
            nc = cfg.catchment_postproc_path(
                ds, res, window_days, slug, available_start, available_end
            )
            if not nc.exists():
                print(f"  [skip] {nc.name}")
                continue

            with xr.open_dataset(str(nc)) as ds_nc:
                da = subset_time_series_by_year(ds_nc["tp_catchment"], use_start, use_end)
                am = get_annual_maxima(da)
                vals.extend(am.values.tolist())

        if vals:
            result[label] = np.array(vals)

    # SMILE models: load or build pooled annual-max caches for the requested period
    for ds_key, smile_entry in cfg.SMILE_CONFIG.items():
        available_start, available_end = get_year_range_smile(smile_entry["model_dir"], ds_key)

        requested_start = start_year if start_year is not None else smile_entry["default_start"]
        requested_end   = end_year   if end_year   is not None else smile_entry["default_end"]

        use_start, use_end = _validate_requested_years(
            requested_start, requested_end, available_start, available_end, label=ds_key
        )

        vals = []
        for slug in cfg.CATCHMENTS:
            am = _load_or_build_smile_annual_maxima_for_period(
                ds_key,
                window_days,
                slug,
                use_start,
                use_end,
                force_recompute=force_recompute,
            )
            vals.extend(am.values.tolist())

        if vals:
            result[ds_key] = np.array(vals)

    return result

def load_annual_maxima_per_catchment(
    window_days: int,
    start_year: int | None = None,
    end_year: int | None = None,
    force_recompute: bool = False,
) -> dict:
    """
    Load annual maxima separately for every catchment, for all SMILE and
    reanalysis models.

    Returns
    -------
    dict[catchment_slug, dict[model_key, np.ndarray]]
        e.g. result["nevina_bergheim"]["era5_0.5"] = array of annual-max values
    """
    from data_smile import get_year_range_smile

    result = {slug: {} for slug in cfg.CATCHMENTS}

    # ── Reanalysis models ────────────────────────────────────────────────────
    for label, ds, res in [
        ("era5_0.5",  "era5",    "0.5x0.5"),
        ("era5_0.25", "era5",    "0.25x0.25"),
        ("senorge",   "senorge", ""),
    ]:
        available_start, available_end = get_cached_year_range(ds, res, window_days)
        if available_start is None:
            print(f"[skip] No {window_days}-day cache found for {label}")
            continue

        use_start, use_end = _validate_requested_years(
            start_year, end_year, available_start, available_end, label=label
        )

        for slug in cfg.CATCHMENTS:
            nc = cfg.catchment_postproc_path(
                ds, res, window_days, slug, available_start, available_end
            )
            if not nc.exists():
                print(f"  [skip] {nc.name}")
                continue

            with xr.open_dataset(str(nc)) as ds_nc:
                da = subset_time_series_by_year(ds_nc["tp_catchment"], use_start, use_end)
                am = get_annual_maxima(da)
                result[slug][label] = am.values

    # ── SMILE models ─────────────────────────────────────────────────────────
    for ds_key, smile_entry in cfg.SMILE_CONFIG.items():
        available_start, available_end = get_year_range_smile(
            smile_entry["model_dir"], ds_key
        )

        requested_start = start_year if start_year is not None else smile_entry["default_start"]
        requested_end   = end_year   if end_year   is not None else smile_entry["default_end"]

        use_start, use_end = _validate_requested_years(
            requested_start, requested_end, available_start, available_end, label=ds_key
        )

        for slug in cfg.CATCHMENTS:
            am = _load_or_build_smile_annual_maxima_for_period(
                ds_key,
                window_days,
                slug,
                use_start,
                use_end,
                force_recompute=force_recompute,
            )
            result[slug][ds_key] = am.values

    return result

def load_daily_values_per_catchment(
    window_days: int,
    start_year: int | None = None,
    end_year: int | None = None,
) -> dict:
    """
    Load ALL daily (or 2-day rolling) precipitation values per catchment,
    for all SMILE and reanalysis models.

    For SMILE, all member time series are concatenated into a single flat
    array per catchment (members × days), giving the marginal daily
    distribution across the ensemble.

    Returns
    -------
    dict[catchment_slug, dict[model_key, np.ndarray]]
        e.g. result["nevina_bergheim"]["cesm2_le"] = array of all daily values
    """
    from data_smile import get_year_range_smile, find_smile_members

    result = {slug: {} for slug in cfg.CATCHMENTS}

    # ── Reanalysis models ────────────────────────────────────────────────────
    for label, ds, res in [
        ("era5_0.5",  "era5",    "0.5x0.5"),
        ("era5_0.25", "era5",    "0.25x0.25"),
        ("senorge",   "senorge", ""),
    ]:
        available_start, available_end = get_cached_year_range(ds, res, window_days)
        if available_start is None:
            print(f"[skip] No {window_days}-day cache found for {label}")
            continue

        use_start, use_end = _validate_requested_years(
            start_year, end_year, available_start, available_end, label=label
        )

        for slug in cfg.CATCHMENTS:
            nc = cfg.catchment_postproc_path(
                ds, res, window_days, slug, available_start, available_end
            )
            if not nc.exists():
                print(f"  [skip] {nc.name}")
                continue

            with xr.open_dataset(str(nc)) as ds_nc:
                da = subset_time_series_by_year(ds_nc["tp_catchment"], use_start, use_end)
                vals = da.values
                result[slug][label] = vals[np.isfinite(vals)]

    # ── SMILE models ─────────────────────────────────────────────────────────
    for ds_key, smile_entry in cfg.SMILE_CONFIG.items():
        model_dir = smile_entry["model_dir"]
        available_start, available_end = get_year_range_smile(model_dir, ds_key)

        requested_start = start_year if start_year is not None else smile_entry["default_start"]
        requested_end   = end_year   if end_year   is not None else smile_entry["default_end"]

        use_start, use_end = _validate_requested_years(
            requested_start, requested_end, available_start, available_end, label=ds_key
        )

        members = find_smile_members(model_dir, ds_key)

        for slug in cfg.CATCHMENTS:
            all_vals: list[float] = []
            for member_id in members:
                member_cache = cfg.smile_member_postproc_path(
                    ds_key, window_days, member_id, slug, available_start, available_end
                )
                if not member_cache.exists():
                    print(f"  [skip] member cache missing: {member_cache.name}")
                    continue

                with xr.open_dataset(str(member_cache), use_cftime=True) as ds_m:
                    da_m = subset_time_series_by_year(
                        ds_m["tp_catchment"], use_start, use_end)
                    vals = da_m.values
                    all_vals.extend(vals[np.isfinite(vals)].tolist())

            if all_vals:
                result[slug][ds_key] = np.array(all_vals)

    return result


# Define percentile mapping table builder
def build_percentile_mapping_table(
    climate_key: str,
    climate_data: np.ndarray,
    reference_data: dict[str, np.ndarray],
    percentiles: tuple[float, ...] = (2.5, 5, 16, 50, 84, 95, 97.5),
) -> pd.DataFrame:
    c = np.asarray(climate_data, dtype=float)
    c = c[np.isfinite(c)]

    rows = []
    for p in percentiles:
        value_mm = float(np.quantile(c, p / 100.0))
        row = {
            "climate_model": climate_key,
            "target_percentile_%": float(p),
            "climate_value_mm": value_mm,
        }

        for ref_key, ref_values in reference_data.items():
            ref = np.asarray(ref_values, dtype=float)
            ref = ref[np.isfinite(ref)]
            row[f"{ref_key}_percentile_%"] = float(100.0 * np.mean(ref <= value_mm))

        rows.append(row)

    return pd.DataFrame(rows)


# Build a summary table of distribution statistics for each model's annual maxima
def build_distribution_summary_table(
    annual_maxima: dict[str, np.ndarray],
) -> pd.DataFrame:
    from plot_style import MODEL_ORDER, MODEL_LABELS

    rows = []
    for key in MODEL_ORDER:
        if key not in annual_maxima:
            continue

        data = np.asarray(annual_maxima[key], dtype=float)
        data = data[np.isfinite(data)]

        rows.append({
            "model": key,
            "label": MODEL_LABELS.get(key, key),
            "n": int(data.size),
            "mean_mm": float(np.mean(data)),
            "std_mm": float(np.std(data, ddof=1)) if data.size > 1 else np.nan,
            "min_mm": float(np.min(data)),
            "q02_5_mm": float(np.quantile(data, 0.025)),
            "q05_mm": float(np.quantile(data, 0.05)),
            "q16_mm": float(np.quantile(data, 0.16)),
            "q25_mm": float(np.quantile(data, 0.25)),
            "median_mm": float(np.quantile(data, 0.50)),
            "q75_mm": float(np.quantile(data, 0.75)),
            "q84_mm": float(np.quantile(data, 0.84)),
            "q95_mm": float(np.quantile(data, 0.95)),
            "q97_5_mm": float(np.quantile(data, 0.975)),
            "max_mm": float(np.max(data)),
            "iqr_mm": float(np.quantile(data, 0.75) - np.quantile(data, 0.25)),
        })

    return pd.DataFrame(rows)

# ── Main orchestration loop ───
def run_all(dataset: str, resolution: str,
            window_days: int = 2,
            force_recompute: bool = False,
            fig_subdir: str = "timeseries_return_hans",
            weight_dir: Path = None,
            analysis_start_year: int | None = None,
            analysis_end_year: int | None = None) -> None:
    """
    Run the full analysis for all catchments defined in config_paths.CATCHMENTS.

    Parameters
    ----------
    dataset : str
        e.g. "era5" or "senorge"
    resolution : str
        e.g. "0.5x0.5" or "0.25x0.25" (use "" for seNorge — no resolution suffix)
    window_days : int
        Accumulation window in days (1 = daily, 2 = 2-day rolling sum)
    force_recompute : bool
        False → reuse cached postprocessed NetCDFs if the complete set exists.
        True  → always recompute from raw data (slow, ~minutes).
    fig_subdir : str
        Subfolder inside both FIGURES_DIR roots where PDFs are saved.
    weight_dir : Path or None
        Override the weight-file search directory (default: cfg.WEIGHTS_DIR).
    """

    # ── Step 1: Discover raw files and infer year range
    # Falls back to scanning the postprocessed cache when raw data is
    # unavailable (e.g. datapeak filesystem not mounted).
    from data_era5 import find_era5_files, load_era5_precipitation, get_year_range
    from data_senorge import find_senorge_files, load_senorge_precipitation, get_year_range_senorge
    from plot_style import make_figure
    
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

    cache_start_year = start_year
    cache_end_year = end_year

    analysis_start_year, analysis_end_year = _validate_requested_years(
        analysis_start_year,
        analysis_end_year,
        cache_start_year,
        cache_end_year,
        label=f"{dataset}/{resolution or 'n/a'}",)
    
    # ── Step 2: Build expected postprocessed .nc paths for every catchment
    expected_nc = {
        slug: cfg.catchment_postproc_path(
            dataset, resolution, window_days, slug, cache_start_year, cache_end_year)
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
            da_catchment = subset_time_series_by_year(
                ds_catchment["tp_catchment"],
                analysis_start_year,
                analysis_end_year,)

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
                da_full_catchment = rolling_accumulation(da_daily, window_days).load()
            else:
                da_full_catchment = da_daily

            _check_series_reasonableness(
                da_full_catchment,
                label=f"{dataset}/{slug}",
                window_days=window_days,)

            ds_out = xr.Dataset({
                "tp_catchment": da_full_catchment,
                "catchment_weight": w_roi.astype("float32"),})
            ds_out.attrs.update({
                "dataset": dataset,
                "resolution": resolution,
                "window_days": window_days,
                "catchment_slug": slug,
                "start_year": cache_start_year,
                "end_year": cache_end_year,
                "units": "mm",
                "source": f"{dataset} postprocessed",})
            save_postproc_dataset(ds_out, nc_path)

            da_catchment = subset_time_series_by_year(
                da_full_catchment,
                analysis_start_year,
                analysis_end_year,)
  
        # ── Figure: save to both roots
        out_paths = cfg.figure_paths(
            dataset, resolution, window_days, slug, analysis_start_year, analysis_end_year, fig_subdir)
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
                               analysis_start_year, analysis_end_year, fig_subdir):
        print(f"  {p.parent}")


# Climate Models Definitions
# ── SMILE reference loader ─────────────────────────────────────────────────────

SMILE_REFERENCE_SPECS = [
    {"dataset": "era5",    "resolution": "0.5x0.5",   "label": "ERA5/0.5°"},
    {"dataset": "era5",    "resolution": "0.25x0.25", "label": "ERA5/0.25°"},
    {"dataset": "senorge", "resolution": "",          "label": "SeNorge/1kmx1km"},]

# Define a loader for the reference values of the Hans event in the reanalysis datasets, to be used in the SMILE comparison
def _load_smile_hans_references(
    window_days: int,
    event_year: int,
    reference_start_year: int | None = None,
    reference_end_year: int | None = None,
) -> dict[str, dict]:
    """
    Load Storm Hans precipitation reference values for all three reanalysis
    reference datasets used in the SMILE comparison.

    The reference value is extracted from the SAME selected period as the SMILE
    analysis, by subsetting the already cached reanalysis time series.

    Returns
    -------
    dict keyed by cfg.smile_reference_tag(...), e.g.
        {
            "era5_0.5": {
                "dataset": "era5",
                "resolution": "0.5x0.5",
                "label": "ERA5/0.5°",
                "start_year": 1985,
                "end_year": 2024,
                "available_cache_start_year": 1941,
                "available_cache_end_year": 2024,
                "per_catchment": {
                    "nevina_bergheim": {"event_value_mm": ...},
                    ...
                },
            },
            ...
        }
    """
    from return_period import get_event_annual_max

    references: dict[str, dict] = {}

    for spec in SMILE_REFERENCE_SPECS:
        ref_dataset = spec["dataset"]
        ref_resolution = spec["resolution"]
        ref_label = spec["label"]
        ref_tag = cfg.smile_reference_tag(ref_dataset, ref_resolution)

        available_start_year, available_end_year = _infer_year_range_from_cache(
            ref_dataset, ref_resolution, window_days
        )
        if available_start_year is None:
            raise FileNotFoundError(
                f"Reference cache not found for {ref_label} ({window_days}-day).\n"
                f"No postprocessed cache files were found in:\n"
                f"  {cfg.postproc_dir(ref_dataset)}\n"
                f"Run the reanalysis section first so the cached reanalysis "
                f"time series exist before running the SMILE comparison."
            )

        use_start_year, use_end_year = _validate_requested_years(
            reference_start_year,
            reference_end_year,
            available_start_year,
            available_end_year,
            label=f"{ref_label} reference",
        )

        if not (use_start_year <= event_year <= use_end_year):
            raise ValueError(
                f"Selected reference period {use_start_year}–{use_end_year} for "
                f"{ref_label} does not include Storm Hans event year {event_year}.\n"
                f"Choose a period containing {event_year}."
            )

        if (use_start_year, use_end_year) == (available_start_year, available_end_year):
            print(
                f"  [ref] {ref_label}: using cached reanalysis period "
                f"{use_start_year}–{use_end_year}."
            )
        else:
            print(
                f"  [ref] {ref_label}: requested period {use_start_year}–{use_end_year} "
                f"exists inside cached reanalysis period "
                f"{available_start_year}–{available_end_year}; using that subset."
            )

        per_catchment: dict[str, dict[str, float]] = {}

        for slug in cfg.CATCHMENTS:
            nc_path = cfg.catchment_postproc_path(
                ref_dataset,
                ref_resolution,
                window_days,
                slug,
                available_start_year,
                available_end_year,
            )
            if not nc_path.exists():
                raise FileNotFoundError(
                    f"Reference cache missing for '{slug}'. Expected:\n  {nc_path}"
                )

            ds = load_postproc_dataset(nc_path)
            da_full = ds["tp_catchment"]
            da_selected = subset_time_series_by_year(
                da_full,
                use_start_year,
                use_end_year,
            )
            event_val, _ = get_event_annual_max(da_selected, event_year)
            per_catchment[slug] = {"event_value_mm": float(event_val)}
            ds.close()

        references[ref_tag] = {
            "dataset": ref_dataset,
            "resolution": ref_resolution,
            "label": ref_label,
            "start_year": use_start_year,
            "end_year": use_end_year,
            "available_cache_start_year": available_start_year,
            "available_cache_end_year": available_end_year,
            "per_catchment": per_catchment,
        }

    return references


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
       1-day or 2-day precipitation time series.
    2. For each member, compute annual maxima (one value per year per member).
    3. Pool all member-year annual maxima into a single flat sample
       (CESM2-LE: 100 members × 40 yr = 4 000 values;
        GFDL-SPEAR: 30 × 40 = 1 200 values).
       Each member-year is treated as an independent realisation of the climate.
    4. Fit GEV to the pooled sample. Create two single-panel figures per catchment:
       a) ERA5/0.5° Hans precipitation → its return period in the SMILE GEV
       b) ERA5/0.5° Hans return period → corresponding precipitation in the SMILE GEV
    """
    from data_smile import (
        find_smile_members,
        find_smile_files_for_member,
        load_smile_precipitation,
    )
    from return_period import get_annual_maxima
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

    print("\n  Loading Storm Hans reference values from cached reanalysis datasets ...")
    event_refs = _load_smile_hans_references(window_days=window_days, event_year=cfg.HANS_SEARCH_YEAR,reference_start_year=start_year, reference_end_year=end_year)

    for slug, title in cfg.CATCHMENTS.items():
        print(f"\n── Catchment: {title} ({slug}) ──")

        stats_path = cfg.smile_yearmax_stats_path(
            dataset, window_days, slug, start_year, end_year)

        # ── Fast path: yearly-max stats already cached ────────────────────────
        if (not force_recompute) and stats_path.exists():
            print(f"  [cache] Year-max stats found → {stats_path.name}")
            ds_stats = load_postproc_dataset(stats_path)

            annual_max_pooled = pd.Series(ds_stats["annual_max_pooled"].values).dropna()
            ds_stats.close()

        # ── Slow path: build from member caches or raw files ──────────────────
        else:
            from data_smile import get_year_range_smile
            avail_start, avail_end = get_year_range_smile(model_dir, dataset)

            weight_path = find_weight_file(dataset, "", slug, weight_dir=w_dir)
            weights = load_weights(weight_path)
            member_annual_maxima: list[pd.Series] = []

            for member_id in members:
                member_cache = cfg.smile_member_postproc_path(
                    dataset, window_days, member_id, slug, avail_start, avail_end
                )
                if (not force_recompute) and member_cache.exists():
                    ds_m = load_postproc_dataset(member_cache)
                    da_m = ds_m["tp_catchment"].load()
                    ds_m.close()
                else:
                    print(f"  [raw] Computing member {member_id} ...")
                    load_from = avail_start - 1 if window_days > 1 else avail_start
                    files = find_smile_files_for_member(
                        model_dir, dataset, member_id, load_from, avail_end
                    )
                    raw_da = load_smile_precipitation(
                        files, load_from, avail_end, unit_mode=unit_mode
                    )
                    w_aligned = align_weights_to_precip(raw_da, weights)
                    precip_roi, w_roi = crop_to_weight_bbox(raw_da, w_aligned)
                    precip_masked = precip_roi.where(w_roi > 0)
                    da_daily = compute_catchment_mean(precip_masked, w_roi).load()
                    if window_days > 1:
                        da_m = rolling_accumulation(da_daily, window_days).load()
                        da_m = da_m.isel(time=(da_m.time.dt.year >= avail_start).values)
                    else:
                        da_m = da_daily

                    _check_series_reasonableness(
                        da_m,
                        label=f"{dataset}/{slug}/member{member_id}",
                        window_days=window_days,
                    )
                    ds_out = xr.Dataset({"tp_catchment": da_m})
                    ds_out.attrs.update({
                        "dataset": dataset, "member": member_id,
                        "window_days": window_days, "catchment_slug": slug,
                        "start_year": avail_start, "end_year": avail_end,
                    })
                    save_postproc_dataset(ds_out, member_cache)

                # ── FIXED: inside member loop, runs after both fast and slow path ──
                da_m_subset = subset_time_series_by_year(da_m, start_year, end_year)
                am_member = get_annual_maxima(da_m_subset).rename(member_id)
                member_annual_maxima.append(am_member)

            # ── FIXED: pool and save stats — runs after member loop completes ──
            print(f"  Computing yearly maxima across all {n_members} members ...")
            annual_max_pooled = pool_member_annual_maxima(member_annual_maxima)

            ds_stats = xr.Dataset({
                "annual_max_pooled": xr.DataArray(
                    annual_max_pooled.values,
                    dims=["sample"],
                    attrs={"units": "mm", "description": "Pooled (member × year) annual maxima"},
                )
            })
            ds_stats.attrs.update({
                "dataset": dataset, "n_members": n_members,
                "window_days": window_days,
                "start_year": start_year, "end_year": end_year,
            })
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            ds_stats.to_netcdf(
                str(stats_path),
                encoding={"annual_max_pooled": {"zlib": True, "complevel": 4}},
            )
            print(f"    [cache] Saved → {stats_path.name}")

        # ── Figures: always generate all three precipitation-reference versions ──
        for ref_tag, ref_info in event_refs.items():
            ref_event_value_mm = ref_info["per_catchment"][slug]["event_value_mm"]

            out_paths = cfg.smile_figure_paths(
                dataset              = dataset,
                window_days          = window_days,
                catchment_slug       = slug,
                start_year           = start_year,
                end_year             = end_year,
                fig_subdir           = fig_subdir,
                reference_dataset    = ref_info["dataset"],
                reference_resolution = ref_info["resolution"],
            )
            make_smile_return_period_figure(
                annual_max_pooled = annual_max_pooled,
                catchment_title   = title,
                dataset           = dataset,
                window_days       = window_days,
                reference_value   = ref_event_value_mm,
                reference_label   = ref_info["label"],
                out_paths         = out_paths,
            )

        print(f"\n[run_all_smile] ✓ All PDFs saved to:")
        for p in cfg.smile_figure_paths(
            dataset              = dataset,
            window_days          = window_days,
            catchment_slug       = next(iter(cfg.CATCHMENTS)),
            start_year           = start_year,
            end_year             = end_year,
            fig_subdir           = fig_subdir,
            reference_dataset    = "era5",
            reference_resolution = "0.5x0.5",
        ):
            print(f"  {p.parent}")


# ── Distribution Difference helpers ───────────────────────────────────────────

def load_smile_annual_maxima_per_year(
    dataset: str,
    window_days: int,
    catchment_slug: str,
    start_year: int,
    end_year: int,
    force_recompute: bool = False,
    weight_dir: Path | None = None,
) -> dict:
    """
    Return per-year distributions of annual maxima for one SMILE dataset
    and one catchment.

    For each calendar year Y in [start_year, end_year]:
        result[Y] = np.array of shape (n_members,)
                    — one annual-maximum value per ensemble member.

    If the member-level postprocessed cache files do not yet exist for the
    requested period they are built from raw data automatically (same logic
    as run_all_smile), so this function works even on a fresh run.

    Parameters
    ----------
    dataset        : SMILE key, e.g. "cesm2_le"
    window_days    : 1 or 2
    catchment_slug : slug from cfg.CATCHMENTS
    start_year     : first year to include
    end_year       : last year to include
    force_recompute: if True, always rebuild from raw data
    weight_dir     : override weight-file search directory
    """
    from data_smile import (
        find_smile_members,
        find_smile_files_for_member,
        load_smile_precipitation,
        get_year_range_smile,
    )
    from return_period import get_annual_maxima

    smile_cfg  = cfg.SMILE_CONFIG[dataset]
    model_dir  = smile_cfg["model_dir"]
    unit_mode  = smile_cfg.get("tp24_unit_mode", "auto")
    members    = find_smile_members(model_dir, dataset)

    # Full available range — used for cache file paths
    avail_start, avail_end = get_year_range_smile(model_dir, dataset)

    # Validate requested analysis years against full available range
    start_year, end_year = _validate_requested_years(
        start_year, end_year, avail_start, avail_end, label=dataset)

    w_dir = weight_dir if weight_dir is not None else cfg.WEIGHTS_DIR
    weight_path = find_weight_file(dataset, "", catchment_slug, weight_dir=w_dir)
    weights = load_weights(weight_path)

    # Accumulate one value per (member, year) — only for requested analysis years
    year_to_vals: dict = {yr: [] for yr in range(start_year, end_year + 1)}

    for member_id in members:
        # Cache path always keyed to FULL available range (no year subfolder)
        member_cache = cfg.smile_member_postproc_path(
            dataset, window_days, member_id, catchment_slug, avail_start, avail_end)

        if (not force_recompute) and member_cache.exists():
            with xr.open_dataset(str(member_cache), use_cftime=True) as ds_m:
                da_m = ds_m["tp_catchment"].load()
        else:
            print(f"    [raw] Building member {member_id} "
                  f"({dataset}/{catchment_slug}/{window_days}day) ...")

            # Load full available range; seed year for rolling handled below
            load_from = avail_start - 1 if window_days > 1 else avail_start

            files = find_smile_files_for_member(
                model_dir, dataset, member_id, load_from, avail_end)
            raw_da = load_smile_precipitation(
                files, load_from, avail_end, unit_mode=unit_mode)

            w_aligned = align_weights_to_precip(raw_da, weights)
            precip_roi, w_roi = crop_to_weight_bbox(raw_da, w_aligned)
            precip_masked = precip_roi.where(w_roi > 0)
            da_daily = compute_catchment_mean(precip_masked, w_roi).load()

            if window_days > 1:
                da_m = rolling_accumulation(da_daily, window_days).load()
                # Remove seed year
                da_m = da_m.isel(time=(da_m.time.dt.year >= avail_start).values)
            else:
                da_m = da_daily

            _check_series_reasonableness(
                da_m,
                label=f"{dataset}/{catchment_slug}/member_{member_id}",
                window_days=window_days,
            )

            ds_out = xr.Dataset({"tp_catchment": da_m})
            ds_out.attrs.update({
                "dataset":        dataset,
                "member":         member_id,
                "window_days":    window_days,
                "catchment_slug": catchment_slug,
                "start_year":     avail_start,
                "end_year":       avail_end,
            })
            save_postproc_dataset(ds_out, member_cache)

        # Extract annual maxima — the year_to_vals loop already filters
        # to only the requested analysis years, so no explicit subset needed
        am = get_annual_maxima(da_m)
        for yr in range(start_year, end_year + 1):
            if yr in am.index:
                year_to_vals[yr].append(float(am.loc[yr]))

    return {
        yr: np.array(vals)
        for yr, vals in year_to_vals.items()
        if len(vals) > 0
    }

# Define a loader for the reference values of the Hans event in the reanalysis datasets, to be used in the distribution difference analysis
def load_reanalysis_values_per_year(
    dataset: str,
    resolution: str,
    window_days: int,
    catchment_slug: str,
    start_year: int | None = None,
    end_year: int | None = None,
    data_type: str = "annual_max",   # "annual_max" or "daily"
) -> dict:
    """
    Load per-year data from a cached reanalysis dataset (ERA5 or SeNorge).

    Returns
    -------
    dict[int, np.ndarray]
        data_type == "annual_max" → {year: array([single_max_value])}
        data_type == "daily"      → {year: array(all_finite_daily_values)}

    This is structurally identical to what load_smile_annual_maxima_per_year
    produces, so compute_distribution_difference works unchanged on the output.
    """
    available_start, available_end = get_cached_year_range(dataset, resolution, window_days)
    if available_start is None:
        raise FileNotFoundError(
            f"No {window_days}-day cache found for {dataset}/{resolution}.\n"
            f"Run the reanalysis pipeline first."
        )

    use_start, use_end = _validate_requested_years(
        start_year, end_year, available_start, available_end,
        label=f"{dataset}/{resolution}",
    )

    nc = cfg.catchment_postproc_path(
        dataset, resolution, window_days, catchment_slug,
        available_start, available_end,
    )
    if not nc.exists():
        raise FileNotFoundError(f"Cache file not found:\n  {nc}")

    with xr.open_dataset(str(nc)) as ds_nc:
        da = subset_time_series_by_year(ds_nc["tp_catchment"], use_start, use_end)
        da = da.load()

    result: dict = {}
    for year in range(use_start, use_end + 1):
        year_mask = da.time.dt.year == year
        da_yr = da.isel(time=year_mask.values)
        vals = da_yr.values
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        if data_type == "annual_max":
            result[year] = np.array([float(vals.max())])
        else:                            # "daily"
            result[year] = vals

    return result


# Define a loader for the daily (or rolling) values of the Hans event in the reanalysis datasets, to be used in the distribution difference analysis
def load_smile_daily_values_per_year(
    dataset: str,
    window_days: int,
    catchment_slug: str,
    start_year: int,
    end_year: int,
    force_recompute: bool = False,
    weight_dir: Path | None = None,
) -> dict:
    """
    Load per-year daily (or rolling) precipitation values for a SMILE dataset,
    concatenated across all ensemble members.

    Returns
    -------
    dict[int, np.ndarray]
        {year: array of all finite daily values across ALL members for that year}

    This parallels load_smile_annual_maxima_per_year but keeps every daily
    value instead of only the per-member annual maximum.  The result feeds
    directly into compute_distribution_difference for the "daily" analysis.
    """
    from data_smile import (
        find_smile_members,
        find_smile_files_for_member,
        load_smile_precipitation,
        get_year_range_smile,)

    smile_cfg  = cfg.SMILE_CONFIG[dataset]
    model_dir  = smile_cfg["model_dir"]
    unit_mode  = smile_cfg.get("tp24_unit_mode", "auto")
    members    = find_smile_members(model_dir, dataset)

    avail_start, avail_end = get_year_range_smile(model_dir, dataset)
    start_year, end_year   = _validate_requested_years(
        start_year, end_year, avail_start, avail_end, label=dataset)

    w_dir       = weight_dir if weight_dir is not None else cfg.WEIGHTS_DIR
    weight_path = find_weight_file(dataset, "", catchment_slug, weight_dir=w_dir)
    weights     = load_weights(weight_path)

    year_to_vals: dict = {yr: [] for yr in range(start_year, end_year + 1)}

    for member_id in members:
        member_cache = cfg.smile_member_postproc_path(
            dataset, window_days, member_id, catchment_slug, avail_start, avail_end)

        if (not force_recompute) and member_cache.exists():
            with xr.open_dataset(str(member_cache), use_cftime=True) as ds_m:
                da_m = ds_m["tp_catchment"].load()
        else:
            print(f"    [raw] Building member {member_id} "
                  f"({dataset}/{catchment_slug}/{window_days}day) ...")
            load_from = avail_start - 1 if window_days > 1 else avail_start
            files = find_smile_files_for_member(
                model_dir, dataset, member_id, load_from, avail_end)
            raw_da = load_smile_precipitation(
                files, load_from, avail_end, unit_mode=unit_mode)
            w_aligned = align_weights_to_precip(raw_da, weights)
            precip_roi, w_roi = crop_to_weight_bbox(raw_da, w_aligned)
            precip_masked = precip_roi.where(w_roi > 0)
            da_daily = compute_catchment_mean(precip_masked, w_roi).load()
            if window_days > 1:
                da_m = rolling_accumulation(da_daily, window_days).load()
                da_m = da_m.isel(time=(da_m.time.dt.year >= avail_start).values)
            else:
                da_m = da_daily
            _check_series_reasonableness(
                da_m,
                label=f"{dataset}/{catchment_slug}/member_{member_id}",
                window_days=window_days,
            )
            ds_out = xr.Dataset({"tp_catchment": da_m})
            ds_out.attrs.update({
                "dataset": dataset, "member": member_id,
                "window_days": window_days, "catchment_slug": catchment_slug,
                "start_year": avail_start, "end_year": avail_end,
            })
            save_postproc_dataset(ds_out, member_cache)

        # Collect daily values year-by-year (analysis window only)
        for yr in range(start_year, end_year + 1):
            year_mask = da_m.time.dt.year == yr
            da_yr = da_m.isel(time=year_mask.values)
            vals = da_yr.values
            vals = vals[np.isfinite(vals)]
            year_to_vals[yr].extend(vals.tolist())

    return {
        yr: np.array(vals)
        for yr, vals in year_to_vals.items()
        if len(vals) > 0}


# Define function to create distribution difference Plots and analysis
def compute_distribution_difference(
    per_year_data: dict,
    window_before: int,
    window_after: int,
    bin_width_mm: float = 5.0,
) -> tuple:
    """
    Compute the average normalised-histogram difference (after − before) over
    all non-overlapping consecutive window pairs in the time series.

    Pairing logic — no year is used twice:
        step = window_before + window_after
        pair k:  before = years[ k*step           : k*step + window_before ]
                 after  = years[ k*step+window_before : (k+1)*step         ]

    For each pair:
        1. Collect all member-year values in the "before" window.
        2. Collect all member-year values in the "after"  window.
        3. Build normalised frequency histograms over shared bin edges.
        4. difference = after_hist − before_hist  (element-wise, sums to ≈ 0)

    Average all pair differences → one scalar per bin.

    Parameters
    ----------
    per_year_data   : dict[int, np.ndarray]  — output of load_smile_annual_maxima_per_year
    window_before   : number of years in the "before" half of each pair
    window_after    : number of years in the "after"  half of each pair
    bin_width_mm    : histogram bin width in mm.
                      5 mm gives ~16 bins for 1-day data (range ≈10–90 mm)
                      and ~25 bins for 2-day data (range ≈15–140 mm).
                      Wide enough so each bin holds multiple values even for
                      the smallest window (e.g. GFDL-SPEAR: 30 members × 1 yr
                      = 30 values → still ≥ 1–2 values per 5 mm bin on average).

    Returns
    -------
    bin_centers : np.ndarray
    avg_diff    : np.ndarray  (same length; values nominally in [−1, +1])
    n_pairs     : int         (number of non-overlapping pairs averaged)
    """
    years = sorted(per_year_data.keys())
    step  = window_before + window_after
    n_years = len(years)

    # Global bin edges derived from the full data range so all pairs share
    # the same axis — essential for a meaningful average difference
    all_vals = np.concatenate(list(per_year_data.values()))
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        return np.array([]), np.array([]), 0

    global_min = float(np.floor(all_vals.min() / bin_width_mm) * bin_width_mm)
    global_max = float(np.ceil(all_vals.max()  / bin_width_mm) * bin_width_mm) + bin_width_mm
    bin_edges  = np.arange(
        global_min, global_max + bin_width_mm * 0.5, bin_width_mm
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    diffs = []
    i = 0
    while i + step <= n_years:
        before_years = years[i                : i + window_before]
        after_years  = years[i + window_before: i + step]

        before_vals = np.concatenate([per_year_data[y] for y in before_years])
        after_vals  = np.concatenate([per_year_data[y] for y in after_years])
        before_vals = before_vals[np.isfinite(before_vals)]
        after_vals  = after_vals[np.isfinite(after_vals)]

        if before_vals.size == 0 or after_vals.size == 0:
            i += step
            continue

        h_before, _ = np.histogram(before_vals, bins=bin_edges)
        h_after,  _ = np.histogram(after_vals,  bins=bin_edges)

        # Normalise to probability mass (each histogram sums to 1)
        if h_before.sum() > 0:
            h_before = h_before / h_before.sum()
        if h_after.sum() > 0:
            h_after  = h_after  / h_after.sum()

        diffs.append(h_after.astype(float) - h_before.astype(float))
        i += step

    n_pairs = len(diffs)
    if n_pairs == 0:
        return bin_centers, np.zeros_like(bin_centers), 0

    avg_diff = np.mean(diffs, axis=0)
    return bin_centers, avg_diff, n_pairs



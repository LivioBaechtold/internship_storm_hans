# catchment_tools.py
# Weight file discovery, grid alignment, weighted catchment-mean computation, NetCDF caching, and the main loop that orchestrates all catchments.

# Import important libraries
import re
import json
from datetime import datetime, timezone
import xarray as xr
import numpy as np
from pathlib import Path
import pandas as pd

import config_paths as cfg
from return_period import get_annual_maxima


# Define catchment processing helpers ───────────────────────────────────────────────
def _infer_year_range_from_cache(
    dataset: str, resolution: str, window_days: int) -> tuple:
    cache_dir = cfg.postproc_dir(dataset) / "catchment_averaged"
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
        f"Expected one of: (latitude, longitude), (lat, lon), (Y, X), (y, x)")

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

# ── Spatial-cache I/O helpers ─────────────────────────────────────────────────

def _chunksizes_for(da: xr.DataArray, time_chunk: int,
                    y_chunk: int, x_chunk: int) -> tuple:
    """Return NetCDF chunksizes aligned to da.dims order."""
    mapping = {"time": time_chunk,
               "Y": y_chunk, "latitude": y_chunk, "lat": y_chunk,
               "X": x_chunk, "longitude": x_chunk, "lon": x_chunk}
    return tuple(min(int(da.sizes[d]), mapping.get(d, int(da.sizes[d])))
                 for d in da.dims)


def save_spatial_netcdf(da: xr.DataArray, out_path: Path, var_name: str,
                        time_chunk: int, y_chunk: int, x_chunk: int,
                        units: str = "mm") -> None:
    """Write one float32 spatial variable to a compressed, chunked NetCDF file.

    units : metadata unit string written to the variable (default 'mm';
            pass 'kg/m2' for SWE / soil-moisture fields).
    """
    da = da.astype("float32")
    da.name = var_name
    da.attrs["units"] = units
    chunksizes = _chunksizes_for(da, time_chunk, y_chunk, x_chunk)
    chunk_map = dict(zip(da.dims, chunksizes))
    da = da.chunk(chunk_map)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    da.to_dataset(name=var_name).to_netcdf(
        str(out_path), mode="w", format="NETCDF4",
        encoding={var_name: {"zlib": True, "complevel": 4,
                             "dtype": "float32", "chunksizes": chunksizes}})
    print(f"  [saved] {out_path.name}  ({dict(da.sizes)})")


def open_precip_cache(cache_path: Path, start_year: int, end_year: int,
                      **open_kwargs) -> xr.DataArray:
    """
    Open a spatial precipitation cache lazily and subset to [start_year, end_year].
    Do NOT wrap in a 'with' block — the file handle must remain open for lazy Dask computation.
    """
    open_kwargs.setdefault("chunks", {})
    ds = xr.open_dataset(str(cache_path), **open_kwargs)
    for varname in ("tp24_mm", "rr_mm"):
        if varname in ds:
            return subset_time_series_by_year(ds[varname], start_year, end_year)
    ds.close()
    raise KeyError(f"No recognised precipitation variable in {cache_path}")

def open_field_cache(cache_path: Path, var_name: str,
                     start_year: int, end_year: int, **open_kwargs) -> xr.DataArray:
    """
    Open a spatial state-field cache (SWE / soil moisture) lazily and subset to
    [start_year, end_year]. Generic counterpart of `open_precip_cache` for an
    explicitly named variable. Do NOT wrap in a 'with' block — the handle must
    stay open for lazy Dask computation.
    """
    open_kwargs.setdefault("chunks", {})
    ds = xr.open_dataset(str(cache_path), **open_kwargs)
    if var_name not in ds:
        ds.close()
        raise KeyError(f"Variable {var_name!r} not found in {cache_path}")
    return subset_time_series_by_year(ds[var_name], start_year, end_year)


def crop_weight_field_to_nonzero_bbox(da: xr.DataArray,
                                      pad_cells: int = 1) -> xr.DataArray:
    """Crop a weight DataArray to the bounding box of its positive cells plus padding."""
    y_dim, x_dim = _spatial_dims(da)
    valid = np.isfinite(da.values) & (da.values > 0)
    if not valid.any():
        raise ValueError("No positive weights found — check the weight file.")
    iy, ix = np.where(valid)
    y0 = max(0, int(iy.min()) - pad_cells)
    y1 = min(da.sizes[y_dim] - 1, int(iy.max()) + pad_cells)
    x0 = max(0, int(ix.min()) - pad_cells)
    x1 = min(da.sizes[x_dim] - 1, int(ix.max()) + pad_cells)
    return da.isel({y_dim: slice(y0, y1 + 1), x_dim: slice(x0, x1 + 1)})


def _mean_step(coord: xr.DataArray, fallback: float) -> float:
    """Mean absolute step size along a coordinate array."""
    vals = np.asarray(coord.values)
    if vals.size < 2:
        return fallback
    return float(np.abs(np.diff(vals)).mean())


def get_plot_extent_and_crs(da: xr.DataArray):
    """
    Infer a tight map extent and CRS from a spatial DataArray.
    Returns (extent_list, cartopy_crs).
    Supports latitude/longitude, lat/lon, and Y/X (UTM) grids.
    """
    import cartopy.crs as ccrs
    if {"longitude", "latitude"}.issubset(da.dims):
        dx = _mean_step(da["longitude"], 0.25)
        dy = _mean_step(da["latitude"], 0.25)
        extent = [float(da["longitude"].min()) - 0.5*dx,
                  float(da["longitude"].max()) + 0.5*dx,
                  float(da["latitude"].min())  - 0.5*dy,
                  float(da["latitude"].max())  + 0.5*dy]
        return extent, ccrs.PlateCarree()
    if {"lon", "lat"}.issubset(da.dims):
        dx = _mean_step(da["lon"], 0.25)
        dy = _mean_step(da["lat"], 0.25)
        extent = [float(da["lon"].min()) - 0.5*dx,
                  float(da["lon"].max()) + 0.5*dx,
                  float(da["lat"].min()) - 0.5*dy,
                  float(da["lat"].max()) + 0.5*dy]
        return extent, ccrs.PlateCarree()
    if {"X", "Y"}.issubset(da.dims):
        dx = _mean_step(da["X"], 1000.0)
        dy = _mean_step(da["Y"], 1000.0)
        extent = [float(da["X"].min()) - 0.5*dx,
                  float(da["X"].max()) + 0.5*dx,
                  float(da["Y"].min()) - 0.5*dy,
                  float(da["Y"].max()) + 0.5*dy]
        return extent, ccrs.UTM(zone=33)
    raise ValueError(f"Unsupported weight-grid dimensions: {da.dims}")


# ── 2day-rolling accumulation helper ───
# ── Rolling-sum accumulation (works for 1-D catchment series AND 2-D spatial fields) ───
def rolling_accumulation(
    da: xr.DataArray,
    window_days: int = 2,
    var_name: str = None,
) -> xr.DataArray:
    """
    Apply a rolling sum over the 'time' dimension for any DataArray.

    Works unchanged for:
      • 1-D catchment-averaged series (used by run_all, SMILE loaders)
      • 2-D spatial fields (used by map-generation notebooks for e.g. 2-day mean maps)

    Parameters
    ----------
    da          : input DataArray with a 'time' dimension
    window_days : accumulation window length in days
    var_name    : explicit name for the output DataArray.
                  If None, the input name (da.name) is preserved.
    """
    out = da.rolling(time=window_days, min_periods=window_days).sum()
    out.attrs["units"] = "mm"
    out.attrs["long_name"] = f"{window_days}-day accumulated precipitation"
    if var_name is not None:
        out.name = var_name
    elif da.name is not None:
        out.name = da.name
    return out


def rolling_change(
    da: xr.DataArray,
    window_days: int = 2,
    var_name: str = None,
) -> xr.DataArray:
    """
    Rolling 'last minus first' change over a trailing time window.

    The value at time t is da(t) - da(t - (window_days - 1)) — the change across
    the SAME trailing window that `rolling_accumulation` sums. For window_days=2
    this is da(t) - da(t-1): the later value minus the earlier one, labelled at
    the later timestep, mirroring the 2-day rolling sum used for precip
    (value[t] = P[t-1] + P[t]). The first (window_days-1) entries are NaN, just
    like the rolling sum.

    Use for state variables such as SWE, where the 2-day quantity is a change
    (SWE gain positive, melt negative). Units are inherited from `da`
    (NOT forced to mm).
    """
    out = da - da.shift(time=window_days - 1)
    out.attrs["long_name"] = f"{window_days}-day change (later minus earlier)"
    if var_name is not None:
        out.name = var_name
    elif da.name is not None:
        out.name = da.name
    return out


def rolling_melt(
    da: xr.DataArray,
    window_days: int = 2,
    var_name: str = None,
) -> xr.DataArray:
    """
    Snowmelt magnitude over a trailing time window: max(0, -ΔSWE).

    The signed change is da(t) - da(t-(window_days-1)) (later minus earlier, the
    same trailing window as `rolling_change`). Snowmelt is the part of that change
    where SWE *decreases*, expressed as a POSITIVE quantity:

        melt(t) = max(0, -(da(t) - da(t-(window_days-1))))

    Accumulation (SWE gain) maps to 0, so the field is a melt flux suitable for a
    snowmelt / compound-flood map: 0 in snow-free seasons, positive during melt,
    with the flood-relevant signal carried by the 90th percentile. The first
    (window_days-1) entries are NaN (inherited from the shift). Units are inherited
    from `da` (NOT forced to mm).
    """
    change = da - da.shift(time=window_days - 1)
    out = (-change).clip(min=0.0)
    out.attrs["long_name"] = f"{window_days}-day snowmelt = max(0, -ΔSWE)"
    if var_name is not None:
        out.name = var_name
    elif da.name is not None:
        out.name = da.name
    return out


def rolling_identity(
    da: xr.DataArray,
    window_days: int = 2,
    var_name: str = None,
) -> xr.DataArray:
    """
    Pass-through 'rolling' op for fields whose window quantity is ALREADY computed
    and stored on disk (e.g. the daily signed-snowmelt ΔSWE cache built in
    load_data_store_postprocessed.ipynb). Returns `da` unchanged so the generic
    window-median / window-p90 compute functions can operate directly on the
    stored difference. `window_days` is accepted for signature compatibility and
    ignored.
    """
    out = da
    if var_name is not None:
        out = out.rename(var_name)
    return out


def rolling_mean(
    da: xr.DataArray,
    window_days: int = 2,
    var_name: str = None,
) -> xr.DataArray:
    """
    Rolling mean over a trailing time window.

    The value at time t is the mean of `da` over [t-(window_days-1), t] — the SAME
    trailing window that `rolling_accumulation` sums. For window_days=2 this is
    mean(da(t-1), da(t)): the 2-day rolling average, labelled at the later
    timestep, mirroring the 2-day rolling sum used for precip. The first
    (window_days-1) entries are NaN, just like the rolling sum.

    Use for state variables such as soil moisture, where the 2-day quantity is an
    average wetness (not an accumulation or a change). Units are inherited from
    `da` (NOT forced to mm).
    """
    out = da.rolling(time=window_days, min_periods=window_days).mean()
    out.attrs["long_name"] = f"{window_days}-day rolling mean"
    if var_name is not None:
        out.name = var_name
    elif da.name is not None:
        out.name = da.name
    return out


# ── CESM2-LE catchment-averaged compound series (precip / SM / snowmelt) ──────
# One cache per (window, catchment, variable) with ALL common members as a
# [member, time] array — consumed by the joint-distribution cell in
# compound_flood_risk_analysis.ipynb.

def common_cesm2_le_members() -> list[str]:
    """
    Member IDs present in ALL three CESM2-LE variable directories (PRECT∩SM∩SWE).

    PRECT has 100 members; SM and SWE exist for only 90 (odd members 001–019
    missing), so the common set is those 90 members.
    """
    from data_smile import find_smile_members
    common = (set(find_smile_members(cfg.CESM2_LE_DIR, "cesm2_le"))
              & set(find_smile_members(cfg.CESM2_LE_SM_DIR, "cesm2_le"))
              & set(find_smile_members(cfg.CESM2_LE_SWE_DIR, "cesm2_le")))
    return sorted(common)


# Per-variable I/O configuration for the compound series (spatial member cache
# path builder, variable name inside that cache, output units, raw dir for the
# available-year lookup).
_CESM2_COMPOUND_SPECS: dict[str, dict] = {
    "precipitation": dict(
        raw_dir_key="CESM2_LE_DIR", cache_var="tp24_mm", units="mm",
        path_fn=lambda mid, s, e: cfg.overall_precip_member_path("cesm2_le", mid, s, e)),
    "soil_moisture": dict(
        raw_dir_key="CESM2_LE_SM_DIR", cache_var="soil_moisture", units="kg/m2",
        path_fn=lambda mid, s, e: cfg.field_daily_cache_path(
            "cesm2_le", "soil_moisture", s, e, member_id=mid)),
    "snowmelt": dict(
        raw_dir_key="CESM2_LE_SWE_DIR", cache_var="swe", units="kg/m2",
        path_fn=lambda mid, s, e: cfg.field_daily_cache_path(
            "cesm2_le", "swe", s, e, member_id=mid)),
}


def _cesm2_compound_window_op(variable: str, da_daily: xr.DataArray,
                              window_days: int) -> xr.DataArray:
    """Window op on a daily catchment series — same ops as the map pipeline:
    precipitation → rolling SUM, soil_moisture → rolling MEAN,
    snowmelt → max(0, −ΔSWE) over the window (rolling_melt; positive melt).
    Swap rolling_melt → rolling_change below for the raw signed ΔSWE instead."""
    if variable == "precipitation":
        return da_daily if window_days == 1 else rolling_accumulation(da_daily, window_days)
    if variable == "soil_moisture":
        return da_daily if window_days == 1 else rolling_mean(da_daily, window_days)
    if variable == "snowmelt":
        if window_days < 2:
            raise ValueError(
                "snowmelt requires window_days >= 2 (ΔSWE = SWE(t) − SWE(t−(N−1))).")
        return rolling_melt(da_daily, window_days)
    raise ValueError(f"Unknown variable {variable!r}; "
                     f"expected one of {sorted(_CESM2_COMPOUND_SPECS)}")


def save_cesm2_le_catchment_field_series(
    variable: str,
    window_days: int,
    catchment_slug: str,
    force: bool = False,
) -> Path:
    """
    Build/save the catchment-averaged CESM2-LE compound series for one
    (variable, window, catchment) over the FULL available record (1920–2034).

    Output: ONE NetCDF [member, time] at cfg.cesm2_catchment_field_path(...),
    containing every member available for ALL three variables (90 members).

    Methodology per member (identical to the existing catchment pipeline):
      spatial daily cache → align + crop weights → weighted catchment mean →
      window op (_cesm2_compound_window_op). NOTE: the catchment mean is taken
      FIRST and the window op applied to the 1-D series — same order as
      run_all_smile; for the nonlinear melt clip this differs slightly from the
      per-pixel spatial-map pipeline.
    """
    from data_smile import get_year_range_smile
    spec = _CESM2_COMPOUND_SPECS[variable]
    avail_s, avail_e = get_year_range_smile(getattr(cfg, spec["raw_dir_key"]), "cesm2_le")
    out_path = cfg.cesm2_catchment_field_path(
        window_days, catchment_slug, variable, avail_s, avail_e)

    if (not force) and out_path.exists():
        print(f"  [skip] exists: {out_path.name}")
        return out_path

    members = common_cesm2_le_members()
    weights = load_weights(find_weight_file("cesm2_le", "", catchment_slug))
    print(f"  [build] {variable} | {window_days}-day | {catchment_slug} | "
          f"{len(members)} members | {avail_s}–{avail_e}")

    member_series: list[xr.DataArray] = []
    for i, mid in enumerate(members, start=1):
        cache = spec["path_fn"](mid, avail_s, avail_e)
        if not cache.exists():
            raise FileNotFoundError(
                f"Spatial member cache missing: {cache}\n"
                "Build the daily spatial caches first in "
                "load_data_store_postprocessed.ipynb (cells "
                "'# %% [Overall precipitation caches …]' and "
                "'# %% [SWE & soil-moisture daily caches …]').")
        da = open_field_cache(cache, spec["cache_var"], avail_s, avail_e)
        w_aligned = align_weights_to_precip(da, weights)
        da_roi, w_roi = crop_to_weight_bbox(da, w_aligned)
        da_daily = compute_catchment_mean(da_roi.where(w_roi > 0), w_roi).load()
        member_series.append(
            _cesm2_compound_window_op(variable, da_daily, window_days).astype("float32"))
        if i % 10 == 0:
            print(f"    {i}/{len(members)} members processed ...")

    # Fixed-width member IDs: netCDF4 cannot serialise the variable-width
    # StringDType that pd.Index produces under numpy 2.
    da_all = xr.concat(member_series, dim="member")
    da_all = da_all.assign_coords(member=np.array(members, dtype="U3"))

    da_all.name = variable
    da_all.attrs.update({
        "units": spec["units"], "window_days": window_days,
        "catchment_slug": catchment_slug,
        "long_name": f"{window_days}-day catchment-averaged {variable} (CESM2-LE)"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    da_all.to_dataset(name=variable).to_netcdf(
        str(out_path), encoding={variable: {"zlib": True, "complevel": 4}})
    print(f"  [saved] {out_path.name}")
    return out_path


def parse_member_selection(selection, available: list[str]) -> list[str]:
    """
    Normalise a member selection against the available member IDs.

    selection : "all" | range string "1-30" | comma string "3,4,27"
                | list of ints/strings, e.g. [3, 4, 5, 27, 35] or ["003", "027"]
    Raises ValueError listing every requested-but-unavailable member.
    """
    if isinstance(selection, str):
        s = selection.strip().lower()
        if s == "all":
            return list(available)
        if "-" in s:
            lo, hi = (int(p) for p in s.split("-", 1))
            requested = [f"{i:03d}" for i in range(lo, hi + 1)]
        else:
            requested = [f"{int(p):03d}" for p in s.split(",")]
    else:
        requested = [f"{int(m):03d}" for m in selection]

    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(
            f"Requested member(s) not available: {', '.join(missing)}.\n"
            f"Available ({len(available)}): {', '.join(available)}\n"
            "CESM2-LE SM & SWE exist for only 90 members (odd members 001–019 "
            "have no data), so the compound series contain exactly those 90 "
            "members for every variable, precipitation included.\n"
            "→ Adjust JD_MEMBERS (joint-distribution cells) or FE_MEMBERS "
            "(frequency-evolution cells) in compound_flood_risk_analysis.ipynb.")
    return requested


def load_cesm2_le_catchment_field_series(
    variable: str,
    window_days: int,
    catchment_slug: str,
    start_year: int | None = None,
    end_year: int | None = None,
    members="all",
) -> xr.DataArray:
    """
    Load one compound-series cache [member, time] and subset by years + members.

    Every unavailable selection raises an informative error stating WHERE to
    build/fix it (notebook + cell) — the notification mechanism used by the
    joint-distribution cell.
    """
    cache_dir = cfg.POSTPROC_DIR / "cesm2_le" / "catchment_averaged"
    pattern = (f"post_processed_cesm2_le_{cfg.acc_tag(window_days)}_"
               f"{catchment_slug}_{variable}_*.nc")
    hits = sorted(cache_dir.glob(pattern))
    if not hits:
        raise FileNotFoundError(
            f"No postprocessed series for variable='{variable}', "
            f"window_days={window_days}, catchment='{catchment_slug}'.\n"
            f"Expected a file matching:\n  {cache_dir / pattern}\n"
            "→ First calculate and save it inside load_data_store_postprocessed.ipynb, "
            "cell \"# %% [CESM2-LE catchment compound series — run once per window "
            f"selection]\" (set WINDOW_DAYS_COMPOUND = {window_days} there and re-run).")

    nc_path = hits[-1]
    m = re.search(r"_(\d{4})-(\d{4})\.nc$", nc_path.name)
    file_s, file_e = int(m.group(1)), int(m.group(2))

    use_s = file_s if start_year is None else int(start_year)
    use_e = file_e if end_year   is None else int(end_year)
    if use_s > use_e or use_s < file_s or use_e > file_e:
        raise ValueError(
            f"Requested time range {use_s}–{use_e} is not covered by the stored "
            f"series {file_s}–{file_e} ({nc_path.name}).\n"
            "→ Adjust JD_START / JD_END in the joint-distribution cell of "
            "compound_flood_risk_analysis.ipynb (or rebuild the series in "
            "load_data_store_postprocessed.ipynb if the stored record is outdated).")

    ds = xr.open_dataset(
        str(nc_path),
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
    da = ds[variable]
    avail_members = [v.decode() if isinstance(v, bytes) else str(v)
                     for v in da["member"].values]
    da = da.assign_coords(member=avail_members)
    da = da.sel(member=parse_member_selection(members, avail_members))
    return subset_time_series_by_year(da, use_s, use_e)


# Define Check Reasonableness for Series
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


# ── Compound absolute-threshold statistics (joint-distribution analysis) ──────
# Normalised compound score  s = x/x_max + y/y_max  for the two variables of a
# joint-distribution pair; s >= threshold marks a compound extreme. Pure array
# maths, no I/O. Consumed by the threshold cell of
# compound_flood_risk_analysis.ipynb, which hands x_max/y_max on to
# plot_style.make_joint_distribution_figure(threshold=...) so the drawn line and
# the printed statistics always use identical denominators.

def compound_threshold_stats(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    threshold: float,
    x_max: float | None = None,
    y_max: float | None = None,
) -> dict:
    """
    Absolute compound-threshold statistics for one (x, y) sample.

    Parameters
    ----------
    x_vals, y_vals : np.ndarray
        Flattened (member × date) values of the two joint-distribution
        variables — the same arrays that are scattered in the figure.
    threshold : float
        Right-hand side of  x/x_max + y/y_max >= threshold  (e.g. 0.9).
    x_max, y_max : float, optional
        Normalisation maxima. Default: the sample maxima of the finite pairs,
        i.e. the maxima of exactly the period/member selection that is plotted.

    Returns
    -------
    dict
        x_max, y_max, threshold : denominators and criterion actually used
        score                   : x/x_max + y/y_max for every point
        mask                    : bool array, finite AND score >= threshold
        n_total, n_exceed, frac_exceed : sample size, exceedances, fraction
        x_at_y0                 : x where the threshold line crosses y = 0
        y_at_x0                 : y where the threshold line crosses x = 0
    """
    x = np.asarray(x_vals, dtype=float).ravel()
    y = np.asarray(y_vals, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(
            f"x_vals and y_vals must have the same length ({x.size} vs {y.size}).")

    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        raise ValueError(
            "compound_threshold_stats: no finite (x, y) pairs — check the "
            "selected variables/period in the joint-distribution cell of "
            "compound_flood_risk_analysis.ipynb.")

    x_max = float(np.nanmax(x[finite])) if x_max is None else float(x_max)
    y_max = float(np.nanmax(y[finite])) if y_max is None else float(y_max)
    if x_max <= 0.0 or y_max <= 0.0:
        raise ValueError(
            f"Normalisation maxima must be > 0 (got x_max={x_max}, y_max={y_max}); "
            "x/x_max + y/y_max is undefined otherwise.")
    if threshold <= 0.0:
        raise ValueError(
            f"threshold must be > 0 (got {threshold}). "
            "→ Set JD_THRESHOLD in the threshold cell of "
            "compound_flood_risk_analysis.ipynb.")
    if threshold > 2.0:
        print(f"  [warning] threshold={threshold:g} is above the maximum possible "
              "score of 2.0 (both variables simultaneously at their maximum) — "
              "no point can satisfy the criterion.")

    score = x / x_max + y / y_max
    mask  = finite & (score >= threshold)
    n_total, n_exceed = int(finite.sum()), int(mask.sum())
    return {
        "x_max":       x_max,
        "y_max":       y_max,
        "threshold":   float(threshold),
        "score":       score,
        "mask":        mask,
        "n_total":     n_total,
        "n_exceed":    n_exceed,
        "frac_exceed": n_exceed / n_total,
        "x_at_y0":     float(threshold) * x_max,
        "y_at_x0":     float(threshold) * y_max,
    }


# ── Compound-extreme frequency evolution (rolling-window analysis) ────────────
# Time-evolution counterpart of the static joint-distribution / threshold cells.
# Pure data preparation + statistics — NO plotting (both figures live in
# plot_style.plot_internal_variability_trend / plot_signal_to_noise_ratio).
#
# Definitions are IDENTICAL to `compound_threshold_stats`: compound severity
# s = x/x_max + y/y_max, exceedance when s >= threshold. The only difference is
# that x_max / y_max are FROZEN on the reference period AND on the selected
# season, so the criterion is ONE fixed line in the (x, y) plane for every
# rolling window and a drifting denominator can never masquerade as a change.
#
# Accepted limitations (deliberate — see the notebook selection block):
#   • no declustering — one storm can exceed on up to FE_WINDOW_DAYS consecutive
#     days, so an "event" here is an exceedance DAY;
#   • rolling windows overlap by (L − FE_ROLL_STEP) years, so consecutive points
#     are not independent — the curve is a display-only low-pass filter.

# Figure-1 spread flags — WHICH spread elements are drawn. "iqr" is the blue
# 25th-75th-percentile band (the box of a classic box plot); "minmax" and
# "p025p975" are the dashed/dotted envelope and are mutually exclusive (they share
# one line style).
FREQ_SPREAD_KINDS   = ("iqr", "minmax", "p025p975")

# HOW the band/envelope numbers are obtained — orthogonal to WHICH of them are
# drawn (FREQ_SPREAD_KINDS above). Exactly two options, both percentile-based:
#   "percentile_empirical" → np.percentile. Because a member rate is an integer
#                            count / L these land exactly on multiples of the rate
#                            quantum 1/L, so the curves form a staircase and the
#                            IQR band can collapse to zero width. Kept as the
#                            reference the grouped version is compared against.
#   "percentile_grouped"   → the interpolated (grouped) percentiles of
#                            `grouped_percentile`, which remove that staircase.
# There is no standard-deviation / "std" option: the ±1σ band was replaced by the
# IQR band and the legacy "std" alias has been removed everywhere.
FREQ_SPREAD_METHODS = ("percentile_empirical", "percentile_grouped")

# Ensemble-table columns Figure 1 reads for each method. The CENTRAL line is
# method-independent and added by `frequency_spread_columns` below.
_FREQ_SPREAD_COLUMNS: dict[str, dict[str, str]] = {
    "percentile_empirical": dict(band_lo="p25",         band_hi="p75",
                                 env_lo="p025",         env_hi="p975"),
    "percentile_grouped":   dict(band_lo="p25_grouped", band_hi="p75_grouped",
                                 env_lo="p025_grouped", env_hi="p975_grouped"),
}


def frequency_spread_columns(spread_method: str) -> dict[str, str]:
    """Ensemble-table column names Figure 1 must read for one `spread_method`.

    Keeps the method → column mapping out of the notebook (orchestration only) and
    out of plot_style (drawing only): the Figure-1 cell looks the columns up here
    and hands plot_style plain arrays.

    Returns
    -------
    dict with keys
        central : ALWAYS "f_mean" — the central line of every figure in this
                  analysis is the ensemble MEAN, whose resolution is 1/(M·L)
                  rather than the 1/L of a rank statistic, and it is the same
                  quantity the signal-to-noise figure divides by sigma. NO median
                  is computed or drawn anywhere in this analysis.
        band_lo/band_hi, env_lo/env_hi : the p25/p75 band and the p2.5/p97.5
                  envelope of the selected method.
    """
    m = str(spread_method).strip().lower()
    if m not in FREQ_SPREAD_METHODS:
        raise ValueError(f"spread_method={spread_method!r} invalid; "
                         f"pick one of {list(FREQ_SPREAD_METHODS)}.")
    cols = dict(_FREQ_SPREAD_COLUMNS[m])
    cols["central"] = "f_mean"
    return cols


# ── Season handling ───────────────────────────────────────────────────────────
def resolve_season_months(season) -> list[int] | None:
    """Month numbers for a season selector; `None` means 'all months'.

    season : "all" | key of cfg.SEASON_MONTHS ("DJF", "MAM", "JJA", "SON", "MAMJ")

    A season is ALWAYS a named key, never an ad-hoc month range: every window the
    analysis may use is defined once in cfg.SEASON_MONTHS, so the filename tag,
    the legend label and the month list can never disagree. Add a new window
    there (plus its label in cfg.SEASON_LABELS) rather than passing months here.
    """
    key = str(season).strip()
    if key.lower() == "all":
        return None
    if key not in cfg.SEASON_MONTHS:
        raise ValueError(f"FE_SEASON/JD_SEASON={season!r} unknown. Use 'all' or one of "
                         f"{sorted(cfg.SEASON_MONTHS)} — new named windows go into "
                         "cfg.SEASON_MONTHS (and cfg.SEASON_LABELS).")
    return list(cfg.SEASON_MONTHS[key])


def season_tag(season) -> str:
    """Filename tag for a season selector ('all' → 'all', otherwise the key)."""
    return str(season).strip()


def season_label(season) -> str:
    """Legend label for a season selector."""
    key = str(season).strip()
    return "all months" if key.lower() == "all" else cfg.SEASON_LABELS.get(key, key)


def subset_season(da: xr.DataArray, season) -> xr.DataArray:
    """Cut a [member, time] series down to the months of `season`.

    The joint-distribution counterpart of the season mask applied inside
    `run_compound_frequency_evolution`, so BOTH analyses reduce the record the
    same way before anything else happens: an N-day window that straddles two
    seasons belongs to the season of its CLOSING day (the time stamp the N-day
    operator writes), and every downstream statistic — sample maxima, threshold
    line, exceedance counts — then sees in-season days only.

    Parameters
    ----------
    da     : DataArray with a `time` coordinate (cftime-safe)
    season : "all" (returns `da` unchanged) | key of cfg.SEASON_MONTHS
             ("DJF", "MAM", "JJA", "SON", "MAMJ") — the same selector
             `resolve_season_months` accepts.

    Returns
    -------
    xr.DataArray with only the in-season time steps.
    """
    months = resolve_season_months(season)
    if months is None:
        return da
    keep = np.isin(da["time"].dt.month.values, months)
    if not keep.any():
        raise ValueError(
            f"No time steps left after restricting to season {season!r} "
            f"(months {sorted(months)}) — check JD_SEASON in the joint-distribution "
            "cell / FE_SEASON in the frequency-evolution cell of "
            "compound_flood_risk_analysis.ipynb.")
    return da.isel(time=keep)


# ── Cache availability ────────────────────────────────────────────────────────
def require_compound_series(catchment_slug: str, variables, window_days: int) -> list[Path]:
    """Verify the CESM2-LE compound-series caches exist for every variable.

    Returns the resolved cache paths; raises FileNotFoundError naming the exact
    notebook cell and settings that build a missing selection.
    """
    cache_dir = cfg.POSTPROC_DIR / "cesm2_le" / "catchment_averaged"
    found, missing = [], []
    for var in variables:
        hits = sorted(cache_dir.glob(f"post_processed_cesm2_le_{cfg.acc_tag(window_days)}_"
                                     f"{catchment_slug}_{var}_*.nc"))
        (found.append(hits[-1]) if hits else missing.append(var))
    if missing:
        raise FileNotFoundError(
            f"No {window_days}-day compound-series cache for {missing} "
            f"(catchment '{catchment_slug}') in\n    {cache_dir}\n"
            "→ BUILD IT FIRST in load_data_store_postprocessed.ipynb, cell\n"
            "     '# %% CESM2-LE compound extremes analysis:'\n"
            f"   set   WINDOW_DAYS_COMPOUND = {window_days}\n"
            f"   ensure COMPOUND_SLUGS      contains '{catchment_slug}'\n"
            f"   ensure COMPOUND_VARIABLES  contains {missing}\n"
            "   then re-run that cell and re-run this one. Its prerequisites (the "
            "1-day spatial member caches) are already built for the full record.")
    print(f"[ok] {window_days}-day caches present: {', '.join(p.name for p in found)}")
    return found


# ── Configuration validation ──────────────────────────────────────────────────
def validate_frequency_evolution_config(config: dict) -> dict:
    """Validate + normalise the FE_* selection block; raise with a clear message.

    Adds `season_months`, `season_tag`, `record_start`/`record_end` and the
    resolved cache paths to a copy of the config.
    """
    c = dict(config)
    if c["catchment"] not in cfg.COMPOUND_CATCHMENTS:
        raise ValueError(f"FE_CATCHMENT={c['catchment']!r} unknown; choose one of "
                         f"{list(cfg.COMPOUND_CATCHMENTS)}.")

    combo = tuple(c["combo"])
    if len(combo) != 2 or combo[0] == combo[1] or any(v not in _CESM2_COMPOUND_SPECS for v in combo):
        raise ValueError(f"FE_COMBO={combo!r} must be an ordered pair of two DIFFERENT "
                         f"variables from {sorted(_CESM2_COMPOUND_SPECS)}.")
    c["combo"] = combo

    n = int(c["window_days"])
    if not 1 <= n <= 4:
        raise ValueError(f"FE_WINDOW_DAYS={n} outside the supported range 1–4.")
    if "snowmelt" in combo and n < 2:
        raise ValueError("FE_WINDOW_DAYS must be >= 2 when 'snowmelt' is selected "
                         "(snowmelt = max(0, −(SWE(t) − SWE(t−(N−1))))).")
    c["window_days"] = n

    paths = require_compound_series(c["catchment"], combo, n)
    m = re.search(r"_(\d{4})-(\d{4})\.nc$", paths[0].name)
    rec_s, rec_e = int(m.group(1)), int(m.group(2))
    c["cache_paths"], c["record_start"], c["record_end"] = paths, rec_s, rec_e

    s, e = int(c["start_year"]), int(c["end_year"])
    if s > e or s < rec_s or e > rec_e:
        raise ValueError(f"FE_START/FE_END = {s}–{e} lies outside the stored record "
                         f"{rec_s}–{rec_e} ({paths[0].name}).")
    c["start_year"], c["end_year"] = s, e

    L, step = int(c["roll_years"]), int(c["roll_step"])
    if not 1 <= L <= (e - s + 1):
        raise ValueError(f"FE_ROLL_YEARS={L} must be between 1 and the analysis-period "
                         f"length {e - s + 1} years ({s}–{e}).")
    if step < 1:
        raise ValueError(f"FE_ROLL_STEP={step} must be >= 1.")
    c["roll_years"], c["roll_step"] = L, step

    if float(c["threshold"]) <= 0.0:
        raise ValueError(f"FE_THRESHOLD={c['threshold']} must be > 0 (max possible score = 2.0).")
    c["threshold"] = float(c["threshold"])

    # The blue band is the interquartile range (25-75%). No "std" alias exists any
    # more — the ±1σ band was replaced by the IQR band and "std" is not accepted.
    spread = tuple(str(v).strip() for v in c["spread_show"])
    bad = [v for v in spread if v not in FREQ_SPREAD_KINDS]
    if bad or not len(spread):
        raise ValueError(f"FE_SPREAD_SHOW={tuple(c['spread_show'])!r} invalid; "
                         f"pick any of {list(FREQ_SPREAD_KINDS)}.")
    if "minmax" in spread and "p025p975" in spread:
        raise ValueError("FE_SPREAD_SHOW: pick EITHER 'minmax' OR 'p025p975' — both are "
                         "drawn as the same dashed envelope and would overlap.")
    c["spread_show"] = spread

    # HOW the band/envelope numbers are obtained — the only two options are the
    # empirical (np.percentile) and the grouped (interpolated) percentiles.
    method = str(c.get("spread_method", "percentile_grouped")).strip().lower()
    if method not in FREQ_SPREAD_METHODS:
        raise ValueError(f"FE_SPREAD_METHOD={c.get('spread_method')!r} invalid; "
                         f"pick one of {list(FREQ_SPREAD_METHODS)}.")
    c["spread_method"] = method

    # The rate quantum every grouped percentile is binned on — 1/L, derived from
    # the window length actually used in this run.
    c["rate_quantum"] = 1.0 / float(L)

    lo, hi = (int(v) for v in c["norm_ref"])

    if lo > hi or lo < rec_s or hi > rec_e:
        raise ValueError(f"FE_NORM_REF={lo}–{hi} must lie inside the stored record "
                         f"{rec_s}–{rec_e}.")
    c["norm_ref"] = (lo, hi)


    c["season_months"] = resolve_season_months(c["season"])
    c["season_tag"]    = season_tag(c["season"])
    return c


# ── Loading, frozen normalisation, severity ───────────────────────────────────
def load_compound_pair(catchment_slug: str, combo: tuple, window_days: int
                       ) -> tuple[xr.DataArray, xr.DataArray]:
    """Full-record [member, time] pair for the two FE_COMBO variables, inner-aligned.

    The N-day operator is already baked into the cache (applied to the FULL record
    when it was built), so every rolling window below is complete by construction.
    """
    out = []
    for var in combo:
        da = load_cesm2_le_catchment_field_series(var, window_days, catchment_slug,
                                                  None, None, "all")
        da.load()      # pull the whole [member, time] array into memory and then
        da.close()     # release the NetCDF handle — the cell is re-run often
        out.append(da.transpose("member", "time"))
    return xr.align(out[0], out[1], join="inner")


def freeze_normalisation_maxima(da_x: xr.DataArray, da_y: xr.DataArray,
                                ref_years: tuple, ref_members="all",
                                season_months: list[int] | None = None
                                ) -> tuple[float, float]:
    """The FROZEN denominators (x_max, y_max) — sample maxima of the reference period.

    Computed ONCE over exactly the population the criterion is later applied to:
    the reference YEARS, the reference MEMBER pool and — when a season is
    selected — ONLY the months of that season. Without the season restriction the
    denominators would come from the annual maxima (e.g. an August storm) while
    the exceedances are counted in MAMJ only, which pushes the threshold line far
    too high and biases every window's count low.

    Held constant across every rolling window and both figures: recomputing them
    per window would move the threshold line and make any change partly an
    artefact of a drifting denominator.

    Parameters
    ----------
    da_x, da_y    : [member, time] arrays of the two FE_COMBO variables
    ref_years     : (first, last) calendar years of the frozen reference period
    ref_members   : member pool for the reference sample ("all" | "1-30" | list)
    season_months : month numbers to keep, or None for all months — pass
                    `config["season_months"]` so reference and analysis agree

    Returns
    -------
    (x_max, y_max) : the two frozen denominators
    """
    mem   = parse_member_selection(ref_members, list(da_x["member"].values))
    rx_da = subset_time_series_by_year(da_x.sel(member=mem), *ref_years)
    ry_da = subset_time_series_by_year(da_y.sel(member=mem), *ref_years)
    if season_months is not None:
        keep  = np.isin(rx_da["time"].dt.month.values, season_months)
        rx_da, ry_da = rx_da.isel(time=keep), ry_da.isel(time=keep)
    rx, ry = rx_da.values.ravel(), ry_da.values.ravel()
    ok = np.isfinite(rx) & np.isfinite(ry)
    if not ok.any():
        raise ValueError(
            f"No finite (x, y) pairs in FE_NORM_REF {ref_years[0]}–{ref_years[1]}"
            + ("" if season_months is None
               else f" restricted to months {sorted(season_months)}") + ".")
    x_max, y_max = float(np.max(rx[ok])), float(np.max(ry[ok]))
    if x_max <= 0.0 or y_max <= 0.0:
        raise ValueError(f"Frozen maxima must be > 0 (got x_max={x_max}, y_max={y_max}).")
    return x_max, y_max

# ── Grouped (interpolated) percentiles of a quantised rate ────────────────────
# A member's rate is an INTEGER exceedance count divided by the window length L,
# so it can only take multiples of the rate quantum w = 1/L. np.percentile then
# also lands exactly on a multiple of w: in a window where 30 of 90 members sit at
# 0.1, p25 and p50 are BOTH 0.1 and the IQR band collapses to zero width, and over
# the full record the percentile curves become a staircase that hides the signal.
# `grouped_percentile` treats each attainable value as the CENTRE of a bin and
# interpolates through the tied block instead of returning the block label, which
# restores a continuous curve without touching the counting itself.

def grouped_percentile(values, q, bin_width: float, zero_half_bin: bool = True):
    """Percentiles of a quantised sample, interpolated THROUGH the tied blocks.

    Pure, per-window and testable: one call handles the M member rates of ONE
    rolling window. See helper/test_grouped_percentile.py for the reference case.

    Method
    ------
    With n = len(values) and target = (q/100) · n, find the bin b whose actual
    member counts satisfy  cum_below(b) < target <= cum_upto(b)  and return

        lower_edge(b) + (target − F) / f · width(b)

    where F is the number of members in bins strictly BELOW b and f the number of
    members IN b. F and f come from the real counts (`np.unique`), so bins that
    happen to be empty are simply absent and are skipped correctly.

    Bin edges
    ---------
    Centred bins EXCEPT at zero, which is a HALF-width bin starting at 0:

        value 0      → [0, w/2)                width w/2
        value k·w    → [k·w − w/2, k·w + w/2)  width w,  k >= 1

    A rate cannot be negative; with a naive symmetric bin at zero the low
    percentiles would come out negative. `zero_half_bin=False` restores that naive
    behaviour and exists only to demonstrate the difference.

    Parameters
    ----------
    values : 1-D array-like
        The M member rates of ONE window, in any order. NaNs are dropped.
    q : float or array-like
        Percentile(s) in [0, 100].
    bin_width : float
        The rate quantum w. Pass 1 / roll_years — NEVER a hardcoded 0.1, so a
        change of L propagates automatically.
    zero_half_bin : bool
        Keep the zero bin at half width (default, correct for a non-negative rate).

    Returns
    -------
    float if `q` is scalar, else np.ndarray with the shape of `q`.
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("grouped_percentile: no finite values in this window.")
    w = float(bin_width)
    if not np.isfinite(w) or w <= 0.0:
        raise ValueError(f"grouped_percentile: bin_width must be > 0 (got {bin_width!r}).")

    q_arr = np.atleast_1d(np.asarray(q, dtype=float))
    if np.any(q_arr < 0.0) or np.any(q_arr > 100.0):
        raise ValueError(f"grouped_percentile: q must lie in [0, 100] (got {q!r}).")

    # Bin index of every member = value / w, which must be an integer by construction.
    k = np.rint(v / w).astype(np.int64)
    off = np.abs(v / w - k)
    if np.any(off > 1e-6):
        raise ValueError(
            f"grouped_percentile: value {v[int(np.argmax(off))]!r} is not a multiple "
            f"of bin_width={w!r}. The rate must be an integer count / L — pass "
            "bin_width = 1 / roll_years.")
    if np.any(k < 0):
        raise ValueError("grouped_percentile: negative rates are not supported.")

    # np.unique returns ONLY the occupied levels, ascending — empty bins never
    # enter the cumulative counts and are therefore skipped by construction.
    levels, counts = np.unique(k, return_counts=True)
    lower = levels * w - 0.5 * w
    if zero_half_bin and levels[0] == 0:
        lower[0] = 0.0                      # a rate cannot be negative
    width  = (levels * w + 0.5 * w) - lower
    cum    = np.cumsum(counts)              # cum_upto(b)
    F      = cum - counts                   # cum_below(b), from ACTUAL counts

    target = (q_arr / 100.0) * v.size
    b = np.clip(np.searchsorted(cum, target, side="left"), 0, levels.size - 1)
    out = lower[b] + (target - F[b]) / counts[b] * width[b]
    out = np.where(target <= 0.0, lower[0], out)     # q = 0 → lowest bin edge
    return float(out[0]) if np.ndim(q) == 0 else out.reshape(np.shape(q))


# ── Annual counts → rolling windows → ensemble statistics ─────────────────────
def annual_exceedance_counts(candidate: np.ndarray, exceed: np.ndarray,
                             years_of_time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse the [member, time] masks to per-calendar-year counts [member, year].

    Window membership is defined by CALENDAR YEAR (cftime/no-leap safe), not by
    fixed day counts, so a season selection simply leaves the non-season days at
    zero. `candidate` is kept in the signature as the documented denominator of
    the counts even though only `exceed` is summed.
    """
    years = np.unique(years_of_time)
    col   = np.searchsorted(years, years_of_time)
    k_ary = np.zeros((exceed.shape[0], years.size), dtype=np.int64)
    for m in range(exceed.shape[0]):
        k_ary[m] = np.bincount(col, weights=exceed[m], minlength=years.size).astype(np.int64)
    return years, k_ary


def rolling_window_counts(years: np.ndarray, k_ary: np.ndarray, roll_years: int,
                          roll_step: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exceedance counts per member in every COMPLETE centred rolling window.

    Returns (window_starts [W], window_centres [W], counts [member, W]).
    Window W = [y0, y0+L−1] is plotted at y0 + (L−1)/2 — half-years are kept, never
    rounded. Only complete windows are produced, so the curve is shorter than the
    analysis period by (L−1)/2 at each end and is NOT padded.
    """
    L, Y = int(roll_years), years.size
    zero  = np.zeros((k_ary.shape[0], 1), dtype=np.int64)
    k_cum = np.concatenate([zero, np.cumsum(k_ary, axis=1)], axis=1)
    i0    = np.arange(0, Y - L + 1, int(roll_step))
    return (years[i0], years[i0] + (L - 1) / 2.0, k_cum[:, i0 + L] - k_cum[:, i0])


# Percentiles computed for every window, in BOTH the empirical and the grouped
# flavour — one list, so the two versions can never drift apart.
FREQ_PERCENTILES = (2.5, 25.0, 75.0, 97.5)


def ensemble_frequency_statistics(starts: np.ndarray, centres: np.ndarray,
                                  counts: np.ndarray, *, roll_years: int
                                  ) -> pd.DataFrame:
    """Per-window ensemble statistics of the compound-extreme rate.

    The rate unit is `cfg.rate_unit_label(season_months)` — 'events per year' for
    an all-months run, 'events per season' for any month subset. The VALUE is the
    same either way (each year contributes exactly one season); only the label
    differs. It is never converted to a per-decade rate, so it stays comparable
    when the window length L changes.

    Per member the rate is K_m(W)/L. No median is computed or drawn anywhere in
    this analysis. `f_mean` is the ensemble MEAN across members
    — the central line of Figure 1 and the numerator of the unitless
    signal-to-noise ratio of Figure 2 (S/N = f_mean / sigma). Its resolution is
    1/(M·L), so it is smooth where every rank statistic is not. `sigma` is the
    sample standard deviation ACROSS MEMBERS — the internal-variability measure
    and the denominator of the S/N ratio. Neither is touched by the grouped
    percentiles added below.

    Percentile columns come in two flavours:
      `p025`/`p25`/`p75`/`p975`           — np.percentile, kept selectable as the
        reference version; because a member rate is an integer count / L these
        land exactly on multiples of the rate quantum 1/L and form a staircase;
      `p025_grouped`/`p25_grouped`/`p75_grouped`/`p975_grouped`
        — `grouped_percentile`, which interpolates THROUGH the tied blocks and
        removes that staircase.

    The rate quantum handed to `grouped_percentile` is derived as 1/roll_years —
    never hardcoded — and every member rate is asserted to be an integer multiple
    of it, so a rate produced some other way fails loudly instead of being binned
    wrongly.

    Returns
    -------
    pd.DataFrame with one row per rolling window: window_start/end/centre,
    n_members, n_events_total, mean_events_per_member, f_mean, sigma, min, max,
    rate_quantum, p25, p75, p025, p975, p025_grouped, p25_grouped, p75_grouped,
    p975_grouped, signal_to_noise.
    """
    L     = float(roll_years)
    rates = counts / L            # [member, window], see the unit note above
    # CHANGE 3: the rate quantum follows the window length, never a literal 0.1.
    bin_width = 1.0 / L

    # The grouped percentiles are only meaningful if every member rate really is
    # an integer count / L — fail loudly rather than bin a differently produced rate.
    scaled = rates * L
    dev = float(np.max(np.abs(scaled - np.rint(scaled)))) if scaled.size else 0.0
    if dev > 1e-9:
        raise ValueError(
            f"ensemble_frequency_statistics: member rates are not multiples of the "
            f"rate quantum 1/roll_years = {bin_width:.6g} (worst deviation {dev:.3g} "
            "counts). The rate must stay an INTEGER exceedance count divided by the "
            "window length L — check annual_exceedance_counts / rolling_window_counts.")

    out = pd.DataFrame({
        "window_start":  starts.astype(int),
        "window_end":    starts.astype(int) + roll_years - 1,
        "window_centre": centres,
        "n_members":     counts.shape[0],
        "n_events_total": counts.sum(axis=0),
        "mean_events_per_member": counts.mean(axis=0),
        "f_mean":   rates.mean(axis=0),
        "sigma":    rates.std(axis=0, ddof=1),
        "min":      rates.min(axis=0),
        "max":      rates.max(axis=0),
        "rate_quantum": bin_width,
        # Empirical (np.percentile) percentiles — kept for comparison.
        "p25":    np.percentile(rates, 25.0, axis=0),
        "p75":    np.percentile(rates, 75.0, axis=0),
        "p025":   np.percentile(rates,  2.5, axis=0),
        "p975":   np.percentile(rates, 97.5, axis=0)})

    # Grouped percentiles, one pure call per window (W ≈ 10^2 — no need to vectorise).
    q_grp = np.asarray(FREQ_PERCENTILES, dtype=float)
    grouped = np.column_stack([grouped_percentile(rates[:, w], q_grp, bin_width)
                               for w in range(rates.shape[1])])          # [4, W]
    for name, row in zip(("p025_grouped", "p25_grouped",
                          "p75_grouped", "p975_grouped"), grouped):
        out[name] = row

    # Signal-to-noise ratio of Figure 2: ensemble MEAN over the across-member
    # standard deviation, unitless — the SAME central quantity Figure 1 draws.
    # Windows where sigma == 0 (every member has the same count) give inf/NaN and
    # are left as such rather than masked.
    with np.errstate(divide="ignore", invalid="ignore"):
        out["signal_to_noise"] = out["f_mean"] / out["sigma"]
    return out


# ── Orchestration ─────────────────────────────────────────────────────────────
def run_compound_frequency_evolution(config: dict) -> dict:
    """Full frequency-evolution analysis for one FE_* selection.

    load pair → freeze maxima (reference years + reference members + SEASON) →
    severity → season/candidate mask → exceedances → annual counts → rolling
    windows → ensemble statistics (mean, σ, envelopes, S/N = mean/σ).
    Returns everything the two figures, the CSV and the JSON need.
    """
    c = validate_frequency_evolution_config(config)
    x_var, y_var = c["combo"]

    da_x, da_y = load_compound_pair(c["catchment"], c["combo"], c["window_days"])
    # The frozen line comes from EXACTLY the population it is later applied to:
    # reference years, reference members and — when set — the season months only.
    x_max, y_max = freeze_normalisation_maxima(
        da_x, da_y, c["norm_ref"], c["norm_ref_members"], c["season_months"])
    n_ref_members = len(parse_member_selection(c["norm_ref_members"],
                                               list(da_x["member"].values)))

    members = parse_member_selection(c["members"], list(da_x["member"].values))
    ax = subset_time_series_by_year(da_x.sel(member=members), c["start_year"], c["end_year"])
    ay = subset_time_series_by_year(da_y.sel(member=members), c["start_year"], c["end_year"])
    xv, yv = ax.values, ay.values

    candidate = np.isfinite(xv) & np.isfinite(yv)
    if c["season_months"] is not None:
        # A window spanning two seasons belongs to the season of its CLOSING day —
        # the convention already used by the seasonal map pipeline.
        candidate &= np.isin(ax["time"].dt.month.values, c["season_months"])[None, :]
    exceed = candidate & ((xv / x_max + yv / y_max) >= c["threshold"])

    years, k_ary = annual_exceedance_counts(candidate, exceed, ax["time"].dt.year.values)
    starts, centres, counts = rolling_window_counts(
        years, k_ary, c["roll_years"], c["roll_step"])
    ens = ensemble_frequency_statistics(
        starts, centres, counts, roll_years=c["roll_years"])

    diag = {
        "n_members":         len(members),
        "n_ref_members":     n_ref_members,
        "n_candidate_days":  int(candidate.sum()),
        "n_exceedance_days": int(exceed.sum()),
        "x_max": x_max, "y_max": y_max,
        "physical_threshold": (f"{x_var}/{x_max:.4g} + {y_var}/{y_max:.4g} "
                               f">= {c['threshold']:g}"),
        "threshold_intercepts": (c["threshold"] * x_max, c["threshold"] * y_max),
        "mean_sigma": float(ens["sigma"].mean()),
        "mean_events_per_member_per_window": float(ens["mean_events_per_member"].mean()),
    }
    return {"config": c, "ensemble": ens, "diagnostics": diag}


def print_frequency_evolution_summary(result: dict) -> None:
    """Console summary — frozen maxima, physical threshold, counts, σ and S/N."""
    c, d, e = result["config"], result["diagnostics"], result["ensemble"]
    x_var, y_var = c["combo"]
    print("=" * 78)
    print(f"COMPOUND FREQUENCY EVOLUTION — {x_var} + {y_var} | {c['catchment']} | "
          f"{c['window_days']}-day | {c['start_year']}–{c['end_year']} | "
          f"season {c['season_tag']}")
    print("=" * 78)
    print(f"Frozen maxima (ref {c['norm_ref'][0]}–{c['norm_ref'][1]}, season "
          f"{c['season_tag']}, {d['n_ref_members']} members): "
          f"max {x_var} = {d['x_max']:.3f}, max {y_var} = {d['y_max']:.3f}")
    print(f"Physical threshold: {d['physical_threshold']}  → intercepts "
          f"({d['threshold_intercepts'][0]:.2f}, 0) and (0, {d['threshold_intercepts'][1]:.2f})")
    print(f"Candidate days: {d['n_candidate_days']:,} | exceedance days (= events): "
          f"{d['n_exceedance_days']:,}")
    print(f"Windows: {len(e)} complete {c['roll_years']}-year windows, centres "
          f"{e['window_centre'].iloc[0]:g}–{e['window_centre'].iloc[-1]:g} "
          f"(step {c['roll_step']} yr)")
    # ONE season-aware unit string for every number printed below.
    unit = cfg.rate_unit_label(c["season_months"])
    col  = frequency_spread_columns(c["spread_method"])
    print(f"First window {e['window_centre'].iloc[0]:g}: {e['f_mean'].iloc[0]:.4g} {unit}"
          f"   |   last window {e['window_centre'].iloc[-1]:g}: "
          f"{e['f_mean'].iloc[-1]:.4g} {unit}   (ensemble MEAN — the line of Figure 1)")
    print(f"Mean σ across members: {d['mean_sigma']:.4g} {unit}")
    print(f"IQR averaged over windows ({c['spread_method']}): "
          f"p25 = {e[col['band_lo']].mean():.4g} … p75 = {e[col['band_hi']].mean():.4g} "
          f"{unit}   (the blue band of Figure 1)")
    # Granularity diagnostic: the empirical p25 can only land on multiples of the
    # rate quantum 1/L, the grouped one interpolates through the tied blocks.
    print(f"Rate quantum 1/L = {1.0 / c['roll_years']:.6g} {unit}  |  distinct p25 values "
          f"over {len(e)} windows: empirical {e['p25'].nunique()}, "
          f"grouped {e['p25_grouped'].nunique()}")
    print(f"S/N (ensemble mean / σ): range {e['signal_to_noise'].min():.2f} … "
          f"{e['signal_to_noise'].max():.2f}   |   first window "
          f"{e['signal_to_noise'].iloc[0]:.2f} → last window "
          f"{e['signal_to_noise'].iloc[-1]:.2f}")


    if d["mean_events_per_member_per_window"] < 5.0:
        print(f"[warning] only {d['mean_events_per_member_per_window']:.1f} events per member "
              f"per {c['roll_years']}-year window — σ across members is dominated by counting "
              "noise. Lengthen FE_ROLL_YEARS before quoting σ quantitatively.")
    print("=" * 78)


def write_frequency_evolution_outputs(result: dict, stem: str, out_paths_fn) -> None:
    """Write the ensemble CSV and a metadata JSON to every output root.

    `stem`         : the figure stem, so the tables carry the same name
    `out_paths_fn` : filename → list of paths (pass the notebook's `fe_figp`), so
                     the tables land next to the PDFs.
    """
    c = result["config"]
    meta = {k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()
            if k != "cache_paths"}
    meta.update({"cache_paths": [str(p) for p in c["cache_paths"]],
                 "diagnostics": {k: (list(v) if isinstance(v, tuple) else v)
                                 for k, v in result["diagnostics"].items()},
                 "package_versions": {"numpy": np.__version__, "pandas": pd.__version__,
                                      "xarray": xr.__version__},
                 "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})

    for fname, payload in ((f"{stem}_ensemble.csv", result["ensemble"]),
                           (f"{stem}_metadata.json", meta)):
        for p in out_paths_fn(fname):
            p.parent.mkdir(parents=True, exist_ok=True)
            (payload.to_csv(p, index=False) if fname.endswith(".csv")
             else p.write_text(json.dumps(payload, indent=2, default=str)))
            print(f"Saved → {p}")


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

# Define Annual Max. Loader for Reanalysis Models
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
    from config_paths import MODEL_ORDER, MODEL_LABELS

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


# ── Public GeoJSON loader (shared by all notebooks) ───────────────────────────

def load_catchments(geojson_files: dict, geojson_dir: Path = None) -> dict:
    """
    Load catchment polygons from GeoJSON files into GeoDataFrames (EPSG:4326).

    Parameters
    ----------
    geojson_files : dict  {slug: filename_string}
    geojson_dir   : Path to the directory containing the GeoJSON files.
                    Defaults to cfg.GEOJSON_DIR if None.

    Returns
    -------
    dict  {slug: gpd.GeoDataFrame}  — all in EPSG:4326.
    """
    import geopandas as gpd
    if geojson_dir is None:
        geojson_dir = cfg.GEOJSON_DIR
    out: dict = {}
    for slug, filename in geojson_files.items():
        path = geojson_dir / filename
        gdf = gpd.read_file(str(path))
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326, allow_override=True)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        out[slug] = gdf
    return out

# data_era5.py
"""
ERA5 file discovery, loading, unit conversion, and optional full-grid caching.
"""

import re
import xarray as xr
import numpy as np
from pathlib import Path
import pandas as pd


def find_era5_files(era5_dir: Path, resolution: str) -> list[Path]:
    """
    Return a chronologically sorted list of ERA5 annual NetCDF files
    for the requested resolution.

    Expected filename pattern:  tp24_<resolution>_<year>.nc
    Example:                    tp24_0.5x0.5_1941.nc
    """
    pattern = re.compile(rf"^tp24_{re.escape(resolution)}_(\d{{4}})\.nc$")
    matched = []
    for f in era5_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            matched.append((int(m.group(1)), f))

    if not matched:
        raise FileNotFoundError(
            f"No ERA5 files found in:\n  {era5_dir}\n"
            f"for resolution '{resolution}'.\n"
            f"Expected filenames like: tp24_{resolution}_1941.nc"
        )

    matched.sort(key=lambda x: x[0])
    return [f for _, f in matched]


def load_era5_precipitation(era5_files: list[Path]) -> xr.DataArray:
    """
    - Converts precipitation from meters to mm
    - Verifies that the time axis is increasing

    Returns
    -------
    xr.DataArray
        Dimensions: (time, latitude, longitude), units: mm
        Values are loaded via Dask
    """
    print(f"  Opening {len(era5_files)} ERA5 files (lazy) ...")
    ds = xr.open_mfdataset([str(f) for f in era5_files], combine="by_coords", coords="minimal", compat="override", chunks={"time": 365},)

    # If an unnecessary singleton ensemble/member dimension exists, remove it
    for _dim in ("number", "ensemble", "ens"):
        if _dim in ds.dims and ds.sizes[_dim] == 1:
            ds = ds.isel({_dim: 0}, drop=True)
        elif _dim in ds.coords:
            ds = ds.drop_vars(_dim)

    da = ds["tp24"] * 1000.0    # meters → millimeters
    da.attrs["units"] = "mm"
    da.name = "tp24_mm"

    # Validate time axis (check before any computation)
    times = da["time"].values
    if not np.all(np.diff(times.astype("int64")) > 0):
        raise ValueError(
            "ERA5 time axis is not strictly increasing after concatenation.\n"
            "Check for duplicate or overlapping annual files.")
    return da


def get_year_range(era5_files: list[Path], resolution: str) -> tuple[int, int]:
    """
    Extract start_year and end_year from the sorted list of ERA5 files
    Relies on the filename pattern tp24_<resolution>_<year>.nc
    """
    pattern = re.compile(rf"^tp24_{re.escape(resolution)}_(\d{{4}})\.nc$")
    years = [int(pattern.match(f.name).group(1)) for f in era5_files]
    return min(years), max(years)


# ── Time-selection utilities (used by map notebook and evaluation) ─────────────
def day_start(dt) -> "pd.Timestamp":
    """Normalise any timestamp to midnight (00:00:00) of the same calendar day."""
    import pandas as pd
    return pd.Timestamp(dt).normalize()


def select_time_range_by_day(da: "xr.DataArray",
                              start_date: str,
                              end_date: str) -> "xr.DataArray":
    """
    Select a calendar-day range from a DataArray, regardless of stored sub-daily
    timestamps. start_date and end_date are inclusive (format: 'YYYY-MM-DD').
    """
    import pandas as pd
    t0 = pd.Timestamp(start_date)
    t1 = pd.Timestamp(end_date) + pd.Timedelta("1D") - pd.Timedelta("1ns")
    return da.sel(time=slice(t0, t1))


def select_single_time_by_day(
    da: "xr.DataArray",
    target_day,
) -> "xr.DataArray":
    """
    Select exactly one time slice for a given calendar day, regardless of
    stored hour/minute/second. Raises KeyError if no match, ValueError if
    more than one match.
    target_day: str 'YYYY-MM-DD' or pd.Timestamp.
    """
    import pandas as pd
    import numpy as np
    target_day = day_start(target_day)
    time_days = pd.to_datetime(da["time"].dt.floor("D").values)
    idx = np.where(time_days == target_day)[0]
    if len(idx) == 0:
        available_times = pd.to_datetime(da["time"].values)
        raise KeyError(
            f"No time step found for calendar day {target_day.date()}.\n"
            f"Available range: {available_times[0].date()} – {available_times[-1].date()}"
        )
    if len(idx) > 1:
        raise ValueError(
            f"More than one time step found for calendar day {target_day.date()}:\n"
            f"{pd.to_datetime(da['time'].values[idx])}\n"
            "Expected only one daily field per day."
        )
    return da.isel(time=int(idx[0]), drop=True)


#Define a helper function to create a slice object that respects the coordinate order (ascending or descending)
def coord_slice(coord: "xr.DataArray", lower: float, upper: float) -> slice:
    """
    Return a slice that respects ascending or descending coordinate order.
    Handles ERA5 latitude which is stored top-to-bottom (descending).
    """
    values = coord.values
    if values[0] <= values[-1]:
        return slice(lower, upper)
    return slice(upper, lower)


def pick_year_file(files: list[Path], year: int) -> Path:
    """
    Pick one annual file from a sorted list based on a 4-digit year in the filename.
    Matches patterns like: tp24_0.5x0.5_2023.nc  or  rr_2023.nc
    """
    year_pattern = re.compile(rf"(^|_){year}\.nc$")
    for f in files:
        if year_pattern.search(f.name):
            return f
    raise FileNotFoundError(
        f"Could not find annual file for year {year} in file list.\n"
        f"Searched {len(files)} files."
    )

# ── Overall-precipitation spatial-cache builders ───────────────────────────────

def save_era5_overall(
    resolution: str,
    era5_dir: Path,
    out_path_fn,
    extent: tuple,
    force: bool = False,
) -> None:
    """
    Build and save the spatially-cropped daily ERA5 precipitation cache.

    Parameters
    ----------
    out_path_fn : callable(dataset, resolution, start_year, end_year) -> Path
                  e.g. cfg.overall_precip_path
    extent      : (west, east, south, north) in degrees
    """
    from catchment_tools import save_spatial_netcdf
    files = find_era5_files(era5_dir, resolution)
    start_year, end_year = get_year_range(files, resolution)
    out_path = out_path_fn("era5", resolution, start_year, end_year)

    if (not force) and out_path.exists():
        print(f"  [skip] ERA5 {resolution} cache exists: {out_path.name}")
        return

    print(f"  Building ERA5 {resolution} DAILY cache ({start_year}–{end_year}) ...")
    west, east, south, north = extent
    buf = 0.6

    ds = xr.open_mfdataset(
        [str(f) for f in files], combine="by_coords", coords="minimal",
        compat="override", chunks={"time": 365, "latitude": 64, "longitude": 64})
    try:
        for _dim in ("number", "ensemble", "ens"):
            if _dim in ds.dims and ds.sizes[_dim] == 1:
                ds = ds.isel({_dim: 0}, drop=True)
            elif _dim in ds.coords:
                ds = ds.drop_vars(_dim)
        da = (ds["tp24"] * 1000.0).sel(
            longitude=slice(west - buf, east + buf),
            latitude=coord_slice(ds["latitude"], south - buf, north + buf))
        save_spatial_netcdf(da, out_path, "tp24_mm",
                            time_chunk=365, y_chunk=64, x_chunk=64)
    finally:
        ds.close()


def compute_era5_annual_median_2d(
    resolution: str,
    start_year: int,
    end_year: int,
    era5_dir: Path,
    cache_path_fn,
    open_cache_fn,
) -> "xr.DataArray":
    """Annual median daily precipitation (mm/year) from ERA5 overall cache."""
    files = find_era5_files(era5_dir, resolution)
    avail_s, avail_e = get_year_range(files, resolution)
    cache = cache_path_fn("era5", resolution, avail_s, avail_e)
    if not cache.exists():
        raise FileNotFoundError(
            f"ERA5 {resolution} overall cache not found.\n{cache}")
    print(f"  Loading ERA5 {resolution} from cache ({start_year}–{end_year}) ...")
    da = open_cache_fn(cache, start_year, end_year)
    out = da.groupby("time.year").sum(dim="time").median(dim="year")
    out.name = "annmedian_precip"
    out.attrs.update({"units": "mm/year",
                      "start_year": start_year, "end_year": end_year})
    return out.compute()


def compute_era5_2day_median_2d(
    resolution: str,
    start_year: int,
    end_year: int,
    era5_dir: Path,
    cache_path_fn,
    open_cache_fn,
    rolling_fn,
    subset_fn,
    window_days: int = 2,
) -> "xr.DataArray":
    """
    Median of 2-day rolling sums (mm) from ERA5 overall 1-day cache.
    rolling_fn : catchment_tools.rolling_accumulation
    subset_fn  : catchment_tools.subset_time_series_by_year
    """
    files = find_era5_files(era5_dir, resolution)
    avail_s, avail_e = get_year_range(files, resolution)
    cache = cache_path_fn("era5", resolution, avail_s, avail_e)
    if not cache.exists():
        raise FileNotFoundError(f"ERA5 {resolution} overall cache not found.\n{cache}")
    print(f"  Loading ERA5 {resolution} (2-day) from cache ({start_year}–{end_year}) ...")
    da = open_cache_fn(cache, start_year - 1, end_year)
    da_2day = rolling_fn(da, window_days=window_days)
    return subset_fn(da_2day, start_year, end_year).median(dim="time").compute()



# ── ERA5 interpolated-to-CESM2-LE grid spatial-cache builders ─────────────────
# Files: tp_era5_interpolated_to_cesm2-le_<year>.nc
# Variable: verify with ncdump -h, assumed to be "tp" in metres → multiply by 1000 for mm
# Coordinates: lat/lon (CESM2-LE grid, NOT latitude/longitude like standard ERA5)

def find_era5_interpolated_files(era5_interp_dir: Path) -> list[Path]:
    """
    Sorted list of ERA5-interpolated (cesm2-le grid) annual NetCDF files in a
    single-variable directory. The variable prefix is matched generically
    (tp / sd / swvl), so one function serves precip, SWE and soil moisture.
    Pattern: <var>_era5_interpolated_to_cesm2-le_<year>.nc
    """
    pattern = re.compile(r"^[A-Za-z0-9]+_era5_interpolated_to_cesm2-le_(\d{4})\.nc$")
    matched = []
    for f in era5_interp_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            matched.append((int(m.group(1)), f))
    if not matched:
        raise FileNotFoundError(
            f"No ERA5-interpolated files found in:\n  {era5_interp_dir}\n"
            f"Expected: <var>_era5_interpolated_to_cesm2-le_<year>.nc"
        )
    matched.sort(key=lambda x: x[0])
    return [f for _, f in matched]



def get_year_range_era5_interp(files: list[Path]) -> tuple[int, int]:
    """Year range from a sorted ERA5-interpolated file list (any variable prefix)."""
    pattern = re.compile(r"^[A-Za-z0-9]+_era5_interpolated_to_cesm2-le_(\d{4})\.nc$")
    years = [int(pattern.match(f.name).group(1)) for f in files]
    return min(years), max(years)



def save_era5_interpolated_overall(
    era5_interp_dir: Path,
    out_path_fn,
    extent: tuple,
    force: bool = False,
) -> None:
    """
    Build and save the spatially-cropped daily ERA5-interpolated precipitation cache.

    IMPORTANT: verify the variable name and units in the raw files with
        ncdump -h <one_file>.nc
    before running. This function assumes variable="tp", units=metres.
    Coordinates are assumed to be lat/lon (CESM2-LE grid).
    """
    from catchment_tools import save_spatial_netcdf
    files = find_era5_interpolated_files(era5_interp_dir)
    start_year, end_year = get_year_range_era5_interp(files)
    out_path = out_path_fn("era5_interpolated", "", start_year, end_year)

    if (not force) and out_path.exists():
        print(f"  [skip] ERA5-interpolated cache exists: {out_path.name}")
        return

    print(f"  Building ERA5-interpolated DAILY cache ({start_year}–{end_year}) ...")
    west, east, south, north = extent
    buf = 0.6

    ds = xr.open_mfdataset(
        [str(f) for f in files], combine="by_coords", coords="minimal",
        compat="override", chunks={"time": 365, "lat": 64, "lon": 64})
    try:
        # Drop singleton dimensions if any
        for _dim in ("number", "ensemble", "ens", "member"):
            if _dim in ds.dims and ds.sizes[_dim] == 1:
                ds = ds.isel({_dim: 0}, drop=True)
            elif _dim in ds.coords:
                ds = ds.drop_vars(_dim)
        # Variable name: verify! Assumed "tp" in metres → mm
        var = "tp" if "tp" in ds else list(ds.data_vars)[0]
        units_raw = str(ds[var].attrs.get("units", "")).strip()
        u = units_raw.lower().replace(" ", "")
        if any(tok in u for tok in ["kgm-2", "kg/m2", "kgm^-2", "mm", "millimetre", "millimeter"]):
            factor = 1.0
        elif u in {"m", "m/day", "md-1", "mday-1", "metre", "meter"} or u.startswith("m/"):
            factor = 1000.0
        else:
            factor = 1000.0
            print(f"  WARNING: ERA5-interp variable '{var}' has unrecognized units={units_raw!r}; "
                  f"defaulting to ×1000 (assuming metres→mm). Verify with ncdump on a raw file.")
        print(f"  ERA5-interp '{var}': units={units_raw!r}, conversion factor={factor}")
        da = (ds[var] * factor).sel(
            lon=slice(west - buf, east + buf),
            lat=coord_slice(ds["lat"], south - buf, north + buf))
        da.attrs["units"] = "mm"
        da.name = "tp24_mm"
        save_spatial_netcdf(da, out_path, "tp24_mm",
                            time_chunk=365, y_chunk=64, x_chunk=64)
    finally:
        ds.close()


def save_era5_interpolated_field_overall(
    era5_interp_dir: Path,
    out_path_fn,            # callable(start_year, end_year) -> Path
    variable: str,          # raw variable in the files, e.g. 'sd' or 'swvl'
    cache_var: str,         # stored cache name, e.g. 'swe' or 'soil_moisture'
    units: str = "kg/m2",
    extent: tuple = None,
    force: bool = False,
) -> None:
    """
    Build/save the ERA5-interpolated daily state-field cache (SWE 'sd' /
    soil moisture 'swvl'). Generic counterpart of `save_era5_interpolated_overall`
    with no metres→mm conversion (values are already kg/m²).
    """
    from catchment_tools import save_spatial_netcdf
    files = find_era5_interpolated_files(era5_interp_dir)
    start_year, end_year = get_year_range_era5_interp(files)
    out_path = out_path_fn(start_year, end_year)
    if (not force) and out_path.exists():
        print(f"  [skip] ERA5-interp {cache_var} cache exists: {out_path.name}")
        return
    print(f"  Building ERA5-interpolated {cache_var} DAILY cache ({start_year}–{end_year}) ...")
    ds = xr.open_mfdataset(
        [str(f) for f in files], combine="by_coords", coords="minimal",
        compat="override", chunks={"time": 365, "lat": 64, "lon": 64})
    try:
        for _dim in ("number", "ensemble", "ens", "member"):
            if _dim in ds.dims and ds.sizes[_dim] == 1:
                ds = ds.isel({_dim: 0}, drop=True)
        if variable not in ds:
            raise KeyError(
                f"Variable {variable!r} not in ERA5-interp files; found {list(ds.data_vars)}")
        da = ds[variable]
        if extent is not None:
            west, east, south, north = extent
            buf = 0.6
            da = da.sel(lon=slice(west - buf, east + buf),
                        lat=coord_slice(ds["lat"], south - buf, north + buf))
        da.attrs["units"] = units
        da.name = cache_var
        save_spatial_netcdf(da, out_path, cache_var,
                            time_chunk=365, y_chunk=64, x_chunk=64, units=units)
    finally:
        ds.close()


def compute_era5_interpolated_2day_median_2d(
    start_year: int,
    end_year: int,
    era5_interp_dir: Path,
    cache_path_fn,
    open_cache_fn,
    rolling_fn,
    subset_fn,
    window_days: int = 2,
) -> "xr.DataArray":
    """Median of 2-day rolling sums (mm) from ERA5-interpolated overall 1-day cache."""
    files = find_era5_interpolated_files(era5_interp_dir)
    avail_s, avail_e = get_year_range_era5_interp(files)
    cache = cache_path_fn("era5_interpolated", "", avail_s, avail_e)
    if not cache.exists():
        raise FileNotFoundError(
            f"ERA5-interpolated overall cache not found.\n{cache}")
    print(f"  Loading ERA5-interpolated (2-day median) from cache ({start_year}–{end_year}) ...")
    da = open_cache_fn(cache, start_year - 1, end_year)
    da_2day = rolling_fn(da, window_days=window_days)
    return subset_fn(da_2day, start_year, end_year).median(dim="time").compute()


def compute_era5_interpolated_2day_p90_2d(
    start_year: int,
    end_year: int,
    era5_interp_dir: Path,
    p90_cache_path: "Path",
    cache_path_fn,
    open_cache_fn,
    rolling_fn,
    subset_fn,
    window_days: int = 2,
    force_recompute: bool = False,
) -> "xr.DataArray":
    """90th percentile of 2-day rolling sums (mm) from ERA5-interpolated cache."""
    if (not force_recompute) and p90_cache_path.exists():
        print(f"  [cache] ERA5-interpolated 2-day p90 ← {p90_cache_path.name}")
        with xr.open_dataset(str(p90_cache_path)) as ds:
            var = "twoday_p90_precip" if "twoday_p90_precip" in ds else "twodayp90_precip"
            return ds[var].load()


    files = find_era5_interpolated_files(era5_interp_dir)
    avail_s, avail_e = get_year_range_era5_interp(files)
    cache = cache_path_fn("era5_interpolated", "", avail_s, avail_e)
    if not cache.exists():
        raise FileNotFoundError(
            f"ERA5-interpolated overall cache not found.\n{cache}")
    print(f"  Computing ERA5-interpolated 2-day p90 ({start_year}–{end_year}) ...")
    da = open_cache_fn(cache, start_year - 1, end_year)
    da_2day = rolling_fn(da, window_days=window_days)
    out = subset_fn(da_2day, start_year, end_year).quantile(0.90, dim="time").compute()
    out = out.drop_vars("quantile", errors="ignore")
    out.name = "twoday_p90_precip"
    out.attrs.update({"units": "mm", "start_year": start_year, "end_year": end_year})
    p90_cache_path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"twoday_p90_precip": out}).to_netcdf(
        str(p90_cache_path),
        encoding={"twoday_p90_precip": {"zlib": True, "complevel": 4}})
    print(f"  [saved] {p90_cache_path.name}")
    return out

# ── Seasonal filtering helpers ─────────────────────────────────────────────────
_SEASON_MONTHS: dict[str, list[int]] = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}


def _filter_season_da(da: "xr.DataArray", season: str) -> "xr.DataArray":
    """Select only time steps whose month belongs to the given season abbreviation."""
    months = _SEASON_MONTHS[season.upper()]
    return da.isel(time=da["time"].dt.month.isin(months))


def compute_era5_interpolated_2day_seasonal_median_2d(
    season: str,
    start_year: int,
    end_year: int,
    era5_interp_dir: "Path",
    median_cache_path: "Path",
    cache_path_fn,
    open_cache_fn,
    rolling_fn,
    subset_fn,
    window_days: int = 2,
    force_recompute: bool = False,
) -> "xr.DataArray":
    """
    Seasonal median of 2-day rolling sums (mm) from ERA5-interpolated cache.

    Rolling sums are computed over the full annual period; only timesteps whose
    month belongs to `season` are retained before taking the pixel-wise median.
    Result: median over ~90×30 values per pixel (days-in-season × years).
    """
    if (not force_recompute) and median_cache_path.exists():
        print(f"  [cache] ERA5-interp seasonal {season} 2-day median ← {median_cache_path.name}")
        with xr.open_dataset(str(median_cache_path)) as _ds:
            return _ds["twoday_seasonal_median"].load()

    files = find_era5_interpolated_files(era5_interp_dir)
    avail_s, avail_e = get_year_range_era5_interp(files)
    cache = cache_path_fn("era5_interpolated", "", avail_s, avail_e)
    if not cache.exists():
        raise FileNotFoundError(
            f"ERA5-interpolated overall cache not found.\n{cache}")

    print(f"  Computing ERA5-interp seasonal {season} 2-day median "
          f"({start_year}–{end_year}) ...")
    da = open_cache_fn(cache, start_year - 1, end_year)
    da_2day = rolling_fn(da, window_days=window_days)
    da_sub = subset_fn(da_2day, start_year, end_year)
    da_season = _filter_season_da(da_sub, season)
    out = da_season.median(dim="time").compute()
    out.name = "twoday_seasonal_median"
    out.attrs.update({"units": "mm", "season": season,
                      "start_year": start_year, "end_year": end_year})
    median_cache_path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"twoday_seasonal_median": out}).to_netcdf(
        str(median_cache_path),
        encoding={"twoday_seasonal_median": {"zlib": True, "complevel": 4}})
    print(f"  [saved] {median_cache_path.name}")
    return out


def compute_era5_interpolated_2day_seasonal_p90_2d(
    season: str,
    start_year: int,
    end_year: int,
    era5_interp_dir: "Path",
    p90_cache_path: "Path",
    cache_path_fn,
    open_cache_fn,
    rolling_fn,
    subset_fn,
    window_days: int = 2,
    force_recompute: bool = False,
) -> "xr.DataArray":
    """
    Seasonal 90th percentile of 2-day rolling sums (mm) from ERA5-interpolated cache.

    Only timesteps in `season`'s months are used; result is the 90th pctile of
    ~90×30 values per pixel (days-in-season × years).
    """
    if (not force_recompute) and p90_cache_path.exists():
        print(f"  [cache] ERA5-interp seasonal {season} 2-day p90 ← {p90_cache_path.name}")
        with xr.open_dataset(str(p90_cache_path)) as _ds:
            return _ds["twoday_seasonal_p90"].load()

    files = find_era5_interpolated_files(era5_interp_dir)
    avail_s, avail_e = get_year_range_era5_interp(files)
    cache = cache_path_fn("era5_interpolated", "", avail_s, avail_e)
    if not cache.exists():
        raise FileNotFoundError(
            f"ERA5-interpolated overall cache not found.\n{cache}")

    print(f"  Computing ERA5-interp seasonal {season} 2-day p90 "
          f"({start_year}–{end_year}) ...")
    da = open_cache_fn(cache, start_year - 1, end_year)
    da_2day = rolling_fn(da, window_days=window_days)
    da_sub = subset_fn(da_2day, start_year, end_year)
    da_season = _filter_season_da(da_sub, season)
    out = da_season.quantile(0.90, dim="time").compute()
    out = out.drop_vars("quantile", errors="ignore")
    out.name = "twoday_seasonal_p90"
    out.attrs.update({"units": "mm", "season": season,
                      "start_year": start_year, "end_year": end_year})
    p90_cache_path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"twoday_seasonal_p90": out}).to_netcdf(
        str(p90_cache_path),
        encoding={"twoday_seasonal_p90": {"zlib": True, "complevel": 4}})
    print(f"  [saved] {p90_cache_path.name}")
    return out
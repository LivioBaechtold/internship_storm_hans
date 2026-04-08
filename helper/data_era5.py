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
    if "number" in ds.dims and ds.sizes["number"] == 1:
        ds = ds.isel(number=0, drop=True)

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